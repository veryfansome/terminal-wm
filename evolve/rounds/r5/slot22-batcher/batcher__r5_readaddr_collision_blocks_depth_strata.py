import re

import torch

NAME = "r5_readaddr_collision_blocks_depth_strata"
DESCRIPTION = (
    "Composes each batch from READ-ADDRESS COLLISION BLOCKS, SYSTEM BLOCKS and a depth-stratified "
    "remainder. Every training sequence's COMMAND text is parsed into a write graph -- mv/cp edges "
    "plus shell output redirections -- and each 'cat P' is walked backwards through the writes that "
    "landed on P earlier in the same trajectory, giving that read's hop count, the address it reads "
    "and the path its content started at. A sequence is described by its deepest such read: the hop "
    "count, the read address with its trailing move index stripped, and a CROSSED flag set when the "
    "content's origin name family differs from the read address's. A ramped share of the batch is "
    "filled from a few read addresses, so those rows carry near-identical read commands with "
    "different answers, and each address's slots are split between crossed and aligned members so "
    "the origin name family is decorrelated from the read address inside the block; a smaller share "
    "is filled from one system image, densifying same-system negatives. Every remaining slot is "
    "drawn under a two-stratum allocation on hop count whose deep share ramps from the corpus rate "
    "to a target, and every draw is Madow systematic probability-proportional-to-size over a "
    "hop-count-sorted, randomly tie-broken frame, so composition hits its target with near-zero "
    "step-to-step variance while membership stays random. The parse chooses which sequences enter a "
    "batch and nothing else: no parsed quantity reaches the model, the loss, or any tensor."
)

_SUFFIX = re.compile(r"\.\d+$")
_REDIR = re.compile(r">>|>")


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _family(path):
    head, _, base = path.rpartition("/")
    return head + "/" + _SUFFIX.sub("", base)


def _deepest_routed_read(cmds):
    writes = {}
    reads = []
    for t, c in enumerate(cmds):
        if not isinstance(c, str) or not c:
            continue
        if ">" in c:
            parts = _REDIR.split(c)
            if len(parts) < 2:
                continue
            right = parts[-1].split()
            if not right:
                continue
            left = parts[0].split()
            srcs = [x for x in left[1:] if "/" in x and not x.startswith("-")]
            if not srcs:
                continue
            writes.setdefault(right[0], []).append((t, srcs[-1]))
            continue
        toks = c.split()
        if (len(toks) == 3 and toks[0] in ("mv", "cp")
                and not toks[1].startswith("-") and not toks[2].startswith("-")):
            writes.setdefault(toks[2], []).append((t, toks[1]))
        elif len(toks) == 2 and toks[0] == "cat" and "/" in toks[1]:
            reads.append((t, toks[1]))
    best_h, best_read, best_origin = 0, None, None
    for t, p in reads:
        cur, ct, h = p, t, 0
        while h < 64:
            prior = [(j, s) for j, s in writes.get(cur, ()) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            h += 1
        if h > best_h:
            best_h, best_read, best_origin = h, p, cur
    return best_h, best_read, best_origin


def _systematic(cum, m, g):
    if m <= 0:
        return torch.empty(0, dtype=torch.long)
    if cum.numel() == 0:
        return torch.zeros(m, dtype=torch.long)
    total = float(cum[-1])
    if not (total > 0.0):
        return torch.zeros(m, dtype=torch.long)
    width = total / float(m)
    u = torch.rand((), generator=g, dtype=torch.float64) * width
    marks = u + width * torch.arange(m, dtype=torch.float64)
    pos = torch.searchsorted(cum, marks.contiguous())
    return pos.clamp_(min=0, max=cum.numel() - 1)


def _split_slots(total, k):
    base = total // k
    rem = total - base * k
    return [base + (1 if j < rem else 0) for j in range(k)]


def make_batcher(fit, bs, seed, alpha_max=1.5, depth_ramp_frac=0.35, depth_cap=6,
                 deep_min=3, deep_frac_max=0.75, block_frac_max=0.75, addr_share=0.6,
                 n_addr_keys=3, cross_frac=0.5, min_block=4, n_block_images=1,
                 ramp_frac=0.3):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    cap = max(1, int(depth_cap))
    dmin = max(1, int(deep_min))

    depths = []
    addr_of = []
    crossed_of = []
    image_of = []
    for s in fit:
        hops, read_p, origin_p = _deepest_routed_read(s.get("cmds", ()) or ())
        depths.append(float(min(hops, cap)))
        image_of.append(str(s.get("image", "?")))
        if read_p is not None and origin_p is not None and hops >= dmin:
            fam_read = _family(read_p)
            addr_of.append(fam_read)
            crossed_of.append(fam_read != _family(origin_p))
        else:
            addr_of.append(None)
            crossed_of.append(False)

    d_all = torch.tensor(depths, dtype=torch.float64)
    all_idx = torch.arange(n, dtype=torch.long)
    deep_idx = all_idx[d_all >= float(dmin)]
    shallow_idx = all_idx[d_all < float(dmin)]
    emp_deep = float(deep_idx.numel()) / float(max(1, n))

    img_groups = {}
    for i, img in enumerate(image_of):
        img_groups.setdefault(img, []).append(i)
    img_keys = sorted(img_groups)
    img_pool = [torch.tensor(img_groups[k], dtype=torch.long) for k in img_keys]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pool])

    addr_groups = {}
    for i, a in enumerate(addr_of):
        if a is not None:
            addr_groups.setdefault(a, []).append(i)
    floor_block = max(2, int(min_block))
    addr_keys = [k for k in sorted(addr_groups) if len(addr_groups[k]) >= floor_block]
    addr_pool = [torch.tensor(addr_groups[k], dtype=torch.long) for k in addr_keys]
    addr_cross = [torch.tensor([i for i in addr_groups[k] if crossed_of[i]], dtype=torch.long)
                  for k in addr_keys]
    addr_align = [torch.tensor([i for i in addr_groups[k] if not crossed_of[i]], dtype=torch.long)
                  for k in addr_keys]
    addr_sizes = torch.tensor([float(p.numel()) for p in addr_pool])

    have_addr = len(addr_pool) > 0
    have_img = len(img_pool) >= 2
    k_addr = min(max(1, int(n_addr_keys)), len(addr_pool)) if have_addr else 0
    k_img = min(max(1, int(n_block_images)), len(img_pool)) if have_img else 0
    blocking = float(block_frac_max) > 0.0 and (have_addr or have_img)
    a_max = max(0.0, float(alpha_max))
    share = min(1.0, max(0.0, float(addr_share)))
    if not have_addr:
        share = 0.0
    elif not have_img:
        share = 1.0

    def next_batch(step, total_steps):
        T = max(1, int(total_steps))
        a_ramp = max(1, int(float(depth_ramp_frac) * T))
        alpha = a_max * _smoothstep(step / a_ramp)
        w = (1.0 + d_all).pow(alpha)

        b_ramp = max(1, int(float(ramp_frac) * T))
        r = _smoothstep(step / b_ramp)

        if blocking:
            n_block = max(0, min(bs, int(round(bs * float(block_frac_max) * r))))
        else:
            n_block = 0
        n_addr = max(0, min(n_block, int(round(n_block * share))))
        n_sys = n_block - n_addr
        n_free = bs - n_block

        def draw(pool, m):
            if m <= 0:
                return torch.empty(0, dtype=torch.long)
            src = pool if pool.numel() > 0 else all_idx
            shuffled = src[torch.randperm(src.numel(), generator=g)]
            ordered = shuffled[torch.argsort(d_all[shuffled], stable=True)]
            return ordered[_systematic(w[ordered].cumsum(0), m, g)]

        parts = []

        if n_free > 0:
            if deep_idx.numel() == 0:
                p_deep = 0.0
            elif shallow_idx.numel() == 0:
                p_deep = 1.0
            else:
                p_deep = emp_deep + (float(deep_frac_max) - emp_deep) * r
                p_deep = min(1.0, max(0.0, p_deep))
            m_deep = max(0, min(n_free, int(round(n_free * p_deep))))
            parts.append(draw(deep_idx, m_deep))
            parts.append(draw(shallow_idx, n_free - m_deep))

        if n_addr > 0:
            picked = torch.multinomial(addr_sizes, k_addr, replacement=False, generator=g)
            for j, m in zip(picked.tolist(), _split_slots(n_addr, k_addr)):
                if m <= 0:
                    continue
                j = int(j)
                nc = int(addr_cross[j].numel())
                na = int(addr_align[j].numel())
                if m >= 2 and nc > 0 and na > 0:
                    m_c = max(1, min(m - 1, int(round(m * float(cross_frac)))))
                    parts.append(draw(addr_cross[j], m_c))
                    parts.append(draw(addr_align[j], m - m_c))
                else:
                    parts.append(draw(addr_pool[j], m))

        if n_sys > 0:
            picked = torch.multinomial(img_sizes, k_img, replacement=False, generator=g)
            pool = torch.cat([img_pool[int(c)] for c in picked.tolist()])
            parts.append(draw(pool, n_sys))

        batch = torch.cat(parts) if parts else torch.zeros(bs, dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
