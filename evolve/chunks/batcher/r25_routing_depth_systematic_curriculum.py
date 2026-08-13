import re

import torch

NAME = "r25_routing_depth_systematic_curriculum"
DESCRIPTION = (
    "Weights every training sequence by the longest content-routing chain its COMMAND text "
    "describes -- mv/cp moves and shell output redirections are read as path-to-path edges and "
    "composed into a per-sequence longest-hop count -- then draws each batch by Madow systematic "
    "probability-proportional-to-size sampling over a depth-sorted, randomly tie-broken index list "
    "rather than iid multinomial draws, so the batch's chain-depth composition and every "
    "sequence's selection rate hit their target with near-zero step-to-step variance while which "
    "sequence fills a given depth slot stays fully random. The depth tilt exponent ramps from 0, "
    "and an image-blocked share of the batch, itself drawn by the same weighted systematic rule, "
    "ramps in alongside it. The parsed depth chooses which sequences enter a batch and nothing "
    "else: it is never handed to the model, the loss, or any tensor the model sees."
)

_REDIR = re.compile(r">>|>")


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _routing_depth(cmds):
    depth = {}
    best = 0
    for c in cmds:
        if not isinstance(c, str) or not c:
            continue
        if ">" in c:
            parts = _REDIR.split(c)
            if len(parts) < 2:
                continue
            dst_tok = parts[-1].split()
            if not dst_tok:
                continue
            dst = dst_tok[0]
            head = parts[0].split()
            srcs = [t for t in head[1:] if "/" in t and not t.startswith("-")]
            if not srcs:
                continue
            d = max(depth.get(s, 0) for s in srcs) + 1
            if d > depth.get(dst, 0):
                depth[dst] = d
            if d > best:
                best = d
            continue
        toks = c.split()
        if len(toks) < 3 or toks[0] not in ("mv", "cp"):
            continue
        src, dst = toks[-2], toks[-1]
        if src.startswith("-") or dst.startswith("-"):
            continue
        if toks[0] == "mv":
            d = depth.pop(src, 0) + 1
            depth[dst] = d
        else:
            d = depth.get(src, 0) + 1
            if d > depth.get(dst, 0):
                depth[dst] = d
        if d > best:
            best = d
    return best


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


def make_batcher(fit, bs, seed, alpha_max=1.5, depth_ramp_frac=0.35, depth_cap=6,
                 n_block_images=1, hard_frac_max=0.75, ramp_frac=0.3):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    cap = max(1, int(depth_cap))
    depths = []
    groups = {}
    for i, s in enumerate(fit):
        d = _routing_depth(s.get("cmds", ()) or ())
        depths.append(float(min(d, cap)))
        groups.setdefault(s.get("image", "?"), []).append(i)
    d_all = torch.tensor(depths, dtype=torch.float64)
    all_idx = torch.arange(n, dtype=torch.long)

    keys = sorted(groups)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in keys]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    a_max = max(0.0, float(alpha_max))
    blocking = len(pools) >= 2 and float(hard_frac_max) > 0.0
    k_imgs = max(1, min(int(n_block_images), len(pools)))

    def next_batch(step, total_steps):
        T = max(1, int(total_steps))
        a_ramp = max(1, int(float(depth_ramp_frac) * T))
        alpha = a_max * _smoothstep(step / a_ramp)
        w = (1.0 + d_all).pow(alpha)

        if blocking:
            b_ramp = max(1, int(float(ramp_frac) * T))
            frac = float(hard_frac_max) * min(1.0, step / b_ramp)
            n_hard = max(0, min(bs, int(round(bs * frac))))
        else:
            n_hard = 0
        n_free = bs - n_hard

        def draw(pool, m):
            shuffled = pool[torch.randperm(pool.numel(), generator=g)]
            ordered = shuffled[torch.argsort(d_all[shuffled], stable=True)]
            cum = w[ordered].cumsum(0)
            return ordered[_systematic(cum, m, g)]

        parts = []
        if n_free > 0:
            parts.append(draw(all_idx, n_free))
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            parts.append(draw(torch.cat([pools[int(c)] for c in chosen]), n_hard))

        batch = torch.cat(parts) if parts else torch.zeros(bs, dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
