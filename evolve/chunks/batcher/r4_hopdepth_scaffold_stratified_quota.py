import torch

NAME = "r4_hopdepth_scaffold_stratified_quota"
DESCRIPTION = (
    "Replays every training sequence's command list through a symbolic content tracker (mv "
    "retires a path and hands its content one hop further, cp and 'read > path' copy it one hop "
    "further, 'read >> path' keeps the deeper of the accumulator and the source, a bare read "
    "reports the hop count standing at its path) and labels the sequence by the hop count of its "
    "deepest routed read. Batch composition is then a stratified quota over those hop-count "
    "strata rather than a draw: a frontier walks the hop scale from one hop up to the deepest "
    "well-populated stratum over the opening fraction of training, an asymmetric kernel around "
    "the frontier (sharp decay above it, mild decay below so shallower strata are retained) turns "
    "into shares, and largest-remainder allocation turns the shares into an exact per-depth count "
    "that every single batch carries. The scheduled slots are drawn from the strata restricted to "
    "a few system images sampled per batch, tilted toward sequences carrying more routed reads, "
    "and the remaining slots are drawn uniformly over the whole split. Falls back to uniform "
    "sampling when no routed read is recoverable."
)

_READ_VERBS = ("cat", "head", "tail")
_BREAK_TOKENS = ("|", "&&", "||", ";", "<", "2>", "2>>")


def _parse_event(cmd):
    toks = cmd.split()
    if not toks:
        return None
    verb = toks[0]
    args = []
    redir = None
    rtgt = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in _BREAK_TOKENS:
            return None
        if t == ">" or t == ">>":
            redir = t
            if i + 1 < len(toks):
                rtgt = toks[i + 1]
                i += 1
        elif t.startswith(">>"):
            redir = ">>"
            rtgt = t[2:] or rtgt
        elif t.startswith(">"):
            redir = ">"
            rtgt = t[1:] or rtgt
        elif t.startswith("-") and len(t) > 1:
            pass
        else:
            args.append(t)
        i += 1
    if verb == "mv" and redir is None and len(args) == 2:
        return ("move", args[0], args[1])
    if verb == "cp" and redir is None and len(args) == 2:
        return ("copy", args[0], args[1])
    if verb in _READ_VERBS and len(args) == 1:
        if redir == ">" and rtgt:
            return ("copy", args[0], rtgt)
        if redir == ">>" and rtgt:
            return ("append", args[0], rtgt)
        if redir is None:
            return ("read", args[0], None)
    return None


def _read_hops(cmds):
    hops = {}
    reads = []
    for cmd in cmds:
        if not isinstance(cmd, str):
            continue
        ev = _parse_event(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "move":
            hops[b] = hops.pop(a, 0) + 1
        elif kind == "copy":
            hops[b] = hops.get(a, 0) + 1
        elif kind == "append":
            hops[b] = max(hops.get(b, 0), hops.get(a, 0) + 1)
        else:
            reads.append(hops.get(a, 0))
    return reads


def _quota(shares, total):
    raw = shares * float(total)
    base = torch.floor(raw)
    left = int(total) - int(base.sum().item())
    if left > 0:
        _, order = torch.sort(raw - base, descending=True, stable=True)
        base[order[:left]] += 1.0
    return [int(v) for v in base.tolist()]


def make_batcher(fit, bs, seed, curr_frac=0.75, ramp_frac=0.35, warm_frac=0.1, tau_up=0.5,
                 lam_down=0.35, n_block_images=1, depth_cap=6, min_stratum=8, demand_pow=1.0):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or bs <= 0 or float(curr_frac) <= 0.0:
        return uniform_batch

    cap = max(1, int(depth_cap))
    labels = []
    member = []
    images = []
    for s in fit:
        hp = _read_hops(s.get("cmds") or [])
        deepest = 0
        load = 0.0
        for d in hp:
            if d >= 1:
                c = min(int(d), cap)
                if c > deepest:
                    deepest = c
                load += float(c)
        labels.append(deepest)
        member.append((1.0 + load) ** float(demand_pow))
        images.append(str(s.get("image", "?")))

    img_names = sorted(set(images))
    img_pos = {k: j for j, k in enumerate(img_names)}

    strata = {}
    blocks = {}
    img_load = [0.0] * len(img_names)
    for i in range(n):
        d = labels[i]
        if d < 1:
            continue
        strata.setdefault(d, []).append(i)
        blocks.setdefault((img_pos[images[i]], d), []).append(i)
        img_load[img_pos[images[i]]] += 1.0

    if not strata:
        return uniform_batch

    def _pack(idxs):
        return (torch.tensor(idxs, dtype=torch.long),
                torch.tensor([member[j] for j in idxs], dtype=torch.float))

    stratum_pool = {d: _pack(v) for d, v in strata.items()}
    block_pool = {k: _pack(v) for k, v in blocks.items()}

    depths = sorted(stratum_pool)
    big = [d for d in depths if len(strata[d]) >= int(min_stratum)]
    d_top = float(max(big) if big else max(depths))
    d_vec = torch.tensor([float(d) for d in depths], dtype=torch.float)
    room = [len(strata[d]) for d in depths]

    img_w = torch.tensor(img_load, dtype=torch.float).clamp_min(1e-6)
    k_imgs = max(1, min(int(n_block_images), len(img_names)))

    def _draw(chosen, d, count):
        parts_i = []
        parts_w = []
        for j in chosen:
            hit = block_pool.get((j, d))
            if hit is not None:
                parts_i.append(hit[0])
                parts_w.append(hit[1])
        if parts_i:
            pool_i = torch.cat(parts_i) if len(parts_i) > 1 else parts_i[0]
            pool_w = torch.cat(parts_w) if len(parts_w) > 1 else parts_w[0]
        else:
            pool_i, pool_w = stratum_pool[d]
        sel = torch.multinomial(pool_w, count, replacement=True, generator=g)
        return pool_i[sel]

    def next_batch(step, total_steps):
        total = max(1, int(total_steps))
        cur = max(1, int(step))
        ramp = min(1.0, float(cur) / max(1.0, float(ramp_frac) * total))
        warm = min(1.0, float(cur) / max(1.0, float(warm_frac) * total))
        frontier = 1.0 + (d_top - 1.0) * ramp
        n_curr = max(0, min(bs, int(round(bs * float(curr_frac) * warm))))

        parts = []
        if n_curr > 0:
            gap = d_vec - frontier
            logw = torch.where(gap > 0.0,
                               -gap / max(1e-3, float(tau_up)),
                               gap * float(lam_down))
            w = torch.exp(logw - logw.max()).clamp_min(1e-8)
            shares = w / w.sum()
            counts = _quota(shares, n_curr)
            chosen = torch.multinomial(img_w, k_imgs, replacement=False,
                                       generator=g).tolist()
            filled = 0
            for pos, d in enumerate(depths):
                c = min(counts[pos], room[pos])
                if c > 0:
                    parts.append(_draw(chosen, d, c))
                    filled += c
            n_curr = filled
        if n_curr < bs:
            parts.append(torch.randint(0, n, (bs - n_curr,), generator=g))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
