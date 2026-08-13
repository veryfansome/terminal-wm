import math

import torch

NAME = "r5_matched_cmdreplicate_routeblocks"
DESCRIPTION = (
    "Builds batches as matched replicate groups: an inverted index maps every command STRING to "
    "the training sequences that issue it, and a group is a set of sequences that all issue the "
    "SAME command string yet answer it with different observations. A command qualifies as a "
    "group anchor only when (a) it occurs in at least min_occ sequences, (b) its base rate over "
    "the split is below max_rate so that grouping actually concentrates replicates beyond what "
    "uniform sampling delivers, and (c) the observations recorded at that command, compared "
    "pairwise under one fixed Gaussian projection, are distinct for at least distinct_min of the "
    "pairs, which removes commands whose observation is constant across the split (empty-output "
    "redirections and moves) and, at a high threshold, commands whose observation takes only as "
    "many values as there are system images. Anchor weight is the distinct-pair fraction times a "
    "bonus for anchors whose path is "
    "touched by an mv edge in the same sequence, where the mv edges and the per-sequence "
    "hop-count of the deepest read reachable through them are parsed from the command strings. "
    "Group members are drawn WITHOUT replacement under a Boltzmann tilt exp(alpha*z) on that "
    "hop-count, so a group holds distinct sequences rather than repeats. A batch is a uniform "
    "part plus n_blocks blocks, each block an anchor group or, with probability img_share, one "
    "system image; the blocked fraction and alpha are annealed up from zero over the opening "
    "fraction of training. Setup reads only command strings and cached observation embeddings of "
    "the training split; nothing is consulted at batch time except the precomputed pools."
)


def _mv_edges(parsed):
    edges = []
    for t, p in enumerate(parsed):
        if len(p) == 3 and p[0] == "mv":
            edges.append((t, p[1], p[2]))
    return edges


def _route_depth(parsed, edges):
    if not edges:
        return 0.0
    wrote = {}
    for t, src, dst in edges:
        wrote.setdefault(dst, []).append((t, src))
    best = 0
    for t, p in enumerate(parsed):
        if len(p) != 2 or p[0] != "cat":
            continue
        cur, ct, d = p[1], t, 0
        while d < 64:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            d += 1
        if d > best:
            best = d
    return float(best)


def _n_steps(s):
    zo = s.get("z_obs")
    if zo is None or not torch.is_tensor(zo) or zo.dim() != 2:
        return 0, None
    cmds = s.get("cmds") or []
    return min(len(cmds), int(zo.shape[0])), zo


def _index_split(fit):
    depth = []
    index = {}
    for i, s in enumerate(fit):
        nst, _ = _n_steps(s)
        if nst <= 0:
            depth.append(0.0)
            continue
        cmds = s["cmds"]
        parsed = [c.split() for c in cmds[:nst]]
        edges = _mv_edges(parsed)
        depth.append(_route_depth(parsed, edges))
        touched = set()
        for _, src, dst in edges:
            touched.add(src)
            touched.add(dst)
        seen = set()
        for t in range(nst):
            c = cmds[t]
            if c in seen:
                continue
            seen.add(c)
            flag = 0.0
            for tokidx in range(1, len(parsed[t])):
                tok = parsed[t][tokidx]
                if tok.startswith("/") and tok in touched:
                    flag = 1.0
                    break
            index.setdefault(c, []).append((i, t, flag))
    return depth, index


def _reference_scale(fit, proj, gen, ref_seqs, ref_steps):
    n = len(fit)
    take = min(int(ref_seqs), n)
    if take < 2:
        return None
    order = torch.randperm(n, generator=gen)[:take].tolist()
    rows = []
    for i in order:
        nst, zo = _n_steps(fit[i])
        if nst <= 0:
            continue
        stride = max(1, nst // max(1, int(ref_steps)))
        for t in range(0, nst, stride):
            rows.append(zo[t].detach().float())
    if len(rows) < 8:
        return None
    x = torch.stack(rows) @ proj
    sq = (x * x).sum(1)
    d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (x @ x.t())).clamp_min(0.0)
    m = x.shape[0]
    val = float(d2.sum().item() / max(1.0, float(m * (m - 1))))
    if not (val == val) or val <= 0.0:
        return None
    return val


def _anchor_stats(fit, groups, proj, tau_abs, chunk_rows):
    fracs = []
    start = 0
    while start < len(groups):
        stop = start
        total = 0
        while stop < len(groups):
            take = len(groups[stop][1])
            if stop > start and total + take > int(chunk_rows):
                break
            total += take
            stop += 1
        rows = []
        for gi in range(start, stop):
            for i, t in groups[gi][1]:
                rows.append(fit[i]["z_obs"][t].detach().float())
        block = torch.stack(rows) @ proj
        off = 0
        for gi in range(start, stop):
            m = len(groups[gi][1])
            x = block[off:off + m]
            off += m
            if m < 2:
                fracs.append(0.0)
                continue
            sq = (x * x).sum(1)
            d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (x @ x.t())).clamp_min(0.0)
            hits = float((d2 > tau_abs).sum().item())
            fracs.append(hits / max(1.0, float(m * (m - 1))))
        start = stop
    return fracs


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, n_blocks=2,
                 img_share=0.2, alpha_max=2.5, proj_dim=32, min_occ=8, max_rate=0.05,
                 occ_cap=32, max_anchors=2000, distinct_tau=1e-5, distinct_min=0.85,
                 route_bonus=1.0, chunk_rows=8192, ref_seqs=128, ref_steps=8):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or float(hard_frac_max) <= 0.0:
        return uniform_only()

    depth, index = _index_split(fit)
    zraw = torch.tensor(depth, dtype=torch.float32)
    zctr = zraw - zraw.mean()
    z = (zctr / zctr.pow(2).mean().sqrt().clamp_min(1e-6)).clamp(-4.0, 4.0)

    by_image = {}
    for i, s in enumerate(fit):
        by_image.setdefault(s.get("image", "?"), []).append(i)
    img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools])
    have_img = len(img_pools) >= 2

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x5BF03635) & 0x7FFFFFFF)

    lo = max(2, int(min_occ))
    hi = max(lo, int(float(max_rate) * n))
    cand = [(c, v) for c, v in index.items() if lo <= len(v) <= hi]
    cand.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    cand = cand[: max(0, int(max_anchors))]

    dim = 0
    for s in fit:
        nst, zo = _n_steps(s)
        if nst > 0:
            dim = int(zo.shape[1])
            break

    anchors = []
    if cand and dim > 0:
        pd = max(4, min(int(proj_dim), dim))
        proj = torch.randn(dim, pd, generator=g_setup)
        ref = _reference_scale(fit, proj, g_setup, ref_seqs, ref_steps)
        if ref is not None:
            groups = []
            for c, v in cand:
                stride = max(1, len(v) // max(1, int(occ_cap)))
                probe = [(i, t) for i, t, _ in v[::stride]][: max(2, int(occ_cap))]
                groups.append((c, probe))
            fracs = _anchor_stats(fit, groups, proj, float(distinct_tau) * ref, chunk_rows)
            for (c, v), frac in zip(cand, fracs):
                if frac < float(distinct_min):
                    continue
                pool = torch.tensor([i for i, _, _ in v], dtype=torch.long)
                rfrac = sum(f for _, _, f in v) / float(len(v))
                w = frac * (1.0 + float(route_bonus) * rfrac)
                if w <= 0.0:
                    continue
                anchors.append((pool, float(w)))

    have_anchor = len(anchors) >= 4
    if not have_anchor and not have_img:
        return uniform_only()

    if have_anchor:
        a_logw = torch.tensor([math.log(max(w, 1e-12)) for _, w in anchors])
        a_zmean = torch.tensor([float(z[p].mean().item()) for p, _ in anchors])
        a_pools = [p for p, _ in anchors]
    else:
        a_logw = torch.zeros(0)
        a_zmean = torch.zeros(0)
        a_pools = []

    kb = max(1, int(n_blocks))
    share_img = 1.0 if not have_anchor else (float(img_share) if have_img else 0.0)

    def tilted_with_replacement(pool, m, alpha):
        w = torch.exp(alpha * (z[pool] - z[pool].max()))
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)
        return pool[torch.multinomial(w, m, replacement=True, generator=g)]

    def tilted_distinct(pool, m, alpha):
        k = min(int(m), int(pool.numel()))
        w = torch.exp(alpha * (z[pool] - z[pool].max()))
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)
        sel = pool[torch.multinomial(w, k, replacement=False, generator=g)]
        if k < m:
            return [sel, torch.randint(0, n, (m - k,), generator=g)]
        return [sel]

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)
        alpha = float(alpha_max) * ramp
        n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            base = n_hard // kb
            rem = n_hard - base * kb
            for j in range(kb):
                m = base + (1 if j < rem else 0)
                if m <= 0:
                    continue
                u = float(torch.rand((1,), generator=g).item())
                if u < share_img:
                    gi = int(torch.multinomial(img_sizes, 1, replacement=True,
                                               generator=g).item())
                    parts.append(tilted_with_replacement(img_pools[gi], m, alpha))
                else:
                    s = alpha * a_zmean + a_logw
                    ws = torch.exp(s - s.max()).clamp_min(1e-12)
                    ai = int(torch.multinomial(ws, 1, replacement=True, generator=g).item())
                    parts.extend(tilted_distinct(a_pools[ai], m, alpha))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
