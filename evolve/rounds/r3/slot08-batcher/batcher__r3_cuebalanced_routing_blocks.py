import torch

NAME = "r3_cuebalanced_routing_blocks"
DESCRIPTION = (
    "Parses the mv edges out of every training sequence's command strings and walks each read "
    "backwards through the edges that wrote its path earlier in the same trajectory. For the "
    "deepest such read it records three quantities over the board — the moves whose source or "
    "destination shares the routed content's leading path prefix: the hop count, the rank of "
    "that content's earliest chain move among all board moves in time order, and the rank of "
    "that content's origin among the distinct files read on the board before the first board "
    "move. Each sequence then carries a sampling weight obtained by iterative proportional "
    "fitting inside hop-count strata, driving both order-rank marginals toward uniform while "
    "each stratum keeps its original share; the weight is raised to a strength exponent, clipped "
    "to a bounded ratio, active from the first step and never annealed. A batch is a base part "
    "drawn from those weights plus blocks; a block is either the pooled union of several system "
    "images or a read-collision neighbourhood (deepest-routed-read command embeddings that are "
    "near-duplicates under a fixed Johnson-Lindenstrauss projection while their read answers are "
    "not duplicates), and a block's slots are allotted round-robin across first-move-rank "
    "classes so every batch carries the classes in near-equal number. Inside a block, membership "
    "follows that weight times a Boltzmann tilt exp(alpha*z) on the hop-count score, with alpha "
    "and the blocked fraction annealed up from zero over the opening fraction of training. "
    "Degrades to the weighted base alone when no block source is available and to uniform "
    "sampling when no routed read is recoverable."
)

_READ_VERBS = ("cat", "head", "tail")
_PREFIX_DEPTH = 3
_N_CUE = 3
_N_DBIN = 4


def _prefix(path, k):
    parts = [q for q in path.split("/") if q]
    return "/".join(parts[:k])


def _scan_seq(cmds):
    mvs = []
    reads = []
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            mvs.append((t, p[1], p[2]))
        elif len(p) == 2 and p[0] in _READ_VERBS:
            reads.append((t, p[1]))
    if not mvs or not reads:
        return None

    wrote = {}
    for t, a, b in mvs:
        wrote.setdefault(b, []).append((t, a))

    best = None
    for t, path in reads:
        cur, ct, d, first = path, t, 0, None
        while d < 64:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            first = ct if first is None else min(first, ct)
            d += 1
        if d >= 1 and (best is None or d > best[0]):
            best = (d, t, path, cur, first)
    if best is None:
        return None

    depth, read_t, _read_path, origin, chain_t0 = best
    pref = _prefix(origin, _PREFIX_DEPTH)

    board = [(t, a, b) for t, a, b in mvs
             if t < read_t and (_prefix(a, _PREFIX_DEPTH) == pref
                                or _prefix(b, _PREFIX_DEPTH) == pref)]
    board.sort()
    fm_rank = -1
    for k, (t, _a, _b) in enumerate(board):
        if t == chain_t0:
            fm_rank = k
            break

    t0 = board[0][0] if board else read_t
    cands = []
    seen = set()
    for t, path in reads:
        if t >= t0:
            break
        if path in seen or _prefix(path, _PREFIX_DEPTH) != pref:
            continue
        seen.add(path)
        cands.append(path)
    orank = cands.index(origin) if origin in cands else -1

    return depth, read_t, fm_rank, orank


def _scan(fit):
    depth = []
    read_at = []
    cue_a = []
    cue_b = []
    for s in fit:
        got = _scan_seq(s.get("cmds") or [])
        if got is None:
            depth.append(0.0)
            read_at.append(-1)
            cue_a.append(_N_CUE)
            cue_b.append(_N_CUE)
            continue
        d, t, fm, orank = got
        zc = s.get("z_cmd")
        zo = s.get("z_obs")
        if zc is None or zo is None:
            t = -1
        elif t >= int(zc.shape[0]) or t >= int(zo.shape[0]):
            t = -1
        depth.append(float(d))
        read_at.append(int(t))
        cue_a.append(min(int(fm), _N_CUE - 1) if fm >= 0 else _N_CUE)
        cue_b.append(min(int(orank), _N_CUE - 1) if orank >= 0 else _N_CUE)
    return depth, read_at, cue_a, cue_b


def _dbin(d):
    if d <= 0:
        return 3
    if d <= 2:
        return 0
    if d <= 4:
        return 1
    return 2


def _rake(raw, axes, iters):
    for _ in range(int(iters)):
        for lab in axes:
            classes = sorted(set(lab.tolist()))
            if len(classes) < 2:
                continue
            target = float(raw.sum()) / len(classes)
            for c in classes:
                mask = lab == c
                s = float(raw[mask].sum())
                if s > 1e-12:
                    raw[mask] = raw[mask] * (target / s)
    return raw


def _cue_weights(depth, cue_a, cue_b, gamma, cap, iters):
    n = len(depth)
    w = torch.ones(n, dtype=torch.float32)
    if float(gamma) <= 0.0 or n == 0:
        return w
    bins = [_dbin(int(d)) for d in depth]
    hi = float(cap) if float(cap) > 1.0 else 1.0
    lo = 1.0 / hi
    for b in range(_N_DBIN):
        rows = [i for i in range(n) if bins[i] == b]
        if len(rows) < 2:
            continue
        la = torch.tensor([cue_a[i] for i in rows], dtype=torch.long)
        lb = torch.tensor([cue_b[i] for i in rows], dtype=torch.long)
        raw = _rake(torch.ones(len(rows), dtype=torch.float32), (la, lb), iters)
        raw = raw / raw.mean().clamp_min(1e-12)
        raw = raw.pow(float(gamma))
        raw = raw / raw.mean().clamp_min(1e-12)
        raw = raw.clamp(lo, hi)
        raw = raw / raw.mean().clamp_min(1e-12)
        for k, i in enumerate(rows):
            w[i] = raw[k]
    return w.clamp_min(1e-8)


def _collision_pools(fit, read_at, gen, proj_dim, nb_k, sim_thresh, ans_max_sim,
                     max_centers, chunk):
    cand = [i for i, t in enumerate(read_at) if t >= 0]
    if len(cand) < 8:
        return []
    sel = torch.tensor(cand, dtype=torch.long)
    if sel.numel() > int(max_centers):
        sel = sel[torch.randperm(sel.numel(), generator=gen)[: int(max_centers)]]
        sel, _ = torch.sort(sel)
    q = torch.stack([fit[int(i)]["z_cmd"][read_at[int(i)]].detach().float() for i in sel])
    a = torch.stack([fit[int(i)]["z_obs"][read_at[int(i)]].detach().float() for i in sel])
    dim = int(q.shape[1])
    pd = max(8, min(int(proj_dim), dim))
    rmat = torch.randn(dim, pd, generator=gen)
    qp = torch.nn.functional.normalize(q @ rmat, dim=1, eps=1e-8)
    ap = torch.nn.functional.normalize(a @ rmat, dim=1, eps=1e-8)
    m = int(sel.numel())
    kk = max(1, min(int(nb_k), m - 1))
    pools = []
    for start in range(0, m, int(chunk)):
        stop = min(m, start + int(chunk))
        rows = torch.arange(start, stop)
        sq = qp[start:stop] @ qp.t()
        sa = ap[start:stop] @ ap.t()
        sq = sq.masked_fill(sa > float(ans_max_sim), -2.0)
        sq[torch.arange(stop - start), rows] = -2.0
        val, idx = torch.topk(sq, kk, dim=1)
        keep = val >= float(sim_thresh)
        for r in range(stop - start):
            nb = idx[r][keep[r]]
            if nb.numel() >= 1:
                pools.append(sel[torch.cat([rows[r].view(1), nb])])
    return pools


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, n_blocks=2,
                 coll_share=0.5, alpha_max=2.5, cue_gamma=1.0, cue_cap=4.0, cue_iters=4,
                 n_block_images=2, proj_dim=128, nb_k=16, sim_thresh=0.85,
                 ans_max_sim=0.995, max_centers=8000, chunk=256):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0:
        return uniform_only()

    depth, read_at, cue_a, cue_b = _scan(fit)
    if max(depth) <= 0.0:
        return uniform_only()

    wcue = _cue_weights(depth, cue_a, cue_b, cue_gamma, cue_cap, cue_iters)
    cdf = torch.cumsum(wcue, dim=0)
    cdf = cdf / cdf[-1].clamp_min(1e-12)

    def base_draw(m):
        if m <= 0:
            return torch.zeros(0, dtype=torch.long)
        u = torch.rand((m,), generator=g)
        return torch.searchsorted(cdf, u).clamp_(0, n - 1)

    zraw = torch.tensor(depth, dtype=torch.float32)
    zctr = zraw - zraw.mean()
    z = (zctr / zctr.pow(2).mean().sqrt().clamp_min(1e-6)).clamp(-4.0, 4.0)
    cls = torch.tensor(cue_a, dtype=torch.long)

    by_image = {}
    for i, s in enumerate(fit):
        by_image.setdefault(s.get("image", "?"), []).append(i)
    img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools])

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0x7FFFFFFF)
    coll_pools = _collision_pools(fit, read_at, g_setup, proj_dim, nb_k, sim_thresh,
                                  ans_max_sim, max_centers, chunk)
    have_coll = len(coll_pools) >= 8
    have_img = len(img_pools) >= 2
    if float(hard_frac_max) <= 0.0 or (not have_coll and not have_img):
        def next_batch_base(step, total_steps):
            return base_draw(bs).tolist()
        return next_batch_base

    centers = torch.tensor([int(p[0]) for p in coll_pools], dtype=torch.long) if have_coll \
        else torch.zeros(0, dtype=torch.long)
    kb = max(1, int(n_blocks))
    k_imgs = max(1, min(int(n_block_images), len(img_pools)))
    share = 0.0 if not have_coll else (1.0 if not have_img else float(coll_share))

    def weights_of(pool, alpha):
        w = torch.exp(alpha * (z[pool] - z[pool].max())) * wcue[pool]
        return torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)

    def balanced_pick(pool, m, alpha, rot):
        if m <= 0:
            return torch.zeros(0, dtype=torch.long)
        base_w = weights_of(pool, alpha)
        pcls = cls[pool]
        picks = []
        leftover = 0
        for j in range(_N_CUE):
            k = m // _N_CUE + (1 if j < m % _N_CUE else 0)
            if k <= 0:
                continue
            c = (rot + j) % _N_CUE
            mask = pcls == c
            if not bool(mask.any()):
                leftover += k
                continue
            w = base_w * mask.to(base_w.dtype)
            picks.append(pool[torch.multinomial(w, k, replacement=True, generator=g)])
        if leftover > 0:
            picks.append(pool[torch.multinomial(base_w, leftover, replacement=True,
                                                generator=g)])
        if not picks:
            return pool[torch.multinomial(base_w, m, replacement=True, generator=g)]
        return torch.cat(picks)

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)
        alpha = float(alpha_max) * ramp
        n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
        rot = int(step) % _N_CUE
        parts = [base_draw(bs - n_hard)]
        if n_hard > 0:
            base = n_hard // kb
            rem = n_hard - base * kb
            for j in range(kb):
                m = base + (1 if j < rem else 0)
                if m <= 0:
                    continue
                u = float(torch.rand((1,), generator=g).item())
                if u < share:
                    ci = int(torch.multinomial(
                        weights_of(centers, alpha), 1, replacement=True, generator=g).item())
                    pool = coll_pools[ci]
                else:
                    gi = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                    pool = torch.cat([img_pools[int(c)] for c in gi])
                parts.append(balanced_pick(pool, m, alpha, rot + j))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
