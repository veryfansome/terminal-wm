import torch

NAME = "r2_routedepth_tilt_readcollision_blocks"
DESCRIPTION = (
    "Composes each batch from the mv-reachability graph parsed out of every training "
    "sequence's command strings: each 'cat P' is walked backwards through the mv edges that "
    "wrote P earlier in the same trajectory, giving the exact number of hops between the read "
    "location and the file the content started in, and a sequence is scored by its deepest "
    "such read. A batch is a uniform part plus blocks; a block is either one system image or a "
    "read-collision neighbourhood — sequences whose deepest routed read carries a near-duplicate "
    "command embedding under a fixed Johnson-Lindenstrauss projection while their read answers "
    "are not duplicates. Membership inside a block follows a Boltzmann tilt exp(alpha*z) on the "
    "hop-count score, with alpha and the blocked fraction annealed up from zero over the opening "
    "fraction of training."
)


def _deepest_routed_read(cmds):
    mvs = []
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            mvs.append((t, p[1], p[2]))
    if not mvs:
        return 0, -1
    wrote = {}
    for t, src, dst in mvs:
        wrote.setdefault(dst, []).append((t, src))
    best_d, best_t = 0, -1
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) != 2 or p[0] != "cat":
            continue
        cur, ct, d = p[1], t, 0
        while d < 64:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            d += 1
        if d > best_d:
            best_d, best_t = d, t
    return best_d, best_t


def _scan(fit):
    depth = []
    read_at = []
    for s in fit:
        d, t = _deepest_routed_read(s.get("cmds") or [])
        zc = s.get("z_cmd")
        zo = s.get("z_obs")
        if zc is None or zo is None:
            t = -1
        elif t >= 0 and (t >= int(zc.shape[0]) or t >= int(zo.shape[0])):
            t = -1
        depth.append(float(d))
        read_at.append(int(t))
    return depth, read_at


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
                 coll_share=0.5, alpha_max=2.5, proj_dim=128, nb_k=16,
                 sim_thresh=0.85, ans_max_sim=0.995, max_centers=8000, chunk=256):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or float(hard_frac_max) <= 0.0:
        return uniform_only()

    depth, read_at = _scan(fit)
    zraw = torch.tensor(depth, dtype=torch.float32)
    zctr = zraw - zraw.mean()
    z = (zctr / zctr.pow(2).mean().sqrt().clamp_min(1e-6)).clamp(-4.0, 4.0)

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
    if not have_coll and not have_img:
        return uniform_only()

    centers = torch.tensor([int(p[0]) for p in coll_pools], dtype=torch.long) if have_coll \
        else torch.zeros(0, dtype=torch.long)
    kb = max(1, int(n_blocks))
    share = 0.0 if not have_coll else (1.0 if not have_img else float(coll_share))

    def tilted_pick(pool, m, alpha):
        w = torch.exp(alpha * (z[pool] - z[pool].max()))
        w = torch.nan_to_num(w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)
        return pool[torch.multinomial(w, m, replacement=True, generator=g)]

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
                if u < share:
                    ci = int(torch.multinomial(
                        torch.exp(alpha * (z[centers] - z[centers].max())).clamp_min(1e-8),
                        1, replacement=True, generator=g).item())
                    parts.append(tilted_pick(coll_pools[ci], m, alpha))
                else:
                    gi = int(torch.multinomial(img_sizes, 1, replacement=True,
                                               generator=g).item())
                    parts.append(tilted_pick(img_pools[gi], m, alpha))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
