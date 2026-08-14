import re

import torch

D_FEAT = 768

NAME = "r6_matchedrisk_ringband_blocks"
DESCRIPTION = (
    "Matched risk-set batching, imported from matched case-control designs: the conditional "
    "likelihood over a matched set cancels every covariate the set was matched on, so only the "
    "un-matched factor can explain who is who. Each training sequence is reduced to its deepest "
    "mv-routed read, parsed from command strings alone by walking a 'cat P' backwards through the "
    "mv edges that wrote P earlier in the same trajectory. A sequence's stratum is (hop count "
    "capped at four, canonical shape of the read path with trailing numeric copy-suffixes stripped "
    "and digit runs collapsed) — so within a stratum the location being read and the number of "
    "moves are held constant and cannot discriminate. Inside a stratum, neighbours of a centre are "
    "ranked by the training objective's OWN confusability ring evaluated between the two read "
    "answers, exp(-tt/lam)*(1-exp(-tt/delta)) on standardized target embeddings under a "
    "variance-preserving Gaussian projection, with lam set from a sampled estimate of the mean "
    "off-diagonal target distance exactly as the loss sets it; neighbours are kept only inside the "
    "ring's pass-band, a fixed fraction of its analytic peak at t* = delta*log(1+lam/delta), which "
    "admits close-but-distinct answers and rejects both near-duplicates and far-apart answers. A "
    "batch is a uniform iid base part plus a few blocks; a block is a centre plus a WITHOUT-"
    "REPLACEMENT draw of its matched neighbours, so the slots hold distinct sequences rather than "
    "repeats. Block strata are chosen off a fixed depth wheel that spends two thirds of the blocked "
    "slots on hop counts three and above. The blocked fraction ramps from zero, so the opening of "
    "training is the plain uniform distribution; falls back to system-image blocks and then to "
    "uniform when no matched set can be built."
)

_READ_VERB = "cat"
_N_DEPTH_CLASS = 4
_WHEEL = (0, 1, 2, 3, 2, 3)
_NUM_SUFFIX = re.compile(r"\.\d+$")
_DIGIT_RUN = re.compile(r"\d+")
_LAM_FRAC = 0.5
_DELTA = 0.05
_MAX_HOPS = 64


def _canon_path(path):
    p = path
    while True:
        q = _NUM_SUFFIX.sub("", p)
        if q == p:
            break
        p = q
    return _DIGIT_RUN.sub("#", p)


def _deepest_routed_read(cmds):
    mvs = []
    for t, c in enumerate(cmds):
        parts = c.split()
        if len(parts) == 3 and parts[0] == "mv":
            mvs.append((t, parts[1], parts[2]))
    if not mvs:
        return None
    wrote = {}
    for t, src, dst in mvs:
        wrote.setdefault(dst, []).append((t, src))
    best = None
    for t, c in enumerate(cmds):
        parts = c.split()
        if len(parts) != 2 or parts[0] != _READ_VERB:
            continue
        cur, ct, hops = parts[1], t, 0
        while hops < _MAX_HOPS:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            hops += 1
        if hops >= 1 and (best is None or hops >= best[0]):
            best = (hops, t, parts[1])
    return best


def _scan(fit):
    rows = []
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        zo = s.get("z_obs")
        if zo is None or not cmds:
            continue
        got = _deepest_routed_read(list(cmds))
        if got is None:
            continue
        hops, read_t, read_path = got
        if read_t < 0 or read_t >= int(zo.shape[0]) or read_t >= len(cmds):
            continue
        cls = min(int(hops), _N_DEPTH_CLASS) - 1
        rows.append((i, read_t, cls, _canon_path(read_path)))
    return rows


def _projection(dim, proj_dim, gen):
    pd = max(8, min(int(proj_dim), int(dim)))
    return torch.randn(int(dim), pd, generator=gen) / float(pd) ** 0.5


def _sq_tt(a, b):
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True)
    return (a2 + b2.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(D_FEAT)


def _mean_offdiag_tt(fit, rmat, gen, stat_n):
    picks = []
    for i, s in enumerate(fit):
        zo = s.get("z_obs")
        if zo is None or int(zo.shape[0]) == 0:
            continue
        picks.append((i, int(zo.shape[0])))
    if not picks:
        return None
    seq_ids = torch.tensor([p[0] for p in picks], dtype=torch.long)
    lens = torch.tensor([p[1] for p in picks], dtype=torch.long)
    m = min(int(stat_n), 4 * int(seq_ids.numel()))
    if m < 8:
        m = min(8, int(seq_ids.numel()))
    rs = torch.randint(0, int(seq_ids.numel()), (m,), generator=gen)
    vecs = []
    for k in range(int(rs.numel())):
        j = int(rs[k])
        n_steps = int(lens[j])
        if n_steps <= 0:
            continue
        t = int(torch.randint(0, n_steps, (1,), generator=gen).item())
        vecs.append(fit[int(seq_ids[j])]["z_obs"][t].detach().float())
    if len(vecs) < 8:
        return None
    y = torch.stack(vecs) @ rmat
    tt = _sq_tt(y, y)
    k = int(y.shape[0])
    total = float(tt.sum().item()) - float(tt.diagonal().sum().item())
    return total / max(1.0, float(k * (k - 1)))


def _ring(tt, lam, delta):
    return torch.exp(-tt / lam) * (1.0 - torch.exp(-tt / delta))


def _ring_peak(lam, delta):
    t_star = delta * float(torch.log(torch.tensor(1.0 + lam / delta)).item())
    a = float(torch.exp(torch.tensor(-t_star / lam)).item())
    b = 1.0 - float(torch.exp(torch.tensor(-t_star / delta)).item())
    return max(1e-8, a * b)


def _matched_pools(fit, rows, gen, proj_dim, nb_k, ring_floor, stratum_cap, stat_n):
    if len(rows) < 4:
        return []
    zo0 = fit[rows[0][0]]["z_obs"]
    dim = int(zo0.shape[1])
    rmat = _projection(dim, proj_dim, gen)
    mean_off = _mean_offdiag_tt(fit, rmat, gen, stat_n)
    if mean_off is None or mean_off <= 1e-8:
        return []
    lam = max(1e-6, _LAM_FRAC * mean_off)
    peak = _ring_peak(lam, _DELTA)

    strata = {}
    for r in rows:
        strata.setdefault((r[2], r[3]), []).append(r)

    raw = []
    for key in sorted(strata):
        members = strata[key]
        if len(members) < 2:
            continue
        sel = list(range(len(members)))
        if len(sel) > int(stratum_cap):
            perm = torch.randperm(len(sel), generator=gen)[: int(stratum_cap)]
            sel = sorted(int(x) for x in perm)
        gidx = torch.tensor([members[j][0] for j in sel], dtype=torch.long)
        answers = torch.stack(
            [fit[members[j][0]]["z_obs"][members[j][1]].detach().float() for j in sel])
        y = answers @ rmat
        ring = _ring(_sq_tt(y, y), lam, _DELTA)
        k = int(y.shape[0])
        ring[torch.arange(k), torch.arange(k)] = -1.0
        kk = max(1, min(int(nb_k), k - 1))
        val, idx = torch.topk(ring, kk, dim=1)
        for r in range(k):
            raw.append((int(gidx[r]), gidx[idx[r]], val[r], key[0]))

    def assemble(floor):
        out = []
        for center, nb_idx, nb_val, cls in raw:
            keep = nb_val >= floor
            nb = nb_idx[keep]
            if int(nb.numel()) >= 1:
                out.append((torch.cat([torch.tensor([center], dtype=torch.long), nb]), cls))
        return out

    pools = assemble(float(ring_floor) * peak)
    if len(pools) < 8:
        pools = assemble(0.0)
    return pools


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, n_blocks=4, nb_k=24,
                 ring_floor=0.25, proj_dim=192, stratum_cap=1024, stat_n=2048,
                 n_block_images=1):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or float(hard_frac_max) <= 0.0:
        return uniform_only()

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0x7FFFFFFF)
    pools = _matched_pools(fit, _scan(fit), g_setup, proj_dim, nb_k, ring_floor,
                           stratum_cap, stat_n)

    by_class = [[] for _ in range(_N_DEPTH_CLASS)]
    for pool, cls in pools:
        if 0 <= cls < _N_DEPTH_CLASS:
            by_class[cls].append(pool)
    all_pools = [p for p, _c in pools]

    if not all_pools:
        by_image = {}
        for i, s in enumerate(fit):
            by_image.setdefault(s.get("image", "?"), []).append(i)
        img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
        if len(img_pools) < 2:
            return uniform_only()
        img_sizes = torch.tensor([float(p.numel()) for p in img_pools])
        k_imgs = max(1, min(int(n_block_images), len(img_pools)))

        def next_batch_img(step, total_steps):
            ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
            ramp = min(1.0, float(step) / ramp_steps)
            n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
            parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
            if n_hard > 0:
                chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([img_pools[int(c)] for c in chosen])
                parts.append(pool[torch.randint(0, int(pool.numel()), (n_hard,), generator=g)])
            batch = torch.cat(parts)
            return batch[torch.randperm(bs, generator=g)].tolist()

        return next_batch_img

    kb = max(1, int(n_blocks))

    def draw_block(pool, m):
        k = int(pool.numel())
        if k >= m:
            keep = torch.randperm(k - 1, generator=g)[: m - 1] + 1
            return torch.cat([pool[:1], pool[keep]])
        return torch.cat([pool, torch.randint(0, n, (m - k,), generator=g)])

    def pick_pool(cls):
        bucket = by_class[cls] if by_class[cls] else all_pools
        j = int(torch.randint(0, len(bucket), (1,), generator=g).item())
        return bucket[j]

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            base = n_hard // kb
            rem = n_hard - base * kb
            for j in range(kb):
                m = base + (1 if j < rem else 0)
                if m <= 0:
                    continue
                cls = _WHEEL[(int(step) + j) % len(_WHEEL)]
                parts.append(draw_block(pick_pool(cls), m))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
