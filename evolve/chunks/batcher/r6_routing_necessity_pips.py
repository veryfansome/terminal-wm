import math

import torch

NAME = "r6_routing_necessity_pips"
DESCRIPTION = (
    "Weights every training sequence by ROUTING NECESSITY, a parser-free geometric quantity read "
    "straight off the frozen embeddings. Inside one sequence, observations that are near-duplicates "
    "of many others (the identical empty render every silent mv and every empty ls produces) are "
    "marked uninformative and are excluded both as candidate answers and as scored positions. For "
    "every remaining step t, e_ref is the smallest squared distance from its observation to any "
    "earlier informative observation (how well the answer is obtainable by carrying some earlier "
    "content forward) and e_bias is the squared distance to the observation of the earlier "
    "informative command whose embedding is most cosine-similar to the command at t (what a "
    "name-lookup on the read path returns). Necessity is the product of two smooth gates, "
    "w = kr/(e_ref+kr) * e_bias/(e_bias+kb), with kr and kb fixed fractions of that sequence's own "
    "mean pairwise observation distance, so it is scale free: w is high exactly where the answer is "
    "some earlier content but the nearest command name does not point at it, near zero on repeats, "
    "on unanswerable filler and on reads a name lookup already solves. The sequence weight is the "
    "sum of its top-k step necessities, raised to alpha over a floor that keeps full support. "
    "Batches are drawn by pi-ps SYSTEMATIC sampling instead of multinomial draws: a uniform "
    "candidate pool per step, Hajek clipping of within-pool inclusion probabilities to 1/bs (which "
    "bounds the importance ratio at pool_mult and makes the batch's indices distinct by "
    "construction), one cumulative pass, and bs equally spaced sample points whose shared offset "
    "walks the golden-ratio Weyl sequence, so each batch's realized composition matches the target "
    "distribution rather than matching it in expectation. No annealing and no image blocking: the "
    "tilt is live for the whole step budget, and same-system negatives are already dense inside "
    "every single sequence."
)

_PHI = 0.6180339887498949
_CAP_ITERS = 12
_CHUNK = 256


def _necessity(obs, cmd, kappa_ref, kappa_bias, dup_tol, dup_frac, top_k):
    n = int(obs.shape[0])
    if n < 3:
        return 0.0
    osq = (obs * obs).sum(dim=1)
    d2 = (osq.unsqueeze(1) + osq.unsqueeze(0) - 2.0 * (obs @ obs.t())).clamp_min(0.0)
    scale = float(d2.mean())
    if not math.isfinite(scale) or scale <= 1e-12:
        return 0.0

    dup = (d2 < dup_tol * scale).sum(dim=1)
    informative = dup <= max(3, int(dup_frac * n))

    unit = cmd / cmd.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sim = unit @ unit.t()

    pos = torch.arange(n)
    cand = (pos.unsqueeze(1) > pos.unsqueeze(0)) & informative.unsqueeze(0)
    has_cand = cand.any(dim=1)

    e_ref = torch.where(cand, d2, torch.full_like(d2, float("inf"))).amin(dim=1)
    j_star = torch.where(cand, sim, torch.full_like(sim, -2.0)).argmax(dim=1)
    e_bias = d2.gather(1, j_star.unsqueeze(1)).squeeze(1)

    kr = kappa_ref * scale
    kb = kappa_bias * scale
    w = (kr / (e_ref + kr)) * (e_bias / (e_bias + kb))
    w = torch.where(has_cand & informative, w, torch.zeros_like(w))
    w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    return float(torch.topk(w, max(1, min(int(top_k), n))).values.sum())


def _capped(w, cap):
    for _ in range(_CAP_ITERS):
        over = w > cap
        if not bool(over.any()):
            break
        excess = float((w[over] - cap).sum())
        w = torch.where(over, torch.full_like(w, cap), w)
        free = ~over
        room = float(w[free].sum())
        if room <= 1e-12 or excess <= 1e-12:
            break
        w = torch.where(free, w * (1.0 + excess / room), w)
    return w.clamp_min(1e-12)


def _score_split(fit, proj, dim, kappa_ref, kappa_bias, dup_tol, dup_frac, top_k):
    n = len(fit)
    score = torch.zeros(n, dtype=torch.float32)
    for start in range(0, n, _CHUNK):
        stop = min(n, start + _CHUNK)
        keep = []
        for i in range(start, stop):
            s = fit[i]
            zo = s.get("z_obs")
            zc = s.get("z_cmd")
            if zo is None or zc is None or zo.dim() != 2 or zc.dim() != 2:
                continue
            if zo.shape[0] != zc.shape[0] or zo.shape[1] != dim or zc.shape[1] != dim:
                continue
            if int(zo.shape[0]) < 3:
                continue
            keep.append((i, int(zo.shape[0]), zo, zc))
        if not keep:
            continue
        obs_all = torch.cat([k[2] for k in keep], dim=0).detach().to(torch.float32) @ proj
        cmd_all = torch.cat([k[3] for k in keep], dim=0).detach().to(torch.float32) @ proj
        off = 0
        for i, ln, _zo, _zc in keep:
            score[i] = _necessity(obs_all[off:off + ln], cmd_all[off:off + ln],
                                  kappa_ref, kappa_bias, dup_tol, dup_frac, top_k)
            off += ln
    return score


def make_batcher(fit, bs, seed, alpha=2.0, w_floor=0.15, pool_mult=16, kappa_ref=0.10,
                 kappa_bias=0.50, dup_tol=0.02, dup_frac=0.20, top_k=3, proj_dim=96):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 4 or bs < 2:
        return uniform_batch

    z0 = fit[0].get("z_obs")
    if z0 is None or z0.dim() != 2:
        return uniform_batch
    dim = int(z0.shape[1])
    rdim = max(8, min(int(proj_dim), dim))

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x5DEECE66) & 0x7FFFFFFF)
    proj = torch.randn(dim, rdim, generator=g_setup) / math.sqrt(float(rdim))
    offset0 = float(torch.rand((1,), generator=g_setup).item())

    score = _score_split(fit, proj, dim, float(kappa_ref), float(kappa_bias),
                         float(dup_tol), float(dup_frac), int(top_k))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(score.max()) <= 0.0:
        return uniform_batch

    floor = float(w_floor) * float(score.mean().clamp_min(1e-8))
    weight = (score + floor).pow(float(alpha))
    weight = torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-12)

    if n < bs:
        def small_batch(step, total_steps):
            return torch.multinomial(weight, bs, replacement=True, generator=g).tolist()
        return small_batch

    pool_size = int(min(n, max(bs, int(pool_mult) * bs)))
    cap = 1.0 / float(bs)
    grid = torch.arange(bs, dtype=torch.float64)

    def next_batch(step, total_steps):
        pool = torch.randperm(n, generator=g)[:pool_size]
        w = weight[pool].double()
        w = w / w.sum().clamp_min(1e-30)
        w = _capped(w, cap)
        cdf = torch.cumsum(w, dim=0)
        cdf[-1] = 1.0
        u = math.fmod(offset0 + float(step) * _PHI, 1.0)
        pts = ((grid + u) / float(bs)).clamp(0.0, 1.0 - 1e-9)
        slot = torch.searchsorted(cdf, pts).clamp_(0, int(pool.numel()) - 1)
        picked = pool[slot]
        return picked[torch.randperm(bs, generator=g)].tolist()

    return next_batch
