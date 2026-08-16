TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

THE CONTRACT — axis 'batcher': Expose make_batcher(fit, bs, seed, **params) -> next_batch(step, total_steps) returning a list of indices of length bs. Must be deterministic given the seed, must own a private generator rather than touching global randomness, and must never mutate the sequences it is handed. This axis composes the in-batch negative pool.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any batcher impl:
    make_batcher(fit, bs, seed, **params) -> next_batch(step, total_steps) -> list[int]
      - fit: the train-split sequence dicts (read-only; may use metadata like s["image"]).
      - returns exactly bs integer indices in [0, len(fit)); called once per training step
        with step in [1, total_steps].
      - must be deterministic given seed, must not mutate fit, and must not touch the
        global torch RNG (own a private torch.Generator).
"""

import torch

NAME = "baseline_uniform"
DESCRIPTION = "Uniform iid sequence sampling; bit-identical to the pre-axis harness RNG stream."


# This impl must stay bit-identical to the pre-axis harness: same generator seeding, same
# randint call, same call order, so archived fitnesses replay exactly.
def make_batcher(fit, bs, seed):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def next_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    return next_batch
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                r6-28-role-unbind-provenance-chase
  its fitness       +0.6142   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r5_decision_direction_metric_quotient
  arch                r6_role_unbind_provenance_chase   params {"d": 192, "heads": 4, "hops": 6, "layers": 4, "null_init": 0.2, "readout_init": 1.5, "recency_init": 0.02, "role_scale": 2.0, "role_thr": 1.6, "stop_scale": 8.0, "stop_thr": 1.6, "temp_init": 0.025}
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              r4_role_bound_path_atom_codes
  head                r2_dualaddress_move_transport

YOUR PARENT'S CURRENT batcher IMPL — r6_sysblock_hardneg_curriculum (this is the code you are mutating):
--------------------------------------------------------------------------------
import torch

NAME = "r6_sysblock_hardneg_curriculum"
DESCRIPTION = (
    "Anneal batch composition from uniform to partially image-blocked (a few systems per "
    "batch), densifying same-system in-batch negatives for the contrastive objectives to "
    "match the eval's same-verb same-system foil geometry."
)


def make_batcher(fit, bs, seed, n_block_images=3, hard_frac_max=0.5, ramp_frac=0.4):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    if len(pools) < 2 or hard_frac_max <= 0.0:
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    k_imgs = max(1, min(int(n_block_images), len(pools)))

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld]; cd-pack [withheld] (not scored).
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld]; cd-pack [withheld] (not scored).
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld]; cd-pack [withheld] (not scored).

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r2_routedepth_tilt_readcollision_blocks (axis batcher)
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

--- r6_matchedrisk_ringband_blocks (axis batcher)
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

--- r5_depth_stratified_epoch_reshuffle_blocks (axis batcher)
import math
import re

import torch

NAME = "r5_depth_stratified_epoch_reshuffle_blocks"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic file-content tracker "
    "(mv / cp / cat-redirect move or copy content between paths, a bare read returns whatever "
    "now sits at its path) and labels the sequence by the hop count of its deepest routed read, "
    "which partitions the training set into disjoint hop-count strata. Every batch reserves a "
    "uniform base share and allocates the remaining slots across strata as exact integer quotas "
    "by largest-remainder rounding of the stratum sizes tilted by exp(beta*hops), with the tilt "
    "annealed up from zero over the opening fraction of training, each quota clamped so no "
    "stratum is visited more than a fixed multiple of the per-sequence rate uniform sampling "
    "would give, and strata smaller than a floor excluded from the quota entirely. Stratum slots "
    "are filled from a per-stratum shuffled epoch cursor, so drawing inside a stratum is without "
    "replacement until that stratum is exhausted and then reshuffled. With annealed probability a "
    "run of slots is instead filled from a collision block: several distinct sequences whose "
    "deepest routed read shares one image, one digit-normalized path shape and one hop count, "
    "pre-filtered once at setup under a fixed random projection so that block members agree on "
    "the read-command embedding and disagree on the read-answer embedding. Falls back to uniform "
    "sampling when no routed read is recoverable."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
_READ_VERBS = ("cat", "head", "tail")


def _norm_path(path):
    return _NUM.sub("#", _HEX.sub("#", path))


def _parse(cmd):
    toks = cmd.split()
    if not toks:
        return None
    verb = toks[0]
    args = []
    redir = None
    rtarget = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in ("|", "&&", "||", ";", "2>", "2>>"):
            return None
        if t in (">", ">>"):
            redir = t
            if i + 1 < len(toks):
                rtarget = toks[i + 1]
            i += 2
            continue
        if t.startswith(">>"):
            redir = ">>"
            rtarget = t[2:] or None
            i += 1
            continue
        if t.startswith(">"):
            redir = ">"
            rtarget = t[1:] or None
            i += 1
            continue
        if t.startswith("-"):
            i += 1
            continue
        args.append(t)
        i += 1
    if verb == "mv" and redir is None and len(args) == 2:
        return ("move", args[0], args[1])
    if verb == "cp" and redir is None and len(args) == 2:
        return ("copy", args[0], args[1])
    if verb in _READ_VERBS and len(args) == 1:
        if redir == ">" and rtarget:
            return ("copy", args[0], rtarget)
        if redir == ">>" and rtarget:
            return ("append", args[0], rtarget)
        if redir is None:
            return ("read", args[0], None)
    return None


def _deepest_routed_read(cmds, depth_cap):
    hops = {}
    best_d = 0
    best_t = -1
    best_path = ""
    count = 0
    for t, cmd in enumerate(cmds):
        ev = _parse(cmd)
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
            d = hops.get(a, 0)
            if d >= 1:
                count += 1
                dd = min(d, depth_cap)
                if dd > best_d:
                    best_d = dd
                    best_t = t
                    best_path = a
    return best_d, best_t, best_path, count


def _refine_block(fit, members, rmat, sim_thresh, ans_max_sim):
    rows = []
    qv = []
    av = []
    for i, t in members:
        zc = fit[i].get("z_cmd")
        zo = fit[i].get("z_obs")
        if zc is None or zo is None or zc.dim() != 2 or zo.dim() != 2:
            return members
        if t >= int(zc.shape[0]) or t >= int(zo.shape[0]):
            continue
        rows.append((i, t))
        qv.append(zc[t].detach().float())
        av.append(zo[t].detach().float())
    if len(rows) < 2:
        return members
    q = torch.nn.functional.normalize(torch.stack(qv) @ rmat, dim=1, eps=1e-8)
    a = torch.nn.functional.normalize(torch.stack(av) @ rmat, dim=1, eps=1e-8)
    qsim = (q @ q[0:1].t()).view(-1).tolist()
    asim = (a @ a.t()).tolist()
    keep = []
    kept_rows = []
    for r in range(len(rows)):
        if r > 0 and qsim[r] < sim_thresh:
            continue
        dup = False
        for kr in kept_rows:
            if asim[r][kr] > ans_max_sim:
                dup = True
                break
        if dup:
            continue
        keep.append(rows[r])
        kept_rows.append(r)
    if len(keep) < 2:
        return rows
    return keep


def make_batcher(fit, bs, seed, base_frac=0.25, depth_beta=1.2, ramp_frac=0.3,
                 block_frac_max=0.5, group_size=4, max_repeat=20.0, min_stratum=16,
                 depth_cap=4, bucket_cap=32, max_filter_buckets=1200, proj_dim=128,
                 sim_thresh=0.6, ans_max_sim=0.995):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n <= 1 or bs <= 0:
        return uniform_batch

    dcap = max(1, int(depth_cap))
    depth = [0] * n
    rcount = [0] * n
    keyed = {}
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        d, t, path, c = _deepest_routed_read(cmds, dcap)
        depth[i] = d
        rcount[i] = c
        if d >= 1 and t >= 0:
            keyed.setdefault((str(s.get("image", "?")), _norm_path(path), d), []).append((i, t))

    strata = [[] for _ in range(dcap + 1)]
    for i in range(n):
        strata[depth[i]].append(i)
    sizes = [len(x) for x in strata]
    if sizes[0] == n:
        return uniform_batch

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0x7FFFFFFF)
    dim = 0
    for s in fit:
        zc = s.get("z_cmd")
        if zc is not None and zc.dim() == 2 and int(zc.shape[0]) > 0:
            dim = int(zc.shape[1])
            break
    rmat = None
    if dim > 0:
        pdim = max(8, min(int(proj_dim), dim))
        rmat = torch.randn(dim, pdim, generator=g_setup)

    blocks = [[] for _ in range(dcap + 1)]
    block_member_w = [[] for _ in range(dcap + 1)]
    n_filtered = 0
    for key in sorted(keyed):
        members = keyed[key]
        if len(members) < 2:
            continue
        cap = max(2, int(bucket_cap))
        if len(members) > cap:
            sel = torch.randperm(len(members), generator=g_setup)[:cap].tolist()
            members = [members[j] for j in sorted(sel)]
        if rmat is not None and n_filtered < int(max_filter_buckets):
            n_filtered += 1
            members = _refine_block(fit, members, rmat, float(sim_thresh), float(ans_max_sim))
        if len(members) < 2:
            continue
        d = int(key[2])
        blocks[d].append(torch.tensor([m[0] for m in members], dtype=torch.long))
        block_member_w[d].append(
            torch.tensor([1.0 + float(rcount[m[0]]) for m in members], dtype=torch.float))

    block_sel_w = [None] * (dcap + 1)
    for d in range(dcap + 1):
        if blocks[d]:
            block_sel_w[d] = torch.tensor([float(b.numel()) ** 0.5 for b in blocks[d]],
                                          dtype=torch.float)

    stratum_idx = [torch.tensor(x, dtype=torch.long) for x in strata]
    perms = [None] * (dcap + 1)
    pos = [0] * (dcap + 1)

    def pull(d):
        if sizes[d] == 0:
            return int(torch.randint(0, n, (1,), generator=g).item())
        p = perms[d]
        if p is None or pos[d] >= int(p.numel()):
            perms[d] = torch.randperm(sizes[d], generator=g)
            pos[d] = 0
            p = perms[d]
        v = int(stratum_idx[d][int(p[pos[d]])])
        pos[d] += 1
        return v

    n_base = max(0, min(bs, int(round(bs * float(base_frac)))))
    n_str = bs - n_base
    gsize = max(2, int(group_size))
    emp = torch.tensor([float(x) for x in sizes], dtype=torch.float)
    lev = torch.arange(dcap + 1, dtype=torch.float)
    floor_n = max(2, int(min_stratum))
    caps = []
    for d in range(dcap + 1):
        if sizes[d] < floor_n:
            caps.append(0)
        elif d == 0:
            caps.append(min(sizes[d], bs))
        else:
            room = int(float(max_repeat) * float(bs) * float(sizes[d]) / float(n))
            caps.append(min(sizes[d], bs, max(1, room)))

    def next_batch(step, total_steps):
        ts = max(1, int(total_steps))
        ramp = max(1, int(float(ramp_frac) * float(ts)))
        r = min(1.0, float(step) / float(ramp))

        counts = [0] * (dcap + 1)
        if n_str > 0:
            w = emp * torch.exp(lev * float(depth_beta) * r)
            tot = float(w.sum())
            if tot <= 0.0:
                counts[0] = n_str
            else:
                p = (w / tot) * float(n_str)
                pl = p.tolist()
                counts = [int(math.floor(x)) for x in pl]
                rem = n_str - sum(counts)
                order = sorted(range(dcap + 1), key=lambda d: (-(pl[d] - counts[d]), -d))
                for k in range(max(0, rem)):
                    counts[order[k % len(order)]] += 1

        surplus = 0
        for d in range(dcap + 1):
            if counts[d] > caps[d]:
                surplus += counts[d] - caps[d]
                counts[d] = caps[d]
        for d in range(dcap, -1, -1):
            if surplus <= 0:
                break
            room = caps[d] - counts[d]
            if room > 0:
                take = min(room, surplus)
                counts[d] += take
                surplus -= take

        block_frac = float(block_frac_max) * r
        out = []
        used = set()
        for d in range(dcap, -1, -1):
            need = counts[d]
            if need <= 0:
                continue
            got = 0
            attempts = 0
            limit = 4 * need + 16
            while got < need and attempts < limit:
                attempts += 1
                use_block = bool(blocks[d]) and \
                    float(torch.rand((1,), generator=g).item()) < block_frac
                if use_block:
                    bi = int(torch.multinomial(block_sel_w[d], 1, generator=g).item())
                    mem = blocks[d][bi]
                    k = min(gsize, int(mem.numel()), need - got)
                    pick = torch.multinomial(block_member_w[d][bi], k, replacement=False,
                                             generator=g)
                    for j in pick.tolist():
                        v = int(mem[j])
                        if v in used:
                            continue
                        out.append(v)
                        used.add(v)
                        got += 1
                        if got >= need:
                            break
                else:
                    v = pull(d)
                    if v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    got += 1

        tries = 0
        while len(out) < bs:
            v = int(torch.randint(0, n, (1,), generator=g).item())
            tries += 1
            if v in used and tries < 8 * bs:
                continue
            out.append(v)
            used.add(v)
        out = out[:bs]
        order = torch.randperm(bs, generator=g).tolist()
        return [out[i] for i in order]

    return next_batch

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.6142 (r6-28-role-unbind-provenance-chase, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
