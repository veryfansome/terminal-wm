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
  id                r6-17-occupancy-transport-name-decoy
  its fitness       +0.0150   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r6_occupancy_transport_name_decoy
  arch                r18_pathstate_latent_transition_worldmodel
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

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca -0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

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

--- r4_hopdepth_scaffold_stratified_quota (axis batcher)
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

--- r5_board_matched_role_contrast_blocks (axis batcher)
import math
import re

import torch

NAME = "r5_board_matched_role_contrast_blocks"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic content tracker that keeps "
    "content IDENTITY, not just hop counts: every read of an untouched path starts a content "
    "thread, every mv/cp/redirect carries a thread to a new path and increments its hop count, and "
    "the directories a thread touches are unioned so a trajectory splits into independent windows. "
    "For each sequence's deepest window the tracker emits two disjoint keys — a BOARD key (how many "
    "threads move, how many moves the window has, the window's maximum hop count, and the "
    "root/arity shape of the read path) holding what a role swap leaves fixed, and a ROLE key (the "
    "read thread's own hop count, its birth rank among the moving threads, and whether it is the "
    "thread that made the window's opening move) holding what a role swap permutes. "
    "Batches are a without-replacement (reshuffled) uniform base plus an annealed block share; a "
    "block is either a whole-system block over a few images or a bucket of sequences sharing one "
    "board key, from which members are drawn round-robin across DISTINCT role keys, so "
    "board-identical, role-differing sequences land in the same in-batch negative pool. Bucket "
    "choice is tilted toward deeper hop counts, larger buckets and more role-diverse buckets; "
    "member choice inside a role is tilted toward sequences with more routed reads. Degrades to "
    "the single-role bucket when a board carries only one role, to whole-system blocks when no "
    "bucket has two members, and to reshuffled uniform sampling when neither exists."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
_READ_VERBS = ("cat", "head", "tail")
_DEPTH_CAP = 6
_MOVE_CAP = 16
_PATH_CAP = 12
_ROOT_KEEP = 3


def _norm(text):
    return _NUM.sub("#", _HEX.sub("#", text))


def _shape(path):
    parts = [p for p in path.split("/") if p]
    root = "/".join(_norm(p) for p in parts[:_ROOT_KEEP])
    return (root, min(len(parts), _PATH_CAP))


def _dirname(path):
    i = path.rfind("/")
    if i > 0:
        return path[:i]
    return "/"


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
            rtarget = toks[i + 1] if i + 1 < len(toks) else None
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


def _trace(cmds):
    loc = {}
    hops = []
    birth = []
    touched = []
    moves = []
    reads = []

    def start(path, base, order):
        tid = len(hops)
        hops.append(int(base))
        birth.append(int(order))
        touched.append({_dirname(path)})
        loc[path] = tid
        return tid

    for i, cmd in enumerate(cmds):
        ev = _parse(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "move":
            tid = loc.pop(a, None)
            if tid is None:
                tid = start(a, 0, i)
                loc.pop(a, None)
            loc[b] = tid
            hops[tid] += 1
            touched[tid].add(_dirname(a))
            touched[tid].add(_dirname(b))
            moves.append((i, tid))
        elif kind == "copy" or kind == "append":
            src = loc.get(a)
            base = hops[src] if src is not None else 0
            if src is not None:
                touched[src].add(_dirname(a))
            tid = loc.get(b)
            if tid is None:
                tid = start(b, base + 1, i)
            elif base + 1 > hops[tid]:
                hops[tid] = base + 1
            touched[tid].add(_dirname(a))
            touched[tid].add(_dirname(b))
            moves.append((i, tid))
        else:
            tid = loc.get(a)
            if tid is None:
                start(a, 0, i)
            else:
                reads.append((i, tid, hops[tid], a))
    return hops, birth, touched, moves, reads


def _components(touched):
    n = len(touched)
    root = list(range(n))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    owner = {}
    for t in range(n):
        for d in touched[t]:
            o = owner.get(d)
            if o is None:
                owner[d] = t
            else:
                ra, rb = find(o), find(t)
                if ra != rb:
                    if ra < rb:
                        root[rb] = ra
                    else:
                        root[ra] = rb
    groups = {}
    for t in range(n):
        groups.setdefault(find(t), []).append(t)
    return groups


def _profile(cmds):
    hops, birth, touched, moves, reads = _trace(cmds)
    if not reads:
        return None, 0.0
    groups = _components(touched)
    home = {}
    for r, members in groups.items():
        for t in members:
            home[t] = r
    demand = 0.0
    best_per_comp = {}
    for (i, t, d, path) in reads:
        if d < 1:
            continue
        demand += float(min(d, _DEPTH_CAP))
        c = home[t]
        cur = best_per_comp.get(c)
        if cur is None or (d, i) > (cur[2], cur[0]):
            best_per_comp[c] = (i, t, d, path)
    if not best_per_comp:
        return None, 0.0
    pick_c = None
    pick_r = None
    for c in sorted(best_per_comp):
        rec = best_per_comp[c]
        if pick_r is None or (rec[2], rec[0]) > (pick_r[2], pick_r[0]):
            pick_c = c
            pick_r = rec
    rt, rd, rpath = pick_r[1], pick_r[2], pick_r[3]
    members = sorted(groups[pick_c])
    comp_moves = [(i, t) for (i, t) in moves if home.get(t) == pick_c]
    movers = [t for t in members if hops[t] >= 1]
    deepest = 0
    for t in movers:
        if min(hops[t], _DEPTH_CAP) > deepest:
            deepest = min(hops[t], _DEPTH_CAP)
    board = (len(movers), min(len(comp_moves), _MOVE_CAP), deepest, _shape(rpath))
    by_birth = sorted(movers, key=lambda t: (birth[t], t))
    birth_rank = by_birth.index(rt) if rt in by_birth else len(by_birth)
    own = [k for k, (i, t) in enumerate(comp_moves) if t == rt]
    opened = 1 if (own and own[0] == 0) else 0
    role = (min(rd, _DEPTH_CAP), min(birth_rank, _MOVE_CAP), opened)
    return (board, role), demand


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, group_size=12,
                 depth_beta=0.8, size_pow=0.75, role_pow=1.5, img_share=0.25,
                 n_block_images=1):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)
    cursor = {"buf": [], "pos": 0}

    def draw_uniform():
        if cursor["pos"] >= len(cursor["buf"]):
            cursor["buf"] = torch.randperm(max(1, n), generator=g).tolist()
            cursor["pos"] = 0
        v = int(cursor["buf"][cursor["pos"]])
        cursor["pos"] += 1
        return v

    def uniform_batch(step, total_steps):
        return [draw_uniform() for _ in range(bs)]

    if n <= 1 or bs <= 0 or float(hard_frac_max) <= 0.0 or int(group_size) < 2:
        return uniform_batch

    demand = [0.0] * n
    keyed = {}
    for i in range(n):
        s = fit[i]
        prof, dm = _profile(s.get("cmds") or [])
        demand[i] = dm
        if prof is None:
            continue
        board, role = prof
        keyed.setdefault(board, {}).setdefault(role, []).append(i)

    bucket_roles = []
    bucket_role_w = []
    weights = []
    for key in sorted(keyed, key=repr):
        rolemap = keyed[key]
        size = 0
        for v in rolemap.values():
            size += len(v)
        if size < 2:
            continue
        role_keys = sorted(rolemap, key=repr)
        deepest = float(key[2])
        bucket_roles.append([torch.tensor(rolemap[rk], dtype=torch.long) for rk in role_keys])
        bucket_role_w.append([
            torch.tensor([1.0 + demand[j] for j in rolemap[rk]], dtype=torch.float)
            for rk in role_keys])
        weights.append((float(size) ** float(size_pow))
                       * math.exp(float(depth_beta) * float(deepest))
                       * (float(len(role_keys)) ** float(role_pow)))

    by_image = {}
    for i in range(n):
        by_image.setdefault(str(fit[i].get("image", "?")), []).append(i)
    img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools], dtype=torch.float)

    have_blocks = len(bucket_roles) > 0
    have_images = len(img_pools) >= 2
    if not have_blocks and not have_images:
        return uniform_batch

    bucket_w = None
    if have_blocks:
        bucket_w = torch.tensor(weights, dtype=torch.float)
        bucket_w = torch.nan_to_num(bucket_w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)

    share = float(img_share)
    if share < 0.0:
        share = 0.0
    if share > 1.0:
        share = 1.0
    if not have_blocks:
        share = 1.0
    if not have_images:
        share = 0.0

    gsize = max(2, int(group_size))
    k_imgs = max(1, min(int(n_block_images), len(img_pools)))
    distinct = n >= bs
    block_cap = 4 * (bs // gsize + 2)
    fill_cap = 4 * bs + 16

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp_steps))
        n_hard = int(round(bs * frac))
        if n_hard < 0:
            n_hard = 0
        if n_hard > bs:
            n_hard = bs

        out = []
        used = set()
        blocks = 0
        while len(out) < n_hard and blocks < block_cap:
            blocks += 1
            k = min(gsize, n_hard - len(out))
            if k <= 0:
                break
            u = float(torch.rand((1,), generator=g).item())
            if u < share:
                chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([img_pools[int(c)] for c in chosen.tolist()])
                cand = pool[torch.randint(0, int(pool.numel()), (3 * k,), generator=g)].tolist()
                for v in cand:
                    v = int(v)
                    if distinct and v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    if len(out) >= n_hard:
                        break
            else:
                bi = int(torch.multinomial(bucket_w, 1, generator=g)[0])
                members = bucket_roles[bi]
                mweights = bucket_role_w[bi]
                nr = len(members)
                order = torch.randperm(nr, generator=g).tolist()
                taken = 0
                probe = 0
                limit = 3 * k + nr
                while taken < k and probe < limit:
                    ri = order[probe % nr]
                    probe += 1
                    j = int(torch.multinomial(mweights[ri], 1, generator=g)[0])
                    v = int(members[ri][j])
                    if distinct and v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    taken += 1
                    if len(out) >= n_hard:
                        break

        tries = 0
        while len(out) < bs and tries < fill_cap:
            tries += 1
            v = draw_uniform()
            if distinct and v in used:
                continue
            out.append(v)
            used.add(v)
        while len(out) < bs:
            out.append(draw_uniform())

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
Beat your parent's fitness of +0.0150 (r6-17-occupancy-transport-name-decoy, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
