TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

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
  id                r4-13-zca-shrunk-target
  its fitness       +0.0262   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r22_prefix_content_xattention
  optim               r18_spectral_capped_transition_readout
  target              zca_shrunk_target_whitening
  batcher             r2_routedepth_tilt_readcollision_blocks   params {"alpha_max": 2.5, "ans_max_sim": 0.995, "chunk": 256, "coll_share": 0.5, "hard_frac_max": 0.75, "max_centers": 8000, "n_blocks": 2, "nb_k": 16, "proj_dim": 128, "ramp_frac": 0.3, "sim_thresh": 0.85}
  stream              r3_srcdst_hashslot_coding
  head                r2_dualaddress_move_transport   params {"pred_weight": 0.0}

YOUR PARENT'S CURRENT batcher IMPL — r2_routedepth_tilt_readcollision_blocks (this is the code you are mutating):
--------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r22_exact_target_equivalence_quotient
  arch                r4_complement_address_transport
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r2_routed_collision_depth_batcher   params {"depth_beta": 0.8, "group_size": 4, "hard_frac_max": 0.5, "ramp_frac": 0.3, "size_pow": 0.5}
  stream              baseline_interleave
  head                r3_occupancy_routed_copy_transport

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r6_sysblock_hardneg_curriculum (axis batcher)
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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0262 (r4-13-zca-shrunk-target, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
