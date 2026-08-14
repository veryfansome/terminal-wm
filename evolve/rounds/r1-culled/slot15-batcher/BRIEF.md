TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board, measured on the deep windows where analytic shortcuts run out.

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
  id                g0-11-masked-endpoint-trunk
  its fitness       +0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r22_masked_endpoint_trunk_task

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

PARENT'S EVAL FEEDBACK: comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the swapped content [withheld]. Next-obs retrieval health [withheld].
comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the sw

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r20_interventional_gain_shell_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r20_interventional_gain_shell_consistency

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot. Invent a genuinely different mechanism, or a novel RECOMBINATION of ideas already in the archive. Do not resubmit an impl you were shown. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before is NOT off-limits. Evolution recombines: a trait that failed ALONE can win in a CHANGED context. You MAY retry a past idea in a new context; argue why the change could flip it.
- LOOK OUTSIDE MACHINE LEARNING — the largest gains in this line of work have come from cross-domain lenses. Read beyond machine learning (neuroscience: predictive coding, hippocampal and episodic memory, place and grid cells; biology; physics; information theory) and translate ONE concrete mechanism into code — equations, not metaphor.
- IGNORE ANY PERFORMANCE FIGURE YOU FIND IN IMPL SOURCE. Some carried mechanisms document measurements taken on a different objective and a different data root, and those numbers mean nothing for what is scored here. Read that source for its MECHANISM — the equations, the interface, what it does and why — and never as a target to match or beat.
- TRACK CONTENT, NOT DISTURBANCE — the scored windows are exactly the ones where knowing THAT something moved is not enough. A mechanism that marks a location as touched, or that keys on the name being asked about, or on which item moved first or last, cancels to zero by construction. Only carrying an item's identity across several hops earns anything.
- NEVER touch the eval, the metric, the split, or the no-leakage guard. The harness re-checks, and a violation makes the candidate unusable regardless of its number.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0037 (g0-11-masked-endpoint-trunk, full budget, inner split).
The unmodified baseline scores +0.0037 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
