TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board, measured on the deep windows where analytic shortcuts run out.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'target': Expose the pure pair make_target(z_obs, z_prev) and to_obs(pred, z_prev) — a target transform and its EXACT inverse back into the fixed observation space. Or set LEARNED = True and expose make(D) -> an nn.Module with the same pair plus reg; its parameters are registered on the net and trained jointly, and it is evaluated through its own inverse, so a collapsed learned target cannot reconstruct and is scored down.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any target impl: expose two pure functions
  make_target(z_obs, z_prev) -> the tensor the model is TRAINED to predict, per cmd step;
      z_obs = true next-obs embedding [n,768], z_prev = previous obs embedding [n,768], zeros
      at the first step.
  to_obs(pred, z_prev) -> the EXACT inverse: a predicted next-obs embedding [n,768] in the fixed
      observation space, for the retrieval eval.
Both must be pure functions of their args. Or set LEARNED = True and expose make(D) -> nn.Module
with the same pair plus reg()."""

NAME = "identity"
DESCRIPTION = "Predict the next observation embedding directly (R4 default)."


def make_target(z_obs, z_prev):
    return z_obs


def to_obs(pred, z_prev):
    return pred
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-00-baseline
  its fitness       +0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           mse
  arch                baseline_transformer
  optim               baseline_adamw
  target              identity
  batcher             baseline_uniform
  stream              baseline_interleave
  head                baseline_passthrough

YOUR PARENT'S CURRENT target IMPL — identity (this is the code you are mutating):
--------------------------------------------------------------------------------
"""Contract for any target impl: expose two pure functions
  make_target(z_obs, z_prev) -> the tensor the model is TRAINED to predict, per cmd step;
      z_obs = true next-obs embedding [n,768], z_prev = previous obs embedding [n,768], zeros
      at the first step.
  to_obs(pred, z_prev) -> the EXACT inverse: a predicted next-obs embedding [n,768] in the fixed
      observation space, for the retrieval eval.
Both must be pure functions of their args. Or set LEARNED = True and expose make(D) -> nn.Module
with the same pair plus reg()."""

NAME = "identity"
DESCRIPTION = "Predict the next observation embedding directly (R4 default)."


def make_target(z_obs, z_prev):
    return z_obs


def to_obs(pred, z_prev):
    return pred
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the swapped content [withheld]. Next-obs retrieval health [withheld].
comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the sw

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r19_obspresent_imagination_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot. Invent a genuinely different mechanism, or a novel RECOMBINATION of ideas already in the archive. Do not resubmit an impl you were shown. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before is NOT off-limits. Evolution recombines: a trait that failed ALONE can win in a CHANGED context. You MAY retry a past idea in a new context; argue why the change could flip it.
- LOOK OUTSIDE MACHINE LEARNING — the largest gains in this line of work have come from cross-domain lenses. Read beyond machine learning (neuroscience: predictive coding, hippocampal and episodic memory, place and grid cells; biology; physics; information theory) and translate ONE concrete mechanism into code — equations, not metaphor.
- IGNORE ANY PERFORMANCE FIGURE YOU FIND IN IMPL SOURCE. Some carried mechanisms document measurements taken on a different objective and a different data root, and those numbers mean nothing for what is scored here. Read that source for its MECHANISM — the equations, the interface, what it does and why — and never as a target to match or beat.
- TRACK CONTENT, NOT DISTURBANCE — the scored windows are exactly the ones where knowing THAT something moved is not enough. A mechanism that marks a location as touched, or that keys on the name being asked about, or on which item moved first or last, cancels to zero by construction. Only carrying an item's identity across several hops earns anything.
- NEVER touch the eval, the metric, the split, or the no-leakage guard. The harness re-checks, and a violation makes the candidate unusable regardless of its number.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0037 (g0-00-baseline, full budget, inner split).
The unmodified baseline scores +0.0037 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
