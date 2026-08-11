TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board, measured on the deep windows where analytic shortcuts run out.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'stream': Owns the token layout. Expose collate(batch, device), extract_cmd_pred(pred_full, batch), flatten_predictions(net, seqs, device) and leakage_ok(net, device). Optionally expose extract_cmd_input(batch), which is what lets an objective opt into the causal context extension. leakage_ok is asserted before scoring and a stream that cannot demonstrate causality is rejected.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any stream impl:
  collate(batch, device) -> dict with tok [B,L,D], types [B,L] in {0,1}, key_pad [B,L] bool,
      tgt [B,maxn,D] (single-vector standardized next-obs target per STEP — the target/eval space
      is FIXED across streams), cmd_mask [B,maxn] bool
  extract_cmd_pred(pred_full [B,L,D], batch) -> [B,maxn,D], the prediction at each step's cmd token
  flatten_predictions(net, seqs, device) -> dict with at least pred/prev/true/cmds/verbs, step order
  leakage_ok(net, device) -> bool, a stream-aware causality probe (corrupt obs_t, cmd_<=t frozen);
      asserted before scoring, and a stream that cannot demonstrate causality is rejected
  extract_cmd_input(batch) -> [B,maxn,D] (optional), what lets an objective opt into WANTS_CTX
"""

import torch

from realenv import seq_worldmodel as M

NAME = "baseline_interleave"
DESCRIPTION = "Single-vector cmd/obs interleave; bit-identical to the pre-axis harness plumbing."


# This impl must stay bit-identical to the pre-axis harness: collate/flatten delegate to the
# seq_worldmodel functions the harness always called, and leakage_ok uses the same seed, toy
# sequence and perturbed index, so archived fitnesses replay exactly.
def collate(batch, device):
    return M.collate(batch, device)


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def flatten_predictions(net, seqs, device):
    return M.flatten_predictions(net, seqs, device)


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0  # corrupt obs_3 (odd index 2*3+1)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

# The cups scoring instrument pins this layout; a stream declaring a different one is
# refused before any GPU time is spent (see eval/adapter.py).
CUPS_LAYOUT = "interleave2"
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-14-pathstate-rssm
  its fitness       +0.0000   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

PARENT'S EVAL FEEDBACK: comp_ca [withheld] over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the swapped content [withheld]. Next-obs retrieval health [withheld].
comp_ca +0.0000 over n=89 deep earnable windows (for reference, the strongest analytic non-tracker on the same windows, h_first, sits at [withheld]) (d2 n=0, d3 n=60, d4+ n=29); native picks [withheld] vs chance [withheld]; under role-swap the same pick is held [withheld] and follows the sw

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r22_observation_occlusion_denoising
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
Beat your parent's fitness of +0.0000 (g0-14-pathstate-rssm, full budget, inner split).
The unmodified baseline scores +0.0037 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
