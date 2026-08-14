TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

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
  id                r3-02-occupancy-routed-copy
  its fitness       -0.0075   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r22_retrieval_composition_renderer
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r2_routed_collision_depth_batcher   params {"depth_beta": 0.8, "group_size": 4, "hard_frac_max": 0.5, "ramp_frac": 0.3, "size_pow": 0.5}
  stream              baseline_interleave
  head                r3_occupancy_routed_copy_transport

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

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r22_exact_target_equivalence_quotient
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- zca_shrunk_target_whitening (axis target)
import torch
import torch.nn as nn

NAME = "zca_shrunk_target_whitening"
DESCRIPTION = (
    "Trains the model in a ZCA-whitened observation space and reads it back through the exact "
    "matching inverse. A running second-moment matrix of the standardized next-observation "
    "targets is kept as a buffer, shrunk toward the identity, eigendecomposed on CPU in float64 "
    "every few steps, and turned into the symmetric pair (S^-p/2, S^+p/2) rescaled so the "
    "transformed targets keep unit average per-dimension variance. make_target left-multiplies "
    "the target by the forward map, so the loss measures error in a Mahalanobis metric that "
    "divides each direction of the target distribution by its own spread to the power p; to_obs "
    "left-multiplies the prediction by the inverse map, returning the fixed observation space "
    "used for retrieval. Both maps are one global matrix, identical for every step, every window "
    "and every candidate, and they are the identity until the running estimate has warmed up. "
    "The symmetric (ZCA) branch of the matrix power is used because it is the whitening closest "
    "to the identity in Frobenius norm, and the shrinkage bounds how far any direction can be "
    "amplified."
)

LEARNED = True

_MOMENTUM = 0.95
_SHRINK = 0.05
_POWER = 0.75
_REFRESH_EVERY = 50
_WARMUP = 150
_EIG_FLOOR = 1e-4


class ZCAShrunkTargetWhitening(nn.Module):
    def __init__(self, dim, momentum=_MOMENTUM, shrink=_SHRINK, power=_POWER,
                 refresh_every=_REFRESH_EVERY, warmup=_WARMUP, eig_floor=_EIG_FLOOR):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.shrink = float(shrink)
        self.power = float(power)
        self.refresh_every = max(1, int(refresh_every))
        self.warmup = max(1, int(warmup))
        self.eig_floor = float(eig_floor)
        self.register_buffer("second_moment", torch.eye(self.dim))
        self.register_buffer("fwd_map", torch.eye(self.dim))
        self.register_buffer("inv_map", torch.eye(self.dim))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))
        self._n_observed = 0

    def _linear_map(self, x, mat):
        if x is None or not torch.is_tensor(x) or x.dim() < 1 or int(x.shape[-1]) != self.dim:
            return x
        m = mat.to(device=x.device, dtype=x.dtype)
        return x @ m.t()

    @torch.no_grad()
    def _rebuild_maps(self):
        s = self.second_moment.detach().to(device="cpu", dtype=torch.float64)
        s = 0.5 * (s + s.t())
        if not bool(torch.isfinite(s).all()):
            return
        eye = torch.eye(self.dim, dtype=s.dtype)
        s = (1.0 - self.shrink) * s + self.shrink * eye
        try:
            evals, evecs = torch.linalg.eigh(s)
        except Exception:
            return
        if not (bool(torch.isfinite(evals).all()) and bool(torch.isfinite(evecs).all())):
            return
        lam = evals.clamp_min(self.eig_floor)
        half = 0.5 * self.power
        f = lam.pow(-half)
        v = lam.pow(half)
        trace_after = lam.pow(1.0 - self.power).sum()
        scale = (float(self.dim) / trace_after.clamp_min(1e-8)).sqrt()
        f = f * scale
        v = v / scale
        fwd = (evecs * f.unsqueeze(0)) @ evecs.t()
        inv = (evecs * v.unsqueeze(0)) @ evecs.t()
        if not (bool(torch.isfinite(fwd).all()) and bool(torch.isfinite(inv).all())):
            return
        self.fwd_map.copy_(fwd.to(dtype=self.fwd_map.dtype))
        self.inv_map.copy_(inv.to(dtype=self.inv_map.dtype))

    @torch.no_grad()
    def _observe(self, z):
        zf = z.detach().reshape(-1, self.dim)
        n = int(zf.shape[0])
        if n < 2:
            return
        zf = zf.to(device=self.second_moment.device, dtype=torch.float32)
        if not bool(torch.isfinite(zf).all()):
            return
        m = (zf.t() @ zf) / float(n)
        self.second_moment.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)
        self._n_observed += 1
        self.seen += 1
        if self._n_observed >= self.warmup and self._n_observed % self.refresh_every == 0:
            self._rebuild_maps()

    def make_target(self, z_obs, z_prev):
        if self.training and torch.is_grad_enabled() and torch.is_tensor(z_obs) \
                and z_obs.dim() >= 1 and int(z_obs.shape[-1]) == self.dim:
            self._observe(z_obs)
        return self._linear_map(z_obs.detach() if torch.is_tensor(z_obs) else z_obs, self.fwd_map)

    def to_obs(self, pred, z_prev):
        return self._linear_map(pred, self.inv_map)

    def reg(self):
        return 0.0


def make(D):
    return ZCAShrunkTargetWhitening(int(D))

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0075 (r3-02-occupancy-routed-copy, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
