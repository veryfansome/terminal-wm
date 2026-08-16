TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

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
  id                r5-25-roletied-address-polyak
  its fitness       +0.0150   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r3_fwl_orthogonalized_ring_contrast
  arch                r23_dual_address_transport_pointer
  optim               r5_roletied_address_polyak_readout
  target              identity
  batcher             r2_routedepth_tilt_readcollision_blocks   params {"alpha_max": 2.5, "ans_max_sim": 0.995, "chunk": 256, "coll_share": 0.5, "hard_frac_max": 0.75, "max_centers": 8000, "n_blocks": 2, "nb_k": 16, "proj_dim": 128, "ramp_frac": 0.3, "sim_thresh": 0.85}
  stream              r4_pathslot_dualaddress_code
  head                r18_transition_forwardmodel_consistency

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

PARENT'S EVAL FEEDBACK: comp_ca +0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca -0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- ridge_zca_whitened_obs (axis target)
import torch
import torch.nn as nn

NAME = "ridge_zca_whitened_obs"
LEARNED = True
DESCRIPTION = (
    "A ridge-regularised ZCA whitening of the observation target space, estimated online from the "
    "training targets and then frozen for the rest of the run. During a warmup phase the target is "
    "the raw next-observation embedding while a running mean and second moment of those targets "
    "are accumulated; once enough rows are seen the covariance is eigendecomposed on CPU in double "
    "precision and a symmetric matrix W = Q diag(g * (lam + eps)^-1/2) Q^T is built, where eps is a "
    "fixed fraction of the mean eigenvalue and g is chosen so the mean per-direction variance of "
    "the transformed target equals that of the untransformed target. From then on the model is "
    "trained to predict (z_obs - mu) @ W, and the inverse pred @ W^-1 + mu returns the prediction "
    "to the fixed observation space the retrieval eval uses. W and mu depend only on frozen "
    "buffers, are updated only inside make_target, hold no trainable parameters, and reg() is "
    "zero; before the freeze both directions are the identity."
)

_WARMUP_BATCHES = 200
_MIN_ROWS = 20000
_RIDGE_FRAC = 0.5


def _work_dtype(t):
    if t.device.type == "mps":
        return torch.float32
    return torch.float64


class RidgeZCAWhitenedTarget(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.width = int(width)
        self.register_buffer("center", torch.zeros(self.width))
        self.register_buffer("mom2", torch.zeros(self.width, self.width))
        self.register_buffer("fwd", torch.eye(self.width))
        self.register_buffer("inv", torch.eye(self.width))
        self.register_buffer("rows", torch.zeros(()))
        self.register_buffer("calls", torch.zeros(()))
        self.register_buffer("ready", torch.zeros(()))
        self.register_buffer("dead", torch.zeros(()))

    def _accumulate(self, z):
        x = z.detach().reshape(-1, self.width).float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = float(x.shape[0])
        if m < 1.0:
            return
        frac = m / (float(self.rows) + m)
        self.center.add_((x.mean(0) - self.center) * frac)
        self.mom2.add_(((x.t() @ x) / m - self.mom2) * frac)
        self.rows.add_(m)

    def _fit(self):
        try:
            mu = self.center.detach().cpu().double()
            m2 = self.mom2.detach().cpu().double()
            cov = m2 - torch.outer(mu, mu)
            cov = 0.5 * (cov + cov.t())
            evals, evecs = torch.linalg.eigh(cov)
            evals = evals.clamp_min(0.0)
            mean_ev = evals.mean().clamp_min(1e-12)
            eps = _RIDGE_FRAC * mean_ev
            w = (evals + eps).rsqrt()
            gain = (mean_ev / (w * w * evals).mean().clamp_min(1e-12)).sqrt()
            w = w * gain
            fwd64 = (evecs * w.unsqueeze(0)) @ evecs.t()
            fwd32 = fwd64.float()
            inv32 = torch.linalg.inv(fwd32.double()).float()
            if not bool(torch.isfinite(fwd32).all()) or not bool(torch.isfinite(inv32).all()):
                raise RuntimeError("non-finite whitening frame")
            self.fwd.copy_(fwd32.to(self.fwd.dtype))
            self.inv.copy_(inv32.to(self.inv.dtype))
            self.ready.fill_(1.0)
        except Exception:
            self.dead.fill_(1.0)

    def make_target(self, z_obs, z_prev):
        with torch.no_grad():
            if float(self.dead) > 0.5:
                return z_obs
            if float(self.ready) < 0.5:
                self._accumulate(z_obs)
                self.calls.add_(1.0)
                if float(self.calls) >= _WARMUP_BATCHES and float(self.rows) >= _MIN_ROWS:
                    self._fit()
                return z_obs
            wd = _work_dtype(z_obs)
            x = z_obs.detach().to(wd) - self.center.to(wd).unsqueeze(0)
            return (x @ self.fwd.to(wd)).to(z_obs.dtype)

    def to_obs(self, pred, z_prev):
        if float(self.dead) > 0.5 or float(self.ready) < 0.5:
            return pred
        wd = _work_dtype(pred)
        x = pred.to(wd) @ self.inv.to(wd) + self.center.to(wd).unsqueeze(0)
        return x.to(pred.dtype)

    def reg(self):
        return 0.0


def make(D):
    return RidgeZCAWhitenedTarget(D)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0150 (r5-25-roletied-address-polyak, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
