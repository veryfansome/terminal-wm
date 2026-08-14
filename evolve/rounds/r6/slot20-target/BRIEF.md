TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

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
  id                r5-27-confusion-discriminant-metric
  its fitness       -0.0000   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r22_retrieval_composition_renderer
  optim               r18_spectral_capped_transition_readout
  target              confusion_discriminant_metric
  batcher             routed_depth_sysblock_batcher   params {"block_frac_max": 0.75, "depth_beta": 0.8, "group_size": 4, "hard_frac_max": 0.5, "n_block_images": 2, "ramp_frac": 0.3, "size_pow": 0.5}
  stream              baseline_interleave
  head                routed_copy_pathchange_transport

YOUR PARENT'S CURRENT target IMPL — confusion_discriminant_metric (this is the code you are mutating):
--------------------------------------------------------------------------------
import torch
import torch.nn as nn

NAME = "confusion_discriminant_metric"
DESCRIPTION = (
    "Trains the model in a globally re-metrized observation space and reads it back through the "
    "exact matching inverse. Two running second-moment buffers are kept over standardized "
    "next-observation targets: a TOTAL moment E[t t^T], and a CONFUSION moment built only from "
    "difference vectors between a target and its k nearest NON-DUPLICATE targets in the same "
    "flattened batch, each pair weighted by the reciprocal of the exact-duplicate class size of "
    "both endpoints. Every few hundred steps both are symmetrized, trace-normalized and "
    "eigendecomposed on CPU in float64: the total moment, shrunk toward the identity, gives the "
    "symmetric whitener B = S^-1/2 and its exact partner B^-1 = S^+1/2; the confusion moment is "
    "carried into the whitened frame as C = B G B, trace-normalized, shrunk toward the identity "
    "and eigendecomposed there, and its eigenvalues become a per-direction gain "
    "(mu/mean(mu))^(p/2), clamped to a bounded range. The forward map "
    "is gain-then-whiten, F = V diag(g) V^T B, rescaled so transformed targets keep unit average "
    "per-dimension variance; the inverse map is B^-1 V diag(1/g) V^T with the reciprocal rescale, "
    "so the pair is algebraically exact. make_target left-multiplies the target by F, so squared "
    "error and every distance in the loss are measured in a metric that stretches the directions "
    "along which confusable observations actually differ and compresses the directions they "
    "share; to_obs left-multiplies the prediction by F^-1, returning the fixed observation space "
    "used for retrieval. Both maps are one global matrix, identical for every step, every window "
    "and every candidate, are the identity until the running estimates have warmed up, and depend "
    "on no per-step side information: to_obs uses neither z_prev nor any observation, so it "
    "behaves identically at a position whose own observation is withheld."
)

LEARNED = True

_MOMENTUM = 0.99
_SHRINK_TOTAL = 0.05
_SHRINK_CONF = 0.10
_POWER = 1.0
_GAIN_MIN = 1.0 / 3.0
_GAIN_MAX = 3.0
_REFRESH_EVERY = 150
_WARMUP = 300
_EIG_FLOOR = 1e-4
_SUBSAMPLE = 192
_NEIGHBORS = 3
_DUP_TOL = 1e-4


class ConfusionDiscriminantMetric(nn.Module):
    def __init__(self, dim, momentum=_MOMENTUM, shrink_total=_SHRINK_TOTAL,
                 shrink_conf=_SHRINK_CONF, power=_POWER, gain_min=_GAIN_MIN, gain_max=_GAIN_MAX,
                 refresh_every=_REFRESH_EVERY, warmup=_WARMUP, eig_floor=_EIG_FLOOR,
                 subsample=_SUBSAMPLE, neighbors=_NEIGHBORS, dup_tol=_DUP_TOL):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.shrink_total = float(shrink_total)
        self.shrink_conf = float(shrink_conf)
        self.power = float(power)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.refresh_every = max(1, int(refresh_every))
        self.warmup = max(1, int(warmup))
        self.eig_floor = float(eig_floor)
        self.subsample = max(8, int(subsample))
        self.neighbors = max(1, int(neighbors))
        self.dup_tol = float(dup_tol)
        self.register_buffer("total_moment", torch.eye(self.dim))
        self.register_buffer("conf_moment", torch.eye(self.dim))
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
    def _accumulate_confusion(self, zf):
        n = int(zf.shape[0])
        if n < 8:
            return
        stride = max(1, n // self.subsample)
        sub = zf[::stride][: self.subsample]
        m = int(sub.shape[0])
        if m < 8:
            return
        sq = sub.pow(2).sum(dim=1, keepdim=True)
        d2 = (sq + sq.t() - 2.0 * (sub @ sub.t())).clamp_min(0.0) / float(self.dim)
        if not bool(torch.isfinite(d2).all()):
            return
        eye = torch.eye(m, device=d2.device, dtype=torch.bool)
        off = ~eye
        denom = off.sum().clamp_min(1).to(d2.dtype)
        ref = (d2 * off.to(d2.dtype)).sum() / denom
        if not bool(torch.isfinite(ref)) or float(ref) <= 0.0:
            return
        dup = d2 <= (self.dup_tol * ref)
        cnt = dup.sum(dim=1).clamp_min(1).to(d2.dtype)
        inv_cnt = cnt.reciprocal()
        masked = d2.masked_fill(dup | eye, float("inf"))
        k = min(self.neighbors, m - 1)
        if k < 1:
            return
        vals, idx = torch.topk(masked, k, dim=1, largest=False)
        ok = torch.isfinite(vals)
        if not bool(ok.any()):
            return
        rows = torch.arange(m, device=d2.device).unsqueeze(1).expand(m, k).reshape(-1)
        cols = idx.reshape(-1)
        diff = sub[rows] - sub[cols]
        w = inv_cnt[rows] * inv_cnt[cols] * ok.reshape(-1).to(d2.dtype)
        total_w = w.sum().clamp_min(1e-6)
        mat = diff.t() @ (diff * (w / total_w).unsqueeze(1))
        if not bool(torch.isfinite(mat).all()):
            return
        self.conf_moment.mul_(self.momentum).add_(
            mat.to(self.conf_moment.dtype), alpha=1.0 - self.momentum)

    @torch.no_grad()
    def _rebuild_maps(self):
        s = self.total_moment.detach().to(device="cpu", dtype=torch.float64)
        g = self.conf_moment.detach().to(device="cpu", dtype=torch.float64)
        s = 0.5 * (s + s.t())
        g = 0.5 * (g + g.t())
        if not (bool(torch.isfinite(s).all()) and bool(torch.isfinite(g).all())):
            return
        eye = torch.eye(self.dim, dtype=s.dtype)
        tau = (s.diagonal().sum() / float(self.dim)).clamp_min(1e-8)
        s_norm = (1.0 - self.shrink_total) * (s / tau) + self.shrink_total * eye
        gt = (g.diagonal().sum() / float(self.dim)).clamp_min(1e-8)
        g_norm = g / gt
        try:
            lam, u_vec = torch.linalg.eigh(s_norm)
        except Exception:
            return
        if not (bool(torch.isfinite(lam).all()) and bool(torch.isfinite(u_vec).all())):
            return
        lam = lam.clamp_min(self.eig_floor)
        b_fwd = (u_vec * lam.pow(-0.5).unsqueeze(0)) @ u_vec.t()
        b_inv = (u_vec * lam.pow(0.5).unsqueeze(0)) @ u_vec.t()
        c_mat = b_fwd @ g_norm @ b_fwd
        c_mat = 0.5 * (c_mat + c_mat.t())
        ct = (c_mat.diagonal().sum() / float(self.dim)).clamp_min(1e-8)
        c_mat = (1.0 - self.shrink_conf) * (c_mat / ct) + self.shrink_conf * eye
        try:
            mu, v_vec = torch.linalg.eigh(c_mat)
        except Exception:
            return
        if not (bool(torch.isfinite(mu).all()) and bool(torch.isfinite(v_vec).all())):
            return
        mu = mu.clamp_min(self.eig_floor)
        mu_bar = mu.mean().clamp_min(1e-8)
        gain = (mu / mu_bar).pow(0.5 * self.power).clamp(self.gain_min, self.gain_max)
        scale = (float(self.dim) / (tau * gain.pow(2).sum()).clamp_min(1e-8)).sqrt()
        w_fwd = (v_vec * gain.unsqueeze(0)) @ v_vec.t()
        w_inv = (v_vec * gain.reciprocal().unsqueeze(0)) @ v_vec.t()
        fwd = (w_fwd @ b_fwd) * scale
        inv = (b_inv @ w_inv) / scale
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
        zf = zf.to(device=self.total_moment.device, dtype=torch.float32)
        if not bool(torch.isfinite(zf).all()):
            return
        m = (zf.t() @ zf) / float(n)
        self.total_moment.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)
        self._accumulate_confusion(zf)
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
    return ConfusionDiscriminantMetric(int(D))
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- identity (axis target)
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
Beat your parent's fitness of -0.0000 (r5-27-confusion-discriminant-metric, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
