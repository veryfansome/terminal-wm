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
  id                r3-03-fwl-orthogonalized-ring
  its fitness       +0.0075   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r3_fwl_orthogonalized_ring_contrast
  arch                r23_dual_address_transport_pointer
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
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

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           whitened_metric_ring_mse
  arch                r22_observation_occlusion_denoising
  optim               r18_spectral_capped_transition_readout
  target              ridge_zca_whitened_obs
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r3_rename_registry_occupancy_routing

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- within_sequence_contrast_equalizer (axis target)
import math

import torch
import torch.nn as nn

NAME = "within_sequence_contrast_equalizer"
DESCRIPTION = (
    "Exactly-invertible spectral reweighting of the prediction target, estimated online from the "
    "training batches. Target rows are cut into trajectories at the all-zero z_prev that marks a "
    "sequence start, exact-duplicate observations are down-weighted by their duplicate count via a "
    "quantized random projection, and two running second moments are accumulated: the total moment "
    "and the moment of the deviations from each trajectory's own mean. A rank-k orthonormal basis "
    "is carried by orthogonal iteration on the total moment; each direction receives a "
    "multiplicative gain equal to its within-trajectory variance fraction raised to a capped, "
    "log-mean-zero power, so directions that vary inside a trajectory are amplified relative to "
    "directions that only separate trajectories. make_target applies the gains inside that subspace "
    "and leaves the orthogonal complement untouched; to_obs applies the reciprocal gains, which is "
    "an exact inverse for any orthonormal basis and any positive gains. The map starts as the "
    "identity, ramps in, and freezes after a warmup; it has no trainable parameters and never reads "
    "commands, paths or step order."
)

LEARNED = True

_EPS = 1e-8


class WithinSequenceContrastEqualizer(nn.Module):
    def __init__(self, d=768, rank=32, alpha=1.0, gain_cap=3.0, warmup_updates=600,
                 ramp_updates=400, ema_decay=0.98, max_rows=1024, hash_dim=8,
                 hash_eps=1e-3, refresh_every=4, init_seed=20260812):
        super().__init__()
        self.d = int(d)
        self.rank = max(1, min(int(rank), self.d))
        self.alpha = float(alpha)
        self.log_gain_cap = math.log(max(1.0 + 1e-6, float(abs(gain_cap))))
        self.warmup_updates = int(warmup_updates)
        self.ramp_updates = max(1, int(ramp_updates))
        self.ema_decay = float(ema_decay)
        self.max_rows = max(8, int(max_rows))
        self.hash_eps = float(hash_eps)
        self.refresh_every = max(1, int(refresh_every))

        gen = torch.Generator().manual_seed(int(init_seed))
        seed_mat = torch.randn(self.d, self.rank, generator=gen)
        q, _ = torch.linalg.qr(seed_mat)
        self.register_buffer("basis", q.t().contiguous())
        self.register_buffer("gain", torch.ones(self.rank))
        self.register_buffer("total_moment", torch.zeros(self.d, self.d))
        self.register_buffer("within_moment", torch.zeros(self.d, self.d))
        self.register_buffer("hash_proj", torch.randn(self.d, max(1, int(hash_dim)),
                                                      generator=gen))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))

    def _ramp(self):
        x = float(int(self.n_updates)) / float(self.ramp_updates)
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _duplicate_weights(self, z):
        try:
            proj = z @ self.hash_proj.to(device=z.device, dtype=z.dtype)
            key = torch.round(proj / self.hash_eps)
            _, inverse, counts = torch.unique(key, dim=0, return_inverse=True,
                                              return_counts=True)
            w = counts.to(z.dtype)[inverse].reciprocal()
        except Exception:
            w = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        return w

    @staticmethod
    def _segment_ids(z_prev):
        starts = z_prev.abs().sum(dim=1) == 0
        if not bool(starts.any()):
            return None
        seg = torch.cumsum(starts.long(), dim=0) - 1
        return seg.clamp_min(0)

    def _orthonormalize(self, rows):
        out = rows.new_zeros(rows.shape)
        count = 0
        for i in range(rows.shape[0]):
            v = rows[i]
            for _ in range(2):
                if count > 0:
                    p = out[:count]
                    v = v - p.t() @ (p @ v)
            nrm = v.norm()
            if not bool(torch.isfinite(nrm)) or float(nrm) < 1e-6:
                return None
            out[count] = v / nrm
            count += 1
        return out

    @torch.no_grad()
    def _accumulate(self, z_obs, z_prev):
        n = z_obs.shape[0]
        if n < 8 or z_obs.shape[-1] != self.d:
            return
        z = z_obs.detach().float()
        prev = z_prev.detach().float() if z_prev is not None else None
        if prev is None or prev.shape != z.shape:
            return
        if not bool(torch.isfinite(z).all()):
            return
        if n > self.max_rows:
            z = z[: self.max_rows]
            prev = prev[: self.max_rows]
            n = self.max_rows

        seg = self._segment_ids(prev)
        w = self._duplicate_weights(z)
        w = w / w.sum().clamp_min(_EPS)
        zw = z * w.unsqueeze(1)
        total = z.t() @ zw

        if seg is None:
            centered = z - (w.unsqueeze(1) * z).sum(dim=0, keepdim=True)
        else:
            n_seg = int(seg.max()) + 1
            seg_w = z.new_zeros(n_seg).index_add_(0, seg, w)
            seg_sum = z.new_zeros(n_seg, self.d).index_add_(0, seg, zw)
            seg_mean = seg_sum / seg_w.clamp_min(_EPS).unsqueeze(1)
            centered = z - seg_mean[seg]
        within = centered.t() @ (centered * w.unsqueeze(1))

        if not (bool(torch.isfinite(total).all()) and bool(torch.isfinite(within).all())):
            return

        rho = self.ema_decay
        self.total_moment.mul_(rho).add_(total, alpha=1.0 - rho)
        self.within_moment.mul_(rho).add_(within, alpha=1.0 - rho)
        self.n_updates += 1

        if int(self.n_updates) % self.refresh_every != 0:
            return

        iterated = self.basis @ self.total_moment
        if not bool(torch.isfinite(iterated).all()):
            return
        new_basis = self._orthonormalize(iterated)
        if new_basis is not None:
            self.basis.copy_(new_basis)

        t_energy = ((self.basis @ self.total_moment) * self.basis).sum(dim=1).clamp_min(0.0)
        w_energy = ((self.basis @ self.within_moment) * self.basis).sum(dim=1).clamp_min(0.0)
        scale = t_energy.mean().clamp_min(_EPS)
        ratio = (w_energy + _EPS * scale) / (t_energy + _EPS * scale)
        log_ratio = ratio.clamp_min(1e-6).log()
        u = 0.5 * self.alpha * self._ramp() * (log_ratio - log_ratio.mean())
        u = u.clamp(-self.log_gain_cap, self.log_gain_cap)
        new_gain = u.exp()
        if bool(torch.isfinite(new_gain).all()):
            self.gain.copy_(new_gain)

    def _rescale(self, x, gain):
        shape = x.shape
        z = x.reshape(-1, self.d)
        basis = self.basis.to(device=z.device, dtype=z.dtype)
        g = gain.to(device=z.device, dtype=z.dtype)
        coeff = z @ basis.t()
        out = z + ((g - 1.0) * coeff) @ basis
        return out.reshape(shape)

    def make_target(self, z_obs, z_prev):
        if self.training and int(self.n_updates) < self.warmup_updates:
            self._accumulate(z_obs, z_prev)
        return self._rescale(z_obs, self.gain)

    def to_obs(self, pred, z_prev):
        return self._rescale(pred, self.gain.reciprocal())

    def reg(self):
        return 0.0


def make(D):
    return WithinSequenceContrastEqualizer(d=int(D))

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
Beat your parent's fitness of +0.0075 (r3-03-fwl-orthogonalized-ring, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
