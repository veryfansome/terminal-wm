import math

import torch
import torch.nn as nn

NAME = "neighbor_contrast_shaped_target"
DESCRIPTION = (
    "Trains the model in a linearly reshaped observation space and reads it back through the exact "
    "matching inverse. Two second-moment matrices are estimated online from the training targets, "
    "under no_grad and with no trainable parameters: the TOTAL covariance of the next-observation "
    "embeddings, and a CONTRAST covariance built from differences between mutually near "
    "observations inside the SAME trajectory. Trajectory boundaries are read off the all-zero "
    "previous-observation row that marks a sequence start; inside each trajectory every row is "
    "paired with its nearest other row whose squared distance exceeds a small fraction of the mean "
    "within-trajectory pair distance (so exact repeats contribute nothing), and only the closer "
    "half of those pairs is kept, so the contrast covariance describes the directions along which "
    "co-occurring, mutually confusable observations actually differ rather than the directions "
    "that carry the most total energy. The forward map is the product of two symmetric matrix "
    "powers, Cov_contrast^(q/2) times Cov_total^(-a/2); each factor is built by an eigendecom"
    "position on CPU in float64 after shrinkage toward the identity, with per-direction gains that "
    "are log-centred and capped so each factor is volume preserving and well conditioned, and a "
    "single global scale keeps the transformed target at unit mean per-dimension variance. The "
    "inverse, Cov_total^(a/2) times Cov_contrast^(-q/2), is assembled from the same eigenpairs and "
    "is therefore an exact algebraic inverse. The map is the identity during a warmup, has both "
    "exponents ramped in over a fixed span while it is periodically rebuilt, and is frozen for the "
    "remainder of training. Setting the contrast exponent to zero reduces the map to a ridge-"
    "shrunk ZCA whitening of the target space, and setting both exponents to zero reduces it to "
    "the identity. to_obs reads only frozen buffers, so it behaves identically whether or not an "
    "observation is present at the position being read."
)

LEARNED = True

_MIN_ROWS = 8
_MIN_PAIRS = 4


def _all_finite(*tensors):
    for t in tensors:
        if not bool(torch.isfinite(t).all()):
            return False
    return True


class NeighborContrastShapedTarget(nn.Module):
    def __init__(self, dim, contrast_power=1.0, whiten_power=0.5, shrink=0.05,
                 gain_cap=3.0, warmup=300, ramp_span=900, rebuild_every=150,
                 freeze_after=1500, accum_stride=2, max_rows=640, dup_frac=1e-3,
                 keep_frac=0.5, eig_floor=1e-3):
        super().__init__()
        self.dim = int(dim)
        self.contrast_power = float(contrast_power)
        self.whiten_power = float(whiten_power)
        self.shrink = min(0.5, max(1e-4, float(shrink)))
        self.log_cap = math.log(max(1.0 + 1e-6, float(gain_cap)))
        self.warmup = max(1, int(warmup))
        self.ramp_span = max(1, int(ramp_span))
        self.rebuild_every = max(1, int(rebuild_every))
        self.freeze_after = max(self.warmup, int(freeze_after))
        self.accum_stride = max(1, int(accum_stride))
        self.max_rows = max(32, int(max_rows))
        self.dup_frac = float(dup_frac)
        self.keep_frac = min(1.0, max(0.05, float(keep_frac)))
        self.eig_floor = float(eig_floor)

        self.register_buffer("row_sum", torch.zeros(self.dim))
        self.register_buffer("row_moment", torch.zeros(self.dim, self.dim))
        self.register_buffer("row_count", torch.zeros(()))
        self.register_buffer("pair_moment", torch.zeros(self.dim, self.dim))
        self.register_buffer("pair_count", torch.zeros(()))
        self.register_buffer("center", torch.zeros(self.dim))
        self.register_buffer("fwd", torch.eye(self.dim))
        self.register_buffer("inv", torch.eye(self.dim))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))
        self.register_buffer("live", torch.zeros(()))

    def _segments(self, z_prev, n, device):
        if not torch.is_tensor(z_prev) or z_prev.dim() != 2 or z_prev.shape[0] != n:
            return torch.zeros(n, dtype=torch.long, device=device)
        starts = z_prev.detach().abs().sum(dim=1) == 0
        if not bool(starts.any()):
            return torch.zeros(n, dtype=torch.long, device=device)
        return (torch.cumsum(starts.long(), dim=0) - 1).clamp_min(0)

    @torch.no_grad()
    def _accumulate(self, z_obs, z_prev):
        if not torch.is_tensor(z_obs) or z_obs.dim() != 2:
            return
        if z_obs.shape[1] != self.dim or z_obs.shape[0] < _MIN_ROWS:
            return
        z = z_obs.detach()
        prev = z_prev if torch.is_tensor(z_prev) else None
        if z.shape[0] > self.max_rows:
            z = z[: self.max_rows]
            if prev is not None:
                prev = prev[: self.max_rows]
        z = z.float()
        if not _all_finite(z):
            return
        n = int(z.shape[0])

        self.row_sum.add_(z.sum(dim=0))
        self.row_moment.add_(z.t() @ z)
        self.row_count.add_(float(n))

        seg = self._segments(prev, n, z.device)
        sq = (z * z).sum(dim=1)
        d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (z @ z.t())).clamp_min(0.0)
        same = seg.unsqueeze(1) == seg.unsqueeze(0)
        eye = torch.eye(n, dtype=torch.bool, device=z.device)
        cand = same & (~eye)
        if int(cand.sum()) < _MIN_PAIRS:
            return
        tau = self.dup_frac * d2[cand].mean()
        usable = cand & (d2 > tau)
        if int(usable.sum()) < _MIN_PAIRS:
            return
        inf = torch.full_like(d2, float("inf"))
        dmin, nn_idx = torch.where(usable, d2, inf).min(dim=1)
        have = torch.isfinite(dmin)
        m = int(have.sum())
        if m < _MIN_PAIRS:
            return
        ordered = torch.sort(dmin[have]).values
        k = max(1, int(round(m * self.keep_frac)))
        thr = ordered[min(k, m) - 1]
        keep = have & (dmin <= thr)
        rows = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if int(rows.numel()) < 2:
            return
        diff = z[rows] - z[nn_idx[rows]]
        if not _all_finite(diff):
            return
        self.pair_moment.add_(diff.t() @ diff)
        self.pair_count.add_(float(rows.numel()))

    @torch.no_grad()
    def _symmetric_power(self, mat, exponent, dtype):
        d = self.dim
        eye = torch.eye(d, dtype=dtype)
        shrunk = (1.0 - self.shrink) * mat + self.shrink * eye
        try:
            evals, evecs = torch.linalg.eigh(shrunk)
        except Exception:
            return None
        if not _all_finite(evals, evecs):
            return None
        lam = evals.clamp_min(self.eig_floor)
        log_gain = 0.5 * float(exponent) * lam.log()
        log_gain = log_gain - log_gain.mean()
        log_gain = log_gain.clamp(-self.log_cap, self.log_cap)
        gain = log_gain.exp()
        fwd = (evecs * gain.unsqueeze(0)) @ evecs.t()
        inv = (evecs * gain.reciprocal().unsqueeze(0)) @ evecs.t()
        if not _all_finite(fwd, inv):
            return None
        return fwd, inv

    @torch.no_grad()
    def _rebuild(self, ramp):
        d = self.dim
        if float(self.row_count) < 512.0 or float(self.pair_count) < 128.0:
            return
        dtype = torch.float64
        mu = (self.row_sum / self.row_count).detach().to("cpu", dtype)
        total = (self.row_moment / self.row_count).detach().to("cpu", dtype)
        total = total - torch.outer(mu, mu)
        total = 0.5 * (total + total.t())
        contrast = (self.pair_moment / self.pair_count).detach().to("cpu", dtype)
        contrast = 0.5 * (contrast + contrast.t())
        if not _all_finite(mu, total, contrast):
            return
        t_tr = torch.diagonal(total).sum()
        c_tr = torch.diagonal(contrast).sum()
        if float(t_tr) <= 0.0 or float(c_tr) <= 0.0:
            return
        raw_total = total
        total = total * (float(d) / t_tr)
        contrast = contrast * (float(d) / c_tr)

        w = self._symmetric_power(total, -self.whiten_power * ramp, dtype)
        if w is None:
            return
        c = self._symmetric_power(contrast, self.contrast_power * ramp, dtype)
        if c is None:
            return
        fwd = c[0] @ w[0]
        inv = w[1] @ c[1]
        if not _all_finite(fwd, inv):
            return
        energy = ((fwd @ raw_total) * fwd).sum()
        if not bool(torch.isfinite(energy)) or float(energy) <= 0.0:
            return
        scale = (float(d) / energy).sqrt()
        fwd = fwd * scale
        inv = inv / scale
        if not _all_finite(fwd, inv):
            return
        self.center.copy_(mu.to(self.center.dtype))
        self.fwd.copy_(fwd.to(self.fwd.dtype))
        self.inv.copy_(inv.to(self.inv.dtype))
        self.live.fill_(1.0)

    def _shift_and_map(self, x, mat, offset, add_offset):
        if not torch.is_tensor(x) or x.dim() < 1 or int(x.shape[-1]) != self.dim:
            return x
        if float(self.live) < 0.5:
            return x
        m = mat.to(device=x.device, dtype=x.dtype)
        o = offset.to(device=x.device, dtype=x.dtype)
        if add_offset:
            return x @ m.t() + o
        return (x - o) @ m.t()

    def make_target(self, z_obs, z_prev):
        if self.training:
            self.calls += 1
            step = int(self.calls)
            if step <= self.freeze_after:
                if step % self.accum_stride == 0:
                    self._accumulate(z_obs, z_prev)
                if step >= self.warmup and (step - self.warmup) % self.rebuild_every == 0:
                    ramp = float(step - self.warmup) / float(self.ramp_span)
                    self._rebuild(min(1.0, max(0.0, ramp)))
        z = z_obs.detach() if torch.is_tensor(z_obs) else z_obs
        return self._shift_and_map(z, self.fwd, self.center, False)

    def to_obs(self, pred, z_prev):
        return self._shift_and_map(pred, self.inv, self.center, True)

    def reg(self):
        return 0.0


def make(D):
    return NeighborContrastShapedTarget(int(D))
