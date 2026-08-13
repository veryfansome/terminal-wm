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
