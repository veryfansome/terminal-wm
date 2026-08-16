import torch
import torch.nn as nn

NAME = "r7_unit_volume_householder_metric"
LEARNED = True
DESCRIPTION = (
    "A learned linear flow on the target space whose Jacobian determinant is pinned to one, so it "
    "can reshape the training geometry but can never shrink or inflate it. The transform is "
    "T(z) = (z R) * g, where R is the orthogonal matrix formed by an even number of Householder "
    "reflections built from trainable vectors, and g is a per-coordinate positive gain whose logs "
    "are bounded by a tanh and then centred to sum to exactly zero. R is orthogonal for every "
    "parameter value and g has unit product, so |det T| = 1 identically; the exact inverse "
    "(pred / g) R_inv replays the same reflections in reverse order. The reflection vectors are "
    "initialised in adjacent identical pairs and the log gains at zero, so the transform is exactly "
    "the identity at step zero and each pair departs from it as a rotation in one learned plane. "
    "Squared error under a unit-determinant reweighting is minimised by gains inversely proportional "
    "to the per-direction residual, so the training signal becomes a relative (geometric-mean) error "
    "criterion in a learned basis instead of an absolute one, and the eval inverse divides the "
    "model's residual by the same gains the training amplified."
)

_N_REFLECTIONS = 64
_LOG_CAP = 1.25
_EPS = 1e-8


class UnitVolumeHouseholderTarget(nn.Module):

    def __init__(self, width, n_reflections=_N_REFLECTIONS, log_cap=_LOG_CAP):
        super().__init__()
        self.width = int(width)
        k = max(2, int(n_reflections))
        if k % 2:
            k += 1
        self.n_reflections = k
        self.log_cap = float(log_cap)
        seeds = torch.randn(k // 2, self.width)
        self.reflectors = nn.Parameter(seeds.repeat_interleave(2, dim=0).contiguous())
        self.log_gain = nn.Parameter(torch.zeros(self.width))

    def _frame(self, device, dtype):
        v = self.reflectors.to(device=device, dtype=dtype)
        return v * torch.rsqrt(v.pow(2).sum(dim=1, keepdim=True).clamp_min(_EPS))

    def _gain(self, device, dtype):
        a = self.log_cap * torch.tanh(self.log_gain.to(device=device, dtype=dtype))
        return torch.exp(a - a.mean())

    def _rotation(self, device, dtype, backward):
        v = self._frame(device, dtype)
        q = torch.eye(self.width, device=device, dtype=dtype)
        order = range(self.n_reflections - 1, -1, -1) if backward else range(self.n_reflections)
        for i in order:
            vi = v[i]
            q = q - 2.0 * torch.matmul(q, vi).unsqueeze(-1) * vi
        return q

    def make_target(self, z_obs, z_prev):
        if z_obs.shape[-1] != self.width or z_obs.numel() == 0:
            return z_obs
        rot = self._rotation(z_obs.device, z_obs.dtype, False)
        return torch.matmul(z_obs, rot) * self._gain(z_obs.device, z_obs.dtype)

    def to_obs(self, pred, z_prev):
        if pred.shape[-1] != self.width or pred.numel() == 0:
            return pred
        rot = self._rotation(pred.device, pred.dtype, True)
        return torch.matmul(pred / self._gain(pred.device, pred.dtype), rot)

    def reg(self):
        return 0.0


def make(D):
    return UnitVolumeHouseholderTarget(D)
