import torch
import torch.nn as nn

NAME = "r6_spectral_equalized_target"
DESCRIPTION = (
    "A learned volume-controlled linear flow on the regression target: an exactly orthonormal "
    "k-frame built from a product of Householder reflections selects k directions of the "
    "observation space, and the target is rescaled by exp(a_i) along direction i and left "
    "untouched everywhere else, so the map is Q diag(e^a, 1) Q^T and its exact inverse is "
    "Q diag(e^-a, 1) Q^T. The regulariser is a whitening pair computed on the same batch of "
    "targets: a log-volume term that steers the frame onto the leading eigendirections of the "
    "target covariance, and an equalisation term that drives the post-transform variance along "
    "each selected direction a fixed fraction of the way from its own variance down to the mean "
    "variance of the untouched complement. The head of the target spectrum — the "
    "empty-versus-content, output-format and length axes that carry most of the corpus variance "
    "and none of the distinction between two file contents — is therefore compressed toward the "
    "tail, so the training loss stops spending its error budget there. Scales are bounded by a "
    "tanh so the inverse can never blow up and the target can never collapse."
)

LEARNED = True

_K = 24
_A_MAX = 1.5
_RHO = 0.5
_SCALE_GAIN = 8.0
_LAM_EQ = 0.05
_LAM_PCA = 0.02
_FRAME_NORM = 20.0
_EPS = 1e-6
_MIN_ROWS = 16
_SEED = 20260814


class SpectralEqualizedTarget(nn.Module):
    def __init__(self, d, k=_K, a_max=_A_MAX, rho=_RHO, scale_gain=_SCALE_GAIN,
                 lam_eq=_LAM_EQ, lam_pca=_LAM_PCA):
        super().__init__()
        self.dim = int(d)
        self.k = max(1, min(int(k), self.dim - 1))
        self.a_max = float(a_max)
        self.rho = float(rho)
        self.scale_gain = float(scale_gain)
        self.lam_eq = float(lam_eq)
        self.lam_pca = float(lam_pca)
        g = torch.Generator().manual_seed(_SEED)
        v = torch.randn(self.k, self.dim, generator=g)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(_EPS) * _FRAME_NORM
        self.frame = nn.Parameter(v)
        self.spec_raw = nn.Parameter(torch.zeros(self.k))
        self.spec_stats = None

    def _unit_frame(self, x):
        v = self.frame
        if v.device != x.device or v.dtype != x.dtype:
            v = v.to(device=x.device, dtype=x.dtype)
        return v / v.norm(dim=1, keepdim=True).clamp_min(_EPS)

    def _log_scale(self, x):
        a = self.a_max * torch.tanh(self.scale_gain * self.spec_raw / self.a_max)
        if a.device != x.device or a.dtype != x.dtype:
            a = a.to(device=x.device, dtype=x.dtype)
        return a

    def _reflect(self, x, v, transposed):
        n = int(v.shape[0])
        order = range(n) if transposed else range(n - 1, -1, -1)
        for i in order:
            vi = v[i]
            x = x - 2.0 * torch.matmul(x, vi).unsqueeze(-1) * vi
        return x

    def _rescale(self, x, a):
        v = self._unit_frame(x)
        t = self._reflect(x, v, True)
        head = t[..., :self.k]
        tail = t[..., self.k:]
        y = torch.cat([head * torch.exp(a), tail], dim=-1)
        return self._reflect(y, v, False), head, tail

    def make_target(self, z_obs, z_prev):
        if z_obs.shape[-1] != self.dim:
            return z_obs
        a = self._log_scale(z_obs)
        out, head, tail = self._rescale(z_obs, a)
        if self.training and torch.is_grad_enabled():
            self.spec_stats = (head, tail, a)
        else:
            self.spec_stats = None
        return out

    def to_obs(self, pred, z_prev):
        if pred.shape[-1] != self.dim:
            return pred
        a = self._log_scale(pred)
        out, _, _ = self._rescale(pred, -a)
        return out

    def reg(self):
        anchor = self.spec_raw.sum() * 0.0 + self.frame.sum() * 0.0
        st = self.spec_stats
        self.spec_stats = None
        if st is None:
            return anchor
        head, tail, a = st
        if head.dim() != 2 or int(head.shape[0]) < _MIN_ROWS or int(tail.shape[-1]) < 1:
            return anchor
        var_head = head.float().var(dim=0, unbiased=False)
        var_tail = tail.float().var(dim=0, unbiased=False).mean()
        log_head = torch.log(var_head + _EPS)
        log_tail = torch.log(var_tail + _EPS)
        goal = ((1.0 - self.rho) * log_head + self.rho * log_tail).detach()
        equalise = ((log_head + 2.0 * a.float() - goal) ** 2).mean()
        log_volume = -log_head.mean()
        out = self.lam_eq * equalise + self.lam_pca * log_volume
        if not bool(torch.isfinite(out)):
            return anchor
        return out.to(anchor.dtype) + anchor


def make(d):
    return SpectralEqualizedTarget(int(d))
