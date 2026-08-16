import math

import torch
import torch.nn as nn

NAME = "r7_local_contrast_expanding_target"
DESCRIPTION = (
    "A learned expansion-only linear flow on the regression target, steered by LOCAL "
    "discriminability instead of global variance. An exactly orthonormal k-frame built from a "
    "product of Householder reflections selects k directions of the observation space and the "
    "target is rescaled by exp(a_i) >= 1 along direction i and left untouched everywhere else, so "
    "the map is Q diag(e^a, 1) Q^T with a >= 0 and its exact inverse Q diag(e^-a, 1) Q^T is a "
    "contraction — it can never amplify prediction error, in any direction. The frame is chosen "
    "by a local-discriminant criterion in the spirit of local Fisher discriminant analysis: on "
    "each batch, every target's m nearest neighbours in the fixed observation space are found "
    "(detached), pairs whose separation is below a small fraction of the mean pairwise separation "
    "are dropped as duplicates, and the frame maximises mean_i [ log Var_local_i - eta * log "
    "Var_global_i ] where Var_local is the variance of surviving nearest-neighbour differences "
    "along direction i. Near-neighbour differences in this corpus are two observations of the "
    "same command shape whose file CONTENT differs, which is exactly the subspace the retrieval "
    "decision reads: the pick between two candidate observations depends only on the "
    "prediction's projection onto their difference. The scales are then pulled by an equalisation "
    "term until the post-transform variance along each selected direction sits a fixed multiple "
    "above the mean variance of the untouched complement, so the training loss must buy accuracy "
    "on the content axes, while the inverse shrinks whatever error is left there before the "
    "candidates are ranked. Scales are bounded by a sigmoid, so the flow is strictly expanding "
    "and bounded."
)

LEARNED = True

_K = 24
_A_MAX = 2.0
_A_BIAS = -2.0
_SCALE_GAIN = 4.0
_RHO = 0.75
_OVER = 2.0
_ETA = 0.5
_LAM_EQ = 0.2
_LAM_DISC = 0.2
_NEIGHBORS = 4
_MAX_ROWS = 384
_DUP_FRAC = 0.002
_MIN_PAIRS = 16
_MIN_ROWS = 32
_FRAME_NORM = 20.0
_EPS = 1e-6
_SEED = 20260814


class LocalContrastExpandingTarget(nn.Module):
    def __init__(self, d, k=_K, a_max=_A_MAX, a_bias=_A_BIAS, scale_gain=_SCALE_GAIN,
                 rho=_RHO, over=_OVER, eta=_ETA, lam_eq=_LAM_EQ, lam_disc=_LAM_DISC,
                 neighbors=_NEIGHBORS, max_rows=_MAX_ROWS, dup_frac=_DUP_FRAC):
        super().__init__()
        self.dim = int(d)
        self.k = max(1, min(int(k), self.dim - 1))
        self.a_max = float(a_max)
        self.a_bias = float(a_bias)
        self.scale_gain = float(scale_gain)
        self.rho = float(rho)
        self.log_over = math.log(max(1e-6, float(over)))
        self.eta = float(eta)
        self.lam_eq = float(lam_eq)
        self.lam_disc = float(lam_disc)
        self.neighbors = max(1, int(neighbors))
        self.max_rows = max(_MIN_ROWS, int(max_rows))
        self.dup_frac = float(dup_frac)
        g = torch.Generator().manual_seed(_SEED)
        v = torch.randn(self.k, self.dim, generator=g)
        v = v / v.norm(dim=1, keepdim=True).clamp_min(_EPS) * _FRAME_NORM
        self.frame = nn.Parameter(v)
        self.spec_raw = nn.Parameter(torch.zeros(self.k))
        self.batch_stats = None

    def _unit_frame(self, x):
        v = self.frame
        if v.device != x.device or v.dtype != x.dtype:
            v = v.to(device=x.device, dtype=x.dtype)
        return v / v.norm(dim=1, keepdim=True).clamp_min(_EPS)

    def _log_scale(self, x):
        a = self.a_max * torch.sigmoid(self.scale_gain * self.spec_raw + self.a_bias)
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
            self.batch_stats = (head, tail, a, z_obs.detach())
        else:
            self.batch_stats = None
        return out

    def to_obs(self, pred, z_prev):
        if pred.shape[-1] != self.dim:
            return pred
        a = self._log_scale(pred)
        out, _, _ = self._rescale(pred, -a)
        return out

    def _local_scatter(self, hs, z_rows):
        r = int(hs.shape[0])
        m = min(self.neighbors, r - 1)
        if m < 1:
            return None
        with torch.no_grad():
            zr = z_rows.float()
            sq = zr.pow(2).sum(dim=1, keepdim=True)
            d2 = (sq + sq.t() - 2.0 * (zr @ zr.t())).clamp_min(0.0)
            eye = torch.eye(r, dtype=torch.bool, device=d2.device)
            mean_off = d2.masked_fill(eye, 0.0).sum() / float(max(1, r * (r - 1)))
            near_d, near_i = torch.topk(d2.masked_fill(eye, float("inf")), m, dim=1,
                                        largest=False)
            keep = (near_d > self.dup_frac * mean_off).to(zr.dtype)
            n_pairs = keep.sum()
        if float(n_pairs) < _MIN_PAIRS:
            return None
        diff = hs.unsqueeze(1) - hs[near_i]
        wsum = (diff.pow(2) * keep.unsqueeze(-1).to(diff.dtype)).sum(dim=(0, 1))
        return wsum / n_pairs.clamp_min(1.0).to(diff.dtype)

    def reg(self):
        anchor = self.spec_raw.sum() * 0.0 + self.frame.sum() * 0.0
        st = self.batch_stats
        self.batch_stats = None
        if st is None:
            return anchor
        head, tail, a, z_obs = st
        if head.dim() != 2 or int(head.shape[0]) < _MIN_ROWS or int(tail.shape[-1]) < 1:
            return anchor
        r = min(int(head.shape[0]), self.max_rows)
        hs = head[:r].float()
        var_head = hs.var(dim=0, unbiased=False)
        var_tail = tail[:r].float().var(dim=0, unbiased=False).mean()
        log_head = torch.log(var_head + _EPS)
        log_tail = torch.log(var_tail + _EPS)
        goal = ((1.0 - self.rho) * log_head
                + self.rho * (log_tail + self.log_over)).detach()
        equalise = ((log_head + 2.0 * a.float() - goal) ** 2).mean()
        out = self.lam_eq * equalise
        local = self._local_scatter(hs, z_obs[:r])
        if local is not None:
            log_local = torch.log(local + _EPS)
            out = out + self.lam_disc * (self.eta * log_head - log_local).mean()
        if not bool(torch.isfinite(out)):
            return anchor
        return out.to(anchor.dtype) + anchor


def make(d):
    return LocalContrastExpandingTarget(int(d))
