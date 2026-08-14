import torch
import torch.nn as nn

NAME = "r2_shrunk_zca_content_metric"
DESCRIPTION = (
    "Trains the model against a shrunk ZCA transform of the observation target. The mean and "
    "covariance of the training targets are accumulated over an opening window of training "
    "steps (statistics of the target distribution only, never of the model's outputs), a single "
    "symmetric operator W = U diag((lam_bar / (lam + rho*lam_bar))**(gamma/2)) U^T is formed "
    "once and rescaled so the total target variance is unchanged, and every later step trains on "
    "(z_obs - mu) W. The eigenvalue floor rho bounds the largest gain, so no direction is "
    "amplified without limit. to_obs applies the exact inverse W^{-1} = U diag(1/g) U^T and adds "
    "mu back, so the retrieval eval runs in the unmodified observation space. The transform is "
    "the identity until the statistics window closes and is frozen from then on."
)

LEARNED = True

_GAMMA = 1.0
_RHO = 0.08
_ACC_CALLS = 192
_ACC_ROWS = 384
_MIN_ROWS_PER_DIM = 8


class ShrunkZcaContentMetric(nn.Module):
    def __init__(self, d, gamma=_GAMMA, rho=_RHO, acc_calls=_ACC_CALLS, acc_rows=_ACC_ROWS):
        super().__init__()
        self.dim = int(d)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.acc_calls = max(1, int(acc_calls))
        self.acc_rows = max(1, int(acc_rows))
        self.register_buffer("w_fwd", torch.eye(self.dim))
        self.register_buffer("w_inv", torch.eye(self.dim))
        self.register_buffer("mu", torch.zeros(1, self.dim))
        self.register_buffer("acc_sum", torch.zeros(self.dim))
        self.register_buffer("acc_gram", torch.zeros(self.dim, self.dim))
        self._rows = 0.0
        self._calls = 0
        self._active = False
        self._sealed = False

    @torch.no_grad()
    def _observe(self, z):
        x = z.detach()
        if x.dim() != 2 or x.shape[-1] != self.dim:
            self._sealed = True
            return
        n = int(x.shape[0])
        if n == 0:
            return
        if n > self.acc_rows:
            x = x[:: max(1, n // self.acc_rows)][: self.acc_rows]
        x = x.to(self.acc_gram.dtype)
        if not bool(torch.isfinite(x).all()):
            return
        self.acc_sum.add_(x.sum(0))
        self.acc_gram.add_(x.t() @ x)
        self._rows += float(x.shape[0])
        self._calls += 1
        if self._calls >= self.acc_calls:
            self._seal()

    @torch.no_grad()
    def _seal(self):
        self._sealed = True
        if self._rows < float(_MIN_ROWS_PER_DIM * self.dim):
            return
        try:
            m = self.acc_sum.detach().cpu().double() / self._rows
            gram = self.acc_gram.detach().cpu().double() / self._rows
            cov = gram - torch.outer(m, m)
            cov = 0.5 * (cov + cov.t())
            evals, evecs = torch.linalg.eigh(cov)
        except Exception:
            return
        evals = torch.nan_to_num(evals, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        bar = evals.mean().clamp_min(1e-12)
        lam = evals + self.rho * bar
        gains = (bar / lam).pow(0.5 * self.gamma)
        denom = (gains.pow(2) * evals).sum().clamp_min(1e-12)
        scale = (evals.sum() / denom).sqrt().clamp(0.5, 2.0)
        gains = gains * scale
        w = (evecs * gains.unsqueeze(0)) @ evecs.t()
        w_i = (evecs * gains.reciprocal().unsqueeze(0)) @ evecs.t()
        if not (bool(torch.isfinite(w).all()) and bool(torch.isfinite(w_i).all())):
            return
        if not bool(torch.isfinite(m).all()):
            return
        dt = self.w_fwd.dtype
        dev = self.w_fwd.device
        self.w_fwd.copy_(w.to(dtype=dt).to(dev))
        self.w_inv.copy_(w_i.to(dtype=dt).to(dev))
        self.mu.copy_(m.view(1, -1).to(dtype=dt).to(dev))
        self._active = True

    def make_target(self, z_obs, z_prev):
        if self.training and not self._sealed:
            self._observe(z_obs)
        if not self._active:
            return z_obs
        return (z_obs - self.mu) @ self.w_fwd

    def to_obs(self, pred, z_prev):
        if not self._active:
            return pred
        return pred @ self.w_inv + self.mu

    def reg(self):
        return 0.0


def make(D):
    return ShrunkZcaContentMetric(D)


def make_target(z_obs, z_prev):
    return z_obs


def to_obs(pred, z_prev):
    return pred
