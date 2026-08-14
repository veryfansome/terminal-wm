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
