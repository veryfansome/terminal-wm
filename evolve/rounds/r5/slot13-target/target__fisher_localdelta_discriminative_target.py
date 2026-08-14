import torch
import torch.nn as nn

NAME = "fisher_localdelta_discriminative_target"
DESCRIPTION = (
    "Trains the model in a linearly re-metricized observation space built from TWO running "
    "second-moment estimates and reads it back through the matching inverse. The first estimate "
    "is the second moment of the standardized next-observation targets, shrunk toward the "
    "identity and turned into a whitening factor of power p on CPU in float64 — a plain global "
    "whitened target space. The second estimate is of LOCAL CHANGE directions: the step-to-step "
    "difference d = z_obs - z_prev is formed for every row whose previous observation exists, "
    "rows with an exactly-zero difference are dropped, and of the rest only the lower half by "
    "difference magnitude is accumulated, each contribution unit-normalized so a few large "
    "differences cannot dominate the direction estimate. The local-change moment is mapped into "
    "the fully whitened frame and eigendecomposed there every few steps; each eigendirection "
    "gets a gain equal to its eigenvalue divided by the mean eigenvalue, raised to a power and "
    "clamped to a bounded range, so directions along which consecutive observations of one "
    "trajectory differ slightly are stretched relative to directions along which the corpus as a "
    "whole varies. The forward map is that gain operator composed with the whitening factor, "
    "rescaled so the transformed targets keep unit average per-dimension variance; the inverse "
    "map is the exact composition in reverse, so the retrieval eval still happens in the fixed "
    "observation space. Both maps are one global matrix, identical for every step, every window "
    "and every candidate, and they are the identity until both running estimates have warmed up. "
    "The previous observation enters only as a statistic used to build the maps — neither map "
    "takes it as an argument, and setting the gain power to zero recovers plain whitening "
    "exactly."
)

LEARNED = True

_MOMENTUM = 0.95
_DELTA_MOMENTUM = 0.9
_SHRINK = 0.05
_DELTA_SHRINK = 0.10
_POWER = 0.75
_DISC_POWER = 0.75
_GAIN_CAP = 3.0
_KEEP_Q = 0.5
_MIN_ROWS = 32
_REFRESH_EVERY = 100
_WARMUP = 200
_EIG_FLOOR = 1e-4


class FisherLocalDeltaDiscriminativeTarget(nn.Module):
    def __init__(self, dim, momentum=_MOMENTUM, delta_momentum=_DELTA_MOMENTUM,
                 shrink=_SHRINK, delta_shrink=_DELTA_SHRINK, power=_POWER,
                 disc_power=_DISC_POWER, gain_cap=_GAIN_CAP, keep_q=_KEEP_Q,
                 min_rows=_MIN_ROWS, refresh_every=_REFRESH_EVERY, warmup=_WARMUP,
                 eig_floor=_EIG_FLOOR):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.delta_momentum = float(delta_momentum)
        self.shrink = float(shrink)
        self.delta_shrink = float(delta_shrink)
        self.power = float(power)
        self.disc_power = float(disc_power)
        self.gain_cap = max(1.0, float(gain_cap))
        self.keep_q = min(1.0, max(0.05, float(keep_q)))
        self.min_rows = max(8, int(min_rows))
        self.refresh_every = max(1, int(refresh_every))
        self.warmup = max(1, int(warmup))
        self.eig_floor = float(eig_floor)
        self.register_buffer("second_moment", torch.eye(self.dim))
        self.register_buffer("delta_moment", torch.eye(self.dim))
        self.register_buffer("fwd_map", torch.eye(self.dim))
        self.register_buffer("inv_map", torch.eye(self.dim))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))
        self._n_observed = 0
        self._n_delta = 0

    def _linear_map(self, x, mat):
        if x is None or not torch.is_tensor(x) or x.dim() < 1 or int(x.shape[-1]) != self.dim:
            return x
        m = mat.to(device=x.device, dtype=x.dtype)
        return x @ m.t()

    @torch.no_grad()
    def _observe_delta(self, zf, pf):
        live = pf.pow(2).sum(dim=1) > 0.0
        if int(live.sum()) < self.min_rows:
            return
        d = zf - pf
        d2 = d.pow(2).sum(dim=1)
        ref = d2[live].mean().clamp_min(1e-12)
        pos = live & (d2 > 1e-4 * ref)
        cnt = int(pos.sum())
        if cnt < self.min_rows:
            return
        k = int(cnt * self.keep_q)
        if k < 8:
            k = 8
        if k > cnt:
            k = cnt
        thr = torch.sort(d2[pos]).values[k - 1]
        keep = pos & (d2 <= thr)
        rows = int(keep.sum())
        if rows < 8:
            return
        dk = d[keep]
        dk = dk / dk.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        md = (dk.t() @ dk) * (float(self.dim) / float(rows))
        if not bool(torch.isfinite(md).all()):
            return
        self.delta_moment.mul_(self.delta_momentum).add_(md, alpha=1.0 - self.delta_momentum)
        self._n_delta += 1

    @torch.no_grad()
    def _observe(self, z, p):
        zf = z.detach().reshape(-1, self.dim)
        n = int(zf.shape[0])
        if n < 2:
            return
        dev = self.second_moment.device
        zf = zf.to(device=dev, dtype=torch.float32)
        if not bool(torch.isfinite(zf).all()):
            return
        m = (zf.t() @ zf) / float(n)
        self.second_moment.mul_(self.momentum).add_(m, alpha=1.0 - self.momentum)
        self._n_observed += 1
        self.seen += 1
        if torch.is_tensor(p) and tuple(p.shape) == tuple(z.shape):
            pf = p.detach().reshape(-1, self.dim).to(device=dev, dtype=torch.float32)
            if bool(torch.isfinite(pf).all()):
                self._observe_delta(zf, pf)
        if (self._n_observed >= self.warmup and self._n_delta >= self.warmup
                and self._n_observed % self.refresh_every == 0):
            self._rebuild_maps()

    @torch.no_grad()
    def _rebuild_maps(self):
        s = self.second_moment.detach().to(device="cpu", dtype=torch.float64)
        s = 0.5 * (s + s.t())
        sd = self.delta_moment.detach().to(device="cpu", dtype=torch.float64)
        sd = 0.5 * (sd + sd.t())
        if not (bool(torch.isfinite(s).all()) and bool(torch.isfinite(sd).all())):
            return
        eye = torch.eye(self.dim, dtype=s.dtype)
        s = (1.0 - self.shrink) * s + self.shrink * eye
        tr = float(sd.diagonal().sum())
        if not (tr == tr) or tr <= 0.0:
            return
        sd = sd * (float(self.dim) / tr)
        sd = (1.0 - self.delta_shrink) * sd + self.delta_shrink * eye
        try:
            mu, vecs = torch.linalg.eigh(s)
        except Exception:
            return
        if not (bool(torch.isfinite(mu).all()) and bool(torch.isfinite(vecs).all())):
            return
        mu = mu.clamp_min(self.eig_floor)
        half = 0.5 * self.power
        w_full = (vecs * mu.pow(-0.5).unsqueeze(0)) @ vecs.t()
        w_fwd = (vecs * mu.pow(-half).unsqueeze(0)) @ vecs.t()
        w_bwd = (vecs * mu.pow(half).unsqueeze(0)) @ vecs.t()
        w_res = (vecs * mu.pow(0.5 * (1.0 - self.power)).unsqueeze(0)) @ vecs.t()
        a = w_full @ sd @ w_full
        a = 0.5 * (a + a.t())
        if not bool(torch.isfinite(a).all()):
            return
        try:
            lam, basis = torch.linalg.eigh(a)
        except Exception:
            return
        if not (bool(torch.isfinite(lam).all()) and bool(torch.isfinite(basis).all())):
            return
        lam = lam.clamp_min(0.0)
        bar = lam.mean().clamp_min(1e-8)
        gain = (lam / bar).clamp_min(1e-8).pow(0.5 * self.disc_power)
        gain = gain.clamp(1.0 / self.gain_cap, self.gain_cap)
        rmat = w_res @ basis
        col = (rmat * rmat).sum(dim=0)
        t = float((gain.pow(2) * col).sum())
        if not (t == t) or t <= 0.0:
            return
        scale = (float(self.dim) / t) ** 0.5
        g_fwd = (basis * gain.unsqueeze(0)) @ basis.t()
        g_bwd = (basis * gain.reciprocal().unsqueeze(0)) @ basis.t()
        fwd = (g_fwd @ w_fwd) * scale
        inv = (w_bwd @ g_bwd) / scale
        if not (bool(torch.isfinite(fwd).all()) and bool(torch.isfinite(inv).all())):
            return
        self.fwd_map.copy_(fwd.to(dtype=self.fwd_map.dtype))
        self.inv_map.copy_(inv.to(dtype=self.inv_map.dtype))

    def make_target(self, z_obs, z_prev):
        if self.training and torch.is_grad_enabled() and torch.is_tensor(z_obs) \
                and z_obs.dim() >= 1 and int(z_obs.shape[-1]) == self.dim:
            self._observe(z_obs, z_prev)
        return self._linear_map(z_obs.detach() if torch.is_tensor(z_obs) else z_obs, self.fwd_map)

    def to_obs(self, pred, z_prev):
        return self._linear_map(pred, self.inv_map)

    def reg(self):
        return 0.0


def make(D):
    return FisherLocalDeltaDiscriminativeTarget(int(D))
