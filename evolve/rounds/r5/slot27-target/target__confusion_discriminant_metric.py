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
