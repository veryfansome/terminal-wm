import torch
import torch.nn as nn

NAME = "confusable_contrast_metric"
DESCRIPTION = (
    "Trains the model in a linearly re-metricized observation space whose metric is the "
    "covariance of UNIT DIFFERENCE DIRECTIONS between mutually confusable observations, and reads "
    "it back through the exact matching inverse. Every few steps a subsample of the batch's "
    "next-observation targets is taken, all pairwise squared distances are formed, each row's "
    "near-duplicate partners are dropped by a per-row relative floor, the k nearest surviving "
    "partners are taken, and the closest half of those pairs by rank are kept. Each kept pair "
    "contributes its difference vector normalized to unit length, and the mean outer product of "
    "those unit differences is folded into a running buffer whose trace is one by construction. "
    "Once enough updates have accumulated, and periodically thereafter, that buffer is rescaled "
    "to unit mean eigenvalue, shrunk toward the "
    "identity, eigendecomposed on CPU in float64, and turned into the symmetric pair "
    "(C^+p/2, C^-p/2), rescaled so the transformed targets keep unit average per-dimension "
    "second moment. make_target left-multiplies the target by the forward map, so the training "
    "loss measures error as a quadratic form in C^p — error along a direction is charged in "
    "proportion to how strongly that direction separates confusable observations, rather than "
    "uniformly; to_obs left-multiplies the prediction by the inverse map, returning the fixed "
    "observation space used for retrieval. Both maps are one global symmetric matrix, identical "
    "for every step, every window and every candidate, they are exactly mutually inverse by "
    "eigenvalue construction, and they are the identity until the running estimate has warmed up. "
    "The exponent p interpolates between the untouched observation metric at p=0 and the full "
    "contrast covariance at p=1, and the shrinkage bounds the map's condition number."
)

LEARNED = True

_MOMENTUM = 0.97
_SHRINK = 0.10
_POWER = 0.70
_SUB_ROWS = 384
_MIN_ROWS = 64
_K_NEIGHBORS = 6
_DUP_FRAC = 0.03
_KEEP_Q = 0.5
_MIN_PAIRS = 32
_ACCUM_EVERY = 4
_WARMUP_ACCUM = 48
_REFRESH_EVERY = 20
_EIG_FLOOR = 1e-4


class ConfusableContrastMetric(nn.Module):
    def __init__(self, dim, momentum=_MOMENTUM, shrink=_SHRINK, power=_POWER,
                 sub_rows=_SUB_ROWS, min_rows=_MIN_ROWS, k_neighbors=_K_NEIGHBORS,
                 dup_frac=_DUP_FRAC, keep_q=_KEEP_Q, min_pairs=_MIN_PAIRS,
                 accum_every=_ACCUM_EVERY, warmup_accum=_WARMUP_ACCUM,
                 refresh_every=_REFRESH_EVERY, eig_floor=_EIG_FLOOR):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.shrink = float(shrink)
        self.power = float(power)
        self.sub_rows = max(16, int(sub_rows))
        self.min_rows = max(4, int(min_rows))
        self.k_neighbors = max(1, int(k_neighbors))
        self.dup_frac = float(dup_frac)
        self.keep_q = min(1.0, max(0.05, float(keep_q)))
        self.min_pairs = max(1, int(min_pairs))
        self.accum_every = max(1, int(accum_every))
        self.warmup_accum = max(1, int(warmup_accum))
        self.refresh_every = max(1, int(refresh_every))
        self.eig_floor = float(eig_floor)
        self.register_buffer("contrast", torch.eye(self.dim) / float(self.dim))
        self.register_buffer("fwd_map", torch.eye(self.dim))
        self.register_buffer("inv_map", torch.eye(self.dim))
        self.register_buffer("accums", torch.zeros((), dtype=torch.long))
        self._calls = 0
        self._accum_count = 0

    def _map_through(self, x, mat):
        if x is None or not torch.is_tensor(x) or x.dim() < 1 or int(x.shape[-1]) != self.dim:
            return x
        m = mat.to(device=x.device, dtype=x.dtype)
        return x @ m.t()

    @torch.no_grad()
    def _rebuild_maps(self):
        c = self.contrast.detach().to(device="cpu", dtype=torch.float64)
        c = 0.5 * (c + c.t())
        if not bool(torch.isfinite(c).all()):
            return
        tr = c.diagonal().sum()
        if not bool(torch.isfinite(tr)) or float(tr) <= 0.0:
            return
        c = c * (float(self.dim) / tr)
        eye = torch.eye(self.dim, dtype=c.dtype)
        c = (1.0 - self.shrink) * c + self.shrink * eye
        try:
            evals, evecs = torch.linalg.eigh(c)
        except Exception:
            return
        if not (bool(torch.isfinite(evals).all()) and bool(torch.isfinite(evecs).all())):
            return
        lam = evals.clamp_min(self.eig_floor)
        half = 0.5 * self.power
        f = lam.pow(half)
        v = lam.pow(-half)
        scale = 1.0 / lam.pow(self.power).mean().clamp_min(1e-8).sqrt()
        f = f * scale
        v = v / scale
        fwd = (evecs * f.unsqueeze(0)) @ evecs.t()
        inv = (evecs * v.unsqueeze(0)) @ evecs.t()
        if not (bool(torch.isfinite(fwd).all()) and bool(torch.isfinite(inv).all())):
            return
        self.fwd_map.copy_(fwd.to(dtype=self.fwd_map.dtype))
        self.inv_map.copy_(inv.to(dtype=self.inv_map.dtype))

    @torch.no_grad()
    def _observe_pairs(self, z):
        zf = z.detach().reshape(-1, self.dim).to(dtype=torch.float32)
        if zf.shape[0] < self.min_rows:
            return False
        if not bool(torch.isfinite(zf).all()):
            return False
        if zf.shape[0] > self.sub_rows:
            sel = torch.randperm(zf.shape[0], device=zf.device)[:self.sub_rows]
            zf = zf.index_select(0, sel)
        n = int(zf.shape[0])
        if n < self.min_rows:
            return False
        sq = zf.pow(2).sum(dim=1, keepdim=True)
        d2 = (sq + sq.t() - 2.0 * (zf @ zf.t())).clamp_min(0.0)
        eye = torch.eye(n, dtype=torch.bool, device=zf.device)
        inf = torch.full_like(d2, float("inf"))
        row_sum = d2.masked_fill(eye, 0.0).sum(dim=1)
        row_mean = row_sum / float(n - 1)
        floor = (self.dup_frac * row_mean).clamp_min(1e-12).unsqueeze(1)
        usable = (~eye) & (d2 > floor)
        d2m = torch.where(usable, d2, inf)
        k = min(self.k_neighbors, n - 1)
        nd, nj = torch.topk(d2m, k, dim=1, largest=False)
        finite = torch.isfinite(nd)
        n_finite = int(finite.sum())
        if n_finite < self.min_pairs:
            return False
        vals = nd[finite]
        thr = torch.quantile(vals, self.keep_q)
        if not bool(torch.isfinite(thr)):
            return False
        keep = finite & (nd <= thr)
        rows, cols = torch.nonzero(keep, as_tuple=True)
        if int(rows.numel()) < self.min_pairs:
            return False
        partners = nj[rows, cols]
        diff = zf.index_select(0, rows) - zf.index_select(0, partners)
        norm = diff.pow(2).sum(dim=1, keepdim=True)
        good = (norm.squeeze(1) > 0.0) & torch.isfinite(norm.squeeze(1))
        if int(good.sum()) < self.min_pairs:
            return False
        diff = diff[good]
        norm = norm[good]
        diff = diff * torch.rsqrt(norm.clamp_min(1e-12))
        if not bool(torch.isfinite(diff).all()):
            return False
        c = (diff.t() @ diff) / float(diff.shape[0])
        if not bool(torch.isfinite(c).all()):
            return False
        c = c.to(device=self.contrast.device, dtype=self.contrast.dtype)
        self.contrast.mul_(self.momentum).add_(c, alpha=1.0 - self.momentum)
        self.accums += 1
        self._accum_count += 1
        return True

    def make_target(self, z_obs, z_prev):
        usable = (torch.is_tensor(z_obs) and z_obs.dim() >= 1
                  and int(z_obs.shape[-1]) == self.dim)
        if usable and self.training and torch.is_grad_enabled():
            self._calls += 1
            if self._calls % self.accum_every == 0 and self._observe_pairs(z_obs):
                if self._accum_count >= self.warmup_accum:
                    if (self._accum_count - self.warmup_accum) % self.refresh_every == 0:
                        self._rebuild_maps()
        src = z_obs.detach() if torch.is_tensor(z_obs) else z_obs
        return self._map_through(src, self.fwd_map)

    def to_obs(self, pred, z_prev):
        return self._map_through(pred, self.inv_map)

    def reg(self):
        return 0.0


def make(D):
    return ConfusableContrastMetric(int(D))
