import math

import torch
import torch.nn as nn

NAME = "rank_gaussianized_contrast_metric_target"
DESCRIPTION = (
    "One full Gaussianization cycle over the observation space: a per-dimension rank transform "
    "first, then a linear re-metrization estimated in the rank-transformed coordinates, read back "
    "through the exact composed inverse. Stage one passes each of the 768 standardized observation "
    "dimensions through its own strictly monotone piecewise-linear map whose x knots are that "
    "dimension's empirical equal-mass quantiles and whose y knots are the standard-normal quantiles "
    "of the same probability levels, fitted once from a stride subsample of the training targets "
    "over the first few hundred steps and then frozen, with per-segment slopes capped to a bounded "
    "multiple of each dimension's median slope, linear outer segments so the map is a bijection on "
    "the whole line, and identity on dimensions with no spread. Stage two runs entirely on those "
    "rank-Gaussianized coordinates: a TOTAL second moment and a CONTRAST second moment are "
    "accumulated online under no_grad with no trainable parameters, the contrast one from "
    "differences between mutually nearest observations inside the SAME trajectory, trajectory "
    "boundaries being read off the all-zero previous-observation row that marks a sequence start, "
    "keeping only the closer half of those pairs so exact repeats and far pairs contribute nothing. "
    "The linear factor is Cov_contrast^(q/2) times Cov_total^(-a/2), each factor built by an "
    "eigendecomposition on CPU in float64 after shrinkage toward the identity, with log-centred "
    "capped per-direction gains so each factor is volume preserving and well conditioned, and a "
    "single global scale that holds the transformed target at unit mean per-dimension variance. "
    "Because every marginal has already been made standard normal, that total second moment IS the "
    "Gaussian-rank (normal-scores) correlation matrix, a bounded-influence estimator whose "
    "eigenvectors cannot be dragged by a handful of heavy-tailed rows the way a raw embedding "
    "covariance can, and the contrast neighbours are chosen by rank-space distance rather than raw "
    "distance. The composite forward map is linear-after-rank; the inverse applies the linear "
    "inverse assembled from the same eigenpairs and then the same knots read backwards, so it is an "
    "exact algebraic inverse of the composite. The two stages are staged in time: rank knots are "
    "fitted and frozen before any second moment is accumulated, the linear exponents ramp in over a "
    "fixed span while the factor is periodically rebuilt, and the whole map is frozen for the "
    "remainder of training. Both stages hold buffers only, read neither z_prev nor any observation "
    "at read time, and are the identity until fitted, so the map behaves identically at a position "
    "whose own observation is withheld."
)

LEARNED = True

_MIN_ACC_ROWS = 8
_MIN_PAIRS = 4


def _all_finite(*tensors):
    for t in tensors:
        if not bool(torch.isfinite(t).all()):
            return False
    return True


class RankGaussianizedContrastMetricTarget(nn.Module):
    def __init__(self, dim, knots=64, marg_fit_calls=200, rows_per_call=64,
                 marg_min_rows=4096, marg_max_rows=12288, calib_rows=4096,
                 slope_ratio=4.0, min_spread=1e-3, gap=1e-4, marg_deadline=3.0,
                 contrast_power=1.0, whiten_power=0.5, shrink=0.05, gain_cap=3.0,
                 lin_warmup=150, lin_ramp_span=500, lin_rebuild_every=125,
                 lin_freeze_after=950, accum_stride=2, max_rows=640,
                 dup_frac=1e-3, keep_frac=0.5, eig_floor=1e-3):
        super().__init__()
        self.dim = int(dim)
        self.segments = max(4, int(knots))
        self.marg_fit_calls = max(1, int(marg_fit_calls))
        self.rows_per_call = max(1, int(rows_per_call))
        self.marg_min_rows = max(64, int(marg_min_rows))
        self.marg_max_rows = max(self.marg_min_rows, int(marg_max_rows))
        self.calib_rows = max(256, int(calib_rows))
        self.slope_ratio = max(1.0, float(slope_ratio))
        self.min_spread = float(min_spread)
        self.gap = float(gap)
        self.marg_deadline = max(1.0, float(marg_deadline)) * float(self.marg_fit_calls)

        self.contrast_power = float(contrast_power)
        self.whiten_power = float(whiten_power)
        self.shrink = min(0.5, max(1e-4, float(shrink)))
        self.log_cap = math.log(max(1.0 + 1e-6, float(gain_cap)))
        self.lin_warmup = max(1, int(lin_warmup))
        self.lin_ramp_span = max(1, int(lin_ramp_span))
        self.lin_rebuild_every = max(1, int(lin_rebuild_every))
        self.lin_freeze_after = max(self.lin_warmup, int(lin_freeze_after))
        self.accum_stride = max(1, int(accum_stride))
        self.max_rows = max(32, int(max_rows))
        self.dup_frac = float(dup_frac)
        self.keep_frac = min(1.0, max(0.05, float(keep_frac)))
        self.eig_floor = float(eig_floor)

        n_knot = self.segments + 1
        grid = torch.linspace(0.0, 1.0, n_knot)
        base = grid.unsqueeze(0).expand(self.dim, n_knot).contiguous()
        self.register_buffer("knot_x", base.clone())
        self.register_buffer("knot_y", base.clone())
        self.register_buffer("marg_ready", torch.zeros(()))
        self.register_buffer("marg_dead", torch.zeros(()))
        self.register_buffer("calls", torch.zeros(()))
        self.register_buffer("lin_t0", torch.zeros(()))

        self.register_buffer("row_sum", torch.zeros(self.dim))
        self.register_buffer("row_moment", torch.zeros(self.dim, self.dim))
        self.register_buffer("row_count", torch.zeros(()))
        self.register_buffer("pair_moment", torch.zeros(self.dim, self.dim))
        self.register_buffer("pair_count", torch.zeros(()))
        self.register_buffer("center", torch.zeros(self.dim))
        self.register_buffer("fwd", torch.eye(self.dim))
        self.register_buffer("inv", torch.eye(self.dim))
        self.register_buffer("lin_live", torch.zeros(()))

        self._rows = []
        self._n_rows = 0

    def _settled(self):
        return float(self.marg_ready) > 0.5 or float(self.marg_dead) > 0.5

    def _levels(self, n_knot, dtype):
        return (torch.arange(n_knot, dtype=dtype) + 0.5) / float(n_knot)

    def _collect_marginal(self, z):
        flat = z.reshape(-1, self.dim)
        n = int(flat.shape[0])
        if n < 1:
            return
        stride = max(1, n // self.rows_per_call)
        take = flat[::stride][: self.rows_per_call].to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(take).all()):
            return
        self._rows.append(take)
        self._n_rows += int(take.shape[0])

    def _fit_marginal(self):
        try:
            if not self._rows:
                return False
            data = torch.cat(self._rows, dim=0)
            self._rows = []
            if int(data.shape[0]) > self.marg_max_rows:
                step = int(data.shape[0]) // self.marg_max_rows + 1
                data = data[::step]
            n = int(data.shape[0])
            if n < self.marg_min_rows:
                return False
            ordered, _ = torch.sort(data, dim=0)
            n_knot = self.segments + 1
            levels = self._levels(n_knot, torch.float64)
            pos = (levels * float(n - 1)).round().long().clamp(0, n - 1)
            kx = ordered.index_select(0, pos).t().to(torch.float64).contiguous()
            del ordered
            spread = kx[:, -1] - kx[:, 0]
            steps = (kx[:, 1:] - kx[:, :-1]).clamp_min(self.gap)
            kx = torch.cat([kx[:, :1], kx[:, :1] + torch.cumsum(steps, dim=1)], dim=1)
            ky_ref = (torch.sqrt(torch.tensor(2.0, dtype=torch.float64))
                      * torch.erfinv(2.0 * levels - 1.0))
            ref_rise = (ky_ref[1:] - ky_ref[:-1]).unsqueeze(0)
            slope = ref_rise / (kx[:, 1:] - kx[:, :-1])
            median = slope.median(dim=1, keepdim=True).values.clamp_min(1e-12)
            slope = torch.max(torch.min(slope, median * self.slope_ratio), median / self.slope_ratio)
            rise = (slope * (kx[:, 1:] - kx[:, :-1])).clamp_min(self.gap)
            ky = torch.cat([torch.zeros(self.dim, 1, dtype=torch.float64),
                            torch.cumsum(rise, dim=1)], dim=1)
            span = ky[:, -1:].clamp_min(1e-12)
            ky = ky * ((ky_ref[-1] - ky_ref[0]) / span)
            calib = data[:: max(1, n // self.calib_rows)].to(torch.float64)
            mapped = self._piecewise(calib, kx, ky)
            centre = mapped.mean(dim=0).unsqueeze(1)
            scale = mapped.std(dim=0).unsqueeze(1).clamp_min(1e-6)
            ky = (ky - centre) / scale
            flat_dims = (spread <= self.min_spread).unsqueeze(1)
            ky = torch.where(flat_dims, kx, ky)
            kx = kx.to(device=self.knot_x.device, dtype=self.knot_x.dtype)
            ky = ky.to(device=self.knot_y.device, dtype=self.knot_y.dtype)
            if not _all_finite(kx, ky):
                return False
            if bool((kx[:, 1:] <= kx[:, :-1]).any()) or bool((ky[:, 1:] <= ky[:, :-1]).any()):
                return False
            self.knot_x.copy_(kx)
            self.knot_y.copy_(ky)
            return True
        except Exception:
            self._rows = []
            return False

    def _piecewise(self, x, src, dst):
        shape = x.shape
        flat = x.reshape(-1, self.dim)
        work = flat.dtype if flat.dtype in (torch.float32, torch.float64) else torch.float32
        v = flat.to(work).t().contiguous()
        s = src.to(device=v.device, dtype=work)
        d = dst.to(device=v.device, dtype=work)
        idx = torch.searchsorted(s.contiguous(), v)
        idx = (idx - 1).clamp_(0, self.segments - 1)
        x0 = torch.gather(s, 1, idx)
        x1 = torch.gather(s, 1, idx + 1)
        y0 = torch.gather(d, 1, idx)
        y1 = torch.gather(d, 1, idx + 1)
        out = y0 + (v - x0) * (y1 - y0) / (x1 - x0)
        return out.t().contiguous().reshape(shape).to(flat.dtype)

    def _shaped(self, x):
        return torch.is_tensor(x) and x.dim() >= 1 and int(x.shape[-1]) == self.dim

    def _marginal_fwd(self, x):
        if float(self.marg_ready) < 0.5 or not self._shaped(x):
            return x
        return self._piecewise(x, self.knot_x, self.knot_y)

    def _marginal_inv(self, y):
        if float(self.marg_ready) < 0.5 or not self._shaped(y):
            return y
        return self._piecewise(y, self.knot_y, self.knot_x)

    def _linear_fwd(self, g):
        if float(self.lin_live) < 0.5 or not self._shaped(g):
            return g
        m = self.fwd.to(device=g.device, dtype=g.dtype)
        o = self.center.to(device=g.device, dtype=g.dtype)
        return (g - o) @ m.t()

    def _linear_inv(self, p):
        if float(self.lin_live) < 0.5 or not self._shaped(p):
            return p
        m = self.inv.to(device=p.device, dtype=p.dtype)
        o = self.center.to(device=p.device, dtype=p.dtype)
        return p @ m.t() + o

    def _segments_of(self, z_prev, n, device):
        if not torch.is_tensor(z_prev) or z_prev.dim() != 2 or z_prev.shape[0] != n:
            return torch.zeros(n, dtype=torch.long, device=device)
        starts = z_prev.detach().abs().sum(dim=1) == 0
        if not bool(starts.any()):
            return torch.zeros(n, dtype=torch.long, device=device)
        return (torch.cumsum(starts.long(), dim=0) - 1).clamp_min(0)

    def _accumulate(self, g_rows, z_prev):
        if not torch.is_tensor(g_rows) or g_rows.dim() != 2:
            return
        if g_rows.shape[1] != self.dim or g_rows.shape[0] < _MIN_ACC_ROWS:
            return
        z = g_rows.detach()
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

        seg = self._segments_of(prev, n, z.device)
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
        dmin, near_idx = torch.where(usable, d2, inf).min(dim=1)
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
        diff = z[rows] - z[near_idx[rows]]
        if not _all_finite(diff):
            return
        self.pair_moment.add_(diff.t() @ diff)
        self.pair_count.add_(float(rows.numel()))

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
        self.lin_live.fill_(1.0)

    def _advance_marginal(self, z):
        self.calls.add_(1.0)
        self._collect_marginal(z)
        ripe = float(self.calls) >= self.marg_fit_calls and self._n_rows >= self.marg_min_rows
        overdue = float(self.calls) >= self.marg_deadline
        if not (ripe or overdue):
            return
        if self._fit_marginal():
            self.marg_ready.fill_(1.0)
        else:
            self.marg_dead.fill_(1.0)
        self._rows = []
        self._n_rows = 0
        self.lin_t0.copy_(self.calls)

    def _advance_linear(self, g, z_prev):
        rel = float(self.calls) - float(self.lin_t0)
        if rel > self.lin_freeze_after or rel < 0.0:
            return
        step = int(rel)
        if step % self.accum_stride == 0:
            self._accumulate(g, z_prev)
        if step >= self.lin_warmup and (step - self.lin_warmup) % self.lin_rebuild_every == 0:
            ramp = float(step - self.lin_warmup) / float(self.lin_ramp_span)
            self._rebuild(min(1.0, max(0.0, ramp)))

    def make_target(self, z_obs, z_prev):
        if not self._shaped(z_obs):
            return z_obs
        live = bool(self.training) and bool(torch.is_grad_enabled())
        with torch.no_grad():
            z = z_obs.detach()
            if live and not self._settled():
                self._advance_marginal(z)
            elif live:
                self.calls.add_(1.0)
            g = self._marginal_fwd(z)
            if live and self._settled():
                self._advance_linear(g, z_prev)
            return self._linear_fwd(g)

    def to_obs(self, pred, z_prev):
        return self._marginal_inv(self._linear_inv(pred))

    def reg(self):
        return 0.0


def make(D):
    return RankGaussianizedContrastMetricTarget(int(D))
