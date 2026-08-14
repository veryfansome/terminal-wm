import torch
import torch.nn as nn

NAME = "equal_mass_normal_score_target"
DESCRIPTION = (
    "Trains the model against the per-dimension NORMAL SCORE of the next observation and reads it "
    "back through the exact matching inverse. Each of the 768 standardized observation dimensions "
    "is passed through its own strictly monotone piecewise-linear map whose x knots are that "
    "dimension's empirical equal-mass quantiles, estimated once from a stride-subsample of the "
    "training targets spread over the first few hundred steps and then frozen, and whose y knots "
    "are the standard-normal quantiles of the same probability levels, so the transformed target "
    "has a standard-normal marginal in every dimension. The map is the probability integral "
    "transform: its local slope is proportional to the density, so it stretches the value regions "
    "where observations PILE UP and compresses the regions where they are already far apart. "
    "Squared error in the transformed space therefore measures the probability MASS lying between "
    "the prediction and the truth rather than the raw distance, which is the quantity a "
    "forced-choice retrieval actually decides on: how many other observations sit between the two. "
    "Confusable observations - same cwd, same exit code, same output shape, different file "
    "content - live inside the dense core of every dimension and have their tiny separations "
    "amplified; observations that are already trivially distinguishable live in the tails and are "
    "compressed. Per-segment slopes are capped to a bounded multiple of each dimension's median "
    "slope so a plateau of exactly repeated values cannot produce an unbounded expansion, the "
    "outer segments extend linearly so the map is a bijection on the whole line, and a dimension "
    "with no spread is left as the identity. The transform reshapes resolution ALONG the value "
    "axis inside each dimension, which no linear re-metrization of the space can express, holds "
    "no trainable parameters, is one global function identical for every step, window and "
    "candidate, is the identity until the knots are fitted, and reads neither z_prev nor any "
    "observation, so it behaves identically at a position whose own observation is withheld."
)

LEARNED = True

_KNOTS = 64
_FIT_CALLS = 200
_ROWS_PER_CALL = 64
_MIN_ROWS = 4096
_MAX_ROWS = 12288
_CALIB_ROWS = 4096
_SLOPE_RATIO = 4.0
_MIN_SPREAD = 1e-3
_GAP = 1e-4


class EqualMassNormalScoreTarget(nn.Module):
    def __init__(self, dim, knots=_KNOTS, fit_calls=_FIT_CALLS, rows_per_call=_ROWS_PER_CALL,
                 min_rows=_MIN_ROWS, max_rows=_MAX_ROWS, calib_rows=_CALIB_ROWS,
                 slope_ratio=_SLOPE_RATIO, min_spread=_MIN_SPREAD, gap=_GAP):
        super().__init__()
        self.dim = int(dim)
        self.segments = max(4, int(knots))
        self.fit_calls = max(1, int(fit_calls))
        self.rows_per_call = max(1, int(rows_per_call))
        self.min_rows = max(64, int(min_rows))
        self.max_rows = max(self.min_rows, int(max_rows))
        self.calib_rows = max(256, int(calib_rows))
        self.slope_ratio = max(1.0, float(slope_ratio))
        self.min_spread = float(min_spread)
        self.gap = float(gap)
        n_knot = self.segments + 1
        grid = torch.linspace(0.0, 1.0, n_knot)
        base = grid.unsqueeze(0).expand(self.dim, n_knot).contiguous()
        self.register_buffer("knot_x", base.clone())
        self.register_buffer("knot_y", base.clone())
        self.register_buffer("ready", torch.zeros(()))
        self.register_buffer("dead", torch.zeros(()))
        self.register_buffer("calls", torch.zeros(()))
        self._rows = []
        self._n_rows = 0

    def _levels(self, n_knot, dtype):
        return (torch.arange(n_knot, dtype=dtype) + 0.5) / float(n_knot)

    def _collect(self, z):
        flat = z.detach().reshape(-1, self.dim)
        n = int(flat.shape[0])
        if n < 1:
            return
        stride = max(1, n // self.rows_per_call)
        take = flat[::stride][: self.rows_per_call].to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(take).all()):
            return
        self._rows.append(take)
        self._n_rows += int(take.shape[0])

    def _fit(self):
        try:
            data = torch.cat(self._rows, dim=0)
            self._rows = []
            if int(data.shape[0]) > self.max_rows:
                step = int(data.shape[0]) // self.max_rows + 1
                data = data[::step]
            n = int(data.shape[0])
            if n < self.min_rows:
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
            if not (bool(torch.isfinite(kx).all()) and bool(torch.isfinite(ky).all())):
                return False
            if bool((kx[:, 1:] <= kx[:, :-1]).any()) or bool((ky[:, 1:] <= ky[:, :-1]).any()):
                return False
            self.knot_x.copy_(kx)
            self.knot_y.copy_(ky)
            return True
        except Exception:
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

    def _usable(self, x):
        if not torch.is_tensor(x) or x.dim() < 1 or int(x.shape[-1]) != self.dim:
            return False
        return float(self.ready) > 0.5 and float(self.dead) < 0.5

    def make_target(self, z_obs, z_prev):
        if torch.is_tensor(z_obs) and z_obs.dim() >= 1 and int(z_obs.shape[-1]) == self.dim \
                and self.training and torch.is_grad_enabled() and float(self.ready) < 0.5 \
                and float(self.dead) < 0.5:
            with torch.no_grad():
                self.calls.add_(1.0)
                self._collect(z_obs)
                if float(self.calls) >= self.fit_calls and self._n_rows >= self.min_rows:
                    if self._fit():
                        self.ready.fill_(1.0)
                    else:
                        self.dead.fill_(1.0)
                    self._rows = []
        if not self._usable(z_obs):
            return z_obs
        with torch.no_grad():
            return self._piecewise(z_obs.detach(), self.knot_x, self.knot_y)

    def to_obs(self, pred, z_prev):
        if not self._usable(pred):
            return pred
        return self._piecewise(pred, self.knot_y, self.knot_x)

    def reg(self):
        return 0.0


def make(D):
    return EqualMassNormalScoreTarget(int(D))
