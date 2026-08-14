import torch
import torch.nn as nn

NAME = "confusable_pair_scatter_metric"
DESCRIPTION = (
    "Trains the model in an invertibly re-metrized observation space whose shape comes from the "
    "CONFUSABLE-PAIR SCATTER rather than from the marginal spread of the targets. Every step, the "
    "batch's standardized next-observation targets are subsampled, all pairwise squared distances "
    "are formed, pairs closer than a fraction of the mean pairwise distance are discarded as "
    "near-duplicates, and each row's k closest surviving partners give difference vectors whose "
    "second moment B is accumulated as a running buffer. B is symmetrized, trace-normalized, "
    "lightly shrunk toward the identity and eigendecomposed on CPU in float64 every few steps; the "
    "forward map is the symmetric matrix power B^(q/2) with its eigen-gains normalized by their "
    "geometric mean and clamped to a fixed band, and the inverse map is the same eigenbasis with "
    "reciprocal gains, so the pair is an exact algebraic inverse. The whole map is then rescaled so "
    "the transformed targets keep unit average per-dimension variance under the running marginal "
    "second moment, which leaves the objective's absolute distance constants in their calibrated "
    "range. A POSITIVE power is used because the retrieval decision is flipped only by prediction "
    "error along directions in which two confusable observations actually differ: the risk "
    "derivative with respect to error variance in a direction is proportional to the candidates' "
    "difference variance in that direction, so error is worth buying down in proportion to B, and "
    "directions that every candidate in a window shares - rendering format, exit line, verb "
    "identity - get a small gain and stop competing for the fixed-rank readout's capacity. Both "
    "maps are one global matrix, identical for every step, every window and every candidate, and "
    "they are the identity until the running estimate has warmed up."
)

LEARNED = True

_MOMENTUM = 0.95
_PAIR_MOMENTUM = 0.98
_SHRINK = 0.05
_POWER = 1.0
_GAIN_MIN = 0.4
_GAIN_MAX = 2.5
_REFRESH_EVERY = 50
_WARMUP = 200
_EIG_FLOOR = 1e-6
_MAX_PAIR_ROWS = 256
_NEIGHBORS = 4
_DUP_FRAC = 0.02
_MIN_ROWS = 8
_INVERSE_TOL = 1e-5


class ConfusablePairScatterMetric(nn.Module):
    def __init__(self, dim, momentum=_MOMENTUM, pair_momentum=_PAIR_MOMENTUM, shrink=_SHRINK,
                 power=_POWER, gain_min=_GAIN_MIN, gain_max=_GAIN_MAX,
                 refresh_every=_REFRESH_EVERY, warmup=_WARMUP, eig_floor=_EIG_FLOOR,
                 max_pair_rows=_MAX_PAIR_ROWS, neighbors=_NEIGHBORS, dup_frac=_DUP_FRAC):
        super().__init__()
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.pair_momentum = float(pair_momentum)
        self.shrink = float(shrink)
        self.power = float(power)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)
        self.refresh_every = max(1, int(refresh_every))
        self.warmup = max(1, int(warmup))
        self.eig_floor = float(eig_floor)
        self.max_pair_rows = max(16, int(max_pair_rows))
        self.neighbors = max(1, int(neighbors))
        self.dup_frac = float(dup_frac)
        self.register_buffer("second_moment", torch.eye(self.dim))
        self.register_buffer("pair_scatter", torch.eye(self.dim))
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
    def _subsample(self, zf):
        n = int(zf.shape[0])
        k = min(self.max_pair_rows, n)
        if n <= k:
            return zf
        gen = torch.Generator()
        gen.manual_seed(9176411 + self._n_observed)
        sel = torch.randperm(n, generator=gen)[:k].to(zf.device)
        return zf.index_select(0, sel)

    @torch.no_grad()
    def _pair_scatter(self, zf):
        s = self._subsample(zf)
        m = int(s.shape[0])
        if m < _MIN_ROWS:
            return None
        sq = (s * s).sum(dim=1)
        d2 = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (s @ s.t())).clamp_min(0.0)
        diag = torch.eye(m, dtype=torch.bool, device=d2.device)
        mean_off = d2.masked_fill(diag, 0.0).sum() / float(m * (m - 1))
        if not bool(torch.isfinite(mean_off)) or float(mean_off) <= 0.0:
            return None
        blocked = diag | (d2 <= self.dup_frac * mean_off)
        masked = torch.where(blocked, torch.full_like(d2, float("inf")), d2)
        k = min(self.neighbors, m - 1)
        vals, idx = torch.topk(masked, k, dim=1, largest=False)
        keep = torch.isfinite(vals)
        if int(keep.sum()) < _MIN_ROWS:
            return None
        rows = torch.arange(m, device=d2.device).unsqueeze(1).expand(m, k)[keep]
        cols = idx[keep]
        diff = s.index_select(0, rows) - s.index_select(0, cols)
        if not bool(torch.isfinite(diff).all()):
            return None
        return (diff.t() @ diff) / float(diff.shape[0])

    @torch.no_grad()
    def _prepare(self, mat, eye):
        m = 0.5 * (mat + mat.t())
        tr = torch.diagonal(m).sum()
        if not bool(torch.isfinite(tr)) or float(tr) <= 0.0:
            return None
        m = m * (float(self.dim) / tr)
        return (1.0 - self.shrink) * m + self.shrink * eye

    @torch.no_grad()
    def _rebuild_maps(self):
        b_raw = self.pair_scatter.detach().to(device="cpu", dtype=torch.float64)
        s_raw = self.second_moment.detach().to(device="cpu", dtype=torch.float64)
        if not (bool(torch.isfinite(b_raw).all()) and bool(torch.isfinite(s_raw).all())):
            return
        eye = torch.eye(self.dim, dtype=torch.float64)
        b = self._prepare(b_raw, eye)
        s = self._prepare(s_raw, eye)
        if b is None or s is None:
            return
        try:
            evals, evecs = torch.linalg.eigh(b)
        except Exception:
            return
        if not (bool(torch.isfinite(evals).all()) and bool(torch.isfinite(evecs).all())):
            return
        beta = evals.clamp_min(self.eig_floor)
        log_mean = beta.log().mean()
        if not bool(torch.isfinite(log_mean)):
            return
        gain = (beta / log_mean.exp()).pow(0.5 * self.power).clamp(self.gain_min, self.gain_max)
        fwd = (evecs * gain.unsqueeze(0)) @ evecs.t()
        inv = (evecs * gain.reciprocal().unsqueeze(0)) @ evecs.t()
        trace_after = ((fwd @ s) * fwd).sum() / float(self.dim)
        if not bool(torch.isfinite(trace_after)) or float(trace_after) <= 0.0:
            return
        scale = trace_after.reciprocal().sqrt()
        fwd = fwd * scale
        inv = inv / scale
        if not (bool(torch.isfinite(fwd).all()) and bool(torch.isfinite(inv).all())):
            return
        round_trip = (fwd @ inv - eye).abs().max()
        if not bool(torch.isfinite(round_trip)) or float(round_trip) > _INVERSE_TOL:
            return
        self.fwd_map.copy_(fwd.to(dtype=self.fwd_map.dtype))
        self.inv_map.copy_(inv.to(dtype=self.inv_map.dtype))

    @torch.no_grad()
    def _observe(self, z):
        zf = z.detach().reshape(-1, self.dim)
        if int(zf.shape[0]) < _MIN_ROWS:
            return
        zf = zf.to(device=self.second_moment.device, dtype=torch.float32)
        if not bool(torch.isfinite(zf).all()):
            return
        n = int(zf.shape[0])
        self.second_moment.mul_(self.momentum).add_((zf.t() @ zf) / float(n),
                                                    alpha=1.0 - self.momentum)
        ps = self._pair_scatter(zf)
        if ps is not None and bool(torch.isfinite(ps).all()):
            self.pair_scatter.mul_(self.pair_momentum).add_(ps, alpha=1.0 - self.pair_momentum)
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
    return ConfusablePairScatterMetric(int(D))
