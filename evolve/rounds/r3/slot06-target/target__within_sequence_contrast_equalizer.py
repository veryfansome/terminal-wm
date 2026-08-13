import math

import torch
import torch.nn as nn

NAME = "within_sequence_contrast_equalizer"
DESCRIPTION = (
    "Exactly-invertible spectral reweighting of the prediction target, estimated online from the "
    "training batches. Target rows are cut into trajectories at the all-zero z_prev that marks a "
    "sequence start, exact-duplicate observations are down-weighted by their duplicate count via a "
    "quantized random projection, and two running second moments are accumulated: the total moment "
    "and the moment of the deviations from each trajectory's own mean. A rank-k orthonormal basis "
    "is carried by orthogonal iteration on the total moment; each direction receives a "
    "multiplicative gain equal to its within-trajectory variance fraction raised to a capped, "
    "log-mean-zero power, so directions that vary inside a trajectory are amplified relative to "
    "directions that only separate trajectories. make_target applies the gains inside that subspace "
    "and leaves the orthogonal complement untouched; to_obs applies the reciprocal gains, which is "
    "an exact inverse for any orthonormal basis and any positive gains. The map starts as the "
    "identity, ramps in, and freezes after a warmup; it has no trainable parameters and never reads "
    "commands, paths or step order."
)

LEARNED = True

_EPS = 1e-8


class WithinSequenceContrastEqualizer(nn.Module):
    def __init__(self, d=768, rank=32, alpha=1.0, gain_cap=3.0, warmup_updates=600,
                 ramp_updates=400, ema_decay=0.98, max_rows=1024, hash_dim=8,
                 hash_eps=1e-3, refresh_every=4, init_seed=20260812):
        super().__init__()
        self.d = int(d)
        self.rank = max(1, min(int(rank), self.d))
        self.alpha = float(alpha)
        self.log_gain_cap = math.log(max(1.0 + 1e-6, float(abs(gain_cap))))
        self.warmup_updates = int(warmup_updates)
        self.ramp_updates = max(1, int(ramp_updates))
        self.ema_decay = float(ema_decay)
        self.max_rows = max(8, int(max_rows))
        self.hash_eps = float(hash_eps)
        self.refresh_every = max(1, int(refresh_every))

        gen = torch.Generator().manual_seed(int(init_seed))
        seed_mat = torch.randn(self.d, self.rank, generator=gen)
        q, _ = torch.linalg.qr(seed_mat)
        self.register_buffer("basis", q.t().contiguous())
        self.register_buffer("gain", torch.ones(self.rank))
        self.register_buffer("total_moment", torch.zeros(self.d, self.d))
        self.register_buffer("within_moment", torch.zeros(self.d, self.d))
        self.register_buffer("hash_proj", torch.randn(self.d, max(1, int(hash_dim)),
                                                      generator=gen))
        self.register_buffer("n_updates", torch.zeros((), dtype=torch.long))

    def _ramp(self):
        x = float(int(self.n_updates)) / float(self.ramp_updates)
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    def _duplicate_weights(self, z):
        try:
            proj = z @ self.hash_proj.to(device=z.device, dtype=z.dtype)
            key = torch.round(proj / self.hash_eps)
            _, inverse, counts = torch.unique(key, dim=0, return_inverse=True,
                                              return_counts=True)
            w = counts.to(z.dtype)[inverse].reciprocal()
        except Exception:
            w = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        return w

    @staticmethod
    def _segment_ids(z_prev):
        starts = z_prev.abs().sum(dim=1) == 0
        if not bool(starts.any()):
            return None
        seg = torch.cumsum(starts.long(), dim=0) - 1
        return seg.clamp_min(0)

    def _orthonormalize(self, rows):
        out = rows.new_zeros(rows.shape)
        count = 0
        for i in range(rows.shape[0]):
            v = rows[i]
            for _ in range(2):
                if count > 0:
                    p = out[:count]
                    v = v - p.t() @ (p @ v)
            nrm = v.norm()
            if not bool(torch.isfinite(nrm)) or float(nrm) < 1e-6:
                return None
            out[count] = v / nrm
            count += 1
        return out

    @torch.no_grad()
    def _accumulate(self, z_obs, z_prev):
        n = z_obs.shape[0]
        if n < 8 or z_obs.shape[-1] != self.d:
            return
        z = z_obs.detach().float()
        prev = z_prev.detach().float() if z_prev is not None else None
        if prev is None or prev.shape != z.shape:
            return
        if not bool(torch.isfinite(z).all()):
            return
        if n > self.max_rows:
            z = z[: self.max_rows]
            prev = prev[: self.max_rows]
            n = self.max_rows

        seg = self._segment_ids(prev)
        w = self._duplicate_weights(z)
        w = w / w.sum().clamp_min(_EPS)
        zw = z * w.unsqueeze(1)
        total = z.t() @ zw

        if seg is None:
            centered = z - (w.unsqueeze(1) * z).sum(dim=0, keepdim=True)
        else:
            n_seg = int(seg.max()) + 1
            seg_w = z.new_zeros(n_seg).index_add_(0, seg, w)
            seg_sum = z.new_zeros(n_seg, self.d).index_add_(0, seg, zw)
            seg_mean = seg_sum / seg_w.clamp_min(_EPS).unsqueeze(1)
            centered = z - seg_mean[seg]
        within = centered.t() @ (centered * w.unsqueeze(1))

        if not (bool(torch.isfinite(total).all()) and bool(torch.isfinite(within).all())):
            return

        rho = self.ema_decay
        self.total_moment.mul_(rho).add_(total, alpha=1.0 - rho)
        self.within_moment.mul_(rho).add_(within, alpha=1.0 - rho)
        self.n_updates += 1

        if int(self.n_updates) % self.refresh_every != 0:
            return

        iterated = self.basis @ self.total_moment
        if not bool(torch.isfinite(iterated).all()):
            return
        new_basis = self._orthonormalize(iterated)
        if new_basis is not None:
            self.basis.copy_(new_basis)

        t_energy = ((self.basis @ self.total_moment) * self.basis).sum(dim=1).clamp_min(0.0)
        w_energy = ((self.basis @ self.within_moment) * self.basis).sum(dim=1).clamp_min(0.0)
        scale = t_energy.mean().clamp_min(_EPS)
        ratio = (w_energy + _EPS * scale) / (t_energy + _EPS * scale)
        log_ratio = ratio.clamp_min(1e-6).log()
        u = 0.5 * self.alpha * self._ramp() * (log_ratio - log_ratio.mean())
        u = u.clamp(-self.log_gain_cap, self.log_gain_cap)
        new_gain = u.exp()
        if bool(torch.isfinite(new_gain).all()):
            self.gain.copy_(new_gain)

    def _rescale(self, x, gain):
        shape = x.shape
        z = x.reshape(-1, self.d)
        basis = self.basis.to(device=z.device, dtype=z.dtype)
        g = gain.to(device=z.device, dtype=z.dtype)
        coeff = z @ basis.t()
        out = z + ((g - 1.0) * coeff) @ basis
        return out.reshape(shape)

    def make_target(self, z_obs, z_prev):
        if self.training and int(self.n_updates) < self.warmup_updates:
            self._accumulate(z_obs, z_prev)
        return self._rescale(z_obs, self.gain)

    def to_obs(self, pred, z_prev):
        return self._rescale(pred, self.gain.reciprocal())

    def reg(self):
        return 0.0


def make(D):
    return WithinSequenceContrastEqualizer(d=int(D))
