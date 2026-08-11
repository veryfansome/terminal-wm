import math

import torch
import torch.nn as nn

NAME = "r23_sibling_contrast_metric"
DESCRIPTION = (
    "Learned-in-the-statistical-sense target metric: an exactly invertible rank-k diagonal "
    "gain in the top principal directions of the standardized observation cloud, where each "
    "direction's gain is the square root of the share of CONFUSABLE-PAIR difference variance "
    "that direction carries (ring-weighted, duplicate-suppressed, EMA-tracked). Training error "
    "is thereby measured in the metric the retrieval pick is actually decided in; the inverse "
    "restores the fixed observation space exactly."
)

LEARNED = True

_EPS = 1e-6


class SiblingContrastMetric(nn.Module):
    def __init__(self, D, k=16, rows=192, update_every=4, gain_ema=0.95,
                 basis_ema=0.90, ramp_updates=150, gain_exponent=1.0,
                 gain_min=0.5, gain_max=2.0, neighbours=8, dup_frac=0.02):
        super().__init__()
        self.D = int(D)
        self.k = max(1, min(int(k), int(D)))
        self.rows = max(16, int(rows))
        self.update_every = max(1, int(update_every))
        self.gain_ema = float(gain_ema)
        self.basis_ema = float(basis_ema)
        self.ramp_updates = max(1, int(ramp_updates))
        self.half_exp = 0.5 * float(gain_exponent)
        self.log_min = math.log(float(gain_min))
        self.log_max = math.log(float(gain_max))
        self.neighbours = max(1, int(neighbours))
        self.dup_frac = max(0.0, float(dup_frac))

        self.register_buffer("basis", torch.zeros(self.D, self.k))
        self.register_buffer("pending", torch.zeros(self.D, self.k))
        self.register_buffer("dirs", torch.zeros(self.D, self.k))
        self.register_buffer("log_gain", torch.zeros(self.k))
        self.register_buffer("log_gain_eff", torch.zeros(self.k))
        self.register_buffer("active_flag", torch.zeros((), dtype=torch.long))

        self._calls = 0
        self._updates = 0
        self._active = False

    def _is_active(self):
        if self._active:
            return True
        if int(self.active_flag.item()) != 0:
            self._active = True
        return self._active

    def _orth(self, mat):
        host = mat.detach().to("cpu", torch.float32).clone()
        host[: self.k, :] += 1e-4 * torch.eye(self.k)
        host = torch.nan_to_num(host, nan=0.0, posinf=1.0, neginf=-1.0)
        q, _ = torch.linalg.qr(host)
        return q.to(device=mat.device, dtype=mat.dtype)

    def _map(self, x, log_g):
        u = self.dirs.to(x.dtype)
        coef = x @ u
        return x + (coef * (torch.exp(log_g.to(x.dtype)) - 1.0)) @ u.t()

    @torch.no_grad()
    def _update(self, z_obs):
        n = int(z_obs.shape[0])
        if n < 16 or int(z_obs.shape[1]) != self.D:
            return
        m = min(n, self.rows)
        z = torch.nan_to_num(z_obs[:m].detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
        centered = z - z.mean(0, keepdim=True)

        if self._active:
            self.dirs.copy_(self.pending)
            seed = self.basis
        else:
            seed = centered[: self.k].t().contiguous()
            if seed.shape[1] < self.k:
                seed = torch.cat(
                    [seed, seed.new_zeros(self.D, self.k - seed.shape[1])], dim=1)

        q_basis = self._orth(seed)
        powered = centered.t() @ (centered @ q_basis) / float(m)
        powered = torch.nan_to_num(powered, nan=0.0, posinf=1e4, neginf=-1e4)
        powered = powered / powered.norm().clamp_min(_EPS)
        if self._active:
            new_basis = self.basis_ema * self.basis + (1.0 - self.basis_ema) * powered
        else:
            new_basis = powered
        self.basis.copy_(new_basis)
        # dirs must stay orthonormal: to_obs inverts the map as I + U(1/g - 1)U^T, which is the
        # exact inverse of I + U(g - 1)U^T only when U^T U = I.
        advanced = torch.nan_to_num(self._orth(new_basis), nan=0.0, posinf=0.0, neginf=0.0)
        self.pending.copy_(advanced)
        if not self._active:
            self.dirs.copy_(advanced)
        dirs = self.dirs

        sq = (z * z).sum(1)
        dist = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (z @ z.t())).clamp_min(0.0) / float(self.D)
        eye = torch.eye(m, dtype=torch.bool, device=z.device)
        dist = dist.masked_fill(eye, 0.0)
        rank = max(1, min(self.neighbours, m - 1))
        mean_off = dist.sum() / float(max(1, m * (m - 1)))
        dup_at = (self.dup_frac * mean_off).clamp_min(1e-9)
        far = torch.full_like(dist, 1e9)
        cand = torch.where(eye | (dist < dup_at), far, dist)
        near = torch.topk(cand, rank, dim=1, largest=False)
        ring = torch.zeros_like(dist).scatter_(
            1, near.indices, (near.values < 1e8).to(dist.dtype))
        ring = torch.maximum(ring, ring.t()).masked_fill(eye, 0.0)
        mass = ring.sum().clamp_min(_EPS)
        t_ring = (ring * dist).sum() / mass

        pk = z @ dirs
        row_mass = ring.sum(1)
        per_dir = 2.0 * ((row_mass.unsqueeze(1) * pk * pk).sum(0) - ((ring @ pk) * pk).sum(0)) / mass
        share = (per_dir.clamp_min(0.0) / t_ring.clamp_min(_EPS)).clamp_min(1e-4)
        fresh = (self.half_exp * torch.log(share)).clamp(self.log_min, self.log_max)
        fresh = torch.where(torch.isfinite(fresh), fresh, torch.zeros_like(fresh))

        trust = (mass / (mass + 1e-2)).clamp(0.0, 1.0)
        if self._active:
            merged = self.log_gain + (1.0 - self.gain_ema) * trust * (fresh - self.log_gain)
        else:
            merged = trust * fresh
        merged = torch.where(torch.isfinite(merged), merged, torch.zeros_like(merged))

        self.log_gain.copy_(merged)
        self._updates += 1
        ramp = min(1.0, self._updates / float(self.ramp_updates))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        self.log_gain_eff.copy_(merged * ramp)
        self._active = True
        self.active_flag.fill_(1)

    def make_target(self, z_obs, z_prev):
        if self.training:
            self._calls += 1
            if (self._calls % self.update_every == 0) or not self._active:
                self._update(z_obs)
        if not self._is_active():
            return z_obs
        return self._map(z_obs, self.log_gain_eff)

    def to_obs(self, pred, z_prev):
        if not self._is_active():
            return pred
        return self._map(pred, -self.log_gain_eff)

    def reg(self):
        return 0.0


def make(D, **params):
    return SiblingContrastMetric(D, **params)
