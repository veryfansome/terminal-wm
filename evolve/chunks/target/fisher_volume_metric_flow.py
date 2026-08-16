import torch
import torch.nn as nn

NAME = "fisher_volume_metric_flow"
DESCRIPTION = (
    "Trains the model in a linearly remapped observation space whose map is LEARNED BY GRADIENT "
    "from the training loss and constrained to have determinant exactly one, and reads the "
    "prediction back through the exact algebraic inverse. The map is W = diag(exp(g)) (I + N_low) "
    "(I + N_up): g is a per-coordinate log-gain that is tanh-capped and then mean-centred, so "
    "diag(exp(g)) has determinant exp(sum(g)) = 1 by construction; N_low and N_up are strictly "
    "lower- and strictly upper-triangular trainable matrices whose Frobenius norms are "
    "renormalized down to a fixed cap inside the forward, so both factors are unit-triangular "
    "(determinant 1, always invertible) with a bounded condition number. The product therefore "
    "preserves volume exactly, which removes the two degenerate solutions a learned target has: "
    "it cannot shrink toward a constant and it cannot inflate to make a contrastive loss trivial. "
    "Every parameter starts at zero, so the map starts as the exact identity and the training "
    "loss alone decides how far it tilts. Two opposing pressures shape it. The main loss pulls W "
    "toward shrinking the directions in which the model's residual is large, which under a fixed "
    "determinant is a whitening of the residual covariance. The module's reg() pushes the other "
    "way with a Fisher-style numerator: it caches, per step, a band-pass CONFUSABILITY ring over "
    "the batch's target-target squared distances (near-identical targets, the repeated-file case, "
    "get about zero weight; close-but-distinct targets get the mass) and returns minus a small "
    "multiple of the log of the ring-weighted mean squared distance between those same pairs "
    "MEASURED IN THE MAPPED SPACE. Because the map is linear the mapped pair differences are just "
    "differences of the targets already computed, so the term is free, the ring weights are "
    "detached, and the gradient reaches only the map. The equilibrium of the two pressures under "
    "the unit-determinant constraint is a discriminant metric: stretch the directions along which "
    "mutually confusable contents actually differ, shrink the directions the model cannot resolve "
    "anyway. make_target and to_obs ignore z_prev entirely, so the read-back behaves identically "
    "whether or not an observation precedes the position being read, and the inverse is assembled "
    "from the same three factors by two unit-triangular solves and a reciprocal gain."
)

LEARNED = True

_EPS = 1e-8


class FisherVolumeMetricFlow(nn.Module):
    def __init__(self, dim, gain_cap=0.35, shear_cap=0.35, spread_beta=0.05,
                 ring_rows=192, lam_frac=0.5, dup_frac=0.05, param_clip=4.0,
                 min_rows=8):
        super().__init__()
        self.dim = int(dim)
        self.gain_cap = float(gain_cap)
        self.shear_cap = float(shear_cap)
        self.spread_beta = float(spread_beta)
        self.ring_rows = max(8, int(ring_rows))
        self.lam_frac = float(lam_frac)
        self.dup_frac = float(dup_frac)
        self.param_clip = float(param_clip)
        self.min_rows = max(4, int(min_rows))

        self.log_gain = nn.Parameter(torch.zeros(self.dim))
        self.shear_low = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.shear_up = nn.Parameter(torch.zeros(self.dim, self.dim))

        ones = torch.ones(self.dim, self.dim)
        self.register_buffer("keep_low", torch.tril(ones, -1))
        self.register_buffer("keep_up", torch.triu(ones, 1))

        self._pair_term = None

    def _gains(self):
        s = self.log_gain.clamp(-50.0, 50.0)
        g = self.gain_cap * torch.tanh(s / self.gain_cap)
        return g - g.mean()

    def _shear(self, raw, keep):
        n = raw.clamp(-self.param_clip, self.param_clip) * keep
        f = n.pow(2).sum().clamp_min(_EPS).sqrt()
        return n * (self.shear_cap / torch.clamp(f, min=self.shear_cap))

    def _factors(self, device, dtype):
        nl = self._shear(self.shear_low, self.keep_low).to(device=device, dtype=dtype)
        nu = self._shear(self.shear_up, self.keep_up).to(device=device, dtype=dtype)
        g = self._gains().to(device=device, dtype=dtype)
        return nl, nu, g

    def _ring_spread(self, z, t):
        n = int(z.shape[0])
        if self.spread_beta <= 0.0 or n < self.min_rows:
            return None
        m = min(n, self.ring_rows)
        zz = z[:m]
        tt = t[:m]
        with torch.no_grad():
            zs = (zz * zz).sum(dim=1)
            zd = (zs.unsqueeze(1) + zs.unsqueeze(0) - 2.0 * (zz @ zz.t())).clamp_min(0.0)
            zd = zd / float(self.dim)
            off = ~torch.eye(m, dtype=torch.bool, device=zz.device)
            mean_off = zd[off].mean().clamp_min(1e-6)
            lam = (self.lam_frac * mean_off).clamp_min(1e-6)
            dup = (self.dup_frac * mean_off).clamp_min(1e-6)
            ring = torch.exp(-zd / lam) * (1.0 - torch.exp(-zd / dup))
            ring = ring * off.to(ring.dtype)
            ring = ring / ring.sum().clamp_min(1e-12)
            if not bool(torch.isfinite(ring).all()):
                return None
        ts = (tt * tt).sum(dim=1)
        td = (ts.unsqueeze(1) + ts.unsqueeze(0) - 2.0 * (tt @ tt.t())).clamp_min(0.0)
        td = td / float(self.dim)
        spread = (ring * td).sum().clamp_min(1e-8)
        return -self.spread_beta * torch.log(spread)

    def make_target(self, z_obs, z_prev):
        if not torch.is_tensor(z_obs) or z_obs.dim() < 1 or int(z_obs.shape[-1]) != self.dim:
            return z_obs
        z = z_obs.detach()
        shape = z.shape
        flat = z.reshape(-1, self.dim)
        nl, nu, g = self._factors(flat.device, flat.dtype)
        y = flat + flat @ nu.t()
        y = y + y @ nl.t()
        y = y * torch.exp(g).unsqueeze(0)
        if self.training:
            self._pair_term = self._ring_spread(flat, y)
        return y.reshape(shape)

    def to_obs(self, pred, z_prev):
        if not torch.is_tensor(pred) or pred.dim() < 1 or int(pred.shape[-1]) != self.dim:
            return pred
        shape = pred.shape
        flat = pred.reshape(-1, self.dim)
        nl, nu, g = self._factors(flat.device, flat.dtype)
        eye = torch.eye(self.dim, device=flat.device, dtype=flat.dtype)
        low_inv = torch.linalg.solve_triangular(eye + nl, eye, upper=False, unitriangular=True)
        up_inv = torch.linalg.solve_triangular(eye + nu, eye, upper=True, unitriangular=True)
        back = (torch.exp(-g).unsqueeze(1) * low_inv.t()) @ up_inv.t()
        return (flat @ back).reshape(shape)

    def reg(self):
        term = self._pair_term
        self._pair_term = None
        if term is None:
            return 0.0
        return term


def make(D):
    return FisherVolumeMetricFlow(int(D))
