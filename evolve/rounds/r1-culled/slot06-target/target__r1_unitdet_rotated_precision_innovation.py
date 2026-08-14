import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r1_unitdet_rotated_precision_innovation"
DESCRIPTION = (
    "Learned target that is an exact affine bijection of the observation space with unit "
    "determinant: t = s * B(z_obs - g(z_prev)), where B is a butterfly product of learned "
    "Givens rotations (orthogonal by construction), s = exp(v - mean(v)) with v a soft-bounded "
    "log-scale vector (sum(log s) = 0 exactly), and g is a zero-initialised low-rank MLP of the "
    "previous observation only. The induced training metric on obs-space error is Mahalanobis "
    "with M = B^T diag(s^2) B and det(M) = 1, so per-direction emphasis can be reallocated but "
    "never globally shrunk; the inverse restores the observation exactly."
)

LEARNED = True

_LAYERS = 10
_PERM_SEED = 20260809
_SCALE_BOUND = 0.47
_HIDDEN = 96
_REG_SCALE = 1e-3
_REG_INNOV = 1e-4


class _ButterflyOrthogonal(nn.Module):
    def __init__(self, d, layers, gen):
        super().__init__()
        self.d = int(d)
        self.pairs = self.d // 2
        self.layers = int(layers)
        self.angle = nn.Parameter(torch.zeros(self.layers, max(1, self.pairs)))
        for l in range(self.layers):
            p = torch.randperm(self.d, generator=gen)
            self.register_buffer(f"perm{l}", p)
            self.register_buffer(f"iperm{l}", torch.argsort(p))

    def _givens(self, x, l, inverse):
        if self.pairs == 0:
            return x
        a = self.angle[l, : self.pairs].to(device=x.device, dtype=x.dtype)
        c = torch.cos(a)
        s = torch.sin(a)
        p2 = 2 * self.pairs
        head = x[:, :p2].reshape(-1, self.pairs, 2)
        u = head[..., 0]
        v = head[..., 1]
        if inverse:
            o0 = c * u + s * v
            o1 = c * v - s * u
        else:
            o0 = c * u - s * v
            o1 = s * u + c * v
        out = torch.stack([o0, o1], dim=-1).reshape(-1, p2)
        if p2 < self.d:
            out = torch.cat([out, x[:, p2:]], dim=1)
        return out

    def forward(self, x):
        shape = x.shape
        y = x.reshape(-1, self.d)
        for l in range(self.layers):
            perm = getattr(self, f"perm{l}").to(y.device)
            y = self._givens(y.index_select(1, perm), l, False)
        return y.reshape(shape)

    def inverse(self, x):
        shape = x.shape
        y = x.reshape(-1, self.d)
        for l in range(self.layers - 1, -1, -1):
            iperm = getattr(self, f"iperm{l}").to(y.device)
            y = self._givens(y, l, True).index_select(1, iperm)
        return y.reshape(shape)


class UnitDetRotatedPrecisionInnovation(nn.Module):
    def __init__(self, d):
        super().__init__()
        rng_state = torch.get_rng_state()
        try:
            gen = torch.Generator().manual_seed(_PERM_SEED)
            self.d = int(d)
            self.rot = _ButterflyOrthogonal(self.d, _LAYERS, gen)
            self.log_scale_raw = nn.Parameter(torch.zeros(self.d))
            h = max(8, int(_HIDDEN))
            self.innov_in = nn.Linear(self.d, h)
            self.innov_out = nn.Linear(h, self.d)
            bound = 1.0 / float(self.d) ** 0.5
            w1 = (torch.rand(h, self.d, generator=gen) * 2.0 - 1.0) * bound
            with torch.no_grad():
                self.innov_in.weight.copy_(w1)
                self.innov_in.bias.zero_()
                self.innov_out.weight.zero_()
                self.innov_out.bias.zero_()
        finally:
            torch.set_rng_state(rng_state)

    def _bounded_log_scale(self):
        u = self.log_scale_raw
        return _SCALE_BOUND * torch.tanh(u / _SCALE_BOUND)

    def _scales(self, ref):
        v = self._bounded_log_scale().to(device=ref.device, dtype=ref.dtype)
        return torch.exp(v - v.mean())

    def _innovation(self, z_prev):
        w1 = self.innov_in.weight.to(device=z_prev.device, dtype=z_prev.dtype)
        b1 = self.innov_in.bias.to(device=z_prev.device, dtype=z_prev.dtype)
        w2 = self.innov_out.weight.to(device=z_prev.device, dtype=z_prev.dtype)
        b2 = self.innov_out.bias.to(device=z_prev.device, dtype=z_prev.dtype)
        return F.linear(F.gelu(F.linear(z_prev, w1, b1)), w2, b2)

    def make_target(self, z_obs, z_prev):
        resid = z_obs - self._innovation(z_prev)
        return self._scales(resid) * self.rot(resid)

    def to_obs(self, pred, z_prev):
        resid = self.rot.inverse(pred / self._scales(pred))
        return resid + self._innovation(z_prev)

    def reg(self):
        v = self._bounded_log_scale()
        return (_REG_SCALE * (v - v.mean()).pow(2).mean()
                + _REG_INNOV * self.innov_out.weight.pow(2).mean())


def make(D):
    return UnitDetRotatedPrecisionInnovation(D)
