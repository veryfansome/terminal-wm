import torch
import torch.nn as nn

NAME = "r8_prev_context_translation_field"
DESCRIPTION = (
    "Learned bounded translation field: make_target = z_obs + phi(z_prev) (MLP + diagonal skip, "
    "zero-init -> exact identity at init), to_obs = pred - phi(z_prev) (exact inverse for any phi). "
    "Identity linear part keeps the fastweights obs-space memory read perfectly aligned; phi acts "
    "only on the in-batch negative geometry, separating cross-context negatives so the contrastive "
    "pressure concentrates on same-context hard ones. Learned generalization of delta/partial "
    "residual (phi = -alpha*z_prev is a special case)."
)

LEARNED = True

_HID = 192
_RHO = 8.0
_REG_LAM = 1e-3


class PrevContextTranslationField(nn.Module):

    def __init__(self, dim, hid=_HID):
        super().__init__()
        self.dim = int(dim)
        self.inp = nn.Linear(self.dim, hid)
        self.act = nn.SiLU()
        self.out = nn.Linear(hid, self.dim)
        self.diag = nn.Parameter(torch.zeros(self.dim))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self._phi_pen = None

    def _phi(self, z_prev, track_pen):
        raw = self.out(self.act(self.inp(z_prev))) + self.diag * z_prev
        norm = raw.norm(dim=-1, keepdim=True)
        phi = raw * (_RHO / (_RHO + norm))
        if track_pen:
            self._phi_pen = (phi * phi).sum(dim=-1).mean()
        return phi

    def make_target(self, z_obs, z_prev):
        return z_obs + self._phi(z_prev, track_pen=True)

    def to_obs(self, pred, z_prev):
        # Must subtract exactly the translation make_target added, so reconstruction is exact for
        # any parameter values and a collapsed field cannot distort the eval.
        return pred - self._phi(z_prev, track_pen=False)

    def reg(self):
        if self._phi_pen is None:
            return self.out.weight.sum() * 0.0
        pen = self._phi_pen
        self._phi_pen = None  # clear so a freed autograd graph is never reused across steps
        return _REG_LAM * pen


def make(dim):
    return PrevContextTranslationField(dim)
