import torch
import torch.nn as nn

NAME = "r6_gls_rotated_frame"
DESCRIPTION = (
    "A LEARNED, exactly invertible, volume-preserving target frame. The observation embedding is "
    "carried through a stack of Givens-rotation layers (a fixed random pairing of the 768 "
    "coordinates per layer, one learned angle per pair, angles initialised to zero so the frame "
    "starts as the identity) and then through a diagonal whose log-scales are bounded by a tanh "
    "and mean-centred, so the diagonal's determinant is exactly one. The inverse replays the "
    "diagonal reciprocal and the same rotations in reverse order with negated angles, in closed "
    "form, so the retrieval eval still happens in the fixed observation space. Because the map is "
    "a linear bijection, numerically equal observations stay numerically equal and every "
    "exact-duplicate equivalence class in the objective survives untouched; what changes is only "
    "the geometry the training loss measures error in. The objective already reweights squared "
    "error per coordinate by a detached inverse-error precision, which is a weighted least "
    "squares metric and is therefore axis aligned in whatever frame the target lives in. Rotating "
    "the target frame turns that axis-aligned reweighting into a full anisotropic metric in a "
    "learned basis, which is the generalized-least-squares correction for correlated residuals; "
    "the bounded det-one diagonal keeps the map non-isometric so both parameter groups move the "
    "loss under any objective."
)

LEARNED = True

_PERM_SEED = 20260614
_LAYERS = 12
_SCALE_RANGE = 0.35
_REG_WEIGHT = 1e-4


class GLSRotatedFrameTarget(nn.Module):

    def __init__(self, d, n_layers=_LAYERS, scale_range=_SCALE_RANGE,
                 reg_weight=_REG_WEIGHT, perm_seed=_PERM_SEED):
        super().__init__()
        self.frame_d = int(d)
        self.n_layers = max(1, int(n_layers))
        self.n_pairs = max(1, self.frame_d // 2)
        self.rot_width = 2 * self.n_pairs
        self.scale_range = float(scale_range)
        self.reg_weight = float(reg_weight)

        gen = torch.Generator().manual_seed(int(perm_seed))
        for li in range(self.n_layers):
            perm = torch.randperm(self.frame_d, generator=gen)
            self.register_buffer(f"perm_{li}", perm, persistent=True)
            self.register_buffer(f"unperm_{li}", torch.argsort(perm), persistent=True)

        self.angles = nn.Parameter(torch.zeros(self.n_layers, self.n_pairs))
        self.log_scale_raw = nn.Parameter(torch.zeros(self.frame_d))

    def scale_vector(self, ref):
        u = self.scale_range * torch.tanh(self.log_scale_raw)
        u = u - u.mean()
        return torch.exp(u).to(device=ref.device, dtype=ref.dtype)

    def rotate(self, x, backward):
        order = range(self.n_layers - 1, -1, -1) if backward else range(self.n_layers)
        sign = -1.0 if backward else 1.0
        w = self.rot_width
        lead = x.shape[:-1]
        for li in order:
            perm = getattr(self, f"perm_{li}").to(x.device)
            unperm = getattr(self, f"unperm_{li}").to(x.device)
            xp = x.index_select(-1, perm)
            head = xp[..., :w]
            pairs = head.reshape(*lead, self.n_pairs, 2)
            theta = (sign * self.angles[li]).to(device=x.device, dtype=x.dtype)
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            a = pairs[..., 0]
            b = pairs[..., 1]
            rot = torch.stack([cos_t * a - sin_t * b, sin_t * a + cos_t * b], dim=-1)
            rot = rot.reshape(*lead, w)
            if w < xp.shape[-1]:
                rot = torch.cat([rot, xp[..., w:]], dim=-1)
            x = rot.index_select(-1, unperm)
        return x

    def make_target(self, z_obs, z_prev):
        if z_obs.shape[-1] != self.frame_d or z_obs.numel() == 0:
            return z_obs
        return self.rotate(z_obs, False) * self.scale_vector(z_obs)

    def to_obs(self, pred, z_prev):
        if pred.shape[-1] != self.frame_d or pred.numel() == 0:
            return pred
        return self.rotate(pred / self.scale_vector(pred), True)

    def reg(self):
        return self.reg_weight * (self.log_scale_raw.pow(2).mean() + self.angles.pow(2).mean())


def make(d):
    return GLSRotatedFrameTarget(int(d))
