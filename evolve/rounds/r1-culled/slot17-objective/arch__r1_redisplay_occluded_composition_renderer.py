import torch

from evolve.chunks.arch.r22_retrieval_composition_renderer import (
    R22RetrievalCompositionRenderer,
)

NAME = "r1_redisplay_occluded_composition_renderer"
DESCRIPTION = (
    "The r22 composition renderer trained under RE-DISPLAY-TARGETED observation occlusion: "
    "each obs token is removed (zeroed + key-padded) with a probability that rises from occ_p "
    "to occ_p_redisplay in proportion to how closely it repeats an EARLIER observation of the "
    "same trajectory, so redundant re-displays are the ones withheld and the surviving evidence "
    "for a repeated content sits further back. Ramped; training-only; eval forward and the "
    "init-RNG stream are unchanged from the r22 renderer; zero new trainable parameters."
)


class R1RedisplayOccludedCompositionRenderer(R22RetrievalCompositionRenderer):
    def __init__(self, occ_p=0.10, occ_p_redisplay=0.55, occ_ramp_start=300,
                 occ_ramp_end=1000, occ_sig_frac=0.35, **params):
        super().__init__(**params)
        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.occ_p_redisplay = max(self.occ_p, min(0.9, float(occ_p_redisplay)))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.occ_sig_frac = max(1e-3, float(occ_sig_frac))
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def _occ_scale(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return 1.0
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return x * x * (3.0 - 2.0 * x)

    @torch.no_grad()
    def _redisplay_score(self, obs, valid_obs):
        B, N, Dv = obs.shape
        if N < 2:
            return obs.new_zeros(B, N)
        sq = obs.pow(2).sum(dim=-1, keepdim=True)
        dd = sq + sq.transpose(1, 2) - 2.0 * torch.bmm(obs, obs.transpose(1, 2))
        dd = dd.clamp_min(0.0) / float(Dv)
        pos = torch.arange(N, device=obs.device)
        strict = (pos.unsqueeze(1) > pos.unsqueeze(0)).unsqueeze(0)
        earlier = strict & valid_obs.unsqueeze(1) & valid_obs.unsqueeze(2)
        cnt = earlier.sum().clamp_min(1).to(dd.dtype)
        sigma = (self.occ_sig_frac * (dd * earlier.to(dd.dtype)).sum() / cnt).clamp_min(1e-3)
        dmin = dd.masked_fill(~earlier, 1e9).amin(dim=2)
        return torch.exp(-dmin / sigma)

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_step += 1
            scale = self._occ_scale()
            B, L = tok_emb.shape[0], tok_emb.shape[1]
            if scale > 0.0 and L >= 2 and B > 0:
                if key_pad is None:
                    key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                key_pad = key_pad.bool()
                n_pair = L // 2
                obs = tok_emb[:, 1:2 * n_pair:2, :]
                valid_obs = ~key_pad[:, 1:2 * n_pair:2]
                rd = self._redisplay_score(obs, valid_obs).to(obs.dtype)
                p = scale * (self.occ_p + (self.occ_p_redisplay - self.occ_p) * rd)
                drop = (torch.rand(B, n_pair, device=tok_emb.device) < p) & valid_obs
                drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                drop_full[:, 1:2 * n_pair:2] = drop
                tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                key_pad = key_pad | drop_full
        return super().forward(tok_emb, types, key_pad)


def build(**params):
    return R1RedisplayOccludedCompositionRenderer(**params)
