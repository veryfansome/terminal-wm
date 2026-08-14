import torch

from evolve.chunks.arch.r20_dualkey_contentgated_transition_worldmodel import (
    R20DualKeyContentGatedWorldModel,
)

NAME = "dualkey_contentgated_occlusion_denoising"
DESCRIPTION = (
    "The dual-key content-gated latent-transition world model (split read/write addressing, "
    "learned content-presence gate, delta-rule path memory) trained under ramped stochastic "
    "observation occlusion: in training mode each observation token is independently zeroed and "
    "marked as padding with a probability that smoothsteps from 0 up to occ_p between "
    "occ_ramp_start and occ_ramp_end optimiser steps. An occluded observation is invalid for the "
    "presence gate, so it contributes no delta-rule write, no decay, no system-summary mass and "
    "no previous-observation read source, which makes it structurally identical to a step whose "
    "result was never seen. Targets are untouched — occlusion only removes input history. Eval "
    "forward is bit-for-bit identical to the unoccluded forward; zero new trainable parameters."
)


class DualKeyContentGatedOcclusionDenoising(R20DualKeyContentGatedWorldModel):
    def __init__(self, occ_p=0.12, occ_ramp_start=300, occ_ramp_end=1000, **params):
        super().__init__(**params)
        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def occlusion_prob(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return self.occ_p
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return self.occ_p * (x * x * (3.0 - 2.0 * x))

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_step += 1
            p = self.occlusion_prob()
            B, L = tok_emb.shape[0], tok_emb.shape[1]
            if p > 0.0 and L >= 2 and B > 0:
                if key_pad is None:
                    key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                key_pad = key_pad.bool()
                n_pair = L // 2
                drop = torch.rand(B, n_pair, device=tok_emb.device) < p
                drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                drop_full[:, 1:2 * n_pair:2] = drop
                tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                key_pad = key_pad | drop_full
        return super().forward(tok_emb, types, key_pad)


def build(**params):
    return DualKeyContentGatedOcclusionDenoising(**params)
