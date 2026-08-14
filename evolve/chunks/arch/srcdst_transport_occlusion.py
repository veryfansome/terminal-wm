import torch
import torch.nn as nn

from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

D = 768

NAME = "srcdst_transport_occlusion"
DESCRIPTION = (
    "The r22 occlusion-trained r18 path-state world model with a two-address latent-transition "
    "memory: each step reads the slot addressed by a SOURCE key, applies the shared "
    "command-conditioned transition to that content, and writes the result by the delta rule into "
    "the slot addressed by a separate DESTINATION key, then subtracts the source content from the "
    "source slot in proportion to a learned erase gate times the angular separation of the two "
    "keys, so a step whose two keys coincide reduces exactly to the single-address r18 update. "
    "The destination projection is initialized as a copy of the source projection, making the "
    "module bit-identical to r18 at initialization; the two projections receive different "
    "gradients (one through the read, one through the write) and separate during training. "
    "Everything else - trunk, file/path delta-rule memories, FiLM views, system summary, "
    "occlusion schedule, transition_from_emb entry point and the _transition_reads signature - is "
    "inherited unchanged."
)


class SrcDstTransportOcclusion(R22ObservationOcclusionDenoising):
    def __init__(self, erase_bias=-1.0, **params):
        super().__init__(**params)
        self.tr_dst = nn.Linear(self.d, self.key_d, bias=False)
        with torch.no_grad():
            self.tr_dst.weight.copy_(self.tr_path.weight)
        self.tr_erase_gate = nn.Linear(self.d, 1)
        nn.init.zeros_(self.tr_erase_gate.weight)
        nn.init.constant_(self.tr_erase_gate.bias, float(erase_bias))

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p_src = self._unit(self.tr_path(x_cmd))
        p_dst = self._unit(self.tr_dst(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        e = torch.sigmoid(self.tr_erase_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            si = p_src[:, i, :]
            di = p_dst[:, i, :]
            s_pre = torch.bmm(si.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            d_pre = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            corr_dst = (v_i - d_pre) * active
            sep = (1.0 - (si * di).sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
            erase = e[:, i].unsqueeze(-1) * sep * active
            write = torch.bmm(di.unsqueeze(2), corr_dst.unsqueeze(1))
            clear = torch.bmm(si.unsqueeze(2), (erase * s_pre).unsqueeze(1))
            mem = decay * mem + write - clear
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return SrcDstTransportOcclusion(**params)
