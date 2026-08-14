import torch
import torch.nn as nn

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r5_dual_key_transport_memory"
DESCRIPTION = (
    "The r22 arch with the latent-transition memory given TWO addressing projections instead of "
    "one: a read address and a write address, both linear functionals of the same command "
    "feature. At each step the memory is read at the read address to give the pre-state, that "
    "pre-state is passed through the shared command-conditioned affine transition operator, and "
    "the result is written by the delta rule at the WRITE address, correcting against the content "
    "currently held there. When the two addresses coincide the step is the r22 read-modify-write "
    "in place; when they differ the step carries content from one slot to another in a single "
    "memory operation, so a chain of such steps composes without needing one network layer per "
    "link. The returned per-step read is still the pre-write content at the read address, so the "
    "prediction at a step never sees that step's own observation. The write projection is "
    "initialized as a copy of the read projection, which makes the module exactly the r22 "
    "function at initialization, and the two diverge under training."
)


class R5DualKeyTransportMemory(R22PrefixContentXAttention):
    def __init__(self, **params):
        super().__init__(**params)
        self.tr_path_dst = nn.Linear(self.d, self.key_d, bias=False)
        with torch.no_grad():
            self.tr_path_dst.weight.copy_(self.tr_path.weight)

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p_read = self._unit(self.tr_path(x_cmd))
        p_write = self._unit(self.tr_path_dst(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ri = p_read[:, i, :]
            wi_key = p_write[:, i, :]
            s_pre = torch.bmm(ri.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            d_pre = torch.bmm(wi_key.unsqueeze(1), mem).squeeze(1)
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
            corr = (v_i - d_pre) * active
            mem = decay * mem + torch.bmm(wi_key.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R5DualKeyTransportMemory(**params)
