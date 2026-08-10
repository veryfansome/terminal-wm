import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r21_evidencegated_imagwrite_algebraic_observer_arch"
DESCRIPTION = (
    "The r18 path-state arch + the r20 parameter-free imagination write, with the "
    "imagined correction scaled by an evidence gate g = evid/(evid+c), evid = accumulated "
    "squared mass of strictly-earlier OBSERVED memory writes. With observed history g "
    "saturates to ~1 (b-arm write preserved, measured); with none g == 0 exactly, so the "
    "history-masked forward is value-identical to the plain r18 forward (no empty-memory "
    "command-decode hallucination — the channel that lifted C1's IMAG_hist and broke its "
    "redirect family floors). Even-length fitness streams are bit-identical to the "
    "r18 forward: zero new parameters, same init RNG, same gradients, same optimizer routing."
)


class R21EvidenceGatedImagWrite(R18PathStateLatentTransition):
    def __init__(self, imag_gate_c=0.25, **params):
        super().__init__(**params)
        # c must be strictly > 0: the gate exists for the exact-zero algebra 0/(0+c) == 0.0, so
        # with no earlier observed write the imagined write is algebraically zero, not small.
        self.imag_gate_c = max(1e-6, float(imag_gate_c))

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        evid = x_cmd.new_zeros(B, 1)
        c = float(self.imag_gate_c)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (v_i - s_pre) * active
            g = evid / (evid + c)
            imagined_corr = g * (w[:, i].unsqueeze(-1) * (delta - s_pre)) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            evid = evid + observed_corr.pow(2).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R21EvidenceGatedImagWrite(**params)
