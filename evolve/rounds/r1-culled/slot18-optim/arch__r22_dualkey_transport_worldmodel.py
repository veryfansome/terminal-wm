import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import D
from evolve.chunks.arch.r20_contentcond_transition_imagwrite import (
    R20ContentCondTransitionImagWrite,
)

NAME = "r22_dualkey_transport_worldmodel"
DESCRIPTION = (
    "Two-address transport for the latent-transition memory. A command now emits a SOURCE "
    "address q = unit(tr_path(cmd)) and a separate DESTINATION address k = unit(tr_dest(cmd)). "
    "Observed steps keep the r18/r20/r21 single-address delta-rule write untouched. On a silent "
    "step (command present, observation absent) the content read at q is transformed by the r20 "
    "content-conditioned transition and DEPOSITED at k with a two-key delta rule "
    "k (delta - k^T mem), while the source is VACATED by -e q (q^T mem); both terms are scaled by "
    "the r21 evidence gate evid/(evid+c). tr_dest is initialised as a copy of tr_path and the "
    "vacate gate starts at sigmoid(-2), so the silent-step write starts near the partner's "
    "imagination write and separates only where transport helps."
)


class R22DualKeyTransport(R20ContentCondTransitionImagWrite):
    def __init__(self, *args, tr2_cap=0.5, imag_gate_c=0.25, move_erase_bias=-2.0, **kwargs):
        super().__init__(*args, tr2_cap=tr2_cap, **kwargs)
        self.imag_gate_c = max(1e-6, float(imag_gate_c))
        self.tr_dest = torch.nn.Linear(self.d, self.key_d, bias=False)
        self.tr_move = torch.nn.Linear(self.d, 1)
        with torch.no_grad():
            self.tr_dest.weight.copy_(self.tr_path.weight)
        torch.nn.init.zeros_(self.tr_move.weight)
        torch.nn.init.constant_(self.tr_move.bias, float(move_erase_bias))

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        q = self._unit(self.tr_path(x_cmd))
        k = self._unit(self.tr_dest(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        e = torch.sigmoid(self.tr_move(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        evid = x_cmd.new_zeros(B, 1)
        c = float(self.imag_gate_c)
        reads = []
        for i in range(n_cmd):
            qi = q[:, i, :]
            ki = k[:, i, :]
            s_src = torch.bmm(qi.unsqueeze(1), mem).squeeze(1)
            s_dst = torch.bmm(ki.unsqueeze(1), mem).squeeze(1)
            reads.append(s_src)
            delta = self._transition(s_src, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_src.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (v_i - s_src) * active
            g = evid / (evid + c)
            amount = g * wi * imag
            deposit = amount * (delta - s_dst)
            vacate = amount * e[:, i].unsqueeze(-1) * s_src
            mem = (
                decay * mem
                + torch.bmm(qi.unsqueeze(2), (observed_corr - vacate).unsqueeze(1))
                + torch.bmm(ki.unsqueeze(2), deposit.unsqueeze(1))
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            evid = evid + observed_corr.pow(2).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R22DualKeyTransport(**params)
