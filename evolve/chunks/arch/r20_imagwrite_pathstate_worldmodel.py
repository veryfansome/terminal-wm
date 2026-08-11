import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_imagwrite_pathstate_worldmodel"
DESCRIPTION = (
    "The r18 path-state latent-transition arch + a PARAMETER-FREE imagination write: a "
    "pair with a valid command but a masked (key_pad) observation writes w_i*(f(s_pre,cmd)-"
    "s_pre) into its path slot, so the net natively forwards obs-missing mutation suffixes "
    "(measurement path b) and composes the read through the r18 stack's own trained machinery. "
    "Identically dead on even-length fully-observed streams: fitness training/eval, "
    "state_dict, init RNG, gradients and optimizer routing are bit-identical to the r18 forward "
    "— the imagination is an emergent eval-time capability of the already-trained operator, "
    "not a new training pressure."
)


class R20ImagWritePathState(R18PathStateLatentTransition):
    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
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
            corr = (v_i - s_pre) * active \
                + (w[:, i].unsqueeze(-1) * (delta - s_pre)) * imag
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R20ImagWritePathState(**params)
