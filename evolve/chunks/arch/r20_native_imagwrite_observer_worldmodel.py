import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_native_imagwrite_observer_worldmodel"
DESCRIPTION = (
    "The r18 path-state transition model plus a parameter-free native imagination "
    "write for valid-command/masked-observation pairs. Fully observed training and fitness "
    "streams are unchanged; an obs-missing mutation can update memory before a later read."
)


class R20NativeImagWriteObserverWorldModel(R18PathStateLatentTransition):
    supports_native_imagwrite = True

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = (0.90 + 0.099 * torch.sigmoid(self.logit_decay)).to(dtype)
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
            value = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (value - s_pre) * active
            imagined_corr = w[:, i].unsqueeze(-1) * (delta - s_pre) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(
                mem, nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp(-1e4, 1e4)
        return torch.nan_to_num(
            torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4
        )


def build(**params):
    return R20NativeImagWriteObserverWorldModel(**params)
