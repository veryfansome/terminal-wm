import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_contentcond_transition_imagwrite"
DESCRIPTION = (
    "The r18 arch with the transition operator's functional form upgraded from command-only "
    "affine to CONTENT-CONDITIONED: delta = affine(s_pre, cmd) + cap*tanh(MLP([rms(s_pre); "
    "cmd_feat])/cap), MLP zero-init (exact r18 function at init), trained by the r18 stack's "
    "own main loss + forward-model head aux — targeting the measured +0.057 operator-form "
    "headroom (brief finding 7) that every endpoint corrector works around; plus the r20 "
    "parameter-free imagination write (reused, attributed) so the net natively forwards the "
    "obs-missing endpoint layout (measurement path b). Bounded residual, unchanged interfaces, "
    "Muon/spectral-cap routing verified safe."
)


class R20ContentCondTransitionImagWrite(R18PathStateLatentTransition):
    def __init__(self, *args, tr2_hidden=192, tr2_cap=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        # New modules are constructed AFTER the entire inherited __init__ so the inherited
        # parameters draw the identical init-RNG stream; reordering breaks bit-identity at init.
        self.tr2_cap = float(tr2_cap)
        th2 = max(32, int(tr2_hidden))
        self.tr2_in = torch.nn.Linear(D + self.d, th2)
        self.tr2_out = torch.nn.Linear(th2, D)
        torch.nn.init.zeros_(self.tr2_out.weight)
        torch.nn.init.zeros_(self.tr2_out.bias)

    def _transition(self, s_pre, cmd_feat):
        base = super()._transition(s_pre, cmd_feat)
        s_n = s_pre * torch.rsqrt(s_pre.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        h = torch.nn.functional.gelu(self.tr2_in(torch.cat([s_n, cmd_feat], dim=-1)))
        res = self.tr2_cap * torch.tanh(self.tr2_out(h) / self.tr2_cap)
        return base + res

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
    return R20ContentCondTransitionImagWrite(**params)
