import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = 'r22_causal_variance_fingerprint_film'
DESCRIPTION = (
    'The r18 path-state arch plus a causal second-order system fingerprint: '
    'the shrunk diagonal variance of strictly earlier raw observations drives a '
    'bias-free, low-rank, bounded FiLM on command predictions. Individual values, '
    'their order, and their mean are discarded; the decoder is zero-initialized, '
    'empty history is structurally zero, and PAD values are excluded.'
)


class R22CausalVarianceFingerprintFilm(R18PathStateLatentTransition):
    def __init__(
        self,
        variance_rank=64,
        variance_kappa=4.0,
        variance_floor=0.25,
        variance_log_clip=2.0,
        variance_mult_cap=0.25,
        variance_shift_cap=0.25,
        variance_rms_cap=0.5,
        **params,
    ):
        super().__init__(**params)
        self.variance_rank = int(variance_rank)
        self.variance_kappa = float(variance_kappa)
        self.variance_floor = float(variance_floor)
        self.variance_log_clip = float(variance_log_clip)
        self.variance_mult_cap = float(variance_mult_cap)
        self.variance_shift_cap = float(variance_shift_cap)
        self.variance_rms_cap = float(variance_rms_cap)

        values = (
            self.variance_kappa,
            self.variance_floor,
            self.variance_log_clip,
            self.variance_mult_cap,
            self.variance_shift_cap,
            self.variance_rms_cap,
        )
        if self.variance_rank < 1:
            raise ValueError('variance_rank must be positive')
        if any((not math.isfinite(v)) or v <= 0.0 for v in values):
            raise ValueError('variance hyperparameters must be finite and positive')

        self.variance_down = nn.Linear(D, self.variance_rank, bias=False)
        self.variance_out = nn.Linear(self.variance_rank, 2 * D, bias=False)
        nn.init.zeros_(self.variance_out.weight)

    def _causal_log_variance(self, tok_emb, valid, n_cmd):
        batch = tok_emb.size(0)
        obs = tok_emb[:, 1::2, :]
        obs_valid = valid[:, 1::2]
        acc_dtype = (
            torch.float32
            if obs.dtype in (torch.float16, torch.bfloat16)
            else obs.dtype
        )
        obs_acc = obs.to(acc_dtype)
        mask = obs_valid.unsqueeze(-1).to(acc_dtype)

        count_inclusive = torch.cumsum(obs_valid.to(acc_dtype), dim=1)
        sum_inclusive = torch.cumsum(obs_acc * mask, dim=1)
        sq_inclusive = torch.cumsum(obs_acc.square() * mask, dim=1)

        zero_count = count_inclusive.new_zeros(batch, 1)
        zero_vec = sum_inclusive.new_zeros(batch, 1, D)
        count = self._pad_steps(
            torch.cat([zero_count, count_inclusive], dim=1), n_cmd
        )
        total = self._pad_steps(
            torch.cat([zero_vec, sum_inclusive], dim=1), n_cmd
        )
        total_sq = self._pad_steps(
            torch.cat([zero_vec, sq_inclusive], dim=1), n_cmd
        )

        denom = count.clamp_min(1.0).unsqueeze(-1)
        mean = total / denom
        population_variance = (total_sq / denom - mean.square()).clamp_min(0.0)

        rho = count / (count + self.variance_kappa)
        variance = (
            rho.unsqueeze(-1) * population_variance
            + (1.0 - rho).unsqueeze(-1)
        )
        variance = torch.nan_to_num(
            variance,
            nan=1.0,
            posinf=1e4,
            neginf=self.variance_floor,
        ).clamp_min(self.variance_floor)
        log_variance = torch.log(variance).clamp(
            -self.variance_log_clip, self.variance_log_clip
        )
        return log_variance, rho

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        batch, length, _ = tok_emb.shape
        if length == 0:
            return pred, h_out

        if key_pad is None:
            valid = torch.ones(
                batch,
                length,
                dtype=torch.bool,
                device=tok_emb.device,
            )
        else:
            valid = ~key_pad.bool()

        n_cmd = (length + 1) // 2
        log_variance, rho = self._causal_log_variance(tok_emb, valid, n_cmd)

        fingerprint_rms = torch.sqrt(
            log_variance.square().mean(dim=-1, keepdim=True) + 1e-12
        )
        code = log_variance / (1.0 + fingerprint_rms)
        latent = F.silu(
            self.variance_down(code.to(self.variance_down.weight.dtype))
        )
        film = self.variance_out(latent)

        multiplier = self.variance_mult_cap * torch.tanh(film[..., :D])
        shift = self.variance_shift_cap * torch.tanh(film[..., D:])
        pred_cmd = pred[:, 0::2, :]
        delta = multiplier.to(pred_cmd.dtype) * pred_cmd + shift.to(pred_cmd.dtype)

        delta_rms = torch.sqrt(delta.square().mean(dim=-1, keepdim=True) + 1e-12)
        cap = torch.clamp(self.variance_rms_cap / delta_rms, max=1.0).detach()
        delta = delta * cap

        correction = (
            rho.unsqueeze(-1).to(delta.dtype)
            * delta
            * valid[:, 0::2].unsqueeze(-1).to(delta.dtype)
        )
        correction = torch.nan_to_num(
            correction,
            nan=0.0,
            posinf=self.variance_rms_cap,
            neginf=-self.variance_rms_cap,
        )

        out = pred.clone()
        out[:, 0::2, :] = torch.nan_to_num(
            pred_cmd + correction,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        return out, h_out


def build(**params):
    return R22CausalVarianceFingerprintFilm(**params)
