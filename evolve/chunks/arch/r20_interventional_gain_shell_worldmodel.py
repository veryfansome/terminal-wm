"""R20 arch: operation-conditioned imagination write with bounded shell calibration.

This is the R18 path-state transition world model plus a small two-scalar intervention
calibrator. Fully observed even-length streams delegate directly to R18. On the declared
odd masked-endpoint layout, alpha controls how much of the learned transition is written
and beta supplies a bounded radial correction at the later read.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_interventional_gain_shell_worldmodel"
DESCRIPTION = (
    "Champion R18 path-state model plus a 12.5K-parameter operation-conditioned "
    "intervention calibrator. On [prefix,c_m,PAD,c_r], it replaces the fixed mutation "
    "write coefficient with alpha=sigmoid(logit(w)+g(c_m,c_r,w)) and applies a bounded "
    "beta correction toward the standardized observation-radius shell at c_r. Fully "
    "observed fitness forwards are exactly the inherited champion forward."
)


class R20InterventionalGainShell(R18PathStateLatentTransition):
    def __init__(
        self,
        imag_calib_hidden=64,
        imag_shell_max=0.75,
        imag_radial_bias=-5.0,
        **params,
    ):
        super().__init__(**params)
        self.imag_shell_max = float(imag_shell_max)
        self._imag_mode = None

        # Do not move the champion/global initialization RNG stream.
        rng_state = torch.get_rng_state()
        try:
            hidden = max(8, int(imag_calib_hidden))
            self.imag_calibrator = nn.Sequential(
                nn.Linear(2 * self.d + 1, hidden),
                nn.GELU(),
                nn.Linear(hidden, 2),
            )
            nn.init.zeros_(self.imag_calibrator[-1].weight)
            nn.init.zeros_(self.imag_calibrator[-1].bias)
            nn.init.constant_(self.imag_calibrator[-1].bias[1], float(imag_radial_bias))
        finally:
            torch.set_rng_state(rng_state)

        self.supports_interventional_calibrator = True

    def imagination_command_features(self, tok_emb, types):
        """Return the exact pre-transformer command features used by the R18 transition."""
        t = types.long().clamp(0, 1)
        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t)
        x = x + self.pos_scale * self._positional(
            tok_emb.size(1), tok_emb.device, x.dtype
        ).unsqueeze(0)
        return self.in_norm(x)[:, 0::2, :]

    def imagination_coeffs(self, x_m, x_r, w_prior):
        w = w_prior.to(x_m.dtype).clamp(1e-4, 1.0 - 1e-4)
        z = self.imag_calibrator(torch.cat([x_m, x_r, w.unsqueeze(-1)], dim=-1))
        alpha = torch.sigmoid(torch.logit(w) + z[..., 0])
        beta = self.imag_shell_max * torch.sigmoid(z[..., 1])
        return alpha, beta

    def imagination_calibrate(self, p0, p1, x_m, x_r, w_prior):
        """Analytic endpoint used by the head: interpolate structured hypotheses, then
        make only a bounded radial correction. No free 768-dimensional innovation exists."""
        alpha, beta = self.imagination_coeffs(x_m, x_r, w_prior)
        pre = p0 + alpha.unsqueeze(-1) * (p1 - p0)
        shell = F.normalize(pre, dim=-1, eps=1e-6) * (float(D) ** 0.5)
        out = pre + beta.unsqueeze(-1) * (shell - pre)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out, alpha, beta

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        # Every fitness stream is even and paired, so this is the exact champion path
        # without a device synchronization or calibrator graph.
        if n_cmd == n_pair:
            return super()._transition_reads(
                x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
            )

        batch = x_cmd.size(0)
        dtype = x_cmd.dtype
        path = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = (0.90 + 0.099 * torch.sigmoid(self.logit_decay)).to(dtype)
        mem = x_cmd.new_zeros(batch, self.key_d, D)
        reads = []

        if self._imag_mode == "off":
            alpha = torch.zeros_like(w)
        elif self._imag_mode == "full":
            alpha = torch.ones_like(w)
        elif n_cmd > 1:
            a, _ = self.imagination_coeffs(
                x_cmd[:, :-1, :], x_cmd[:, 1:, :], w[:, :-1]
            )
            alpha = torch.cat([a, w[:, -1:]], dim=1)
        else:
            alpha = w

        for i in range(n_cmd):
            pi = path[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])

            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                observed = (valid_cmd[:, i] & valid_obs[:, i]).to(dtype).unsqueeze(-1)
                imagined = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(batch, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                observed = x_cmd.new_zeros(batch, 1)
                imagined = x_cmd.new_zeros(batch, 1)

            value = (1.0 - wi) * obs_i + wi * delta
            correction = (value - s_pre) * observed
            correction = correction + alpha[:, i].unsqueeze(-1) * (delta - s_pre) * imagined
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), correction.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.nan_to_num(
            torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4
        )

    def forward(self, tok_emb, types, key_pad):
        pred, hidden = super().forward(tok_emb, types, key_pad)
        length = tok_emb.size(1)
        if self._imag_mode is not None or length < 3 or length % 2 == 0:
            return pred, hidden

        batch = tok_emb.size(0)
        valid = (
            ~key_pad.bool()
            if key_pad is not None
            else torch.ones(batch, length, dtype=torch.bool, device=tok_emb.device)
        )
        n_pair = length // 2
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        masked = valid_cmd[:, :n_pair] & ~valid_obs
        if not bool(masked.any().item()):
            return pred, hidden

        count = masked.sum(dim=1)
        m_idx = masked.long().argmax(dim=1)
        n_cmd = valid_cmd.size(1)
        next_idx = (m_idx + 1).clamp_max(n_cmd - 1)
        row_ok = count == 1
        row_ok = row_ok & ((m_idx + 1) < n_cmd)
        row_ok = row_ok & valid_cmd.gather(1, next_idx.unsqueeze(1)).squeeze(1)

        prefix_cumsum = (valid_cmd[:, :n_pair] & valid_obs).long().cumsum(dim=1)
        prefix_count = torch.where(
            m_idx > 0,
            prefix_cumsum.gather(
                1, (m_idx - 1).clamp_min(0).unsqueeze(1)
            ).squeeze(1),
            torch.zeros_like(m_idx),
        )
        row_ok = row_ok & (prefix_count == m_idx)
        positions = torch.arange(length, device=tok_emb.device).unsqueeze(0)
        tail_live = (valid & (positions > (2 * m_idx + 2).unsqueeze(1))).any(dim=1)
        row_ok = row_ok & ~tail_live
        if not bool(row_ok.any().item()):
            return pred, hidden

        rows = row_ok.nonzero(as_tuple=False).squeeze(1)
        m = m_idx[rows]
        x_cmd = self.imagination_command_features(tok_emb, types)
        x_m = x_cmd[rows, m]
        x_r = x_cmd[rows, m + 1]
        w = torch.sigmoid(self.tr_mut_gate(x_m)).squeeze(-1)
        _, beta = self.imagination_coeffs(x_m, x_r, w)

        read_pos = 2 * m + 2
        pre = pred[rows, read_pos]
        shell = F.normalize(pre, dim=-1, eps=1e-6) * (float(D) ** 0.5)
        pred = pred.clone()
        pred[rows, read_pos] = pre + beta.unsqueeze(-1) * (shell - pre)
        return pred, hidden


def build(**params):
    return R20InterventionalGainShell(**params)
