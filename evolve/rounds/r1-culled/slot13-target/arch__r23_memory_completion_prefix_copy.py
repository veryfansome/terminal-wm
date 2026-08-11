import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r20_interventional_gain_shell_worldmodel import (
    R20InterventionalGainShell,
)

D = 768

NAME = "r23_memory_completion_prefix_copy"
DESCRIPTION = (
    "The r20 interventional-gain path-state world model plus a prefix-observation copy "
    "channel whose addressing is CONTENT-COMPLETED: the query is the command position's own "
    "current predicted content (RMS-normalized, alongside the r22-style hidden-state query) "
    "and the keys carry the raw prefix observation embeddings as well as their hidden states, "
    "so the recurrent path memory's blurry content estimate snaps onto the exact earlier "
    "observation it most resembles. Values are raw 768-d prefix observations, injected at "
    "command positions through a zero-init (D,D) readout and a match-confidence gate."
)


class R23MemoryCompletionPrefixCopy(R20InterventionalGainShell):
    def __init__(self, xattn_dim=64, xattn_gate_bias=-2.0, **params):
        super().__init__(**params)
        rng_state = torch.get_rng_state()
        try:
            self.xattn_dim = max(4, int(xattn_dim))
            self.xq_h = nn.Linear(self.d, self.xattn_dim, bias=False)
            self.xk_h = nn.Linear(self.d, self.xattn_dim, bias=False)
            self.xq_c = nn.Linear(D, self.xattn_dim, bias=False)
            self.xk_c = nn.Linear(D, self.xattn_dim, bias=False)
            nn.init.zeros_(self.xq_c.weight)
            self.x_out = nn.Linear(D, D)
            nn.init.zeros_(self.x_out.weight)
            nn.init.zeros_(self.x_out.bias)
            self.x_gate = nn.Linear(self.d + 2, 1)
            nn.init.constant_(self.x_gate.bias, float(xattn_gate_bias))
        finally:
            torch.set_rng_state(rng_state)

    @staticmethod
    def _rms_unit(x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        batch, length, _ = tok_emb.shape
        n_pair = length // 2
        if length < 3 or n_pair == 0:
            return pred, h_out
        n_cmd = (length + 1) // 2

        valid = (
            ~key_pad.bool()
            if key_pad is not None
            else torch.ones(batch, length, dtype=torch.bool, device=tok_emb.device)
        )
        valid_obs = valid[:, 1::2]
        valid_cmd = valid[:, 0::2]

        h_cmd = h_out[:, 0::2, :]
        h_obs = h_out[:, 1::2, :]
        obs_val = tok_emb[:, 1::2, :]
        pred_cmd = pred[:, 0::2, :]

        q = self.xq_h(h_cmd) + self.xq_c(self._rms_unit(pred_cmd).to(self.xq_c.weight.dtype))
        k = self.xk_h(h_obs) + self.xk_c(self._rms_unit(obs_val).to(self.xk_c.weight.dtype))
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(self.xattn_dim))

        ci = torch.arange(n_cmd, device=tok_emb.device).unsqueeze(1)
        pj = torch.arange(n_pair, device=tok_emb.device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        has_key = allowed.any(dim=2, keepdim=True)
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        retrieved = torch.bmm(attn.to(obs_val.dtype), obs_val)
        retrieved = torch.nan_to_num(retrieved, nan=0.0, posinf=1e4, neginf=-1e4)

        strength = (retrieved.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(h_cmd.dtype)
        peak = attn.amax(dim=2, keepdim=True).to(h_cmd.dtype)
        gate = torch.sigmoid(self.x_gate(torch.cat([h_cmd, strength, peak], dim=-1)))

        contrib = gate.to(retrieved.dtype) * self.x_out(
            retrieved.to(self.x_out.weight.dtype)).to(retrieved.dtype)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R23MemoryCompletionPrefixCopy(**params)
