import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r22_prefix_content_xattention"
DESCRIPTION = (
    "The r18 path-state arch + ONE new native channel: a single-head full-rank "
    "cross-attention over the strictly-earlier prefix OBSERVATIONS (keys/queries from the "
    "causal hidden states, values = raw 768-d obs embeddings — the frozen-probe computation "
    "in-trunk), injected into the command-position predictions through a zero-init (D,D) "
    "readout and a sigmoid gate, AFTER the sysfilm styling. Bit-for-bit identical to the r18 forward at init; "
    "Muon captures the new (64,176) addressing pair, the spectral cap the new (D,D) readout."
)


class R22PrefixContentXAttention(R18PathStateLatentTransition):
    def __init__(self, xattn_dim=64, xattn_gate_bias=-2.0, **params):
        super().__init__(**params)
        # New modules are constructed AFTER the entire inherited __init__ so the inherited
        # parameters draw the identical init-RNG stream; reordering breaks bit-identity at init.
        self.xattn_dim = int(xattn_dim)
        self.xq = nn.Linear(self.d, self.xattn_dim, bias=False)
        self.xk = nn.Linear(self.d, self.xattn_dim, bias=False)
        self.x_out = nn.Linear(D, D)
        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)
        self.x_gate = nn.Linear(self.d + 1, 1)
        nn.init.constant_(self.x_gate.bias, float(xattn_gate_bias))

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=tok_emb.device)
        valid_obs = valid[:, 1::2]
        h_cmd = h_out[:, 0::2, :]
        h_obs = h_out[:, 1::2, :]
        obs_val = tok_emb[:, 1::2, :]

        q = self.xq(h_cmd)
        k = self.xk(h_obs)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.xattn_dim)

        ci = torch.arange(n_cmd, device=tok_emb.device).unsqueeze(1)
        pj = torch.arange(n_pair, device=tok_emb.device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))
        o = torch.bmm(attn.to(obs_val.dtype), obs_val)
        o = torch.nan_to_num(o, nan=0.0, posinf=1e4, neginf=-1e4)

        feat = (o.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(h_cmd.dtype)
        gate = torch.sigmoid(self.x_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate.to(o.dtype) * self.x_out(o.to(self.x_out.weight.dtype)).to(o.dtype)
        contrib = contrib * valid[:, 0::2].unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R22PrefixContentXAttention(**params)
