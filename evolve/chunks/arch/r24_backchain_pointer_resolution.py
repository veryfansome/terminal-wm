import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r24_backchain_pointer_resolution"
DESCRIPTION = (
    "The r18 path-state trunk with r22's prefix-content cross-attention, plus a K-hop BACKWARD "
    "POINTER CHASE over the command sequence. Every command's raw 768-d embedding is mapped by "
    "one shared feature layer and two learned diagonal role gates through a SINGLE shared address "
    "projection, giving a first-path address and a second-path address in one common address "
    "space. A masked softmax over strictly-earlier commands, biased toward recency, forms a "
    "step-to-step follow matrix: from a step, jump to the earlier step whose second-path address "
    "matches this step's first-path address. Each step also emits a learned absorb probability "
    "from its command features and its own observation. Starting at a command position, the chase "
    "iterates the follow matrix K times, accumulating absorbed mass; the accumulated mass over "
    "earlier steps is renormalized into a convex combination of their raw observation embeddings "
    "and injected into that command's prediction through a small identity-initialized (D,D) "
    "readout under a sigmoid gate fed the absorbed mass and the chase sharpness. One "
    "absorb-everywhere hop reduces to a single-hop prefix copy; more hops compose several address "
    "matches into one retrieval. Strictly causal: all mass moves to strictly earlier positions."
)

_NEG = -1e9


class R24BackchainPointerResolution(R22PrefixContentXAttention):
    def __init__(
        self,
        chase_hops=8,
        addr_hidden=256,
        obs_feat_d=64,
        term_hidden=96,
        chase_temp=0.5,
        chase_gate_bias=-2.0,
        role_init=0.75,
        role_noise=0.5,
        recency_init=-1.25,
        readout_init=0.05,
        **params,
    ):
        super().__init__(**params)
        self.chase_hops = max(1, int(chase_hops))
        self.addr_hidden = max(32, int(addr_hidden))
        self.obs_feat_d = max(8, int(obs_feat_d))

        self.chase_ln = nn.LayerNorm(D)
        self.chase_in = nn.Linear(D, self.addr_hidden)
        self.role_src = nn.Parameter(
            torch.randn(self.addr_hidden) * float(role_noise) + float(role_init))
        self.role_dst = nn.Parameter(
            torch.randn(self.addr_hidden) * float(role_noise) - float(role_init))
        self.addr_proj = nn.Linear(self.addr_hidden, self.key_d, bias=False)

        self.obs_feat = nn.Linear(D, self.obs_feat_d)
        th = max(16, int(term_hidden))
        self.absorb_in = nn.Linear(self.addr_hidden + self.obs_feat_d, th)
        self.absorb_out = nn.Linear(th, 1)
        nn.init.normal_(self.absorb_out.weight, std=0.01)
        nn.init.constant_(self.absorb_out.bias, 0.0)

        t0 = max(1e-2, float(chase_temp))
        self.chase_log_temp = nn.Parameter(torch.tensor(math.log(t0)))
        self.chase_recency = nn.Parameter(torch.tensor(float(recency_init)))

        self.chase_out = nn.Linear(D, D)
        with torch.no_grad():
            self.chase_out.weight.copy_(torch.eye(D) * float(readout_init))
            self.chase_out.bias.zero_()
        self.chase_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_gate.bias, float(chase_gate_bias))

    def _follow_matrix(self, cmd_raw, valid_cmd):
        a = F.gelu(self.chase_in(self.chase_ln(cmd_raw))).float()
        addr_src = self._unit(self.addr_proj(a * torch.sigmoid(self.role_src.float())))
        addr_dst = self._unit(self.addr_proj(a * torch.sigmoid(self.role_dst.float())))

        n = a.size(1)
        device = a.device
        temp = self.chase_log_temp.float().exp().clamp(0.05, 5.0)
        scores = torch.bmm(addr_src, addr_dst.transpose(1, 2)) / temp

        idx = torch.arange(n, device=device)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1).clamp_min(0).float()
        scores = scores - F.softplus(self.chase_recency.float()) * gap.unsqueeze(0)

        strict = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        allowed = strict & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allowed, _NEG)
        follow = torch.softmax(scores, dim=2)
        has_key = allowed.any(dim=2, keepdim=True)
        follow = torch.where(has_key, follow, torch.zeros_like(follow))
        return a, torch.nan_to_num(follow, nan=0.0, posinf=0.0, neginf=0.0)

    def _absorb_prob(self, a, obs_pad, obs_live):
        f = torch.cat([a, self.obs_feat(obs_pad.to(self.obs_feat.weight.dtype)).float()], dim=-1)
        logit = self.absorb_out(F.gelu(self.absorb_in(f)))
        return torch.sigmoid(logit.float()) * obs_live.unsqueeze(-1).float()

    def _backchain(self, follow, absorb):
        carry = (1.0 - absorb) * follow
        alpha = follow
        acc = alpha * absorb.transpose(1, 2)
        for _ in range(self.chase_hops - 1):
            alpha = torch.bmm(alpha, carry)
            acc = acc + alpha * absorb.transpose(1, 2)
        return torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 4:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair < 2:
            return pred, h_out

        device = tok_emb.device
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = tok_emb[:, 0::2, :]
        obs_pad = self._pad_steps(tok_emb[:, 1::2, :], n_cmd)
        obs_live = self._pad_steps(valid_obs, n_cmd) & valid_cmd

        a, follow = self._follow_matrix(cmd_raw, valid_cmd)
        absorb = self._absorb_prob(a, obs_pad, obs_live)
        acc = self._backchain(follow, absorb)

        mass = acc.sum(dim=2, keepdim=True)
        sharp = acc.amax(dim=2, keepdim=True)
        copy = torch.bmm(acc, obs_pad.float()) / mass.clamp_min(1e-4)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)

        rms = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass.clamp(0.0, 1.0), sharp.clamp(0.0, 1.0), rms], dim=-1)
        gi = torch.cat([h_out[:, 0::2, :].float(), feat], dim=-1)
        gate = torch.sigmoid(self.chase_gate(gi.to(self.chase_gate.weight.dtype))).float()

        read = self.chase_out(copy.to(self.chase_out.weight.dtype)).float()
        contrib = gate * read * valid_cmd.unsqueeze(-1).float()
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(dtype)
        return pred, h_out


def build(**params):
    return R24BackchainPointerResolution(**params)
