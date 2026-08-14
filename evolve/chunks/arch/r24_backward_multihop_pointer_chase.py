import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r24_backward_multihop_pointer_chase"
DESCRIPTION = (
    "The r22 prefix-content arch plus a tied-weight MULTI-HOP BACKWARD POINTER CHASE over the "
    "strictly-earlier command prefix. Every command's raw 768-d embedding passes through one "
    "shared two-layer GELU address trunk and two sets of unit-normalized heads, giving each step "
    "a SOURCE address and a DESTINATION address in n_addr parallel key_d channels. A command's "
    "own source address is the hop-0 query; each hop scores that query against the DESTINATION "
    "addresses of all strictly-earlier steps (cosine summed over channels, learned temperature), "
    "softmax-attends, reads the raw observation paired with the matched step as that hop's "
    "candidate content, and then replaces the query with the matched step's SOURCE address under "
    "a learned gate, so hop k+1 searches for whatever put content at the location hop k came "
    "from. The per-hop candidate contents are combined by a softmax over learned halting logits "
    "that see the hidden state, a projection of the candidate content, the attention sharpness, "
    "the content RMS, its cosine to a learned null-observation direction, and a hop embedding; "
    "the combination is injected into the command-position prediction through a zero-init (D,D) "
    "readout under a sigmoid gate, so the forward is bit-identical to r22 at init. Only strictly "
    "earlier commands and their paired observations are visible at any position. The address "
    "heads form identical-shape (key_d, addr_hidden) pairs and the readout is (D,D)."
)


class R24BackwardMultihopPointerChase(R22PrefixContentXAttention):
    def __init__(
        self,
        chase_hops=6,
        addr_hidden=256,
        n_addr=2,
        content_proj=32,
        hop_emb=16,
        chase_logit_scale=8.0,
        chase_step_bias=1.0,
        chase_gate_bias=-2.0,
        **params,
    ):
        super().__init__(**params)
        self.chase_hops = max(1, int(chase_hops))
        self.addr_hidden = max(32, int(addr_hidden))
        self.n_addr = max(1, int(n_addr))
        self.content_proj_d = max(4, int(content_proj))
        self.hop_emb_d = max(2, int(hop_emb))

        self.chase_ln = nn.LayerNorm(D)
        self.chase_in = nn.Linear(D, self.addr_hidden)
        self.chase_mid = nn.Linear(self.addr_hidden, self.addr_hidden)
        self.chase_src = nn.ModuleList(
            [nn.Linear(self.addr_hidden, self.key_d, bias=False) for _ in range(self.n_addr)]
        )
        self.chase_dst = nn.ModuleList(
            [nn.Linear(self.addr_hidden, self.key_d, bias=False) for _ in range(self.n_addr)]
        )

        self.chase_logit_scale = nn.Parameter(torch.tensor(float(chase_logit_scale)))
        self.chase_null = nn.Parameter(torch.randn(D) * (1.0 / math.sqrt(D)))
        self.chase_content = nn.Linear(D, self.content_proj_d)
        self.chase_hop_emb = nn.Embedding(self.chase_hops, self.hop_emb_d)

        self.chase_step_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_step_gate.bias, float(chase_step_bias))

        self.chase_halt = nn.Linear(self.d + self.content_proj_d + 3 + self.hop_emb_d, 1)
        nn.init.zeros_(self.chase_halt.bias)

        self.chase_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_gate.bias, float(chase_gate_bias))

        self.chase_out = nn.Linear(D, D)
        nn.init.zeros_(self.chase_out.weight)
        nn.init.zeros_(self.chase_out.bias)

    def _chase_features(self, attn, content):
        sharp = attn.amax(dim=2, keepdim=True)
        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        null_cos = (
            F.normalize(content, dim=-1) * F.normalize(self.chase_null, dim=0)
        ).sum(dim=-1, keepdim=True)
        return torch.cat([sharp, rms, null_cos], dim=-1)

    def _pointer_chase(self, cmd_raw, obs_mem, h_cmd, allowed, n_cmd, n_pair):
        A = self.n_addr
        B = cmd_raw.size(0)
        K = self.key_d
        device = cmd_raw.device

        a_h = F.gelu(self.chase_mid(F.gelu(self.chase_in(self.chase_ln(cmd_raw)))))
        src = torch.stack([self._unit(m(a_h)) for m in self.chase_src], dim=0)
        dst = torch.stack([self._unit(m(a_h)) for m in self.chase_dst], dim=0)

        src_mem = src[:, :, :n_pair, :].reshape(A * B, n_pair, K)
        dst_mem = dst[:, :, :n_pair, :].reshape(A * B, n_pair, K).transpose(1, 2)
        q = src.reshape(A * B, n_cmd, K)

        allowed_rep = allowed.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, n_pair)
        has_key = allowed.any(dim=2, keepdim=True)
        scale = self.chase_logit_scale.clamp(0.5, 64.0)

        contents = []
        halts = []
        sharps = []
        for k in range(self.chase_hops):
            scores = torch.bmm(q, dst_mem).view(A, B, n_cmd, n_pair).sum(dim=0) * scale
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~allowed, neg)
            attn = torch.softmax(scores, dim=2)
            attn = torch.where(has_key, attn, torch.zeros_like(attn))

            content = torch.bmm(attn, obs_mem)
            content = torch.nan_to_num(content, nan=0.0, posinf=1e4, neginf=-1e4)
            feats = self._chase_features(attn, content)

            hop_code = self.chase_hop_emb(
                torch.tensor(k, device=device, dtype=torch.long)
            ).view(1, 1, self.hop_emb_d).expand(B, n_cmd, self.hop_emb_d)
            halt_in = torch.cat([h_cmd, self.chase_content(content), feats, hop_code], dim=-1)
            halts.append(self.chase_halt(halt_in))
            contents.append(content)
            sharps.append(feats[..., 0:1])

            if k + 1 == self.chase_hops:
                break

            step_gate = torch.sigmoid(self.chase_step_gate(torch.cat([h_cmd, feats], dim=-1)))
            attn_rep = attn.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, n_pair)
            follow = torch.bmm(attn_rep, src_mem)
            gate_rep = step_gate.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, 1)
            q = self._unit(gate_rep * follow + (1.0 - gate_rep) * q)

        halt_logits = torch.cat(halts, dim=-1)
        weights = torch.softmax(halt_logits, dim=-1)
        stacked = torch.stack(contents, dim=2)
        mixed = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        peak = torch.cat(sharps, dim=-1).amax(dim=-1, keepdim=True)
        return torch.nan_to_num(mixed, nan=0.0, posinf=1e4, neginf=-1e4), peak

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        device = tok_emb.device
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        wdt = self.chase_in.weight.dtype
        cmd_raw = tok_emb[:, 0::2, :].to(wdt)
        obs_mem = tok_emb[:, 1::2, :].to(wdt)
        h_cmd = h_out[:, 0::2, :].to(wdt)

        ci = torch.arange(n_cmd, device=device).unsqueeze(1)
        pj = torch.arange(n_pair, device=device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & (valid_obs & valid_cmd[:, :n_pair]).unsqueeze(1)

        mixed, peak = self._pointer_chase(cmd_raw, obs_mem, h_cmd, allowed, n_cmd, n_pair)

        rms = (mixed.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        null_cos = (
            F.normalize(mixed, dim=-1) * F.normalize(self.chase_null, dim=0)
        ).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.chase_gate(torch.cat([h_cmd, rms, peak, null_cos], dim=-1)))

        contrib = gate * self.chase_out(mixed)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R24BackwardMultihopPointerChase(**params)
