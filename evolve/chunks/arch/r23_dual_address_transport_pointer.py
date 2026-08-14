import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r23_dual_address_transport_pointer"
DESCRIPTION = (
    "The r22 prefix-content arch plus a DUAL-ADDRESS transport memory and a pointer-copy "
    "readout, on a trunk widened to the reference baseline's shape (d=192, ffn 4x). Each "
    "command's raw 768-d embedding is passed through one shared GELU layer and two separate "
    "unit-normalized address heads, giving a SOURCE address and a DESTINATION address per step. "
    "At every step the model reads the content currently held at the source address, writes "
    "either that read content (transport) or the step's own observation (exposure) at the "
    "destination address under a delta rule, and erases the source, all in one batched rank-2 "
    "update of a [key_d, 768] content store. The value retrieved at the source address is then "
    "used as the QUERY of a masked softmax over the strictly-earlier observation embeddings, "
    "whose output copies one raw prefix observation verbatim; a gated blend of the retrieved "
    "value and the copied observation is injected into the command-position prediction through "
    "a zero-init (D,D) readout, so the forward is bit-identical to r22 at init. The address "
    "heads form an identical-shape (key_d, hidden) pair and the readout is (D,D)."
)


class R23DualAddressTransportPointer(R22PrefixContentXAttention):
    def __init__(
        self,
        addr_hidden=256,
        ptr_dim=64,
        tp_gate_bias=-2.0,
        move_gate_bias=0.0,
        write_gate_bias=1.0,
        erase_gate_bias=-1.0,
        tp_decay=0.999,
        d=192,
        ffn_mult=4,
        **params,
    ):
        super().__init__(d=d, ffn_mult=ffn_mult, **params)
        self.addr_hidden = max(32, int(addr_hidden))
        self.ptr_dim = max(8, int(ptr_dim))
        self.tp_decay = float(tp_decay)

        self.addr_ln = nn.LayerNorm(D)
        self.addr_in = nn.Linear(D, self.addr_hidden)
        self.addr_src = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.addr_dst = nn.Linear(self.addr_hidden, self.key_d, bias=False)

        self.tp_move_gate = nn.Linear(self.d + self.addr_hidden, 1)
        self.tp_write_gate = nn.Linear(self.d + self.addr_hidden, 1)
        self.tp_erase_gate = nn.Linear(self.d + self.addr_hidden, 1)
        nn.init.constant_(self.tp_move_gate.bias, float(move_gate_bias))
        nn.init.constant_(self.tp_write_gate.bias, float(write_gate_bias))
        nn.init.constant_(self.tp_erase_gate.bias, float(erase_gate_bias))

        self.ptr_q = nn.Linear(D, self.ptr_dim, bias=False)
        self.ptr_k = nn.Linear(D, self.ptr_dim, bias=False)

        self.tp_alpha = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.tp_alpha.bias, 0.0)
        self.tp_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.tp_gate.bias, float(tp_gate_bias))

        self.tp_out = nn.Linear(D, D)
        nn.init.zeros_(self.tp_out.weight)
        nn.init.zeros_(self.tp_out.bias)

    def _transport_reads(self, a_src, a_dst, obs_raw, g_move, g_write, g_erase, obs_live,
                         n_cmd, n_pair):
        B = a_src.size(0)
        mem = a_src.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ai = a_src[:, i, :]
            di = a_dst[:, i, :]
            s_i = torch.bmm(ai.unsqueeze(1), mem).squeeze(1)
            reads.append(s_i)
            if i < n_pair:
                obs_i = obs_raw[:, i, :]
                live = obs_live[:, i].unsqueeze(-1)
            else:
                obs_i = mem.new_zeros(B, D)
                live = mem.new_zeros(B, 1)
            gm = g_move[:, i, :]
            v_i = gm * s_i + (1.0 - gm) * obs_i
            cur_d = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            w_write = (g_write[:, i, :] * live) * (v_i - cur_d)
            w_erase = -(g_erase[:, i, :] * gm * live) * s_i
            upd_a = torch.stack([di, ai], dim=2)
            upd_v = torch.stack([w_write, w_erase], dim=1)
            mem = torch.baddbmm(mem, upd_a, upd_v, beta=self.tp_decay)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        out = torch.stack(reads, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

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
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = tok_emb[:, 0::2, :]
        obs_raw = tok_emb[:, 1::2, :]
        h_cmd = h_out[:, 0::2, :].to(dtype)

        a_h = torch.nn.functional.gelu(self.addr_in(self.addr_ln(cmd_raw)))
        a_src = self._unit(self.addr_src(a_h)).to(dtype)
        a_dst = self._unit(self.addr_dst(a_h)).to(dtype)

        g_in = torch.cat([h_cmd, a_h.to(dtype)], dim=-1)
        g_move = torch.sigmoid(self.tp_move_gate(g_in)).to(dtype)
        g_write = torch.sigmoid(self.tp_write_gate(g_in)).to(dtype)
        g_erase = torch.sigmoid(self.tp_erase_gate(g_in)).to(dtype)

        obs_live = (valid_obs & valid_cmd[:, :n_pair]).to(dtype)

        read = self._transport_reads(
            a_src, a_dst, obs_raw, g_move, g_write, g_erase, obs_live, n_cmd, n_pair)

        q = self.ptr_q(read)
        k = self.ptr_k(obs_raw)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.ptr_dim)

        ci = torch.arange(n_cmd, device=device).unsqueeze(1)
        pj = torch.arange(n_pair, device=device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))
        copy = torch.bmm(attn.to(dtype), obs_raw)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True).to(dtype)

        rms_read = (read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        rms_copy = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([rms_read, rms_copy, sharp], dim=-1)
        gi = torch.cat([h_cmd, feat], dim=-1)

        alpha = torch.sigmoid(self.tp_alpha(gi))
        blend = alpha * copy + (1.0 - alpha) * read
        contrib = torch.sigmoid(self.tp_gate(gi)) * self.tp_out(blend)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R23DualAddressTransportPointer(**params)
