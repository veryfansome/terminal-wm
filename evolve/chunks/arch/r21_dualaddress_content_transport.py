import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

D = 768

NAME = "r21_dualaddress_content_transport"
DESCRIPTION = (
    "The occlusion-denoised r18 trunk with its single-address latent-transition memory replaced "
    "by a DUAL-ADDRESS CONTENT TRANSPORTER. Each step derives a read address and a write address "
    "from one shared nonlinear map of (command feature, causal trunk state, raw-command address "
    "hint), reads the content currently held at the read address, and writes to the write address "
    "a carry-gated blend of that retrieved content (optionally edited by the shared command-"
    "conditioned transition operator) and the step's own observation, then erases a learned "
    "fraction of the retrieved content at the read address, scaled by how far the two addresses "
    "differ so a single-address step is a no-op. A step whose observation carries no information "
    "therefore RELOCATES what it retrieved instead of overwriting a slot with an empty "
    "observation, so content identity survives an arbitrarily long chain of renames. The write is "
    "active on command validity rather than observation validity, and a missing observation forces "
    "the carry branch, which makes the parent's ramped observation occlusion train the transport "
    "path directly. The transporter owns its own slow memory decay; the (D,D) transition readout "
    "is small-random rather than zero so every new parameter has a live gradient path to the "
    "scored command prediction from the first step."
)


class R21DualAddressContentTransport(R22ObservationOcclusionDenoising):
    def __init__(
        self,
        addr_hidden=192,
        carry_bias=0.0,
        transport_erase_bias=-1.0,
        mv_decay_init=0.998,
        tr_read_init=1e-3,
        **params,
    ):
        super().__init__(**params)
        hidden = max(32, int(addr_hidden))
        self.addr_hidden = hidden
        feat_d = 2 * self.d + self.key_d

        self.mv_addr_in = nn.Linear(feat_d, hidden)
        self.mv_read = nn.Linear(hidden, self.key_d, bias=False)
        self.mv_write = nn.Linear(hidden, self.key_d, bias=False)
        self.mv_carry = nn.Linear(feat_d + self.d, 1)
        self.mv_erase = nn.Linear(feat_d, 1)
        nn.init.constant_(self.mv_carry.bias, float(carry_bias))
        nn.init.constant_(self.mv_erase.bias, float(transport_erase_bias))

        decay = min(0.9985, max(0.9015, float(mv_decay_init)))
        frac = (decay - 0.90) / 0.099
        self.logit_mv_decay = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac))))

        std = max(1e-6, float(tr_read_init))
        nn.init.normal_(self.tr_read.weight, mean=0.0, std=std)
        nn.init.zeros_(self.tr_read.bias)

    def _occlude(self, tok_emb, key_pad):
        if not self.training:
            return tok_emb, key_pad
        self.occ_step += 1
        p = self._occ_prob()
        B, L = tok_emb.shape[0], tok_emb.shape[1]
        if p <= 0.0 or L < 2 or B == 0:
            return tok_emb, key_pad
        if key_pad is None:
            key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
        key_pad = key_pad.bool()
        n_pair = L // 2
        drop = torch.rand(B, n_pair, device=tok_emb.device) < p
        drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
        drop_full[:, 1:2 * n_pair:2] = drop
        tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
        key_pad = key_pad | drop_full
        return tok_emb, key_pad

    def _transport_reads(self, x_cmd, h_cmd, h_obs, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype

        hint = self._unit(self.tr_path(x_cmd))
        feat = torch.cat([x_cmd, h_cmd, hint], dim=-1)
        addr_h = torch.nn.functional.gelu(self.mv_addr_in(feat))
        a_read = self._unit(self.mv_read(addr_h))
        a_write = self._unit(self.mv_write(addr_h))

        mut = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        align = (a_read * a_write).sum(dim=-1).clamp(-1.0, 1.0)
        erase = torch.sigmoid(self.mv_erase(feat)).squeeze(-1) * (1.0 - align * align)

        if n_pair:
            carry_in = torch.cat([feat[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            carry_pair = torch.sigmoid(self.mv_carry(carry_in)).squeeze(-1)
        else:
            carry_pair = x_cmd.new_zeros(B, 0)

        carry = self._pad_steps(carry_pair, n_cmd)
        seen = self._pad_steps(valid_obs.to(dtype), n_cmd)
        obs_pad = self._pad_steps(obs_tok.to(dtype), n_cmd)
        cmd_ok = valid_cmd.to(dtype)

        decay = (0.90 + 0.099 * torch.sigmoid(self.logit_mv_decay)).to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ri = a_read[:, i, :]
            wi = a_write[:, i, :]
            ports = torch.bmm(torch.stack([ri, wi], dim=1), mem)
            s_src = ports[:, 0, :]
            s_dst = ports[:, 1, :]
            reads.append(s_src)

            edited = self._transition(s_src, x_cmd[:, i, :])
            mu = mut[:, i].unsqueeze(-1)
            carried = (1.0 - mu) * s_src + mu * edited

            seen_i = seen[:, i].unsqueeze(-1)
            c_i = carry[:, i].unsqueeze(-1) * seen_i + (1.0 - seen_i)
            v_i = (1.0 - c_i) * obs_pad[:, i, :] + c_i * carried

            active = cmd_ok[:, i].unsqueeze(-1)
            corr_w = (v_i - s_dst) * active
            corr_e = (-(erase[:, i].unsqueeze(-1) * c_i) * s_src) * active
            addr_pair = torch.stack([wi, ri], dim=2)
            corr_pair = torch.stack([corr_w, corr_e], dim=1)
            mem = decay * mem + torch.bmm(addr_pair, corr_pair)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        tok_emb, key_pad = self._occlude(tok_emb, key_pad)

        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * valid_obs.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transport_reads(
            x_cmd, h_cmd0, h_obs, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
        )
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R21DualAddressContentTransport(**params)
