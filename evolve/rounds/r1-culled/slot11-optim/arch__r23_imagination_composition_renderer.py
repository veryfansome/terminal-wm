import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r20_endpoint_imagination_worldmodel import (
    R20EndpointImaginationWorldModel,
)

D = 768

NAME = "r23_imagination_composition_renderer"
DESCRIPTION = (
    "Crossover of the r20 endpoint-imagination world model and the r22 retrieval-composition "
    "renderer on the shared r18 path-state latent-transition trunk. Keeps the r22 zero-init "
    "RMS-capped composition MLP over the retrieved-content channels, and promotes the r20 "
    "imaginer from a branch that only fires on obs-missing suffixes to an always-on eval "
    "channel: at every command position j the model hard-selects the most recent earlier "
    "command m on the same path (command-embedding cosine, the same rule the r20 head uses to "
    "mine its training pairs), runs the shared imaginer cross-attention over the plan-time "
    "prefix pairs strictly before m queried by (c_m, c_j), and adds the result through a "
    "zero-init per-channel gain and a position gate. Zero contribution at init; strictly "
    "causal (only observations at pair index < m < j enter)."
)


class R23ImaginationCompositionRenderer(R20EndpointImaginationWorldModel):
    def __init__(
        self,
        comp_hidden=192,
        comp_rms_cap=1.0,
        comp_gate_bias=-1.0,
        imag_path_thresh=0.60,
        imag_gate_bias=-1.0,
        imag_dk=128,
        imag_dv=192,
        imag_heads=4,
        imag_qf=192,
        imag_hid=384,
        **params,
    ):
        super().__init__(
            imag_dk=int(imag_dk),
            imag_dv=int(imag_dv),
            imag_heads=int(imag_heads),
            imag_qf=int(imag_qf),
            imag_hid=int(imag_hid),
            **params,
        )
        self.comp_rms_cap = float(comp_rms_cap)
        self.imag_path_thresh = float(imag_path_thresh)

        comp_in_dim = 3 * D + self.d + self.ctx_d + self.sys_d
        ch = max(32, int(comp_hidden))
        self.comp_ln = nn.LayerNorm(comp_in_dim)
        self.comp_in = nn.Linear(comp_in_dim, ch)
        self.comp_out = nn.Linear(ch, D)
        nn.init.zeros_(self.comp_out.weight)
        nn.init.zeros_(self.comp_out.bias)
        self.comp_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.comp_gate.bias, float(comp_gate_bias))

        self.imag_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.imag_gate.bias, float(imag_gate_bias))
        self.imag_gain = nn.Parameter(torch.zeros(D))

    @staticmethod
    def _attend_pairs(im, q, k, v, mask):
        B, N, _ = q.shape
        P = k.size(1)
        H = im.h
        dkh = im.dk // H
        dvh = im.dv // H
        qh = q.view(B, N, H, dkh).transpose(1, 2)
        kh = k.view(B, P, H, dkh).transpose(1, 2)
        vh = v.view(B, P, H, dvh).transpose(1, 2)
        sc = torch.einsum("bhnd,bhpd->bhnp", qh, kh) / math.sqrt(max(1, dkh))
        sc = sc.masked_fill(~mask.unsqueeze(1), -1e30)
        att = torch.nan_to_num(torch.softmax(sc, dim=-1), nan=0.0)
        out = torch.einsum("bhnp,bhpd->bhnd", att, vh).transpose(1, 2).reshape(B, N, im.dv)
        return out * mask.any(dim=2, keepdim=True).to(out.dtype)

    def _imagine_stream(self, cmd_raw, obs_raw, valid_cmd, valid_obs, n_cmd, n_pair):
        B = cmd_raw.size(0)
        device = cmd_raw.device
        dtype = cmd_raw.dtype
        if n_pair == 0 or n_cmd == 0:
            return cmd_raw.new_zeros(B, n_cmd, D)

        im = self.imaginer
        cmd_c = torch.nan_to_num(cmd_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        obs_c = torch.nan_to_num(obs_raw, nan=0.0, posinf=1e4, neginf=-1e4)

        cu = self._unit(cmd_c)
        sim = torch.bmm(cu, cu.transpose(1, 2))
        pos = torch.arange(n_cmd, device=device)
        earlier = pos.unsqueeze(0) < pos.unsqueeze(1)
        same = (sim > self.imag_path_thresh) & valid_cmd.unsqueeze(1) & earlier.unsqueeze(0)
        posf = pos.view(1, 1, n_cmd).expand(B, n_cmd, n_cmd)
        m_idx = torch.where(same, posf, torch.full_like(posf, -1)).amax(dim=2)
        has_m = (m_idx >= 0) & valid_cmd
        mc = m_idx.clamp_min(0)

        observed = valid_cmd[:, :n_pair] & valid_obs
        ppos = torch.arange(n_pair, device=device)
        pmask = observed.unsqueeze(1) & (ppos.view(1, 1, n_pair) < mc.unsqueeze(2))
        pmask = pmask & has_m.unsqueeze(2)
        live = pmask.any(dim=2)

        pair_cat = torch.cat([cmd_c[:, :n_pair, :], obs_c], dim=-1)
        k = im.k_proj(pair_cat)
        v = im.v_proj(pair_cat)

        c_m = torch.gather(cmd_c, 1, mc.unsqueeze(-1).expand(B, n_cmd, D))
        c_r = cmd_c

        am = self._attend_pairs(im, im.qm(c_m), k, v, pmask)
        ar = self._attend_pairs(im, im.qr(c_r), k, v, pmask)
        qf = torch.tanh(im.qf(torch.cat([c_m, c_r], dim=-1)))
        z = im.norm(torch.cat([am, ar, qf], dim=-1))
        out = torch.nan_to_num(im.mlp(z), nan=0.0, posinf=1e4, neginf=-1e4)
        return out * live.unsqueeze(-1).to(dtype)

    def forward(self, tok_emb, types, key_pad):
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

        comp_z = torch.cat(
            [r_obs.to(x.dtype), rv, rp, h_cmd0, r_ctx.to(x.dtype), s_cmd.to(x.dtype)], dim=-1
        )
        comp_h = torch.nn.functional.gelu(self.comp_in(self.comp_ln(comp_z)))
        comp_raw = torch.nan_to_num(self.comp_out(comp_h), nan=0.0, posinf=1e4, neginf=-1e4)
        rms = (comp_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        scale = (self.comp_rms_cap / rms.clamp_min(self.comp_rms_cap)).detach()
        comp = comp_raw * scale
        g_comp = torch.sigmoid(self.comp_gate(torch.cat([h_cmd0, read_feat, rms.to(x.dtype)], dim=-1)))
        target_read = target_read + (g_comp * comp).to(dtype)

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        imag_stream = self._imagine_stream(
            tok_emb[:, 0::2, :], obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
        )
        g_imag = torch.sigmoid(self.imag_gate(torch.cat([h_cmd0, read_feat], dim=-1))).to(dtype)
        imag_contrib = g_imag * (imag_stream * self.imag_gain.to(dtype))

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype) + imag_contrib
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R23ImaginationCompositionRenderer(**params)
