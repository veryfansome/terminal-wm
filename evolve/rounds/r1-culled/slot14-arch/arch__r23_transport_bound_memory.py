import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r23_transport_bound_memory"
DESCRIPTION = (
    "The r18 path-state trunk whose latent-transition register is replaced by a TWO-ADDRESS "
    "TRANSPORT memory: one shared address code is extracted from the raw command embedding in "
    "two ordered role slots (source, destination) through a common projection, and each step "
    "reads the content standing at the source slot and re-writes it, optionally edited by the "
    "shared transition operator, at the destination slot while erasing the source — so a chain "
    "of moves carries a content vector from address to address with no observation needed. "
    "Trained under run-structured observation occlusion so obs-free command-driven forwarding "
    "is in-distribution rather than an eval-time special case."
)


class R23TransportBoundMemory(R18PathStateLatentTransition):
    def __init__(
        self,
        path_d=96,
        addr_hidden=256,
        tcm_decay_init=0.995,
        tcm_gate_bias=-1.5,
        mv_bias=-1.5,
        erase_bias=0.0,
        edit_bias=-2.0,
        gate_weight_scale=0.1,
        readout_identity=0.25,
        occ_run_p=0.35,
        occ_max_run=3,
        occ_iid_p=0.05,
        occ_ramp_start=200,
        occ_ramp_end=900,
        **params,
    ):
        super().__init__(**params)

        self.path_d = max(8, int(path_d))
        ah = max(32, int(addr_hidden))

        self.addr_in = nn.Linear(D, ah)
        self.addr_src = nn.Linear(ah, self.path_d)
        self.addr_dst = nn.Linear(ah + self.path_d, self.path_d)
        self.addr_proj = nn.Linear(self.path_d, self.key_d, bias=False)
        self.role_mix = nn.Parameter(torch.eye(2))
        self.query_mix = nn.Parameter(torch.tensor([1.0, 0.0]))

        self.tcm_op_gates = nn.Linear(D, 3)
        with torch.no_grad():
            self.tcm_op_gates.weight.mul_(float(gate_weight_scale))
            self.tcm_op_gates.bias.copy_(
                torch.tensor([float(mv_bias), float(erase_bias), float(edit_bias)])
            )

        self.tcm_write = nn.Linear(2 * self.d, 1)
        nn.init.constant_(self.tcm_write.bias, 1.0)

        self.tcm_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.tcm_gate.bias, float(tcm_gate_bias))

        dq = min(max(float(tcm_decay_init), 0.9501), 0.9998)
        t = (dq - 0.95) / 0.0499
        self.tcm_logit_decay = nn.Parameter(torch.tensor(math.log(t / (1.0 - t))))

        with torch.no_grad():
            self.tr_read.weight.copy_(torch.eye(D) * float(readout_identity))
            self.tr_read.bias.zero_()

        self.occ_run_p = max(0.0, min(1.0, float(occ_run_p)))
        self.occ_iid_p = max(0.0, min(0.9, float(occ_iid_p)))
        self.occ_max_run = max(1, int(occ_max_run))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def _occ_scale(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return 1.0
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return x * x * (3.0 - 2.0 * x)

    def _occlude(self, tok_emb, key_pad):
        B, L = tok_emb.shape[0], tok_emb.shape[1]
        n_pair = L // 2
        if B == 0 or n_pair < 3:
            return tok_emb, key_pad
        scale = self._occ_scale()
        if scale <= 0.0 or (self.occ_run_p <= 0.0 and self.occ_iid_p <= 0.0):
            return tok_emb, key_pad

        device = tok_emb.device
        if key_pad is None:
            key_pad = torch.zeros(B, L, dtype=torch.bool, device=device)
        key_pad = key_pad.bool()

        pos = torch.arange(n_pair, device=device).unsqueeze(0)
        fire = torch.rand(B, 1, device=device) < (self.occ_run_p * scale)
        length = torch.randint(1, self.occ_max_run + 1, (B, 1), device=device)
        start = (torch.rand(B, 1, device=device) * float(n_pair)).long().clamp(0, n_pair - 1)
        run = fire & (pos >= start) & (pos < start + length)
        iid = torch.rand(B, n_pair, device=device) < (self.occ_iid_p * scale)
        drop = run | iid
        drop = drop & (~drop).any(dim=1, keepdim=True)

        full = torch.zeros(B, L, dtype=torch.bool, device=device)
        full[:, 1 : 2 * n_pair : 2] = drop
        tok_emb = tok_emb.masked_fill(full.unsqueeze(-1), 0.0)
        return tok_emb, key_pad | full

    def _addresses(self, cmd_raw):
        u = torch.nn.functional.gelu(self.addr_in(cmd_raw))
        z1 = self.addr_src(u)
        z2 = self.addr_dst(torch.cat([u, z1], dim=-1))
        m = self.role_mix
        s_code = m[0, 0] * z1 + m[0, 1] * z2
        d_code = m[1, 0] * z1 + m[1, 1] * z2
        q_code = self.query_mix[0] * z1 + self.query_mix[1] * z2
        return (
            self._unit(self.addr_proj(s_code)),
            self._unit(self.addr_proj(d_code)),
            self._unit(self.addr_proj(q_code)),
        )

    def _transport_reads(self, cmd_raw, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype

        cmd_raw = torch.nan_to_num(cmd_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        a_s, a_d, a_q = self._addresses(cmd_raw)

        gates = self.tcm_op_gates(cmd_raw)
        vc = valid_cmd.to(dtype)
        g_mv = torch.sigmoid(gates[..., 0]).to(dtype) * vc
        g_er = g_mv * torch.sigmoid(gates[..., 1]).to(dtype)
        alpha = torch.sigmoid(gates[..., 2]).to(dtype).unsqueeze(-1)

        if n_pair:
            wgt_in = torch.cat([x_cmd[:, :n_pair, :], self.obs_proj(obs_tok)], dim=-1)
            g_obs = torch.sigmoid(self.tcm_write(wgt_in)).squeeze(-1).to(dtype)
            g_obs = g_obs * (valid_obs & valid_cmd[:, :n_pair]).to(dtype)
            g_obs = self._pad_steps(g_obs.unsqueeze(-1), n_cmd).squeeze(-1)
        else:
            g_obs = x_cmd.new_zeros(B, n_cmd)

        obs_pad = self._pad_steps(obs_tok, n_cmd).to(dtype)

        gb = self.tr_out(torch.nn.functional.gelu(self.tr_in(x_cmd)))
        gamma_all = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta_all = gb[..., D:]

        decay = (0.95 + 0.0499 * torch.sigmoid(self.tcm_logit_decay)).to(dtype)

        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            As = a_s[:, i, :].to(dtype)
            Ad = a_d[:, i, :].to(dtype)
            Aq = a_q[:, i, :].to(dtype)
            A = torch.stack([As, Ad, Aq], dim=1)

            R = torch.bmm(A, mem)
            s_i = R[:, 0, :]
            d_i = R[:, 1, :]
            r_i = R[:, 2, :]
            reads.append(r_i)

            ai = alpha[:, i, :]
            edited = s_i * (1.0 + gamma_all[:, i, :]) + beta_all[:, i, :]
            v_t = (1.0 - ai) * s_i + ai * edited

            corr_d = g_mv[:, i].unsqueeze(-1) * (v_t - decay * d_i)
            corr_s = -g_er[:, i].unsqueeze(-1) * (decay * s_i)

            c_dq = (Ad * Aq).sum(dim=-1, keepdim=True)
            c_sq = (As * Aq).sum(dim=-1, keepdim=True)
            o_i = decay * r_i + c_dq * corr_d + c_sq * corr_s
            corr_q = g_obs[:, i].unsqueeze(-1) * (obs_pad[:, i, :] - o_i)

            corr = torch.stack([corr_s, corr_d, corr_q], dim=1)
            mem = decay * mem + torch.bmm(A.transpose(1, 2), corr)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_step += 1
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
        cmd_raw = tok_emb[:, 0::2, :]
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

        tcm = self._transport_reads(
            cmd_raw, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
        )
        tcm_x = tcm.to(x.dtype)
        tcm_rms = (tcm_x.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        g_tcm = torch.sigmoid(
            self.tcm_gate(torch.cat([h_cmd0, read_feat, tcm_rms], dim=-1))
        ).to(dtype)
        target_read = target_read + g_tcm * tcm.to(dtype)

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tcm_x)

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
    return R23TransportBoundMemory(**params)
