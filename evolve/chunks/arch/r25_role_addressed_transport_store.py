import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r25_role_addressed_transport_store"
DESCRIPTION = (
    "The r18 path-state trunk whose latent-transition memory is replaced by a ROLE-ADDRESSED "
    "TRANSPORT STORE. Every command's raw 768-d embedding passes through one shared address "
    "trunk and is read out under three roles — query, move-source, move-destination — via "
    "role-specific pre-activation biases and diagonal gates feeding ONE shared projection, so "
    "all three roles land in a single address space. Each role code is a per-head softmax over "
    "slots (multi-head learned hashing) with a learned temperature. The store keeps a value "
    "memory and a parallel scalar mass memory, so a read is a mass-normalized convex combination "
    "of the raw observation embeddings written so far, occupancy-weighted across heads. At each "
    "step the store first reads at the query address (the emitted per-step read), then reads at "
    "the source address, passes that content through the shared latent-transition operator "
    "f(s,cmd)=s*(1+gamma)+beta, and writes the result at the DESTINATION address with a "
    "content-presence gate, while a separate gate writes the step's own observation at the query "
    "address; per-address erase strengths and a source-delete strength are learned. Because the "
    "value written at a destination is whatever the source address currently holds, chained "
    "renames compose without any resolved routing being supplied from outside. Carries the "
    "learned observation content-presence gate (prototype distance + MLP) into the file-memory "
    "write amount, the forgetting factor, the system summary and the previous-observation "
    "channel. Injected through the inherited (D,D) transition readout, initialised to a small "
    "multiple of the identity, under the inherited sigmoid gate."
)


class R25RoleAddressedTransportStore(R18PathStateLatentTransition):
    def __init__(
        self,
        addr_hidden=256,
        addr_heads=4,
        addr_slots=16,
        addr_temp=0.5,
        role_bias_scale=0.3,
        role_gate_scale=0.5,
        gate_hidden=128,
        move_bias=-1.0,
        obs_write_bias=1.5,
        content_bias=1.0,
        erase_bias=-1.0,
        presence_hidden=128,
        presence_scale=0.5,
        presence_margin=1.0,
        presence_bias=2.0,
        transport_readout_init=0.1,
        **params,
    ):
        super().__init__(**params)
        self.addr_hidden = max(32, int(addr_hidden))
        self.addr_heads = max(1, int(addr_heads))
        self.addr_slots = max(2, int(addr_slots))

        self.addr_ln = nn.LayerNorm(D)
        self.addr_in = nn.Linear(D, self.addr_hidden)
        self.role_bias = nn.Parameter(torch.randn(3, self.addr_hidden) * float(role_bias_scale))
        self.role_gate = nn.Parameter(torch.randn(3, self.addr_hidden) * float(role_gate_scale))
        self.addr_proj = nn.Linear(self.addr_hidden, self.addr_heads * self.addr_slots, bias=False)
        self.addr_log_temp = nn.Parameter(torch.tensor(math.log(max(1e-2, float(addr_temp)))))
        self.occ_logit = nn.Parameter(torch.tensor(0.0))

        gh = max(16, int(gate_hidden))
        self.gate_in = nn.Linear(self.d + self.addr_hidden + self.key_d, gh)
        self.move_out = nn.Linear(gh, 1)
        nn.init.constant_(self.move_out.bias, float(move_bias))
        self.obs_write_out = nn.Linear(gh, 1)
        nn.init.constant_(self.obs_write_out.bias, float(obs_write_bias))
        self.content_out = nn.Linear(gh + 2, 1)
        nn.init.constant_(self.content_out.bias, float(content_bias))
        self.store_erase = nn.Linear(gh, 3)
        nn.init.constant_(self.store_erase.bias, float(erase_bias))
        self.store_logit_decay = nn.Parameter(torch.tensor(2.0))

        ph = max(16, int(presence_hidden))
        self.pres_proto = nn.Parameter(torch.zeros(D))
        self.pres_scale = nn.Parameter(torch.tensor(float(presence_scale)))
        self.pres_margin = nn.Parameter(torch.tensor(float(presence_margin)))
        self.pres_in = nn.Linear(2 * self.d, ph)
        self.pres_out = nn.Linear(ph, 1)
        nn.init.zeros_(self.pres_out.weight)
        nn.init.constant_(self.pres_out.bias, float(presence_bias))

        with torch.no_grad():
            self.tr_read.weight.copy_(torch.eye(D) * float(transport_readout_init))
            self.tr_read.bias.zero_()

    def _presence(self, obs_tok, obs_x, h_obs, valid_obs):
        if obs_tok.size(1) == 0:
            return obs_x.new_zeros(obs_x.size(0), 0)
        diff = obs_tok - self.pres_proto.view(1, 1, -1).to(obs_tok.dtype)
        d2 = diff.pow(2).mean(dim=-1)
        mlp = self.pres_out(torch.tanh(self.pres_in(torch.cat([obs_x, h_obs], dim=-1)))).squeeze(-1)
        logit = self.pres_scale * (d2.to(mlp.dtype) - self.pres_margin) + mlp
        a = torch.sigmoid(logit.clamp(-30.0, 30.0))
        return a * valid_obs.to(a.dtype)

    def _role_codes(self, cmd_raw):
        base = self.addr_in(self.addr_ln(cmd_raw))
        tau = self.addr_log_temp.exp().clamp(0.05, 4.0)
        codes = []
        for r in range(3):
            hr = F.gelu(base + self.role_bias[r]) * torch.sigmoid(self.role_gate[r])
            z = self.addr_proj(hr)
            z = z.view(z.size(0), z.size(1), self.addr_heads, self.addr_slots)
            z = z - z.mean(dim=-1, keepdim=True)
            z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            codes.append(torch.softmax(z / tau, dim=-1))
        return base, codes[0], codes[1], codes[2]

    def _store_read(self, mem, mass, code):
        B, H, K, V = mem.shape
        num = torch.bmm(code.reshape(B * H, 1, K), mem.reshape(B * H, K, V)).view(B, H, V)
        den = (code * mass).sum(dim=-1)
        val_h = num / (den.unsqueeze(-1) + 1e-4)
        ref = F.softplus(self.occ_logit) + 0.1
        sat = den / (den + ref)
        w = sat / (sat.sum(dim=-1, keepdim=True) + 1e-4)
        val = (w.unsqueeze(-1) * val_h).sum(dim=1)
        return val, den.mean(dim=-1, keepdim=True)

    def _transport_reads(self, cmd_raw, x_cmd, obs_tok, valid_cmd, pres_cmd, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        H = self.addr_heads
        K = self.addr_slots

        base, code_read, code_src, code_dst = self._role_codes(cmd_raw.to(dtype))
        gate_feat = torch.cat([x_cmd, F.gelu(base), self.tr_path(x_cmd)], dim=-1)
        gate_h = torch.tanh(self.gate_in(gate_feat))

        live = valid_cmd.to(dtype)
        g_move_all = torch.sigmoid(
            self.move_out(gate_h).squeeze(-1) + self.tr_mut_gate(x_cmd).squeeze(-1)
        ) * live
        g_obs_all = torch.sigmoid(self.obs_write_out(gate_h).squeeze(-1)) * pres_cmd.to(dtype)
        erase_all = torch.sigmoid(self.store_erase(gate_h))
        decay = (0.90 + 0.099 * torch.sigmoid(self.store_logit_decay)).to(dtype)

        mem = x_cmd.new_zeros(B, H, K, D)
        mass = x_cmd.new_zeros(B, H, K)
        reads = []
        for i in range(n_cmd):
            c_read = code_read[:, i]
            c_src = code_src[:, i]
            c_dst = code_dst[:, i]

            val_read, _ = self._store_read(mem, mass, c_read)
            reads.append(val_read)

            val_src, occ_src = self._store_read(mem, mass, c_src)
            v_move = self._transition(val_src, x_cmd[:, i, :])
            rms_src = (val_src.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
            content = torch.sigmoid(
                self.content_out(torch.cat([gate_h[:, i], rms_src, occ_src], dim=-1))
            )

            g_ins = g_move_all[:, i].unsqueeze(-1) * content
            g_del = g_ins * erase_all[:, i, 2:3]
            g_obs = g_obs_all[:, i].unsqueeze(-1)

            if i < n_pair:
                v_obs = obs_tok[:, i, :].to(dtype)
            else:
                v_obs = x_cmd.new_zeros(B, D)

            keep = (
                (1.0 - (g_obs * erase_all[:, i, 0:1]).unsqueeze(-1) * c_read)
                * (1.0 - (g_ins * erase_all[:, i, 1:2]).unsqueeze(-1) * c_dst)
                * (1.0 - g_del.unsqueeze(-1) * c_src)
            )
            w_obs = g_obs.unsqueeze(-1) * c_read
            w_ins = g_ins.unsqueeze(-1) * c_dst
            codes = torch.stack([w_obs, w_ins], dim=-1)
            vals = torch.stack([v_obs, v_move], dim=1).unsqueeze(1).expand(B, H, 2, D)

            mem = torch.baddbmm(
                (decay * mem * keep.unsqueeze(-1)).reshape(B * H, K, D),
                codes.reshape(B * H, K, 2),
                vals.reshape(B * H, 2, D),
            ).view(B, H, K, D)
            mass = decay * mass * keep + w_obs + w_ins

        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

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
        obs_x_all = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x_all)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        cmd_raw = tok_emb[:, 0::2, :]
        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]
        obs_x = obs_x_all[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        a_pair = self._presence(obs_tok, obs_x, h_obs, valid_obs)
        live_pair = (valid_cmd[:, :n_pair] & valid_obs).to(x.dtype)
        pres_pair = a_pair.to(x.dtype) * live_pair
        pres_cmd = self._pad_steps(pres_pair, n_cmd)

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        beta = self._pad_steps(amount_pair, n_cmd) * pres_cmd

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = 1.0 - (1.0 - decay.to(x.dtype)) * pres_cmd

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
            sal = sal * pres_pair.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * pres_pair.unsqueeze(-1).to(dtype)
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
            cmd_raw, x_cmd, obs_tok, valid_cmd, pres_cmd, n_cmd, n_pair
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
    return R25RoleAddressedTransportStore(**params)
