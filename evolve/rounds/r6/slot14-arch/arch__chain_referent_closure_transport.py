import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

D = 768

NAME = "chain_referent_closure_transport"
DESCRIPTION = (
    "The occlusion-denoised r18 path-state trunk plus a CHAIN-REFERENT CLOSURE: every command "
    "asks, from its raw embedding alone, which strictly earlier command last wrote the location "
    "it is about, and a one-shot transitive solve carries content along that pointer chain "
    "regardless of how many hops it has. The pointer is a cosine match with a learned temperature "
    "between a read-role projection of the querying command and a write-role projection of each "
    "earlier command, plus a learned weighted-L1 mismatch term over a low-dimensional code (exact "
    "string agreement is not a bilinear form, so a dot product alone cannot express it), minus a "
    "learned recency slope, and minus a learned CONSUMPTION penalty: the mass already claimed on a "
    "key by strictly earlier queries is subtracted from its logit, so one earlier write cannot be "
    "the referent of every later read. The register of a step is its own observation when a learned "
    "gate over that observation's hidden state says the observation carries content, and otherwise "
    "whatever its own referent holds — so a step with an empty or occluded observation relays "
    "instead of storing emptiness, and register = own*obs + (1-own)*A@register is solved in closed "
    "form by one unit-lower-triangular solve rather than unrolled. The retrieved content is passed "
    "through the trunk's shared command-conditioned transition operator and injected at every "
    "command position through a zero-initialised (D,D) readout, so the net is exactly the "
    "occlusion-denoised r18 at initialisation. Strictly causal: a step reads only registers of "
    "strictly earlier steps, and its own observation gate can only affect later steps."
)


class ChainReferentClosureTransport(R22ObservationOcclusionDenoising):
    def __init__(
        self,
        cc_d=128,
        cc_pd=32,
        cc_tau=10.0,
        cc_recency_init=0.1,
        cc_kappa_init=0.5,
        cc_gate_bias=-1.0,
        **params,
    ):
        super().__init__(**params)
        self.cc_d = max(8, int(cc_d))
        self.cc_pd = max(4, int(cc_pd))

        self.cc_ln = nn.LayerNorm(D)
        self.cc_q = nn.Linear(D, self.cc_d, bias=False)
        self.cc_k = nn.Linear(D, self.cc_d, bias=False)
        self.cc_p = nn.Linear(D, self.cc_pd, bias=False)
        self.cc_diff = nn.Linear(self.cc_pd, 1)
        nn.init.zeros_(self.cc_diff.bias)
        self.cc_keybias = nn.Linear(D, 1)
        nn.init.zeros_(self.cc_keybias.bias)
        self.cc_null = nn.Linear(D, 1)
        nn.init.zeros_(self.cc_null.bias)
        self.cc_own = nn.Linear(D + 2 * self.d, 1)
        nn.init.zeros_(self.cc_own.bias)
        self.cc_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.cc_gate.bias, float(cc_gate_bias))
        self.cc_read = nn.Linear(D, D)
        nn.init.zeros_(self.cc_read.weight)
        nn.init.zeros_(self.cc_read.bias)

        self.cc_logit_tau = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(1e-3, float(cc_tau)))))
        )
        self.cc_logit_recency = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(1e-4, float(cc_recency_init)))))
        )
        self.cc_logit_kappa = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(1e-4, float(cc_kappa_init)))))
        )

    def _chain_closure(self, cmd_raw, h_cmd0, h_obs_pad, obs_pad, valid_cmd, obs_present, n_cmd):
        B = cmd_raw.size(0)
        device = cmd_raw.device
        dtype = h_cmd0.dtype

        u = self.cc_ln(cmd_raw).to(dtype)
        q = self._unit(self.cc_q(u))
        k = self._unit(self.cc_k(u))
        pcode = self.cc_p(u)

        tau = torch.nn.functional.softplus(self.cc_logit_tau).to(dtype)
        scores = tau * torch.bmm(q, k.transpose(1, 2))
        mismatch = (pcode.unsqueeze(2) - pcode.unsqueeze(1)).abs()
        scores = scores + self.cc_diff(mismatch).squeeze(-1)
        scores = scores + self.cc_keybias(u).transpose(1, 2)

        idx = torch.arange(n_cmd, device=device)
        gap = (idx.view(n_cmd, 1) - idx.view(1, n_cmd)).clamp_min(0).to(dtype)
        recency = torch.nn.functional.softplus(self.cc_logit_recency).to(dtype)
        scores = scores - recency * gap.unsqueeze(0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)

        allowed = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0) & valid_cmd.unsqueeze(1)
        neg = torch.finfo(scores.dtype).min
        null = self.cc_null(u)
        kappa = torch.nn.functional.softplus(self.cc_logit_kappa).to(dtype)
        live = valid_cmd.to(dtype)

        consumed = scores.new_zeros(B, n_cmd)
        rows = []
        pre_used = []
        for j in range(n_cmd):
            row = scores[:, j, :] - kappa * consumed
            row = row.masked_fill(~allowed[:, j, :], neg)
            full = torch.cat([row, null[:, j, :]], dim=1)
            a = torch.softmax(full, dim=1)[:, :n_cmd] * live[:, j].unsqueeze(-1)
            a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
            pre_used.append((a * consumed).sum(dim=1, keepdim=True))
            rows.append(a)
            consumed = consumed + a

        attn = torch.stack(rows, dim=1)
        pre_used = torch.stack(pre_used, dim=1)

        own = torch.sigmoid(self.cc_own(torch.cat([u, h_cmd0, h_obs_pad], dim=-1)))
        own = own * (valid_cmd & obs_present).unsqueeze(-1).to(dtype)

        transfer = (1.0 - own) * attn
        eye = torch.eye(n_cmd, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        registers = self._solve_lower(eye - transfer, own * obs_pad.to(dtype))
        registers = torch.nan_to_num(registers, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        pulled = torch.nan_to_num(torch.bmm(attn, registers), nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True)
        mass = attn.sum(dim=2, keepdim=True)
        return pulled, sharp, mass, pre_used

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_step += 1
            p_occ = self._occ_prob()
            Bo, Lo = tok_emb.shape[0], tok_emb.shape[1]
            if p_occ > 0.0 and Lo >= 2 and Bo > 0:
                if key_pad is None:
                    key_pad = torch.zeros(Bo, Lo, dtype=torch.bool, device=tok_emb.device)
                key_pad = key_pad.bool()
                np_occ = Lo // 2
                drop = torch.rand(Bo, np_occ, device=tok_emb.device) < p_occ
                drop_full = torch.zeros(Bo, Lo, dtype=torch.bool, device=tok_emb.device)
                drop_full[:, 1:2 * np_occ:2] = drop
                tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                key_pad = key_pad | drop_full

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
        cmd_raw = tok_emb[:, 0::2, :]

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

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        obs_pad = self._pad_steps(obs_for_prev, n_cmd)
        h_obs_pad = self._pad_steps(h_obs, n_cmd)
        obs_present = self._pad_steps(valid_obs, n_cmd)
        pulled, sharp, mass, pre_used = self._chain_closure(
            cmd_raw, h_cmd0, h_obs_pad, obs_pad, valid_cmd, obs_present, n_cmd
        )
        chain_render = self._transition(pulled, x_cmd)
        chain_feat = torch.cat(
            [
                (pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                sharp,
                mass,
                pre_used,
            ],
            dim=-1,
        )
        g_cc = torch.sigmoid(self.cc_gate(torch.cat([h_cmd0, chain_feat.to(x.dtype)], dim=-1)))
        cc_contrib = torch.nan_to_num(
            g_cc.to(dtype) * self.cc_read(chain_render), nan=0.0, posinf=1e4, neginf=-1e4
        )

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype) + cc_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return ChainReferentClosureTransport(**params)
