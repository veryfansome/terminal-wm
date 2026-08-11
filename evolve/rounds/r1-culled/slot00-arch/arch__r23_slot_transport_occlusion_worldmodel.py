import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r23_slot_transport_occlusion_worldmodel"
DESCRIPTION = (
    "The r18 path-state world model plus a LOCATION-SLOT TRANSPORT memory: a shared "
    "path-key encoder reads three role-specific keys (source, destination, query) out of the "
    "raw command embedding, and a command-only step moves content between slots -- the value "
    "written into the destination slot is the value READ from the source slot (identity at "
    "init, then a learned affine edit), and the source slot is delta-cleared. Reads are taken "
    "before writes, so the channel is causal, and it updates on steps whose observation is "
    "absent, which the r18 delta-rule and latent-transition memories cannot do. The transport "
    "read joins the retrieval mixture as a fourth peer channel. Trained under ramped "
    "observation occlusion with contiguous silent spans, so a chain of unobserved mutations "
    "followed by a read is an in-distribution training regime; the r18 latent-transition "
    "memory also takes an imagined write on occluded steps. Eval forward is deterministic and "
    "consumes no RNG."
)


class R23SlotTransportOcclusionWorldModel(R18PathStateLatentTransition):
    supports_native_imagwrite = True

    def __init__(
        self,
        tp_key_d=64,
        tp_hidden=192,
        tp_val_hidden=192,
        tp_gscale=0.5,
        tp_move_bias=0.0,
        tp_clear_bias=-1.0,
        tp_bind_bias=1.0,
        tp_out_bias=-2.0,
        tp_decay_logit=3.0,
        occ_p=0.12,
        occ_span_p=0.35,
        occ_span_max=3,
        occ_ramp_start=200,
        occ_ramp_end=900,
        **params,
    ):
        super().__init__(**params)

        self.tp_key_d = max(8, int(tp_key_d))
        self.tp_gscale = float(tp_gscale)
        th = max(32, int(tp_hidden))
        vh = max(32, int(tp_val_hidden))

        self.tp_cmd_in = nn.Linear(D, th)
        self.tp_src_role = nn.Linear(th, th)
        self.tp_dst_role = nn.Linear(th, th)
        self.tp_qry_role = nn.Linear(th, th)
        self.tp_key = nn.Linear(th, self.tp_key_d, bias=False)

        self.tp_move_gate = nn.Linear(th + self.d, 1)
        self.tp_clear_gate = nn.Linear(th + self.d, 1)
        self.tp_bind_gate = nn.Linear(th + self.d, 1)
        nn.init.constant_(self.tp_move_gate.bias, float(tp_move_bias))
        nn.init.constant_(self.tp_clear_gate.bias, float(tp_clear_bias))
        nn.init.constant_(self.tp_bind_gate.bias, float(tp_bind_bias))

        self.tp_val_in = nn.Linear(th, vh)
        self.tp_val_out = nn.Linear(vh, 2 * D)
        nn.init.zeros_(self.tp_val_out.weight)
        nn.init.zeros_(self.tp_val_out.bias)

        self.tp_logit_decay = nn.Parameter(torch.tensor(float(tp_decay_logit)))

        self.tp_logit = nn.Linear(self.d + 4, 1)
        nn.init.zeros_(self.tp_logit.weight)
        nn.init.zeros_(self.tp_logit.bias)

        self.tp_out_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.tp_out_gate.bias, float(tp_out_bias))

        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.occ_span_p = max(0.0, min(1.0, float(occ_span_p)))
        self.occ_span_max = max(1, int(occ_span_max))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def _occ_ramp(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return 1.0
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return x * x * (3.0 - 2.0 * x)

    def _apply_occlusion(self, tok_emb, key_pad):
        B, L = tok_emb.shape[0], tok_emb.shape[1]
        n_pair = L // 2
        if B < 1 or n_pair < 1:
            return tok_emb, key_pad
        self.occ_step += 1
        ramp = self._occ_ramp()
        p = self.occ_p * ramp
        sp = self.occ_span_p * ramp
        if p <= 0.0 and sp <= 0.0:
            return tok_emb, key_pad

        dev = tok_emb.device
        if key_pad is None:
            key_pad = torch.zeros(B, L, dtype=torch.bool, device=dev)
        key_pad = key_pad.bool()

        drop = torch.rand(B, n_pair, device=dev) < p
        if sp > 0.0 and n_pair >= 2:
            idx = torch.arange(n_pair, device=dev)
            choose = torch.rand(B, device=dev) < sp
            start = (1 + (torch.rand(B, device=dev) * (n_pair - 1)).long()).clamp(
                1, n_pair - 1
            )
            span_len = (1 + (torch.rand(B, device=dev) * self.occ_span_max).long()).clamp(
                1, self.occ_span_max
            )
            span = (idx.unsqueeze(0) >= start.unsqueeze(1)) & (
                idx.unsqueeze(0) < (start + span_len).unsqueeze(1)
            )
            drop = drop | (span & choose.unsqueeze(1))

        full = torch.zeros(B, L, dtype=torch.bool, device=dev)
        full[:, 1 : 2 * n_pair : 2] = drop
        full = full & ~key_pad
        tok_emb = tok_emb.masked_fill(full.unsqueeze(-1), 0.0)
        return tok_emb, key_pad | full

    def _transport_reads(
        self, cmd_raw, h_cmd0, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
    ):
        B = h_cmd0.size(0)
        dtype = h_cmd0.dtype
        if n_cmd == 0:
            return h_cmd0.new_zeros(B, 0, D)

        g0 = torch.nn.functional.gelu(self.tp_cmd_in(cmd_raw.to(dtype)))
        k_src = self._unit(self.tp_key(torch.nn.functional.gelu(self.tp_src_role(g0))))
        k_dst = self._unit(self.tp_key(torch.nn.functional.gelu(self.tp_dst_role(g0))))
        k_qry = self._unit(self.tp_key(torch.nn.functional.gelu(self.tp_qry_role(g0))))

        gate_in = torch.cat([g0, h_cmd0], dim=-1)
        g_move = torch.sigmoid(self.tp_move_gate(gate_in))
        g_clear = torch.sigmoid(self.tp_clear_gate(gate_in))
        g_bind = torch.sigmoid(self.tp_bind_gate(gate_in))

        vo = self.tp_val_out(torch.nn.functional.gelu(self.tp_val_in(g0)))
        v_gamma = torch.tanh(vo[..., :D]) * self.tp_gscale
        v_beta = vo[..., D:]

        decay = (0.95 + 0.05 * torch.sigmoid(self.tp_logit_decay)).to(dtype)
        live = valid_cmd.to(dtype)
        obs_d = obs_tok.to(dtype)

        mem = h_cmd0.new_zeros(B, self.tp_key_d, D)
        reads = []
        for i in range(n_cmd):
            kq = k_qry[:, i, :]
            ks = k_src[:, i, :]
            kd = k_dst[:, i, :]

            v_qry = torch.bmm(kq.unsqueeze(1), mem).squeeze(1)
            reads.append(v_qry)

            li = live[:, i].unsqueeze(-1)
            v_src = torch.bmm(ks.unsqueeze(1), mem).squeeze(1)
            v_dst = torch.bmm(kd.unsqueeze(1), mem).squeeze(1)
            moved = v_src * (1.0 + v_gamma[:, i, :]) + v_beta[:, i, :]

            w_move = g_move[:, i, :] * li
            w_clear = w_move * g_clear[:, i, :]
            dst_corr = (moved - v_dst) * w_move
            src_corr = -(v_src * w_clear)

            if i < n_pair:
                bind = g_bind[:, i, :] * (
                    valid_obs[:, i] & valid_cmd[:, i]
                ).to(dtype).unsqueeze(-1)
                qd = (kq * kd).sum(dim=-1, keepdim=True)
                qs = (kq * ks).sum(dim=-1, keepdim=True)
                v_qry_post = v_qry + qd * dst_corr + qs * src_corr
                qry_corr = (obs_d[:, i, :] - v_qry_post) * bind
            else:
                qry_corr = torch.zeros_like(dst_corr)

            keys3 = torch.stack([kd, ks, kq], dim=2)
            corr3 = torch.stack([dst_corr, src_corr, qry_corr], dim=1)
            mem = decay * mem + torch.bmm(keys3, corr3)
            mem = torch.nan_to_num(
                mem, nan=0.0, posinf=1e4, neginf=-1e4
            ).clamp(-1e4, 1e4)

        return torch.nan_to_num(
            torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4
        )

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = (0.90 + 0.099 * torch.sigmoid(self.logit_decay)).to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            value = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (value - s_pre) * active
            imagined_corr = w[:, i].unsqueeze(-1) * (delta - s_pre) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(
            torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4
        )

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            tok_emb, key_pad = self._apply_occlusion(tok_emb, key_pad)

        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(
            B, L, device=device, dtype=torch.bool
        )

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(
            L, device, x.dtype
        ).unsqueeze(0)
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

        tp_read = self._transport_reads(
            self._pad_steps(cmd_raw, n_cmd),
            h_cmd0,
            self._pad_steps(obs_tok, n_cmd),
            valid_cmd,
            self._pad_steps(valid_obs, n_cmd),
            n_cmd,
            n_pair,
        )

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        rt = tp_read.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )
        tp_feat = (rt.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()

        logits3 = self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1))
        logit_tp = self.tp_logit(torch.cat([h_cmd0, read_feat, tp_feat], dim=-1))
        mix = torch.softmax(torch.cat([logits3, logit_tp], dim=-1), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
            + mix[:, :, 3:4] * tp_read.to(dtype)
        )

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
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
        g_tp_out = torch.sigmoid(
            self.tp_out_gate(torch.cat([h_cmd0, read_feat, tp_feat], dim=-1))
        ).to(dtype)
        pred_cmd = pred_cmd + g_tp_out * tp_read.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R23SlotTransportOcclusionWorldModel(**params)
