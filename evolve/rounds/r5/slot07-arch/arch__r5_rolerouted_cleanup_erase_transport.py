import math

import torch
import torch.nn as nn

D = 768

NAME = "r5_rolerouted_cleanup_erase_transport"
DESCRIPTION = (
    "Content-gated latent-transition transport memory whose read and write addresses are produced "
    "by a ROLE ROUTER over a shared bank of address extractors, cleaned up against a shared "
    "address codebook, and paired with a destructive read. Four extractors emit unit vectors in "
    "one common key space: two linear ones from the trunk command feature (the write one "
    "initialised as a copy of the read one) and two from a GELU feature of the raw frozen command "
    "embedding. A softmax router driven only by that command feature — no positions, no "
    "observations — mixes the bank separately for the read role and the write role, so which "
    "extractor supplies an address is a multiplicative function of the command rather than a fixed "
    "linear map of it; the router is initialised biased to read-from-slot-0 and write-to-slot-1, "
    "which are identical at init, so both addresses start equal. Each mixed address is then "
    "blended by a learned gate toward its softmax projection onto a learned codebook of address "
    "directions, so two commands whose addresses are merely close are pulled onto the same "
    "direction. Per step the memory reads its read slot, then subtracts the read content from that "
    "slot in proportion to a command-conditioned erase gate, then delta-rule writes the "
    "content-gated transition value into the write slot. The observation-presence gate, the shared "
    "latent-transition operator and its transition_from_emb entry point, the file and path "
    "delta-rule memories, the system summary and all readouts are carried over unchanged."
)


class R5RoleRoutedCleanupEraseTransport(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        presence_hidden=128,
        presence_scale=0.5,
        presence_margin=1.0,
        presence_bias=2.0,
        addr_hidden=256,
        route_bias=3.0,
        route_temp=1.0,
        n_codes=96,
        code_temp=0.25,
        snap_bias=-2.0,
        erase_bias=-2.0,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        self.tr_gscale = float(tr_gscale)
        self.n_slots = 4
        self.addr_hidden = max(32, int(addr_hidden))
        self.n_codes = max(2, int(n_codes))
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(3, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.tr_path = nn.Linear(self.d, self.key_d, bias=False)
        self.tr_path_w = nn.Linear(self.d, self.key_d, bias=False)
        with torch.no_grad():
            self.tr_path_w.weight.copy_(self.tr_path.weight)

        self.tr_addr_norm = nn.LayerNorm(D)
        self.tr_addr_in = nn.Linear(D, self.addr_hidden)
        self.tr_slot_a = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.tr_slot_b = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.tr_route = nn.Linear(self.addr_hidden, 2 * self.n_slots)
        with torch.no_grad():
            self.tr_route.weight.normal_(0.0, 0.02)
            rb = torch.zeros(2, self.n_slots)
            rb[0, 0] = float(route_bias)
            rb[1, 1] = float(route_bias)
            self.tr_route.bias.copy_(rb.view(-1))
        self.tr_route_log_temp = nn.Parameter(torch.tensor(math.log(max(1e-2, float(route_temp)))))
        self.tr_codes = nn.Parameter(torch.randn(self.n_codes, self.key_d))
        self.tr_code_log_temp = nn.Parameter(torch.tensor(math.log(max(1e-2, float(code_temp)))))
        self.tr_snap_gate = nn.Parameter(torch.tensor(float(snap_bias)))
        self.tr_erase = nn.Linear(self.addr_hidden, 1)
        with torch.no_grad():
            self.tr_erase.weight.normal_(0.0, 0.02)
            self.tr_erase.bias.fill_(float(erase_bias))

        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        ph = max(16, int(presence_hidden))
        self.pres_proto = nn.Parameter(torch.zeros(D))
        self.pres_scale = nn.Parameter(torch.tensor(float(presence_scale)))
        self.pres_margin = nn.Parameter(torch.tensor(float(presence_margin)))
        self.pres_in = nn.Linear(2 * self.d, ph)
        self.pres_out = nn.Linear(ph, 1)
        nn.init.zeros_(self.pres_out.weight)
        nn.init.constant_(self.pres_out.bias, float(presence_bias))

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _cleanup(self, a):
        a = self._unit(a)
        codes = self._unit(self.tr_codes).to(a.dtype)
        temp = self.tr_code_log_temp.exp().clamp(0.02, 5.0).to(a.dtype)
        sim = torch.matmul(a, codes.transpose(0, 1)) / temp
        q = self._unit(torch.matmul(torch.softmax(sim, dim=-1), codes))
        g = torch.sigmoid(self.tr_snap_gate).to(a.dtype)
        return self._unit((1.0 - g) * a + g * q)

    def _addresses(self, x_cmd, cmd_raw):
        dtype = x_cmd.dtype
        u = torch.nn.functional.gelu(self.tr_addr_in(self.tr_addr_norm(cmd_raw.to(dtype))))
        s0 = self._unit(self.tr_path(x_cmd))
        s1 = self._unit(self.tr_path_w(x_cmd))
        s2 = self._unit(self.tr_slot_a(u))
        s3 = self._unit(self.tr_slot_b(u))
        slots = torch.stack([s0, s1, s2, s3], dim=2)

        B, n = x_cmd.size(0), x_cmd.size(1)
        temp = self.tr_route_log_temp.exp().clamp(0.1, 5.0).to(dtype)
        logits = self.tr_route(u).view(B, n, 2, self.n_slots) / temp
        wts = torch.softmax(logits, dim=-1)
        mixed = torch.matmul(wts, slots)

        p_r = self._cleanup(mixed[:, :, 0, :])
        p_w = self._cleanup(mixed[:, :, 1, :])
        erase = torch.sigmoid(self.tr_erase(u)).squeeze(-1)
        return p_r, p_w, erase

    def _presence(self, obs_tok, obs_x, h_obs, valid_obs):
        if obs_tok.size(1) == 0:
            return obs_x.new_zeros(obs_x.size(0), 0)
        diff = obs_tok - self.pres_proto.view(1, 1, -1).to(obs_tok.dtype)
        d2 = diff.pow(2).mean(dim=-1)
        mlp = self.pres_out(torch.tanh(self.pres_in(torch.cat([obs_x, h_obs], dim=-1)))).squeeze(-1)
        logit = self.pres_scale * (d2.to(mlp.dtype) - self.pres_margin) + mlp
        a = torch.sigmoid(logit.clamp(-30.0, 30.0))
        return a * valid_obs.to(a.dtype)

    def _transition_reads(self, x_cmd, cmd_raw, obs_tok, valid_cmd, pres, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p_r, p_w, erase = self._addresses(x_cmd, cmd_raw)
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ri = p_r[:, i, :]
            wi_key = p_w[:, i, :]
            live = valid_cmd[:, i].to(dtype).unsqueeze(-1)
            s_pre = torch.bmm(ri.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                a_i = pres[:, i].to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                a_i = s_pre.new_zeros(B, 1)
            mut = w[:, i].unsqueeze(-1)
            base_i = a_i * obs_i + (1.0 - a_i) * s_pre
            v_i = (1.0 - mut) * base_i + mut * delta
            e_i = erase[:, i].to(dtype).unsqueeze(-1) * live
            mem = mem - torch.bmm(ri.unsqueeze(2), (e_i * s_pre).unsqueeze(1))
            s_dst = torch.bmm(wi_key.unsqueeze(1), mem).squeeze(1)
            corr = (v_i - s_dst) * live
            mem = decay * mem + torch.bmm(wi_key.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 2)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x_all = self.obs_proj(tok_emb)
        x = torch.where((t == 1).unsqueeze(-1), obs_x_all, cmd_x)
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
        obs_x = obs_x_all[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        a_pair = self._presence(obs_tok, obs_x, h_obs, valid_obs)
        live_pair = (valid_cmd[:, :n_pair] & valid_obs).to(x.dtype)
        pres_pair = a_pair.to(x.dtype) * live_pair

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        pres_cmd = self._pad_steps(pres_pair, n_cmd)
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

        tr_reads = self._transition_reads(
            x_cmd, cmd_raw, obs_tok, valid_cmd, pres_pair, n_cmd, n_pair
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
    return R5RoleRoutedCleanupEraseTransport(**params)
