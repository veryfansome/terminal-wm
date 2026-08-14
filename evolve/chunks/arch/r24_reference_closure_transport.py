import math

import torch
import torch.nn as nn

D = 768

NAME = "r24_reference_closure_transport"
DESCRIPTION = (
    "The r18 latent-transition trunk plus a REFERENCE-CLOSURE transport branch and a learned "
    "observation-informativeness gate. Every command position carries a content register C_i "
    "defined by C_i = g_i * obs_i + (1 - g_i) * sum_{j<i} A_ij C_j, where A is a strictly "
    "lower-triangular soft reference graph scored from raw command embeddings by a bilinear "
    "term, a learned absolute-difference mismatch term, a per-key bias, a learned recency decay "
    "and a null slot that lets a row keep less than unit mass. Because the recursion is linear "
    "in C, the whole transitive closure is obtained exactly by one batched unit-lower-triangular "
    "solve (I - diag(1-g) A) C = diag(g) OBS, so a reference chain of any length resolves in a "
    "single pass instead of one attention layer per hop. The content pulled at position i, "
    "sum_{j<i} A_ij C_j, is rendered through the arch's SHARED command-conditioned affine "
    "transition operator and injected into the command-position prediction through a zero-init "
    "(D,D) readout gated on chain confidence. The r18 per-path transition memory is retained and "
    "its write base is a learned soft blend of the step's observation and the slot's current "
    "content, so a step whose observation carries no content composes the operator on the "
    "retrieved state instead of overwriting it. Strictly causal: position i reads only registers "
    "of strictly earlier steps, which depend only on strictly earlier observations."
)


class R24ReferenceClosureTransport(nn.Module):
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
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        ref_d=128,
        ref_pd=64,
        ref_recency_init=0.1,
        obs_info_bias=4.0,
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
        self.ref_d = max(8, int(ref_d))
        self.ref_pd = max(8, int(ref_pd))
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
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

        self.obs_info = nn.Linear(self.d, 1)
        nn.init.constant_(self.obs_info.bias, float(obs_info_bias))

        self.ref_ln = nn.LayerNorm(D)
        self.ref_q = nn.Linear(D, self.ref_d, bias=False)
        self.ref_k = nn.Linear(D, self.ref_d, bias=False)
        self.ref_p = nn.Linear(D, self.ref_pd, bias=False)
        self.ref_diff = nn.Linear(self.ref_pd, 1)
        nn.init.zeros_(self.ref_diff.bias)
        self.ref_keybias = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_keybias.bias)
        self.ref_null = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_null.bias)
        self.ref_own = nn.Linear(D + self.d, 1)
        nn.init.zeros_(self.ref_own.bias)
        self.ref_gate = nn.Linear(self.d + 3, 1)
        nn.init.zeros_(self.ref_gate.bias)
        self.ref_read = nn.Linear(D, D)
        nn.init.zeros_(self.ref_read.weight)
        nn.init.zeros_(self.ref_read.bias)
        rr = max(1e-4, float(ref_recency_init))
        self.ref_recency = nn.Parameter(torch.tensor(math.log(math.expm1(rr))))

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

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        info = torch.sigmoid(self.obs_info(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                active = x_cmd.new_zeros(B, 1)
            fi = info[:, i].unsqueeze(-1)
            base_i = fi * obs_i + (1.0 - fi) * s_pre
            wi = w[:, i].unsqueeze(-1)
            if i >= n_pair:
                wi = wi * 0.0
            v_i = (1.0 - wi) * base_i + wi * delta
            corr = (v_i - s_pre) * active
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def _reference_closure(self, cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd):
        B = cmd_raw.size(0)
        device = cmd_raw.device
        dtype = h_cmd0.dtype

        u = self.ref_ln(cmd_raw).to(dtype)
        q = self.ref_q(u)
        k = self.ref_k(u)
        p = self.ref_p(u)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(self.ref_d))
        mismatch = (p.unsqueeze(2) - p.unsqueeze(1)).abs()
        scores = scores + self.ref_diff(mismatch).squeeze(-1)
        scores = scores + self.ref_keybias(u).transpose(1, 2)

        idx = torch.arange(n_cmd, device=device)
        gap = (idx.view(n_cmd, 1) - idx.view(1, n_cmd)).clamp_min(0).to(dtype)
        scores = scores - torch.nn.functional.softplus(self.ref_recency).to(dtype) * gap.unsqueeze(0)

        allowed = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0) & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        full = torch.cat([scores, self.ref_null(u)], dim=2)
        attn = torch.softmax(full, dim=2)[:, :, :n_cmd]
        attn = attn * valid_cmd.unsqueeze(-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        own = torch.sigmoid(self.ref_own(torch.cat([u, h_cmd0], dim=-1)))
        own = own * valid_cmd.unsqueeze(-1).to(dtype)

        transfer = (1.0 - own) * attn
        eye = torch.eye(n_cmd, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        system = eye - transfer
        rhs = own * obs_pad.to(dtype)
        registers = self._solve_lower(system, rhs)
        registers = torch.nan_to_num(registers, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        pulled = torch.bmm(attn, registers)
        pulled = torch.nan_to_num(pulled, nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True)
        mass = attn.sum(dim=2, keepdim=True)
        return pulled, sharp, mass

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
        pulled, sharp, mass = self._reference_closure(cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd)
        chain_render = self._transition(pulled, x_cmd)
        chain_feat = torch.cat(
            [(pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(), sharp, mass], dim=-1
        )
        g_ref = torch.sigmoid(self.ref_gate(torch.cat([h_cmd0, chain_feat], dim=-1)))
        ref_contrib = g_ref * self.ref_read(chain_render)
        ref_contrib = torch.nan_to_num(ref_contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype) + ref_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R24ReferenceClosureTransport(**params)
