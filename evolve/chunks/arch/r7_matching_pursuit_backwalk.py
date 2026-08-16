import math

import torch
import torch.nn as nn

D = 768

NAME = "r7_matching_pursuit_backwalk"
DESCRIPTION = (
    "A causal transformer trunk plus a MATCHING-PURSUIT BACKWARD CHAIN WALK over raw command "
    "embeddings. A mean-pooled embedding of 'mv SRC DST' is additive in its two path arguments, so "
    "a read address is advanced one hop backwards by pulling the best-matching earlier command "
    "atom and SUBTRACTING the current address from it: z <- W(mu) - a*z + b. That telescopes "
    "f(P_k) -> f(P_k-1) -> ... along a chain of silent moves without ever separating source from "
    "destination. Atoms live in a learned address space, deflated by a learned orthonormal "
    "null-frame and centered on the CAUSAL PREFIX MEAN of the atoms seen so far, so the shared "
    "verb and path-prefix mass is removed and matching keys on the distinguishing argument. Each "
    "hop also harvests, from the same residual, the observation of the earlier command that read "
    "that path; an adaptive-computation-time halting head mixes the per-hop harvests into one "
    "content prediction. Hop selection is pushed strictly backwards in command index by a learned "
    "soft monotonicity penalty on a soft selected-index pointer, which is what disambiguates the "
    "move that wrote a path from the move that later read it. Hop 0 is exact-address retrieval; "
    "hop h resolves a depth-h chain."
)


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12))


def _rms(x):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


class R7MatchingPursuitBackwalk(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        addr_d=128,
        hops=6,
        n_null=8,
        halt_hidden=96,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        tau_init=6.0,
        back_lambda_init=2.0,
        cluster_beta_init=12.0,
        cluster_thresh_init=-0.2,
        cluster_prior_init=1.0,
        halt_bias=0.0,
        halt_kappa_init=8.0,
        halt_thresh_init=0.5,
        step_alpha_init=0.7,
        harvest_scale=0.25,
        harvest_gate_bias=-2.0,
        fuse_gate_bias=-1.0,
        **unused,
    ):
        super().__init__()
        self.D = D
        self.d = max(8, int(d))
        self.layers = max(1, int(layers))
        self.addr_d = max(8, int(addr_d))
        self.hops = max(1, int(hops))
        self.n_null = max(1, min(int(n_null), self.addr_d - 1))
        self.tr_gscale = float(tr_gscale)
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

        self.addr_ln = nn.LayerNorm(D)
        self.addr = nn.Linear(D, self.addr_d, bias=False)
        self.null_basis = nn.Parameter(torch.randn(self.n_null, self.addr_d) * 0.2)

        self.move_metric = nn.Parameter(torch.ones(self.addr_d))
        self.harv_metric = nn.Parameter(torch.ones(self.addr_d))
        t0 = math.log(math.expm1(max(1e-3, float(tau_init))))
        self.tau_move = nn.Parameter(torch.tensor(t0))
        self.tau_harv = nn.Parameter(torch.tensor(t0))
        l0 = math.log(math.expm1(max(1e-3, float(back_lambda_init))))
        self.lam_back = nn.Parameter(torch.tensor(l0))
        b0 = math.log(math.expm1(max(1e-3, float(cluster_beta_init))))
        self.clus_beta = nn.Parameter(torch.tensor(b0))
        self.clus_thresh = nn.Parameter(torch.tensor(float(cluster_thresh_init)))
        p0 = math.log(math.expm1(max(1e-3, float(cluster_prior_init))))
        self.clus_prior = nn.Parameter(torch.tensor(p0))
        self.harv_shift = nn.Parameter(torch.zeros(self.addr_d))

        self.move_key_bias = nn.Linear(D, 1)
        self.harv_key_bias = nn.Linear(D, 1)
        for lin in (self.move_key_bias, self.harv_key_bias):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.move_null = nn.Linear(self.addr_d, 1)
        self.harv_null = nn.Linear(self.addr_d, 1)
        for lin in (self.move_null, self.harv_null):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.step_lin = nn.Linear(self.addr_d, self.addr_d, bias=False)
        with torch.no_grad():
            self.step_lin.weight.copy_(torch.eye(self.addr_d))
        self.step_alpha = nn.Parameter(torch.tensor(float(step_alpha_init)))
        self.step_bias = nn.Parameter(torch.zeros(self.addr_d))

        hh = max(16, int(halt_hidden))
        self.halt_in = nn.Linear(self.d + 5, hh)
        self.halt_out = nn.Linear(hh, 1)
        nn.init.normal_(self.halt_out.weight, std=0.02)
        nn.init.constant_(self.halt_out.bias, float(halt_bias))
        self.halt_kappa = nn.Parameter(torch.tensor(float(halt_kappa_init)))
        self.halt_thresh = nn.Parameter(torch.tensor(float(halt_thresh_init)))

        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        nn.init.normal_(self.tr_out.weight, std=0.01)
        nn.init.zeros_(self.tr_out.bias)

        self.harv_out = nn.Linear(D, D)
        with torch.no_grad():
            self.harv_out.weight.copy_(torch.eye(D) * float(harvest_scale))
            self.harv_out.bias.zero_()
        self.harv_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.harv_gate.bias, float(harvest_gate_bias))

        self.content_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d, self.d)
        nn.init.constant_(self.fuse_gate.bias, float(fuse_gate_bias))

        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

    def _positional(self, L, device, dtype):
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        half = (self.d + 1) // 2
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        ang = pos * div
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(ang[:, : pe[:, 0::2].size(1)])
        pe[:, 1::2] = torch.cos(ang[:, : pe[:, 1::2].size(1)])
        return pe

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _null_frame(self):
        rows = []
        for i in range(self.n_null):
            v = self.null_basis[i]
            for w in rows:
                v = v - (v * w).sum() * w
            rows.append(v * torch.rsqrt(v.pow(2).sum().clamp_min(1e-8)))
        return torch.stack(rows, dim=0)

    @staticmethod
    def _deflate(a, Q):
        return a - torch.matmul(torch.matmul(a, Q.transpose(0, 1)), Q)

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

    def _residual_atoms(self, atom, allowed, live):
        n = atom.size(1)
        wsum = torch.cumsum(atom * live, dim=1) - atom * live
        cnt = (torch.cumsum(live, dim=1) - live).clamp_min(1.0)
        g = wsum / cnt
        gap = atom - g
        a_sq = atom.pow(2).sum(dim=-1)
        cross = torch.bmm(g, atom.transpose(1, 2))
        g_sq = g.pow(2).sum(dim=-1)
        key_inv = torch.rsqrt((a_sq.unsqueeze(1) - 2.0 * cross + g_sq.unsqueeze(2)).clamp_min(1e-8))
        num = torch.bmm(gap, atom.transpose(1, 2)) - (gap * g).sum(dim=-1, keepdim=True)
        qn = torch.rsqrt(gap.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12))
        sim = num * qn * key_inv

        beta = torch.nn.functional.softplus(self.clus_beta).to(atom.dtype)
        gate = torch.sigmoid(beta * (sim - self.clus_thresh)) * allowed.to(atom.dtype)
        mass = gate.sum(dim=2, keepdim=True)
        prior = torch.nn.functional.softplus(self.clus_prior).to(atom.dtype)
        ctr = (torch.bmm(gate, atom) + prior * g) / (mass + prior)
        return _rms(atom - _clean(ctr))

    def _walk(self, res, key_u, obs_val, allowed_move, allowed_harv, h_cmd0, Q):
        B, n, _ = res.shape
        dtype = res.dtype
        device = res.device
        neg = torch.finfo(dtype).min

        jgrid = torch.arange(n, device=device, dtype=dtype).view(1, 1, n)
        tau_m = torch.nn.functional.softplus(self.tau_move).to(dtype)
        tau_h = torch.nn.functional.softplus(self.tau_harv).to(dtype)
        lam = torch.nn.functional.softplus(self.lam_back).to(dtype)

        keys = _unit(res).transpose(1, 2)

        mb = self.move_key_bias(key_u).squeeze(-1).unsqueeze(1).to(dtype)
        hb = self.harv_key_bias(key_u).squeeze(-1).unsqueeze(1).to(dtype)

        z = res
        z0u = _unit(z)
        ptr = torch.arange(n, device=device, dtype=dtype).view(1, n).expand(B, n)
        remaining = res.new_ones(B, n, 1)
        content = obs_val.new_zeros(B, n, obs_val.size(-1))
        top_w = res.new_zeros(B, n, 1)

        for _ in range(self.hops + 1):
            back = -lam * torch.relu(jgrid - ptr.unsqueeze(-1) + 1.0)

            c_h = torch.bmm(_unit((z + self.harv_shift) * self.harv_metric), keys)
            s_h = (tau_h * c_h + hb + back).masked_fill(~allowed_harv, neg)
            a_h = _clean(torch.softmax(torch.cat([s_h, self.harv_null(z)], dim=2), dim=2)[:, :, :n])
            h_mass = a_h.sum(dim=2, keepdim=True)
            h_max = a_h.amax(dim=2, keepdim=True)
            got = _clean(torch.bmm(a_h, obs_val))

            c_m = torch.bmm(_unit(z * self.move_metric), keys)
            s_m = (tau_m * c_m + mb + back).masked_fill(~allowed_move, neg)
            a_m = _clean(torch.softmax(torch.cat([s_m, self.move_null(z)], dim=2), dim=2)[:, :, :n])
            m_mass = a_m.sum(dim=2, keepdim=True)
            m_max = a_m.amax(dim=2, keepdim=True)

            drift = (_unit(z) * z0u).sum(dim=2, keepdim=True)
            feats = torch.cat([h_mass, h_max, m_mass, m_max, drift], dim=2)
            p = torch.sigmoid(
                self.halt_out(torch.tanh(self.halt_in(torch.cat([h_cmd0, feats], dim=2))))
                + self.halt_kappa * (h_max - self.halt_thresh))
            w = p * remaining
            remaining = remaining * (1.0 - p)
            content = content + w * got
            top_w = torch.maximum(top_w, w)

            mu = _rms(torch.bmm(a_m, res))
            mm = m_mass.squeeze(2)
            sel = (a_m * jgrid).sum(dim=2) / mm.clamp_min(1e-6)
            ptr = mm * sel + (1.0 - mm) * ptr
            z_new = _rms(self._deflate(
                self.step_lin(mu) - self.step_alpha * z + self.step_bias, Q))
            z = _rms(m_mass * z_new + (1.0 - m_mass) * z)

        return _clean(content), _clean(1.0 - remaining), _clean(top_w)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        if types is None:
            t = torch.zeros(B, L, dtype=torch.long, device=device)
        else:
            t = types.long().clamp(0, 2)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(
            B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 1).unsqueeze(-1), obs_x, cmd_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = _clean(h_base) * valid.unsqueeze(-1).to(x.dtype)

        n = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        cmd_raw = tok_emb[:, 0::2, :]
        obs_val = self._pad_steps(tok_emb[:, 1::2, :], n).to(x.dtype)

        valid_cmd = valid[:, 0::2]
        vo = valid[:, 1::2]
        if n_pair < n:
            vo = torch.cat([vo, vo.new_zeros(B, n - n_pair)], dim=1)

        qi = torch.arange(n, device=device).view(1, n, 1)
        kj = torch.arange(n, device=device).view(1, 1, n)
        earlier = kj < qi
        allowed_move = earlier & valid_cmd.unsqueeze(1) & valid_cmd.unsqueeze(2)
        allowed_harv = allowed_move & vo.unsqueeze(1)

        key_u = self.addr_ln(cmd_raw)
        Q = self._null_frame().to(x.dtype)
        atom = _rms(self._deflate(self.addr(key_u).to(x.dtype), Q))

        live = valid_cmd.unsqueeze(-1).to(x.dtype)
        res = self._residual_atoms(atom, allowed_move, live)

        content, used, top_w = self._walk(
            res, key_u, obs_val, allowed_move, allowed_harv, h_cmd0, Q)

        render = self._transition(content, x_cmd)
        render = _clean(render) * valid_cmd.unsqueeze(-1).to(x.dtype)

        c_rms = (render.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        g = torch.sigmoid(self.harv_gate(torch.cat([h_cmd0, used, top_w, c_rms], dim=-1)))

        mem_h = self.content_to_h(render)
        fuse = torch.sigmoid(self.fuse_gate(torch.cat([h_cmd0, mem_h], dim=-1)))
        h_cmd = self.out_norm(h_cmd0 + fuse * mem_h)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + g * self.harv_out(render)
        pred[:, 0::2, :] = _clean(pred_cmd)

        pred = _clean(pred * valid.unsqueeze(-1).to(pred.dtype)).to(dtype)
        h_out = _clean(h_out * valid.unsqueeze(-1).to(h_out.dtype))
        return pred, h_out


def build(**params):
    return R7MatchingPursuitBackwalk(**params)
