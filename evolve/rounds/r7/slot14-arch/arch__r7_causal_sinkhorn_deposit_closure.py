import math

import torch
import torch.nn as nn

D = 768
NEG = -1.0e9

NAME = "r7_causal_sinkhorn_deposit_closure"
DESCRIPTION = (
    "A reference-shaped causal trunk (d=192, 4 layers, ffn 4x) whose only memory is a CAUSAL "
    "SINKHORN DEPOSIT-CLOSURE over command positions. Every command gets a pickup query and a "
    "deposit key from one shared GELU reducer on its raw 768-d embedding; the low-rank asymmetric "
    "bilinear score between pickup_i and deposit_j (j<i) is turned into a soft PARTIAL MATCHING by "
    "a dustbin-augmented Sinkhorn in log space, alternating row normalization with a PREFIX "
    "(logcumsumexp) column normalization so column mass is only ever shared with earlier rows and "
    "the layer stays strictly causal. The row dustbin is the origination mass: what a command "
    "keeps of its own observation instead of inheriting. Deposits are then resolved by exact "
    "transitive closure of the strictly-lower-triangular matching via repeated squaring "
    "(dep = sum_t P^t (own * obs), computed in ceil(log2 n) steps), and the value pulled at a "
    "command is P @ dep, which depends only on strictly earlier observations. The pulled content "
    "passes through a shared command-conditioned affine render operator (also exposed as "
    "transition_from_emb) and enters the prediction twice: fused into the command hidden state "
    "through a gated projection, and added through a zero-init (D,D) readout. No delta-rule "
    "associative memory, no verb quotient, no system FiLM, no recency prior."
)


class R7CausalSinkhornDepositClosure(nn.Module):
    def __init__(
        self,
        d=192,
        layers=4,
        heads=4,
        ffn_mult=4,
        dropout=0.1,
        link_hidden=192,
        link_d=64,
        sink_iters=2,
        sink_tau=1.0,
        null_cap=4.0,
        closure_pow=6,
        read_floor=0.2,
        tr_hidden=192,
        tr_gscale=0.5,
        obs_info_bias=2.0,
        fuse_bias=-1.0,
        route_bias=0.0,
        null_row_bias=0.0,
        null_col_bias=0.0,
        overlap_init=0.25,
        route_init=1e-3,
        **unused,
    ):
        super().__init__()
        self.D = D
        self.d = max(8, int(d))
        self.layers = max(1, int(layers))
        self.link_hidden = max(16, int(link_hidden))
        self.link_d = max(4, int(link_d))
        self.sink_iters = max(0, int(sink_iters))
        self.sink_tau = max(1e-3, float(sink_tau))
        self.null_cap = float(null_cap)
        self.closure_pow = max(1, int(closure_pow))
        self.read_floor = min(1.0, max(1e-3, float(read_floor)))
        self.tr_gscale = float(tr_gscale)
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

        self.link_ln = nn.LayerNorm(D)
        self.link_in = nn.Linear(D, self.link_hidden)
        self.link_pickup = nn.Linear(self.link_hidden, self.link_d, bias=False)
        self.link_deposit = nn.Linear(self.link_hidden, self.link_d, bias=False)
        self.link_overlap = nn.Linear(self.link_hidden, self.link_d, bias=False)
        self.overlap_scale = nn.Parameter(torch.full((1,), float(overlap_init)))
        self.pickup_bias = nn.Linear(self.link_hidden, 1)
        self.deposit_bias = nn.Linear(self.link_hidden, 1)
        self.null_pickup = nn.Linear(self.link_hidden, 1)
        self.null_deposit = nn.Linear(self.link_hidden, 1)
        nn.init.zeros_(self.pickup_bias.weight)
        nn.init.zeros_(self.pickup_bias.bias)
        nn.init.zeros_(self.deposit_bias.weight)
        nn.init.zeros_(self.deposit_bias.bias)
        nn.init.constant_(self.null_pickup.bias, float(null_row_bias))
        nn.init.constant_(self.null_deposit.bias, float(null_col_bias))

        self.obs_info = nn.Linear(self.d, 1)
        nn.init.constant_(self.obs_info.bias, float(obs_info_bias))

        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)

        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(self.d + 3, self.d)
        self.route_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.fuse_gate.bias, float(fuse_bias))
        nn.init.constant_(self.route_gate.bias, float(route_bias))

        self.route_out = nn.Linear(D, D)
        nn.init.normal_(self.route_out.weight, std=max(1e-8, float(route_init)))
        nn.init.zeros_(self.route_out.bias)

        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half))
        )
        wave = pos * div
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        n_even = pe[:, 0::2].size(1)
        n_odd = pe[:, 1::2].size(1)
        pe[:, 0::2] = torch.sin(wave[:, :n_even])
        pe[:, 1::2] = torch.cos(wave[:, :n_odd])
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

    def _matching(self, cmd_raw, valid_cmd):
        B, n, _ = cmd_raw.shape
        device = cmd_raw.device
        dtype = cmd_raw.dtype

        z = torch.nn.functional.gelu(self.link_in(self.link_ln(cmd_raw)))
        pick = self.link_pickup(z)
        dep = self.link_deposit(z)
        ov = self._unit(self.link_overlap(z))

        score = torch.bmm(pick, dep.transpose(1, 2)) / math.sqrt(float(self.link_d))
        score = score + self.overlap_scale.to(dtype) * torch.bmm(ov, ov.transpose(1, 2))
        score = score + self.pickup_bias(z) + self.deposit_bias(z).transpose(1, 2)

        idx = torch.arange(n, device=device)
        strictly_earlier = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        allowed = strictly_earlier & valid_cmd.unsqueeze(1) & valid_cmd.unsqueeze(2)

        f = torch.nan_to_num(score / self.sink_tau, nan=0.0, posinf=1e4, neginf=-1e4)
        f = f.clamp(-30.0, 30.0).masked_fill(~allowed, NEG)

        cap = self.null_cap
        row_dustbin = cap * torch.tanh(self.null_pickup(z))
        col_dustbin = (cap * torch.tanh(self.null_deposit(z))).transpose(1, 2)

        for _ in range(self.sink_iters):
            prefix = torch.logcumsumexp(f, dim=1)
            prefix = torch.logaddexp(prefix, col_dustbin.expand_as(prefix))
            f = (f - prefix).masked_fill(~allowed, NEG)
            row = torch.logsumexp(f, dim=2, keepdim=True)
            row = torch.logaddexp(row, row_dustbin)
            f = (f - row).masked_fill(~allowed, NEG)

        if self.sink_iters == 0:
            row = torch.logsumexp(f, dim=2, keepdim=True)
            row = torch.logaddexp(row, row_dustbin)
            f = (f - row).masked_fill(~allowed, NEG)

        P = torch.exp(f.clamp(max=0.0))
        P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        P = P * allowed.to(P.dtype)
        return P

    def _closure(self, P, source, n):
        steps = min(self.closure_pow, max(1, int(math.ceil(math.log2(max(2, n))))))
        acc = source
        pow_mat = P
        for t in range(steps):
            acc = acc + torch.bmm(pow_mat, acc)
            acc = torch.nan_to_num(acc, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            if t + 1 < steps:
                pow_mat = torch.bmm(pow_mat, pow_mat)
                pow_mat = torch.nan_to_num(pow_mat, nan=0.0, posinf=0.0, neginf=0.0)
        return acc

    def _render(self, content, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        shift = gb[..., D:]
        return content * (1.0 + gamma) + shift

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._render(s_pre, cmd_feat)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        if pad_mask is not None:
            valid = ~pad_mask
        else:
            valid = torch.ones(B, L, device=device, dtype=torch.bool)

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

        cmd_feat = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = self._pad_steps(h_base[:, 1::2, :], n_cmd)
        cmd_raw = tok_emb[:, 0::2, :]
        obs_raw = self._pad_steps(tok_emb[:, 1::2, :], n_cmd)
        valid_cmd = valid[:, 0::2]
        valid_obs = self._pad_steps(valid[:, 1::2].unsqueeze(-1), n_cmd).squeeze(-1)

        P = self._matching(cmd_raw, valid_cmd)
        row_mass = P.sum(dim=2).clamp(0.0, 1.0)
        live = (valid_cmd & valid_obs).to(dtype)
        origination = (1.0 - row_mass) * torch.sigmoid(self.obs_info(h_obs)).squeeze(-1) * live

        source = origination.unsqueeze(-1).to(dtype) * obs_raw
        deposit = self._closure(P.to(dtype), source, n_cmd)
        pulled = torch.bmm(P.to(dtype), deposit)
        pulled = pulled / row_mass.clamp_min(self.read_floor).unsqueeze(-1).to(dtype)
        pulled = torch.nan_to_num(pulled, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        rendered = self._render(pulled.to(x.dtype), cmd_feat)
        rendered = torch.nan_to_num(rendered, nan=0.0, posinf=1e4, neginf=-1e4)

        feat = torch.cat(
            [
                (pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(x.dtype),
                P.amax(dim=2, keepdim=True).to(x.dtype),
                row_mass.unsqueeze(-1).to(x.dtype),
            ],
            dim=-1,
        )
        gate_in = torch.cat([h_cmd0, feat], dim=-1)

        mem_h = self.read_to_h(rendered.to(x.dtype))
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(gate_in)) * mem_h)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd

        pred = self.head(h_out).clone()
        contrib = torch.sigmoid(self.route_gate(gate_in)) * self.route_out(rendered.to(x.dtype))
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        pred_cmd = pred[:, 0::2, :] + contrib.to(pred.dtype)
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(
            pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4
        )
        h_out = torch.nan_to_num(
            h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4
        )
        return pred, h_out


def build(**params):
    return R7CausalSinkhornDepositClosure(**params)
