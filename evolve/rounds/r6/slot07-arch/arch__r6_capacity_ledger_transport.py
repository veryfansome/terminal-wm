import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "r6_capacity_ledger_transport"
DESCRIPTION = (
    "A location ledger resolved by CAPACITY-DEPLETED address matching. One shared feature layer "
    "over the raw command embedding is read twice under two learned additive role biases before "
    "the nonlinearity, giving each step a source address and a destination address in one common "
    "key space. A ledger carries, per earlier step, how much unconsumed content still sits at that "
    "step's destination. Step j scores every earlier step i by source/destination agreement plus "
    "the LOG of i's remaining availability, plus a learned null option and a recency tilt; the "
    "softmax over that is the step's link row. The link row pulls the earlier steps' resolved "
    "content, and the pull DEPLETES the matched availability by a learned consume amount, so a "
    "location whose content has already been moved away can no longer be matched — the erase that "
    "a move performs falls out of the capacity constraint instead of being hand-built. Content is "
    "carried as a coefficient row over raw observation embeddings, so the recurrence "
    "content(j) = take_j * link_j @ content(<j) + (1 - take_j) * obs_j resolves an arbitrarily deep "
    "chain in a single causal pass over n steps at n-by-n cost, never touching a step's own or a "
    "later observation. The resolved content is rendered through the shared command-conditioned "
    "affine transition operator and injected into the command-position prediction through a "
    "match-mass-scaled sigmoid gate and an identity-initialised (D,D) readout, and is also fused "
    "into the command hidden state. No delta-rule file/path memory, no verb quotient, no FiLM "
    "stack: the ledger is the memory."
)

_NEG = -1.0e4
_EPS = 1e-8
_CLIP = 1e4


class R6CapacityLedgerTransport(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        addr_hidden=192,
        gate_hidden=96,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        link_temp=0.25,
        recency_init=-1.0,
        avail_eps=1e-4,
        presence_scale=0.5,
        presence_margin=1.0,
        take_bias=1.0,
        write_bias=1.0,
        consume_bias=0.0,
        read_gate_bias=-0.5,
        readout_init=0.5,
        dst_init_noise=0.02,
        tr_mix_bias=-2.0,
        tr_out_std=0.01,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.addr_hidden = max(16, int(addr_hidden))
        self.avail_eps = float(avail_eps)
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
        self.addr_in = nn.Linear(D, self.addr_hidden)
        self.role_src = nn.Parameter(torch.randn(self.addr_hidden) * 0.5)
        self.role_dst = nn.Parameter(torch.randn(self.addr_hidden) * 0.5)
        self.src_key = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        self.dst_key = nn.Linear(self.addr_hidden, self.key_d, bias=False)
        with torch.no_grad():
            self.dst_key.weight.copy_(
                self.src_key.weight
                + torch.randn_like(self.src_key.weight) * float(dst_init_noise)
            )

        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-2, float(link_temp)))))
        self.log_recency = nn.Parameter(torch.tensor(float(recency_init)))
        self.null_out = nn.Linear(self.addr_hidden + self.d, 1)
        nn.init.zeros_(self.null_out.weight)
        nn.init.zeros_(self.null_out.bias)

        self.pres_proto = nn.Parameter(torch.zeros(D))
        self.pres_scale = nn.Parameter(torch.tensor(float(presence_scale)))
        self.pres_margin = nn.Parameter(torch.tensor(float(presence_margin)))

        gh = max(16, int(gate_hidden))
        self.gate_in = nn.Linear(self.addr_hidden + self.d + 1, gh)
        self.gate_out = nn.Linear(gh, 3)
        nn.init.normal_(self.gate_out.weight, std=0.02)
        with torch.no_grad():
            self.gate_out.bias.copy_(
                torch.tensor([float(take_bias), float(write_bias), float(consume_bias)])
            )

        self.read_gate_in = nn.Linear(self.addr_hidden + self.d + 2, gh)
        self.read_gate_out = nn.Linear(gh, 1)
        nn.init.normal_(self.read_gate_out.weight, std=0.02)
        nn.init.constant_(self.read_gate_out.bias, float(read_gate_bias))

        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        nn.init.normal_(self.tr_out.weight, std=float(tr_out_std))
        nn.init.zeros_(self.tr_out.bias)
        self.tr_mix = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mix.bias, float(tr_mix_bias))

        self.content_readout = nn.Linear(D, D)
        with torch.no_grad():
            self.content_readout.weight.copy_(torch.eye(D) * float(readout_init))
            self.content_readout.bias.zero_()

        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 2, self.d)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

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
    def _clean(x):
        return torch.nan_to_num(x, nan=0.0, posinf=_CLIP, neginf=-_CLIP)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = F.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _addresses(self, cmd_raw):
        z = self.addr_in(self.addr_ln(cmd_raw))
        feat = F.gelu(z)
        src = self._unit(self.src_key(F.gelu(z + self.role_src)))
        dst = self._unit(self.dst_key(F.gelu(z + self.role_dst)))
        return feat, src, dst

    def _ledger(self, src, dst, null, take, write, consume, valid_cmd):
        B, n, _ = src.shape
        device = src.device
        dtype = src.dtype

        temp = self.log_temp.exp().clamp(0.05, 5.0).to(dtype)
        scores = torch.bmm(src, dst.transpose(1, 2)) / temp
        idx = torch.arange(n, device=device)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1).clamp_min(0).to(dtype)
        scores = scores - F.softplus(self.log_recency).to(dtype) * gap.unsqueeze(0)

        eye_n = torch.eye(n, device=device, dtype=dtype)
        vc = valid_cmd.to(dtype)
        avail = src.new_zeros(B, n)
        trows = []
        routed_rows = []
        mass_cols = []
        sharp_cols = []

        for j in range(n):
            if j == 0:
                link = None
                rrow = src.new_zeros(B, n)
                mass = src.new_zeros(B, 1)
                sharp = src.new_zeros(B, 1)
            else:
                logit = scores[:, j, :j] + torch.log(avail[:, :j].clamp_min(0.0) + self.avail_eps)
                logit = logit.masked_fill(~valid_cmd[:, :j], _NEG)
                full = torch.cat([logit, null[:, j : j + 1]], dim=1)
                probs = torch.softmax(full, dim=1)
                link = probs[:, :j]
                mass = link.sum(dim=1, keepdim=True)
                sharp = link.amax(dim=1, keepdim=True)
                rrow = torch.bmm(link.unsqueeze(1), torch.stack(trows, dim=1)).squeeze(1)

            routed_rows.append(rrow)
            mass_cols.append(mass)
            sharp_cols.append(sharp)

            tj = take[:, j : j + 1] * mass
            trow = tj * rrow + (1.0 - tj) * eye_n[j].view(1, n)
            trows.append(self._clean(trow * vc[:, j : j + 1]))

            if j == 0:
                kept = avail[:, :0]
            else:
                kept = avail[:, :j] * (1.0 - consume[:, j : j + 1] * link)
            avail = torch.cat(
                [kept, write[:, j : j + 1], src.new_zeros(B, n - j - 1)], dim=1
            )

        return (
            self._clean(torch.stack(routed_rows, dim=1)),
            torch.cat(mass_cols, dim=1).unsqueeze(-1),
            torch.cat(sharp_cols, dim=1).unsqueeze(-1),
        )

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 2)
        pad_mask = key_pad.bool() if key_pad is not None else None
        if pad_mask is not None:
            valid = ~pad_mask
        else:
            valid = torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x_all = self.obs_proj(tok_emb)
        x = torch.where((t == 1).unsqueeze(-1), obs_x_all, cmd_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, cmd_x.dtype).unsqueeze(0)
        x = self.in_norm(x)
        dtype = x.dtype

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = self._clean(h_base) * valid.unsqueeze(-1).to(dtype)

        n = (L + 1) // 2
        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        cmd_raw = tok_emb[:, 0::2, :].to(dtype)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        obs_live = self._pad_steps(valid_obs, n) & valid_cmd
        obs_pad = self._pad_steps(tok_emb[:, 1::2, :].to(dtype), n) * obs_live.unsqueeze(-1).to(dtype)

        diff = obs_pad - self.pres_proto.view(1, 1, -1).to(dtype)
        d2 = diff.pow(2).mean(dim=-1)
        pres = torch.sigmoid(
            (self.pres_scale.to(dtype) * (d2 - self.pres_margin.to(dtype))).clamp(-30.0, 30.0)
        )
        pres = pres * obs_live.to(dtype)

        feat, src, dst = self._addresses(cmd_raw)
        null = self.null_out(torch.cat([feat, h_cmd0], dim=-1)).squeeze(-1)

        gates = self.gate_out(
            F.gelu(self.gate_in(torch.cat([feat, h_cmd0, pres.unsqueeze(-1)], dim=-1)))
        )
        vc = valid_cmd.to(dtype)
        take = torch.sigmoid(gates[..., 0])
        write = torch.sigmoid(gates[..., 1]) * vc
        consume = torch.sigmoid(gates[..., 2])

        route, mass, sharp = self._ledger(src, dst, null, take, write, consume, valid_cmd)
        routed = self._clean(torch.bmm(route, obs_pad))

        tmix = torch.sigmoid(self.tr_mix(h_cmd0))
        rendered = routed + tmix * (self._transition(routed, x_cmd) - routed)
        rendered = self._clean(rendered)

        rg = torch.sigmoid(
            self.read_gate_out(
                torch.tanh(self.read_gate_in(torch.cat([feat, h_cmd0, mass, sharp], dim=-1)))
            )
        )
        inject = self._clean(rg * mass * self.content_readout(rendered)) * vc.unsqueeze(-1)

        mem_h = self.read_to_h(rendered)
        fuse_in = torch.cat([h_cmd0, mem_h, mass, sharp], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred[:, 0::2, :] = self._clean(pred[:, 0::2, :] + inject)

        pred = self._clean(pred * valid.unsqueeze(-1).to(pred.dtype))
        h_out = self._clean(h_out * valid.unsqueeze(-1).to(h_out.dtype))
        return pred, h_out


def build(**params):
    return R6CapacityLedgerTransport(**params)
