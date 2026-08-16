import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "r7_slot_permutation_transport_bank"
DESCRIPTION = (
    "A location->content SLOT BANK updated by soft permutation operators. Every command is "
    "unbound twice by a learned elementwise sign-mask over the raw command token and pushed "
    "through ONE SHARED projection into a source address and a destination address, then "
    "cleaned up against a learned slot codebook with a softmax at learned temperature, so the "
    "same path reaches the same slot whether it appears in `cat P` or in `mv P Q`. Each step "
    "reads the content held at its source slot, and a move gate applies the rank-two operator "
    "M <- M + b (f(c_src) - c_dst)^T - a c_src^T, which is exactly a permutation matrix acting "
    "on the slot axis when the two addresses are one-hot, so a chain of moves COMPOSES into the "
    "product of its operators; an observation gate then delta-rule writes the step's own "
    "observation at the source slot AFTER the read, keeping the recurrence strictly causal. The "
    "prediction is a four-way convex mixture of the trunk head, the slot read (identity at "
    "init, so a retrieved content can be emitted verbatim), the previous observation, and a "
    "command-similarity lookup over earlier observations. No FiLM, no delta-rule file/path "
    "memories, no per-step transition MLP."
)


class R7SlotPermutationTransportBank(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        ffn_mult=2,
        dropout=0.1,
        key_d=64,
        n_slots=64,
        code_dims=128,
        code_prior=0.25,
        addr_temp=10.0,
        write_cap=20.0,
        xform_rank=80,
        lookup_d=96,
        lookup_temp=6.0,
        decay_init=0.995,
        move_bias=-1.0,
        store_bias=1.5,
        mix_trunk_bias=1.5,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = D
        self.d = max(8, int(d))
        self.layers = max(1, int(layers))
        self.key_d = max(4, int(key_d))
        self.n_slots = max(2, int(n_slots))
        self.xform_rank = max(4, int(xform_rank))
        self.lookup_d = max(4, int(lookup_d))
        self.write_cap = max(1.0, float(write_cap))
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

        self.addr_norm = nn.LayerNorm(D)
        cd = max(1, min(int(code_dims), D))
        prior = torch.full((D,), float(code_prior))
        prior[D - cd:] = 1.0
        self.g_src = nn.Parameter(prior.clone())
        self.g_dst = nn.Parameter(prior.clone() * 0.999 + torch.randn(D) * 0.02)
        self.addr_proj = nn.Linear(D, self.key_d, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(self.n_slots, self.key_d))
        self.addr_temp = nn.Parameter(torch.tensor(math.log(math.expm1(max(1e-3, float(addr_temp))))))

        self.move_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.move_gate.bias, float(move_bias))
        self.store_gate = nn.Linear(3 * self.d, 1)
        nn.init.constant_(self.store_gate.bias, float(store_bias))

        self.xf_cmd = nn.Linear(self.d, self.xform_rank)
        self.xf_val = nn.Linear(D, self.xform_rank, bias=False)
        self.xf_out = nn.Linear(self.xform_rank, D)
        nn.init.zeros_(self.xf_out.weight)
        nn.init.zeros_(self.xf_out.bias)

        self.look_q = nn.Linear(D, self.lookup_d, bias=False)
        self.look_k = nn.Linear(D, self.lookup_d, bias=False)
        self.look_null = nn.Linear(D, 1)
        nn.init.zeros_(self.look_null.bias)
        self.look_temp = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(1e-3, float(lookup_temp)))))
        )

        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 4, self.d)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)
        self.read_out = nn.Linear(D, D)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)
        self.mix = nn.Linear(self.d + 4, 4)
        nn.init.zeros_(self.mix.weight)
        with torch.no_grad():
            self.mix.bias.copy_(torch.tensor([float(mix_trunk_bias), 0.0, 0.0, 0.0]))

        lo, hi = 0.95, 0.9999
        dv = min(max(float(decay_init), lo + 1e-4), hi - 1e-4)
        frac = (dv - lo) / (hi - lo)
        self.decay_lo = lo
        self.decay_span = hi - lo
        self.logit_decay = nn.Parameter(torch.tensor(math.log(frac / (1.0 - frac))))

    def _positional(self, L, device, dtype):
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        idx = torch.arange(self.d, device=device, dtype=dtype).unsqueeze(0)
        div = torch.exp(-(math.log(10000.0) / max(1.0, float(self.d))) * (2.0 * torch.floor(idx / 2.0)))
        ang = pos * div
        even = (torch.arange(self.d, device=device) % 2 == 0).unsqueeze(0)
        return torch.where(even, torch.sin(ang), torch.cos(ang))

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _rms(x):
        return (x.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()

    def _slot_address(self, u, gain):
        v = self._unit(self.addr_proj(u * gain))
        k = self._unit(self.slot_keys).to(v.dtype)
        logits = torch.matmul(v, k.transpose(0, 1)) * F.softplus(self.addr_temp).to(v.dtype)
        return torch.softmax(logits, dim=-1)

    def _lookup(self, u, obs_pad, valid_cmd, n_cmd):
        q = self._unit(self.look_q(u))
        k = self._unit(self.look_k(u))
        scores = torch.matmul(q, k.transpose(1, 2)) * F.softplus(self.look_temp).to(q.dtype)
        idx = torch.arange(n_cmd, device=u.device)
        allow = (idx.view(-1, 1) > idx.view(1, -1)).unsqueeze(0) & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allow, torch.finfo(scores.dtype).min)
        full = torch.cat([scores, self.look_null(u)], dim=2)
        w = torch.softmax(full, dim=2)[:, :, :n_cmd]
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.bmm(w.to(obs_pad.dtype), obs_pad)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

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

        cmd_raw = tok_emb[:, 0::2, :]
        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]
        obs_xf = obs_x[:, 1::2, :]
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        u = self.addr_norm(cmd_raw)
        a_src = self._slot_address(u, self.g_src)
        a_dst = self._slot_address(u, self.g_dst)

        move = torch.sigmoid(self.move_gate(x_cmd)).squeeze(-1) * valid_cmd.to(x.dtype)
        if n_pair:
            store_in = torch.cat([x_cmd[:, :n_pair, :], h_obs, obs_xf], dim=-1)
            store = torch.sigmoid(self.store_gate(store_in)).squeeze(-1)
            store = store * (valid_cmd[:, :n_pair] & valid_obs).to(x.dtype)
        else:
            store = x.new_zeros(B, 0)

        xf_c = self.xf_cmd(x_cmd)
        lam = (self.decay_lo + self.decay_span * torch.sigmoid(self.logit_decay)).to(dtype)

        mem = tok_emb.new_zeros(B, self.n_slots, D)
        reads = []
        for i in range(n_cmd):
            a = a_src[:, i, :].to(dtype)
            b = a_dst[:, i, :].to(dtype)
            c_src = torch.bmm(a.unsqueeze(1), mem).squeeze(1)
            reads.append(c_src)
            c_dst = torch.bmm(b.unsqueeze(1), mem).squeeze(1)
            na = a.pow(2).sum(dim=-1, keepdim=True)
            nb = b.pow(2).sum(dim=-1, keepdim=True)
            sa = (1.0 / (na + 0.02)).clamp(max=self.write_cap)
            sb = (1.0 / (nb + 0.02)).clamp(max=self.write_cap)
            overlap = ((a * b).sum(dim=-1, keepdim=True) * torch.rsqrt(na * nb + 1e-12)).clamp(0.0, 1.0)
            gi = move[:, i].unsqueeze(-1).to(dtype)
            moved = c_src + self.xf_out(torch.tanh(self.xf_val(c_src.to(x.dtype)) + xf_c[:, i, :])).to(dtype)
            put = gi * sb * (moved - c_dst)
            take = -gi * sa * (1.0 - overlap) * c_src
            addr_pair = torch.stack([b, a], dim=2)
            corr_pair = torch.stack([put, take], dim=1)
            mem = lam * mem + torch.bmm(addr_pair, corr_pair)
            if i < n_pair:
                c_now = torch.bmm(a.unsqueeze(1), mem).squeeze(1)
                wi = store[:, i].unsqueeze(-1).to(dtype)
                upd = wi * sa * (obs_tok[:, i, :] - c_now)
                mem = mem + torch.bmm(a.unsqueeze(2), upd.unsqueeze(1))
            mem = mem.clamp(-1e4, 1e4)
        read = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

        obs_masked = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), obs_masked[:, :n_cmd - 1, :]], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)
        if n_pair >= n_cmd:
            obs_pad = obs_masked[:, :n_cmd, :]
        else:
            obs_pad = torch.cat(
                [obs_masked, tok_emb.new_zeros(B, n_cmd - n_pair, D)], dim=1
            )

        look = torch.nan_to_num(
            self._lookup(u, obs_pad, valid_cmd, n_cmd), nan=0.0, posinf=1e4, neginf=-1e4
        )

        feats = torch.cat(
            [
                self._rms(read.to(x.dtype)),
                self._rms(prev_obs.to(x.dtype)),
                self._rms(look.to(x.dtype)),
                a_src.amax(dim=-1, keepdim=True).to(x.dtype),
            ],
            dim=-1,
        )

        mem_h = self.read_to_h(read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, feats], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out)
        trunk_cmd = pred[:, 0::2, :]

        mix = torch.softmax(self.mix(torch.cat([h_cmd0, feats], dim=-1)), dim=-1).to(dtype)
        read_channel = read + self.read_out(read.to(x.dtype)).to(dtype)
        pred_cmd = (
            mix[..., 0:1] * trunk_cmd
            + mix[..., 1:2] * read_channel
            + mix[..., 2:3] * prev_obs
            + mix[..., 3:4] * look
        )

        pred = pred.clone()
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)
        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R7SlotPermutationTransportBank(**params)
