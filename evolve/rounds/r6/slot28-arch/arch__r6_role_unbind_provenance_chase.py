import hashlib
import math

import torch
import torch.nn as nn

D = 768

NAME = "r6_role_unbind_provenance_chase"
DESCRIPTION = (
    "A causal transformer trunk plus a PROVENANCE CHASE built by algebraic role UNBINDING of "
    "the command token, in place of every forward delta-rule store. The command stream binds each "
    "path argument to a +/-1 role vector by elementwise product, which is its own inverse, so a "
    "single learned per-coordinate diagonal recovers 'the path this command takes FROM' and 'the "
    "path this command leaves it AT' as two vectors in one common address space; the three "
    "diagonals are initialised to the role vectors themselves and are free to move. Every command "
    "emits an IN address (a convex blend of the source-role and read-role unbindings) and an OUT "
    "address (destination-role and read-role), blended by a gate driven by the command's own coded "
    "slot mass, which counts how many paths the command names. One masked softmax over strictly "
    "earlier commands, with a learned recency penalty and a learned NULL column in cosine units, "
    "gives a single provenance matrix T: the probability that what a command's IN address refers "
    "to was last put there by an earlier command. The same T is iterated, because the answer to "
    "'who put it there' is again a command with an IN address. Mass leaves the chase at each hop "
    "with a stop probability that the same slot-mass statistic drives, so one-path commands (whose "
    "observation shows the content) absorb and two-path commands (whose observation is empty) pass "
    "through. The absorbed mass is renormalised into a convex combination of raw earlier "
    "observation embeddings and enters the command prediction through an identity-initialised "
    "(768,768) readout under a gate fed the absorbed mass, the chase sharpness and the read "
    "magnitude, and separately into the trunk state through a small gated projection. All mass "
    "moves strictly backward by an explicit mask, so a command's prediction is exactly constant in "
    "its own and every later observation."
)

_NEG = -1e9


def _rademacher(key, n):
    nbytes = max(1, (n + 7) // 8)
    digest = hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=nbytes).digest()
    return torch.tensor(
        [1.0 if (digest[i >> 3] >> (i & 7)) & 1 else -1.0 for i in range(n)],
        dtype=torch.float32,
    )


class R6RoleUnbindProvenanceChase(nn.Module):
    def __init__(
        self,
        d=192,
        layers=4,
        heads=4,
        ffn_mult=2,
        dropout=0.1,
        hops=6,
        code_off=640,
        code_dims=128,
        temp_init=0.025,
        null_init=0.20,
        recency_init=0.02,
        stop_scale=8.0,
        stop_thr=1.6,
        role_scale=2.0,
        role_thr=1.6,
        readout_init=1.5,
        mass_w=4.0,
        sharp_w=2.0,
        content_bias=-2.9,
        fuse_bias=-1.0,
        **unused,
    ):
        super().__init__()
        self.D = D
        self.d = max(2, int(d))
        self.hops = max(1, int(hops))
        off = max(0, min(int(code_off), D))
        cd = max(0, min(int(code_dims), D - off))
        self.code_off = off
        self.code_dims = cd

        ffn_h = max(self.d, int(float(ffn_mult) * self.d))
        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            max(1, int(heads)),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, max(1, int(layers)), enable_nested_tensor=False)
        self.state_norm = nn.LayerNorm(self.d)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.addr_src = nn.Parameter(self._role_diag("src"))
        self.addr_dst = nn.Parameter(self._role_diag("dst"))
        self.addr_read = nn.Parameter(self._role_diag("read"))

        mass = torch.zeros(D)
        if cd > 0:
            mass[off:off + cd] = 1.0 / float(cd)
        self.slot_mass = nn.Parameter(mass)

        self.in_gate = nn.Linear(self.d + 1, 2)
        self.out_gate = nn.Linear(self.d + 1, 2)
        for g in (self.in_gate, self.out_gate):
            nn.init.zeros_(g.weight)
            nn.init.zeros_(g.bias)
            with torch.no_grad():
                g.weight[0, self.d] = float(role_scale)
                g.bias[0] = -float(role_scale) * float(role_thr)
                g.weight[1, self.d] = -float(role_scale)
                g.bias[1] = float(role_scale) * float(role_thr)

        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.null_level = nn.Parameter(torch.tensor(float(null_init)))
        self.null_ctx = nn.Linear(self.d, 1)
        nn.init.zeros_(self.null_ctx.weight)
        nn.init.zeros_(self.null_ctx.bias)
        r0 = max(1e-4, float(recency_init))
        self.recency = nn.Parameter(torch.tensor(math.log(math.expm1(r0))))

        self.stop_scale = nn.Parameter(torch.tensor(float(stop_scale)))
        self.stop_thr = nn.Parameter(torch.tensor(float(stop_thr)))
        self.stop_ctx = nn.Linear(2 * self.d, 1)
        nn.init.zeros_(self.stop_ctx.weight)
        nn.init.zeros_(self.stop_ctx.bias)

        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(self.d + 3, 1)
        nn.init.zeros_(self.fuse_gate.weight)
        nn.init.constant_(self.fuse_gate.bias, float(fuse_bias))
        self.content_out = nn.Linear(D, D)
        with torch.no_grad():
            self.content_out.weight.copy_(torch.eye(D) * float(readout_init))
            self.content_out.bias.zero_()
        self.content_gate = nn.Linear(self.d + 3, 1)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.constant_(self.content_gate.bias, float(content_bias))
        with torch.no_grad():
            self.content_gate.weight[0, self.d] = float(mass_w)
            self.content_gate.weight[0, self.d + 1] = float(sharp_w)

    def _role_diag(self, role):
        v = torch.zeros(D)
        if self.code_dims > 0:
            v[self.code_off:self.code_off + self.code_dims] = _rademacher(
                "\x00role\x00" + role, self.code_dims)
        return v

    def _positional(self, L, device, dtype):
        half = self.d // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        ang = pos * div
        pe[:, 0:2 * half:2] = torch.sin(ang)
        pe[:, 1:2 * half:2] = torch.cos(ang)
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

    @staticmethod
    def _clean(x):
        return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

    def _provenance(self, cmd_raw, h_cmd, valid_cmd, n_cmd):
        device = cmd_raw.device
        dtype = cmd_raw.dtype

        u_src = cmd_raw * self.addr_src
        u_dst = cmd_raw * self.addr_dst
        u_read = cmd_raw * self.addr_read
        slot = (cmd_raw * cmd_raw * self.slot_mass).sum(dim=-1, keepdim=True)
        slot = self._clean(slot)

        gate_in = torch.cat([h_cmd, slot], dim=-1)
        gi = torch.sigmoid(self.in_gate(gate_in))
        go = torch.sigmoid(self.out_gate(gate_in))
        q = self._unit(gi[..., 0:1] * u_src + gi[..., 1:2] * u_read)
        k = self._unit(go[..., 0:1] * u_dst + go[..., 1:2] * u_read)

        cos = self._clean(torch.bmm(q, k.transpose(1, 2)))
        idx = torch.arange(n_cmd, device=device)
        gap = (idx.view(1, n_cmd, 1) - idx.view(1, 1, n_cmd) - 1).clamp_min(0).to(dtype)
        score = cos - torch.nn.functional.softplus(self.recency).to(dtype) * gap

        temp = self.log_temp.exp().clamp(0.005, 2.0).to(dtype)
        logits = score / temp
        allow = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0)
        allow = allow & valid_cmd.unsqueeze(1) & valid_cmd.unsqueeze(2)
        logits = logits.masked_fill(~allow, _NEG)
        null = (self.null_level.to(dtype) + self.null_ctx(h_cmd)) / temp
        w = torch.softmax(torch.cat([logits, null], dim=2), dim=2)[:, :, :n_cmd]
        w = torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w * allow.to(dtype), slot

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        if pad_mask is not None:
            valid = ~pad_mask
        else:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self._clean(self.tf(x, mask=causal, src_key_padding_mask=pad_mask))
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        dtype = h_base.dtype

        hn = self.state_norm(h_base)
        h_cmd = hn[:, 0::2, :]
        h_obs = self._pad_steps(hn[:, 1::2, :], n_cmd)
        valid_cmd = valid[:, 0::2]
        valid_obs = self._pad_steps(valid[:, 1::2], n_cmd) & valid_cmd

        cmd_raw = self._clean(tok_emb[:, 0::2, :].to(dtype))
        obs_raw = self._pad_steps(self._clean(tok_emb[:, 1::2, :].to(dtype)), n_cmd)
        obs_raw = obs_raw * valid_obs.unsqueeze(-1).to(dtype)

        trans, slot = self._provenance(cmd_raw, h_cmd, valid_cmd, n_cmd)

        stop_logit = self.stop_scale.to(dtype) * (self.stop_thr.to(dtype) - slot)
        stop_logit = stop_logit + self.stop_ctx(torch.cat([h_cmd, h_obs], dim=-1))
        stop = torch.sigmoid(stop_logit).squeeze(-1) * valid_obs.to(dtype)
        stop_row = stop.unsqueeze(1)

        alive = trans
        acc = alive * stop_row
        alive = alive * (1.0 - stop_row)
        for _ in range(self.hops - 1):
            alive = torch.bmm(alive, trans)
            acc = acc + alive * stop_row
            alive = alive * (1.0 - stop_row)
        acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

        mass = acc.sum(dim=2, keepdim=True)
        sharp = acc.amax(dim=2, keepdim=True)
        evidence = self._clean(torch.bmm(acc, obs_raw))
        content = self._clean(evidence / mass.clamp_min(0.05))
        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass.clamp(0.0, 1.0), sharp.clamp(0.0, 1.0), rms], dim=-1)

        gate_in = torch.cat([h_cmd, feat], dim=-1)
        mem_h = self.read_to_h(content)
        fused = self.out_norm(h_base[:, 0::2, :] + torch.sigmoid(self.fuse_gate(gate_in)) * mem_h)

        contrib = torch.sigmoid(self.content_gate(gate_in)) * self.content_out(evidence)
        contrib = self._clean(contrib) * valid_cmd.unsqueeze(-1).to(dtype)

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = fused
        pred = self.head(h_out).clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib

        pred = self._clean(pred * valid.unsqueeze(-1).to(pred.dtype))
        h_out = self._clean(h_out * valid.unsqueeze(-1).to(h_out.dtype))
        return pred, h_out


def build(**params):
    return R6RoleUnbindProvenanceChase(**params)
