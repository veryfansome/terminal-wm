import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 768

NAME = "r7_bagdiff_address_arithmetic_store"
DESCRIPTION = (
    "A causal trunk over interleaved cmd/obs embeddings paired with a BAG-DIFFERENCE ADDRESS "
    "ARITHMETIC slot store. One shared linear address map A is applied to every raw command "
    "embedding; a learned per-role offset vector is subtracted to give either a single-argument "
    "READ address or a two-argument JOINT address. A move's destination is never read out of the "
    "command by its own projection: the store first resolves which currently-OCCUPIED slot the "
    "joint address overlaps (occupancy-biased attention), then DERIVES the destination as the "
    "vector difference rho*joint - eta*a_src + b, exploiting the additivity of mean-pooled "
    "bag-of-token embeddings. The content standing at the resolved source is passed through a "
    "shared command-conditioned affine transition operator and written at the derived address; "
    "the source's occupancy is erased. Slots are addressed DNC-style by a blend of content "
    "matching and least-used allocation, with keys that move toward the address they were "
    "written at, so an address that has never been seen claims a free slot and is thereafter "
    "content-addressable. A chain of moves therefore composes by identity: whatever content sits "
    "at the resolved source is what arrives at the derived destination, to arbitrary depth. The "
    "store read is blended with the previous-observation channel and injected into the "
    "command-position prediction through an identity-initialised (D,D) readout under a sigmoid "
    "gate."
)


class R7BagDiffAddressArithmeticStore(nn.Module):
    def __init__(
        self,
        d=192,
        layers=4,
        heads=4,
        dropout=0.1,
        ffn_mult=4,
        key_d=64,
        slots=24,
        addr_hidden=256,
        gate_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        tr_init_scale=0.02,
        readout_init=0.25,
        inv_tau_read=8.0,
        inv_tau_write=8.0,
        alloc_gamma=1.0,
        kappa_read=1.0,
        kappa_src=1.5,
        kappa_write=0.5,
        match_sharp=12.0,
        match_thresh=0.5,
        scale_read_init=1.0,
        scale_move_init=1.4,
        obs_write_bias=1.5,
        move_bias=0.0,
        del_bias=1.0,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = max(8, int(key_d))
        self.slots = max(2, int(slots))
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

        self.addr_ln = nn.LayerNorm(D)
        self.addr_map = nn.Linear(D, self.key_d, bias=False)
        self.c_read = nn.Parameter(torch.zeros(self.key_d))
        self.c_move = nn.Parameter(torch.zeros(self.key_d))
        self.log_scale_read = nn.Parameter(torch.tensor(math.log(float(scale_read_init))))
        self.log_scale_move = nn.Parameter(torch.tensor(math.log(float(scale_move_init))))

        self.rho = nn.Parameter(torch.tensor(1.0))
        self.eta = nn.Parameter(torch.tensor(1.0))
        self.b_dst = nn.Parameter(torch.zeros(self.key_d))

        self.slot_key_init = nn.Parameter(torch.randn(self.slots, self.key_d))

        ah = max(32, int(addr_hidden))
        self.addr_feat = nn.Linear(D, ah)
        gh = max(16, int(gate_hidden))
        self.gate_in = nn.Linear(self.d + ah, gh)
        self.gate_out = nn.Linear(gh, 3)
        with torch.no_grad():
            self.gate_out.bias.copy_(
                torch.tensor([float(obs_write_bias), float(move_bias), float(del_bias)])
            )

        self.log_inv_tau_read = nn.Parameter(torch.tensor(math.log(float(inv_tau_read))))
        self.log_inv_tau_write = nn.Parameter(torch.tensor(math.log(float(inv_tau_write))))
        self.raw_alloc_gamma = nn.Parameter(torch.tensor(float(alloc_gamma)))
        self.kappa_read = nn.Parameter(torch.tensor(float(kappa_read)))
        self.kappa_src = nn.Parameter(torch.tensor(float(kappa_src)))
        self.kappa_write = nn.Parameter(torch.tensor(float(kappa_write)))
        self.match_sharp = nn.Parameter(torch.tensor(float(match_sharp)))
        self.match_thresh = nn.Parameter(torch.tensor(float(match_thresh)))
        self.occ_logit_decay = nn.Parameter(torch.tensor(3.0))

        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        with torch.no_grad():
            self.tr_out.weight.mul_(float(tr_init_scale))
            self.tr_out.bias.zero_()
        self.tr_mut_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, 1.0)

        self.tr_read = nn.Linear(D, D)
        with torch.no_grad():
            self.tr_read.weight.copy_(torch.eye(D) * float(readout_init))
            self.tr_read.bias.zero_()
        self.tr_read_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.tr_read_gate.bias, -1.0)

        self.read_mix = nn.Linear(self.d + 4, 2)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(self.d + 4, self.d)
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
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-6)

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

    def _transition(self, s_pre, cmd_feat):
        hin = F.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        shift = gb[..., D:]
        return s_pre * (1.0 + gamma) + shift

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _write_weights(self, keys, occ, addr, inv_tau_w, kap_w, gamma):
        sim = torch.bmm(keys, addr.unsqueeze(-1)).squeeze(-1)
        content = torch.softmax(sim * inv_tau_w + kap_w * torch.log(occ + 1e-4), dim=-1)

        used = (occ.clamp(0.0, 1.0) + 1e-6).pow(gamma)
        ordered, order = torch.sort(used, dim=-1)
        prod = torch.cumprod(ordered, dim=-1)
        prefix = torch.cat([torch.ones_like(prod[:, :1]), prod[:, :-1]], dim=-1)
        free = (1.0 - ordered).clamp_min(0.0) * prefix
        alloc = torch.zeros_like(occ).scatter(-1, order, free)

        best = (sim * occ).amax(dim=-1, keepdim=True)
        mgate = torch.sigmoid(
            self.match_sharp.clamp(0.1, 50.0) * (best - self.match_thresh)
        )
        return mgate * content + (1.0 - mgate) * alloc

    def _store(self, cmd_raw, x_cmd, obs_pad, gates, has_obs, valid_cmd, n_cmd):
        B = cmd_raw.size(0)
        dtype = x_cmd.dtype
        M = self.slots
        eps_occ = 1e-4

        z = self.addr_map(self.addr_ln(cmd_raw).to(dtype))
        solo = self._unit((z - self.c_read) * self.log_scale_read.exp().clamp(0.05, 20.0))
        joint_raw = (z - self.c_move) * self.log_scale_move.exp().clamp(0.05, 20.0)

        g_obs_all = gates[..., 0] * has_obs
        g_move_all = gates[..., 1] * valid_cmd
        g_del_all = gates[..., 2]
        mut_all = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)

        inv_tau_r = self.log_inv_tau_read.exp().clamp(0.1, 60.0)
        inv_tau_w = self.log_inv_tau_write.exp().clamp(0.1, 60.0)
        gamma_a = F.softplus(self.raw_alloc_gamma).clamp(0.2, 8.0)
        kap_r = F.softplus(self.kappa_read)
        kap_s = F.softplus(self.kappa_src)
        kap_w = F.softplus(self.kappa_write)
        lam = (0.90 + 0.099 * torch.sigmoid(self.occ_logit_decay)).to(dtype)

        keys = self._unit(self.slot_key_init).unsqueeze(0).expand(B, M, self.key_d)
        keys = keys.to(dtype).contiguous()
        vals = cmd_raw.new_zeros(B, M, D).to(dtype)
        occ = cmd_raw.new_zeros(B, M).to(dtype)

        reads = []
        sharps = []
        masses = []
        for i in range(n_cmd):
            q = solo[:, i, :]
            log_occ = torch.log(occ + eps_occ)

            sim_r = torch.bmm(keys, q.unsqueeze(-1)).squeeze(-1)
            alpha = torch.softmax(sim_r * inv_tau_r + kap_r * log_occ, dim=-1)
            reads.append(torch.bmm(alpha.unsqueeze(1), vals).squeeze(1))
            sharps.append(alpha.amax(dim=-1, keepdim=True))
            masses.append((alpha * occ).sum(dim=-1, keepdim=True))

            jt = joint_raw[:, i, :]
            sim_s = torch.bmm(keys, self._unit(jt).unsqueeze(-1)).squeeze(-1)
            beta = torch.softmax(sim_s * inv_tau_r + kap_s * log_occ, dim=-1)
            a_src = torch.bmm(beta.unsqueeze(1), keys).squeeze(1)
            v_src = torch.bmm(beta.unsqueeze(1), vals).squeeze(1)
            m_src = (beta * occ).sum(dim=-1, keepdim=True)

            a_dst = self._unit(self.rho * jt - self.eta * a_src + self.b_dst)
            moved = self._transition(v_src, x_cmd[:, i, :])
            mi = mut_all[:, i].unsqueeze(-1)
            v_dst = mi * moved + (1.0 - mi) * v_src
            g_mv = g_move_all[:, i].unsqueeze(-1) * m_src

            occ = occ * lam

            g_ob = g_obs_all[:, i].unsqueeze(-1)
            w_obs = self._write_weights(keys, occ, q, inv_tau_w, kap_w, gamma_a) * g_ob
            occ_mid = 1.0 - (1.0 - occ) * (1.0 - w_obs)
            w_mv = self._write_weights(keys, occ_mid, a_dst, inv_tau_w, kap_w, gamma_a) * g_mv

            keep = (1.0 - w_obs) * (1.0 - w_mv)
            keys = self._unit(
                keep.unsqueeze(-1) * keys
                + w_obs.unsqueeze(-1) * q.unsqueeze(1)
                + w_mv.unsqueeze(-1) * a_dst.unsqueeze(1)
            )
            vals = (
                keep.unsqueeze(-1) * vals
                + w_obs.unsqueeze(-1) * obs_pad[:, i, :].unsqueeze(1)
                + w_mv.unsqueeze(-1) * v_dst.unsqueeze(1)
            )
            vals = self._clean(vals).clamp(-1e4, 1e4)

            occ = 1.0 - (1.0 - occ_mid) * (1.0 - w_mv)
            del_w = beta * g_mv * g_del_all[:, i].unsqueeze(-1)
            occ = (occ * (1.0 - del_w)).clamp(0.0, 1.0)

        read = self._clean(torch.stack(reads, dim=1))
        sharp = torch.stack(sharps, dim=1)
        mass = torch.stack(masses, dim=1)
        return read, sharp, mass

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device

        if L == 0:
            return tok_emb.new_zeros(B, 0, D), tok_emb.new_zeros(B, 0, self.d)

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(
            B, L, device=device, dtype=torch.bool
        )

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self._clean(self.tf(x, mask=causal, src_key_padding_mask=pad_mask))
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2
        dtype = x.dtype

        cmd_raw = tok_emb[:, 0::2, :]
        obs_tok = tok_emb[:, 1::2, :]
        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]

        valid_cmd = valid[:, 0::2].to(dtype)
        valid_obs = valid[:, 1::2]
        live_pair = (valid_obs & valid[:, 0::2][:, :n_pair]).to(dtype)
        has_obs = self._pad_steps(live_pair, n_cmd)
        obs_pad = self._pad_steps(obs_tok * live_pair.unsqueeze(-1).to(obs_tok.dtype), n_cmd)
        obs_pad = obs_pad.to(dtype)

        a_feat = F.gelu(self.addr_feat(self.addr_ln(cmd_raw).to(dtype)))
        gate_h = torch.tanh(self.gate_in(torch.cat([h_cmd0, a_feat], dim=-1)))
        gates = torch.sigmoid(self.gate_out(gate_h))

        read, sharp, mass = self._store(
            cmd_raw, x_cmd, obs_pad, gates, has_obs, valid_cmd, n_cmd
        )

        if n_cmd > 1:
            prev_obs = torch.cat(
                [obs_pad.new_zeros(B, 1, D), self._pad_steps(obs_pad, n_cmd - 1)], dim=1
            )
        else:
            prev_obs = obs_pad.new_zeros(B, n_cmd, D)

        feat = torch.cat(
            [
                (read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (prev_obs.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                sharp,
                mass,
            ],
            dim=-1,
        )
        ctl = torch.cat([h_cmd0, feat], dim=-1)

        mix = torch.softmax(self.read_mix(ctl), dim=-1)
        blend = mix[:, :, 0:1] * read + mix[:, :, 1:2] * prev_obs

        mem_h = self.read_to_h(blend)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(ctl)) * mem_h)

        contrib = torch.sigmoid(self.tr_read_gate(ctl)) * self.tr_read(blend)
        contrib = self._clean(contrib * valid_cmd.unsqueeze(-1))

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred[:, 0::2, :] = self._clean(pred[:, 0::2, :] + contrib)

        pred = self._clean(pred * valid.unsqueeze(-1).to(pred.dtype))
        h_out = self._clean(h_out * valid.unsqueeze(-1).to(h_out.dtype))
        return pred, h_out


def build(**params):
    return R7BagDiffAddressArithmeticStore(**params)
