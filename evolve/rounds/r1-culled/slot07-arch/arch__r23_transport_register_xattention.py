import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r23_transport_register_xattention"
DESCRIPTION = (
    "The r22 prefix-content cross-attention arch + a SLOT-CONTENT TRANSPORT REGISTER: a "
    "[slots+null, items] occupancy matrix initialised to the identity over the fully-observed "
    "prefix pairs, driven at every valid command position by a rank-one soft-permutation update "
    "R <- R*(1-g*alpha)*(1-g*o*omega) + (g*omega) (alpha^T R), where alpha/omega are cosine "
    "address distributions over earlier slots plus a null sink and g/o are a move and an "
    "overwrite gate. The pre-write read alpha^T R is a distribution over the SHOWN observations "
    "whose weights are a product of per-hop transports, decoded back to 768-d by a small-identity "
    "readout under a sigmoid gate. The write is gated by valid_cmd only, so a command whose "
    "observation is masked still transports content; the read is masked to slots strictly earlier "
    "than the current command, so a position can never reach its own or a later observation."
)


class R23TransportRegisterXAttention(R22PrefixContentXAttention):
    def __init__(
        self,
        reg_dim=64,
        reg_hidden=192,
        reg_move_bias=-3.0,
        reg_overwrite_bias=0.0,
        reg_gate_bias=-3.0,
        reg_out_scale=0.25,
        reg_temp=10.0,
        reg_clamp=4.0,
        **params,
    ):
        super().__init__(**params)
        # New modules are constructed AFTER the entire inherited __init__ so the inherited
        # parameters draw the identical init-RNG stream; reordering breaks bit-identity at init.
        self.reg_dim = max(8, int(reg_dim))
        self.reg_clamp = float(reg_clamp)
        hq = max(32, int(reg_hidden))
        self.reg_kbody = nn.Linear(D, hq)
        self.reg_qbody = nn.Linear(D + self.d, hq)
        self.reg_slot_k = nn.Linear(hq, self.reg_dim, bias=False)
        self.reg_qsrc = nn.Linear(hq, self.reg_dim, bias=False)
        self.reg_qdst = nn.Linear(hq, self.reg_dim, bias=False)
        self.reg_null = nn.Parameter(torch.randn(self.reg_dim) * 0.2)
        self.reg_ctl = nn.Linear(hq, 2)
        nn.init.zeros_(self.reg_ctl.weight)
        with torch.no_grad():
            self.reg_ctl.bias.copy_(
                torch.tensor([float(reg_move_bias), float(reg_overwrite_bias)])
            )
        self.reg_logit_temp = nn.Parameter(torch.tensor(math.log(max(1.0, float(reg_temp)))))
        self.reg_out = nn.Linear(D, D)
        with torch.no_grad():
            self.reg_out.weight.copy_(torch.eye(D) * float(reg_out_scale))
            self.reg_out.bias.zero_()
        self.reg_gate = nn.Linear(self.d + 1, 1)
        nn.init.constant_(self.reg_gate.bias, float(reg_gate_bias))

    def _addresses(self, qb, keys, allowed, proj):
        q = self._unit(proj(qb))
        temp = torch.exp(self.reg_logit_temp.clamp(0.0, 4.0)).to(q.dtype)
        scores = torch.bmm(q, keys.transpose(1, 2)) * temp
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=2)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        n_cmd = (L + 1) // 2
        S = L // 2
        if S < 1 or n_cmd < 2:
            return pred, h_out

        device = tok_emb.device
        hdtype = h_out.dtype

        if key_pad is not None:
            valid = ~key_pad.bool()
        else:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        live = valid_cmd[:, :S] & valid_obs

        cmd_raw = tok_emb[:, 0::2, :].to(hdtype)
        h_cmd = h_out[:, 0::2, :]

        kb = torch.nn.functional.gelu(self.reg_kbody(cmd_raw[:, :S, :]))
        qb = torch.nn.functional.gelu(self.reg_qbody(torch.cat([cmd_raw, h_cmd], dim=-1)))

        kslot = self._unit(self.reg_slot_k(kb))
        knull = self._unit(self.reg_null).view(1, 1, -1).to(kslot.dtype).expand(B, 1, self.reg_dim)
        keys = torch.cat([kslot, knull], dim=1)

        ci = torch.arange(n_cmd, device=device).unsqueeze(1)
        sj = torch.arange(S, device=device).unsqueeze(0)
        allowed_slot = (sj < ci).unsqueeze(0) & live.unsqueeze(1)
        allowed = torch.cat(
            [allowed_slot, torch.ones(B, n_cmd, 1, dtype=torch.bool, device=device)], dim=2
        )

        a_all = self._addresses(qb, keys, allowed, self.reg_qsrc)
        w_all = self._addresses(qb, keys, allowed, self.reg_qdst)

        ctl = torch.sigmoid(self.reg_ctl(qb)).to(a_all.dtype)
        g_all = ctl[..., 0:1] * valid_cmd.unsqueeze(-1).to(a_all.dtype)
        o_all = ctl[..., 1:2]

        rdt = a_all.dtype
        row_keep = torch.ones(1, S + 1, 1, device=device, dtype=rdt)
        row_keep[:, S, :] = 0.0
        R = torch.zeros(B, S + 1, S, device=device, dtype=rdt)
        R[:, :S, :] = torch.eye(S, device=device, dtype=rdt).unsqueeze(0) * live.unsqueeze(-1).to(rdt)

        reads = []
        for i in range(n_cmd):
            a = a_all[:, i, :]
            w = w_all[:, i, :]
            g = g_all[:, i, :]
            o = o_all[:, i, :]
            c = torch.bmm(a.unsqueeze(1), R).squeeze(1)
            reads.append(c)
            keep = (1.0 - g * a) * (1.0 - g * o * w)
            R = R * keep.unsqueeze(-1) + (g * w).unsqueeze(-1) * c.unsqueeze(1)
            R = R * row_keep
            R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, self.reg_clamp)

        W = torch.stack(reads, dim=1)

        V = tok_emb[:, 1::2, :] * live.unsqueeze(-1).to(tok_emb.dtype)
        out = torch.bmm(W.to(V.dtype), V)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        feat = (out.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(hdtype)
        gate = torch.sigmoid(self.reg_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate.to(out.dtype) * self.reg_out(out.to(self.reg_out.weight.dtype)).to(out.dtype)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R23TransportRegisterXAttention(**params)
