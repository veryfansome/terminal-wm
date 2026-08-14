import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_retrieval_composition_renderer import (
    R22RetrievalCompositionRenderer,
)

D = 768

NAME = "r24_pointer_unbind_transport"
DESCRIPTION = (
    "The r22 retrieval-composition arch plus a POINTER-UNBINDING transport table. Each "
    "command's raw pooled embedding is mapped by one affine head into an address code and "
    "quotiented against a learned orthonormal boilerplate basis, then unit-normalized. A table "
    "of items is carried along the sequence: item j holds the raw observation embedding of "
    "step j as an immutable value and a MUTABLE unit address, initialized to step j's own "
    "address code. At every command the model soft-matches its address code against the "
    "addresses currently held by strictly-earlier items, with a learned temperature, a learned "
    "existence bias and a null slot that absorbs mass when nothing matches; the matched value "
    "is read out. The matched address is then removed from the command code by Gram-Schmidt "
    "projection under a learned unbinding strength, and the unit residual is used as the new "
    "address of the matched item, dragged there in proportion to a learned per-command move "
    "gate times the match weight. Values are never rewritten, only re-addressed. The read is "
    "injected into the command-position prediction through a gated zero-init (D,D) readout, so "
    "the forward is identical to r22 at init."
)


class R24PointerUnbindTransport(R22RetrievalCompositionRenderer):
    def __init__(
        self,
        ptr_dim=96,
        ptr_basis=6,
        ptr_tau=8.0,
        ptr_move_bias=0.0,
        ptr_null_init=0.0,
        ptr_gate_bias=-2.0,
        ptr_unbind_init=2.0,
        **params,
    ):
        super().__init__(**params)
        self.ptr_dim = max(16, int(ptr_dim))
        self.ptr_basis_n = max(1, int(ptr_basis))
        self.ptr_ln = nn.LayerNorm(D)
        self.ptr_addr = nn.Linear(D, self.ptr_dim)
        self.ptr_codebook = nn.Parameter(torch.randn(self.ptr_basis_n, self.ptr_dim) * 0.2)
        self.ptr_log_tau = nn.Parameter(torch.tensor(math.log(max(1e-2, float(ptr_tau)))))
        self.ptr_null = nn.Parameter(torch.tensor(float(ptr_null_init)))
        self.ptr_unbind = nn.Parameter(torch.tensor(float(ptr_unbind_init)))
        self.ptr_move = nn.Linear(self.d, 1)
        nn.init.constant_(self.ptr_move.bias, float(ptr_move_bias))
        self.ptr_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.ptr_gate.bias, float(ptr_gate_bias))
        self.ptr_out = nn.Linear(D, D)
        nn.init.zeros_(self.ptr_out.weight)
        nn.init.zeros_(self.ptr_out.bias)

    @staticmethod
    def _ptr_unit(x):
        n2 = x.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-4)
        return x * torch.rsqrt(n2)

    def _ptr_quotient_basis(self):
        vs = []
        for i in range(self.ptr_basis_n):
            v = self.ptr_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum().clamp_min(1e-6))
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _pointer_route(self, g, mu, ex, vals, n_cmd):
        B = g.size(0)
        dtype = g.dtype
        tau = torch.exp(self.ptr_log_tau).clamp(0.05, 64.0).to(dtype)
        lam = torch.sigmoid(self.ptr_unbind).to(dtype)
        log_ex = torch.log(ex.to(dtype).clamp_min(1e-4))
        null = self.ptr_null.to(dtype).view(1, 1).expand(B, 1)

        addr = g
        reads = [vals.new_zeros(B, D)]
        mass = [g.new_zeros(B, 1)]
        peak = [g.new_zeros(B, 1)]

        for i in range(1, n_cmd):
            gi = g[:, i, :]
            past = addr[:, :i, :]
            sim = torch.bmm(past, gi.unsqueeze(2)).squeeze(2)
            logits = tau * sim + log_ex[:, :i]
            w = torch.softmax(torch.cat([logits, null], dim=1), dim=1)[:, :i]

            reads.append(torch.bmm(w.unsqueeze(1), vals[:, :i, :]).squeeze(1))
            mass.append(w.sum(dim=1, keepdim=True))
            peak.append(w.amax(dim=1, keepdim=True))

            src = self._ptr_unit(torch.bmm(w.unsqueeze(1), past).squeeze(1))
            proj = (gi * src).sum(dim=-1, keepdim=True)
            dst = self._ptr_unit(gi - lam * proj * src)

            drag = (mu[:, i].unsqueeze(-1) * w).unsqueeze(-1)
            moved = self._ptr_unit(past + drag * (dst.unsqueeze(1) - past))
            addr = torch.cat([moved, addr[:, i:, :]], dim=1)

        read = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        feat = torch.cat([torch.stack(mass, dim=1), torch.stack(peak, dim=1)], dim=-1)
        return read, torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=0.0)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        device = tok_emb.device
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = tok_emb[:, 0::2, :]
        obs_raw = tok_emb[:, 1::2, :]
        h_cmd = h_out[:, 0::2, :].to(dtype)

        g = self._quotient(self.ptr_addr(self.ptr_ln(cmd_raw)), self._ptr_quotient_basis()).to(dtype)
        mu = torch.sigmoid(self.ptr_move(h_cmd)).squeeze(-1).to(dtype)

        live = (valid_obs & valid_cmd[:, :n_pair]).to(dtype)
        ex = self._pad_steps(live * (1.0 - mu[:, :n_pair]), n_cmd)
        vals = self._pad_steps(obs_raw * valid_obs.unsqueeze(-1).to(dtype), n_cmd)

        read, feat2 = self._pointer_route(g, mu, ex, vals, n_cmd)
        rms = (read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([feat2, rms.to(dtype)], dim=-1)

        gate = torch.sigmoid(self.ptr_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate * self.ptr_out(read)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R24PointerUnbindTransport(**params)
