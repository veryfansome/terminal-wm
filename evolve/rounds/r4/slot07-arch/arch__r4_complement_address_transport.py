import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r22_retrieval_composition_renderer import (
    R22RetrievalCompositionRenderer,
)

D = 768

NAME = "r4_complement_address_transport"
DESCRIPTION = (
    "The r22 retrieval-composition arch, trained under ramped stochastic observation "
    "occlusion, plus a COMPLEMENT-ADDRESS TRANSPORT POINTER. Every command's raw embedding is "
    "mapped by one linear map, rescaled and offset by a soft verb mixture, into a single "
    "address code. A causal slot bank holds (unit address, occupancy, mixture-over-prefix-"
    "observations) triples. At each step a softmax over the occupied earlier slots, scored by "
    "address cosine plus occupancy, selects the slot the command reads; the destination address "
    "written by that step is the component of the step's own address code ORTHOGONAL to the "
    "selected slot's address, blended with a directly learned destination head. A transport gate "
    "decides whether the new slot carries the selected slot's observation mixture or a pointer to "
    "the step's own observation, and an erase gate decays the selected slot's occupancy. The "
    "resulting per-step mixture is contracted against the prefix observation embeddings and "
    "injected into the command-position prediction through a zero-init (D,D) readout, so the "
    "forward is bit-identical to r22 at initialization. Reads depend only on strictly earlier "
    "commands and observations."
)


class R4ComplementAddressTransport(R22RetrievalCompositionRenderer):
    def __init__(
        self,
        addr_d=96,
        addr_verbs=6,
        addr_hidden=192,
        match_temp=8.0,
        occ_weight=2.0,
        move_bias=0.0,
        write_bias=1.0,
        erase_bias=1.0,
        dst_mix_bias=2.0,
        ptr_gate_bias=0.0,
        occ_p=0.12,
        occ_ramp_start=300,
        occ_ramp_end=1000,
        **params,
    ):
        super().__init__(**params)

        self.addr_d = max(16, int(addr_d))
        self.addr_verbs = max(2, int(addr_verbs))
        ah = max(32, int(addr_hidden))
        self.addr_hidden = ah

        self.addr_map = nn.Linear(D, self.addr_d)
        self.addr_ctx_in = nn.Linear(D, ah)
        self.addr_verb = nn.Linear(ah, self.addr_verbs)
        self.addr_scale = nn.Parameter(torch.ones(self.addr_verbs))
        self.addr_offset = nn.Parameter(torch.zeros(self.addr_verbs, self.addr_d))

        self.query_adjust = nn.Linear(ah, self.addr_d)
        nn.init.zeros_(self.query_adjust.weight)
        nn.init.zeros_(self.query_adjust.bias)

        self.dst_direct = nn.Linear(ah, self.addr_d)
        self.dst_mix = nn.Linear(ah, 1)
        nn.init.constant_(self.dst_mix.bias, float(dst_mix_bias))

        gd = self.d + ah
        self.op_move = nn.Linear(gd, 1)
        self.op_write = nn.Linear(gd, 1)
        self.op_erase = nn.Linear(gd, 1)
        nn.init.constant_(self.op_move.bias, float(move_bias))
        nn.init.constant_(self.op_write.bias, float(write_bias))
        nn.init.constant_(self.op_erase.bias, float(erase_bias))

        self.log_match_temp = nn.Parameter(
            torch.tensor(math.log(max(0.2, float(match_temp))))
        )
        self.occ_weight = nn.Parameter(torch.tensor(float(occ_weight)))

        self.ptr_out = nn.Linear(D, D)
        nn.init.zeros_(self.ptr_out.weight)
        nn.init.zeros_(self.ptr_out.bias)
        self.ptr_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.ptr_gate.bias, float(ptr_gate_bias))

        self.occl_p = max(0.0, min(0.9, float(occ_p)))
        self.occl_start = max(0, int(occ_ramp_start))
        self.occl_end = max(self.occl_start + 1, int(occ_ramp_end))
        self.register_buffer("occl_step", torch.zeros((), dtype=torch.long))

    def _occlusion_prob(self):
        s = int(self.occl_step)
        if s <= self.occl_start:
            return 0.0
        if s >= self.occl_end:
            return self.occl_p
        x = (s - self.occl_start) / float(self.occl_end - self.occl_start)
        return self.occl_p * (x * x * (3.0 - 2.0 * x))

    def _address_code(self, cmd_raw):
        ctx = torch.nn.functional.gelu(self.addr_ctx_in(cmd_raw))
        pi = torch.softmax(self.addr_verb(ctx), dim=-1)
        base = self.addr_map(cmd_raw)
        scale = torch.matmul(pi, self.addr_scale.unsqueeze(-1))
        offset = torch.matmul(pi, self.addr_offset)
        code = scale * base - offset
        return torch.nan_to_num(code, nan=0.0, posinf=1e4, neginf=-1e4), ctx

    def _transport_pointer(self, code, ctx, h_cmd, live_slot, valid_cmd, n_cmd):
        B = code.size(0)
        dtype = code.dtype
        device = code.device

        temp = torch.exp(self.log_match_temp).clamp(0.05, 60.0).to(dtype)
        occ_w = self.occ_weight.to(dtype)

        gate_in = torch.cat([h_cmd.to(dtype), ctx.to(dtype)], dim=-1)
        g_move = torch.sigmoid(self.op_move(gate_in))
        g_write = torch.sigmoid(self.op_write(gate_in))
        g_erase = torch.sigmoid(self.op_erase(gate_in))
        g_dst = torch.sigmoid(self.dst_mix(ctx)).to(dtype)
        dst_free = self.dst_direct(ctx).to(dtype)

        query = self._unit(code + self.query_adjust(ctx).to(dtype))
        eye = torch.eye(n_cmd, device=device, dtype=dtype)
        valid_f = valid_cmd.to(dtype)
        neg = -1e4

        addr_bank = None
        mix_bank = None
        occ_bank = None
        live_bank = None

        reads = []
        sharps = []
        occ_hits = []

        for i in range(n_cmd):
            if addr_bank is None:
                mix_i = code.new_zeros(B, n_cmd)
                src_unit = code.new_zeros(B, self.addr_d)
                sharp_i = code.new_zeros(B, 1)
                occ_i = code.new_zeros(B, 1)
                alpha = None
            else:
                cos = torch.bmm(addr_bank, query[:, i, :].unsqueeze(2)).squeeze(2)
                logits = temp * cos + occ_w * occ_bank
                logits = logits + (live_bank - 1.0) * (-neg)
                has = live_bank.sum(dim=1, keepdim=True) > 0.0
                alpha = torch.softmax(logits, dim=1)
                alpha = torch.where(has, alpha, torch.zeros_like(alpha))
                alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
                mix_i = torch.bmm(alpha.unsqueeze(1), mix_bank).squeeze(1)
                src_unit = self._unit(torch.bmm(alpha.unsqueeze(1), addr_bank).squeeze(1))
                sharp_i = alpha.amax(dim=1, keepdim=True)
                occ_i = (alpha * occ_bank).sum(dim=1, keepdim=True)

            reads.append(mix_i)
            sharps.append(sharp_i)
            occ_hits.append(occ_i)

            code_i = code[:, i, :]
            proj = (code_i * src_unit).sum(dim=-1, keepdim=True)
            residual = code_i - proj * src_unit
            gd_i = g_dst[:, i, :]
            dst_i = gd_i * residual + (1.0 - gd_i) * dst_free[:, i, :]

            mv_i = g_move[:, i, :].to(dtype)
            addr_i = self._unit(mv_i * dst_i + (1.0 - mv_i) * code_i)
            self_row = eye[i].unsqueeze(0).expand(B, n_cmd) * live_slot[:, i : i + 1]
            new_mix = mv_i * mix_i + (1.0 - mv_i) * self_row
            new_occ = (g_write[:, i, :].to(dtype) * valid_f[:, i : i + 1]).squeeze(-1)

            addr_i = torch.nan_to_num(addr_i, nan=0.0, posinf=0.0, neginf=0.0)
            new_mix = torch.nan_to_num(new_mix, nan=0.0, posinf=0.0, neginf=0.0)

            if addr_bank is None:
                addr_bank = addr_i.unsqueeze(1)
                mix_bank = new_mix.unsqueeze(1)
                occ_bank = new_occ.unsqueeze(1)
                live_bank = valid_f[:, i : i + 1]
            else:
                decay = 1.0 - (mv_i * g_erase[:, i, :].to(dtype)) * alpha
                occ_bank = torch.cat(
                    [occ_bank * decay.clamp(0.0, 1.0), new_occ.unsqueeze(1)], dim=1
                )
                addr_bank = torch.cat([addr_bank, addr_i.unsqueeze(1)], dim=1)
                mix_bank = torch.cat([mix_bank, new_mix.unsqueeze(1)], dim=1)
                live_bank = torch.cat([live_bank, valid_f[:, i : i + 1]], dim=1)

        coeff = torch.stack(reads, dim=1)
        sharp = torch.stack(sharps, dim=1)
        occ_hit = torch.stack(occ_hits, dim=1)
        return coeff, sharp, occ_hit

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if key_pad is None:
            base_pad = torch.zeros(B, L, dtype=torch.bool, device=device)
        else:
            base_pad = key_pad.bool()

        tok_in = tok_emb
        pad_in = base_pad
        if self.training and B > 0 and L >= 2:
            self.occl_step += 1
            p = self._occlusion_prob()
            if p > 0.0:
                n_pair0 = L // 2
                drop = torch.rand(B, n_pair0, device=device) < p
                drop_full = torch.zeros(B, L, dtype=torch.bool, device=device)
                drop_full[:, 1 : 2 * n_pair0 : 2] = drop
                tok_in = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                pad_in = base_pad | drop_full

        pred, h_out = super().forward(tok_in, types, pad_in)

        if L < 3:
            return pred, h_out

        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        valid_cmd = ~base_pad[:, 0::2]
        valid_obs = ~base_pad[:, 1::2]
        live_pair = (valid_cmd[:, :n_pair] & valid_obs).to(dtype)
        live_slot = self._pad_steps(live_pair, n_cmd)

        obs_val = tok_in[:, 1::2, :] * valid_obs.unsqueeze(-1).to(dtype)
        obs_val = self._pad_steps(obs_val, n_cmd)

        cmd_raw = tok_emb[:, 0::2, :]
        h_cmd = h_out[:, 0::2, :]

        code, ctx = self._address_code(cmd_raw)
        coeff, sharp, occ_hit = self._transport_pointer(
            code, ctx, h_cmd, live_slot, valid_cmd, n_cmd
        )

        content = torch.bmm(coeff, obs_val)
        content = torch.nan_to_num(content, nan=0.0, posinf=1e4, neginf=-1e4)

        mass = coeff.sum(dim=-1, keepdim=True)
        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([sharp, occ_hit, mass, rms], dim=-1).to(h_cmd.dtype)

        gate = torch.sigmoid(self.ptr_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate.to(dtype) * self.ptr_out(content)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R4ComplementAddressTransport(**params)
