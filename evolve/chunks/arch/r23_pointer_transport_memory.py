import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)
from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

D = 768

NAME = "r23_pointer_transport_memory"
DESCRIPTION = (
    "The r22 occlusion-denoised r18 path-state trunk plus a TWO-ADDRESS pointer-transport "
    "memory: every command emits a source address and a destination address in one shared "
    "verb-quotiented address space, and the slot memory update is a soft register move "
    "(the content read at the source is deposited at the destination and removed from the "
    "source) followed by a delta-rule write of the observation at the source address. The "
    "source-address read is injected into the command-position prediction through a zero-init "
    "(D,D) readout whose gate is fed an explicit slot-occupancy scalar, so the contribution "
    "vanishes and the plain trunk head stands alone when no content has been stored at the "
    "address being read. Bit-for-bit identical to the r22 forward at initialization."
)


class R23PointerTransportMemory(R22ObservationOcclusionDenoising):
    def __init__(
        self,
        tp_rms_cap=1.0,
        tp_move_bias=-1.0,
        tp_write_bias=1.0,
        tp_out_bias=-2.0,
        **params,
    ):
        super().__init__(**params)
        self.tp_rms_cap = float(tp_rms_cap)
        self.tp_src_sel = nn.Linear(self.d, self.d)
        self.tp_dst_sel = nn.Linear(self.d, self.d)
        self.tp_addr = nn.Linear(self.d, self.key_d, bias=False)
        self.tp_move_gate = nn.Linear(2 * self.d, 1)
        self.tp_write_gate = nn.Linear(2 * self.d, 1)
        self.tp_read = nn.Linear(D, D)
        self.tp_out_gate = nn.Linear(self.d + 2, 1)
        nn.init.eye_(self.tp_src_sel.weight)
        nn.init.zeros_(self.tp_src_sel.bias)
        nn.init.zeros_(self.tp_read.weight)
        nn.init.zeros_(self.tp_read.bias)
        nn.init.constant_(self.tp_move_gate.bias, float(tp_move_bias))
        nn.init.constant_(self.tp_write_gate.bias, float(tp_write_bias))
        nn.init.constant_(self.tp_out_gate.bias, float(tp_out_bias))

    def _occlude(self, tok_emb, key_pad):
        if not self.training:
            return tok_emb, key_pad
        self.occ_step += 1
        p = self._occ_prob()
        B, L = tok_emb.shape[0], tok_emb.shape[1]
        if p <= 0.0 or L < 2 or B == 0:
            return tok_emb, key_pad
        if key_pad is None:
            key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
        key_pad = key_pad.bool()
        n_pair = L // 2
        drop = torch.rand(B, n_pair, device=tok_emb.device) < p
        drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
        drop_full[:, 1:2 * n_pair:2] = drop
        tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
        return tok_emb, key_pad | drop_full

    def _cmd_features(self, tok_emb, L, n_cmd):
        device = tok_emb.device
        cmd_tok = tok_emb[:, 0::2, :]
        z = self.cmd_proj(cmd_tok)
        idx0 = torch.zeros(cmd_tok.shape[0], n_cmd, dtype=torch.long, device=device)
        pe = self._positional(L, device, z.dtype)[0::2].unsqueeze(0)
        return self.in_norm(z + self.type_emb(idx0) + self.pos_scale * pe)

    def _transport_reads(self, x_cmd, h_cmd, h_obs, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.shape[0]
        dtype = x_cmd.dtype

        Q = self._verb_basis()
        a = self._quotient(self.tp_addr(self.tp_src_sel(x_cmd)), Q)
        b = self._quotient(self.tp_addr(self.tp_dst_sel(x_cmd)), Q)
        ab = (a * b).sum(dim=-1, keepdim=True)

        mv = torch.sigmoid(self.tp_move_gate(torch.cat([x_cmd, h_cmd], dim=-1))).squeeze(-1)
        mv = mv * valid_cmd.to(dtype)

        if n_pair:
            wr_in = torch.cat([h_cmd[:, :n_pair, :], h_obs], dim=-1)
            wr = torch.sigmoid(self.tp_write_gate(wr_in)).squeeze(-1)
            wr = wr * (valid_cmd[:, :n_pair] & valid_obs).to(dtype)
            wr = self._pad_steps(wr, n_cmd)
            obs = self._pad_steps(obs_tok.to(dtype), n_cmd)
        else:
            wr = x_cmd.new_zeros(B, n_cmd)
            obs = x_cmd.new_zeros(B, n_cmd, D)

        mem = x_cmd.new_zeros(B, self.key_d, D)
        occ = x_cmd.new_zeros(B, self.key_d)
        reads = []
        confs = []

        for i in range(n_cmd):
            ai = a[:, i, :]
            bi = b[:, i, :]
            s = torch.bmm(ai.unsqueeze(1), mem).squeeze(1)
            c = (ai * occ).sum(dim=-1, keepdim=True)
            reads.append(s)
            confs.append(c)

            mi = mv[:, i].unsqueeze(-1)
            wi = wr[:, i].unsqueeze(-1)

            shrink = 1.0 + mi * (ab[:, i, :] - 1.0)
            s_post = s * shrink
            c_post = c * shrink

            carry = mi * s
            deposit = wi * (obs[:, i, :] - s_post)
            addr_pair = torch.stack([bi, ai], dim=2)
            val_pair = torch.stack([carry, deposit - carry], dim=1)
            mem = torch.nan_to_num(
                mem + torch.bmm(addr_pair, val_pair), nan=0.0, posinf=1e4, neginf=-1e4
            )

            c_carry = mi * c
            c_dep = wi * (1.0 - c_post)
            occ = torch.nan_to_num(
                occ + bi * c_carry + ai * (c_dep - c_carry), nan=0.0, posinf=1e2, neginf=-1e2
            )

        r = torch.nan_to_num(
            torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-1e4, 1e4)
        cf = torch.nan_to_num(
            torch.stack(confs, dim=1), nan=0.0, posinf=1e2, neginf=-1e2
        ).clamp(-1e2, 1e2)
        return r, cf

    def forward(self, tok_emb, types, key_pad):
        tok_emb, key_pad = self._occlude(tok_emb, key_pad)
        pred, h_out = R18PathStateLatentTransition.forward(self, tok_emb, types, key_pad)

        B, L, _ = tok_emb.shape
        if L == 0:
            return pred, h_out

        device = tok_emb.device
        n_cmd = (L + 1) // 2
        n_pair = L // 2

        if key_pad is not None:
            valid = ~key_pad.bool()
        else:
            valid = torch.ones(B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        x_cmd = self._cmd_features(tok_emb, L, n_cmd)
        h_cmd = h_out[:, 0::2, :].to(x_cmd.dtype)
        h_obs = h_out[:, 1::2, :].to(x_cmd.dtype)
        obs_tok = tok_emb[:, 1::2, :]

        r, cf = self._transport_reads(
            x_cmd, h_cmd, h_obs, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair
        )

        rms = (r.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        scale = (self.tp_rms_cap / rms.clamp_min(self.tp_rms_cap)).detach()
        g = torch.sigmoid(self.tp_out_gate(torch.cat([h_cmd, cf, rms], dim=-1)))
        contrib = g * self.tp_read((r * scale).to(self.tp_read.weight.dtype)).to(g.dtype)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R23PointerTransportMemory(**params)
