import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r25_address_transport_memory"
DESCRIPTION = (
    "The r22 arch (r18 path-state trunk + prefix-content cross-attention) plus a forward "
    "ADDRESS-TRANSPORT MEMORY: one associative store mem[key_d, 768] carried left-to-right over the "
    "command steps. Every command's raw 768-d embedding goes through one shared feature layer and "
    "then three (key_d, hidden) projections — a shared one plus a source-role and a destination-role "
    "one — whose sums are unit-normalized into a SOURCE address and a DESTINATION address in one "
    "common address space; the destination address doubles as the read address. At each step the "
    "module reads the content currently held at both addresses, then applies three learned scalar "
    "gates: a move gate that writes (content_at_source - content_at_destination) at the destination, "
    "an erase gate scaled by one minus the source/destination address cosine that subtracts the "
    "source content at the source address, and a write gate that delta-rule-writes this step's raw "
    "observation embedding at the destination address. The two rank-one updates are applied as one "
    "batched outer product with a learned decay. The move update is identically zero whenever the "
    "two role addresses coincide, so single-path commands leave the store untouched; a chain of "
    "two-path commands relocates a stored content one address per step, and the read at a later "
    "command is a single lookup at that command's destination address regardless of chain length. "
    "The read is injected into the command-position prediction through an identity-scaled (D,D) "
    "readout under a sigmoid gate fed the read RMS, the address split and the stored mass. Reads at "
    "a step use only the store built from strictly earlier observations and commands, so no "
    "prediction sees its own observation. Muon captures the three (key_d, hidden) address "
    "projections, the spectral cap the (D,D) readout."
)


class R25AddressTransportMemory(R22PrefixContentXAttention):
    def __init__(
        self,
        transport_hidden=176,
        move_bias=-1.0,
        erase_bias=0.0,
        write_bias=1.0,
        transport_gate_bias=-1.0,
        readout_init=0.1,
        decay_init=0.995,
        **params,
    ):
        super().__init__(**params)
        ah = max(32, int(transport_hidden))
        self.transport_hidden = ah

        self.tp_ln = nn.LayerNorm(D)
        self.tp_in = nn.Linear(D, ah)
        self.tp_share = nn.Linear(ah, self.key_d, bias=False)
        self.tp_src = nn.Linear(ah, self.key_d, bias=False)
        self.tp_dst = nn.Linear(ah, self.key_d, bias=False)

        self.tp_move = nn.Linear(ah + self.d + 1, 1)
        nn.init.constant_(self.tp_move.bias, float(move_bias))
        self.tp_erase = nn.Linear(ah + self.d + 1, 1)
        nn.init.constant_(self.tp_erase.bias, float(erase_bias))
        self.tp_write = nn.Linear(2 * self.d + 1, 1)
        nn.init.constant_(self.tp_write.bias, float(write_bias))

        self.tp_out = nn.Linear(D, D)
        with torch.no_grad():
            self.tp_out.weight.copy_(torch.eye(D) * float(readout_init))
            self.tp_out.bias.zero_()
        self.tp_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.tp_gate.bias, float(transport_gate_bias))

        z = min(max(float(decay_init), 0.9005), 0.9995)
        s = (z - 0.90) / 0.0999
        self.tp_logit_decay = nn.Parameter(torch.tensor(math.log(s / (1.0 - s))))

    def _role_addresses(self, cmd_raw):
        f = F.gelu(self.tp_in(self.tp_ln(cmd_raw)))
        shared = self.tp_share(f)
        a_src = self._unit(shared + self.tp_src(f))
        a_dst = self._unit(shared + self.tp_dst(f))
        return f, a_src, a_dst

    def _transport_reads(self, a_src, a_dst, obs_raw, g_move, g_erase, g_write, lam, n_cmd):
        B = a_src.size(0)
        mem = a_src.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ad = a_dst[:, i, :]
            asr = a_src[:, i, :]
            v_dst = torch.bmm(ad.unsqueeze(1), mem).squeeze(1)
            v_src = torch.bmm(asr.unsqueeze(1), mem).squeeze(1)
            reads.append(v_dst)

            moved = g_move[:, i, :] * (v_src - v_dst)
            v_post = v_dst + moved
            u_dst = moved + g_write[:, i, :] * (obs_raw[:, i, :] - v_post)
            u_src = -g_erase[:, i, :] * v_src

            addr = torch.stack([ad, asr], dim=2)
            upd = torch.stack([u_dst, u_src], dim=1)
            mem = lam[:, i, :].unsqueeze(-1) * mem + torch.bmm(addr, upd)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.stack(reads, dim=1)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 2:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        device = tok_emb.device
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        wdt = self.tp_in.weight.dtype
        cmd_raw = torch.nan_to_num(
            tok_emb[:, 0::2, :].to(wdt), nan=0.0, posinf=1e4, neginf=-1e4)
        obs_raw = self._pad_steps(
            torch.nan_to_num(tok_emb[:, 1::2, :].to(wdt), nan=0.0, posinf=1e4, neginf=-1e4),
            n_cmd,
        )
        h_cmd = h_out[:, 0::2, :].to(wdt)
        h_obs = self._pad_steps(h_out[:, 1::2, :].to(wdt), n_cmd)

        live_cmd = valid_cmd.to(wdt).unsqueeze(-1)
        live_obs = (self._pad_steps(valid_obs, n_cmd) & valid_cmd).to(wdt).unsqueeze(-1)

        f, a_src, a_dst = self._role_addresses(cmd_raw)
        role_cos = (a_src * a_dst).sum(dim=-1, keepdim=True)
        role_split = (1.0 - role_cos).clamp(0.0, 1.0)

        cmd_gate_in = torch.cat([f, h_cmd, role_cos], dim=-1)
        g_move = torch.sigmoid(self.tp_move(cmd_gate_in)) * live_cmd
        g_erase = torch.sigmoid(self.tp_erase(cmd_gate_in)) * live_cmd * role_split
        g_write = torch.sigmoid(
            self.tp_write(torch.cat([h_cmd, h_obs, role_cos], dim=-1))) * live_obs

        decay = 0.90 + 0.0999 * torch.sigmoid(self.tp_logit_decay)
        lam = decay * live_cmd + (1.0 - live_cmd)

        v_read = self._transport_reads(
            a_src, a_dst, obs_raw, g_move, g_erase, g_write, lam, n_cmd)
        v_read = torch.nan_to_num(v_read, nan=0.0, posinf=1e4, neginf=-1e4)

        rms = (v_read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        stored = torch.cumsum(g_write, dim=1) - g_write
        stored = stored / (stored + 1.0)
        gate = torch.sigmoid(
            self.tp_gate(torch.cat([h_cmd, rms, role_split, stored], dim=-1)))

        contrib = gate * self.tp_out(v_read) * live_cmd
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R25AddressTransportMemory(**params)
