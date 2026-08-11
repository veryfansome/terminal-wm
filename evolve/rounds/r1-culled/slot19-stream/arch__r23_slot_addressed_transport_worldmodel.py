import torch
import torch.nn as nn

from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

NAME = "r23_slot_addressed_transport_worldmodel"
DESCRIPTION = (
    "r22 (r18 path-state world model under ramped observation occlusion) whose latent-transition "
    "memory is re-addressed: the read key comes from the command's SOURCE path slot and the write "
    "keys from its DESTINATION / destination-parent / directory-join slots, decoded exactly from "
    "the reserved coordinate blocks the r23 path-slot stream writes into the command token. A step "
    "therefore reads the content currently held at the source address, pushes it through the shared "
    "transition operator, overwrites the destination addresses with it and erases the source, so a "
    "chain of moves transports content instead of marking touched paths. Command-code occlusion is "
    "ramped alongside observation occlusion so the learned fallback addressing stays alive."
)

D = 768
N_SLOT = 5
SLOT_K = 40
BASIS_SEED = 20260809
CODE_SCALE = float(SLOT_K) ** 0.5


def slot_index():
    g = torch.Generator().manual_seed(BASIS_SEED)
    return torch.randperm(D, generator=g)[: N_SLOT * SLOT_K].view(N_SLOT, SLOT_K).contiguous()


class R23SlotAddressedTransport(R22ObservationOcclusionDenoising):
    def __init__(self, code_drop=0.10, code_drop_ramp=600, learned_addr_scale=0.15, **params):
        super().__init__(**params)
        self.code_drop = max(0.0, min(0.9, float(code_drop)))
        self.code_drop_ramp = max(1, int(code_drop_ramp))
        self.register_buffer("slot_flat", slot_index().reshape(-1))
        self.register_buffer("code_step", torch.zeros((), dtype=torch.long))

        self.rd_gate = nn.Linear(self.d, 2)
        self.w_amt = nn.Linear(self.d, 3)
        self.e_amt = nn.Linear(self.d, 1)
        self.tr_path_w = nn.Linear(self.d, self.key_d, bias=False)
        self.addr = nn.Parameter(torch.empty(SLOT_K, self.key_d))
        self.learned_addr = nn.Parameter(torch.tensor(float(learned_addr_scale)))

        nn.init.orthogonal_(self.addr)
        nn.init.normal_(self.tr_path_w.weight, std=0.02)
        nn.init.zeros_(self.rd_gate.weight)
        with torch.no_grad():
            self.rd_gate.bias.copy_(torch.tensor([1.5, -1.5]))
        nn.init.zeros_(self.w_amt.weight)
        with torch.no_grad():
            self.w_amt.bias.copy_(torch.tensor([2.0, -1.0, -1.0]))
        nn.init.zeros_(self.e_amt.weight)
        nn.init.constant_(self.e_amt.bias, 0.0)
        self._cmd_raw = None

    def _code_keep_prob(self):
        if not self.training or self.code_drop <= 0.0:
            return 1.0
        x = min(1.0, float(int(self.code_step)) / float(self.code_drop_ramp))
        return 1.0 - self.code_drop * (x * x * (3.0 - 2.0 * x))

    def _decode_slots(self, n_cmd):
        raw = self._cmd_raw
        if raw is None:
            return None
        raw = raw[:, :n_cmd]
        a = raw.index_select(-1, self.slot_flat)
        a = a.reshape(raw.size(0), raw.size(1), N_SLOT, SLOT_K) / CODE_SCALE
        keep = self._code_keep_prob()
        if keep < 1.0:
            m = (torch.rand(a.size(0), a.size(1), 1, 1, device=a.device) < keep).to(a.dtype)
            a = a * m
        return self._unit(a)

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        a = self._decode_slots(n_cmd)
        learned_r = self.learned_addr * self._unit(self.tr_path(x_cmd))
        learned_w = (self.learned_addr * self._unit(self.tr_path_w(x_cmd))).unsqueeze(2)
        if a is None:
            p_r = self._unit(self.tr_path(x_cmd))
            p_w = self._unit(self.tr_path_w(x_cmd)).unsqueeze(2).expand(B, n_cmd, 3, self.key_d)
        else:
            a = a.to(x_cmd.dtype)
            g_r = torch.softmax(self.rd_gate(x_cmd), dim=-1)
            a_read = g_r[..., 0:1] * a[:, :, 0] + g_r[..., 1:2] * a[:, :, 1]
            p_r = self._unit(torch.matmul(a_read, self.addr) + learned_r)
            p_w = self._unit(torch.matmul(a[:, :, 2:5], self.addr) + learned_w)

        keys = torch.cat([p_r.unsqueeze(2), p_w], dim=2)
        amt_w = torch.sigmoid(self.w_amt(x_cmd)).unsqueeze(-1)
        amt_e = torch.sigmoid(self.e_amt(x_cmd))
        w_mut = torch.sigmoid(self.tr_mut_gate(x_cmd))

        decay = (0.90 + 0.099 * torch.sigmoid(self.logit_decay)).to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ki = keys[:, i].to(dtype)
            cur = torch.bmm(ki, mem)
            s_src = cur[:, 0]
            reads.append(s_src)
            delta = self._transition(s_src, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_src.new_zeros(B, D)
                active = x_cmd.new_zeros(B, 1)
            wi = w_mut[:, i]
            value = (1.0 - wi) * obs_i + wi * delta
            corr_w = (value.unsqueeze(1) - cur[:, 1:]) * amt_w[:, i] * active.unsqueeze(1)
            corr_e = (-s_src * amt_e[:, i] * active).unsqueeze(1)
            corr = torch.cat([corr_e, corr_w], dim=1)
            mem = decay * mem + torch.bmm(ki.transpose(1, 2), corr)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        self._cmd_raw = tok_emb[:, 0::2]
        if self.training:
            self.code_step += 1
        return super().forward(tok_emb, types, key_pad)


def build(**params):
    return R23SlotAddressedTransport(**params)
