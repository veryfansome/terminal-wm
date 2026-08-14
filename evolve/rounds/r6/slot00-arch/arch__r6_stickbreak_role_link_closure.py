import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_retrieval_composition_renderer import (
    R22RetrievalCompositionRenderer,
)

D = 768

NAME = "r6_stickbreak_role_link_closure"
DESCRIPTION = (
    "The r22 retrieval-composition arch under the same ramped observation-occlusion schedule, "
    "with the parent's sequential complement-address slot bank REPLACED by a STICK-BREAKING "
    "ROLE-LINK CLOSURE. Each command's raw embedding is read by one source-role map and by that "
    "same map plus a relocation offset scaled by a learned 'this command relocates' gate, and both "
    "role vectors are pushed through ONE shared nonlinear filler encoder, so the place a "
    "single-argument command reads and the place a two-argument command writes are addresses in "
    "the same code by construction. Between commands, selection weights are stick-breaking rather "
    "than softmax: w_ij = sigmoid(sel_ij) * prod over the strictly intervening k of "
    "(1 - sigmoid(sel_ik)) * (1 - sigmoid(shd_ik)), where sel matches a read's source address "
    "against an earlier command's destination address and shd matches it against an earlier "
    "command's SOURCE address offset by that command's relocation logit. The product picks the "
    "most recent writer to the queried location and cancels it once a later command has taken the "
    "content away, computed in closed form from two cumulative sums. Content is then defined by "
    "one strictly lower-triangular recursion, register_j = own_j*obs_j + (1-own_j)*sum_k w_jk "
    "register_k, solved exactly by a unit-triangular forward substitution, so a chain of "
    "contentless relocations composes to unbounded depth in a single pass; the readout at a "
    "command is the strictly earlier registers pulled through the same weights, injected into the "
    "command-position prediction through an identity-initialised (D,D) map with a small learned "
    "gain. Reads depend only on strictly earlier commands and observations."
)


class R6StickBreakRoleLinkClosure(R22RetrievalCompositionRenderer):
    def __init__(
        self,
        link_d=64,
        link_hidden=192,
        link_temp=4.0,
        move_bias=0.0,
        move_init_scale=0.25,
        own_bias=0.0,
        link_bias=-1.0,
        shd_bias=-1.0,
        ptr_gate_bias=0.0,
        link_gain=0.05,
        occ_p=0.12,
        occ_ramp_start=300,
        occ_ramp_end=1000,
        **params,
    ):
        super().__init__(**params)

        self.link_d = max(8, int(link_d))
        lh = max(32, int(link_hidden))
        self.link_hidden = lh

        self.link_ln = nn.LayerNorm(D)
        self.role_src = nn.Linear(D, lh)
        self.role_move = nn.Linear(D, lh)
        with torch.no_grad():
            self.role_move.weight.mul_(float(move_init_scale))
            self.role_move.bias.mul_(float(move_init_scale))
        self.move_gate = nn.Linear(D, 1)
        nn.init.constant_(self.move_gate.bias, float(move_bias))

        self.fill_in = nn.Linear(lh, lh)
        self.fill_out = nn.Linear(lh, self.link_d)

        self.key_bias = nn.Linear(D, 1)
        nn.init.zeros_(self.key_bias.weight)
        nn.init.zeros_(self.key_bias.bias)

        self.log_link_temp = nn.Parameter(
            torch.tensor(math.log(max(0.1, float(link_temp))))
        )
        self.link_bias = nn.Parameter(torch.tensor(float(link_bias)))
        self.shd_bias = nn.Parameter(torch.tensor(float(shd_bias)))

        self.own_gate = nn.Linear(2 * self.d, 1)
        nn.init.constant_(self.own_gate.bias, float(own_bias))

        self.ptr_out = nn.Linear(D, D)
        with torch.no_grad():
            self.ptr_out.weight.copy_(torch.eye(D))
            self.ptr_out.bias.zero_()
        self.ptr_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.ptr_gate.bias, float(ptr_gate_bias))
        self.link_gain = nn.Parameter(torch.tensor(float(link_gain)))

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

    def _link_addresses(self, cmd_raw):
        u = self.link_ln(cmd_raw)
        r_src = self.role_src(u)
        move_logit = self.move_gate(u)
        r_dst = r_src + torch.sigmoid(move_logit) * self.role_move(u)
        a_src = self._unit(self.fill_out(F.gelu(self.fill_in(r_src))))
        a_dst = self._unit(self.fill_out(F.gelu(self.fill_in(r_dst))))
        return a_src, a_dst, self.key_bias(u), move_logit

    def _link_weights(self, a_src, a_dst, key_bias, move_logit, valid_cmd, n_cmd):
        device = a_src.device
        dtype = a_src.dtype

        temp = torch.exp(self.log_link_temp).clamp(0.05, 60.0).to(dtype)
        sel = temp * torch.bmm(a_src, a_dst.transpose(1, 2))
        sel = sel + key_bias.transpose(1, 2).to(dtype) + self.link_bias.to(dtype)
        shd = temp * torch.bmm(a_src, a_src.transpose(1, 2))
        shd = shd + move_logit.transpose(1, 2).to(dtype) + self.shd_bias.to(dtype)

        idx = torch.arange(n_cmd, device=device)
        strict = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0)
        allowed = strict & valid_cmd.unsqueeze(1)
        af = allowed.to(dtype)

        block = -(F.softplus(sel) + F.softplus(shd)) * af
        cum = torch.cumsum(block, dim=2)
        rowtot = cum[:, :, -1:]
        logw = F.logsigmoid(sel) + (rowtot - cum)
        logw = torch.where(allowed, logw, torch.full_like(logw, -1e4))
        w = torch.exp(logw.clamp(max=0.0))
        return torch.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    def _closure_reads(self, w, own, obs_pad, n_cmd):
        B = w.size(0)
        device = w.device
        dtype = w.dtype

        eye = torch.eye(n_cmd, device=device, dtype=dtype).unsqueeze(0).expand(B, n_cmd, n_cmd)
        system = eye - (1.0 - own) * w
        rhs = own * obs_pad
        registers = self._solve_lower(system, rhs)
        registers = torch.nan_to_num(
            registers, nan=0.0, posinf=1e4, neginf=-1e4
        ).clamp(-1e4, 1e4)

        pulled = torch.bmm(w, registers)
        return torch.nan_to_num(pulled, nan=0.0, posinf=1e4, neginf=-1e4)

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
        if n_pair == 0 or n_cmd < 2:
            return pred, h_out

        valid_cmd = ~base_pad[:, 0::2]
        valid_obs = ~base_pad[:, 1::2]
        live_pair = (valid_cmd[:, :n_pair] & valid_obs).to(dtype)
        live_slot = self._pad_steps(live_pair, n_cmd)

        obs_val = tok_in[:, 1::2, :] * valid_obs.unsqueeze(-1).to(dtype)
        obs_pad = self._pad_steps(obs_val, n_cmd)

        cmd_raw = tok_emb[:, 0::2, :]
        h_cmd = h_out[:, 0::2, :]
        h_obs_pad = self._pad_steps(h_out[:, 1::2, :], n_cmd)

        a_src, a_dst, key_bias, move_logit = self._link_addresses(cmd_raw)
        w = self._link_weights(a_src, a_dst, key_bias, move_logit, valid_cmd, n_cmd)

        own = torch.sigmoid(
            self.own_gate(torch.cat([h_cmd, h_obs_pad], dim=-1))
        ).to(dtype)
        own = own * live_slot.unsqueeze(-1)

        pulled = self._closure_reads(w, own, obs_pad, n_cmd)

        mass = w.sum(dim=-1, keepdim=True)
        sharp = w.amax(dim=-1, keepdim=True)
        src_own = torch.bmm(w, own)
        rms = (pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([sharp, mass, src_own, rms], dim=-1).to(h_cmd.dtype)

        gate = torch.sigmoid(self.ptr_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate.to(dtype) * self.link_gain.to(dtype) * self.ptr_out(pulled)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R6StickBreakRoleLinkClosure(**params)
