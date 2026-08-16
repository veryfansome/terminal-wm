import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "mutable_key_register_file"
DESCRIPTION = (
    "The r22 prefix-content trunk with its backward pointer chase replaced by a FORWARD "
    "REGISTER FILE whose slot KEYS are mutable and whose slot CONTENTS are immutable. A bank of "
    "n_slots registers carries a unit location key, a convex coefficient row over the "
    "observations seen so far, and an occupancy scalar. Each command's raw 768-d embedding runs "
    "through one shared two-layer feature trunk, then three separate nonlinear role branches "
    "(read, source, destination) that all map through ONE shared address projection, so a read "
    "address and an earlier command's destination address live in the same space and can match. "
    "Scanning the sequence once, every step first READS — a temperature-sharpened softmax over "
    "slot keys, weighted by occupancy and renormalized, pulls that slot's coefficient row, so the "
    "retrieved content is exactly a convex mixture of strictly earlier raw observations — and "
    "only then WRITES. The write does two things under gates computed from the command features "
    "and the step's own observation: a RELOCATION that softmax-matches slots against the source "
    "address, weights the match by a confidence sigmoid on the peak similarity, and slides the "
    "matched slot's key toward the destination address, renormalizing back to the sphere; and an "
    "INSTALL that binds this step's observation into the slot the read address already occupies "
    "whenever the occupancy-masked peak similarity says that address is already held, and "
    "otherwise into a register allocated by a straight-through hard argmax over one minus "
    "occupancy plus a learned per-slot tiebreak, setting that slot's key to the read address and "
    "raising its occupancy. Occupancy leaks by a learned factor, so allocation recycles the "
    "stalest register. A content "
    "therefore keeps one register for the whole trajectory while its key is rewritten by each "
    "move, and a read of the final location returns the observation that content was introduced "
    "with, at any chain depth, with no hop budget. The mixture is injected into the command-"
    "position prediction through an identity-scaled (D,D) readout under a sigmoid gate fed the "
    "retrieved mass, the peak share and the mixture RMS. Strictly causal: a step reads state "
    "built only from strictly earlier steps, and the coefficient rows only ever carry weight on "
    "strictly earlier observations."
)

_EPS = 1e-4


def _inv_softplus(x):
    v = max(1e-4, float(x))
    return math.log(math.expm1(v)) if v < 20.0 else v


def _logit(x):
    v = min(1.0 - 1e-6, max(1e-6, float(x)))
    return math.log(v / (1.0 - v))


class MutableKeyRegisterFile(R22PrefixContentXAttention):
    def __init__(
        self,
        n_slots=32,
        addr_hidden=256,
        role_hidden=128,
        obs_feat_d=64,
        ctrl_hidden=96,
        read_sharp=10.0,
        move_sharp=10.0,
        alloc_sharp=8.0,
        alloc_tiebreak=0.01,
        conf_sharp=6.0,
        conf_thresh=0.5,
        hit_sharp=6.0,
        hit_thresh=0.5,
        free_penalty=4.0,
        occ_leak=0.99,
        key_init_scale=0.5,
        move_bias=0.0,
        install_bias=0.0,
        readout_init=0.05,
        out_gate_bias=-2.0,
        **params,
    ):
        super().__init__(**params)
        self.n_slots = max(2, int(n_slots))
        self.addr_hidden = max(32, int(addr_hidden))
        self.role_hidden = max(16, int(role_hidden))
        self.obs_feat_d = max(8, int(obs_feat_d))

        self.regmem_ln = nn.LayerNorm(D)
        self.regmem_feat_in = nn.Linear(D, self.addr_hidden)
        self.regmem_feat_mid = nn.Linear(self.addr_hidden, self.addr_hidden)
        self.regmem_role_read = nn.Linear(self.addr_hidden, self.role_hidden)
        self.regmem_role_src = nn.Linear(self.addr_hidden, self.role_hidden)
        self.regmem_role_dst = nn.Linear(self.addr_hidden, self.role_hidden)
        self.regmem_addr = nn.Linear(self.role_hidden, self.key_d, bias=False)
        self.regmem_key0 = nn.Parameter(
            torch.randn(self.n_slots, self.key_d) * float(key_init_scale))

        self.regmem_obs_feat = nn.Linear(D, self.obs_feat_d)
        ch = max(16, int(ctrl_hidden))
        self.regmem_ctrl_in = nn.Linear(self.addr_hidden + self.obs_feat_d, ch)
        self.regmem_move_out = nn.Linear(ch, 1)
        self.regmem_install_out = nn.Linear(ch, 1)
        nn.init.constant_(self.regmem_move_out.bias, float(move_bias))
        nn.init.constant_(self.regmem_install_out.bias, float(install_bias))

        self.regmem_read_sharp = nn.Parameter(torch.tensor(_inv_softplus(read_sharp)))
        self.regmem_move_sharp = nn.Parameter(torch.tensor(_inv_softplus(move_sharp)))
        self.regmem_alloc_sharp = nn.Parameter(torch.tensor(_inv_softplus(alloc_sharp)))
        self.regmem_conf_sharp = nn.Parameter(torch.tensor(_inv_softplus(conf_sharp)))
        self.regmem_hit_sharp = nn.Parameter(torch.tensor(_inv_softplus(hit_sharp)))
        self.regmem_free_pen = nn.Parameter(torch.tensor(_inv_softplus(free_penalty)))
        self.regmem_alloc_bias = nn.Parameter(
            torch.randn(self.n_slots) * float(alloc_tiebreak))
        self.regmem_conf_thresh = nn.Parameter(torch.tensor(float(conf_thresh)))
        self.regmem_hit_thresh = nn.Parameter(torch.tensor(float(hit_thresh)))
        self.regmem_leak = nn.Parameter(
            torch.tensor(_logit((min(0.999, max(0.90, float(occ_leak))) - 0.90) / 0.0999)))

        self.regmem_out = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.regmem_out.weight.copy_(torch.eye(D) * float(readout_init))
        self.regmem_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.regmem_gate.bias, float(out_gate_bias))

    @staticmethod
    def _regmem_unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _regmem_pad(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(shape)], dim=1)

    def _regmem_addresses(self, cmd_raw):
        f = F.gelu(self.regmem_feat_mid(F.gelu(self.regmem_feat_in(self.regmem_ln(cmd_raw)))))
        a_read = self._regmem_unit(self.regmem_addr(F.gelu(self.regmem_role_read(f))).float())
        a_src = self._regmem_unit(self.regmem_addr(F.gelu(self.regmem_role_src(f))).float())
        a_dst = self._regmem_unit(self.regmem_addr(F.gelu(self.regmem_role_dst(f))).float())
        return f, a_read, a_src, a_dst

    def _regmem_scan(self, a_read, a_src, a_dst, move_p, install_p, n_cmd):
        B = a_read.size(0)
        device = a_read.device

        s_read = F.softplus(self.regmem_read_sharp.float()).clamp(0.25, 64.0)
        s_move = F.softplus(self.regmem_move_sharp.float()).clamp(0.25, 64.0)
        s_alloc = F.softplus(self.regmem_alloc_sharp.float()).clamp(0.25, 64.0)
        s_conf = F.softplus(self.regmem_conf_sharp.float()).clamp(0.25, 64.0)
        s_hit = F.softplus(self.regmem_hit_sharp.float()).clamp(0.25, 64.0)
        free_pen = F.softplus(self.regmem_free_pen.float()).clamp(0.25, 32.0)
        alloc_bias = self.regmem_alloc_bias.float().view(1, -1)
        leak = 0.90 + 0.0999 * torch.sigmoid(self.regmem_leak.float())
        conf_thresh = self.regmem_conf_thresh.float()
        hit_thresh = self.regmem_hit_thresh.float()

        keys = self._regmem_unit(self.regmem_key0.float()).unsqueeze(0).expand(B, -1, -1)
        keys = keys.contiguous()
        coefs_state = torch.zeros(B, self.n_slots, n_cmd, device=device, dtype=torch.float32)
        occ = torch.zeros(B, self.n_slots, device=device, dtype=torch.float32)
        eye = torch.eye(n_cmd, device=device, dtype=torch.float32)

        out_coef = []
        out_mass = []
        out_peak = []
        for i in range(n_cmd):
            ar = a_read[:, i, :]
            sim_r = torch.bmm(keys, ar.unsqueeze(2)).squeeze(2)
            p_r = torch.softmax(sim_r * s_read, dim=1)
            w_r = p_r * occ
            mass = w_r.sum(dim=1, keepdim=True)
            denom = mass.clamp_min(_EPS)
            share = w_r / denom
            out_coef.append(torch.bmm(share.unsqueeze(1), coefs_state).squeeze(1))
            out_mass.append(mass)
            out_peak.append(share.amax(dim=1, keepdim=True) * mass.clamp(0.0, 1.0))

            mv = move_p[:, i].unsqueeze(1)
            ins = install_p[:, i].unsqueeze(1)

            sim_s = torch.bmm(keys, a_src[:, i, :].unsqueeze(2)).squeeze(2)
            p_s = torch.softmax(sim_s * s_move, dim=1)
            peak_s = (sim_s - free_pen * (1.0 - occ)).amax(dim=1, keepdim=True)
            conf = torch.sigmoid(s_conf * (peak_s - conf_thresh))
            w_m = (mv * conf * p_s * occ).clamp(0.0, 1.0)
            keys = self._regmem_unit(
                keys + w_m.unsqueeze(-1) * (a_dst[:, i, :].unsqueeze(1) - keys))

            peak_r = (sim_r - free_pen * (1.0 - occ)).amax(dim=1, keepdim=True)
            hit = torch.sigmoid(s_hit * (peak_r - hit_thresh))
            alloc_logit = (1.0 - occ) * s_alloc + alloc_bias
            p_alloc_soft = torch.softmax(alloc_logit, dim=1)
            p_alloc_hard = F.one_hot(alloc_logit.argmax(dim=1), self.n_slots).float()
            p_alloc = p_alloc_hard + p_alloc_soft - p_alloc_soft.detach()
            w_i = (ins * (hit * share + (1.0 - hit) * p_alloc)).clamp(0.0, 1.0)
            onehot = eye[i].view(1, 1, n_cmd)
            coefs_state = coefs_state + w_i.unsqueeze(-1) * (onehot - coefs_state)
            keys = self._regmem_unit(keys + w_i.unsqueeze(-1) * (ar.unsqueeze(1) - keys))
            occ = occ * leak
            occ = occ + w_i * (1.0 - occ)

        coef = torch.stack(out_coef, dim=1)
        mass_t = torch.stack(out_mass, dim=1)
        peak_t = torch.stack(out_peak, dim=1)
        coef = torch.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        return coef, mass_t, peak_t

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 2:
            return pred, h_out
        n_cmd = (L + 1) // 2
        if n_cmd < 1:
            return pred, h_out

        device = tok_emb.device
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        wdt = self.regmem_feat_in.weight.dtype
        cmd_raw = tok_emb[:, 0::2, :].to(wdt)
        obs_raw = self._regmem_pad(tok_emb[:, 1::2, :].to(wdt), n_cmd)
        obs_live = (self._regmem_pad(valid_obs, n_cmd) & valid_cmd).float()

        f, a_read, a_src, a_dst = self._regmem_addresses(cmd_raw)
        ctrl = F.gelu(self.regmem_ctrl_in(torch.cat([f, self.regmem_obs_feat(obs_raw)], dim=-1)))
        move_p = torch.sigmoid(self.regmem_move_out(ctrl).float()).squeeze(-1) * obs_live
        install_p = torch.sigmoid(self.regmem_install_out(ctrl).float()).squeeze(-1) * obs_live

        coef, mass_t, peak_t = self._regmem_scan(a_read, a_src, a_dst, move_p, install_p, n_cmd)

        read = torch.bmm(coef, obs_raw.float())
        read = torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4)
        rms = (read.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass_t.clamp(0.0, 1.0), peak_t.clamp(0.0, 1.0), rms], dim=-1)

        gi = torch.cat([h_out[:, 0::2, :].float(), feat], dim=-1)
        gate = torch.sigmoid(self.regmem_gate(gi.to(self.regmem_gate.weight.dtype))).float()
        contrib = gate * self.regmem_out(read.to(self.regmem_out.weight.dtype)).float()
        contrib = contrib * valid_cmd.unsqueeze(-1).float()
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return MutableKeyRegisterFile(**params)
