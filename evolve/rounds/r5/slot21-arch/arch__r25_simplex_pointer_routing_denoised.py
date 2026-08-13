import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r25_simplex_pointer_routing_denoised"
DESCRIPTION = (
    "The r22 prefix-content arch (r18 path-state trunk + prefix-observation cross-attention) with "
    "two additions. (1) A POINTER-SIMPLEX routing store: the store's values are distributions over "
    "STEP INDICES of the same window rather than 768-d content, so its size is [key_d, n_steps]. "
    "Addresses come from a soft-thresholded, sign-preserving coordinate map of two fixed disjoint "
    "coordinate blocks of the raw command embedding, pushed through ONE shared projection with a "
    "per-role additive bias, so the same block pattern yields the same address whichever block it "
    "sits in. At each command the model reads the distribution at its source address, projects it "
    "back to the simplex (rectify, learned power sharpening, sum-normalise, linear mass ramp), and "
    "writes at its destination address either that distribution (transport) or the one-hot of the "
    "current step (exposure) under a delta rule, erasing the source by a gated amount; all gates "
    "are functions of the command token and the causal hidden state only, never of any observation "
    "value. The distribution read at a command is masked to strictly earlier live steps, "
    "renormalised, and used to mix the raw observation embeddings of those steps; the mixture is "
    "injected into the command-position prediction through a small identity-initialised (D,D) "
    "readout under a sigmoid gate fed the pointer mass and peak. Composition of moves is therefore "
    "a product of near-stochastic pointer updates and the retrieved vector is always a convex "
    "combination of observations that actually occurred, with no content passing through the store. "
    "(2) Ramped stochastic observation occlusion during training only: each observation token is "
    "independently zeroed and key-padded with a probability ramping to occ_p, which also removes "
    "that step from the pointer's origin pool while leaving move transport intact. The eval-mode "
    "forward has no occlusion."
)

_EPS = 1e-6
_MAXP = 4.0


class R25SimplexPointerRoutingDenoised(R22PrefixContentXAttention):
    def __init__(
        self,
        src_lo=0,
        src_hi=352,
        dst_lo=352,
        dst_hi=704,
        addr_thresh=1.25,
        ptr_key_d=128,
        gate_hidden=96,
        ptr_gate_bias=-2.0,
        ptr_readout_init=0.05,
        ptr_move_bias=0.0,
        ptr_write_bias=1.0,
        ptr_erase_bias=0.0,
        ptr_decay=0.999,
        ptr_mass_tau=0.25,
        occ_p=0.12,
        occ_ramp_start=300,
        occ_ramp_end=1000,
        **params,
    ):
        super().__init__(**params)

        lo_s = max(0, min(int(src_lo), D - 1))
        lo_d = max(0, min(int(dst_lo), D - 1))
        span = min(int(src_hi) - lo_s, int(dst_hi) - lo_d, D - lo_s, D - lo_d)
        if span < 8:
            lo_s, lo_d = 0, D // 2
            span = D // 2
        self.addr_src_lo = lo_s
        self.addr_dst_lo = lo_d
        self.addr_block = int(span)

        self.ptr_decay = float(ptr_decay)
        self.ptr_key_d = max(16, int(ptr_key_d))

        self.addr_proj = nn.Linear(self.addr_block, self.ptr_key_d, bias=False)
        self.addr_role = nn.Parameter(torch.zeros(2, self.ptr_key_d))
        th = max(1e-3, float(addr_thresh))
        self.addr_thresh_raw = nn.Parameter(torch.tensor(math.log(math.expm1(th))))

        gh = max(16, int(gate_hidden))
        self.ptr_feat = nn.Linear(D, gh)
        self.ptr_move = nn.Linear(self.d + gh, 1)
        self.ptr_write = nn.Linear(self.d + gh, 1)
        self.ptr_erase = nn.Linear(self.d + gh, 1)
        nn.init.constant_(self.ptr_move.bias, float(ptr_move_bias))
        nn.init.constant_(self.ptr_write.bias, float(ptr_write_bias))
        nn.init.constant_(self.ptr_erase.bias, float(ptr_erase_bias))

        self.ptr_log_sharp = nn.Parameter(torch.tensor(-6.0))
        tau = max(1e-2, float(ptr_mass_tau))
        self.ptr_mass_raw = nn.Parameter(torch.tensor(math.log(math.expm1(tau))))

        self.ptr_out = nn.Linear(D, D)
        with torch.no_grad():
            self.ptr_out.weight.copy_(torch.eye(D) * float(ptr_readout_init))
            self.ptr_out.bias.zero_()
        self.ptr_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.ptr_gate.bias, float(ptr_gate_bias))

        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def _occ_prob(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return self.occ_p
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return self.occ_p * (x * x * (3.0 - 2.0 * x))

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

    def _role_addresses(self, cmd_raw):
        xs = cmd_raw[..., self.addr_src_lo:self.addr_src_lo + self.addr_block]
        xd = cmd_raw[..., self.addr_dst_lo:self.addr_dst_lo + self.addr_block]
        theta = F.softplus(self.addr_thresh_raw).to(cmd_raw.dtype)
        xs = torch.sign(xs) * torch.relu(xs.abs() - theta)
        xd = torch.sign(xd) * torch.relu(xd.abs() - theta)
        wd = self.addr_proj.weight.dtype
        ks = self.addr_proj(xs.to(wd)).to(cmd_raw.dtype) + self.addr_role[0].to(
            cmd_raw.dtype).view(1, 1, -1)
        kd = self.addr_proj(xd.to(wd)).to(cmd_raw.dtype) + self.addr_role[1].to(
            cmd_raw.dtype).view(1, 1, -1)
        return self._unit(ks), self._unit(kd)

    def _sharp_exponent(self):
        return (1.0 + F.softplus(self.ptr_log_sharp)).clamp(1.0, _MAXP)

    def _pointer_simplex(self, p):
        pos = torch.relu(p)
        mass = pos.sum(dim=-1, keepdim=True)
        x = pos.clamp_min(_EPS).pow(self._sharp_exponent().to(p.dtype))
        x = x / x.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        tau = F.softplus(self.ptr_mass_raw).to(p.dtype).clamp_min(0.05)
        ramp = (mass / tau).clamp(0.0, 1.0)
        return torch.nan_to_num(x * ramp, nan=0.0, posinf=0.0, neginf=0.0)

    def _pointer_route(self, a_src, a_dst, g_move, g_write, g_erase, live, cmd_valid, n_cmd):
        B = a_src.size(0)
        dtype = a_src.dtype
        mem = a_src.new_zeros(B, self.ptr_key_d, n_cmd)
        eye = torch.eye(n_cmd, device=a_src.device, dtype=dtype)
        reads = []
        for i in range(n_cmd):
            ai = a_src[:, i, :]
            di = a_dst[:, i, :]
            p_i = torch.bmm(ai.unsqueeze(1), mem).squeeze(1)
            reads.append(p_i)
            q_i = self._pointer_simplex(p_i)
            self_i = eye[i].view(1, -1) * live[:, i].unsqueeze(-1)
            gm = g_move[:, i, :]
            v_i = gm * q_i + (1.0 - gm) * self_i
            cur = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            cv = cmd_valid[:, i].unsqueeze(-1)
            w_write = (g_write[:, i, :] * cv) * (v_i - cur)
            w_erase = -(g_erase[:, i, :] * gm * cv) * p_i
            upd_a = torch.stack([di, ai], dim=2)
            upd_v = torch.stack([w_write, w_erase], dim=1)
            mem = torch.baddbmm(mem, upd_a, upd_v, beta=self.ptr_decay)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        out = torch.stack(reads, dim=1)
        return torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        tok_emb, key_pad = self._occlude(tok_emb, key_pad)
        pred, h_out = super().forward(tok_emb, types, key_pad)

        B, L, _ = tok_emb.shape
        if L < 4:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair < 1:
            return pred, h_out

        device = tok_emb.device
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        obs_live = self._pad_steps(valid_obs, n_cmd) & valid_cmd

        cmd_raw = tok_emb[:, 0::2, :]
        obs_pad = self._pad_steps(tok_emb[:, 1::2, :], n_cmd)
        obs_pad = obs_pad * obs_live.unsqueeze(-1).to(dtype)
        h_cmd = h_out[:, 0::2, :].to(dtype)

        a_src, a_dst = self._role_addresses(cmd_raw)

        g_feat = F.gelu(self.ptr_feat(cmd_raw)).to(dtype)
        gi = torch.cat([h_cmd, g_feat], dim=-1)
        g_move = torch.sigmoid(self.ptr_move(gi)).to(dtype)
        g_write = torch.sigmoid(self.ptr_write(gi)).to(dtype)
        g_erase = torch.sigmoid(self.ptr_erase(gi)).to(dtype)

        p_raw = self._pointer_route(
            a_src, a_dst, g_move, g_write, g_erase,
            obs_live.to(dtype), valid_cmd.to(dtype), n_cmd)

        pos = torch.relu(p_raw)
        mass = pos.sum(dim=-1, keepdim=True)
        x = pos.clamp_min(_EPS).pow(self._sharp_exponent().to(dtype))

        idx = torch.arange(n_cmd, device=device)
        causal = (idx.view(1, -1) < idx.view(-1, 1)).unsqueeze(0)
        allowed = causal & obs_live.unsqueeze(1)
        x = x * allowed.to(dtype)
        has_key = allowed.any(dim=-1, keepdim=True).to(dtype)
        p_norm = x / x.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        p_norm = torch.nan_to_num(p_norm * has_key, nan=0.0, posinf=0.0, neginf=0.0)

        copy = torch.bmm(p_norm, obs_pad)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)

        sharp = p_norm.amax(dim=-1, keepdim=True)
        rms = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass.clamp(0.0, 1.0), sharp, rms], dim=-1)
        gate = torch.sigmoid(self.ptr_gate(torch.cat([h_cmd, feat], dim=-1))).to(dtype)

        read = self.ptr_out(copy.to(self.ptr_out.weight.dtype)).to(dtype)
        contrib = gate * read * valid_cmd.unsqueeze(-1).to(dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R25SimplexPointerRoutingDenoised(**params)
