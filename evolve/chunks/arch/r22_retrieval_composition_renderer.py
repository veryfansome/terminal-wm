"""R22 arch: RETRIEVAL-COMPOSITION RENDERER — the r18 path-state world model with ONE
trunk-internal upgrade at its measured content-attribution carrier: a zero-init, RMS-capped
NONLINEAR COMPOSER that lets the three retrieved-content channels (file-memory view, path-memory
read, raw file-memory retrieval) interact VECTOR-WISE, conditioned on the trunk state, the
retrieval context and the system summary, before entering the prediction. Identical to the r18
function at init; trained by the MAIN loss inside the single pass (no aux, no second forward,
no mining).

WHY THIS SITE (measured, R22 inventor-3 probes on the canonical pod-lane r18 checkpoints
r18_stack_rssm.s{0,1}.st4000, instrument-exact b-layout + wrong-history donors built on
TRAIN-image windows, 3072 genuine windows, strict-tie top-1 vs 63 same-verb foils):
  * The masked-endpoint CONTENT differential (CA* = b_top1 - wrong_top1) is NOT carried by trunk
    attention: zeroing all read-side memory channels collapses CA* 0.5608 -> 0.1172 (s0) /
    0.5588 -> 0.1318 (s1). The delta-rule memories reading PREFIX OBSERVATIONS are the carrier:
    no_filemem 0.5608 -> 0.4541 (s0) / 0.2432 (s1); no_pathmem -> 0.4448 (s0).
    Per family (s0): mv rides the FILE memory (0.4152 -> 0.2697 without it), mkdir rides the
    PATH memory (0.4764 -> 0.0472 without it).
  * What the memories DELIVER at the masked read is related-but-not-the-answer content that
    needs TRANSFORMATION: median cos(retrieved, z_r) — mv r_obs 0.303, redir:prod> 0.102;
    and the r18 arch's ONE shared diagonal-affine FiLM is family-conflicted: it lifts
    redir:prod> 0.102 -> 0.357 but DESTROYS mkdir's 0.767 -> 0.040 (mkdir escapes via the
    scalar mix to the path channel). The channels can only be scalar-mixed (read_mix softmax);
    content vectors never interact.
  * The syscond FiLM channel is content-differential-NEGATIVE: removing it RAISES CA*
    (0.5608 -> 0.5785 s0, 0.5588 -> 0.5925 s1) while lowering b_top1 — a shared-template
    render that helps both arms. System information should enter CONDITIONED ON retrieved
    content, not as a content-free output modulation.
  * Measured earlier (R20 finding 7, full 3-seed): at the analogous transform site
    (s_pre_m, c_m) -> z_r on the frozen r18 arch's memory content, the r18 affine
    functional form SATURATES where an MLP adds +0.057 top-1 (0.510 affine vs 0.569 MLP).

MECHANISM. At every command position the r18 forward computes r_obs / r_view (file memory + FiLM
view), read_path (path memory), prev_obs, mixes them with SCALAR softmax weights into
target_read, and injects. This arch adds, at the same point:

    z    = LN([r_obs, r_view, read_path, h_cmd0, r_ctx, s_cmd])
    comp = W2 gelu(W1 z)          # W2 ZERO-INIT  -> exactly the r18 function at init
    comp = comp * min(1, cap/RMS(comp))   (detached per-row RMS projection, cap=1.0 —
                                           the R10/R11 norm-inflation rail, applied
                                           structurally since the co-designed optim's
                                           spectral cap targets only (D,D) matrices)
    target_read += sigmoid(gate([h_cmd0, read_feat, RMS(comp)])) * comp

so the composed content flows through BOTH r18 injection routes (fuse into h_cmd and the
direct gate), exactly like real retrieved content. Everything else — memories, FiLM, syscond,
transition channel, `transition_from_emb` (head-aux co-design point), the forward contract —
is the r18 forward verbatim.

Why this can move IMAG_CA where the R20/R21 families did not: the composer's content inputs are
PREFIX-DERIVED (memory reads); any use of them is content-attributable by construction (they
differ between the real-prefix and donor-prefix arms), while command-decode components
(h_cmd0-driven) cancel between arms. It is trained by the main loss at every scored position
from step ~0 (not a detached observer; not an eval-time-only branch; not a post-hoc fine-tune),
and it upgrades the one functional form the measurements show saturating, on the one channel
the decomposition measures as carrying the selective quantity.

Causal/leak-safe: every composer input is an existing strictly-causal channel of the r18 forward
at the same position; a function of causal signals is causal. PAD-safe: all inputs are already
PAD-value-invariant (memory writes gated by validity, attention key-padded, prev zeroed).
NaN-safe: LN + bounded gelu + RMS cap + nan_to_num. Zero new (D,D) or (key_d, *) matrices, so
the co-designed optimizer's Muon/spectral-cap routing is untouched.
"""

import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r22_retrieval_composition_renderer"
DESCRIPTION = (
    "The r18 path-state world model + a zero-init, RMS-capped nonlinear COMPOSITION "
    "RENDERER over the retrieved-content channels (raw file-memory retrieval, FiLM view, "
    "path-memory read; conditioned on trunk state, retrieval context and system summary), "
    "injected into target_read so retrieved prefix contents can interact vector-wise instead "
    "of only scalar-mixing. Grounded in the R22 pathway decomposition (the delta-rule memories "
    "carry the masked-endpoint content differential; the shared diagonal FiLM is family-"
    "conflicted) and R20 finding 7 (the affine transform saturates where an MLP adds +0.057). "
    "Bit-identical to the r18 forward at init; trained by the main loss; no aux, no new forward."
)


class R22RetrievalCompositionRenderer(R18PathStateLatentTransition):
    def __init__(self, comp_hidden=192, comp_rms_cap=1.0, comp_gate_bias=-1.0, **params):
        super().__init__(**params)          # r18 params drawn FIRST -> identical init RNG
        self.comp_rms_cap = float(comp_rms_cap)
        comp_in_dim = 3 * D + self.d + self.ctx_d + self.sys_d
        ch = max(32, int(comp_hidden))
        self.comp_ln = nn.LayerNorm(comp_in_dim)
        self.comp_in = nn.Linear(comp_in_dim, ch)
        self.comp_out = nn.Linear(ch, D)
        nn.init.zeros_(self.comp_out.weight)   # exactly the r18 function at init
        nn.init.zeros_(self.comp_out.bias)
        self.comp_gate = nn.Linear(self.d + 4, 1)
        nn.init.constant_(self.comp_gate.bias, float(comp_gate_bias))

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 1)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 0).unsqueeze(-1), cmd_x, obs_x)
        x = x + self.type_emb(t) + self.pos_scale * self._positional(L, device, x.dtype).unsqueeze(0)
        x = self.in_norm(x)

        causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        h_base = self.tf(x, mask=causal, src_key_padding_mask=pad_mask)
        h_base = torch.nan_to_num(h_base, nan=0.0, posinf=1e4, neginf=-1e4)
        h_base = h_base * valid.unsqueeze(-1).to(h_base.dtype)

        n_cmd = (L + 1) // 2
        n_pair = L // 2

        x_cmd = x[:, 0::2, :]
        h_cmd0 = h_base[:, 0::2, :]
        h_obs = h_base[:, 1::2, :]
        obs_tok = tok_emb[:, 1::2, :]

        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]
        active_pair = valid_cmd[:, :n_pair] & valid_obs

        if n_pair:
            gate_in = torch.cat([h_cmd0[:, :n_pair, :], h_obs[:, :n_pair, :]], dim=-1)
            amount_pair = torch.sigmoid(self.write_gate(gate_in)).squeeze(-1)
        else:
            amount_pair = x.new_zeros(B, 0)

        write_active = self._pad_steps(active_pair, n_cmd)
        beta = self._pad_steps(amount_pair, n_cmd) * write_active.to(x.dtype)

        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        lam = torch.where(write_active, decay.to(x.dtype).expand_as(beta), torch.ones_like(beta))

        Q = self._verb_basis()
        q_file = self._quotient(self.file_read(x_cmd), Q)
        k_file = self._quotient(self.file_write(x_cmd), Q)
        ctx = self.ctx_proj(x_cmd).to(dtype)
        value_file = torch.cat([self._pad_steps(obs_tok, n_cmd), ctx], dim=-1)

        q_path = self._unit(self.path_read(h_cmd0))
        k_path = self._unit(self.path_write(h_cmd0))
        value_path = self._pad_steps(obs_tok, n_cmd)

        read_file = self._chunked_delta_reads(q_file, k_file, value_file, beta, lam)
        read_path = self._chunked_delta_reads(q_path, k_path, value_path, beta, lam)

        r_obs = read_file[..., :D]
        r_ctx = read_file[..., D:]

        film_h = torch.tanh(self.film_in(torch.cat([h_cmd0.to(dtype), r_ctx], dim=-1)))
        film = self.film_out(film_h)
        gamma = 1.0 + film[..., :D]
        shift = film[..., D:]
        r_view = torch.nan_to_num(gamma * r_obs + shift, nan=0.0, posinf=1e4, neginf=-1e4)

        # -- causal system-identity summary (r18, unchanged).
        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * valid_obs.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * valid_obs.unsqueeze(-1).to(dtype)
        if n_cmd > 1:
            prev_src = self._pad_steps(obs_for_prev, n_cmd - 1)
            prev_obs = torch.cat([tok_emb.new_zeros(B, 1, D), prev_src], dim=1)
        else:
            prev_obs = tok_emb.new_zeros(B, n_cmd, D)

        rv = r_view.to(x.dtype)
        rp = read_path.to(x.dtype)
        ro = prev_obs.to(x.dtype)
        read_feat = torch.cat(
            [
                (rv.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (rp.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
                (ro.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(),
            ],
            dim=-1,
        )

        mix = torch.softmax(self.read_mix(torch.cat([h_cmd0, read_feat], dim=-1)), dim=-1).to(dtype)
        target_read = (
            mix[:, :, 0:1] * r_view
            + mix[:, :, 1:2] * read_path
            + mix[:, :, 2:3] * prev_obs
        )

        # -- NEW: nonlinear composition renderer over the retrieved-content channels. --------
        comp_z = torch.cat(
            [r_obs.to(x.dtype), rv, rp, h_cmd0, r_ctx.to(x.dtype), s_cmd.to(x.dtype)], dim=-1
        )
        comp_h = torch.nn.functional.gelu(self.comp_in(self.comp_ln(comp_z)))
        comp_raw = self.comp_out(comp_h)                       # exactly 0 at init (zero-init W2)
        comp_raw = torch.nan_to_num(comp_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        rms = (comp_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        scale = (self.comp_rms_cap / rms.clamp_min(self.comp_rms_cap)).detach()  # <=1 projection
        comp = comp_raw * scale
        g_comp = torch.sigmoid(self.comp_gate(torch.cat([h_cmd0, read_feat, rms.to(x.dtype)], dim=-1)))
        target_read = target_read + (g_comp * comp).to(dtype)
        # ------------------------------------------------------------------------------------

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair)
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R22RetrievalCompositionRenderer(**params)
