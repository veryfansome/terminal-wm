TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

THE CONTRACT — axis 'arch': Expose build(**params) -> an nn.Module whose forward(tok_emb, types, key_pad) returns (pred, h). Tokens interleave command and observation embeddings; types marks which is which; key_pad marks padding. Predict at EVERY position — the harness reads command positions by stride. The module MUST be causal: perturbing a later observation may not move an earlier command's prediction, and the guard checks this before anything is scored.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any arch impl: expose `build(**params) -> nn.Module` whose
  forward(tok_emb [B,L,768], types [B,L] in {0,1}, key_pad [B,L] bool) -> (pred [B,L,768], h [B,L,dh])
predicting at EVERY position; the harness reads command positions as pred[:, 0::2]. Input tokens
are frozen 768-d embeddings and the module must map back to 768-d target space. Params come from
the genome's chunks.arch.params.

The module MUST be causal: a command-position prediction may not depend on its own or a later
observation token. The per-genome no-leakage guard rejects any arch that fails this.
"""

from realenv import seq_worldmodel as M

NAME = "baseline_transformer"
DESCRIPTION = "R4 causal transformer over interleaved cmd/obs frozen embeddings (SeqWorldModel)."


def build(d=192, layers=4, heads=4, dropout=0.1):
    return M.SeqWorldModel("jepa", d=d, layers=layers, heads=heads, dropout=dropout)
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                r5-08-depth-stratified-epoch
  its fitness       -0.0075   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r4_complement_address_transport
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r5_depth_stratified_epoch_reshuffle_blocks   params {"ans_max_sim": 0.995, "base_frac": 0.25, "block_frac_max": 0.5, "bucket_cap": 32, "depth_beta": 1.2, "depth_cap": 4, "group_size": 4, "max_filter_buckets": 1200, "max_repeat": 20.0, "min_stratum": 16, "proj_dim": 128, "ramp_frac": 0.3, "sim_thresh": 0.6}
  stream              baseline_interleave
  head                r3_occupancy_routed_copy_transport

YOUR PARENT'S CURRENT arch IMPL — r4_complement_address_transport (this is the code you are mutating):
--------------------------------------------------------------------------------
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
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- srcdst_transport_occlusion (axis arch)
import torch
import torch.nn as nn

from evolve.chunks.arch.r22_observation_occlusion_denoising import (
    R22ObservationOcclusionDenoising,
)

D = 768

NAME = "srcdst_transport_occlusion"
DESCRIPTION = (
    "The r22 occlusion-trained r18 path-state world model with a two-address latent-transition "
    "memory: each step reads the slot addressed by a SOURCE key, applies the shared "
    "command-conditioned transition to that content, and writes the result by the delta rule into "
    "the slot addressed by a separate DESTINATION key, then subtracts the source content from the "
    "source slot in proportion to a learned erase gate times the angular separation of the two "
    "keys, so a step whose two keys coincide reduces exactly to the single-address r18 update. "
    "The destination projection is initialized as a copy of the source projection, making the "
    "module bit-identical to r18 at initialization; the two projections receive different "
    "gradients (one through the read, one through the write) and separate during training. "
    "Everything else - trunk, file/path delta-rule memories, FiLM views, system summary, "
    "occlusion schedule, transition_from_emb entry point and the _transition_reads signature - is "
    "inherited unchanged."
)


class SrcDstTransportOcclusion(R22ObservationOcclusionDenoising):
    def __init__(self, erase_bias=-1.0, **params):
        super().__init__(**params)
        self.tr_dst = nn.Linear(self.d, self.key_d, bias=False)
        with torch.no_grad():
            self.tr_dst.weight.copy_(self.tr_path.weight)
        self.tr_erase_gate = nn.Linear(self.d, 1)
        nn.init.zeros_(self.tr_erase_gate.weight)
        nn.init.constant_(self.tr_erase_gate.bias, float(erase_bias))

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p_src = self._unit(self.tr_path(x_cmd))
        p_dst = self._unit(self.tr_dst(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        e = torch.sigmoid(self.tr_erase_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            si = p_src[:, i, :]
            di = p_dst[:, i, :]
            s_pre = torch.bmm(si.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            d_pre = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            corr_dst = (v_i - d_pre) * active
            sep = (1.0 - (si * di).sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
            erase = e[:, i].unsqueeze(-1) * sep * active
            write = torch.bmm(di.unsqueeze(2), corr_dst.unsqueeze(1))
            clear = torch.bmm(si.unsqueeze(2), (erase * s_pre).unsqueeze(1))
            mem = decay * mem + write - clear
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return SrcDstTransportOcclusion(**params)

--- r22_retrieval_composition_renderer (axis arch)
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
    "of only scalar-mixing. Bit-identical to the r18 forward at init; trained by the main "
    "loss; no aux, no new forward."
)


class R22RetrievalCompositionRenderer(R18PathStateLatentTransition):
    def __init__(self, comp_hidden=192, comp_rms_cap=1.0, comp_gate_bias=-1.0, **params):
        super().__init__(**params)
        # New modules are constructed AFTER the entire inherited __init__ so the inherited
        # parameters draw the identical init-RNG stream; reordering breaks bit-identity at init.
        self.comp_rms_cap = float(comp_rms_cap)
        comp_in_dim = 3 * D + self.d + self.ctx_d + self.sys_d
        ch = max(32, int(comp_hidden))
        self.comp_ln = nn.LayerNorm(comp_in_dim)
        self.comp_in = nn.Linear(comp_in_dim, ch)
        self.comp_out = nn.Linear(ch, D)
        nn.init.zeros_(self.comp_out.weight)
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

        comp_z = torch.cat(
            [r_obs.to(x.dtype), rv, rp, h_cmd0, r_ctx.to(x.dtype), s_cmd.to(x.dtype)], dim=-1
        )
        comp_h = torch.nn.functional.gelu(self.comp_in(self.comp_ln(comp_z)))
        comp_raw = self.comp_out(comp_h)
        comp_raw = torch.nan_to_num(comp_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        rms = (comp_raw.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        scale = (self.comp_rms_cap / rms.clamp_min(self.comp_rms_cap)).detach()
        comp = comp_raw * scale
        g_comp = torch.sigmoid(self.comp_gate(torch.cat([h_cmd0, read_feat, rms.to(x.dtype)], dim=-1)))
        target_read = target_read + (g_comp * comp).to(dtype)

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

--- r5_dualaddress_transport_closure (axis arch)
import math

import torch
import torch.nn as nn

D = 768

NAME = "r5_dualaddress_transport_closure"
DESCRIPTION = (
    "The reference-closure transport trunk with its per-step latent memory turned into a "
    "TWO-ADDRESS content transporter. One shared linear address map is applied to two fixed, "
    "equal-width coordinate regions of the raw command token: the first gives the step's READ "
    "address, the second its WRITE address, and the two coincide exactly when a coding stream "
    "puts the same pattern in both regions. Each step reads the content currently held at its "
    "read address, blends it with the step's own observation through a learned "
    "observation-informativeness gate and the shared command-conditioned affine transition "
    "operator, and writes the result at its WRITE address with the delta rule, then subtracts a "
    "learned fraction of the read content back out of the read address, scaled by how far the two "
    "addresses differ, so a step with a single address never erases itself. A step whose "
    "observation carries nothing therefore relocates the content it retrieved instead of storing "
    "an empty observation, and a chain of such steps carries one content across arbitrarily many "
    "hops. The parallel delta-rule file and path memories, the reference-closure branch with its "
    "one-shot transitive solve, the system-summary film and every readout are retained. Strictly "
    "causal: a step's read precedes its own write, so it sees only earlier observations. The "
    "transporter owns no private addressing parameters: when a caller does not supply the raw "
    "command token the two addresses come from the path memory's own read/write map pair, which "
    "the trunk trains and reads through at every prediction."
)


class R5DualAddressTransportClosure(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        ref_d=128,
        ref_pd=64,
        ref_recency_init=0.1,
        obs_info_bias=4.0,
        addr_w=224,
        src_lo=0,
        dst_lo=224,
        erase_bias=-2.0,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        self.tr_gscale = float(tr_gscale)
        self.ref_d = max(8, int(ref_d))
        self.ref_pd = max(8, int(ref_pd))
        self.addr_w = max(8, min(int(addr_w), D))
        self.src_lo = max(0, min(int(src_lo), D - self.addr_w))
        self.dst_lo = max(0, min(int(dst_lo), D - self.addr_w))
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(2, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.addr = nn.Linear(self.addr_w, self.key_d, bias=False)
        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_erase = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)
        nn.init.constant_(self.tr_erase.bias, float(erase_bias))
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        self.obs_info = nn.Linear(self.d, 1)
        nn.init.constant_(self.obs_info.bias, float(obs_info_bias))

        self.ref_ln = nn.LayerNorm(D)
        self.ref_q = nn.Linear(D, self.ref_d, bias=False)
        self.ref_k = nn.Linear(D, self.ref_d, bias=False)
        self.ref_p = nn.Linear(D, self.ref_pd, bias=False)
        self.ref_diff = nn.Linear(self.ref_pd, 1)
        nn.init.zeros_(self.ref_diff.bias)
        self.ref_keybias = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_keybias.bias)
        self.ref_null = nn.Linear(D, 1)
        nn.init.zeros_(self.ref_null.bias)
        self.ref_own = nn.Linear(D + self.d, 1)
        nn.init.zeros_(self.ref_own.bias)
        self.ref_gate = nn.Linear(self.d + 3, 1)
        nn.init.zeros_(self.ref_gate.bias)
        self.ref_read = nn.Linear(D, D)
        nn.init.zeros_(self.ref_read.weight)
        nn.init.zeros_(self.ref_read.bias)
        rr = max(1e-4, float(ref_recency_init))
        self.ref_recency = nn.Parameter(torch.tensor(math.log(math.expm1(rr))))

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _step_addresses(self, x_cmd, cmd_raw):
        if cmd_raw is None:
            return self._unit(self.path_read(x_cmd)), self._unit(self.path_write(x_cmd))
        dtype = x_cmd.dtype
        s0 = self.src_lo
        d0 = self.dst_lo
        w = self.addr_w
        a_read = self._unit(self.addr(cmd_raw[..., s0:s0 + w].to(dtype)))
        a_write = self._unit(self.addr(cmd_raw[..., d0:d0 + w].to(dtype)))
        return a_read, a_write

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair, cmd_raw=None):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        a_read, a_write = self._step_addresses(x_cmd, cmd_raw)
        mut = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        info = torch.sigmoid(self.obs_info(x_cmd)).squeeze(-1)
        align = (a_read * a_write).sum(dim=-1).clamp(-1.0, 1.0)
        erase = torch.sigmoid(self.tr_erase(x_cmd)).squeeze(-1) * (1.0 - align * align)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            ri = a_read[:, i, :]
            wi = a_write[:, i, :]
            s_src = torch.bmm(ri.unsqueeze(1), mem).squeeze(1)
            reads.append(s_src)
            delta = self._transition(s_src, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                gi = mut[:, i].unsqueeze(-1)
            else:
                obs_i = s_src.new_zeros(B, D)
                active = x_cmd.new_zeros(B, 1)
                gi = mut[:, i].unsqueeze(-1) * 0.0
            fi = info[:, i].unsqueeze(-1)
            base_i = fi * obs_i + (1.0 - fi) * s_src
            v_i = (1.0 - gi) * base_i + gi * delta
            s_dst = torch.bmm(wi.unsqueeze(1), mem).squeeze(1)
            corr_w = (v_i - s_dst) * active
            corr_e = (-erase[:, i].unsqueeze(-1) * s_src) * active
            addr_pair = torch.stack([wi, ri], dim=2)
            corr_pair = torch.stack([corr_w, corr_e], dim=1)
            mem = decay * mem + torch.bmm(addr_pair, corr_pair)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def _reference_closure(self, cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd):
        B = cmd_raw.size(0)
        device = cmd_raw.device
        dtype = h_cmd0.dtype

        u = self.ref_ln(cmd_raw).to(dtype)
        q = self.ref_q(u)
        k = self.ref_k(u)
        p = self.ref_p(u)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(float(self.ref_d))
        mismatch = (p.unsqueeze(2) - p.unsqueeze(1)).abs()
        scores = scores + self.ref_diff(mismatch).squeeze(-1)
        scores = scores + self.ref_keybias(u).transpose(1, 2)

        idx = torch.arange(n_cmd, device=device)
        gap = (idx.view(n_cmd, 1) - idx.view(1, n_cmd)).clamp_min(0).to(dtype)
        scores = scores - torch.nn.functional.softplus(self.ref_recency).to(dtype) * gap.unsqueeze(0)

        allowed = (idx.view(n_cmd, 1) > idx.view(1, n_cmd)).unsqueeze(0) & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        full = torch.cat([scores, self.ref_null(u)], dim=2)
        attn = torch.softmax(full, dim=2)[:, :, :n_cmd]
        attn = attn * valid_cmd.unsqueeze(-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        own = torch.sigmoid(self.ref_own(torch.cat([u, h_cmd0], dim=-1)))
        own = own * valid_cmd.unsqueeze(-1).to(dtype)

        transfer = (1.0 - own) * attn
        eye = torch.eye(n_cmd, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        system = eye - transfer
        rhs = own * obs_pad.to(dtype)
        registers = self._solve_lower(system, rhs)
        registers = torch.nan_to_num(registers, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        pulled = torch.bmm(attn, registers)
        pulled = torch.nan_to_num(pulled, nan=0.0, posinf=1e4, neginf=-1e4)
        sharp = attn.amax(dim=2, keepdim=True)
        mass = attn.sum(dim=2, keepdim=True)
        return pulled, sharp, mass

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
        cmd_raw = tok_emb[:, 0::2, :]

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

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(
            x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair, cmd_raw
        )
        g_tr = torch.sigmoid(self.tr_read_gate(h_cmd0))
        tr_contrib = g_tr * self.tr_read(tr_reads.to(x.dtype))

        obs_pad = self._pad_steps(obs_for_prev, n_cmd)
        pulled, sharp, mass = self._reference_closure(cmd_raw, h_cmd0, obs_pad, valid_cmd, n_cmd)
        chain_render = self._transition(pulled, x_cmd)
        chain_feat = torch.cat(
            [(pulled.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt(), sharp, mass], dim=-1
        )
        g_ref = torch.sigmoid(self.ref_gate(torch.cat([h_cmd0, chain_feat], dim=-1)))
        ref_contrib = g_ref * self.ref_read(chain_render)
        ref_contrib = torch.nan_to_num(ref_contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        sf_h = torch.tanh(self.sysfilm_in(torch.cat([h_cmd0, s_cmd.to(x.dtype)], dim=-1)))
        sf = self.sysfilm_out(sf_h).to(dtype)
        g_sys = sf[..., :D]
        b_sys = sf[..., D:]

        h_out = self.out_norm(h_base).clone()
        h_out[:, 0::2, :] = h_cmd
        pred = self.head(h_out).clone()
        pred_cmd = pred[:, 0::2, :] + torch.sigmoid(self.direct_gate(fuse_in)).to(dtype) * target_read
        pred_cmd = pred_cmd + tr_contrib.to(dtype) + ref_contrib.to(dtype)
        pred_cmd = pred_cmd * (1.0 + g_sys) + b_sys
        pred[:, 0::2, :] = torch.nan_to_num(pred_cmd, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.nan_to_num(pred * valid.unsqueeze(-1).to(pred.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        h_out = torch.nan_to_num(h_out * valid.unsqueeze(-1).to(h_out.dtype), nan=0.0, posinf=1e4, neginf=-1e4)
        return pred, h_out


def build(**params):
    return R5DualAddressTransportClosure(**params)

--- r19_obspresent_imagination_worldmodel (axis arch)
import math

import torch
import torch.nn as nn

D = 768

NAME = "r19_obspresent_imagination_worldmodel"
DESCRIPTION = (
    "R18 per-path latent-transition world model + an obs-ABSENT IMAGINATION path: a third "
    "slot type (type 2, an imagined command with no observation, at a command position) lets "
    "the learned transition operator f(s,cmd)=s*(1+gamma)+beta be COMPOSED without an obs — the "
    "imagined command writes to its path slot using its current retrieved content s_pre in place "
    "of the missing obs, and its paired obs slot contributes nothing (obs_present decoupled from "
    "valid_cmd across every obs branch). Bit-for-bit the R18 stack when no type-2 slots exist; "
    "strictly causal (the write depends only on s_pre + the command, never on future/absent obs)."
)


class R19ObsPresentImaginationWorldModel(nn.Module):
    def __init__(
        self,
        d=176,
        layers=4,
        heads=4,
        key_d=64,
        ctx_d=96,
        n_verb=8,
        film_hidden=128,
        sys_d=64,
        sysfilm_hidden=128,
        tr_hidden=192,
        tr_gscale=0.5,
        ffn_mult=2,
        dropout=0.1,
        chunk_size=16,
        **unused,
    ):
        super().__init__()
        if "k" in unused:
            key_d = unused["k"]
        if "chunk" in unused:
            chunk_size = unused["chunk"]

        self.D = D
        self.d = int(d)
        self.layers = max(1, int(layers))
        self.key_d = int(key_d)
        self.ctx_d = int(ctx_d)
        self.n_verb = max(1, int(n_verb))
        self.sys_d = max(8, int(sys_d))
        self.chunk_size = max(1, int(chunk_size))
        self.tr_gscale = float(tr_gscale)
        ffn_h = max(self.d, int(float(ffn_mult) * self.d))

        self.cmd_proj = nn.Linear(D, self.d)
        self.obs_proj = nn.Linear(D, self.d)
        self.type_emb = nn.Embedding(3, self.d)
        self.in_norm = nn.LayerNorm(self.d)
        self.pos_scale = nn.Parameter(torch.tensor(0.2))

        enc = nn.TransformerEncoderLayer(
            self.d,
            int(heads),
            ffn_h,
            float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc, self.layers, enable_nested_tensor=False)

        self.file_read = nn.Linear(self.d, self.key_d, bias=False)
        self.file_write = nn.Linear(self.d, self.key_d, bias=False)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, self.key_d) * 0.2)
        self.ctx_proj = nn.Linear(self.d, self.ctx_d)

        self.path_read = nn.Linear(self.d, self.key_d, bias=False)
        self.path_write = nn.Linear(self.d, self.key_d, bias=False)

        self.write_gate = nn.Linear(2 * self.d, 1)

        fh = max(16, int(film_hidden))
        self.film_in = nn.Linear(self.d + self.ctx_d, fh)
        self.film_out = nn.Linear(fh, 2 * D)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

        self.sys_sal = nn.Linear(self.d, 1)
        self.sys_val = nn.Linear(self.d, self.sys_d)
        sh = max(16, int(sysfilm_hidden))
        self.sysfilm_in = nn.Linear(self.d + self.sys_d, sh)
        self.sysfilm_out = nn.Linear(sh, 2 * D)
        nn.init.zeros_(self.sysfilm_out.weight)
        nn.init.zeros_(self.sysfilm_out.bias)

        self.read_mix = nn.Linear(self.d + 3, 3)
        self.read_to_h = nn.Linear(D, self.d)
        self.fuse_gate = nn.Linear(2 * self.d + 3, self.d)
        self.direct_gate = nn.Linear(2 * self.d + 3, 1)
        self.out_norm = nn.LayerNorm(self.d)
        self.head = nn.Linear(self.d, D)

        self.tr_path = nn.Linear(self.d, self.key_d, bias=False)
        th = max(32, int(tr_hidden))
        self.tr_in = nn.Linear(self.d, th)
        self.tr_out = nn.Linear(th, 2 * D)
        self.tr_mut_gate = nn.Linear(self.d, 1)
        self.tr_read = nn.Linear(D, D)
        nn.init.zeros_(self.tr_read.weight)
        nn.init.zeros_(self.tr_read.bias)
        self.tr_read_gate = nn.Linear(self.d, 1)
        nn.init.constant_(self.tr_mut_gate.bias, -1.0)
        nn.init.constant_(self.tr_read_gate.bias, 0.0)

        init_decay = (0.985 - 0.90) / 0.099
        self.logit_decay = nn.Parameter(torch.tensor(math.log(init_decay / (1.0 - init_decay))))

        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.fuse_gate.bias, -1.0)
        nn.init.constant_(self.direct_gate.bias, -2.0)

    def _positional(self, L, device, dtype):
        half = (self.d + 1) // 2
        pos = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)
        div = torch.exp(
            torch.arange(half, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(1, half - 1))
        )
        pe = torch.zeros(L, self.d, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d // 2])
        return pe

    @staticmethod
    def _unit(x):
        return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True) + 1e-12)

    @staticmethod
    def _pad_steps(x, n):
        cur = x.size(1)
        if cur == n:
            return x
        if cur > n:
            return x[:, :n]
        pad_shape = (x.size(0), n - cur) + tuple(x.shape[2:])
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _quotient(self, k, Q):
        coef = torch.matmul(k, Q.transpose(0, 1))
        return self._unit(k - torch.matmul(coef, Q))

    def _solve_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        C = system.size(1)
        for i in range(C):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                corr = torch.bmm(system[:, i : i + 1, :i], prev).squeeze(1)
                yi = yi - corr
            yi = yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6)
            parts.append(yi)
        return torch.stack(parts, dim=1)

    def _chunked_delta_reads(self, q, k, value, beta, lam):
        B, N, K = q.shape
        V = value.size(-1)
        if N == 0:
            return value.new_zeros(B, 0, V)

        dtype = value.dtype
        q = q.to(dtype)
        k = k.to(dtype)
        beta = beta.to(dtype)
        lam = lam.to(dtype).clamp(0.90, 1.0)

        mem = value.new_zeros(B, K, V)
        outs = []
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            qc = q[:, start:end, :]
            kc = k[:, start:end, :]
            vc = value[:, start:end, :]
            bc = beta[:, start:end]
            lc = lam[:, start:end]
            C = end - start

            prefix = torch.cumprod(lc, dim=1)
            before = torch.cat(
                [torch.ones(B, 1, device=value.device, dtype=dtype), prefix[:, :-1]], dim=1
            )
            denom = prefix.clamp_min(1e-6)
            between = before.unsqueeze(2) / denom.unsqueeze(1)

            strict = torch.tril(torch.ones(C, C, device=value.device, dtype=torch.bool), diagonal=-1)
            strict = strict.unsqueeze(0).to(dtype)

            kk = torch.bmm(kc, kc.transpose(1, 2))
            lower = kk * between * bc.unsqueeze(1) * strict

            rhs = vc - before.unsqueeze(-1) * torch.bmm(kc, mem)
            eye = torch.eye(C, device=value.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            err = self._solve_lower(eye + lower, rhs)
            err = torch.nan_to_num(err, nan=0.0, posinf=1e4, neginf=-1e4)

            qk = torch.bmm(qc, kc.transpose(1, 2))
            weights = qk * between * bc.unsqueeze(1) * strict
            read = before.unsqueeze(-1) * torch.bmm(qc, mem) + torch.bmm(weights, err)
            outs.append(torch.nan_to_num(read, nan=0.0, posinf=1e4, neginf=-1e4))

            end_factor = prefix[:, -1]
            end_between = end_factor.unsqueeze(1) / denom
            contrib = torch.bmm(kc.transpose(1, 2), err * (bc * end_between).unsqueeze(-1))
            mem = end_factor.view(B, 1, 1) * mem + contrib
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        return torch.cat(outs, dim=1)

    def _transition(self, s_pre, cmd_feat):
        hin = torch.nn.functional.gelu(self.tr_in(cmd_feat))
        gb = self.tr_out(hin)
        gamma = torch.tanh(gb[..., :D]) * self.tr_gscale
        beta = gb[..., D:]
        return s_pre * (1.0 + gamma) + beta

    def transition_from_emb(self, s_pre, cmd_emb):
        """Apply the shared transition operator to a current-content estimate `s_pre` [N,D]
        and a raw standardized command embedding `cmd_emb` [N,D]; returns the post-transition
        content [N,D]. Entry point for the head aux that supervises the operator."""
        idx0 = torch.zeros(cmd_emb.size(0), dtype=torch.long, device=cmd_emb.device)
        cmd_feat = self.in_norm(self.cmd_proj(cmd_emb) + self.type_emb(idx0))
        return self._transition(s_pre, cmd_feat)

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, imagined, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            imagined_i = imagined[:, i]
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                obs_present_i = valid_obs[:, i] & ~imagined_i
            else:
                obs_i = s_pre.new_zeros(B, D)
                obs_present_i = torch.zeros(B, dtype=torch.bool, device=x_cmd.device)
            wi = w[:, i].unsqueeze(-1)
            base_i = torch.where(obs_present_i.unsqueeze(-1), obs_i, s_pre)
            write_active_i = valid_cmd[:, i] & (obs_present_i | imagined_i)
            v_i = (1.0 - wi) * base_i + wi * delta
            corr = (v_i - s_pre) * write_active_i.to(dtype).unsqueeze(-1)
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, tok_emb, types, key_pad):
        B, L, _ = tok_emb.shape
        device = tok_emb.device
        dtype = tok_emb.dtype

        if L == 0:
            h0 = tok_emb.new_zeros(B, 0, self.d)
            return tok_emb.new_zeros(B, 0, D), h0

        t = types.long().clamp(0, 2)
        pad_mask = key_pad.bool() if key_pad is not None else None
        valid = ~pad_mask if pad_mask is not None else torch.ones(B, L, device=device, dtype=torch.bool)

        cmd_x = self.cmd_proj(tok_emb)
        obs_x = self.obs_proj(tok_emb)
        x = torch.where((t == 1).unsqueeze(-1), obs_x, cmd_x)
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
        imagined = (t[:, 0::2] == 2)
        obs_present_pair = valid_obs & ~imagined[:, :n_pair]
        active_pair = valid_cmd[:, :n_pair] & obs_present_pair

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

        if n_pair:
            sal = torch.sigmoid(self.sys_sal(h_obs)).squeeze(-1)
            sal = sal * obs_present_pair.to(sal.dtype)
            v_sys = torch.tanh(self.sys_val(h_obs))
            num = torch.cumsum(sal.unsqueeze(-1) * v_sys, dim=1)
            den = torch.cumsum(sal, dim=1).unsqueeze(-1)
            s_incl = num / (den + 1e-6)
            s_cmd = torch.cat([s_incl.new_zeros(B, 1, self.sys_d), s_incl], dim=1)
            s_cmd = self._pad_steps(s_cmd, n_cmd)
        else:
            s_cmd = x.new_zeros(B, n_cmd, self.sys_d)

        obs_for_prev = obs_tok * obs_present_pair.unsqueeze(-1).to(dtype)
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

        mem_h = self.read_to_h(target_read.to(x.dtype))
        fuse_in = torch.cat([h_cmd0, mem_h, read_feat], dim=-1)
        h_cmd = self.out_norm(h_cmd0 + torch.sigmoid(self.fuse_gate(fuse_in)) * mem_h)

        tr_reads = self._transition_reads(x_cmd, obs_tok, valid_cmd, valid_obs, imagined, n_cmd, n_pair)
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
    return R19ObsPresentImaginationWorldModel(**params)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0075 (r5-08-depth-stratified-epoch, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
