TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

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
  id                r3-06-within-sequence-contrast
  its fitness       +0.0000   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r22_prefix_content_xattention
  optim               r18_spectral_capped_transition_readout
  target              within_sequence_contrast_equalizer
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r18_transition_forwardmodel_consistency

YOUR PARENT'S CURRENT arch IMPL — r22_prefix_content_xattention (this is the code you are mutating):
--------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r22_prefix_content_xattention"
DESCRIPTION = (
    "The r18 path-state arch + ONE new native channel: a single-head full-rank "
    "cross-attention over the strictly-earlier prefix OBSERVATIONS (keys/queries from the "
    "causal hidden states, values = raw 768-d obs embeddings — the frozen-probe computation "
    "in-trunk), injected into the command-position predictions through a zero-init (D,D) "
    "readout and a sigmoid gate, AFTER the sysfilm styling. Bit-for-bit identical to the r18 forward at init; "
    "Muon captures the new (64,176) addressing pair, the spectral cap the new (D,D) readout."
)


class R22PrefixContentXAttention(R18PathStateLatentTransition):
    def __init__(self, xattn_dim=64, xattn_gate_bias=-2.0, **params):
        super().__init__(**params)
        # New modules are constructed AFTER the entire inherited __init__ so the inherited
        # parameters draw the identical init-RNG stream; reordering breaks bit-identity at init.
        self.xattn_dim = int(xattn_dim)
        self.xq = nn.Linear(self.d, self.xattn_dim, bias=False)
        self.xk = nn.Linear(self.d, self.xattn_dim, bias=False)
        self.x_out = nn.Linear(D, D)
        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)
        self.x_gate = nn.Linear(self.d + 1, 1)
        nn.init.constant_(self.x_gate.bias, float(xattn_gate_bias))

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=tok_emb.device)
        valid_obs = valid[:, 1::2]
        h_cmd = h_out[:, 0::2, :]
        h_obs = h_out[:, 1::2, :]
        obs_val = tok_emb[:, 1::2, :]

        q = self.xq(h_cmd)
        k = self.xk(h_obs)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.xattn_dim)

        ci = torch.arange(n_cmd, device=tok_emb.device).unsqueeze(1)
        pj = torch.arange(n_pair, device=tok_emb.device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))
        o = torch.bmm(attn.to(obs_val.dtype), obs_val)
        o = torch.nan_to_num(o, nan=0.0, posinf=1e4, neginf=-1e4)

        feat = (o.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt().to(h_cmd.dtype)
        gate = torch.sigmoid(self.x_gate(torch.cat([h_cmd, feat], dim=-1)))
        contrib = gate.to(o.dtype) * self.x_out(o.to(self.x_out.weight.dtype)).to(o.dtype)
        contrib = contrib * valid[:, 0::2].unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R22PrefixContentXAttention(**params)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca +0.0000 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r22_exact_target_equivalence_quotient
  arch                r22_prefix_content_xattention
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r23_routed_content_alignment

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r24_backchain_pointer_resolution (axis arch)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r24_backchain_pointer_resolution"
DESCRIPTION = (
    "The r18 path-state trunk with r22's prefix-content cross-attention, plus a K-hop BACKWARD "
    "POINTER CHASE over the command sequence. Every command's raw 768-d embedding is mapped by "
    "one shared feature layer and two learned diagonal role gates through a SINGLE shared address "
    "projection, giving a first-path address and a second-path address in one common address "
    "space. A masked softmax over strictly-earlier commands, biased toward recency, forms a "
    "step-to-step follow matrix: from a step, jump to the earlier step whose second-path address "
    "matches this step's first-path address. Each step also emits a learned absorb probability "
    "from its command features and its own observation. Starting at a command position, the chase "
    "iterates the follow matrix K times, accumulating absorbed mass; the accumulated mass over "
    "earlier steps is renormalized into a convex combination of their raw observation embeddings "
    "and injected into that command's prediction through a small identity-initialized (D,D) "
    "readout under a sigmoid gate fed the absorbed mass and the chase sharpness. One "
    "absorb-everywhere hop reduces to a single-hop prefix copy; more hops compose several address "
    "matches into one retrieval. Strictly causal: all mass moves to strictly earlier positions."
)

_NEG = -1e9


class R24BackchainPointerResolution(R22PrefixContentXAttention):
    def __init__(
        self,
        chase_hops=8,
        addr_hidden=256,
        obs_feat_d=64,
        term_hidden=96,
        chase_temp=0.5,
        chase_gate_bias=-2.0,
        role_init=0.75,
        role_noise=0.5,
        recency_init=-1.25,
        readout_init=0.05,
        **params,
    ):
        super().__init__(**params)
        self.chase_hops = max(1, int(chase_hops))
        self.addr_hidden = max(32, int(addr_hidden))
        self.obs_feat_d = max(8, int(obs_feat_d))

        self.chase_ln = nn.LayerNorm(D)
        self.chase_in = nn.Linear(D, self.addr_hidden)
        self.role_src = nn.Parameter(
            torch.randn(self.addr_hidden) * float(role_noise) + float(role_init))
        self.role_dst = nn.Parameter(
            torch.randn(self.addr_hidden) * float(role_noise) - float(role_init))
        self.addr_proj = nn.Linear(self.addr_hidden, self.key_d, bias=False)

        self.obs_feat = nn.Linear(D, self.obs_feat_d)
        th = max(16, int(term_hidden))
        self.absorb_in = nn.Linear(self.addr_hidden + self.obs_feat_d, th)
        self.absorb_out = nn.Linear(th, 1)
        nn.init.normal_(self.absorb_out.weight, std=0.01)
        nn.init.constant_(self.absorb_out.bias, 0.0)

        t0 = max(1e-2, float(chase_temp))
        self.chase_log_temp = nn.Parameter(torch.tensor(math.log(t0)))
        self.chase_recency = nn.Parameter(torch.tensor(float(recency_init)))

        self.chase_out = nn.Linear(D, D)
        with torch.no_grad():
            self.chase_out.weight.copy_(torch.eye(D) * float(readout_init))
            self.chase_out.bias.zero_()
        self.chase_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_gate.bias, float(chase_gate_bias))

    def _follow_matrix(self, cmd_raw, valid_cmd):
        a = F.gelu(self.chase_in(self.chase_ln(cmd_raw))).float()
        addr_src = self._unit(self.addr_proj(a * torch.sigmoid(self.role_src.float())))
        addr_dst = self._unit(self.addr_proj(a * torch.sigmoid(self.role_dst.float())))

        n = a.size(1)
        device = a.device
        temp = self.chase_log_temp.float().exp().clamp(0.05, 5.0)
        scores = torch.bmm(addr_src, addr_dst.transpose(1, 2)) / temp

        idx = torch.arange(n, device=device)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1).clamp_min(0).float()
        scores = scores - F.softplus(self.chase_recency.float()) * gap.unsqueeze(0)

        strict = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        allowed = strict & valid_cmd.unsqueeze(1)
        scores = scores.masked_fill(~allowed, _NEG)
        follow = torch.softmax(scores, dim=2)
        has_key = allowed.any(dim=2, keepdim=True)
        follow = torch.where(has_key, follow, torch.zeros_like(follow))
        return a, torch.nan_to_num(follow, nan=0.0, posinf=0.0, neginf=0.0)

    def _absorb_prob(self, a, obs_pad, obs_live):
        f = torch.cat([a, self.obs_feat(obs_pad.to(self.obs_feat.weight.dtype)).float()], dim=-1)
        logit = self.absorb_out(F.gelu(self.absorb_in(f)))
        return torch.sigmoid(logit.float()) * obs_live.unsqueeze(-1).float()

    def _backchain(self, follow, absorb):
        carry = (1.0 - absorb) * follow
        alpha = follow
        acc = alpha * absorb.transpose(1, 2)
        for _ in range(self.chase_hops - 1):
            alpha = torch.bmm(alpha, carry)
            acc = acc + alpha * absorb.transpose(1, 2)
        return torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)
        B, L, _ = tok_emb.shape
        if L < 4:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair < 2:
            return pred, h_out

        device = tok_emb.device
        dtype = tok_emb.dtype
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        cmd_raw = tok_emb[:, 0::2, :]
        obs_pad = self._pad_steps(tok_emb[:, 1::2, :], n_cmd)
        obs_live = self._pad_steps(valid_obs, n_cmd) & valid_cmd

        a, follow = self._follow_matrix(cmd_raw, valid_cmd)
        absorb = self._absorb_prob(a, obs_pad, obs_live)
        acc = self._backchain(follow, absorb)

        mass = acc.sum(dim=2, keepdim=True)
        sharp = acc.amax(dim=2, keepdim=True)
        copy = torch.bmm(acc, obs_pad.float()) / mass.clamp_min(1e-4)
        copy = torch.nan_to_num(copy, nan=0.0, posinf=1e4, neginf=-1e4)

        rms = (copy.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        feat = torch.cat([mass.clamp(0.0, 1.0), sharp.clamp(0.0, 1.0), rms], dim=-1)
        gi = torch.cat([h_out[:, 0::2, :].float(), feat], dim=-1)
        gate = torch.sigmoid(self.chase_gate(gi.to(self.chase_gate.weight.dtype))).float()

        read = self.chase_out(copy.to(self.chase_out.weight.dtype)).float()
        contrib = gate * read * valid_cmd.unsqueeze(-1).float()
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(dtype)
        return pred, h_out


def build(**params):
    return R24BackchainPointerResolution(**params)

--- r24_backward_multihop_pointer_chase (axis arch)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.arch.r22_prefix_content_xattention import R22PrefixContentXAttention

D = 768

NAME = "r24_backward_multihop_pointer_chase"
DESCRIPTION = (
    "The r22 prefix-content arch plus a tied-weight MULTI-HOP BACKWARD POINTER CHASE over the "
    "strictly-earlier command prefix. Every command's raw 768-d embedding passes through one "
    "shared two-layer GELU address trunk and two sets of unit-normalized heads, giving each step "
    "a SOURCE address and a DESTINATION address in n_addr parallel key_d channels. A command's "
    "own source address is the hop-0 query; each hop scores that query against the DESTINATION "
    "addresses of all strictly-earlier steps (cosine summed over channels, learned temperature), "
    "softmax-attends, reads the raw observation paired with the matched step as that hop's "
    "candidate content, and then replaces the query with the matched step's SOURCE address under "
    "a learned gate, so hop k+1 searches for whatever put content at the location hop k came "
    "from. The per-hop candidate contents are combined by a softmax over learned halting logits "
    "that see the hidden state, a projection of the candidate content, the attention sharpness, "
    "the content RMS, its cosine to a learned null-observation direction, and a hop embedding; "
    "the combination is injected into the command-position prediction through a zero-init (D,D) "
    "readout under a sigmoid gate, so the forward is bit-identical to r22 at init. Only strictly "
    "earlier commands and their paired observations are visible at any position. The address "
    "heads form identical-shape (key_d, addr_hidden) pairs and the readout is (D,D)."
)


class R24BackwardMultihopPointerChase(R22PrefixContentXAttention):
    def __init__(
        self,
        chase_hops=6,
        addr_hidden=256,
        n_addr=2,
        content_proj=32,
        hop_emb=16,
        chase_logit_scale=8.0,
        chase_step_bias=1.0,
        chase_gate_bias=-2.0,
        **params,
    ):
        super().__init__(**params)
        self.chase_hops = max(1, int(chase_hops))
        self.addr_hidden = max(32, int(addr_hidden))
        self.n_addr = max(1, int(n_addr))
        self.content_proj_d = max(4, int(content_proj))
        self.hop_emb_d = max(2, int(hop_emb))

        self.chase_ln = nn.LayerNorm(D)
        self.chase_in = nn.Linear(D, self.addr_hidden)
        self.chase_mid = nn.Linear(self.addr_hidden, self.addr_hidden)
        self.chase_src = nn.ModuleList(
            [nn.Linear(self.addr_hidden, self.key_d, bias=False) for _ in range(self.n_addr)]
        )
        self.chase_dst = nn.ModuleList(
            [nn.Linear(self.addr_hidden, self.key_d, bias=False) for _ in range(self.n_addr)]
        )

        self.chase_logit_scale = nn.Parameter(torch.tensor(float(chase_logit_scale)))
        self.chase_null = nn.Parameter(torch.randn(D) * (1.0 / math.sqrt(D)))
        self.chase_content = nn.Linear(D, self.content_proj_d)
        self.chase_hop_emb = nn.Embedding(self.chase_hops, self.hop_emb_d)

        self.chase_step_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_step_gate.bias, float(chase_step_bias))

        self.chase_halt = nn.Linear(self.d + self.content_proj_d + 3 + self.hop_emb_d, 1)
        nn.init.zeros_(self.chase_halt.bias)

        self.chase_gate = nn.Linear(self.d + 3, 1)
        nn.init.constant_(self.chase_gate.bias, float(chase_gate_bias))

        self.chase_out = nn.Linear(D, D)
        nn.init.zeros_(self.chase_out.weight)
        nn.init.zeros_(self.chase_out.bias)

    def _chase_features(self, attn, content):
        sharp = attn.amax(dim=2, keepdim=True)
        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        null_cos = (
            F.normalize(content, dim=-1) * F.normalize(self.chase_null, dim=0)
        ).sum(dim=-1, keepdim=True)
        return torch.cat([sharp, rms, null_cos], dim=-1)

    def _pointer_chase(self, cmd_raw, obs_mem, h_cmd, allowed, n_cmd, n_pair):
        A = self.n_addr
        B = cmd_raw.size(0)
        K = self.key_d
        device = cmd_raw.device

        a_h = F.gelu(self.chase_mid(F.gelu(self.chase_in(self.chase_ln(cmd_raw)))))
        src = torch.stack([self._unit(m(a_h)) for m in self.chase_src], dim=0)
        dst = torch.stack([self._unit(m(a_h)) for m in self.chase_dst], dim=0)

        src_mem = src[:, :, :n_pair, :].reshape(A * B, n_pair, K)
        dst_mem = dst[:, :, :n_pair, :].reshape(A * B, n_pair, K).transpose(1, 2)
        q = src.reshape(A * B, n_cmd, K)

        allowed_rep = allowed.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, n_pair)
        has_key = allowed.any(dim=2, keepdim=True)
        scale = self.chase_logit_scale.clamp(0.5, 64.0)

        contents = []
        halts = []
        sharps = []
        for k in range(self.chase_hops):
            scores = torch.bmm(q, dst_mem).view(A, B, n_cmd, n_pair).sum(dim=0) * scale
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~allowed, neg)
            attn = torch.softmax(scores, dim=2)
            attn = torch.where(has_key, attn, torch.zeros_like(attn))

            content = torch.bmm(attn, obs_mem)
            content = torch.nan_to_num(content, nan=0.0, posinf=1e4, neginf=-1e4)
            feats = self._chase_features(attn, content)

            hop_code = self.chase_hop_emb(
                torch.tensor(k, device=device, dtype=torch.long)
            ).view(1, 1, self.hop_emb_d).expand(B, n_cmd, self.hop_emb_d)
            halt_in = torch.cat([h_cmd, self.chase_content(content), feats, hop_code], dim=-1)
            halts.append(self.chase_halt(halt_in))
            contents.append(content)
            sharps.append(feats[..., 0:1])

            if k + 1 == self.chase_hops:
                break

            step_gate = torch.sigmoid(self.chase_step_gate(torch.cat([h_cmd, feats], dim=-1)))
            attn_rep = attn.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, n_pair)
            follow = torch.bmm(attn_rep, src_mem)
            gate_rep = step_gate.unsqueeze(0).expand(A, -1, -1, -1).reshape(A * B, n_cmd, 1)
            q = self._unit(gate_rep * follow + (1.0 - gate_rep) * q)

        halt_logits = torch.cat(halts, dim=-1)
        weights = torch.softmax(halt_logits, dim=-1)
        stacked = torch.stack(contents, dim=2)
        mixed = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        peak = torch.cat(sharps, dim=-1).amax(dim=-1, keepdim=True)
        return torch.nan_to_num(mixed, nan=0.0, posinf=1e4, neginf=-1e4), peak

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
        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=device)
        valid_cmd = valid[:, 0::2]
        valid_obs = valid[:, 1::2]

        wdt = self.chase_in.weight.dtype
        cmd_raw = tok_emb[:, 0::2, :].to(wdt)
        obs_mem = tok_emb[:, 1::2, :].to(wdt)
        h_cmd = h_out[:, 0::2, :].to(wdt)

        ci = torch.arange(n_cmd, device=device).unsqueeze(1)
        pj = torch.arange(n_pair, device=device).unsqueeze(0)
        allowed = (pj < ci).unsqueeze(0) & (valid_obs & valid_cmd[:, :n_pair]).unsqueeze(1)

        mixed, peak = self._pointer_chase(cmd_raw, obs_mem, h_cmd, allowed, n_cmd, n_pair)

        rms = (mixed.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        null_cos = (
            F.normalize(mixed, dim=-1) * F.normalize(self.chase_null, dim=0)
        ).sum(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.chase_gate(torch.cat([h_cmd, rms, peak, null_cos], dim=-1)))

        contrib = gate * self.chase_out(mixed)
        contrib = contrib * valid_cmd.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = pred.clone()
        pred[:, 0::2, :] = pred[:, 0::2, :] + contrib.to(pred.dtype)
        return pred, h_out


def build(**params):
    return R24BackwardMultihopPointerChase(**params)

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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0000 (r3-06-within-sequence-contrast, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
