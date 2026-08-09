"""R22 arch: NATIVE FULL-RANK PREFIX-CONTENT CROSS-ATTENTION CHANNEL — the frozen-probe
computation (the record's strongest positive capability measurement: cross-attention over
the plan-time prefix queried by the endpoint commands, +0.16-0.30 over the trained
command-only ceiling on every genuine family, differential +0.181 dedup, R20 findings 2/5)
transplanted INTO the champion trunk as a zero-init native channel, trained by the
untouched main loss inside the single pass.

THE MEASURED GAP THIS FILLS (R22 inventor-7 diagnostics, canonical full-scale champion
checkpoints s0/s1, protocol validated against the round's CA* baseline 0.5611/0.5614):
the champion's realized masked-endpoint outcome TRACKS cross-position prefix-content
availability (train-window CA* on mv 0.850 where answer-similar content exists at
non-same-path prefix positions vs 0.300 where it does not; mkdir b_top1 0.993 vs 0.718),
and on the PRIOR-ONLY windows that dominate the lowest-CA family (76% of mv) the OWN
prefix's raw observation set still carries a +0.228 mean-cosine content differential
toward the answer over a presence-matched donor prefix (prod> +0.197, echo> +0.247,
ln +0.230, mkdir +0.018). Yet the champion has NO full-rank positional route to that
evidence: every 768-d content path into the prediction is a SLOT-STORE read keyed by a
single command projection (file/path/transition delta memories, prev-obs copy), while
positional attention exists only inside the d=176 transformer whose content must exit
through a rank-176 head; and the only dedicated system-evidence channel is a diagonal
style FiLM from a 64-d summary that the round's decomposition measured CA-ADVERSE
(no_syscond raises train-window CA* on both seeds: +0.018/+0.034).

MECHANISM (one channel; the champion forward is executed VERBATIM first, then the
channel adds a gated residual to the command-position predictions):
    keys    k_j = xk(h_out[obs position j])          # position-contextual, path-aware (causal h)
    query   q_i = xq(h_out[cmd position i])          # at the masked-endpoint read: has attended c_m,c_r
    attn    a_ij = softmax_j<i( q_i . k_j / sqrt(dk) )   # strictly-earlier VALID pairs only
    value   v_j = raw z_obs_j                        # the probe-exact 768-d frozen value space
    out_i   pred_cmd_i += sigmoid(x_gate([h_i, rms(o_i)])) * x_out(o_i),  o_i = sum_j a_ij v_j
x_out is a ZERO-INIT (768,768) square readout -> the arch is the champion function
bit-for-bit at init, and the co-designed optimizer's routing captures the new channel by
the champion's own signatures: xq/xk are (64,176) addressing projections (join the Muon
orthogonalized-momentum group), x_out is a (D,D) square content readout (joins the
spectral-norm-capped group) -- the exact rails the champion stack already trains under.
Sharp attention retrieves specific cross-position content (the mv-source / moved-file /
link-target evidence the read command's slot key cannot reach); diffuse attention
aggregates a query-conditioned SYSTEM-CONTENT prototype (the +0.23 prior-only
differential) -- one computation serving both regimes, injected AFTER the sysfilm
styling so the measured CA-adverse diagonal style transform never touches it.

WHY THE WRONG ARM CANNOT SHARE THE GAIN: the channel's payload is the prefix
observations themselves, which differ mechanically between the real-prefix and
donor-prefix arms; any learned use of it is content-attributable, while decode
components (query-side command information) fire identically in both arms and cancel
in IMAG_CA. Training only ever sees coherent own-trajectory prefixes, so no donor/
incoherence feature exists to learn.

Causal / leak-free: attention is strictly lower-triangular over (cmd i, pair j) with
j < i and masked by pair validity (the b-layout PAD slot is excluded by key_pad, so
PAD-value invariance is structural); obs_t reaches only commands > t. Eval-unconditional:
the identical computation runs on every stream, train and eval -- no masked-pair branch,
no key_pad statistics beyond the standard validity mask the parent forward itself uses.
NaN-safe (masked-empty rows emit exactly 0; nan_to_num on the contribution).

Distinct from: the r20/r21 imagination-write family (eval-conditional masked-pair
writes; this has no conditional branch and trains from step ~0); the R21 detached
observers (private losses + gated overrides, measured subsumed; this has no private
loss, no detachment, no override -- the main loss owns it); r13 syscond (64-d diagonal
STYLE conditioning; this is full-rank CONTENT aggregation injected outside the style
transform); the champion's own memories (single-command-keyed slot stores; this is
positional attention with values in the raw obs space).
"""

import math

import torch
import torch.nn as nn

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

D = 768

NAME = "r22_prefix_content_xattention"
DESCRIPTION = (
    "Champion r18 path-state arch + ONE new native channel: a single-head full-rank "
    "cross-attention over the strictly-earlier prefix OBSERVATIONS (keys/queries from the "
    "causal hidden states, values = raw 768-d obs embeddings — the frozen-probe computation "
    "in-trunk), injected into the command-position predictions through a zero-init (D,D) "
    "readout and a sigmoid gate, AFTER the sysfilm styling. Champion bit-for-bit at init; "
    "Muon captures the new (64,176) addressing pair, the spectral cap the new (D,D) readout."
)


class R22PrefixContentXAttention(R18PathStateLatentTransition):
    def __init__(self, xattn_dim=64, xattn_gate_bias=-2.0, **params):
        super().__init__(**params)
        self.xattn_dim = int(xattn_dim)
        # (64, d=176) addressing projections — same signature as the delta-memory keys, so the
        # co-designed optimizer routes them into the Muon orthogonalized-momentum group.
        self.xq = nn.Linear(self.d, self.xattn_dim, bias=False)
        self.xk = nn.Linear(self.d, self.xattn_dim, bias=False)
        # (D, D) square content readout, ZERO-INIT -> exact champion function at init; the
        # co-designed optimizer's spectral cap targets this signature (norm-calibration rail).
        self.x_out = nn.Linear(D, D)
        nn.init.zeros_(self.x_out.weight)
        nn.init.zeros_(self.x_out.bias)
        self.x_gate = nn.Linear(self.d + 1, 1)
        nn.init.constant_(self.x_gate.bias, float(xattn_gate_bias))

    def forward(self, tok_emb, types, key_pad):
        pred, h_out = super().forward(tok_emb, types, key_pad)  # champion forward VERBATIM
        B, L, _ = tok_emb.shape
        if L < 3:
            return pred, h_out
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_pair == 0:
            return pred, h_out

        valid = ~key_pad.bool() if key_pad is not None else torch.ones(
            B, L, dtype=torch.bool, device=tok_emb.device)
        valid_obs = valid[:, 1::2]                                   # [B, n_pair]
        h_cmd = h_out[:, 0::2, :]                                    # [B, n_cmd, d] causal
        h_obs = h_out[:, 1::2, :]                                    # [B, n_pair, d] causal
        obs_val = tok_emb[:, 1::2, :]                                # [B, n_pair, D] raw values

        q = self.xq(h_cmd)                                           # [B, n_cmd, dk]
        k = self.xk(h_obs)                                           # [B, n_pair, dk]
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.xattn_dim)

        # strictly-earlier pairs only: obs of pair j (position 2j+1) visible to command i
        # (position 2i) iff j < i; plus pair validity (excludes the b-layout PAD slot).
        ci = torch.arange(n_cmd, device=tok_emb.device).unsqueeze(1)   # [n_cmd,1]
        pj = torch.arange(n_pair, device=tok_emb.device).unsqueeze(0)  # [1,n_pair]
        allowed = (pj < ci).unsqueeze(0) & valid_obs.unsqueeze(1)      # [B, n_cmd, n_pair]

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)                     # [B, n_cmd, 1]
        attn = torch.softmax(scores, dim=2)
        attn = torch.where(has_key, attn, torch.zeros_like(attn))      # empty rows -> exact 0
        o = torch.bmm(attn.to(obs_val.dtype), obs_val)                 # [B, n_cmd, D]
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
