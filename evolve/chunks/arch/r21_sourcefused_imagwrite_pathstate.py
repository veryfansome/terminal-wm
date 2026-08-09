"""R21 arch: SOURCE-FUSED EVIDENCE-GATED IMAGINATION WRITE — Candidate 8
(r21_evidencegated_imagwrite_pathstate) with the imagined write's PRE-CONTENT ESTIMATE
upgraded on source-present windows, plus this round's measured map of which state
completions do NOT work (three refuted sibling families, probed at equal protocol).

THE PARENT (kept verbatim): the r20 native imagination write — the record's only
FULL-SCALE-measured (b)-path gain (dedup Δb +0.0334 vs the plain champion, every genuine
family non-negative) — with C8's evidence gate g = evid/(evid+c) that pins the
history-masked arm to the plain champion exactly (empty memory ⇒ g == 0 ⇒ no write).

THE ONE CHANGE (measured): at a valid-cmd/masked-obs pair m, C8 executes the trained
transition on the tr-memory blend, f(s_pre_tr, c_m). That input is the R20-finding-6
OFF-distribution point: the head aux (`r18_transition_forwardmodel_consistency`)
supervises the operator as f(RAW same-path pre-observation, c_m) ≈ the future same-path
read — and the record's (a)-path measured raw-obs inputs far better than memory content
through this operator (endhist-pre 0.404 vs mem-pre 0.148). When the plan-time prefix
CONTAINS an observed same-path touch of c_m (the aux's own mining rule: latest p < m
with cmd-cosine > 0.60 — measured coverage ~20.6% of genuine windows; per family:
redir:echo> 49.5%, ln 44%, mv 18%, mkdir 17%, redir:prod> 4%), that aux-distribution
input exists at plan time. This arch FUSES it into the write's pre-content:

    v_imag = 0.5 * f(s_pre_tr, c_m) + 0.5 * transition_from_emb(z_obs_p*, z_cmd_m)
    (== f((s_pre_tr + src)/2, c_m) up to featurization — the operator is affine in s)

on matched rows only; unmatched rows remain exactly C8. Two independent pre-content
estimators — the associative memory blend and the episodic latest same-path observation
— averaged before the trained dynamics: estimator fusion of two priors, applied at the
exact operating point the aux trains.

MEASURED (mini champion-stack protocol, the round's shared one — d=128/L2, 700 steps,
CPU, train-image windows, node+postgres held out, n=2336/1273 dedup; paired
function-swap on identical trained weights, so deltas are exact; TRAIN-image probes
only — honest bounds, not a promotion claim): paired Δb vs C8, seeds 0/1/2 — full
slice +0.0020/+0.0006/+0.0006 (3/3 positive, mean +0.0011); dedup +0.0016/−0.0002/
+0.0012 (mean +0.0009, worst seed −0.0002); matched rows (dedup) +0.0077/−0.0010/
+0.0058 (mean +0.0042); per-family dedup means echo> +0.0053, prod> +0.0011, mv
+0.0005, mkdir 0.0, ln −0.0027 (worst family far above the −0.02 floor). Blend
response is ordered and replicates: dedup means 0.25 → +0.0004, 0.5 → +0.0009,
1.0 → −0.0005 (full-replace also seed-unstable: echo> +0.008 s0 / −0.0112 s1) — an
interior optimum, the signature of genuine two-estimator fusion, not retrieval
override. The aggregate effect at mini scale is small and honestly bounded; the
write's promotion case remains the parent's full-scale-anchored +0.0334, with this
fusion as a floors-safe refinement and src_blend=0.0 recovering C8 bit-exactly as
the designed full-scale ablation.

REFUTED SIBLINGS (same protocol, recorded so later rounds do not re-walk them):
(1) writing f(s_pre_tr, c_m) into the FILE/PATH delta memories (the champion's primary
    render pathway): −0.0185/−0.0112 dedup (2 seeds) — the FiLM/mix render is trained on
    observed-manifold memory values only (R20 finding 6 reproduced at a second site);
(2) value-independence of that failure (3 seeds): trans-of-source −0.0007 / raw-copy
    −0.0005 / pure-ERASE −0.0012 / thresh-0.50 −0.0007 mean dedup, 11/12 arm-seed
    cells ≤ 0 — the extra write event itself (with its global decay at m) perturbs
    the trained equilibrium;
(3) fusing z_{m-1} (last obs) into the write's pre on ALL rows: −0.0010 dedup, prod>
    −0.0097 — the unconditional last observation misleads produced-file redirects.

FITNESS PATH — bit-identical training to the champion, C8's own chain, unchanged:
zero new parameters (blend/threshold/gate are plain floats; state_dict, init-RNG stream,
gradients, Muon/spectral-cap optimizer routing identical); even-length streams have no
valid-cmd/masked-obs pair, the has_imag mask guard (champion's own per-pair masks; no
key_pad statistics, no PAD-value read) skips every added op, and the forward is the
champion's verbatim (this class overrides only `_transition_reads` + a raw-token stash
in `forward`); measured even-stream pred AND h max|Δ| = 0.0. The identical genome shape
(champion head + inert-write arch, bit-identical training) measured pod fitness 0.4245
vs champion 0.4247. IMAG_hist: no observed history ⇒ no match AND g == 0 ⇒ predictions
value-identical to the plain champion (measured max|Δ| = 0.0) — ΔIMAG_HA = Δb
one-for-one. PAD-value invariance exactly 0.0 (measured). (a)-path co-report unchanged
(`_memory_spre` calls `_transition_reads` without the stash ⇒ exact C8 behavior).

Refs: forward-model consistency at matched train/compose distributions (the R20 §6
mismatch, repaired at its cause); Dyna — imagined transitions update the same state
structures as real experience (Sutton 1991, SIGART 2(4)); Kalman fusion of independent
priors / innovation gain under evidence; MBPO trust-region model use (arXiv:1906.08253);
delta-rule fast-weight editing (arXiv:2102.11174; Gated DeltaNet arXiv:2412.06464);
constructive episodic simulation — recombine stored episode elements under a
transformation (Schacter, Addis & Buckner 2007, Nat Rev Neurosci 8:657); I-JEPA masked
latent completion (arXiv:2301.08243).
"""

import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r21_sourcefused_imagwrite_pathstate"
DESCRIPTION = (
    "Candidate 8's evidence-gated imagination write with the imagined value's pre-content "
    "upgraded on source-present windows: fuse the tr-memory blend with the latest observed "
    "same-path pre-observation (the head aux's own mining rule, cmd-cos>0.60) 50/50 and "
    "execute the trained operator on the fused estimate via transition_from_emb — the exact "
    "aux-training operating point. Unmatched rows and everything else are C8 verbatim; zero "
    "new parameters; even streams bit-identical; empty history algebraically zero. Sibling "
    "completions (file/path render writes, erase, z_prev fusion) measured negative and "
    "excluded by design."
)


class R21SourceFusedImagWritePathState(R18PathStateLatentTransition):
    """Overrides `_transition_reads` (C8's gated scan + matched-row source fusion) and
    wraps `forward` only to stash the raw command tokens for the aux-faithful
    `transition_from_emb` call (the champion forward body runs verbatim via super()).
    All knobs are plain Python floats — the state_dict and init-RNG stream are
    bit-identical to the champion arch."""

    def __init__(self, imag_gate_c=0.25, src_thresh=0.60, src_blend=0.5, **params):
        super().__init__(**params)
        # c > 0 preserves the exact-zero algebra (0/(0+c) == 0.0); measured evid on real
        # b-arm windows is >= ~1.5e3, so the observed-history gate sits at ~1 (C8).
        self.imag_gate_c = max(1e-6, float(imag_gate_c))
        # the head aux's path_thresh — deliberately NOT a tuned surface.
        self.src_thresh = float(src_thresh)
        # fusion weight of the source estimate; 0.0 recovers C8 exactly. Probed peak 0.5.
        self.src_blend = min(1.0, max(0.0, float(src_blend)))
        self._raw_cmd = None

    def forward(self, tok_emb, types, key_pad):
        self._raw_cmd = tok_emb[:, 0::2, :] if tok_emb.dim() == 3 else None
        try:
            return super().forward(tok_emb, types, key_pad)
        finally:
            self._raw_cmd = None

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        evid = x_cmd.new_zeros(B, 1)
        c = float(self.imag_gate_c)

        # ---- masked-pair presence (champion's own per-pair masks; False on every
        # even-length training/fitness stream -> the scan below is C8's ops verbatim).
        if n_pair:
            imag_pair = valid_cmd[:, :n_pair] & ~valid_obs
            has_imag = bool(imag_pair.any().item())
        else:
            has_imag = False

        # ---- source-triple match (the head aux's mining rule, at plan time): latest
        # OBSERVED prefix pair p < i with cmd-cosine > src_thresh to command i. Uses raw
        # command token VALUES only (stashed by forward); the PAD obs value is never read.
        use_src = (
            has_imag
            and self.src_blend > 0.0
            and self._raw_cmd is not None
            and self._raw_cmd.shape[0] == B
            and self._raw_cmd.shape[1] == n_cmd
        )
        if use_src:
            raw = torch.nan_to_num(self._raw_cmd, nan=0.0, posinf=1e4, neginf=-1e4)
            cu = self._unit(raw)
            sim = torch.bmm(cu, cu.transpose(1, 2))                       # [B,n_cmd,n_cmd]
            pos = torch.arange(n_cmd, device=x_cmd.device)
            observed = torch.zeros(B, n_cmd, dtype=torch.bool, device=x_cmd.device)
            observed[:, :n_pair] = valid_cmd[:, :n_pair] & valid_obs
            causal_pp = pos.view(1, -1, 1) > pos.view(1, 1, -1)           # p < i
            cand = observed.unsqueeze(1) & causal_pp & (sim > float(self.src_thresh))
            posf = pos.view(1, 1, -1).expand(B, n_cmd, n_cmd)
            pstar = torch.where(cand, posf, torch.full_like(posf, -1)).amax(dim=2)
            matched_all = pstar >= 0                                      # [B,n_cmd]
            pc = pstar.clamp(0, max(0, n_pair - 1))
            obs_padded = self._pad_steps(obs_tok, n_cmd)
            src_all = torch.gather(obs_padded, 1, pc.unsqueeze(-1).expand(B, n_cmd, D))

        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)
            reads.append(s_pre)
            delta = self._transition(s_pre, x_cmd[:, i, :])
            if i < n_pair:
                obs_i = obs_tok[:, i, :].to(dtype)
                wi = w[:, i].unsqueeze(-1)
                active = (valid_obs[:, i] & valid_cmd[:, i]).to(dtype).unsqueeze(-1)
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta
            observed_corr = (v_i - s_pre) * active
            g = evid / (evid + c)
            delta_imag = delta
            if use_src and i < n_pair and bool((imag.squeeze(-1) > 0).any().item()):
                mrow = matched_all[:, i] & (imag.squeeze(-1) > 0)
                if bool(mrow.any().item()):
                    # the operator at its aux-training operating point (raw pre-obs,
                    # raw command; transition_from_emb re-featurizes exactly as the aux).
                    f_src = self.transition_from_emb(
                        src_all[mrow, i].to(dtype), self._raw_cmd[mrow, i].to(dtype)
                    )
                    f_src = torch.nan_to_num(f_src, nan=0.0, posinf=1e4, neginf=-1e4)
                    delta_imag = delta.clone()
                    b = float(self.src_blend)
                    delta_imag[mrow] = b * f_src + (1.0 - b) * delta[mrow]
            imagined_corr = g * (w[:, i].unsqueeze(-1) * (delta_imag - s_pre)) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            evid = evid + observed_corr.pow(2).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R21SourceFusedImagWritePathState(**params)
