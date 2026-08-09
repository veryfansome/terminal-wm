"""R20 arch: CONTENT-CONDITIONED TRANSITION OPERATOR + the imagination write — attack the one
measured headroom no other R20 mechanism touches: the r18 transition's FUNCTIONAL FORM.

FINDING 7 (round brief): a fresh operator trained on the frozen r18 arch's memory content,
(s_pre_m, c_m) -> z_r, scores 0.510 with the r18 arch's exact command-only AFFINE form — i.e.
AT the lexical floor (0.511) — while an MLP on the same inputs scores 0.569 (+0.057). The
r18 operator cannot express content-dependent edits: gamma, beta are functions of the
COMMAND ONLY, a restriction chosen for a closed-form linear scan that was then NOT used (the
<=16-step sequential loop ships — see the r18 `_transition_reads` docstring). So the
restriction buys nothing at runtime and measurably costs +0.057 of operator-level composition
at full r18 scale. Every other R20 mechanism freezes this operator and works AROUND it
(endpoint correctors / calibrators / renderers / de-novo regressors); this mutation upgrades
the operator ITSELF, inside the r18 stack's existing training pressure.

MECHANISM — two changes on the r18 path-state arch:

1. CONTENT-CONDITIONED TRANSITION RESIDUAL (the new capacity). `_transition` becomes
       delta = s_pre*(1+gamma(cmd)) + beta(cmd)                    (r18 affine, unchanged)
             + cap * tanh( W2 gelu( W1 [rms_norm(s_pre); cmd_feat] ) / cap )   (NEW, W2 ZERO-INIT)
   Zero-init -> the arch computes exactly the r18 function at step 0 (the r18 stack's own
   fade-in discipline: tr_read / film_out / sysfilm_out are all zero-init; cf. ReZero,
   arXiv:2003.04887). The rms-normalization of s_pre makes the residual robust to the measured
   memory-vs-raw norm shift (eval s_pre_m mean norm 11.5 vs mined-triple s_pre_k 18.4 vs raw
   obs ~23-27 — measured on the mini, see proposal), which the affine is not. The residual is
   soft-clipped per channel (|res| <= cap = 2.0 in ~unit-variance standardized obs space) so the
   per-path recurrence s -> f(s) stays bounded even though f now depends on s. MODIFY/APPEND
   edits ("remove one name from this listing", "append to this file") are content-dependent
   transformations an affine-in-content-with-command-only-coefficients operator cannot
   represent; this is exactly the mutated-cell content a symbolic tracker cannot compute (where
   the v3 margin lives), so the same capacity that serves imagination is in play FOR fitness
   (mut_gate 0.946 — the arch heavily uses this write in-distribution), not merely protected
   from it. Trained by the r18 stack's own losses: the main loss through the memory reads, and
   the (co-designed) head aux supervising `transition_from_emb` on mined mutation triples.

2. THE PARAMETER-FREE IMAGINATION WRITE (taken VERBATIM from `r20_imagwrite_pathstate_
   worldmodel`, attributed): a pair with a valid command but a masked (key_pad) observation
   writes w_i*(f(s_pre, c_m) - s_pre) into its path slot, so the net natively forwards the
   declared obs-missing endpoint layout [prefix, c_m, PAD-obs, c_r] and the LATER read
   composes post-mutation content through the r18 arch's untouched machinery — now with the
   richer f. Identically dead on even-length streams (pairs pad together in every harness
   collate), so fitness training/eval never executes it.

WHY THIS LEVER AND NOT ANOTHER (all measured, image-disjoint TRAIN-window minis, 2 seeds,
protocol byte-matching the prior R20 minis — floor 0.5404 full / 0.5032 dedup, n=2315/1400):
the write-family COMPOSITION is at its own ceiling — an ORACLE that substitutes the TRUE
mutation observation into the masked slot scores only ~+0.003 above the write, a fully
in-distribution self-rollout (dream the mutation's obs, re-forward fully-observed) TIES the
write (residual-error cosine 0.98: same predictor), endpoint ensembling is dead, a re-read
refinement pass is negative, and endpoint scale calibration is worth only ~+0.005/+0.002. So
further (b)-path gains cannot come from better composition of a frozen operator; the operator
itself is the remaining lever. At the mini's 700-step budget the recruited residual
(||res||/||base|| = 0.42 on eval inputs) is outcome-neutral (write-endpoint and main-task tie
the r18 arch within seed noise; raw-pre operator composition improves slightly, +0.01..+0.03);
the repo's own measurement doctrine (evolve/CLAUDE.md: proxy under-trains slow-converging
memory mechanisms, documented rank-inversions) is why the full-budget 3-seed run — where
finding 7's +0.057 was measured — is the measurement this file's mechanism needs.

FITNESS PATH: NOT bit-identical (deliberately — the capacity aims at the mutated cells);
protected by the r18 stack's proven fade-in discipline (exact r18 function at init,
verified bit-identical pred AND h on even streams), a bounded residual, unchanged interfaces,
and the paired 2x2 mini showing main-task deltas <= 0.0013 across all quadrants and both
seeds. New params (~330K at full r18 scale) are constructed AFTER the entire r18 __init__,
so every r18 parameter draws the IDENTICAL init-RNG stream; the residual's two Linears
route to AdamW under the co-designed optim (shapes (192, D+d) and (D, 192) match neither the
Muon (key_d, *) signature nor the (D,D) spectral-cap signature — verified against the real
make()). The imag_direct (a)-path instrument works unchanged (`transition_from_emb` /
`_transition_reads` present; s_pre at m is read strictly before the write on the instrument's
layout) and now measures the trained operator.

LEAK-FREE: the residual reads s_pre (writes strictly < i) and cmd_feat (the current command) —
no new information flow; the masked pair's obs VALUE is never read (PAD-value perturbation
Delta == 0.0 exactly at the prediction position, verified; prefix-obs and c_m perturbations DO
move the endpoint; even-stream causality bit-identical to the r18 arch at init).

Refs: DeltaProduct — richer per-token memory transforms beyond one delta step, arXiv:2502.10297;
Gated DeltaNet erase-capable writes, arXiv:2412.06464; RSSM/PlaNet/Dreamer latent transitions
f(s, a) are MLPs of [state; action], never action-only affines, arXiv:1811.04551 /
arXiv:1912.01603; ReZero zero-init residual fade-in, arXiv:2003.04887; observation-dropout
world models (missing obs stood in by the model), arXiv:1910.13038.
"""

import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_contentcond_transition_imagwrite"
DESCRIPTION = (
    "The r18 arch with the transition operator's functional form upgraded from command-only "
    "affine to CONTENT-CONDITIONED: delta = affine(s_pre, cmd) + cap*tanh(MLP([rms(s_pre); "
    "cmd_feat])/cap), MLP zero-init (exact r18 function at init), trained by the r18 stack's "
    "own main loss + forward-model head aux — targeting the measured +0.057 operator-form "
    "headroom (brief finding 7) that every endpoint corrector works around; plus the r20 "
    "parameter-free imagination write (reused, attributed) so the net natively forwards the "
    "obs-missing endpoint layout (measurement path b). Bounded residual, unchanged interfaces, "
    "Muon/spectral-cap routing verified safe."
)


class R20ContentCondTransitionImagWrite(R18PathStateLatentTransition):
    """r18 path-state subclass. Overrides `_transition` (content-conditioned residual,
    zero-init) and `_transition_reads` (the reused imagination write). Everything else — trunk,
    memories, gates, FiLM, `transition_from_emb` (which now routes through the richer
    `_transition`) — is inherited verbatim. New modules are constructed AFTER the full r18
    __init__ so the r18 parameters draw the identical init-RNG stream."""

    def __init__(self, *args, tr2_hidden=192, tr2_cap=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.tr2_cap = float(tr2_cap)
        th2 = max(32, int(tr2_hidden))
        # Shapes chosen to dodge the co-designed optim's routing signatures:
        # (th2, D + d) and (D, th2) are neither (key_d, *) [Muon] nor (D, D) [spectral cap].
        self.tr2_in = torch.nn.Linear(D + self.d, th2)
        self.tr2_out = torch.nn.Linear(th2, D)
        torch.nn.init.zeros_(self.tr2_out.weight)
        torch.nn.init.zeros_(self.tr2_out.bias)

    def _transition(self, s_pre, cmd_feat):
        """Content-conditioned transition: r18 affine + bounded zero-init residual on
        [rms-normalized current content; command feature]. Exact r18 function at init.
        NOTE: f now depends on s_pre nonlinearly, so the r18 docstring's diagonal-linear-
        scan equivalence no longer applies even in principle (the shipped sequential loop never
        used it)."""
        base = super()._transition(s_pre, cmd_feat)
        s_n = s_pre * torch.rsqrt(s_pre.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        h = torch.nn.functional.gelu(self.tr2_in(torch.cat([s_n, cmd_feat], dim=-1)))
        res = self.tr2_cap * torch.tanh(self.tr2_out(h) / self.tr2_cap)
        return base + res

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        """r18 scan + the r20 imagination write (taken verbatim from
        r20_imagwrite_pathstate_worldmodel): a valid-cmd/masked-obs pair writes
        w_i*(f(s_pre, cmd) - s_pre); identically dead on even-length fully-observed streams
        (the fitness path), where this reduces bit-for-bit to the r18 scan at init."""
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))                          # [B,n_cmd,key_d]
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)       # [B,n_cmd]
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        reads = []
        for i in range(n_cmd):
            pi = p[:, i, :]
            s_pre = torch.bmm(pi.unsqueeze(1), mem).squeeze(1)       # writes < i (causal)
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
            v_i = (1.0 - wi) * obs_i + wi * delta                    # r18 write value
            corr = (v_i - s_pre) * active \
                + (w[:, i].unsqueeze(-1) * (delta - s_pre)) * imag   # imagination write
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R20ContentCondTransitionImagWrite(**params)
