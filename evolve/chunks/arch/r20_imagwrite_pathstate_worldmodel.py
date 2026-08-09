"""R20 arch: IMAGINATION-WRITE path-state world model — the r18 path-state arch plus ONE
parameter-free branch that makes the net natively forward an OBS-MISSING suffix, so the
imagination the r18 stack ALREADY TRAINS becomes reachable at composition time. Nothing else
changes: zero new parameters, zero new training pressure, and the branch is identically dead
on every fully-observed stream — the trained model is bit-for-bit the r18 path-state arch.

THE GAP (measured, R20 brief): the r18 path-state arch cannot imagine a mutation->read outcome
without observation feedback (hardened companion: genuine dedup endhist-pre margin -0.066 /
mem-pre -0.342), and its native masked forward is nearly blind to the mutation (+0.0085 over
the no-mutation counterfactual, P4 probe) for a MECHANICAL reason visible in the code:
`_transition_reads` gates every memory write with `active = valid_obs & valid_cmd`, so a pair
whose observation is masked — the declared path-(b) endpoint layout
[prefix, cmd_m, PAD-obs, cmd_r] — writes NOTHING. The mutation never enters the path state and
the read at cmd_r retrieves the STALE pre-mutation content (the counterfactual twin the eval
punishes). Meanwhile the r18 head aux (`r18_transition_forwardmodel_consistency`,
aux_weight 1.0) already trains the transition operator f(s_pre, cmd) as an obs-calibrated
forward model on mined (pre, mutating-cmd, future-read) triples, and the trained read path
(spectral-capped tr_read + gates + syscond FiLM) already knows how to render memory content
into predictions. The ONLY missing piece of the imagination loop is the write when obs is
absent — this file adds exactly that piece:

    corr_i = (v_i - s_pre_i) * [valid_obs & valid_cmd]                    (r18, unchanged)
           + w_i * (delta_i - s_pre_i) * [valid_cmd & ~valid_obs]         (NEW imagination write)

with delta_i = f(s_pre_i, cmd_i) the r18 arch's own command-conditioned transition and
w_i = sigmoid(tr_mut_gate) its own mutation detector (mut_gate_mean 0.946 on real mutations).
The mutation is applied in latent state exactly as the fully-observed write would apply it,
minus the unavailable obs blend, and the LATER read composes post-mutation content through the
r18 arch's untouched machinery — trunk attention over prefix + c_m + c_r, path/file memories
over the prefix, transition memory carrying the imagined edit, 3-way read mix, syscond FiLM.

EVIDENCE (image-disjoint TRAIN-window probe, mini 2-layer/d128 stack, 700 steps, n=2315
genuine heldout windows, hardened-style per-window-oracle floor 0.5404 full / 0.5032 dedup):
the untouched-training net evaluated with this branch scores ABOVE floor on every mutation
family with NO imagination-specific training at all — margins +0.081 full / +0.110 dedup
(mv +0.064/+0.117, redir:prod> +0.142/+0.117, redir:echo> +0.049/+0.045, mkdir +0.061/+0.145,
ln +0.087/+0.105) — while an in-pass masked-endpoint InfoNCE aux (also tested, both with and
without this branch) DEGRADED both the main task and the endpoint (-0.065/-0.002 full), so the
aux was dropped and this mutation ships PURE: the imagination is emergent from the r18 stack's
existing losses, not from new training. Same-weights write-ON vs write-OFF ablation isolates
the branch's contribution (+0.032/+0.026 full margin in the main-only training, +0.026/+0.017
with the r18 head aux; positive on all 5 families in both trainings).

WHY THE FITNESS PATH IS PROVABLY UNCHANGED: training/eval streams (baseline_interleave) are
even-length with pairs either both-valid or both-padded, so [valid_cmd & ~valid_obs] is
identically False — verified bit-identical forward (pred AND h) to the verbatim r18 class on
even streams at the r18 reference config, including after weight perturbation. ZERO new
parameters -> state_dict, init-RNG stream, gradients, optimizer routing (Muon keys + the (D,D)
spectral-cap signature) all bit-identical: score_genome retrains literally the same model
(verified: a 30-step training with the r18 head + r18 optim produced a bit-identical state_dict
vs the r18 arch). The imag_direct (a)-path instrument is also bit-identical (s_pre at m
is read STRICTLY BEFORE the write at m; verified on the instrument's own layout), so
operator-composition companion numbers stay byte-comparable with the r18 arch's own numbers.

LEAK-FREE: the masked pair's obs VALUE is never read (the obs term is multiplied by the
r18 arch's own active-gate, and key_pad excludes the slot from attention) — PAD-value
perturbation Delta == 0.0 exactly at the prediction position; prefix-obs and mutation-command
perturbations DO move the endpoint (the mechanism reads history and c_m). Future-obs leakage
on normal streams is 0.0 (inherited; re-verified).

Refs: latent overshooting (PlaNet, arXiv:1811.04551); masked latent prediction from context
(I-JEPA, arXiv:2301.08243; Masked World Models, arXiv:2206.14244); fast-weight/delta-rule
memory edits (arXiv:2102.11174); gated/erase-capable delta writes (arXiv:2412.06464).
"""

import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r20_imagwrite_pathstate_worldmodel"
DESCRIPTION = (
    "Champion r18 path-state latent-transition arch + a PARAMETER-FREE imagination write: a "
    "pair with a valid command but a masked (key_pad) observation writes w_i*(f(s_pre,cmd)-"
    "s_pre) into its path slot, so the net natively forwards obs-missing mutation suffixes "
    "(measurement path b) and composes the read through the champion's own trained machinery. "
    "Identically dead on even-length fully-observed streams: fitness training/eval, "
    "state_dict, init RNG, gradients and optimizer routing are bit-identical to the champion "
    "— the imagination is an emergent eval-time capability of the already-trained operator, "
    "not a new training pressure."
)


class R20ImagWritePathState(R18PathStateLatentTransition):
    """Subclass overriding ONLY `_transition_reads` (the imagination-write branch); every other
    r18 method — trunk, memories, gates, `_transition`, `transition_from_emb`, forward —
    is inherited verbatim."""

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
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
                # NEW: valid command whose observation is MASKED -> imagination write.
                # Identically 0 on fully-observed streams (pairs pad together), so the
                # fitness path never executes it. The obs VALUE is never read here.
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta                    # r18 write value
            corr = (v_i - s_pre) * active \
                + (w[:, i].unsqueeze(-1) * (delta - s_pre)) * imag   # NEW imagination write
            mem = decay * mem + torch.bmm(pi.unsqueeze(2), corr.unsqueeze(1))
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R20ImagWritePathState(**params)
