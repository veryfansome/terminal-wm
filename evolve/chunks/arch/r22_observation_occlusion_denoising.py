"""R22 arch: OBSERVATION-OCCLUSION DENOISING on the champion world model — the champion
`r18_pathstate_latent_transition_worldmodel` with ONE training-time change and ZERO
eval-time change: during the single sanctioned training pass, each (cmd, obs) pair's
OBSERVATION token is independently occluded with a small ramped probability, using the
frozen instrument's EXACT masked-slot semantics (token zeroed + internally key-padded,
type kept = obs). Eval-mode forward is the champion bit-for-bit (code-identity: the
subclass only edits its inputs when `self.training` and the ramped p > 0).

THE MEASURED GAP THIS ATTACKS (R22 record + this inventor's probes): the champion's
content-attributable imagination (CA 0.5471 anchor) is earned by its trunk ZERO-SHOT on
an input regime it never trains on — in the ordinary interleave stream every observation
between mutation and read is present by the time the read is supervised, while the
instrument forwards an obs-missing masked-endpoint layout. This inventor measured, on the
canonical full-scale champion checkpoints (s0/s1, TRAIN-image windows, instrument-exact
layouts + wrong-history donors), that the trained equilibrium is at a LOCAL OPTIMUM with
respect to every eval-time content-channel edit tried: (1) shielding the content memories
from intervention-step writes moves the b-arm by ~0 (db +0.0002/-0.0008; only the wrong
arm moves); (2) sketch-cued episodic snapping toward the window's own prefix observations
never raises the b-arm (db -0.0006..-0.0025 across gates/temperatures, both seeds); (3)
even an ORACLE-cued snap (cue = z_r) cannot raise it (db -0.004 at alpha=0.5, -0.437 at
alpha=1.0 — the local-episodic channel is fully banked; composition, not retrieval, is
what remains, consistent with the R20 shortcut audit). Conclusion: post-hoc edits of the
trained function are exhausted — the remaining lever is the TRAINING-TIME INPUT
DISTRIBUTION of the trunk itself.

MECHANISM (train-only, role-agnostic, loss-free): with probability p (smoothstep-ramped
0 -> occ_p between occ_ramp_start and occ_ramp_end training forwards), an observation
token is removed from the model's INPUT exactly as the measurement removes it: the token
zeroed AND key-padded, so attention masks it, the file/path/transition memories skip the
write (active_pair False), prev-obs and the system summary exclude it — the identical
downstream consequences the instrument's PAD slot produces (verified bit-identical:
forwarding the occluded-input construction equals forwarding the instrument's own
b-layout, max|delta| = 0). Supervision is untouched: the occluded pair's own command is
still supervised (its target comes from the batch target tensor, and a command's own
observation is never causally visible to its own prediction), so occlusion only degrades
the CONTEXT of LATER rows. Every role therefore trains under occasional missing evidence:
a read after an occluded mutation IS the imagination condition (expected ~15 genuine
mutation->read windows per bs-64 batch at p=0.12, ~59k over the 4000-step pass), a
revisit after an occluded first touch is cued recall from state rather than copy, and the
~88% unoccluded remainder anchors the observed-regime function the fitness metric scores.
The main loss, objective, batcher, head aux, stream, and eval are all untouched — no
mined pairs, no second forward, no constructed layouts, no auxiliary task, no new
parameters (one non-trainable step-counter buffer).

WHY THIS CAN MOVE IMAG_CA WHERE R20/R21 MECHANISMS DID NOT: the quantity is
CA = b - wrong. Occlusion training contains only coherent own-trajectory prefixes, so no
donor/incoherence detector can be learned (the wrong arm's inputs stay in-distribution);
command-decode components cancel between arms by construction. The only thing this
mechanism can learn is to answer occluded reads from surviving PREFIX CONTENT — i.e., a
b-arm content gain, which does not transfer to content-mismatched donor prefixes. Honest
failure modes: the trunk routes occluded reads through command priors (CA stays ~0 and
nothing promotes), or the 12% context degradation costs fitness beyond the eps=0.002
band. Both are decided only by the round's full-scale measurement.

Ramp rationale: warmup lets the champion equilibrium form first (house pattern: head aux
ramp 400, batcher ramp 30%); full occlusion strength holds for the last ~3000 steps.
p=0.12 sizes the imagination-condition incidence (~15 rows/batch) without dominating the
observed regime; occ_p is a genome param.

Causal/leak-safe: occlusion happens only in training mode; the harness's leakage guard
and all scoring run under net.eval() where this forward is the champion's byte-identical
code path. The edit touches only observation positions (never commands), only the
model's own INPUT VIEW (never targets), and only the current batch. RNG: global
generator, seeded per run by the harness (the r16 conditioning-dropout precedent);
during the p=0 warmup no draws are made, so the warmup RNG stream is bit-identical to
the champion's.

Optimizer/head co-design intact: zero new trainable parameters — Muon still routes the
(key_d, d) addressing pair, the spectral cap still routes the unique (D, D) tr_read;
`transition_from_emb` / `_transition_reads` inherited, so the r18 head aux and the
(a)-path operator co-report attach unchanged.
"""
import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    R18PathStateLatentTransition,
)

NAME = "r22_observation_occlusion_denoising"
DESCRIPTION = (
    "Champion r18 path-state world model trained under ramped stochastic observation "
    "occlusion: in training mode each obs token is independently removed (zeroed + "
    "key-padded — the frozen instrument's exact masked-slot semantics, verified "
    "bit-identical) with probability ramping to occ_p, so occluded-evidence prediction — "
    "including the mutation->read imagination condition — becomes an in-distribution "
    "training regime for the SAME trunk the instrument measures. Eval forward is the "
    "champion bit-for-bit; zero new trainable parameters; loss/head/batcher untouched."
)


class R22ObservationOcclusionDenoising(R18PathStateLatentTransition):
    def __init__(self, occ_p=0.12, occ_ramp_start=300, occ_ramp_end=1000, **params):
        super().__init__(**params)
        self.occ_p = max(0.0, min(0.9, float(occ_p)))
        self.occ_ramp_start = max(0, int(occ_ramp_start))
        self.occ_ramp_end = max(self.occ_ramp_start + 1, int(occ_ramp_end))
        # non-trainable training-forward counter (persisted in state_dict for ckpt fidelity)
        self.register_buffer("occ_step", torch.zeros((), dtype=torch.long))

    def _occ_prob(self):
        s = int(self.occ_step)
        if s <= self.occ_ramp_start:
            return 0.0
        if s >= self.occ_ramp_end:
            return self.occ_p
        x = (s - self.occ_ramp_start) / float(self.occ_ramp_end - self.occ_ramp_start)
        return self.occ_p * (x * x * (3.0 - 2.0 * x))   # smoothstep

    def forward(self, tok_emb, types, key_pad):
        if self.training:
            self.occ_step += 1
            p = self._occ_prob()
            B, L = tok_emb.shape[0], tok_emb.shape[1]
            if p > 0.0 and L >= 2 and B > 0:
                if key_pad is None:
                    key_pad = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                key_pad = key_pad.bool()
                n_pair = L // 2
                # i.i.d. per-pair Bernoulli(p) over OBSERVATION slots only (odd positions).
                drop = torch.rand(B, n_pair, device=tok_emb.device) < p
                drop_full = torch.zeros(B, L, dtype=torch.bool, device=tok_emb.device)
                drop_full[:, 1:2 * n_pair:2] = drop
                # the frozen instrument's masked-slot semantics: zero token + key-padded,
                # type untouched (stays obs). Non-inplace: inputs are never mutated.
                tok_emb = tok_emb.masked_fill(drop_full.unsqueeze(-1), 0.0)
                key_pad = key_pad | drop_full
        return super().forward(tok_emb, types, key_pad)


def build(**params):
    return R22ObservationOcclusionDenoising(**params)
