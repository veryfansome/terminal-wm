"""R21 arch: EVIDENCE-GATED IMAGINATION WRITE — the r20 native imagination-write
(measured at full scale: dedup b-margin +0.2827 vs the plain r18 path-state arch's
+0.2493, Δb = +0.0334, every genuine family non-negative, six independent R20
trainings agreeing at +0.03–0.04) with its ONE measured defect repaired: the write
may no longer execute from a vacuous state.

THE DEFECT (localized by Candidate 1's R21 BLOCK): on the instrument's history-masked
arm the transition memory is EMPTY, so the imagined write w*(f(s_pre,c_m)-s_pre)
degenerates to the pure command decode w*beta(c_m) — a history-free hallucination that
lifted IMAG_hist by +0.0668 dedup (redir:prod> +0.151, redir:echo> +0.176), consumed
the differential (C1 ΔIMAG_HA −0.0262) and breached both redirect family floors.

THE REPAIR (this file): the imagined correction is scaled by an EVIDENCE GATE
    g_i = evid_i / (evid_i + c),   evid_i = Σ_{j<i} ||observed_corr_j||²
the accumulated squared mass of the memory's OBSERVED (real-obs) writes strictly
before i. No key_pad statistic is read beyond the r18 arch's own active/imag masks;
the same code runs identically in both instrument arms. With any observed history
evid saturates g -> ~1 (measured on 2336 real TRAIN-image windows with trained
nets, 2 seeds: evid >= 1510 at every imagination step, so g >= 0.99983 at c=0.25;
b-arm prediction delta vs the ungated write mean 3.9e-5 / max 1.2e-3 on ~27-norm
vectors; b_top1 identical to 4 decimals, pooled and per-family); with NO observed
history evid == 0 exactly, so g == 0, the write is ALGEBRAICALLY zero, memory stays
empty, and the history-masked forward is value-identical to the plain r18 arch's
(measured max|Δ| = 0.0 across all windows on both seeds, while the ungated write
moves the same predictions by up to 1.48 — the contamination channel, removed
exactly). This is the
round's zero-on-empty-history honesty construction (Inventor 2's evidence-only algebra)
lifted from the head to the arch, and it is standard estimation theory: an innovation
update from a zero-information prior gets zero gain (Kalman); model rollouts are trusted
only where the model has evidence (MBPO, arXiv:1906.08253); constructive episodic
simulation composes stored episode elements — an empty store simulates nothing
(Schacter–Addis–Buckner 2007). A world model that "imagines" file content with no
observed evidence is hallucinating, and the gate repairs exactly that.

WHY THE FITNESS PATH IS PROVABLY UNCHANGED (inherited from the r20 write, re-verified):
training/fitness streams are even-length with pairs both-valid or both-padded, so the
imag mask is identically False and the gated term contributes exact zeros; ZERO new
parameters (the gate constant is a plain float attribute) -> state_dict, init-RNG
stream, gradients, optimizer routing (Muon keys + the (D,D) spectral-cap signature) are
bit-identical to the r18 arch; verified even-stream forward max|Δ| = 0.0 (pred AND h)
against the verbatim r18 class, and the r20 imagwrite genome measured pod fitness
0.4245 vs the r18 arch's 0.4247 (|Δ| = 0.0002, within run noise) on this identical
training.
The (a)-path operator instrument reads `_transition_reads` on a fully-observed prefix
layout with no masked-obs pair, so it is untouched (verified max|Δ| = 0.0).

LEAK-FREE: the masked pair's obs VALUE is never read (the obs term is gated by the
r18 arch's own active mask; evid accumulates only active-gated observed corrections;
key_pad excludes the slot from attention) — PAD-value perturbation Δ == 0.0 exactly.

Refs: Kalman innovation gain under prior information; MBPO "When to Trust Your Model"
(arXiv:1906.08253); gated/erase-capable delta writes (Gated DeltaNet, arXiv:2412.06464);
fast-weight delta-rule memory (arXiv:2102.11174); constructive episodic simulation
(Schacter, Addis & Buckner 2007, Nat Rev Neurosci 8:657); masked latent prediction
(I-JEPA, arXiv:2301.08243).
"""

import torch

from evolve.chunks.arch.r18_pathstate_latent_transition_worldmodel import (
    D,
    R18PathStateLatentTransition,
)

NAME = "r21_evidencegated_imagwrite_algebraic_observer_arch"
DESCRIPTION = (
    "The r18 path-state arch + the r20 parameter-free imagination write, with the "
    "imagined correction scaled by an evidence gate g = evid/(evid+c), evid = accumulated "
    "squared mass of strictly-earlier OBSERVED memory writes. With observed history g "
    "saturates to ~1 (b-arm write preserved, measured); with none g == 0 exactly, so the "
    "history-masked forward is value-identical to the plain r18 forward (no empty-memory "
    "command-decode hallucination — the channel that lifted C1's IMAG_hist and broke its "
    "redirect family floors). Even-length fitness streams are bit-identical to the "
    "r18 forward: zero new parameters, same init RNG, same gradients, same optimizer routing."
)


class R21EvidenceGatedImagWrite(R18PathStateLatentTransition):
    """Subclass overriding ONLY `_transition_reads` (the evidence-gated imagination-write
    branch); every other r18 method — trunk, memories, gates, `_transition`,
    `transition_from_emb`, forward — is inherited verbatim. `imag_gate_c` is a plain
    Python float (NOT a Parameter/buffer): the state_dict and init-RNG stream are
    bit-identical to the r18 arch."""

    def __init__(self, imag_gate_c=0.25, **params):
        super().__init__(**params)
        # c > 0 is required for the exact-zero algebra (0/(0+c) == 0.0); measured
        # evid on real b-arm windows is >= ~1.5e3, so any c in (0, ~10] leaves the
        # observed-history gate saturated at ~1 — there is no tuning surface here.
        self.imag_gate_c = max(1e-6, float(imag_gate_c))

    def _transition_reads(self, x_cmd, obs_tok, valid_cmd, valid_obs, n_cmd, n_pair):
        B = x_cmd.size(0)
        dtype = x_cmd.dtype
        p = self._unit(self.tr_path(x_cmd))                          # [B,n_cmd,key_d]
        w = torch.sigmoid(self.tr_mut_gate(x_cmd)).squeeze(-1)       # [B,n_cmd]
        decay = 0.90 + 0.099 * torch.sigmoid(self.logit_decay)
        decay = decay.to(dtype)
        mem = x_cmd.new_zeros(B, self.key_d, D)
        evid = x_cmd.new_zeros(B, 1)   # accumulated squared mass of OBSERVED writes < i
        c = float(self.imag_gate_c)
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
                # r20 imagination write: valid command whose observation is MASKED.
                # Identically 0 on fully-observed streams (pairs pad together), so the
                # fitness path never executes it. The obs VALUE is never read here.
                imag = (valid_cmd[:, i] & ~valid_obs[:, i]).to(dtype).unsqueeze(-1)
            else:
                obs_i = s_pre.new_zeros(B, D)
                wi = w[:, i].unsqueeze(-1) * 0.0
                active = x_cmd.new_zeros(B, 1)
                imag = x_cmd.new_zeros(B, 1)
            v_i = (1.0 - wi) * obs_i + wi * delta                    # r18 write value
            observed_corr = (v_i - s_pre) * active
            # R21 EVIDENCE GATE: exact 0 with no strictly-earlier observed write
            # (0/(0+c) == 0.0), saturates to ~1 under any real observed history.
            g = evid / (evid + c)
            imagined_corr = g * (w[:, i].unsqueeze(-1) * (delta - s_pre)) * imag
            mem = decay * mem + torch.bmm(
                pi.unsqueeze(2), (observed_corr + imagined_corr).unsqueeze(1)
            )
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            evid = evid + observed_corr.pow(2).sum(dim=-1, keepdim=True)
        return torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)


def build(**params):
    return R21EvidenceGatedImagWrite(**params)
