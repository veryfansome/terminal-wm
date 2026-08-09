"""objective chunk (R23, WANTS_CTX): COMMAND-RESIDUAL within-command contrastive — the
r12 anti-retrieval ring loss (verbatim base) PLUS a content-routing term that strips the
command-decodable component out of both prediction and target and contrasts on the RESIDUAL, so
command-decode earns ZERO credit for it.

THE DIAGNOSIS (operator-verified, R23 brief): the ordinary next-obs loss is cheaply satisfied by
COMMAND-DECODE (trained command-only ceiling 0.51-0.56 on the genuine families), and IMAG_CA cancels
command-decode by construction (real-prefix vs presence-matched content-mismatched donor, same
command) — so a loss whose cheapest descent is command-decode nets ~0 content-attributable margin.
inv7's control settles it: transplanting the frozen probe's EXACT cross-attention into the trunk
under this ordinary loss REGRESSED CA (-0.011); the same computation under a dedicated
endpoint-retrieval objective banked +0.18. The binding variable is the OBJECTIVE, not the data or the
architecture. To move IMAG_CA the loss must route gradient onto the command-RESIDUAL — the part of the
target the command alone cannot predict, which is exactly what IMAG_CA rewards.

THE MECHANISM (uses ctx["cmd"], the command-INPUT embedding per row):
  1. Same-command siblings, in-batch. The e5 command embedding is a deterministic function of the
     command STRING, so exact-command rows have near-zero embedding distance while distinct commands
     are far away — MEASURED gap on TRAIN (dockerfs3-e5ft): same-string pairs sqdist <= 0.0019,
     distinct-string pairs sqdist >= 17.96 (n=1.1M pairs). A hard threshold at 1.0 isolates exact
     siblings with zero error. Under the sysblock hard-negative batcher the flattened batch is ~1785 cmd-rows and
     ~50% of them have an exact-command sibling IN-BATCH (measured 848-912/batch across the ramp) — a
     DENSE signal, not the sparse per-64-row picture.
  2. Command-conditional mean = the command-decodable prediction. cmd_hat_i = leave-one-out mean of
     the batch targets over row i's exact-command siblings (detached). MEASURED cos(cmd_hat, tgt) on
     sibling rows = 0.60-0.68 (matches the GLOBAL exact-command-mean cos 0.64), residual ||tgt-cmd_hat||
     = ~0.6 * ||tgt||. cmd_hat is the honest in-batch E[tgt|cmd]; the residual r_tgt = tgt - cmd_hat is
     the history-content the command cannot supply.
  3. Within-command residual contrastive. r_pred_i = pred_i - cmd_hat_i (gradient only through pred),
     r_tgt_i = tgt_i - cmd_hat_i (detached). Each sibling row i must retrieve its OWN residual r_tgt_i
     against its exact-command group's residuals {r_tgt_j : cmd_j == cmd_i} in the eval's per-dim-mean
     squared-L2 geometry (temperature _RTAU). Same-command same-answer pairs (target sqdist < _DUP)
     are masked as false negatives (their content does not diverge). This is the eval's own decision
     restricted to the IMAG_CA-deciding foils: same command, divergent content — rank YOUR content
     above your command-twin's content.

WHY command-decode is an INSUFFICIENT minimizer (measured on a real batch, step 2000, 512 sibling
rows): the within-command residual NLL at pred = cmd_hat (pure command-decode, r_pred = 0) = 7.73 —
WORSE than the uniform ceiling ln(n) = 6.24 and far above the content optimum pred = tgt (3.27),
because r_pred = 0 makes every row's residual identical and the group cannot be resolved. The
descent direction at command-decode aligns with the content residual: mean cos(-grad, r_tgt) = 0.838.
So the command-mean prediction — the exact failure mode the brief names — is a near-MAXIMIZER of the
added term; the only way down is to make pred_i carry the row-specific history-content.

WHY IMAG_CA (not just fitness): the term's supervised quantity is the command-RESIDUAL, the same
object IMAG_CA isolates (real vs content-mismatched donor at a fixed command). Command-decode
components are removed from both sides before the contrastive, so a command-decoding solution gains
nothing here — a gain requires genuine history-content use, which does not transfer to a
content-mismatched donor prefix (wrong arm) and therefore shows up as +CA, not as a decode shortcut.

CONTRACT / SAFETY:
  * WANTS_CTX = True; signature loss(pred, tgt, ctx). Pure function of (pred, tgt, ctx["cmd"]); NO
    module state, NO trainable params, NO RNG, NO in-place edits of inputs. cmd_hat / masks / gate all
    DETACHED. Adds a few [n,n] ops at the same n (r12 already runs this scale).
  * Causal: ctx["cmd"] is a model INPUT (the command), never a future obs — leakage_ok is untouched.
  * NaN-safe: the diagonal is always same-command (self, sqdist 0), so every gated row's masked
    softmax has >=1 finite entry; clamp_min on counts/denominators; n<2 -> MSE anchor only; the
    residual term is 0 when no sibling row has a divergent-content twin. Harness fails closed on
    non-finite loss regardless.
  * Anti-collapse: the r12 base is anti-collapse-safe unchanged; a constant prediction gives r_pred_i
    = c - cmd_hat_i (varies with i, does not match r_tgt) and a command-mean prediction gives r_pred=0
    (measured near-maximal, above uniform) — neither minimizes the added term. Collapse and
    command-decode are both strictly penalized.
"""

import torch
import torch.nn.functional as F

WANTS_CTX = True

NAME = "r23_command_residual_contrastive"
DESCRIPTION = (
    "Champion r12 anti-retrieval ring loss (verbatim base) PLUS a within-command residual contrastive: "
    "strip the leave-one-out exact-command-conditional mean (the command-decodable component, via "
    "ctx['cmd']) from prediction and target and require each sibling row to retrieve its OWN content "
    "residual against its exact-command group in the eval's squared-L2 geometry. Command-decode "
    "(r_pred=0) is a near-maximizer of the added term, so the loss routes gradient onto the "
    "history-content the ordinary next-obs loss leaves unbanked — the quantity IMAG_CA rewards."
)

# ---- r12 constants (unchanged) ----
_TEMP = 0.25
_GAMMA = 1.0
_ANCHOR = 0.05
_BETA = 0.5
_EPS = 1e-2
_WMIN, _WMAX = 0.25, 4.0
_KAPPA = 4.0
_DELTA = 0.05
_LAM_FRAC = 0.5
_LAMBDA_REP = 0.1
_MARGIN = 0.5
_TAU_R = 0.25
_GEPS = 1e-3

# ---- command-residual term constants ----
_CMD_EXACT = 1.0    # cmd-embedding sqdist below which two rows are the SAME exact command.
                    # MEASURED gap (dockerfs3-e5ft train): same-string <= 0.0019, distinct >= 17.96.
_RTAU = 0.25        # residual-contrastive temperature (eval per-dim-mean sqL2, matches _TEMP)
_DUP = 0.05         # per-dim-mean target sqdist below which a same-command sibling is the SAME
                    # ANSWER (false negative — content does not diverge). Matches r12 _DELTA scale.
_LAMBDA_RESID = 0.5  # weight of the content-routing term (r12 base stays dominant;
                     # the term is the ONLY gradient on content at the command-decode equilibrium).


def _pdmean_sq(a, b, d):
    an = (a * a).sum(1, keepdim=True)
    bn = (b * b).sum(1, keepdim=True)
    return (an + bn.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(d)


def loss(pred, tgt, ctx):
    n, d = pred.shape

    # ================= r12 BASE (verbatim) =================
    mse_anchor = ((pred - tgt) ** 2).mean()
    if n < 2:
        return mse_anchor

    with torch.no_grad():
        mse_d = ((pred - tgt) ** 2).mean(dim=0)
        w = (1.0 / (mse_d + _EPS)).pow(_BETA)
        w = w / w.mean().clamp_min(1e-12)
        w = w.clamp(_WMIN, _WMAX)
        w = w / w.mean().clamp_min(1e-12)
        sw = w.sqrt().unsqueeze(0)

    pw = pred * sw
    tw = tgt * sw
    pw_sq = (pw * pw).sum(dim=1, keepdim=True)
    tw_sq = (tw * tw).sum(dim=1, keepdim=True)
    dist2 = pw_sq + tw_sq.t() - 2.0 * (pw @ tw.t())
    dist2 = dist2.clamp_min(0.0) / float(d)

    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        tt = (tw_sq + tw_sq.t() - 2.0 * (tw @ tw.t())).clamp_min(0.0) / float(d)
        mean_off = (tt.sum() / (n * (n - 1))).clamp_min(_EPS)
        lam = (_LAM_FRAC * mean_off).clamp_min(_EPS)
        confus = torch.exp(-tt / lam)
        dupmask = 1.0 - torch.exp(-tt / _DELTA)
        ring = (confus * dupmask).masked_fill(eye, 0.0)
        a_raw = 1.0 + _KAPPA * ring
        row_mean = a_raw.masked_fill(eye, 0.0).sum(dim=1, keepdim=True) / (n - 1)
        a = (a_raw / row_mean.clamp_min(1e-6)).masked_fill(eye, 1.0)
        log_a = a.clamp_min(1e-6).log()
        mass = ring.sum(dim=1)
        q = ring / mass.clamp_min(1e-6).unsqueeze(1)
        gate = mass / (mass + _GEPS)

    logits = -dist2 / _TEMP + log_a
    labels = torch.arange(n, device=pred.device)
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    with torch.no_grad():
        p_true = (-nll).exp().clamp(0.0, 1.0)
        focal = (1.0 - p_true).pow(_GAMMA)
    listwise = (focal * nll).mean()

    d_true = dist2.diagonal()
    d_conf = (q * dist2).sum(dim=1)
    rep = (gate * F.softplus((d_true + _MARGIN - d_conf) / _TAU_R)).mean()

    base = listwise + _ANCHOR * mse_anchor + _LAMBDA_REP * rep

    # ================= COMMAND-RESIDUAL CONTENT ROUTING (new) =================
    cmd = ctx["cmd"]                                              # [n, D] command-INPUT embedding
    with torch.no_grad():
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        cn = (cmd * cmd).sum(dim=1, keepdim=True)
        dcmd = (cn + cn.t() - 2.0 * (cmd @ cmd.t())).clamp_min(0.0)   # [n,n] command sqdist
        same_cmd = dcmd < _CMD_EXACT                             # exact-command (incl diagonal)
        sib = same_cmd & ~eye                                    # LOO siblings
        sib_cnt = sib.sum(dim=1)                                 # [n]

        # command-conditional mean (the command-decodable component), detached, LOO.
        cmd_hat = (sib.float() @ tgt) / sib_cnt.clamp_min(1).unsqueeze(1)   # [n, D]

        # false-negative guard: a same-command sibling whose FULL target is the same answer.
        tt_plain = _pdmean_sq(tgt, tgt, d)                      # [n,n] per-dim-mean target sqdist
        dup_off = (tt_plain < _DUP) & ~eye
        cols = same_cmd & ~dup_off                              # softmax columns: self + divergent sibs
        divergent_sib = (same_cmd & ~dup_off & ~eye).sum(dim=1)  # [n]
        row_gate = (divergent_sib >= 1).float()                # rows with >=1 divergent content twin

    resid = pred.new_zeros(())
    denom = row_gate.sum()
    if float(denom) > 0.0:
        r_pred = pred - cmd_hat                                 # grad via pred
        r_tgt = tgt - cmd_hat                                   # detached content residual
        r_dist2 = _pdmean_sq(r_pred, r_tgt.detach(), d)         # [n,n], grad via r_pred
        rlogits = (-r_dist2 / _RTAU).masked_fill(~cols, float("-inf"))
        rlogp = F.log_softmax(rlogits, dim=1)
        rnll = -rlogp.diagonal()                               # positive = self (own residual)
        resid = (row_gate * rnll).sum() / denom.clamp_min(1.0)

    return base + _LAMBDA_RESID * resid
