"""R22 head: IN-PASS MASKED-ENDPOINT TRUNK TASK — the champion forward-model consistency
aux VERBATIM, plus a second training term that forwards the TRUNK ITSELF on in-batch-built
obs-missing masked-endpoint layouts (the instrument's exact (b)-layout) and supervises the
prediction at the read slot with an eval-geometry contrastive loss. Gradients flow into the
whole trunk. ZERO new parameters — the state_dict, module set, and optimizer routing
(Muon/spectral-cap signatures) are bit-identical to the champion's.

WHY THE TRUNK (the R20/R21 record, engaged): the champion's IMAG_CA (0.5471) comes from its
TRUNK — every bolt-on module measured ~0 content-attributable at full scale (R21: honest
observers' mini gains +0.05-0.09 compressed to <= +0.004, 4x-replicated subsumption; the
evidence-gated write family's HA gain F1-decomposed to ~86% presence-gated command-decode,
content channel -0.0005). Meanwhile NOTHING in the single training pass ever poses the
masked-endpoint problem to the trunk: by the time a read at r is scored in the normal
interleave stream, obs_{m..r-1} are present, so the trunk's 0.776 dedup b-top1 on the masked
layout is pure out-of-distribution generalization. The r20 masked-window head posed the task
inside the pass but on a DEDICATED module with detached inputs ("so it cannot destabilize
the trunk") — and the instrument reads the trunk, not the module. The refuted r19 attempt
posed it to the trunk POST-HOC (unstable all-params fine-tune at the end, non-reproducing).
This head does the one untried thing: pose the task to the trunk, INSIDE the pass, ramped
and anchored by the main loss — closing the train/measure distribution gap on the object
the frozen instrument actually measures.

MINING (metadata-free, measured on TRAIN images with the champion's canonical s0 ckpt):
the arch's own trained mutation gate w_mut = sigmoid(tr_mut_gate(in_norm(cmd_proj(z_cmd) +
type_emb[0]))) separates mutation commands from reads (intervene mean 0.958, 99.7% > 0.5;
reads 0.40; fires on EVERY family incl. redir:prod> 0.99 / mkdir 0.826 — where the champion
head's cmd-cosine rule has 0.0 coverage). Rule: for each command position j with an observed
label (and w_mut(j) <= floor), k(j) = the NEAREST EARLIER position with w_mut(k) > floor
(floor 0.8, swept: every family supplies true pairs, hidden-cause labels 0.7%); build
  [cmd_0, obs_0, ..., cmd_{k-1}, obs_{k-1}, cmd_k, PAD-obs, cmd_j]
(prefix pairs < k kept, obs at k masked, steps (k, j) CUT, read cmd at 2k+2 — byte-matching
the frozen instrument's build_b_layout semantics) and supervise pred[2k+2] against z_j.
Because k is the LAST gate-crossing position before j, hidden causes are structurally
suppressed (measured 0.7% of pairs); ~56% of harvested genuine windows are recovered
exactly and the rest become visible-cause compressed-layout examples (the same task
family). The gate starts undifferentiated (bias -1.0 => w~0.27 < floor) so the term is
SILENT until the trunk's own training matures the gate — an automatic curriculum; if the
gate never differentiates the term stays 0.0 and the genome degrades to the champion.

LOSS (eval geometry): per-dim-mean squared-L2 logits InfoNCE (tau 0.25) over the mined
labels + sampled batch targets, negatives importance-weighted by READ-COMMAND cosine (the
instrument's same-verb foil protocol, emulated without metadata), near-duplicate targets
(cos > 0.98) masked as false negatives, small MSE anchor for norm calibration, pair-weighted
by the detached gate value, smoothstep-ramped.

WHY THIS RAISES CONTENT-ATTRIBUTION (CA) RATHER THAN DECODE: same-verb-weighted negatives
make command-decode insufficient on exactly the deciding foils (same read verb ties decode
priors; content discriminates). The training distribution contains ONLY coherent
(prefix, endpoints) pairs — an incoherence/donor detector has no training signal, so the
coherence-saboteur class named in the prereg cannot be learned from this term; the
wrong-history arm's behavior stays the honest misled-by-content the champion already shows.

Causal / leak-free: eval forward untouched (wrap adds no module, registers no params, never
re-points forward — the arch's forward is bit-identical to the plain champion's at eval).
The masked forward's inputs are strictly plan-time (gather sources are positions <= 2k plus
the read command 2j); z_j enters ONLY as a loss label (sanctioned — the champion aux already
consumes future obs as labels); the future read command is a train-only aux INPUT mirroring
the sanctioned endpoint formulation (the measurement itself supplies c_r as the query) and
never touches any scored prediction.

Refs: I-JEPA masked latent prediction (arXiv:2301.08243); CPC/InfoNCE (arXiv:1807.03748);
debiased contrastive false-negative masking (arXiv:2007.00224); the R20/R21 measured record
(subsumption + F1 decode decomposition) for the trunk-not-module targeting.
"""

import math

import torch
import torch.nn.functional as F

NAME = "r22_masked_endpoint_trunk_task"
DESCRIPTION = (
    "Champion forward-model consistency verbatim + an in-pass MASKED-ENDPOINT task posed to "
    "the TRUNK itself: mine (last-gated-mutation k -> later read j) pairs with the arch's own "
    "detached mutation gate (floor 0.8, measured), build the frozen instrument's exact "
    "obs-missing compressed layout in-batch, forward the same net, and train pred[2k+2] "
    "against z_j with a same-verb-weighted L2-InfoNCE + MSE anchor (ramped, pair-weighted). "
    "Zero new parameters; eval forward bit-identical; silent until the gate matures; "
    "champion-term-only on archs without the gate."
)

_DEFAULTS = {
    # ---- champion forward-model consistency term (verbatim constants) ----
    "row_frac": 0.6,       # fraction of batch rows the aux mining runs on
    "path_thresh": 0.60,   # frozen cmd-cosine floor for "same path"
    "change_floor": 0.25,  # min mean-sq change ||obs_i-obs_j||^2 to call it a mutation
    "max_examples": 512,   # cap mined triples per step (cost control)
    "cos_weight": 0.10,    # operator -> obs_j cosine reconstruction
    "mse_weight": 0.02,    # small metric anchor
    "aux_weight": 1.0,
    "ramp_steps": 400,     # smoothstep ramp so early training is main-loss-dominated
    # ---- NEW: in-pass masked-endpoint trunk task ----
    "imag_weight": 0.5,        # overall weight of the trunk task
    "imag_ramp_steps": 800,    # its own (slower) smoothstep ramp
    "imag_gate_floor": 0.8,    # w_mut floor for a mined mutation slot (swept on train data)
    "imag_max_pairs": 64,      # per-step cap on masked windows (cost control)
    "imag_neg_extra": 128,     # extra sampled batch-target negatives
    "imag_tau": 0.25,          # temperature on per-dim-mean sqL2 logits (eval geometry)
    "imag_kappa": 4.0,         # same-verb (read-cmd cosine) negative up-weighting
    "imag_dup_cos": 0.98,      # near-duplicate targets masked as false negatives
    "imag_mse_anchor": 0.05,   # small norm-calibration anchor
    "imag_period": 2,          # run the trunk task every 2nd step (amortized cost control)
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _interleave_layout_ok(b):
    for k in ("tok", "types", "key_pad", "tgt", "cmd_mask"):
        if k not in b:
            return False
    tok, types, key_pad, tgt, cmd_mask = b["tok"], b["types"], b["key_pad"], b["tgt"], b["cmd_mask"]
    if tok.dim() != 3 or tgt.dim() != 3 or types.dim() != 2 or key_pad.dim() != 2:
        return False
    if tok.shape[:2] != types.shape or tok.shape[:2] != key_pad.shape:
        return False
    if cmd_mask.shape != tgt.shape[:2] or tok.shape[0] != tgt.shape[0] or tok.shape[2] != tgt.shape[2]:
        return False
    if tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


def _has_gate(net):
    for attr in ("tr_mut_gate", "cmd_proj", "type_emb", "in_norm"):
        if getattr(net, attr, None) is None:
            return False
    return True


def wrap(net, D, **params):
    """No forward re-point, no module cycle, no new module, NO NEW PARAMETERS: both terms use
    the ARCH's own modules by reference. Returns a config dict; the champion term is disabled
    if the arch lacks `transition_from_emb`, the trunk task if it lacks the mutation-gate
    featurization path (passthrough-equivalent in both cases)."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    cfg["_imag_disabled"] = not _has_gate(net)
    return cfg


# ==== champion term: forward-model consistency on the shared transition operator (VERBATIM) ====

@torch.no_grad()
def _mine_triples(cmd, obs, valid, path_thresh, change_floor):
    """Per row, mine (row, i, k, j) same-path triples with obs changed across k, plus a weight.
    cmd/obs [B,maxn,D] standardized; valid [B,maxn] bool. FULLY VECTORIZED — identical to the
    champion head's mining (r18_transition_forwardmodel_consistency)."""
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))                         # [B,maxn,maxn] cmd cosine
    vmask = valid.bool()                                           # [B,maxn]

    pos = torch.arange(maxn, device=device)
    lower = pos.unsqueeze(1) > pos.unsqueeze(0)                    # [maxn,maxn] p<k (rows=k)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)                    # [maxn,maxn] p>k
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)           # [B,maxn,maxn]

    before = same_path & lower.unsqueeze(0)
    after = same_path & upper.unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)

    has_i = i_idx >= 0
    has_j = j_idx < maxn
    triple_ok = has_i & has_j & vmask

    ic = i_idx.clamp(0, maxn - 1)
    jc = j_idx.clamp(0, maxn - 1)
    obs_i = torch.gather(obs, 1, ic.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    obs_j = torch.gather(obs, 1, jc.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    change = (obs_i - obs_j).pow(2).mean(dim=-1)
    sim_ki = torch.gather(sim, 2, ic.unsqueeze(-1)).squeeze(-1)
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)
    w = sim_ki * sim_kj * change

    keep = triple_ok & (change >= change_floor) & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    sel_i = i_idx[sel_b, sel_k]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_i, sel_k, sel_j, sel_w


def _transition_term(cfg, batch, net, device, ramp):
    """The champion aux, bit-for-bit (same mining, weights, loss math, RNG call)."""
    if cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = obs[r, ti].detach()
    cmd_k = cmd[r, tk].detach()
    tgt = obs[r, tj].detach()
    pre = torch.nan_to_num(pre, nan=0.0, posinf=1e4, neginf=-1e4)
    cmd_k = torch.nan_to_num(cmd_k, nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

    pred = op(pre, cmd_k)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    total = float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err
    return float(cfg["aux_weight"]) * ramp * total


# ==== NEW term: the in-pass masked-endpoint trunk task ====

@torch.no_grad()
def _gate_values(net, cmd_flat):
    """The arch's own mutation gate on raw standardized command embeddings, via the same
    featurization `transition_from_emb` uses (cmd_proj + cmd type row + in_norm; position
    omitted). Detached — mining pressure never shapes the gate directly."""
    idx0 = torch.zeros(cmd_flat.size(0), dtype=torch.long, device=cmd_flat.device)
    x = net.in_norm(net.cmd_proj(cmd_flat) + net.type_emb(idx0))
    return torch.sigmoid(net.tr_mut_gate(x)).squeeze(-1)


def _imag_term(cfg, batch, net, device, step):
    if cfg.get("_imag_disabled", True):
        return 0.0
    if float(cfg["imag_weight"]) <= 0.0:
        return 0.0
    period = max(1, int(cfg["imag_period"]))
    if (step % period) != 0:
        return 0.0
    ramp = _smoothstep(step / max(1.0, float(cfg["imag_ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    tgt_full = batch["tgt"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0
    L = tok.shape[1]
    Dd = tok.shape[2]
    cmd = tok[:, 0::2][:, :maxn]                                   # [B,maxn,D] raw z_cmd

    with torch.no_grad():
        w = _gate_values(net, torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)
                         .reshape(-1, Dd)).reshape(B, maxn)
        w = torch.nan_to_num(w, nan=0.0)

    floor = float(cfg["imag_gate_floor"])
    gated = (w > floor) & cmd_mask                                 # mutation-like slots
    pos = torch.arange(maxn, device=device)
    gpos = torch.where(gated, pos.unsqueeze(0).expand(B, maxn),
                       torch.full((B, maxn), -1, dtype=torch.long, device=device))
    kmax = gpos.cummax(dim=1).values                               # latest gated <= j
    k_of = torch.cat([torch.full((B, 1), -1, dtype=torch.long, device=device),
                      kmax[:, :-1]], dim=1)                        # strictly earlier
    elig = cmd_mask & (~gated) & (k_of >= 1)                       # labeled j, non-gated, prefix>=1
    if not bool(elig.any()):
        return 0.0

    nz = elig.nonzero(as_tuple=False)
    rows, js = nz[:, 0], nz[:, 1]
    ks = k_of[rows, js]
    pw = w[rows, ks]                                               # pair weight = gate at k
    P = rows.numel()
    cap = int(cfg["imag_max_pairs"])
    if P > cap:
        sel = torch.multinomial(pw.clamp_min(1e-6).float(), cap, replacement=False)
        rows, js, ks, pw = rows[sel], js[sel], ks[sel], pw[sel]
        P = cap

    # ---- build the instrument's (b)-layout in-batch (compressed, obs-missing) ----
    # [cmd_0, obs_0, ..., cmd_{k-1}, obs_{k-1}, cmd_k, PAD-obs, cmd_j]; read pred at 2k+2.
    mm = ks
    rr = js
    Lm = int(2 * int(mm.max().item()) + 3)
    p2 = torch.arange(Lm, device=device).unsqueeze(0).expand(P, Lm)
    end_col = (2 * mm + 2).unsqueeze(1)
    src = torch.where(p2 == end_col, (2 * rr).unsqueeze(1), p2.clamp_max(L - 1))
    keep = (p2 <= (2 * mm).unsqueeze(1)) | (p2 == end_col)         # prefix+cmd_k+cmd_j live
    tok2 = tok[rows].gather(1, src.unsqueeze(-1).expand(P, Lm, Dd))
    tok2 = tok2 * keep.unsqueeze(-1).to(tok2.dtype)                # PAD/right-pad zeroed
    types2 = ((p2 % 2 == 1) & (p2 <= (2 * mm + 1).unsqueeze(1))).long()  # instrument semantics
    kp2 = ~keep
    tok2 = torch.nan_to_num(tok2, nan=0.0, posinf=1e4, neginf=-1e4)

    pred_all, _ = net(tok2, types2, kp2)                           # SECOND trunk forward (grads in)
    pred = pred_all[torch.arange(P, device=device), 2 * mm + 2]
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    zj = torch.nan_to_num(tgt_full[rows, rr].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    cj = cmd[rows, rr].detach()

    # ---- extra negatives: sampled batch targets (+ their commands, for verb weighting) ----
    flat_t = tgt_full[cmd_mask].detach()
    flat_c = cmd[cmd_mask].detach()
    extra = int(cfg["imag_neg_extra"])
    if flat_t.shape[0] > extra > 0:
        es = torch.randint(0, flat_t.shape[0], (extra,), device=device)
        neg_t, neg_c = flat_t[es], flat_c[es]
    else:
        neg_t, neg_c = flat_t, flat_c
    cand_t = torch.cat([zj, neg_t], dim=0)                         # [C,D]; positive of row i = col i
    cand_c = torch.cat([cj, neg_c], dim=0)
    cand_t = torch.nan_to_num(cand_t, nan=0.0, posinf=1e4, neginf=-1e4)

    d2 = (pred.pow(2).sum(1, keepdim=True) + cand_t.pow(2).sum(1).unsqueeze(0)
          - 2.0 * pred @ cand_t.t()).clamp_min(0.0) / float(Dd)    # per-dim-mean sqL2 (eval metric)

    ar = torch.arange(P, device=device)
    with torch.no_grad():
        vsim = (F.normalize(cj, dim=-1) @ F.normalize(cand_c, dim=-1).t()).clamp(-1.0, 1.0)
        a = 1.0 + float(cfg["imag_kappa"]) * vsim.clamp_min(0.0)   # same-verb foil emulation
        tsim = F.normalize(zj, dim=-1) @ F.normalize(cand_t, dim=-1).t()
        dup = tsim > float(cfg["imag_dup_cos"])                    # false negatives
        dup[ar, ar] = False
        loga = a.clamp_min(1e-9).log().masked_fill(dup, float("-inf"))
        loga[ar, ar] = 0.0                                         # positive weight exactly 1

    logits = -d2 / float(cfg["imag_tau"]) + loga
    nll = F.cross_entropy(logits, ar, reduction="none")
    wn = (pw.to(pred.dtype) / pw.to(pred.dtype).sum().clamp_min(_EPS)).detach()
    nce = (wn * nll).sum()
    anchor = (wn * (pred - zj).pow(2).mean(dim=-1)).sum()
    return float(cfg["imag_weight"]) * ramp * (nce + float(cfg["imag_mse_anchor"]) * anchor)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0
    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    step = cfg["_step"]
    ramp = _smoothstep(step / max(1.0, float(cfg["ramp_steps"])))

    total = 0.0
    if ramp > 0.0:
        t = _transition_term(cfg, batch, net, device, ramp)        # champion term (verbatim)
        if torch.is_tensor(t):
            if bool(torch.isfinite(t).item()):
                total = total + t
        elif t:
            total = total + t
    m = _imag_term(cfg, batch, net, device, step)                  # NEW trunk task
    if torch.is_tensor(m):
        if bool(torch.isfinite(m).item()):
            total = total + m
    elif m:
        total = total + m

    if torch.is_tensor(total) and not bool(torch.isfinite(total).item()):
        return 0.0
    return total


def leak_safe(mod, params):
    """Forward untouched (wrap adds no module, registers no params, never re-points forward).
    The champion term consumes future obs strictly as loss labels; the trunk task's masked
    forward gathers ONLY positions <= 2k plus the read command 2j (verified construction) and
    consumes z_j strictly as a loss label. Validate params are finite and in range."""
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        0.0 < vals["row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["change_floor"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["imag_weight"] >= 0.0,
        vals["imag_ramp_steps"] >= 1.0,
        0.0 < vals["imag_gate_floor"] < 1.0,
        vals["imag_max_pairs"] >= 1.0,
        vals["imag_neg_extra"] >= 0.0,
        vals["imag_tau"] > 0.0,
        vals["imag_kappa"] >= 0.0,
        0.0 < vals["imag_dup_cos"] <= 1.0,
        vals["imag_mse_anchor"] >= 0.0,
        vals["imag_period"] >= 1.0,
    ]
    return all(checks)
