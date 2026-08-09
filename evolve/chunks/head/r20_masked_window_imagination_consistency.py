"""R20 head: champion FORWARD-MODEL CONSISTENCY verbatim + a MASKED-WINDOW IMAGINATION
aux that trains the co-designed r20 arch's `imaginer` on in-batch-mined mutation->read
windows — the missing training signal for direct-endpoint imagination.

THE R20 TARGET (measured): the champion composes its obs-calibrated transition operator
at -0.105 vs the lexical floor on mutation->read windows, while a frozen-embedding probe
(cross-attention over the plan-time prefix queried by the two endpoint commands) proves
+0.16-0.30 headroom exists on every family. Nothing in the champion's single training
pass ever POSES the imagination problem: by the time a read at r is scored, the obs at
m..r-1 are present in the stream, so no mechanism is ever trained to predict z_r from
(H_{<m}, c_m, c_r) alone. The refuted r19 attempt posed it POST-HOC (unstable all-params
InfoNCE fine-tune, non-reproducing); this head poses it INSIDE the pass, on a dedicated
module with stationary (frozen-embedding) inputs, so it cannot destabilize the trunk.

MECHANISM (train-only; eval forward untouched — wrap adds no module, never re-points):
  1. CHAMPION TERM, VERBATIM: mine same-path (i < k < j) triples by frozen cmd-cosine
     with the observation changed across k, and require the arch's shared operator
     f(obs_i, cmd_k) ~ obs_j (cos+MSE, change-weighted, ramped). Identical code, weights
     and RNG consumption to `r18_transition_forwardmodel_consistency` — the fitness-
     earning aux is preserved bit-for-bit.
  2. NEW IMAGINATION TERM: mine same-path (k -> j = nearest later touch) ENDPOINT pairs
     (no earlier-touch requirement — ~90% of genuine window targets have NO local source
     observation, and requiring one would mis-match the measurement distribution),
     weighted by sim_kj * (floor + w_mut(k)) where w_mut is the arch's OWN learned
     mutation-gate (detached) — the in-distribution mutation detector (0.946 on real
     mutations). Each mined pair becomes a masked-window JEPA example: predict z_obs_j
     from (raw prefix pairs < k, raw cmd_k, raw cmd_j) via net.imaginer, trained with
     an L2-InfoNCE in the eval geometry (per-dim-mean sqL2 logits, tau=0.25, duplicate-
     label masking) + a small MSE anchor for norm calibration. Gradients flow ONLY into
     imaginer params (all inputs are detached frozen embeddings; the w_mut factor is
     computed under no_grad) — the trunk's training trajectory is untouched.

WHY THIS CAPTURES THE SIGNAL WHERE r18/r19 FAILED: the imaginer's inputs are raw frozen
embeddings, so the training and measurement input distributions are IDENTICAL by
construction (the r18 operator's train/compose mismatch cannot occur); the read command
c_r is an input (the r18 operator never saw it); and cross-attention composes distributed
prefix evidence (source-content transport for mv/redir, system-styled priors for mkdir)
instead of editing a single content estimate. Hypothesis-tested end-to-end on TRAIN-image
windows (held-out seqs, exact genuine windows, this exact module + loss on the noisy
embedding-mined pool): +0.166 aggregate over the lexical floor, every family positive,
history-ON minus trained-history-OFF +0.309.

Causal / leak-free: eval forward untouched; future obs_j enters ONLY as a loss label;
future cmd_j is a train-only aux INPUT mirroring the sanctioned endpoint formulation
(the measurement itself supplies c_r as the query) and never touches any scored
prediction. Disabled (champion-term-only) on archs without `imaginer`; fully disabled
(hard 0.0) on archs with neither `imaginer` nor `transition_from_emb`.

Refs: I-JEPA masked latent prediction (arXiv:2301.08243); V-JEPA 2 action-conditioned
predictor on frozen features (arXiv:2506.09985); CPC/InfoNCE (arXiv:1807.03748);
debiased/false-negative-aware contrastive (arXiv:2007.00224); constructive episodic
simulation (Schacter, Addis & Buckner 2007, Nat Rev Neurosci 8:657).
"""

import math

import torch
import torch.nn as nn

NAME = "r20_masked_window_imagination_consistency"
DESCRIPTION = (
    "Champion r18 forward-model-consistency aux VERBATIM + a masked-window imagination "
    "aux: mines same-path endpoint (k -> nearest later touch j) pairs in-batch (weighted "
    "by the arch's own detached mutation gate), and trains the co-designed arch's "
    "`imaginer` (cross-attention over raw plan-time prefix pairs queried by the two "
    "endpoint commands) to predict z_obs_j with an eval-geometry L2-InfoNCE + MSE anchor. "
    "Imagination gradients touch only imaginer params; eval forward untouched; champion-"
    "equivalent on archs without `imaginer`."
)

_DEFAULTS = {
    # ---- champion transition-consistency term (verbatim r18 defaults) ----
    "row_frac": 0.6,       # fraction of batch rows the aux mining runs on
    "path_thresh": 0.60,   # frozen cmd-cosine floor for "same path"
    "change_floor": 0.25,  # min mean-sq change |obs_i - obs_j|^2 to call it a mutation
    "max_examples": 512,   # cap mined triples per step (cost control)
    "cos_weight": 0.10,    # operator -> obs_j cosine reconstruction
    "mse_weight": 0.02,    # small metric anchor
    "aux_weight": 1.0,
    "ramp_steps": 400,     # smoothstep ramp so early training is main-loss-dominated
    # ---- NEW imagination term ----
    "imag_weight": 1.0,        # weight on the imagination aux (imaginer params only)
    "imag_ramp_steps": 400,    # own smoothstep ramp (same schedule family)
    "imag_path_thresh": 0.60,  # same-path floor for endpoint mining
    "imag_max_examples": 256,  # cap mined endpoint pairs per step
    "imag_tau": 0.25,          # L2-InfoNCE temperature (eval-geometry, proven value)
    "imag_dup_delta": 0.05,    # per-dim sq-dist below which two labels are the SAME answer
    "imag_mse": 0.05,          # small MSE anchor (norm calibration; the r19 blowup guard)
    "imag_wfloor": 0.15,       # mining-weight floor added to the detached mutation gate
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


def wrap(net, D, **params):
    """No forward re-point, no module cycle, no new module: the operator and the imaginer
    are the ARCH's own (shared by reference). Returns a config dict; each term is disabled
    independently when the arch lacks its module (passthrough-equivalent if both absent)."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    cfg["_imag_disabled"] = not isinstance(getattr(net, "imaginer", None), nn.Module)
    return cfg


@torch.no_grad()
def _mine_triples(cmd, obs, valid, path_thresh, change_floor):
    """Per row, mine (row, i, k, j) same-path triples with obs changed across k, plus a weight.
    cmd/obs [B,maxn,D] standardized; valid [B,maxn] bool. FULLY VECTORIZED — one batched
    similarity matmul + tensor top/min selection, no per-row/per-position Python loop and no
    Python-scalar float()/.item() (which forced tens of thousands of MPS device->host syncs).
    Numerically equivalent to the original mining (nearest earlier / nearest later same-path
    touch; change>=floor; positive finite weight), modulo tie-breaking. Returns device tensors
    (sel_b, sel_i, sel_k, sel_j, sel_w), each 1-D over the surviving triples."""
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))                         # [B,maxn,maxn] cmd cosine
    vmask = valid.bool()                                           # [B,maxn]

    pos = torch.arange(maxn, device=device)
    lower = pos.unsqueeze(1) > pos.unsqueeze(0)                    # [maxn,maxn] p<k  (rows=k, cols=p)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)                    # [maxn,maxn] p>k
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)          # [B,maxn,maxn], p valid & sim>thr

    before = same_path & lower.unsqueeze(0)                        # [B,maxn,maxn] candidate earlier p
    after = same_path & upper.unsqueeze(0)                         # candidate later p
    # nearest earlier same-path touch i = max p over `before`; nearest later j = min p over `after`.
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)     # [B,maxn], -1 if none
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)    # [B,maxn], maxn if none

    has_i = i_idx >= 0
    has_j = j_idx < maxn
    triple_ok = has_i & has_j & vmask                             # k valid, both touches exist

    ic = i_idx.clamp(0, maxn - 1)
    jc = j_idx.clamp(0, maxn - 1)
    obs_i = torch.gather(obs, 1, ic.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    obs_j = torch.gather(obs, 1, jc.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    change = (obs_i - obs_j).pow(2).mean(dim=-1)                   # [B,maxn]
    sim_ki = torch.gather(sim, 2, ic.unsqueeze(-1)).squeeze(-1)    # [B,maxn] sim[b,k,i]
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)    # [B,maxn] sim[b,k,j]
    w = sim_ki * sim_kj * change                                  # [B,maxn]

    keep = triple_ok & (change >= change_floor) & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)                      # [N,2] -> (row b, position k)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    sel_i = i_idx[sel_b, sel_k]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_i, sel_k, sel_j, sel_w


@torch.no_grad()
def _mine_endpoint_pairs(cmd, obs, valid, net, path_thresh, wfloor):
    """Mine (row, k, j) ENDPOINT pairs: j = the nearest LATER same-path touch of k (frozen
    cmd-cosine > path_thresh). NO earlier-touch requirement — the imagination windows'
    target content is mostly NOT locally observed, and requiring a source observation
    would mis-match that distribution. Weight = sim_kj * (wfloor + w_mut(k)) with w_mut
    the arch's own mutation gate on the reconstructed command feature (detached — the
    same reconstruction `transition_from_emb` uses; falls back to wfloor-only when the
    arch lacks the gate). Fully vectorized; no RNG. Returns (sel_b, sel_k, sel_j, sel_w)."""
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
    sim = torch.bmm(cu, cu.transpose(1, 2))                         # [B,maxn,maxn]
    vmask = valid.bool()

    pos = torch.arange(maxn, device=device)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)                    # [maxn,maxn] p>k
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)
    after = same_path & upper.unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)    # [B,maxn]
    has_j = (j_idx < maxn) & vmask

    w_mut = None
    try:
        cp = getattr(net, "cmd_proj", None)
        inn = getattr(net, "in_norm", None)
        tmg = getattr(net, "tr_mut_gate", None)
        te = getattr(net, "type_emb", None)
        if (isinstance(cp, nn.Module) and isinstance(inn, nn.Module)
                and isinstance(tmg, nn.Module) and isinstance(te, nn.Module)):
            idx0 = torch.zeros(B, maxn, dtype=torch.long, device=device)
            x_cmd = inn(cp(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)) + te(idx0))
            w_mut = torch.sigmoid(tmg(x_cmd)).squeeze(-1)                        # [B,maxn]
    except Exception:
        w_mut = None
    if w_mut is None:
        w_mut = torch.zeros(B, maxn, device=device, dtype=cmd.dtype)

    jc = j_idx.clamp(0, maxn - 1)
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)                  # [B,maxn]
    w = sim_kj * (float(wfloor) + w_mut)

    keep = has_j & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_k, sel_j, sel_w


def _imag_nce(pred, tgt, w, tau, dup_delta, mse_w):
    """Eval-geometry L2-InfoNCE over the mined examples' labels + a small MSE anchor.
    Logits are negative per-dim-mean squared L2 (the retrieval metric's decision
    variable) / tau; near-duplicate labels (per-dim sqdist < dup_delta) are masked out
    of the negatives (false-negative guard — the same answer read twice). Per-example
    weights w are pre-normalized. Anti-collapse: a constant prediction leaves the
    softmax row-uniform over distinct labels (NLL pinned > 0) and the MSE anchor
    strictly positive."""
    n, d = pred.shape
    mse = ((pred - tgt) ** 2).mean(dim=-1)
    if n < 4:
        return (w * mse).sum()
    dist2 = (pred.pow(2).sum(1, keepdim=True) + tgt.pow(2).sum(1) - 2.0 * pred @ tgt.t()) \
        .clamp_min(0.0) / float(d)
    with torch.no_grad():
        tt = (tgt.pow(2).sum(1, keepdim=True) + tgt.pow(2).sum(1) - 2.0 * tgt @ tgt.t()) \
            .clamp_min(0.0) / float(d)
        dup = (tt < float(dup_delta)) & ~torch.eye(n, dtype=torch.bool, device=pred.device)
    logits = (-dist2 / float(tau)).masked_fill(dup, -1e30)
    nll = -torch.log_softmax(logits, dim=1).diagonal()
    return (w * nll).sum() + float(mse_w) * (w * mse).sum()


def _transition_term(cfg, batch, net, device, ramp):
    """The champion r18 forward-model-consistency term, logic verbatim (including its
    randperm row subsample — the ONLY RNG the aux consumes, matching the champion head's
    per-step RNG consumption exactly)."""
    if cfg.get("_disabled", True) or float(cfg.get("aux_weight", 0.0)) <= 0.0:
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
    cmd = tok[sel][:, 0::2][:, :maxn]                              # [nr,maxn,D] standardized z_cmd
    obs = tok[sel][:, 1::2][:, :maxn]                              # [nr,maxn,D] standardized z_obs
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        # keep the strongest (largest weight) triples
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = obs[r, ti].detach()                                     # [N,D] pre-mutation content estimate
    cmd_k = cmd[r, tk].detach()                                   # [N,D] the mutating command
    tgt = obs[r, tj].detach()                                     # [N,D] future post-mutation read (LABEL)
    pre = torch.nan_to_num(pre, nan=0.0, posinf=1e4, neginf=-1e4)
    cmd_k = torch.nan_to_num(cmd_k, nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

    pred = op(pre, cmd_k)                                         # SHARED arch operator (grad flows in)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    total = float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err
    return float(cfg["aux_weight"]) * ramp * total


def _imagination_term(cfg, batch, net, device, ramp):
    """The NEW masked-window imagination term. Trains ONLY net.imaginer parameters:
    every input is a detached frozen embedding; the mutation-gate mining factor is
    computed under no_grad. The trunk's gradient stream is untouched."""
    if cfg.get("_imag_disabled", True) or float(cfg.get("imag_weight", 0.0)) <= 0.0:
        return 0.0
    imaginer = getattr(net, "imaginer", None)
    if not isinstance(imaginer, nn.Module):
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 2:
        return 0.0

    cmd = tok[:, 0::2][:, :maxn].detach()                          # [B,maxn,D]
    obs = tok[:, 1::2][:, :maxn].detach()                          # [B,maxn,D]
    valid = cmd_mask

    sb, sk, sj, w = _mine_endpoint_pairs(
        cmd, obs, valid, net, float(cfg["imag_path_thresh"]), float(cfg["imag_wfloor"])
    )
    if sb.numel() < 2:
        return 0.0
    if sb.numel() > int(cfg["imag_max_examples"]):
        w, order = torch.topk(w, int(cfg["imag_max_examples"]))
        sb = sb[order]; sk = sk[order]; sj = sj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pair_cat = torch.cat([cmd, obs], dim=-1)[sb]                   # [N,maxn,2D] raw frozen prefix
    pos = torch.arange(maxn, device=device)
    pmask = valid[sb] & (pos.unsqueeze(0) < sk.unsqueeze(1))       # strictly-earlier valid pairs
    c_m = cmd[sb, sk]
    c_r = cmd[sb, sj]
    lab = torch.nan_to_num(obs[sb, sj], nan=0.0, posinf=1e4, neginf=-1e4)

    pred = imaginer(pair_cat, pmask, c_m, c_r)                     # grads -> imaginer params ONLY
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    total = _imag_nce(pred, lab, w, cfg["imag_tau"], cfg["imag_dup_delta"], cfg["imag_mse"])
    return float(cfg["imag_weight"]) * ramp * total


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if cfg.get("_disabled", True) and cfg.get("_imag_disabled", True):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    ramp_imag = _smoothstep(cfg["_step"] / max(1.0, float(cfg["imag_ramp_steps"])))

    total = 0.0
    if ramp > 0.0:
        total = total + _transition_term(cfg, batch, net, device, ramp)
    if ramp_imag > 0.0:
        total = total + _imagination_term(cfg, batch, net, device, ramp_imag)

    if torch.is_tensor(total):
        if not bool(torch.isfinite(total).item()):
            return 0.0
    return total


def leak_safe(mod, params):
    """Forward untouched (wrap adds no module, never re-points forward). The champion
    term consumes future obs_j strictly as a loss LABEL; the imagination term consumes
    future obs_j strictly as a loss LABEL and the future COMMAND c_j only as a train-
    only aux input to an eval-inactive module (the sanctioned endpoint formulation) —
    no scored prediction sees anything but its own history. Validate params finite and
    in range."""
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
        -1.0 <= vals["imag_path_thresh"] < 1.0,
        vals["imag_max_examples"] >= 2.0,
        vals["imag_tau"] > 0.0,
        vals["imag_dup_delta"] >= 0.0,
        vals["imag_mse"] >= 0.0,
        vals["imag_wfloor"] >= 0.0,
    ]
    return all(checks)
