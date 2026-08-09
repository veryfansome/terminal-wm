"""R18 head: FORWARD-MODEL CONSISTENCY on the arch's per-path latent-transition operator.

Co-designed with `r18_pathstate_latent_transition_worldmodel`. That arch predicts a mutated
path's CURRENT content by writing, at the mutating command k, a learned latent transition
value f(s_pre_k, cmd_k) into the path slot, which a LATER read j>k retrieves. The main MSE
loss only supervises each command's OWN next observation; it never tells the operator that
the value it writes at k must equal what the path will READ at j. This train-only aux adds
that pressure directly on the SHARED operator.

MECHANISM (train-only; eval forward BIT-IDENTICAL to the wrapped arch — wrap adds no module,
never re-points forward): per trajectory, mine same-path TRIPLES (i < k < j) where i is the
nearest earlier same-path touch of k, j the nearest later one (path identity by frozen
command-embedding cosine — `cat P`, `echo hi>P`, `cat P` sit close in the shared path
subspace), and the observation CHANGED across k (‖obs_i − obs_j‖ large — a real mutation).
Then require the arch's own operator to be a good forward model:
    f( pre = obs_i , cmd = cmd_k )  ≈  obs_j        (cosine + MSE, change-weighted)
where obs_i is the pre-mutation content estimate, cmd_k the mutating command, obs_j the future
post-mutation read. Gradients flow into the shared tr_in/tr_out and the trunk input
projection, so the operator used at INFERENCE becomes a genuine (pre-content, command) →
post-content dynamics — exactly the mutated-cell content a symbolic tracker cannot compute.

DISTINCT from prior heads: this is NOT a contrastive retrieval-RANK loss vs a twin
(#4 twinmargin) and NOT a probe on the trunk hidden state representing the delta (r17
mutation/stc probes). It is a REGRESSION forward-model consistency that trains the actual
transition OPERATOR the arch runs — a supervised latent-dynamics term (Dreamer/RSSM
self-consistency), the head half of the co-designed stack.

Composability: disabled (hard 0.0, passthrough-equivalent) on any arch that does not expose
`transition_from_emb` (i.e. every arch but the co-designed one). Causal/leak-free: future
obs_j enters ONLY as a loss label; the mined pre/cmd are strictly earlier; forward is
untouched.
"""

import math

import torch
import torch.nn as nn

NAME = "r18_transition_forwardmodel_consistency"
DESCRIPTION = (
    "Train-only forward-model consistency on the r18 latent-transition arch's shared operator: "
    "mines same-path (pre, mutating-cmd, future-read) triples and requires f(obs_pre, cmd) to "
    "reconstruct the future post-mutation observation (cosine+MSE, change-weighted). Eval forward "
    "untouched; disabled (0.0) on archs without the transition operator. Co-designed head half of "
    "the r18 transition world-model stack."
)

_DEFAULTS = {
    "row_frac": 0.6,       # fraction of batch rows the aux mining runs on
    "path_thresh": 0.60,   # frozen cmd-cosine floor for "same path"
    "change_floor": 0.25,  # min mean-sq change ‖obs_i−obs_j‖² to call it a mutation
    "max_examples": 512,   # cap mined triples per step (cost control)
    "cos_weight": 0.10,    # operator -> obs_j cosine reconstruction
    "mse_weight": 0.02,    # small metric anchor
    "aux_weight": 1.0,
    "ramp_steps": 400,     # smoothstep ramp so early training is main-loss-dominated
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
    """No forward re-point, no module cycle, no new module: the operator is the ARCH's own
    `transition_from_emb`, shared by reference. Returns a config dict; disabled if the arch
    does not expose the operator (passthrough-equivalent)."""
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
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


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
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

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    """Forward untouched (wrap adds no module, never re-points forward); the aux consumes future
    obs_j strictly as a loss LABEL and mines pre/cmd from strictly-earlier positions. Validate
    params are finite and in range."""
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
    ]
    return all(checks)
