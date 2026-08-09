"""R20 head: DUAL-PRE forward-model consistency — the r18 aux VERBATIM plus a second
supervision arm that trains the SAME shared transition operator on its DEPLOYMENT input
distribution: the arch's own memory content s_pre.

THE SEAM THIS CLOSES (prior findings 6+7). The r18 head supervises
transition_from_emb(pre, cmd) -> future read with pre = RAW past observations (obs_i) only. At
composition time — both the (a)-path companion and the (b)-path imagination write — the
operator is applied to the arch's MEMORY content s_pre, a distribution it never trained on:
measured collapse endhist-pre 0.404 >> mem-pre 0.148 (finding 6), r18 re-baseline mem-pre
margin -0.342 dedup. Finding 7's de-risk showed the memory content is USABLE — a fresh operator
trained on (s_pre_m, c_m) -> z_r recovers 0.510 (affine, = floor: the r18 operator FORM has no
capacity to use content) / 0.569 (MLP, +0.057). This head runs exactly that experiment INSIDE
the sanctioned single pass, on the r18 head's own mined triples:

  arm 1 (r18 aux, verbatim): f(obs_i, cmd_k) ~ obs_j           weight 1.0  (cos 0.10 + mse 0.02)
  arm 2 (NEW, mem-pre):       f(s_pre_k, cmd_k) ~ obs_j        weight mem_arm_w = 0.5x arm 1
    s_pre_k = the arch's OWN transition-memory read at k, computed by the arch's input block +
    `_transition_reads` under no_grad on the SAME selected rows (no transformer trunk — the
    cheap path the (a)-path instrument `imag_direct._memory_spre` validated), DETACHED — it is
    an input like obs_i, so gradient flows only through the operator call (tr_in/tr_out [+ the
    co-designed arch's content residual] + cmd_proj/type_emb/in_norm), the exact same parameter
    surface as arm 1.

ONE mining pass serves both arms (same triples, same weights): the per-step RNG draw count is
IDENTICAL to the r18 head (one randperm), so with mem_arm_w=0 training is bit-equal to the
r18 head under the same seed (verified), and with mem_arm_w>0 the ONLY training-trajectory
change is the mem-arm gradient itself — clean attribution. CO-DESIGN: on the r18 arch this
arm can only re-linearize (finding 7: affine-on-memory saturates at floor); it is paired with
`r20_contentcond_transition_imagwrite`, whose rms-normalized content residual is the capacity
that can act on it. HONEST MINI RESULT (2x2 quadrant probe, 2 seeds, 700-step d=128 minis):
at that budget the arm is outcome-neutral — main task unchanged (deltas <= 0.0013 in all
quadrants), operator mem-pre composition unmoved, small consistent raw-pre improvement with
the capacity arch (+0.01..+0.03 full) — while the residual IS recruited (||res||/||base||
0.42); the repo's documented proxy-inversion doctrine for memory-mechanism training (700-step
minis under-train them; evolve/CLAUDE.md) is why the full-budget 3-seed measurement — the
scale where finding 7's +0.057 was measured — is the one that would settle it. Composability:
the mem arm silently disables (raw-obs-arm-only, still verbatim) on any arch lacking the
input-block/_transition_reads surface; the whole head disables (hard 0.0) without
`transition_from_emb`, as the r18 head does.

Causal/leak-free: identical to the raw-obs arm — obs_j enters ONLY as a loss label; s_pre_k is
computed from pairs strictly before k on fully-observed training rows; forward is untouched
(wrap adds no module, never re-points forward; eval bit-identical to the wrapped arch).

Refs: the r18 head (this file's arm 1, verbatim); learned-simulator distribution-shift
/ train-on-own-state corrections (DAgger arXiv:1011.0686; scheduled sampling arXiv:1506.03099 —
here applied to the OPERATOR's input distribution, not the trunk's); Dreamer/RSSM latent
self-consistency (arXiv:1912.01603).
"""

import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CH

NAME = "r20_dualpre_transition_consistency"
DESCRIPTION = (
    "The r18 forward-model-consistency aux VERBATIM (raw-obs pre arm, weight 1.0) plus a "
    "mem-pre arm supervising the SAME shared transition operator on the arch's own memory "
    "content s_pre_k (computed via the arch's input block + _transition_reads under no_grad, "
    "detached) toward the same mined future reads — training the operator on its deployment "
    "distribution (brief findings 6+7) inside the single pass. One mining pass, RNG-draw count "
    "identical to the r18 head; mem arm auto-disables on archs without the memory surface."
)

_DEFAULTS = dict(CH._DEFAULTS)
_DEFAULTS.update({
    "mem_arm_w": 0.5,      # mem-pre arm weight relative to the raw-obs arm's cos/mse weights
})

_MEM_ATTRS = ("cmd_proj", "obs_proj", "type_emb", "in_norm", "_positional", "_transition_reads")


def wrap(net, D, **params):
    """Delegate to the r18 wrap (no module, no forward re-point), then add the mem-arm
    config + capability check."""
    cfg = CH.wrap(net, D, **{k: v for k, v in params.items() if k in CH._DEFAULTS})
    cfg["mem_arm_w"] = float(params.get("mem_arm_w", _DEFAULTS["mem_arm_w"]))
    mem_ok = all(hasattr(net, a) for a in _MEM_ATTRS) and hasattr(net, "pos_scale")
    cfg["_mem_disabled"] = bool(cfg.get("_disabled", True)) or not mem_ok
    return cfg


@torch.no_grad()
def _memory_pre(net, rows_tok, rows_types, valid, device):
    """s_pre at every command index for the selected rows, via the arch's OWN input block +
    _transition_reads (no transformer trunk). rows_tok [nr, L, D]; valid [nr, maxn]."""
    L = rows_tok.shape[1]
    t = rows_types.long().clamp(0, 1)
    x = torch.where((t == 0).unsqueeze(-1), net.cmd_proj(rows_tok), net.obs_proj(rows_tok))
    x = x + net.type_emb(t) + net.pos_scale * net._positional(L, device, x.dtype).unsqueeze(0)
    x = net.in_norm(x)
    maxn = valid.shape[1]
    x_cmd = x[:, 0::2][:, :maxn]
    obs = rows_tok[:, 1::2][:, :maxn]
    return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not CH._interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = CH._smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]      # the ONE RNG draw (same as the r18 head)
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = CH._mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(CH._EPS)).detach()

    cmd_k = torch.nan_to_num(cmd[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(obs[r, tj].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    gu = CH._unit(tgt)

    def arm(pre):
        pred = torch.nan_to_num(op(pre, cmd_k), nan=0.0, posinf=1e4, neginf=-1e4)
        pu = CH._unit(pred)
        cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
        return float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err

    # -- arm 1: the r18 aux, verbatim (raw-obs pre) --
    pre_raw = torch.nan_to_num(obs[r, ti].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    total = arm(pre_raw)

    # -- arm 2: mem-pre (deployment distribution), on the SAME triples --
    mem_w = float(cfg.get("mem_arm_w", 0.0))
    if mem_w > 0.0 and not cfg.get("_mem_disabled", True):
        reads = _memory_pre(net, tok[sel], batch["types"][sel], valid, device)
        pre_mem = torch.nan_to_num(reads[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
        total = total + mem_w * arm(pre_mem)

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    """r18 checks + mem_arm_w range. Forward untouched; future obs_j is a loss label only;
    s_pre_k uses pairs strictly before k."""
    p = dict(params or {})
    mw = p.pop("mem_arm_w", _DEFAULTS["mem_arm_w"])
    try:
        mw = float(mw)
    except Exception:
        return False
    if not math.isfinite(mw) or mw < 0.0 or mw > 10.0:
        return False
    return CH.leak_safe(mod, p)
