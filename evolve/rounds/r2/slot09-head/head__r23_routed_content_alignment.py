import math

import torch

NAME = "r23_routed_content_alignment"
DESCRIPTION = (
    "Train-only supervision of an arch's prefix-content cross-attention. Mines content-transport "
    "events from the token stream alone — an observation numerically equal to a strictly earlier "
    "observation whose command text differs — and applies a multi-positive log-loss to the arch's "
    "own attention row at the later read, moving its mass onto the earlier occurrence(s) of that "
    "same content. Prefix reads whose commands are nearest the queried command but whose content "
    "differs are rank-selected as hard negatives and importance-boosted in the denominator; "
    "same-content positions are never negatives. Degenerate observations are removed by a "
    "duplicate-class-size cap and by suppressing the batch's most frequent observation class. "
    "Hidden states are read through a forward hook on the net, so no forward is re-pointed and no "
    "module is re-registered; the head adds no parameters and leaves the eval forward untouched. "
    "Disabled on archs without the prefix-content attention surface."
)

_DEFAULTS = {
    "align_w": 0.3,
    "kappa": 4.0,
    "n_stale": 2,
    "dup_eps": 1e-4,
    "cmd_eps": 1e-4,
    "max_class": 6,
    "modal_frac": 0.05,
    "modal_probes": 64,
    "max_examples": 512,
    "min_valid": 8,
    "ramp_steps": 300,
}

_EPS = 1e-8
_ATTN_ATTRS = ("xq", "xk", "x_out")


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _pair_sqdist(a):
    s = (a * a).sum(dim=-1)
    g = torch.bmm(a, a.transpose(1, 2))
    d = s.unsqueeze(2) + s.unsqueeze(1) - 2.0 * g
    return d.clamp_min(0.0) / float(a.size(-1))


def _layout_ok(b):
    for k in ("tok", "types", "key_pad", "tgt", "cmd_mask"):
        if k not in b:
            return False
    tok, types, key_pad, tgt, cmd_mask = b["tok"], b["types"], b["key_pad"], b["tgt"], b["cmd_mask"]
    if tok.dim() != 3 or tgt.dim() != 3 or types.dim() != 2 or key_pad.dim() != 2:
        return False
    if tok.shape[:2] != types.shape or tok.shape[:2] != key_pad.shape:
        return False
    if cmd_mask.shape != tgt.shape[:2] or tok.shape[0] != tgt.shape[0]:
        return False
    if tok.shape[1] != 2 * cmd_mask.shape[1]:
        return False
    return True


def _attn_surface_ok(net):
    for a in _ATTN_ATTRS:
        if not callable(getattr(net, a, None)):
            return False
    try:
        return int(getattr(net, "xattn_dim")) > 0
    except Exception:
        return False


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    cfg["D"] = int(D)
    cfg["_step"] = 0
    state = {"h": None, "tok": None}
    cfg["_state"] = state
    ok = _attn_surface_ok(net)
    cfg["_disabled"] = not ok
    if ok:

        def _capture(module, inputs, output):
            if not torch.is_grad_enabled():
                return
            if isinstance(output, (tuple, list)) and len(output) == 2:
                state["tok"] = inputs[0] if len(inputs) else None
                state["h"] = output[1]

        net.register_forward_hook(_capture)
    return cfg


@torch.no_grad()
def _modal_mask(obs, valid, dup_eps, modal_frac, probes):
    B, n, dim = obs.shape
    out = torch.zeros(B * n, dtype=torch.bool, device=obs.device)
    flat = obs.reshape(B * n, dim)
    idx = torch.nonzero(valid.reshape(-1), as_tuple=False).squeeze(1)
    m = int(idx.numel())
    if m < 8:
        return out.view(B, n)
    s = min(int(probes), m)
    pick = torch.linspace(0, m - 1, s, device=obs.device).round().long().clamp(0, m - 1)
    a = flat[idx]
    ref = a[pick]
    d2 = (a * a).sum(-1, keepdim=True) + (ref * ref).sum(-1).unsqueeze(0) - 2.0 * (a @ ref.t())
    hit = (d2.clamp_min(0.0) / float(dim)) <= dup_eps
    cnt = hit.sum(dim=0)
    best = int(torch.argmax(cnt).item())
    if int(cnt[best].item()) >= max(2, int(modal_frac * m)):
        out[idx] = hit[:, best]
    return out.view(B, n)


@torch.no_grad()
def _mine(obs, cmd, valid, cfg):
    dup_eps = float(cfg["dup_eps"])
    cmd_eps = float(cfg["cmd_eps"])
    n = valid.shape[1]

    same_content = (_pair_sqdist(obs) <= dup_eps) & valid.unsqueeze(1) & valid.unsqueeze(2)
    same_cmd = _pair_sqdist(cmd) <= cmd_eps

    class_size = same_content.sum(dim=2)
    modal = _modal_mask(obs, valid, dup_eps, float(cfg["modal_frac"]), int(cfg["modal_probes"]))
    usable = valid & (class_size <= int(cfg["max_class"])) & (~modal)

    order = torch.arange(n, device=valid.device)
    earlier = (order.unsqueeze(1) > order.unsqueeze(0)).unsqueeze(0)
    allowed = earlier & valid.unsqueeze(1)

    positives = (allowed & same_content & (~same_cmd)
                 & usable.unsqueeze(1) & usable.unsqueeze(2))
    eligible = positives.any(dim=2) & usable

    negatives = allowed & (~same_content)

    cu = _unit(cmd)
    csim = torch.bmm(cu, cu.transpose(1, 2)).masked_fill(~negatives, -2.0)
    k = max(1, min(int(cfg["n_stale"]), n))
    top_v, top_i = csim.topk(k, dim=2)
    ring = torch.zeros_like(negatives)
    ring.scatter_(2, top_i, top_v > -1.5)
    ring = ring & negatives

    den_mask = positives | negatives
    return positives, den_mask, ring, eligible


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    weight = float(cfg.get("align_w", 0.0))
    if weight <= 0.0:
        return 0.0
    state = cfg.get("_state")
    if state is None:
        return 0.0
    h = state.get("h")
    seen_tok = state.get("tok")
    state["h"] = None
    state["tok"] = None
    if h is None or not _layout_ok(batch):
        return 0.0
    tok = batch["tok"]
    if seen_tok is not tok:
        return 0.0

    cmd_mask = batch["cmd_mask"].bool()
    key_pad = batch["key_pad"].bool()
    B, n = cmd_mask.shape
    if n < 2 or h.dim() != 3 or h.shape[0] != B or h.shape[1] != 2 * n:
        return 0.0
    valid = cmd_mask & (~key_pad[:, 0::2]) & (~key_pad[:, 1::2])
    if int(valid.sum().item()) < int(cfg["min_valid"]):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    obs = tok[:, 1::2, :].detach().float()
    cmd = tok[:, 0::2, :].detach().float()
    obs = torch.nan_to_num(obs, nan=0.0, posinf=1e4, neginf=-1e4)
    cmd = torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)

    positives, den_mask, ring, eligible = _mine(obs, cmd, valid, cfg)
    rows = torch.nonzero(eligible, as_tuple=False)
    e = int(rows.shape[0])
    if e == 0:
        return 0.0
    cap = int(cfg["max_examples"])
    if e > cap:
        keep = torch.linspace(0, e - 1, cap, device=rows.device).round().long().clamp(0, e - 1)
        rows = rows[keep]
    bi = rows[:, 0]
    ji = rows[:, 1]

    h_cmd = h[:, 0::2, :]
    h_obs = h[:, 1::2, :]
    q = net.xq(h_cmd[bi, ji])
    k = net.xk(h_obs[bi])
    scale = 1.0 / math.sqrt(float(int(getattr(net, "xattn_dim"))))
    scores = (q.unsqueeze(1) * k).sum(dim=-1) * scale
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4).float()

    pos_r = positives[bi, ji]
    den_r = den_mask[bi, ji]
    ring_r = ring[bi, ji]

    boost = ring_r.to(scores.dtype) * math.log(1.0 + max(0.0, float(cfg["kappa"])))
    ninf = torch.full_like(scores, float("-inf"))
    den = torch.where(den_r, scores + boost, ninf)
    num = torch.where(pos_r, scores, ninf)
    nll = (torch.logsumexp(den, dim=1) - torch.logsumexp(num, dim=1)).clamp_min(0.0)

    out = weight * ramp * nll.mean()
    if not bool(torch.isfinite(out).item()):
        return 0.0
    return out


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        0.0 <= vals["align_w"] <= 10.0,
        0.0 <= vals["kappa"] <= 100.0,
        vals["n_stale"] >= 1.0,
        0.0 < vals["dup_eps"] <= 1.0,
        0.0 < vals["cmd_eps"] <= 1.0,
        vals["max_class"] >= 1.0,
        0.0 < vals["modal_frac"] <= 1.0,
        vals["modal_probes"] >= 1.0,
        vals["max_examples"] >= 1.0,
        vals["min_valid"] >= 1.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
