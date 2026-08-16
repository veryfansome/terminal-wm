import math

import torch
import torch.nn.functional as F

NAME = "r7_move_ablation_counterfactual_contrast"
DESCRIPTION = (
    "A parameter-free head: the wrapped forward is the unwrapped forward bit-for-bit (no extra "
    "readout, no injected memory, no re-pointed forward), and the whole mechanism is a train-time "
    "INTERVENTIONAL auxiliary that runs the arch on two edited copies of the same rows and scores "
    "the paired difference between them. Label-free mining on the batch alone finds a read step j "
    "whose observation duplicates an earlier contentful observation under a different command, "
    "with at least one content-free step between them; content-free steps are identified as the "
    "one large duplicate cluster every observation-space batch carries, because a move or a "
    "redirect prints nothing and so all such steps share one observation embedding. The BROKEN arm "
    "masks out every content-free step between the content's first exposure and the read, which "
    "removes the only mechanism that could have delivered that content to that location; the "
    "CONTROL arm masks the SAME NUMBER of steps chosen from steps that cannot change what sits "
    "there — earlier contentful reads that are not that content, and steps after the read. The "
    "loss makes the control arm name the content and requires the broken arm's log-probability of "
    "the same content to sit a margin BELOW the control's, over one shared candidate set of the "
    "row's observations. Because both arms carry the same number of masked steps and the same read "
    "command at the same position, a predictor that keys on the location being read, or on how "
    "much of the stream is masked, gets exactly zero margin and is penalised; only sensitivity to "
    "WHICH moves happened can satisfy it. No parameters are introduced, so every gradient the aux "
    "produces lands on the architecture's own routing and readout."
)

_DEFAULTS = {
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "min_first": 1,
    "max_span": 16,
    "max_ablate": 10,
    "max_rows": 16,
    "cand_temp": 0.5,
    "margin": 1.5,
    "pos_weight": 0.5,
    "diff_weight": 1.0,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


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
    if tok.shape[1] != 2 * tgt.shape[1] or tok.shape[1] % 2 != 0:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


@torch.no_grad()
def _mine(tok, cmd_mask, cfg):
    B, L, Dm = tok.shape
    n = min(L // 2, cmd_mask.shape[1])
    if n < 4:
        return None
    dev = tok.device
    dm = float(Dm)

    c = torch.nan_to_num(tok[:, 0:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    o = torch.nan_to_num(tok[:, 1:2 * n:2, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    valid = cmd_mask[:, :n].bool()

    osq = o.pow(2).sum(-1)
    csq = c.pow(2).sum(-1)
    d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
           - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / dm
    d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
           - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / dm

    eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
    vv = valid.unsqueeze(2) & valid.unsqueeze(1)
    off = vv & (~eye)
    cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
    ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
    ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

    dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
    dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

    max_dup = int(cfg["max_dup"])
    clus = dup_o.sum(dim=2)
    contentful = valid & (clus <= max_dup)
    contentless = valid & (clus > max_dup)

    idx = torch.arange(n, device=dev)
    before = (idx.view(1, n, 1) > idx.view(1, 1, n))

    pair = dup_o & (~dup_c) & contentful.unsqueeze(2) & contentful.unsqueeze(1)
    earlier = pair & before
    has_p = earlier.any(dim=2)
    big = torch.full((1, 1, n), n, device=dev, dtype=torch.long)
    first_i = torch.where(earlier, idx.view(1, 1, n), big).amin(dim=2)

    trivial = (dup_c & dup_o & before).any(dim=2)

    brk = contentless.unsqueeze(1) & before & (idx.view(1, 1, n) > first_i.unsqueeze(2))
    n_brk = brk.sum(dim=2)

    span = (idx.view(1, n) - first_i).clamp_min(0)
    eligible = (valid & contentful & has_p & (~trivial)
                & (n_brk >= 1)
                & (n_brk <= int(cfg["max_ablate"]))
                & (first_i >= int(cfg["min_first"]))
                & (span <= int(cfg["max_span"])))
    score = torch.where(eligible, n_brk.float() + 0.01 * span.float(),
                        torch.full_like(n_brk, -1, dtype=torch.float32))

    best_s, best_j = score.max(dim=1)
    rows = torch.nonzero(best_s >= 0.5, as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return None
    cap = max(1, int(cfg["max_rows"]))
    if rows.numel() > cap:
        _, order = torch.topk(best_s[rows], cap)
        rows = rows[order]

    b_idx = rows
    j_idx = best_j[b_idx]
    E = int(b_idx.numel())

    dup_row = dup_o[b_idx, j_idx]
    contentful_e = contentful[b_idx]
    valid_e = valid[b_idx]
    is_j = F.one_hot(j_idx, n).bool()
    jv = j_idx.view(E, 1)
    kk = idx.view(1, n)

    brk_e = brk[b_idx, j_idx]
    need = brk_e.sum(dim=1)

    pool = (valid_e & (~dup_row) & (~is_j)
            & ((contentful_e & (kk < jv)) | (kk > jv)))
    rank = pool.long().cumsum(dim=1) - 1
    ctrl_e = pool & (rank >= 0) & (rank < need.view(E, 1))

    cand_mask = contentful_e & (kk <= jv)
    pos_mask = cand_mask & (dup_row | is_j)

    return {"n": n, "E": E, "b_idx": b_idx, "j_idx": j_idx, "brk": brk_e, "ctrl": ctrl_e,
            "cand_mask": cand_mask, "pos_mask": pos_mask, "obs": o[b_idx], "obs_sq": osq[b_idx]}


def _arm(tok_e, kp_e, step_mask, n):
    E, L, _ = tok_e.shape
    pad = torch.zeros(E, L, dtype=torch.bool, device=tok_e.device)
    pad[:, 0:2 * n:2] = step_mask
    pad[:, 1:2 * n:2] = step_mask
    tk = tok_e.masked_fill(pad.unsqueeze(-1), 0.0)
    return tk, (kp_e.bool() | pad)


def _log_pos(p, cand, cand_sq, cand_mask, pos_mask, dm, temp):
    dot = torch.bmm(cand, p.unsqueeze(2)).squeeze(2)
    d2 = (p.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / dm
    logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
    logp = torch.log_softmax(logits, dim=1)
    return torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    return {"cfg": cfg, "step": 0, "layout": None, "D": int(D)}


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    cfg = st["cfg"]
    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0

    layout = st.get("layout")
    if layout is None:
        layout = bool(_layout_ok(batch))
        st["layout"] = layout
    if not layout:
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    mined = _mine(tok, batch["cmd_mask"], cfg)
    if mined is None:
        return 0.0

    n = mined["n"]
    E = mined["E"]
    b_idx = mined["b_idx"]
    j_idx = mined["j_idx"]

    tok_e = tok[b_idx]
    types_e = batch["types"][b_idx]
    kp_e = batch["key_pad"][b_idx]

    tok_c, kp_c = _arm(tok_e, kp_e, mined["ctrl"], n)
    tok_b, kp_b = _arm(tok_e, kp_e, mined["brk"], n)

    pred, _ = net(torch.cat([tok_c, tok_b], dim=0),
                  torch.cat([types_e, types_e], dim=0),
                  torch.cat([kp_c, kp_b], dim=0))
    if pred.dim() != 3 or pred.size(-1) != st["D"] or pred.size(0) != 2 * E:
        return 0.0

    cmd_pred = pred[:, 0::2, :][:, :n, :]
    Dm = cmd_pred.size(-1)
    gidx = j_idx.view(E, 1, 1).expand(E, 1, Dm)
    p_ctrl = torch.nan_to_num(cmd_pred[:E].gather(1, gidx).squeeze(1),
                              nan=0.0, posinf=1e4, neginf=-1e4).float()
    p_brk = torch.nan_to_num(cmd_pred[E:].gather(1, gidx).squeeze(1),
                             nan=0.0, posinf=1e4, neginf=-1e4).float()

    cand = mined["obs"]
    cand_sq = mined["obs_sq"]
    cand_mask = mined["cand_mask"]
    pos_mask = mined["pos_mask"]
    dm = float(cand.size(-1))
    temp = max(1e-3, float(cfg["cand_temp"]))

    s_ctrl = _log_pos(p_ctrl, cand, cand_sq, cand_mask, pos_mask, dm, temp)
    s_brk = _log_pos(p_brk, cand, cand_sq, cand_mask, pos_mask, dm, temp)

    l_pos = -s_ctrl.mean()
    l_diff = F.softplus(s_brk - s_ctrl + float(cfg["margin"])).mean()
    total = float(cfg["pos_weight"]) * l_pos + float(cfg["diff_weight"]) * l_diff

    out_loss = ramp * float(cfg["aux_weight"]) * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss.to(tok.dtype)


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
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["min_first"] >= 0.0,
        vals["max_span"] >= 2.0,
        vals["max_ablate"] >= 1.0,
        vals["max_rows"] >= 1.0,
        vals["cand_temp"] > 0.0,
        vals["margin"] >= 0.0,
        vals["pos_weight"] >= 0.0,
        vals["diff_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
