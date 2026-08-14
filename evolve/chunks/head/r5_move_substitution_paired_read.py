import math

import torch
import torch.nn.functional as F

NAME = "r5_move_substitution_paired_read"
DESCRIPTION = (
    "Train-only, parameter-free head that builds a within-batch counterfactual of the move chain "
    "and trains the arch's own command-position prediction on the PAIRED difference between the "
    "two arms. Label-free mining from embeddings alone: observations that are near-exact "
    "duplicates of many others in the row mark content-free (mutating) steps; a read step is a "
    "step whose observation duplicates an earlier step's observation under a non-duplicate "
    "command with at least one mutating step in between. For the deepest such (expose i, read j) "
    "pair per row, a counterfactual copy of the row is built by REPLACING the command embedding "
    "of every mutating step strictly between i and j with the command embedding of another "
    "mutating step of the same row taken from outside that interval (falling back to another "
    "selected row's mutating command). Sequence length, pad pattern, observation tokens and every "
    "non-mutating command are untouched, so the two arms differ only in which moves were issued. "
    "Both arms are run through the unmodified net in one concatenated forward with dropout "
    "disabled, and each prediction is scored as a log-probability over the row's contentful "
    "observations by negative mean-squared distance. The loss is cross-entropy on the native arm "
    "plus a hinge requiring the native arm's log-probability of the read content to exceed the "
    "counterfactual arm's by a margin, plus a small mean-squared anchor, ramped in by a "
    "smoothstep. wrap registers nothing and never re-points forward; eval forward is bit-"
    "identical to the arch's own."
)

_DEFAULTS = {
    "row_frac": 0.5,
    "max_rows": 24,
    "dup_frac": 0.02,
    "cmd_dup_frac": 0.02,
    "max_dup": 2,
    "mut_min": 3,
    "cand_temp": 0.5,
    "margin": 1.0,
    "nat_weight": 1.0,
    "ca_weight": 1.0,
    "mse_weight": 0.02,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


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
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    return cfg


def _pick_from_set(member, count, n, device):
    big = torch.full((member.size(0), n), n, dtype=torch.long, device=device)
    idxn = torch.arange(n, device=device).view(1, n).expand(member.size(0), n)
    tagged = torch.where(member, idxn, big)
    ordered, _ = torch.sort(tagged, dim=1)
    u = torch.rand(member.size(0), n, device=device)
    slot = (u * count.clamp_min(1).view(-1, 1).to(u.dtype)).long().clamp(0, n - 1)
    return torch.gather(ordered, 1, slot).clamp(0, n - 1)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None:
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    if not _layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    types = batch["types"]
    key_pad = batch["key_pad"]
    valid = batch["cmd_mask"].bool()
    B, n = valid.shape
    if n < 4:
        return 0.0
    dev = tok.device

    max_dup = int(cfg["max_dup"])
    mut_min = max(max_dup + 1, int(cfg["mut_min"]))

    with torch.no_grad():
        cmd_all = _clean(tok[:, 0::2, :][:, :n, :].float())
        obs_all = _clean(tok[:, 1::2, :][:, :n, :].float())
        Dm = float(obs_all.size(-1))

        osq = obs_all.pow(2).sum(-1)
        csq = cmd_all.pow(2).sum(-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(obs_all, obs_all.transpose(1, 2))).clamp_min(0.0) / Dm
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(cmd_all, cmd_all.transpose(1, 2))).clamp_min(0.0) / Dm

        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        clus = dup_o.sum(dim=2)
        contentful = valid & (clus <= max_dup)
        mutation = valid & (clus >= mut_min)

        idxn = torch.arange(n, device=dev)
        tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
        earlier = dup_o & (~dup_c) & contentful.unsqueeze(1) & contentful.unsqueeze(2) & tri

        neg1 = torch.full((1, 1, n), -1, dtype=torch.long, device=dev)
        last_i = torch.where(earlier, idxn.view(1, 1, n), neg1).amax(dim=2)
        has_p = earlier.any(dim=2)

        mcs = torch.cumsum(mutation.long(), dim=1)
        jm1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        mc_j = torch.gather(mcs, 1, jm1)
        mc_i = torch.gather(mcs, 1, last_i.clamp_min(0))
        n_between = mc_j - mc_i

        mined = has_p & contentful & valid & (last_i >= 0) & (n_between > 0)
        depth = (idxn.view(1, n) - last_i).clamp_min(0)
        score = torch.where(mined, depth, torch.full_like(depth, -1))
        best = score.argmax(dim=1)
        best_depth = torch.gather(score, 1, best.view(-1, 1)).squeeze(1)
        row_ok = best_depth >= 0

        rows = torch.nonzero(row_ok, as_tuple=False).squeeze(1)
        if rows.numel() == 0:
            return 0.0
        budget = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
        budget = min(budget, int(cfg["max_rows"]))
        if rows.numel() > budget:
            _, order = torch.topk(best_depth[rows], budget)
            rows = rows[order]
        R = int(rows.numel())

        j_sel = best[rows]
        i_sel = last_i[rows, j_sel]

        span = idxn.view(1, n).expand(R, n)
        onpath = mutation[rows] & (span > i_sel.view(-1, 1)) & (span < j_sel.view(-1, 1))
        offpath = mutation[rows] & (~onpath)
        cnt_off = offpath.sum(dim=1)
        cnt_on = onpath.sum(dim=1)
        if int(cnt_on.min().item()) < 1:
            return 0.0
        use_alt = cnt_off < 1
        if R < 2 and bool(use_alt.any().item()):
            return 0.0

        donor_off = _pick_from_set(offpath, cnt_off, n, dev)
        donor_on = _pick_from_set(onpath, cnt_on, n, dev)
        shift = (torch.arange(R, device=dev) + 1) % max(1, R)
        donor_alt = donor_on[shift]

        src_step = torch.where(use_alt.view(-1, 1), donor_alt, donor_off).clamp(0, n - 1)
        src_row = torch.where(use_alt, rows[shift], rows).view(-1, 1).expand(R, n)
        donor_cmd = tok[:, 0::2, :][:, :n, :][src_row, src_step]

        cand = obs_all[rows]
        cand_sq = osq[rows]
        seen = span <= j_sel.view(-1, 1)
        cand_mask = contentful[rows] & seen
        pos_mask = (dup_o[rows, j_sel] | F.one_hot(j_sel, n).bool()) & cand_mask
        if not bool(pos_mask.any(dim=1).all().item()):
            return 0.0

        cmd_r = tok[rows][:, 0::2, :][:, :n, :]
        obs_r = tok[rows][:, 1::2, :][:, :n, :]
        cmd_sw = torch.where(onpath.unsqueeze(-1), donor_cmd.to(cmd_r.dtype), cmd_r)
        tok_nat = torch.stack([cmd_r, obs_r], dim=2).reshape(R, 2 * n, cmd_r.size(-1))
        tok_sw = torch.stack([cmd_sw, obs_r], dim=2).reshape(R, 2 * n, cmd_r.size(-1))
        tok2 = torch.cat([tok_nat, tok_sw], dim=0)
        types2 = torch.cat([types[rows], types[rows]], dim=0)
        kp2 = torch.cat([key_pad[rows], key_pad[rows]], dim=0)

    was_training = bool(net.training)
    net.eval()
    try:
        pred2, _ = net(tok2, types2, kp2)
    finally:
        net.train(was_training)

    if pred2.dim() != 3 or pred2.size(1) < 2 * n or pred2.size(-1) != cand.size(-1):
        return 0.0

    pcmd = pred2[:, 0::2, :][:, :n, :]
    ar = torch.arange(R, device=dev)
    p_nat = _clean(pcmd[:R][ar, j_sel].float())
    p_swp = _clean(pcmd[R:][ar, j_sel].float())

    temp = max(1e-3, float(cfg["cand_temp"]))

    def _logp_pos(q):
        dot = torch.bmm(cand, q.unsqueeze(2)).squeeze(2)
        d2 = (q.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / Dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)

    lp_nat = _logp_pos(p_nat)
    lp_swp = _logp_pos(p_swp)

    l_nat = -(lp_nat.mean())
    l_ca = torch.relu(float(cfg["margin"]) - (lp_nat - lp_swp)).mean()
    l_mse = (p_nat - cand[ar, j_sel]).pow(2).mean()

    total = (float(cfg["nat_weight"]) * l_nat
             + float(cfg["ca_weight"]) * l_ca
             + float(cfg["mse_weight"]) * l_mse)
    out_loss = float(cfg["aux_weight"]) * ramp * total
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
        0.0 < vals["row_frac"] <= 1.0,
        vals["max_rows"] >= 1.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["mut_min"] >= 1.0,
        vals["cand_temp"] > 0.0,
        vals["margin"] >= 0.0,
        vals["nat_weight"] >= 0.0,
        vals["ca_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
