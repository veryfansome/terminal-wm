import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r5_closure_routing_memory"
DESCRIPTION = (
    "Content routing solved as a truncated Neumann series over a causal path-match matrix, added "
    "to the arch's own prediction. The stream is indexed by COMMAND SLOT, n = (L + 1) // 2, so a "
    "trailing command whose observation is absent -- the scored read -- is a first-class row; "
    "pairing is never inferred from the length parity and is read per row from key_pad. Each "
    "command embedding is mapped by two extractors through one shared path projection into a "
    "source key and a destination key; a causal softmax with a learned null slot and a learned "
    "recency decay yields two attention matrices (read-at-source, read-at-destination). A per-step "
    "transport gate and a simplex pair of mixing coefficients build a strictly lower-triangular "
    "routing matrix M from command embeddings alone, and the per-step content register "
    "V = sum_p M^p ((1-gate)*obs*obs_live) is formed by `hops` batched matmuls, so a move chain of "
    "depth d is resolved by the d-th term instead of by a per-step loop. Transport needs no "
    "observation; only the content deposit does, and a slot with no observation deposits nothing "
    "and is read by nobody, since M and the read attention are both strictly lower triangular. The "
    "read at a command position is the source-key attention over V, gated and scaled by a "
    "zero-initialised per-dimension vector, so the wrapped net equals the unwrapped net at "
    "initialisation. No pooled statistic enters the forward path. Train-time aux mines, "
    "label-free, steps whose observation duplicates an earlier step's observation under a "
    "different command, and trains the read and the arch's prediction at those steps with a "
    "within-sequence multi-positive squared-L2 InfoNCE over the sequence's other observations plus "
    "an MSE anchor; every statistic it forms is restricted to slots that carry an observation."
)

_DEFAULTS = {
    "key_d": 64,
    "hid": 192,
    "hops": 7,
    "init_temp": 0.2,
    "init_decay": 0.03,
    "init_null": 1.0,
    "sym_init_noise": 0.02,
    "dup_frac": 0.03,
    "cmd_dup_frac": 0.02,
    "max_dup": 3,
    "max_examples": 256,
    "cand_temp": 0.5,
    "nce_weight": 1.0,
    "pred_weight": 0.5,
    "mse_weight": 0.05,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _cmd_slots(length):
    return (int(length) + 1) // 2


def _fit_rows(x, n):
    m = x.size(1)
    if m == n:
        return x
    if m > n:
        return x[:, :n]
    pad = x.new_zeros(x.size(0), n - m, *tuple(x.shape[2:]))
    return torch.cat([x, pad], dim=1)


def _split_stream(tok, key_pad, n):
    B = tok.size(0)
    L = tok.size(1)
    dev = tok.device
    c = _fit_rows(_clean(tok[:, 0::2, :]), n)
    o = _fit_rows(_clean(tok[:, 1::2, :]), n)
    if key_pad is None:
        live_c = torch.ones(B, n, dtype=torch.bool, device=dev)
        live_o = (torch.arange(n, device=dev).view(1, n) < (L // 2)).expand(B, n)
    else:
        lv = ~key_pad.bool()
        live_c = _fit_rows(lv[:, 0::2], n)
        live_o = _fit_rows(lv[:, 1::2], n)
    return c, o, live_c, live_o


def _shapes_ok(b):
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
    return tgt.shape[1] == _cmd_slots(tok.shape[1])


def _types_interleaved(types, key_pad):
    if types is None or types.dim() != 2:
        return False
    if key_pad is None:
        even = types[:, 0::2].reshape(-1)
        odd = types[:, 1::2].reshape(-1)
    else:
        live = ~key_pad.bool()
        even = types[:, 0::2][live[:, 0::2]]
        odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0:
        return False
    if not bool((even == 0).all().item()):
        return False
    if odd.numel() == 0:
        return True
    return bool((odd == 1).all().item())


def _layout_for(cache, length, types, key_pad):
    key = int(length)
    known = cache.get(key)
    if known is None:
        known = _types_interleaved(types, key_pad)
        cache[key] = known
    return known


class _ClosureRouter(nn.Module):
    def __init__(self, d_model, key_d, hid, hops, init_temp, init_decay, init_null,
                 sym_init_noise):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = int(key_d)
        self.hops = max(1, int(hops))
        h = int(hid)

        self.src_in = nn.Linear(self.d_model, h)
        self.dst_in = nn.Linear(self.d_model, h)
        with torch.no_grad():
            noise = torch.randn_like(self.src_in.weight) * float(sym_init_noise)
            self.dst_in.weight.copy_(self.src_in.weight + noise)
            self.dst_in.bias.copy_(self.src_in.bias)
        self.path_key = nn.Linear(h, self.key_d, bias=False)

        self.ctl_in = nn.Linear(self.d_model, h)
        self.move_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        self.mix_coef = nn.Linear(h, 2)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.read_gate.bias, 0.0)
        nn.init.zeros_(self.mix_coef.weight)
        with torch.no_grad():
            self.mix_coef.bias.copy_(torch.tensor([1.5, -1.5]))

        t0 = max(1e-3, float(init_temp))
        d0 = max(1e-4, float(init_decay))
        self.log_temp = nn.Parameter(torch.tensor(math.log(t0)))
        self.log_decay = nn.Parameter(torch.tensor(math.log(math.expm1(d0))))
        self.null_src = nn.Parameter(torch.full((1,), float(init_null)))
        self.null_dst = nn.Parameter(torch.full((1,), float(init_null)))
        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def route_chain(self, tok, key_pad):
        if tok is None or tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        n = _cmd_slots(tok.size(1))
        if n < 2:
            return None
        B = tok.size(0)
        dev = tok.device
        dt = tok.dtype

        c_raw, o_raw, live_c, live_o = _split_stream(tok, key_pad, n)
        c = c_raw.float()
        o = o_raw.float()

        cf = F.gelu(self.ctl_in(c))
        g = torch.sigmoid(self.move_gate(cf))
        rg = torch.sigmoid(self.read_gate(cf))
        ab = torch.softmax(self.mix_coef(cf), dim=-1)

        q = _unit(self.path_key(F.gelu(self.src_in(c))))
        k = _unit(self.path_key(F.gelu(self.dst_in(c))))

        temp = self.log_temp.exp().clamp(1e-2, 1e2)
        decay = F.softplus(self.log_decay)
        pos = torch.arange(n, device=dev)
        gap = (pos.view(n, 1) - pos.view(1, n)).float().clamp_min(0.0)
        allow = (pos.view(n, 1) > pos.view(1, n)).unsqueeze(0) & live_c.unsqueeze(2) & live_c.unsqueeze(1)

        bias = (-decay * gap).unsqueeze(0)
        l_src = torch.bmm(q, k.transpose(1, 2)) / temp + bias
        l_dst = torch.bmm(k, k.transpose(1, 2)) / temp + bias
        l_src = l_src.masked_fill(~allow, _NEG)
        l_dst = l_dst.masked_fill(~allow, _NEG)

        a_src = torch.softmax(
            torch.cat([l_src, self.null_src.view(1, 1, 1).expand(B, n, 1).to(l_src.dtype)], dim=2),
            dim=2)[:, :, :n]
        a_dst = torch.softmax(
            torch.cat([l_dst, self.null_dst.view(1, 1, 1).expand(B, n, 1).to(l_dst.dtype)], dim=2),
            dim=2)[:, :, :n]

        move = g * (ab[..., 0:1] * a_src + ab[..., 1:2] * a_dst)
        u = (1.0 - g) * o * live_o.unsqueeze(-1).float()
        v = u
        for _ in range(self.hops):
            u = torch.bmm(move, u)
            v = v + u
        reads = _clean(torch.bmm(a_src, v))
        contrib = rg * reads * self.out_scale.view(1, 1, -1)
        contrib = _clean(contrib) * live_c.unsqueeze(-1).float()
        return reads, contrib.to(dt)


def wrap(net, D, **params):
    prev = getattr(net, "_closure_router_state", None)
    if prev is not None:
        return prev

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ClosureRouter(int(D), int(cfg["key_d"]), int(cfg["hid"]), int(cfg["hops"]),
                         float(cfg["init_temp"]), float(cfg["init_decay"]),
                         float(cfg["init_null"]), float(cfg["sym_init_noise"]))
    net.closure_router = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D), "layout": {}}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        out = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if not (isinstance(out, tuple) and len(out) >= 1):
            return out
        pred = out[0]
        if not torch.is_tensor(pred) or pred.dim() != 3 or pred.size(-1) != state["D"]:
            return out
        if not torch.is_tensor(tok_emb) or tok_emb.dim() != 3 or tok_emb.size(-1) != state["D"]:
            return out
        L = tok_emb.size(1)
        n = _cmd_slots(L)
        if n < 2:
            return out
        if not _layout_for(state["layout"], L, types, key_pad):
            return out
        routed = mod.route_chain(tok_emb, key_pad)
        if routed is None:
            return out
        reads, contrib = routed
        if pred.size(1) == L:
            slots = torch.arange(0, L, 2, device=pred.device)
            spread = torch.zeros_like(pred).index_copy(1, slots, contrib.to(pred.dtype))
            new_pred = pred + spread
            pred_cmd = new_pred[:, 0::2, :]
        elif pred.size(1) == n:
            new_pred = pred + contrib.to(pred.dtype)
            pred_cmd = new_pred
        else:
            return out
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads, "pred_cmd": pred_cmd}
        return (new_pred,) + tuple(out[1:])

    net.forward = _forward
    net._closure_router_state = state
    return state


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    cfg = st["cfg"]
    mod = st["mod"]
    stash = st.get("stash")
    st["stash"] = None

    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0
    if not _shapes_ok(batch):
        return 0.0

    tok = batch["tok"]
    if not _layout_for(st["layout"], tok.size(1), batch["types"], batch["key_pad"]):
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    B, n = batch["cmd_mask"].shape
    if n < 3:
        return 0.0

    c_raw, o_raw, live_c, live_o = _split_stream(tok, batch["key_pad"], n)
    valid = batch["cmd_mask"].bool() & live_c & live_o
    if not bool(valid.any().item()):
        return 0.0

    pred_cmd = None
    if stash is not None and stash["tok"] is tok:
        reads = stash["reads"]
        pred_cmd = stash["pred_cmd"]
    else:
        routed = mod.route_chain(tok, batch["key_pad"])
        if routed is None:
            return 0.0
        reads = routed[0]
    if reads.size(1) < n:
        return 0.0
    reads = reads[:, :n, :].float()
    if pred_cmd is not None and pred_cmd.size(1) >= n:
        pred_cmd = pred_cmd[:, :n, :].float()
    else:
        pred_cmd = None

    c = c_raw.float()
    o = o_raw.float()
    Dm = float(o.size(-1))
    dev = o.device

    with torch.no_grad():
        osq = o.pow(2).sum(-1)
        csq = c.pow(2).sum(-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / Dm
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / Dm

        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        off = valid.unsqueeze(2) & valid.unsqueeze(1) & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        contentful = valid & (dup_o.sum(dim=2) <= int(cfg["max_dup"]))
        idxn = torch.arange(n, device=dev)
        tri = (idxn.view(n, 1) > idxn.view(1, n)).unsqueeze(0)
        hit = dup_o & (~dup_c) & tri & contentful.unsqueeze(2) & contentful.unsqueeze(1)
        mined = hit.any(dim=2)
        nz = torch.nonzero(mined, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        b_idx = nz[:, 0]
        i_idx = nz[:, 1]

        far = torch.full((B, n, n), n, dtype=torch.long, device=dev)
        first_j = torch.where(hit, idxn.view(1, 1, n).expand(B, n, n), far).amin(dim=2)
        span = (i_idx - first_j[b_idx, i_idx]).clamp_min(0).float()
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(span, cap)
            b_idx = b_idx[order]
            i_idx = i_idx[order]

        seen = idxn.view(1, n) <= i_idx.view(-1, 1)
        cand_mask = contentful[b_idx] & seen
        pos_mask = (dup_o[b_idx, i_idx] | F.one_hot(i_idx, n).bool()) & cand_mask
        keep = pos_mask.any(dim=1) & (cand_mask.sum(dim=1) >= 2)
        if not bool(keep.any().item()):
            return 0.0
        b_idx = b_idx[keep]
        i_idx = i_idx[keep]
        cand_mask = cand_mask[keep]
        pos_mask = pos_mask[keep]
        cand_sq = osq[b_idx]

    cand = o[b_idx]
    tgt = o[b_idx, i_idx]
    temp = max(1e-3, float(cfg["cand_temp"]))

    def _rank_loss(query):
        dot = torch.bmm(cand, query.unsqueeze(2)).squeeze(2)
        d2 = (query.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / Dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return -(torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)).mean()

    r_sel = reads[b_idx, i_idx]
    total = float(cfg["nce_weight"]) * _rank_loss(r_sel)
    total = total + float(cfg["mse_weight"]) * (r_sel - tgt).pow(2).mean()
    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * _rank_loss(pred_cmd[b_idx, i_idx])

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
        vals["key_d"] >= 4.0,
        vals["hid"] >= 8.0,
        1.0 <= vals["hops"] <= 32.0,
        vals["init_temp"] > 0.0,
        vals["init_decay"] > 0.0,
        abs(vals["init_null"]) <= 20.0,
        vals["sym_init_noise"] >= 0.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cand_temp"] > 0.0,
        vals["nce_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
