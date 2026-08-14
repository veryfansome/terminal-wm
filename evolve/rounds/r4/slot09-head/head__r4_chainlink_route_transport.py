import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r4_chainlink_route_transport"
DESCRIPTION = (
    "Eval-active multi-hop content router over command positions, plus a train-time supervision "
    "of the routing distribution itself. Every command embedding is mapped by one shared encoder "
    "to a read address and a move-destination address; a per-step move gate and write gate are "
    "read from the command together with that step's observation, and each step's effective write "
    "address is a gate-blend of its destination address and its own read address, so a non-moving "
    "step writes where it reads. A strictly-causal softmax over read-address / write-address "
    "agreement gives a one-hop link matrix A; folding the move gate into it yields the linear "
    "recurrence content(i) = sum_j A[i,j] * (move_j * content(j) + (1 - move_j) * obs_j), solved "
    "exactly as one unit-lower-triangular system. The solve returns the absorption matrix T whose "
    "row i is a sub-stochastic distribution over EARLIER OBSERVATIONS, and the routed content "
    "T @ obs is injected into the command-position predictions through a sigmoid gate and a "
    "zero-initialised (D,D) readout, so the wrapped net is the unwrapped net at initialisation and "
    "no position ever reads its own or any later observation. Train-time aux, mined from the batch "
    "embeddings alone: exact-duplicate observation clusters label each step as content-bearing or "
    "content-free and supervise the move and write gates; a read-back step whose observation "
    "exactly duplicates earlier content with a content-free step in between supplies a "
    "cross-entropy on T's mass over that earlier-occurrence set, a cosine/MSE anchor on the routed "
    "content, an address-spread and read-versus-destination separation hinge, and a forward-model "
    "consistency term requiring the arch's shared latent-transition operator to carry the earlier "
    "content through the intervening content-free command."
)

_DEFAULTS = {
    "key_d": 64,
    "hid": 256,
    "link_temp": 0.25,
    "obs_dup_frac": 0.02,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "mut_min": 4,
    "max_examples": 256,
    "gate_weight": 0.5,
    "route_weight": 1.0,
    "anchor_weight": 0.05,
    "transition_weight": 0.1,
    "spread_weight": 0.2,
    "spread_margin": 0.3,
    "move_sep_margin": 0.4,
    "aux_weight": 1.0,
    "ramp_steps": 300,
    "dst_init_noise": 0.02,
}

_EPS = 1e-8
_CLIP = 1e4
_PROB_EPS = 1e-6


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=_CLIP, neginf=-_CLIP)


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _bce(p, y):
    q = p.clamp(_PROB_EPS, 1.0 - _PROB_EPS)
    return -(y * torch.log(q) + (1.0 - y) * torch.log1p(-q))


def _pad_steps(x, n):
    cur = x.size(1)
    if cur == n:
        return x
    if cur > n:
        return x[:, :n]
    shape = (x.size(0), n - cur) + tuple(x.shape[2:])
    return torch.cat([x, x.new_zeros(shape)], dim=1)


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
    if tok.shape[1] != 2 * tgt.shape[1] or tok.shape[2] != tgt.shape[2]:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


def _solve_absorb(mmat, rhs):
    if mmat.size(1) == 0:
        return rhs
    if mmat.device.type != "mps":
        try:
            return torch.linalg.solve_triangular(-mmat, rhs, upper=False, unitriangular=True)
        except Exception:
            pass
    rows = []
    n = mmat.size(1)
    for i in range(n):
        r = rhs[:, i, :]
        if i > 0:
            prev = torch.stack(rows, dim=1)
            r = r + torch.bmm(mmat[:, i:i + 1, :i], prev).squeeze(1)
        rows.append(r)
    return torch.stack(rows, dim=1)


class _ChainLinkRouter(nn.Module):

    def __init__(self, d_model, key_d, hid, link_temp, dst_init_noise):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = int(key_d)
        self.link_temp = max(1e-3, float(link_temp))
        h = int(hid)

        self.path_in = nn.Linear(self.d_model, h)
        self.src_key = nn.Linear(h, self.key_d, bias=False)
        self.dst_key = nn.Linear(h, self.key_d, bias=False)
        with torch.no_grad():
            noise = torch.randn_like(self.src_key.weight) * float(dst_init_noise)
            self.dst_key.weight.copy_(self.src_key.weight + noise)

        self.obs_in = nn.Linear(self.d_model, h)
        self.step_in = nn.Linear(2 * h, h)
        self.move_out = nn.Linear(h, 1)
        self.write_out = nn.Linear(h, 1)
        nn.init.constant_(self.move_out.bias, 0.0)
        nn.init.constant_(self.write_out.bias, 1.0)

        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.read_gate.bias, 0.0)
        self.read_out = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)

    def route(self, tok, types, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        if L < 1:
            return None
        n = (L + 1) // 2
        n_pair = L // 2
        dev = tok.device
        dt = tok.dtype

        c = _clean(tok[:, 0::2, :])
        if key_pad is None:
            vc = torch.ones(B, n, dtype=torch.bool, device=dev)
            vo = torch.ones(B, n_pair, dtype=torch.bool, device=dev)
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2]
            vo = v[:, 1::2]
        vo_pad = _pad_steps(vo.unsqueeze(-1).to(dt), n).squeeze(-1) > 0.5

        g = F.gelu(self.path_in(c))
        q = _unit(self.src_key(g))
        kd = _unit(self.dst_key(g))

        if n_pair > 0:
            o = _pad_steps(_clean(tok[:, 1::2, :]), n)
        else:
            o = tok.new_zeros(B, n, self.d_model)

        og = F.gelu(self.obs_in(o))
        sg = F.gelu(self.step_in(torch.cat([g, og], dim=-1)))
        live_obs = (vo_pad & vc).to(dt)
        move = torch.sigmoid(self.move_out(sg)).squeeze(-1) * live_obs
        beta = torch.sigmoid(self.write_out(sg)).squeeze(-1) * live_obs

        w = _unit(move.unsqueeze(-1) * kd + (1.0 - move).unsqueeze(-1) * q)

        scores = torch.bmm(q, w.transpose(1, 2)) / self.link_temp
        scores = scores + torch.log(beta + _PROB_EPS).unsqueeze(1)

        pos = torch.arange(n, device=dev)
        causal = (pos.unsqueeze(1) > pos.unsqueeze(0)).unsqueeze(0)
        allowed = causal & vc.unsqueeze(1) & vo_pad.unsqueeze(1) & vc.unsqueeze(2)

        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg)
        has_key = allowed.any(dim=2, keepdim=True)
        link = torch.softmax(scores, dim=2)
        link = torch.where(has_key, link, torch.zeros_like(link))

        mmat = link * move.unsqueeze(1)
        rmat = link * (1.0 - move).unsqueeze(1)
        absorb = _solve_absorb(mmat, rmat)
        absorb = _clean(absorb).clamp(-2.0, 2.0)

        routed = torch.bmm(absorb, o)
        routed = _clean(routed)

        gate = torch.sigmoid(self.read_gate(g))
        contrib = gate * self.read_out(routed)
        contrib = _clean(contrib) * vc.unsqueeze(-1).to(dt)

        return {"absorb": absorb, "routed": routed, "contrib": contrib,
                "q": q, "kd": kd, "w": w, "move": move, "beta": beta}


def wrap(net, D, **params):
    prior = getattr(net, "_chainlink_state", None)
    if prior is not None:
        return prior

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ChainLinkRouter(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                           float(cfg["link_temp"]), float(cfg["dst_init_noise"]))
    net.chainlink_router = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"]:
            return pred, h
        if tok_emb.dim() != 3 or tok_emb.size(-1) != state["D"]:
            return pred, h
        out = mod.route(tok_emb, types, key_pad)
        if out is None:
            return pred, h
        contrib = out["contrib"]
        cmd_len = pred[:, 0::2, :].size(1)
        k = min(int(contrib.size(1)), int(cmd_len))
        if k < 1:
            return pred, h
        new_pred = pred.clone()
        cmd_view = new_pred[:, 0::2, :]
        cmd_view[:, :k, :] = cmd_view[:, :k, :] + contrib[:, :k, :].to(pred.dtype)
        if torch.is_grad_enabled() and mod.training:
            stash = dict(out)
            stash["tok"] = tok_emb
            state["stash"] = stash
        return new_pred, h

    net.forward = _forward
    net._chainlink_state = state
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
    if not _layout_ok(batch):
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    B, n = valid.shape
    if n < 3:
        return 0.0

    if stash is not None and stash.get("tok") is tok:
        out = stash
    else:
        out = mod.route(tok, batch["types"], batch["key_pad"])
    if out is None:
        return 0.0

    absorb = out["absorb"][:, :n, :n]
    routed = out["routed"][:, :n, :]
    q = out["q"][:, :n, :]
    kd = out["kd"][:, :n, :]
    w = out["w"][:, :n, :]
    move = out["move"][:, :n]
    beta = out["beta"][:, :n]

    c = _clean(tok[:, 0::2, :][:, :n, :])
    o = _clean(tok[:, 1::2, :][:, :n, :])
    dim = float(o.size(-1))
    dev = o.device
    idxn = torch.arange(n, device=dev)
    cap = max(1, int(cfg["max_examples"]))

    with torch.no_grad():
        osq = o.pow(2).sum(dim=-1)
        csq = c.pow(2).sum(dim=-1)
        d2o = (osq.unsqueeze(2) + osq.unsqueeze(1)
               - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / dim
        d2c = (csq.unsqueeze(2) + csq.unsqueeze(1)
               - 2.0 * torch.bmm(c, c.transpose(1, 2))).clamp_min(0.0) / dim

        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["obs_dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        clus = dup_o.sum(dim=2)
        contentful = valid & (clus <= int(cfg["max_dup"]))
        mutation = valid & (clus >= int(cfg["mut_min"]))

        tri = (idxn.unsqueeze(1) > idxn.unsqueeze(0)).unsqueeze(0)
        origin = (dup_o & tri & (~dup_c)
                  & contentful.unsqueeze(2) & contentful.unsqueeze(1))
        has_origin = origin.any(dim=2)

        small = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)
        big = torch.full((1, 1, n), n, device=dev, dtype=torch.long)
        last_origin = torch.where(origin, idxn.view(1, 1, n), small).amax(dim=2)
        first_origin = torch.where(origin, idxn.view(1, 1, n), big).amin(dim=2)

        mcs = torch.cumsum(mutation.long(), dim=1)
        jm1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        mut_between = (torch.gather(mcs, 1, jm1)
                       - torch.gather(mcs, 1, last_origin.clamp_min(0))) > 0

        mined = has_origin & contentful & mut_between & valid
        nz = torch.nonzero(mined, as_tuple=False)
        n_mined = int(nz.shape[0])
        if n_mined > 0:
            b_idx = nz[:, 0]
            j_idx = nz[:, 1]
            depth = (idxn.view(1, n) - first_origin.clamp(0, n - 1)).clamp_min(0)
            if n_mined > cap:
                _, order = torch.topk(depth[b_idx, j_idx].float(), cap)
                b_idx = b_idx[order]
                j_idx = j_idx[order]
                n_mined = cap
            pos_mask = origin[b_idx, j_idx]

        distinct = off & (~dup_o) & contentful.unsqueeze(2) & contentful.unsqueeze(1)

    total = routed.sum() * 0.0

    gw = float(cfg["gate_weight"])
    if gw > 0.0:
        vf = valid.to(move.dtype)
        want_move = mutation.to(move.dtype)
        want_write = (contentful | mutation).to(beta.dtype)
        gate_err = _bce(move, want_move) + _bce(beta, want_write)
        total = total + gw * ((gate_err * vf).sum() / vf.sum().clamp_min(1.0))

    if n_mined > 0:
        rw = float(cfg["route_weight"])
        if rw > 0.0:
            mass = (absorb[b_idx, j_idx] * pos_mask.to(absorb.dtype)).sum(dim=1)
            total = total + rw * (-torch.log(mass.clamp_min(0.0) + _PROB_EPS)).mean()

        aw = float(cfg["anchor_weight"])
        if aw > 0.0:
            r_sel = routed[b_idx, j_idx]
            t_sel = o[b_idx, j_idx]
            cos_err = 1.0 - (_unit(r_sel) * _unit(t_sel)).sum(dim=-1).clamp(-1.0, 1.0)
            mse_err = (r_sel - t_sel).pow(2).mean(dim=-1)
            total = total + aw * (cos_err.mean() + 0.2 * mse_err.mean())

    sw = float(cfg["spread_weight"])
    if sw > 0.0:
        cos_ww = torch.bmm(w, w.transpose(1, 2)).clamp(-1.0, 1.0)
        df = distinct.to(cos_ww.dtype)
        spread = ((F.relu(cos_ww - float(cfg["spread_margin"])) * df).sum()
                  / df.sum().clamp_min(1.0))
        cos_qd = (q * kd).sum(dim=-1).clamp(-1.0, 1.0)
        mf = mutation.to(cos_qd.dtype)
        movesep = ((F.relu(cos_qd - float(cfg["move_sep_margin"])) * mf).sum()
                   / mf.sum().clamp_min(1.0))
        total = total + sw * (spread + movesep)

    tw = float(cfg["transition_weight"])
    op = getattr(net, "transition_from_emb", None)
    n_tr = 0
    if tw > 0.0 and callable(op) and n_mined > 0:
        with torch.no_grad():
            src = last_origin[b_idx, j_idx]
            mut_in = (mutation[b_idx] & (idxn.view(1, n) < j_idx.view(-1, 1))
                      & (idxn.view(1, n) > src.view(-1, 1)))
            k_sel = torch.where(mut_in, idxn.view(1, n),
                                torch.full_like(mut_in, -1, dtype=torch.long)).amax(dim=1)
            keep = k_sel >= 0
            tb = b_idx[keep]
            tj = j_idx[keep]
            tsrc = src[keep]
            tk = k_sel[keep]
            n_tr = int(tb.shape[0])
        if n_tr > 0:
            pre = o[tb, tsrc]
            cmd_k = c[tb, tk]
            gold = o[tb, tj]
            pred_tr = _clean(op(pre, cmd_k))
            cos_err = 1.0 - (_unit(pred_tr) * _unit(gold)).sum(dim=-1).clamp(-1.0, 1.0)
            mse_err = (pred_tr - gold).pow(2).mean(dim=-1)
            total = total + tw * (cos_err.mean() + 0.2 * mse_err.mean())

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
        vals["key_d"] >= 1.0,
        vals["hid"] >= 8.0,
        vals["link_temp"] > 0.0,
        0.0 < vals["obs_dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["mut_min"] > vals["max_dup"],
        vals["max_examples"] >= 1.0,
        vals["gate_weight"] >= 0.0,
        vals["route_weight"] >= 0.0,
        vals["anchor_weight"] >= 0.0,
        vals["transition_weight"] >= 0.0,
        vals["spread_weight"] >= 0.0,
        -1.0 <= vals["spread_margin"] <= 1.0,
        -1.0 <= vals["move_sep_margin"] <= 1.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["dst_init_noise"] >= 0.0,
    ]
    return all(checks)
