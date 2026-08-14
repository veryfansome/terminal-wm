import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r6_chain_closure_link_router"
DESCRIPTION = (
    "A provenance link matrix closed in one triangular solve. Every command position emits two "
    "unit keys from its own token; the link logit from a later command i to an earlier command j "
    "is a SIGNED bilinear form, <ka_i,ka_j> + s_j * <kb_i,kb_j>, where s_j = 1 - 2*mu_j is set by "
    "the same per-command depositor gate mu_j that decides what j asserts. A depositor (s_j=-1) is "
    "linked to when the querying command's location matches the location j wrote INTO; an asserter "
    "(s_j=+1) is linked to when the locations match with the same role. A learned recency tilt, a "
    "per-column assert score and a per-row null column complete the row-softmax A. Content is then "
    "defined by the fixed point V = diag(mu) A V + (1-mu) * OBS, i.e. what a command asserts is "
    "either the content it carried in (depositor) or the observation it revealed (asserter). "
    "Because diag(mu)A is strictly lower triangular, the fixed point is exact and is obtained by a "
    "single unit-lower-triangular solve, and the read at command i is R = A V — a closed-form "
    "transitive closure of the move chain rather than a stepwise slot machine. The read exists at "
    "EVERY command position including the last command of an odd-length sequence, whose observation "
    "is absent because it is the answer, and it enters the prediction there through an identity-"
    "initialised (D,D) readout times a per-command gate. Train-time aux carries a one-hot origin "
    "channel through the SAME solve, so the closure exposes, per position, the distribution over "
    "which earlier command originated the content it is now reading; duplicate observations "
    "separated by content-free steps supply a label-free positive set for that distribution and for "
    "a squared-L2 InfoNCE on the read itself, weighted by how many content-free steps the chain "
    "crossed, plus a BCE that calibrates mu against whether a step produced content. No aux and no "
    "origin channel at eval."
)

_DEFAULTS = {
    "key_d": 64,
    "id_scale": 2.0,
    "role_scale": 4.0,
    "rec_scale": 1.0,
    "wo_init": 0.02,
    "dup_frac": 0.05,
    "max_dup": 3,
    "depth_cap": 6.0,
    "base_w": 0.25,
    "bonus_w": 1.0,
    "max_examples": 256,
    "nce_temp": 0.5,
    "nce_weight": 0.5,
    "origin_weight": 0.5,
    "mse_weight": 0.05,
    "gate_weight": 0.4,
    "ent_weight": 0.01,
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


def _inv_softplus(v):
    v = max(1e-4, float(v))
    if v > 20.0:
        return v
    return math.log(math.expm1(v))


def _layout_ok(b):
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


def _solve_unit_lower(sysm, rhs):
    if sysm.device.type != "mps":
        try:
            return torch.linalg.solve_triangular(sysm, rhs, upper=False, unitriangular=True)
        except Exception:
            pass
    outs = []
    for i in range(sysm.size(1)):
        yi = rhs[:, i, :]
        if outs:
            prev = torch.stack(outs, dim=1)
            yi = yi - torch.bmm(sysm[:, i:i + 1, :i], prev).squeeze(1)
        outs.append(yi)
    return torch.stack(outs, dim=1)


class _ChainClosureRouter(nn.Module):
    def __init__(self, d_model, key_d, id_scale, role_scale, rec_scale, wo_init):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(4, int(key_d))

        self.k_id = nn.Linear(self.d_model, self.key_d, bias=False)
        self.k_role = nn.Linear(self.d_model, self.key_d, bias=False)

        self.move_score = nn.Linear(self.d_model, 1)
        self.assert_score = nn.Linear(self.d_model, 1)
        self.null_score = nn.Linear(self.d_model, 1)
        self.read_gate = nn.Linear(self.d_model, 1)
        nn.init.zeros_(self.move_score.bias)
        nn.init.zeros_(self.assert_score.weight)
        nn.init.zeros_(self.assert_score.bias)
        nn.init.zeros_(self.null_score.weight)
        nn.init.zeros_(self.null_score.bias)
        nn.init.zeros_(self.read_gate.bias)

        self.log_id = nn.Parameter(torch.tensor(_inv_softplus(id_scale)))
        self.log_role = nn.Parameter(torch.tensor(_inv_softplus(role_scale)))
        self.log_rec = nn.Parameter(torch.tensor(_inv_softplus(rec_scale)))

        self.out_lin = nn.Linear(self.d_model, self.d_model, bias=False)
        with torch.no_grad():
            self.out_lin.weight.copy_(torch.eye(self.d_model) * float(wo_init))

    def run(self, tok, key_pad, want_trace):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, Dm = tok.shape
        n = (L + 1) // 2
        if n < 2:
            return None
        dev = tok.device
        dt = tok.dtype

        c = torch.nan_to_num(tok[:, 0::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = torch.nan_to_num(tok[:, 1::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        if o.size(1) < n:
            o = torch.cat([o, o.new_zeros(B, n - o.size(1), Dm)], dim=1)

        if key_pad is None:
            vc = torch.ones(B, n, dtype=torch.bool, device=dev)
            vo = torch.ones(B, n, dtype=torch.bool, device=dev)
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2]
            vo = v[:, 1::2]
            if vo.size(1) < n:
                vo = torch.cat(
                    [vo, torch.zeros(B, n - vo.size(1), dtype=torch.bool, device=dev)], dim=1)
        obs = o * vo.unsqueeze(-1).to(dt)

        qa = _unit(self.k_id(c))
        qb = _unit(self.k_role(c))
        mv_logit = self.move_score(c).squeeze(-1)
        mu = torch.sigmoid(mv_logit)
        sgn = 1.0 - 2.0 * mu

        s_id = F.softplus(self.log_id).to(dt)
        s_role = F.softplus(self.log_role).to(dt)
        s_rec = F.softplus(self.log_rec).to(dt)

        sim_a = torch.bmm(qa, qa.transpose(1, 2))
        sim_b = torch.bmm(qb, qb.transpose(1, 2))
        col = self.assert_score(c).squeeze(-1).unsqueeze(1)
        idx = torch.arange(n, device=dev, dtype=dt)
        rel = torch.log1p((idx.view(1, n, 1) - idx.view(1, 1, n)).clamp_min(0.0))

        logits = s_id * sim_a + s_role * sgn.unsqueeze(1) * sim_b + col - s_rec * rel
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-60.0, 60.0)
        allow = torch.tril(
            torch.ones(n, n, device=dev, dtype=torch.bool), diagonal=-1).unsqueeze(0) & vc.unsqueeze(1)
        logits = logits.masked_fill(~allow, _NEG)
        full = torch.cat([self.null_score(c), logits], dim=2)
        attn = torch.softmax(full, dim=2)
        a = attn[:, :, 1:] * allow.to(dt)

        if want_trace:
            eye_n = torch.eye(n, device=dev, dtype=dt).unsqueeze(0).expand(B, n, n)
            rhs_val = torch.cat([obs, eye_n], dim=2)
        else:
            rhs_val = obs
        rhs = (1.0 - mu).unsqueeze(-1) * rhs_val
        sysm = torch.eye(n, device=dev, dtype=dt).unsqueeze(0) - mu.unsqueeze(-1) * a
        vv = _solve_unit_lower(sysm, rhs)
        vv = torch.nan_to_num(vv, nan=0.0, posinf=1e4, neginf=-1e4)
        reads = torch.bmm(a, vv)
        reads = torch.nan_to_num(reads, nan=0.0, posinf=1e4, neginf=-1e4)

        r_obs = reads[..., :Dm]
        origin = reads[..., Dm:] if want_trace else None

        g = torch.sigmoid(self.read_gate(c))
        contrib = g * self.out_lin(r_obs) * vc.unsqueeze(-1).to(dt)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)

        return {"contrib": contrib, "reads": r_obs, "origin": origin, "attn": attn,
                "mv_logit": mv_logit, "vc": vc}


def wrap(net, D, **params):
    existing = getattr(net, "_chain_closure_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ChainClosureRouter(int(D), cfg["key_d"], cfg["id_scale"], cfg["role_scale"],
                              cfg["rec_scale"], cfg["wo_init"])
    net.chain_closure_router = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "layout": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"]:
            return pred, h
        if tok_emb.dim() != 3 or tok_emb.size(1) < 2 or tok_emb.size(1) != pred.size(1):
            return pred, h
        trace = bool(torch.is_grad_enabled() and mod.training)
        out = mod.run(tok_emb, key_pad, trace)
        if out is None:
            return pred, h
        contrib = out["contrib"]
        n = min(contrib.size(1), (pred.size(1) + 1) // 2)
        if n < 1:
            return pred, h
        add = torch.zeros_like(pred)
        add[:, 0:2 * n:2, :] = contrib[:, :n, :]
        merged = pred + add
        if trace:
            state["stash"] = {"tok": tok_emb, "reads": out["reads"], "origin": out["origin"],
                              "attn": out["attn"], "mv_logit": out["mv_logit"], "vc": out["vc"]}
        return merged, h

    net.forward = _forward
    net._chain_closure_state = state
    return state


def aux_loss(head_state, batch, net, device):
    st = head_state
    if st is None:
        return 0.0
    cfg = st["cfg"]
    stash = st.get("stash")
    st["stash"] = None

    if float(cfg["aux_weight"]) <= 0.0:
        return 0.0
    if stash is None or stash.get("origin") is None:
        return 0.0
    layout = st.get("layout")
    if layout is None:
        layout = bool(_layout_ok(batch))
        st["layout"] = layout
    if not layout:
        return 0.0

    tok = batch["tok"]
    if stash["tok"] is not tok:
        return 0.0

    st["step"] = int(st.get("step", 0)) + 1
    ramp = _smoothstep(st["step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    reads = stash["reads"]
    origin = stash["origin"]
    attn = stash["attn"]
    mv_logit = stash["mv_logit"]
    n = reads.size(1)
    if n < 3:
        return 0.0
    cm = batch["cmd_mask"].bool()
    if cm.size(1) < n or origin.size(2) != n:
        return 0.0

    B = tok.size(0)
    Dm = tok.size(-1)
    dev = tok.device
    dt = reads.dtype
    valid = stash["vc"][:, :n] & cm[:, :n]
    o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4)

    total = reads.sum() * 0.0

    with torch.no_grad():
        if not bool(valid.any().item()):
            return 0.0
        osq = o.pow(2).sum(-1)
        d2 = (osq.unsqueeze(2) + osq.unsqueeze(1)
              - 2.0 * torch.bmm(o, o.transpose(1, 2))).clamp_min(0.0) / float(Dm)
        eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
        vv2 = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv2 & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2.dtype)
        ref = ((d2 * off).sum(dim=(1, 2)) / cnt).clamp_min(1e-8)
        dup = off & (d2 <= (float(cfg["dup_frac"]) * ref).view(B, 1, 1))
        clus = dup.sum(dim=2)
        mut = valid & (clus > int(cfg["max_dup"]))
        flat_clus = clus.masked_fill(~valid, -1).reshape(-1)
        top = int(torch.argmax(flat_clus).item())
        if int(flat_clus[top].item()) > int(cfg["max_dup"]):
            proto = o.reshape(-1, Dm)[top]
            d2p = (osq + proto.pow(2).sum() - 2.0 * (o * proto.view(1, 1, Dm)).sum(-1)).clamp_min(0.0)
            d2p = d2p / float(Dm)
            mut = mut | (valid & (d2p <= float(cfg["dup_frac"]) * ref.view(B, 1)))
        content = valid & (~mut)
        mcs = torch.cumsum(mut.long(), dim=1)

        idxn = torch.arange(n, device=dev)
        tri = idxn.view(1, n, 1) > idxn.view(1, 1, n)
        pair = dup & tri & content.unsqueeze(2) & content.unsqueeze(1)
        has_p = pair.any(dim=2)
        big = torch.full((B, n, n), n, device=dev, dtype=torch.long)
        first_j = torch.where(pair, idxn.view(1, 1, n).expand(B, n, n), big).amin(dim=2)
        im1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        depth = (torch.gather(mcs, 1, im1)
                 - torch.gather(mcs, 1, first_j.clamp(0, n - 1))).clamp_min(0)
        mined = has_p & content & valid & (depth >= 1)
        nz = torch.nonzero(mined, as_tuple=False)

    if nz.numel() > 0:
        with torch.no_grad():
            b_idx = nz[:, 0]
            i_idx = nz[:, 1]
            dcap = float(cfg["depth_cap"])
            w_sel = (float(cfg["base_w"])
                     + float(cfg["bonus_w"]) * depth[b_idx, i_idx].to(dt).clamp(max=dcap))
            capn = int(cfg["max_examples"])
            if b_idx.numel() > capn:
                jitter = torch.rand_like(w_sel) * 1e-3
                _, order = torch.topk(w_sel + jitter, capn)
                b_idx = b_idx[order]
                i_idx = i_idx[order]
                w_sel = w_sel[order]
            wn = (w_sel / w_sel.sum().clamp_min(_EPS)).to(dt)
            cand_mask = content[b_idx] & (idxn.view(1, n) < i_idx.view(-1, 1))
            pos_mask = dup[b_idx, i_idx] & cand_mask
            cand_sq = osq[b_idx]

        cand = o[b_idx]
        r_sel = reads[b_idx, i_idx]
        tgt_o = o[b_idx, i_idx]

        if float(cfg["nce_weight"]) > 0.0:
            d2r = (cand_sq + r_sel.pow(2).sum(-1, keepdim=True)
                   - 2.0 * torch.bmm(cand, r_sel.unsqueeze(2)).squeeze(2)).clamp_min(0.0) / float(Dm)
            lg = (-d2r / max(1e-3, float(cfg["nce_temp"]))).masked_fill(~cand_mask, _NEG)
            logp = torch.log_softmax(lg, dim=1)
            nce = -torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)
            total = total + float(cfg["nce_weight"]) * (wn * nce).sum()

        if float(cfg["origin_weight"]) > 0.0:
            e_sel = origin[b_idx, i_idx]
            p_pos = (e_sel * pos_mask.to(dt)).sum(dim=1).clamp(1e-6, 1.0)
            total = total + float(cfg["origin_weight"]) * (wn * (-p_pos.log())).sum()

        if float(cfg["mse_weight"]) > 0.0:
            total = total + float(cfg["mse_weight"]) * (
                wn * (r_sel - tgt_o).pow(2).mean(dim=-1)).sum()

        if float(cfg["ent_weight"]) > 0.0:
            arow = attn[b_idx, i_idx]
            ent = -(arow * (arow + 1e-6).log()).sum(dim=1)
            total = total + float(cfg["ent_weight"]) * (wn * ent).sum()

    if float(cfg["gate_weight"]) > 0.0:
        vf = valid.to(dt)
        bce = F.binary_cross_entropy_with_logits(
            mv_logit[:, :n], mut.to(dt), reduction="none")
        total = total + float(cfg["gate_weight"]) * (bce * vf).sum() / vf.sum().clamp_min(1.0)

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


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
        vals["id_scale"] > 0.0,
        vals["role_scale"] > 0.0,
        vals["rec_scale"] > 0.0,
        abs(vals["wo_init"]) <= 1.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["depth_cap"] >= 1.0,
        vals["base_w"] >= 0.0,
        vals["bonus_w"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["nce_weight"] >= 0.0,
        vals["origin_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["gate_weight"] >= 0.0,
        vals["ent_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
