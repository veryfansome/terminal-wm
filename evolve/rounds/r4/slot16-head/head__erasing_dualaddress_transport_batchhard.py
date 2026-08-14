import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "erasing_dualaddress_transport_batchhard"
DESCRIPTION = (
    "A causal two-address transport memory added on top of the arch's own prediction, plus the "
    "transition-operator forward-model consistency aux. Every command embedding goes through one "
    "shared trunk into a SOURCE key and a DESTINATION key, where the destination key is the source "
    "key displaced by a zero-initialised command-conditioned shift in the same key space, so both "
    "addresses start identical and separate only as they are trained. At each command slot the "
    "memory is read at the source key and at the destination key in one batched matmul, and then "
    "updated by two simultaneous rank-one delta rules: the destination moves toward the value just "
    "read at the source, and the source moves toward a blend of this step's observation and a "
    "gated ERASURE of its own current value, so a location that has been moved away from stops "
    "returning what it used to hold. Every gate (move, erase, record, readout) is a function of the "
    "command embedding alone; the observation enters only as a write VALUE, scaled by a learned "
    "informativeness gate and by the observation slot being present, so a command slot whose "
    "observation is absent still reads. The read at a slot is taken strictly before that slot's "
    "write and is injected into the arch's prediction at every command slot through a "
    "zero-initialised per-dimension scale, so the wrapped net is the unwrapped net at "
    "initialisation. The train-time aux is label-free: it mines slots whose observation duplicates "
    "an earlier slot's observation under a different command with at least one content-free step "
    "between them, and applies a batch-hard triplet hinge with a soft-minimum over the earlier "
    "non-duplicate observations, so the memory read must be nearer the content actually present "
    "than to the nearest stale content, plus a small squared-error anchor; on archs exposing a "
    "shared transition operator it also requires f(obs_pre, cmd) to reconstruct the later "
    "post-mutation observation on same-path triples."
)

_DEFAULTS = {
    "key_d": 32,
    "hid": 192,
    "move_bias": 0.0,
    "erase_bias": -2.0,
    "rec_bias": 1.0,
    "out_gate_bias": 0.0,
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "max_examples": 192,
    "margin_frac": 0.25,
    "tau_frac": 0.10,
    "triplet_weight": 0.5,
    "anchor_weight": 0.05,
    "fm_row_frac": 0.6,
    "fm_path_thresh": 0.60,
    "fm_change_floor": 0.25,
    "fm_max_examples": 512,
    "fm_cos_weight": 0.10,
    "fm_mse_weight": 0.02,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


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
    if tok.shape[2] != tgt.shape[2] or tok.shape[1] != 2 * tgt.shape[1]:
        return False
    live = ~key_pad.bool()
    if not bool(live.any().item()):
        return False
    even = types[:, 0::2][live[:, 0::2]]
    odd = types[:, 1::2][live[:, 1::2]]
    if even.numel() == 0 or odd.numel() == 0:
        return False
    return bool((even == 0).all().item()) and bool((odd == 1).all().item())


class _ErasingDualAddressTransport(nn.Module):
    def __init__(self, d_model, key_d, hid, move_bias, erase_bias, rec_bias, out_gate_bias):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(1, int(key_d))
        h = max(8, int(hid))

        self.arg_trunk = nn.Linear(self.d_model, h)
        self.src_ext = nn.Linear(h, h)
        self.dst_shift = nn.Linear(h, h)
        nn.init.zeros_(self.dst_shift.weight)
        nn.init.zeros_(self.dst_shift.bias)
        self.path_key = nn.Linear(h, self.key_d, bias=False)

        self.obs_ctx = nn.Linear(self.d_model, h)
        self.info_gate = nn.Linear(h, 1)
        self.move_gate = nn.Linear(h, 1)
        self.erase_gate = nn.Linear(h, 1)
        self.rec_gate = nn.Linear(h, 1)
        self.out_gate = nn.Linear(h, 1)
        nn.init.constant_(self.info_gate.bias, 0.0)
        nn.init.constant_(self.move_gate.bias, float(move_bias))
        nn.init.constant_(self.erase_gate.bias, float(erase_bias))
        nn.init.constant_(self.rec_gate.bias, float(rec_bias))
        nn.init.constant_(self.out_gate.bias, float(out_gate_bias))

        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def run(self, tok, types, key_pad):
        if tok is None or tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None, None
        B, L, _ = tok.shape
        if L < 1:
            return None, None
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        dev = tok.device
        dt = tok.dtype

        c = _clean(tok[:, 0::2, :])
        o = _clean(tok[:, 1::2, :])

        if key_pad is None:
            vc = torch.ones(B, n_cmd, dtype=torch.bool, device=dev)
            vo = torch.ones(B, n_pair, dtype=torch.bool, device=dev)
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2]
            vo = v[:, 1::2]
        if types is None:
            imagined = torch.zeros(B, n_cmd, dtype=torch.bool, device=dev)
        else:
            imagined = types.long()[:, 0::2] == 2

        f = F.gelu(self.arg_trunk(c))
        u = F.gelu(self.src_ext(f))
        ka = _unit(self.path_key(u))
        kb = _unit(self.path_key(u + self.dst_shift(f)))
        keys = torch.stack([ka, kb], dim=2)

        g_move = torch.sigmoid(self.move_gate(f))
        g_erase = torch.sigmoid(self.erase_gate(f))
        g_rec = torch.sigmoid(self.rec_gate(f))
        g_out = torch.sigmoid(self.out_gate(f))

        if n_pair > 0:
            obs_live = (vo & ~imagined[:, :n_pair]).to(dt).unsqueeze(-1)
            info = torch.sigmoid(self.info_gate(F.gelu(self.obs_ctx(o)))) * obs_live
        else:
            info = None

        act = vc.to(dt).unsqueeze(-1)
        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        for i in range(n_cmd):
            ki = keys[:, i, :, :]
            rd = torch.bmm(ki, mem)
            s_i = rd[:, 0, :]
            d_i = rd[:, 1, :]
            reads.append(s_i)

            if i < n_pair:
                o_i = o[:, i, :]
                rho = g_rec[:, i, :] * info[:, i, :]
            else:
                o_i = torch.zeros_like(s_i)
                rho = torch.zeros_like(g_rec[:, i, :])

            a_i = act[:, i, :]
            upd_dst = g_move[:, i, :] * (s_i - d_i) * a_i
            upd_src = (rho * (o_i - s_i) - (1.0 - rho) * g_erase[:, i, :] * s_i) * a_i
            upd = torch.stack([upd_src, upd_dst], dim=1)
            upd = _clean(upd).clamp(-1e3, 1e3)

            mem = torch.baddbmm(mem, ki.transpose(1, 2), upd).clamp(-1e4, 1e4)

        reads_t = _clean(torch.stack(reads, dim=1))
        contrib = g_out * (reads_t * self.out_scale.view(1, 1, -1)) * act
        return reads_t, _clean(contrib)


def wrap(net, D, **params):
    existing = getattr(net, "_erasing_transport_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ErasingDualAddressTransport(
        int(D), int(cfg["key_d"]), int(cfg["hid"]), float(cfg["move_bias"]),
        float(cfg["erase_bias"]), float(cfg["rec_bias"]), float(cfg["out_gate_bias"]),
    )
    net.erasing_transport = mod

    state = {
        "cfg": cfg,
        "mod": mod,
        "step": 0,
        "stash": None,
        "D": int(D),
        "fm_off": not callable(getattr(net, "transition_from_emb", None)),
    }

    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if not torch.is_tensor(pred) or pred.dim() != 3 or pred.size(-1) != state["D"]:
            return pred, h
        if tok_emb.dim() != 3 or pred.size(1) != tok_emb.size(1) or tok_emb.size(1) < 1:
            return pred, h
        reads_t, contrib = mod.run(tok_emb, types, key_pad)
        if contrib is None:
            return pred, h
        m = min(contrib.size(1), (pred.size(1) + 1) // 2)
        if m < 1:
            return pred, h
        add = torch.zeros_like(pred)
        add[:, 0:2 * m:2, :] = contrib[:, :m, :].to(pred.dtype)
        out = pred + add
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads_t}
        return out, h

    net.forward = _forward
    net._erasing_transport_state = state
    return state


def _transport_terms(cfg, mod, stash, batch):
    tok = batch["tok"]
    valid = batch["cmd_mask"].bool()
    B, n = valid.shape
    if n < 3:
        return 0.0

    if stash is not None and stash.get("tok") is tok:
        reads_t = stash["reads"]
    else:
        reads_t, _ = mod.run(tok, batch["types"], batch["key_pad"])
    if reads_t is None or reads_t.size(1) < n:
        return 0.0
    reads_t = reads_t[:, :n, :].float()

    c = _clean(tok[:, 0::2, :][:, :n, :]).float()
    o = _clean(tok[:, 1::2, :][:, :n, :]).float()
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
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        off = vv & (~eye)
        cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2o.dtype)
        ref_o = ((d2o * off).sum(dim=(1, 2)) / cnt).clamp_min(_EPS)
        ref_c = ((d2c * off).sum(dim=(1, 2)) / cnt).clamp_min(_EPS)

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        deg = dup_o.sum(dim=2)
        max_dup = int(cfg["max_dup"])
        contentful = valid & (deg <= max_dup)
        mutation = valid & (deg > max_dup)
        mcs = torch.cumsum(mutation.long(), dim=1)

        tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
        pair = dup_o & (~dup_c) & contentful.unsqueeze(1) & contentful.unsqueeze(2)
        earlier = pair & tri
        has_p = earlier.any(dim=2)

        idxn = torch.arange(n, device=dev)
        big = torch.full((1, 1, n), n, device=dev, dtype=torch.long)
        small = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)
        first_i = torch.where(earlier, idxn.view(1, 1, n), big).amin(dim=2)
        last_i = torch.where(earlier, idxn.view(1, 1, n), small).amax(dim=2)

        jm1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        mc_j = torch.gather(mcs, 1, jm1)
        mc_i = torch.gather(mcs, 1, last_i.clamp_min(0))
        mut_between = (mc_j - mc_i) > 0

        mined = has_p & contentful & mut_between & valid
        nz = torch.nonzero(mined, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        b_idx = nz[:, 0]
        j_idx = nz[:, 1]

        depth = (idxn.view(1, n) - first_i).clamp_min(0)
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(depth[b_idx, j_idx].float(), cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]

        seen = idxn.view(1, n) <= j_idx.view(-1, 1)
        self_hot = F.one_hot(j_idx, n).bool()
        neg_mask = contentful[b_idx] & seen & (~dup_o[b_idx, j_idx]) & (~self_hot)
        has_neg = neg_mask.any(dim=1)
        ref_sel = ref_o[b_idx]

    r_sel = reads_t[b_idx, j_idx]
    tgt_o = o[b_idx, j_idx]
    cand = o[b_idx]

    d_pos = (r_sel - tgt_o).pow(2).mean(dim=-1)
    d_all = (cand - r_sel.unsqueeze(1)).pow(2).mean(dim=-1)

    tau = (float(cfg["tau_frac"]) * ref_sel).clamp_min(1e-4)
    logits = (-d_all / tau.unsqueeze(1)).masked_fill(~neg_mask, _NEG)
    d_neg = -tau * torch.logsumexp(logits, dim=1)

    margin = float(cfg["margin_frac"]) * ref_sel
    hinge = F.relu(margin + d_pos - d_neg) * has_neg.to(d_pos.dtype)
    hinge = hinge.sum() / has_neg.to(d_pos.dtype).sum().clamp_min(1.0)

    return float(cfg["triplet_weight"]) * hinge + float(cfg["anchor_weight"]) * d_pos.mean()


def _fm_terms(cfg, net, batch, device):
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    cos_w = float(cfg["fm_cos_weight"])
    mse_w = float(cfg["fm_mse_weight"])
    if cos_w <= 0.0 and mse_w <= 0.0:
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["fm_row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    with torch.no_grad():
        R = cmd.size(0)
        cu = _unit(_clean(cmd))
        sim = torch.bmm(cu, cu.transpose(1, 2))
        vmask = valid.bool()

        pos = torch.arange(maxn, device=device)
        lower = pos.unsqueeze(1) > pos.unsqueeze(0)
        upper = pos.unsqueeze(1) < pos.unsqueeze(0)
        same_path = (sim > float(cfg["fm_path_thresh"])) & vmask.unsqueeze(1)

        before = same_path & lower.unsqueeze(0)
        after = same_path & upper.unsqueeze(0)
        posf = pos.view(1, 1, maxn).expand(R, maxn, maxn)
        i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)
        j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)

        triple_ok = (i_idx >= 0) & (j_idx < maxn) & vmask
        ic = i_idx.clamp(0, maxn - 1)
        jc = j_idx.clamp(0, maxn - 1)
        obs_i = torch.gather(obs, 1, ic.unsqueeze(-1).expand(R, maxn, obs.size(-1)))
        obs_j = torch.gather(obs, 1, jc.unsqueeze(-1).expand(R, maxn, obs.size(-1)))
        change = (obs_i - obs_j).pow(2).mean(dim=-1)
        w_all = (torch.gather(sim, 2, ic.unsqueeze(-1)).squeeze(-1)
                 * torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1) * change)

        keep = (triple_ok & (change >= float(cfg["fm_change_floor"]))
                & torch.isfinite(w_all) & (w_all > 0.0))
        nz = torch.nonzero(keep, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        r = nz[:, 0]
        k = nz[:, 1]
        w = w_all[r, k]
        ti = i_idx[r, k]
        tj = j_idx[r, k]
        fm_cap = int(cfg["fm_max_examples"])
        if r.numel() > fm_cap:
            w, order = torch.topk(w, fm_cap)
            r = r[order]
            k = k[order]
            ti = ti[order]
            tj = tj[order]
        w = (w.to(cmd.dtype) / w.to(cmd.dtype).sum().clamp_min(_EPS))

        pre = _clean(obs[r, ti])
        cmd_k = _clean(cmd[r, k])
        gold = _clean(obs[r, tj])

    pred = _clean(op(pre, cmd_k))
    cos_err = (w * (1.0 - (_unit(pred) * _unit(gold)).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - gold).pow(2).mean(dim=-1)).sum()
    return cos_w * cos_err + mse_w * mse_err


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

    total = _transport_terms(cfg, mod, stash, batch)
    if not st["fm_off"]:
        total = total + _fm_terms(cfg, net, batch, device)
    if not torch.is_tensor(total):
        return 0.0

    out = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out).item()):
        return 0.0
    return out.to(batch["tok"].dtype)


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
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["margin_frac"] >= 0.0,
        vals["tau_frac"] > 0.0,
        vals["triplet_weight"] >= 0.0,
        vals["anchor_weight"] >= 0.0,
        0.0 < vals["fm_row_frac"] <= 1.0,
        -1.0 <= vals["fm_path_thresh"] < 1.0,
        vals["fm_change_floor"] >= 0.0,
        vals["fm_max_examples"] >= 1.0,
        vals["fm_cos_weight"] >= 0.0,
        vals["fm_mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
