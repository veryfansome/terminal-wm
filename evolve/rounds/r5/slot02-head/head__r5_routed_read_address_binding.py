import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r5_routed_read_address_binding"
DESCRIPTION = (
    "A head-owned dual-address transport store plus a differentiable pointer-chase loss on its "
    "own addresses. Every command embedding is mapped by one shared GELU layer to a unit SOURCE "
    "address and a unit DESTINATION address; a causal delta-rule store reads at the source, "
    "writes at the destination either the step's observation or a command-conditioned affine edit "
    "(FiLM gamma/beta) of the value just read, and the read is injected into the arch's "
    "command-position prediction through a zero-init (D,D) projection, so the wrapped net is the "
    "unwrapped net at initialisation. Train-time aux mines, from batch embeddings only, steps "
    "whose observation duplicates an earlier step's observation under a different command with at "
    "least one content-free step in between. On each mined pair it runs a soft state-tracking "
    "recurrence over the intervening commands in ADDRESS space -- start at the earlier step's "
    "destination address, at each intervening command open a sigmoid gate by the match between "
    "that command's source address and the tracked address and slew the tracked address toward "
    "that command's destination address -- and trains the endpoint by cross-entropy against every "
    "in-row address, plus the mirror recurrence run backwards. The same mined steps carry a "
    "within-sequence squared-L2 InfoNCE on the store's read and on the arch's final prediction, a "
    "small MSE anchor, and a source/destination agreement term on the contentful steps."
)

_EPS = 1e-8
_NEG = -1e9

_DEFAULTS = {
    "key_d": 64,
    "addr_hidden": 256,
    "gate_hidden": 128,
    "edit_gscale": 0.5,
    "decay": 0.999,
    "dst_init_noise": 0.02,
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "max_examples": 192,
    "cand_temp": 0.5,
    "addr_temp": 0.10,
    "gate_thr": 0.30,
    "gate_temp": 0.25,
    "pred_weight": 1.0,
    "read_weight": 0.5,
    "mse_weight": 0.05,
    "addr_weight": 0.5,
    "self_addr_weight": 0.10,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _pad_to(x, n):
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


class _AddressTransportStore(nn.Module):
    def __init__(self, d_model, key_d, addr_hidden, gate_hidden, edit_gscale, dst_init_noise,
                 decay):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(8, int(key_d))
        self.edit_gscale = float(edit_gscale)
        self.decay = float(decay)
        ah = max(32, int(addr_hidden))
        gh = max(16, int(gate_hidden))

        self.addr_norm = nn.LayerNorm(self.d_model)
        self.addr_in = nn.Linear(self.d_model, ah)
        self.addr_src = nn.Linear(ah, self.key_d, bias=False)
        self.addr_dst = nn.Linear(ah, self.key_d, bias=False)
        with torch.no_grad():
            noise = torch.randn_like(self.addr_src.weight) * float(dst_init_noise)
            self.addr_dst.weight.copy_(self.addr_src.weight + noise)

        self.cmd_feat = nn.Linear(ah, gh)
        self.obs_feat = nn.Linear(self.d_model, gh)
        self.move_gate = nn.Linear(2 * gh, 1)
        self.store_gate = nn.Linear(2 * gh, 1)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.store_gate.bias, 1.0)

        self.edit_in = nn.Linear(ah, gh)
        self.edit_out = nn.Linear(gh, 2 * self.d_model)
        nn.init.zeros_(self.edit_out.weight)
        nn.init.zeros_(self.edit_out.bias)

        self.read_gate = nn.Linear(ah, 1)
        nn.init.constant_(self.read_gate.bias, 0.0)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def addresses(self, cmd_raw):
        feat = F.gelu(self.addr_in(self.addr_norm(cmd_raw)))
        return feat, _unit(self.addr_src(feat)), _unit(self.addr_dst(feat))

    def run(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        if n_cmd < 1 or n_pair < 1:
            return None
        dev = tok.device

        c = torch.nan_to_num(tok[:, 0::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = torch.nan_to_num(tok[:, 1::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        if key_pad is None:
            vc = torch.ones(B, n_cmd, dtype=torch.bool, device=dev)
            vo = torch.ones(B, n_pair, dtype=torch.bool, device=dev)
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2]
            vo = v[:, 1::2]

        feat, a_src, a_dst = self.addresses(c)
        o_pad = _pad_to(o, n_cmd)
        live = _pad_to((vc[:, :n_pair] & vo).to(tok.dtype).unsqueeze(-1), n_cmd)

        cf = F.gelu(self.cmd_feat(feat))
        of = F.gelu(self.obs_feat(o_pad))
        gmix = torch.cat([cf, of], dim=-1)
        move = torch.sigmoid(self.move_gate(gmix))
        store = torch.sigmoid(self.store_gate(gmix))

        eb = self.edit_out(F.gelu(self.edit_in(feat)))
        gamma = torch.tanh(eb[..., :self.d_model]) * self.edit_gscale
        shift = eb[..., self.d_model:]

        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        for i in range(n_cmd):
            ai = a_src[:, i, :]
            di = a_dst[:, i, :]
            s_i = torch.bmm(ai.unsqueeze(1), mem).squeeze(1)
            reads.append(s_i)
            edited = s_i * (1.0 + gamma[:, i, :]) + shift[:, i, :]
            m_i = move[:, i, :]
            v_i = m_i * edited + (1.0 - m_i) * o_pad[:, i, :]
            cur = torch.bmm(di.unsqueeze(1), mem).squeeze(1)
            w_i = (v_i - cur) * store[:, i, :] * live[:, i, :]
            w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            mem = torch.baddbmm(mem, di.unsqueeze(2), w_i.unsqueeze(1), beta=self.decay)
            mem = torch.nan_to_num(mem, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        reads_t = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        contrib = torch.sigmoid(self.read_gate(feat)) * self.out_proj(reads_t)
        contrib = contrib * vc.unsqueeze(-1).to(contrib.dtype)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"reads": reads_t, "contrib": contrib, "a_src": a_src, "a_dst": a_dst}


def wrap(net, D, **params):
    existing = getattr(net, "_routed_addr_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params or {})

    mod = _AddressTransportStore(
        int(D), int(cfg["key_d"]), int(cfg["addr_hidden"]), int(cfg["gate_hidden"]),
        float(cfg["edit_gscale"]), float(cfg["dst_init_noise"]), float(cfg["decay"]))
    net.routed_addr_store = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        out = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        pred, hidden = out[0], out[1]
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 2:
            return pred, hidden
        res = mod.run(tok_emb, key_pad)
        if res is None:
            return pred, hidden
        contrib = res["contrib"]
        new_pred = pred.clone()
        cmd_view = new_pred[:, 0::2, :]
        m = min(cmd_view.size(1), contrib.size(1))
        cmd_view[:, :m, :] = cmd_view[:, :m, :] + contrib[:, :m, :].to(new_pred.dtype)
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": res["reads"], "pred": new_pred,
                              "a_src": res["a_src"], "a_dst": res["a_dst"]}
        return new_pred, hidden

    net.forward = _forward
    net._routed_addr_state = state
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

    if stash is not None and stash["tok"] is tok:
        reads_t = stash["reads"]
        pred_cmd = stash["pred"][:, 0::2, :]
        a_src = stash["a_src"]
        a_dst = stash["a_dst"]
    else:
        res = mod.run(tok, batch["key_pad"])
        if res is None:
            return 0.0
        reads_t = res["reads"]
        pred_cmd = None
        a_src = res["a_src"]
        a_dst = res["a_dst"]

    reads_t = _pad_to(reads_t, n)
    a_src = _pad_to(a_src, n)
    a_dst = _pad_to(a_dst, n)
    if pred_cmd is not None:
        pred_cmd = _pad_to(pred_cmd, n)

    c = torch.nan_to_num(tok[:, 0::2, :], nan=0.0, posinf=1e4, neginf=-1e4)[:, :n, :].float()
    o = torch.nan_to_num(tok[:, 1::2, :], nan=0.0, posinf=1e4, neginf=-1e4)[:, :n, :].float()
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
        ref_o = (d2o * off).sum(dim=(1, 2)) / cnt
        ref_c = (d2c * off).sum(dim=(1, 2)) / cnt

        dup_o = off & (d2o <= (float(cfg["dup_frac"]) * ref_o).view(B, 1, 1))
        dup_c = off & (d2c <= (float(cfg["cmd_dup_frac"]) * ref_c).view(B, 1, 1))

        clus = dup_o.sum(dim=2)
        max_dup = int(cfg["max_dup"])
        contentful = valid & (clus <= max_dup)
        mutation = valid & (clus > max_dup)
        mcs = torch.cumsum(mutation.long(), dim=1)

        idxn = torch.arange(n, device=dev)
        tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
        earlier = (dup_o & (~dup_c) & contentful.unsqueeze(2) & contentful.unsqueeze(1) & tri)

        miss = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)
        last_i = torch.where(earlier, idxn.view(1, 1, n), miss).amax(dim=2)
        has_prev = last_i >= 0

        jm1 = (idxn.view(1, n) - 1).clamp_min(0).expand(B, n)
        mc_j = torch.gather(mcs, 1, jm1)
        mc_i = torch.gather(mcs, 1, last_i.clamp_min(0))
        n_mut = (mc_j - mc_i).clamp_min(0)

        mined = has_prev & contentful & valid & (n_mut >= 1)
        nz = torch.nonzero(mined, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        b_idx = nz[:, 0]
        j_idx = nz[:, 1]
        i_idx = last_i[b_idx, j_idx]
        wt = n_mut[b_idx, j_idx].to(o.dtype)

        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(wt, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]
            i_idx = i_idx[order]
            wt = wt[order]

        w = wt / wt.sum().clamp_min(_EPS)
        E = b_idx.numel()
        seen = idxn.view(1, n) <= j_idx.view(-1, 1)
        cand_mask = contentful[b_idx] & seen
        pos_mask = (dup_o[b_idx, j_idx] | F.one_hot(j_idx, n).bool()) & cand_mask
        cand_sq = osq[b_idx]
        between = (valid[b_idx]
                   & (idxn.view(1, n) > i_idx.view(-1, 1))
                   & (idxn.view(1, n) < j_idx.view(-1, 1)))
        valid_e = valid[b_idx]

    cand = o[b_idx]
    temp = max(1e-3, float(cfg["cand_temp"]))

    def _nce(q):
        qf = q.float()
        dot = torch.bmm(cand, qf.unsqueeze(2)).squeeze(2)
        d2 = (qf.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / Dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return -(w * torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)).sum()

    r_sel = reads_t[b_idx, j_idx].float()
    tgt_o = o[b_idx, j_idx]
    total = float(cfg["read_weight"]) * _nce(r_sel)
    total = total + float(cfg["mse_weight"]) * (w * (r_sel - tgt_o).pow(2).mean(dim=-1)).sum()
    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * _nce(pred_cmd[b_idx, j_idx])

    addr_w = float(cfg["addr_weight"])
    if addr_w > 0.0:
        at = max(1e-3, float(cfg["addr_temp"]))
        gt = max(1e-3, float(cfg["gate_temp"]))
        thr = float(cfg["gate_thr"])
        A_src = a_src[b_idx].float()
        A_dst = a_dst[b_idx].float()
        ar = torch.arange(E, device=dev)

        fwd = A_dst[ar, i_idx]
        for t in range(n):
            act = between[:, t].to(fwd.dtype).unsqueeze(-1)
            gate = torch.sigmoid(
                ((A_src[:, t, :] * fwd).sum(-1, keepdim=True) - thr) / gt) * act
            fwd = _unit((1.0 - gate) * fwd + gate * A_dst[:, t, :])
        log_f = ((A_src * fwd.unsqueeze(1)).sum(-1) / at).masked_fill(~valid_e, _NEG)
        loss_fwd = F.cross_entropy(log_f, j_idx, reduction="none")

        bwd = A_src[ar, j_idx]
        for t in range(n - 1, -1, -1):
            act = between[:, t].to(bwd.dtype).unsqueeze(-1)
            gate = torch.sigmoid(
                ((A_dst[:, t, :] * bwd).sum(-1, keepdim=True) - thr) / gt) * act
            bwd = _unit((1.0 - gate) * bwd + gate * A_src[:, t, :])
        log_b = ((A_dst * bwd.unsqueeze(1)).sum(-1) / at).masked_fill(~valid_e, _NEG)
        loss_bwd = F.cross_entropy(log_b, i_idx, reduction="none")

        total = total + addr_w * (w * (loss_fwd + loss_bwd)).sum()

    self_w = float(cfg["self_addr_weight"])
    if self_w > 0.0:
        agree = (a_src * a_dst).sum(-1).float()
        cmf = contentful.to(agree.dtype)
        total = total + self_w * (((1.0 - agree) * cmf).sum() / cmf.sum().clamp_min(1.0))

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
        vals["key_d"] >= 8.0,
        vals["addr_hidden"] >= 32.0,
        vals["gate_hidden"] >= 16.0,
        vals["edit_gscale"] >= 0.0,
        0.0 < vals["decay"] <= 1.0,
        vals["dst_init_noise"] >= 0.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cand_temp"] > 0.0,
        vals["addr_temp"] > 0.0,
        -1.0 <= vals["gate_thr"] <= 1.0,
        vals["gate_temp"] > 0.0,
        vals["pred_weight"] >= 0.0,
        vals["read_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["addr_weight"] >= 0.0,
        vals["self_addr_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
