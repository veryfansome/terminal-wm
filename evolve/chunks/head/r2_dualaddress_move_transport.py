import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r2_dualaddress_move_transport"
DESCRIPTION = (
    "Adds a dual-address transport memory on top of the arch's own prediction: every command "
    "embedding is mapped through two extractors and one SHARED path-to-key encoder into a source "
    "key and a destination key, and a delta-rule outer-product memory is updated causally by "
    "writing, at the destination key, a gated blend of the value currently read at the source key "
    "and the step's observation. The memory read at each command position is injected into the "
    "arch's prediction through a per-dimension scale initialised to zero, so the wrapped net is "
    "the unwrapped net at initialisation. Train-time aux mines, label-free and from the batch "
    "alone, steps whose observation exactly duplicates an earlier step's observation under a "
    "different command with at least one content-free step in between, and trains the memory read "
    "and the final prediction with a within-sequence squared-L2 InfoNCE against the other "
    "observations seen so far, plus a small MSE anchor."
)

_DEFAULTS = {
    "key_d": 48,
    "hid": 256,
    "cand_temp": 0.5,
    "dup_frac": 0.05,
    "cmd_dup_frac": 0.01,
    "max_dup": 3,
    "max_examples": 192,
    "read_weight": 1.0,
    "pred_weight": 0.5,
    "mse_weight": 0.05,
    "aux_weight": 1.0,
    "ramp_steps": 300,
    "sym_init_noise": 0.02,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


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


class _DualAddressTransport(nn.Module):
    def __init__(self, d_model, key_d, hid, sym_init_noise):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = int(key_d)
        h = int(hid)

        self.src_ext = nn.Linear(self.d_model, h)
        self.dst_ext = nn.Linear(self.d_model, h)
        with torch.no_grad():
            noise = torch.randn_like(self.src_ext.weight) * float(sym_init_noise)
            self.dst_ext.weight.copy_(self.src_ext.weight + noise)
            self.dst_ext.bias.copy_(self.src_ext.bias)

        self.path_key = nn.Linear(h, self.key_d, bias=False)

        self.cmd_ctx = nn.Linear(self.d_model, h)
        self.obs_ctx = nn.Linear(self.d_model, h)
        self.gate_mix = nn.Linear(2 * h, h)
        self.move_gate = nn.Linear(h, 1)
        self.write_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.read_gate.bias, 0.0)

        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def run(self, tok, types, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None, None
        B, L, Dm = tok.shape
        n = L // 2
        if n < 1:
            return None, None
        dev = tok.device
        dt = tok.dtype

        c = torch.nan_to_num(tok[:, 0::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4)

        if key_pad is None:
            vc = torch.ones(B, n, dtype=torch.bool, device=dev)
            vo = vc
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2][:, :n]
            vo = v[:, 1::2][:, :n]
        active = (vc & vo).to(dt).unsqueeze(-1)

        cf = F.gelu(self.cmd_ctx(c))
        of = F.gelu(self.obs_ctx(o))
        gf = F.gelu(self.gate_mix(torch.cat([cf, of], dim=-1)))
        move = torch.sigmoid(self.move_gate(gf))
        beta = torch.sigmoid(self.write_gate(gf))
        rgate = torch.sigmoid(self.read_gate(cf))

        ks = _unit(self.path_key(F.gelu(self.src_ext(c))))
        kd = _unit(self.path_key(F.gelu(self.dst_ext(c))))

        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        for i in range(n):
            ksi = ks[:, i, :].unsqueeze(1)
            kdi = kd[:, i, :].unsqueeze(1)
            r_i = torch.bmm(ksi, mem).squeeze(1)
            reads.append(r_i)
            cur_d = torch.bmm(kdi, mem).squeeze(1)
            m_i = move[:, i, :]
            v_i = m_i * r_i + (1.0 - m_i) * o[:, i, :]
            w_i = (v_i - cur_d) * beta[:, i, :] * active[:, i, :]
            w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            mem = torch.baddbmm(mem, kdi.transpose(1, 2), w_i.unsqueeze(1))

        reads_t = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        contrib = rgate * (reads_t * self.out_scale.view(1, 1, -1))
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)
        return reads_t, contrib


def wrap(net, D, **params):
    if getattr(net, "_dualaddr_transport_state", None) is not None:
        return net._dualaddr_transport_state

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _DualAddressTransport(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                                float(cfg["sym_init_noise"]))
    net.dualaddr_transport = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 2:
            return pred, h
        reads_t, contrib = mod.run(tok_emb, types, key_pad)
        if contrib is None:
            return pred, h
        n = contrib.size(1)
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        cmd_view[:, :n, :] = cmd_view[:, :n, :] + contrib
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads_t, "pred": out}
        return out, h

    net.forward = _forward
    net._dualaddr_transport_state = state
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
        pred_cmd = stash["pred"][:, 0::2, :][:, :n, :]
    else:
        reads_t, _ = mod.run(tok, batch["types"], batch["key_pad"])
        pred_cmd = None
    if reads_t is None:
        return 0.0
    reads_t = reads_t[:, :n, :].float()

    c = torch.nan_to_num(tok[:, 0::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    o = torch.nan_to_num(tok[:, 1::2, :][:, :n, :], nan=0.0, posinf=1e4, neginf=-1e4).float()
    Dm = float(o.size(-1))
    dev = o.device

    max_dup = int(cfg["max_dup"])
    mut_min = max_dup + 1

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
        contentful = valid & (clus <= max_dup)
        mutation = valid & (clus >= mut_min)
        mcs = torch.cumsum(mutation.long(), dim=1)

        pair = dup_o & (~dup_c) & contentful.unsqueeze(1) & contentful.unsqueeze(2)
        tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1).unsqueeze(0)
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
        dep = depth[b_idx, j_idx].float()
        cap = int(cfg["max_examples"])
        if b_idx.numel() > cap:
            _, order = torch.topk(dep, cap)
            b_idx = b_idx[order]
            j_idx = j_idx[order]

        seen = idxn.view(1, n) <= j_idx.view(-1, 1)
        cand_mask = contentful[b_idx] & seen
        pos_mask = (dup_o[b_idx, j_idx] | F.one_hot(j_idx, n).bool()) & cand_mask
        cand_sq = osq[b_idx]

    cand = o[b_idx]
    tgt_o = o[b_idx, j_idx]
    temp = max(1e-3, float(cfg["cand_temp"]))

    def _nce(q):
        dot = torch.bmm(cand, q.unsqueeze(2)).squeeze(2)
        d2 = (q.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / Dm
        logits = (-d2 / temp).masked_fill(~cand_mask, _NEG)
        logp = torch.log_softmax(logits, dim=1)
        return -(torch.logsumexp(logp.masked_fill(~pos_mask, _NEG), dim=1)).mean()

    r_sel = reads_t[b_idx, j_idx]
    total = float(cfg["read_weight"]) * _nce(r_sel)
    total = total + float(cfg["mse_weight"]) * (r_sel - tgt_o).pow(2).mean()
    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * _nce(pred_cmd[b_idx, j_idx].float())

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
        vals["key_d"] >= 1.0,
        vals["hid"] >= 8.0,
        vals["cand_temp"] > 0.0,
        0.0 < vals["dup_frac"] < 1.0,
        0.0 < vals["cmd_dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["sym_init_noise"] >= 0.0,
    ]
    return all(checks)
