import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r5_chainhop_hindsight_transport"
DESCRIPTION = (
    "A dual-address move-transport memory over the command stream with per-hop hindsight "
    "supervision. Each command embedding is mapped by two extractors through one shared "
    "path-to-key encoder into a source address and a destination address; a delta-rule "
    "outer-product store is rolled forward causally, reading at the source address, writing a "
    "gated blend of that read and the step's observation at the destination address, and "
    "subtracting a move-gated erase at the source address, so a move relocates content instead "
    "of duplicating it. The rollout covers EVERY command position, including a trailing command "
    "that has no following observation: the read at a position is taken strictly before that "
    "position's write, so the injected memory read exists and is unchanged whether or not the "
    "sequence is truncated after the command. The read is injected into the arch's prediction "
    "through a per-dimension scale initialised to zero, so the wrapped net is the unwrapped net "
    "at initialisation. Train-time aux, label-free from the batch alone: steps whose observation "
    "duplicates an earlier step's observation under a different command with content-free steps "
    "in between are mined as (exposure, re-read) pairs; besides the endpoint InfoNCE on the read "
    "and on the prediction, a multiple-instance term takes the content-free steps lying between "
    "the exposure and the re-read, scores for each of them the store's value AT ITS OWN "
    "DESTINATION ADDRESS against the other observations seen so far, and trains the "
    "softly-selected best of them to hold the exposed content. A role term drives the two "
    "addresses of a contentful step together and the two addresses of a content-free step apart."
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
    "hop_weight": 0.6,
    "hop_hops": 8,
    "hop_examples": 96,
    "hop_temp": 0.25,
    "role_weight": 0.15,
    "role_margin": 0.4,
    "aux_weight": 1.0,
    "ramp_steps": 300,
    "sym_init_noise": 0.02,
    "erase_bias": -2.0,
}

_EPS = 1e-8
_NEG = -1e9


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _pad_time(x, n):
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


class _ChainHopTransport(nn.Module):
    def __init__(self, d_model, key_d, hid, sym_init_noise, erase_bias):
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
        self.erase_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.move_gate.bias, 0.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.erase_gate.bias, float(erase_bias))
        nn.init.constant_(self.read_gate.bias, 0.0)

        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def rollout(self, tok, types, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None, None, None, None
        B, L, _ = tok.shape
        n_cmd = (L + 1) // 2
        if n_cmd < 1:
            return None, None, None, None
        dev = tok.device
        dt = tok.dtype

        c = torch.nan_to_num(tok[:, 0::2, :], nan=0.0, posinf=1e4, neginf=-1e4)
        o = _pad_time(torch.nan_to_num(tok[:, 1::2, :], nan=0.0, posinf=1e4, neginf=-1e4), n_cmd)

        if key_pad is None:
            vc = torch.ones(B, n_cmd, dtype=torch.bool, device=dev)
            vo = vc
        else:
            v = ~key_pad.bool()
            vc = v[:, 0::2]
            vo = _pad_time(v[:, 1::2].unsqueeze(-1), n_cmd).squeeze(-1)
        active = (vc & vo).to(dt).unsqueeze(-1)

        cf = F.gelu(self.cmd_ctx(c))
        of = F.gelu(self.obs_ctx(o))
        gf = F.gelu(self.gate_mix(torch.cat([cf, of], dim=-1)))
        move = torch.sigmoid(self.move_gate(gf))
        beta = torch.sigmoid(self.write_gate(gf))
        erase = torch.sigmoid(self.erase_gate(gf))
        rgate = torch.sigmoid(self.read_gate(cf))

        ks = _unit(self.path_key(F.gelu(self.src_ext(c))))
        kd = _unit(self.path_key(F.gelu(self.dst_ext(c))))
        cs = (ks * kd).sum(dim=-1, keepdim=True)

        mem = tok.new_zeros(B, self.key_d, self.d_model)
        reads = []
        dsts = []
        for i in range(n_cmd):
            ksi = ks[:, i, :].unsqueeze(1)
            kdi = kd[:, i, :].unsqueeze(1)
            r_i = torch.bmm(ksi, mem).squeeze(1)
            reads.append(r_i)
            cur_d = torch.bmm(kdi, mem).squeeze(1)
            m_i = move[:, i, :]
            a_i = active[:, i, :]
            v_i = m_i * r_i + (1.0 - m_i) * o[:, i, :]
            w_i = (v_i - cur_d) * beta[:, i, :] * a_i
            w_i = torch.nan_to_num(w_i, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            e_i = m_i * erase[:, i, :] * r_i * a_i
            e_i = torch.nan_to_num(e_i, nan=0.0, posinf=1e3, neginf=-1e3).clamp(-1e3, 1e3)
            k2 = torch.cat([kdi, ksi], dim=1)
            w2 = torch.cat([w_i.unsqueeze(1), (-e_i).unsqueeze(1)], dim=1)
            mem = torch.baddbmm(mem, k2.transpose(1, 2), w2)
            dsts.append(cur_d + w_i - cs[:, i, :] * e_i)

        reads_t = torch.nan_to_num(torch.stack(reads, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        dsts_t = torch.nan_to_num(torch.stack(dsts, dim=1), nan=0.0, posinf=1e4, neginf=-1e4)
        contrib = rgate * (reads_t * self.out_scale.view(1, 1, -1)) * vc.unsqueeze(-1).to(dt)
        contrib = torch.nan_to_num(contrib, nan=0.0, posinf=1e4, neginf=-1e4)
        return reads_t, dsts_t, contrib, cs


def wrap(net, D, **params):
    if getattr(net, "_chainhop_transport_state", None) is not None:
        return net._chainhop_transport_state

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ChainHopTransport(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                             float(cfg["sym_init_noise"]), float(cfg["erase_bias"]))
    net.chainhop_transport = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 1:
            return pred, h
        reads_t, dsts_t, contrib, cs = mod.rollout(tok_emb, types, key_pad)
        if contrib is None:
            return pred, h
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        n = min(contrib.size(1), cmd_view.size(1))
        if n < 1:
            return pred, h
        cmd_view[:, :n, :] = cmd_view[:, :n, :] + contrib[:, :n, :]
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "reads": reads_t, "dsts": dsts_t,
                              "cs": cs, "pred": out}
        return out, h

    net.forward = _forward
    net._chainhop_transport_state = state
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
        dsts_t = stash["dsts"]
        cs_t = stash["cs"]
        pred_cmd = stash["pred"][:, 0::2, :][:, :n, :]
    else:
        reads_t, dsts_t, _, cs_t = mod.rollout(tok, batch["types"], batch["key_pad"])
        pred_cmd = None
    if reads_t is None:
        return 0.0
    if reads_t.size(1) < n:
        return 0.0
    reads_t = reads_t[:, :n, :].float()
    dsts_t = dsts_t[:, :n, :].float()
    cs_t = cs_t[:, :n, 0].float()

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
        have = bool(nz.numel() > 0)

        cont_w = (contentful & valid).to(cs_t.dtype)
        mut_w = (mutation & valid).to(cs_t.dtype)

        if have:
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

            E = int(b_idx.numel())
            i_sel = last_i[b_idx, j_idx].clamp_min(0)
            rowi = idxn.view(1, n).expand(E, n)
            between = ((rowi > i_sel.view(-1, 1)) & (rowi < j_idx.view(-1, 1))
                       & mutation[b_idx] & valid[b_idx])
            H = max(1, min(int(cfg["hop_hops"]), n))
            neg_one = torch.full((E, n), -1.0, device=dev, dtype=torch.float32)
            hop_scores = torch.where(between, rowi.to(torch.float32), neg_one)
            hop_vals = torch.topk(hop_scores, H, dim=1).values
            hop_valid = hop_vals >= 0.0
            hop_idx = hop_vals.clamp_min(0.0).long()

    role = ((1.0 - cs_t) * cont_w).sum() / cont_w.sum().clamp_min(1.0)
    role = role + ((F.relu(cs_t - float(cfg["role_margin"])) * mut_w).sum()
                   / mut_w.sum().clamp_min(1.0))
    total = float(cfg["role_weight"]) * role

    if have:
        cand = o[b_idx]
        tgt_o = o[b_idx, j_idx]
        temp = max(1e-3, float(cfg["cand_temp"]))

        def _member_logp(q3, cm, pm, csq3, cnd):
            dot = torch.bmm(q3, cnd.transpose(1, 2))
            d2 = (q3.pow(2).sum(-1, keepdim=True) + csq3.unsqueeze(1)
                  - 2.0 * dot).clamp_min(0.0) / Dm
            logits = (-d2 / temp).masked_fill(~cm.unsqueeze(1), _NEG)
            logp = torch.log_softmax(logits, dim=2)
            return torch.logsumexp(logp.masked_fill(~pm.unsqueeze(1), _NEG), dim=2)

        def _nce(q):
            return -(_member_logp(q.unsqueeze(1), cand_mask, pos_mask, cand_sq, cand)).mean()

        r_sel = reads_t[b_idx, j_idx]
        total = total + float(cfg["read_weight"]) * _nce(r_sel)
        total = total + float(cfg["mse_weight"]) * (r_sel - tgt_o).pow(2).mean()
        if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
            total = total + float(cfg["pred_weight"]) * _nce(pred_cmd[b_idx, j_idx].float())

        if float(cfg["hop_weight"]) > 0.0 and bool(hop_valid.any().item()):
            E2 = min(E, max(1, int(cfg["hop_examples"])))
            hb = b_idx[:E2]
            hj = hop_idx[:E2]
            hv = hop_valid[:E2]
            qh = dsts_t[hb.view(-1, 1).expand(-1, H), hj]
            mlp = _member_logp(qh, cand_mask[:E2], pos_mask[:E2], cand_sq[:E2], cand[:E2])
            mlp = torch.where(hv, mlp, torch.full_like(mlp, _NEG))
            wts = torch.softmax(mlp.detach() / max(1e-3, float(cfg["hop_temp"])), dim=1)
            wts = wts * hv.to(wts.dtype)
            wts = wts / wts.sum(dim=1, keepdim=True).clamp_min(1e-6)
            safe = torch.where(hv, mlp, torch.zeros_like(mlp))
            per_row = -(wts * safe).sum(dim=1)
            rows = hv.any(dim=1).to(per_row.dtype)
            total = total + float(cfg["hop_weight"]) * (
                (per_row * rows).sum() / rows.sum().clamp_min(1.0))

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
        vals["hop_weight"] >= 0.0,
        vals["hop_hops"] >= 1.0,
        vals["hop_examples"] >= 1.0,
        vals["hop_temp"] > 0.0,
        vals["role_weight"] >= 0.0,
        -1.0 <= vals["role_margin"] <= 1.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
        vals["sym_init_noise"] >= 0.0,
    ]
    return all(checks)
