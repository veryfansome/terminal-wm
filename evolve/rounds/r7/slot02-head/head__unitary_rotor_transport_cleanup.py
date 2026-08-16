import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "unitary_rotor_transport_cleanup"
DESCRIPTION = (
    "A vector-symbolic (unitary-binding) board memory that transports content by an exactly "
    "invertible rotation, with an error-correcting cleanup after every hop. One D-dimensional "
    "memory vector S holds the whole board as a superposition sum_p bind(A_p, content_p). The "
    "768 dimensions are read as 384 planes; an address A is a per-plane rotation angle, so "
    "bind(A, v) is a block-diagonal orthogonal rotation and unbind is its exact transpose: "
    "unbind(A, bind(A, v)) = v to machine precision, and rotation preserves inner products, so "
    "crosstalk from other bound items stays decorrelated instead of biasing the read. Each "
    "command position emits, from the raw command embedding alone, three addresses and three "
    "gates: its own path address (generated from the command feature after the learned verb "
    "subspace is projected out, so 'cat P' and 'mv P Q' can agree on P), a source address, a "
    "destination address, a move amount, a write amount and a read gate. The step order is "
    "read, then move, then write, so the read at a command sees only strictly earlier commands "
    "and observations. A move is the rank-one algebraic edit "
    "S <- S + m * (bind(A_dst, v) - bind(A_src, v)) where v = cleanup(unbind(A_src, S)) and "
    "cleanup is a null-augmented softmax over the observation embeddings already seen in this "
    "window, scored by cosine. Because v is snapped onto an actually-observed content vector "
    "before it is subtracted, the erase at the source is exact rather than approximate, so a "
    "chain of moves does not accumulate residue with depth: the same content is re-quantised "
    "after every hop. A write is the gated addition S <- S + w * bind(A_own, obs). Both the raw "
    "unbound read and the cleaned-up read are injected additively into the arch's prediction at "
    "command positions through per-dimension scales. Train-time aux mines positions whose "
    "observation duplicates an earlier observation, down-weighting immediate repeats, and runs "
    "an InfoNCE in the eval's squared-L2 decision variable over the causally visible "
    "observations on the cleaned read, the raw read and the injected prediction; a secondary "
    "same-path forward-model term supervises the arch's shared transition operator when it "
    "exposes one. No auxiliary runs at eval."
)

_DEFAULTS = {
    "hid": 192,
    "n_verb": 8,
    "temp_init": 0.07,
    "null_init": 2.0,
    "clean_init": 0.02,
    "dup_frac": 0.05,
    "max_dup": 3,
    "max_examples": 256,
    "nce_temp": 0.5,
    "near_weight": 0.25,
    "read_weight": 1.0,
    "raw_weight": 0.25,
    "pred_weight": 0.5,
    "mse_weight": 0.05,
    "trans_weight": 0.5,
    "path_row_frac": 0.6,
    "path_thresh": 0.60,
    "path_change_floor": 0.25,
    "path_max_examples": 512,
    "path_cos_weight": 0.10,
    "path_mse_weight": 0.02,
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


class _RotorBoardMemory(nn.Module):
    def __init__(self, d_model, hid, n_verb, temp_init, null_init, clean_init):
        super().__init__()
        self.d_model = int(d_model)
        self.half = self.d_model // 2
        self.n_verb = max(1, int(n_verb))
        h = max(16, int(hid))
        self.feat = nn.Linear(self.d_model, h)
        self.feat_norm = nn.LayerNorm(h)
        self.verb_codebook = nn.Parameter(torch.randn(self.n_verb, h) * 0.2)
        self.path_angle = nn.Linear(h, self.half)
        self.src_angle = nn.Linear(h, self.half)
        self.dst_angle = nn.Linear(h, self.half)
        self.move_gate = nn.Linear(h, 1)
        self.write_gate = nn.Linear(h, 1)
        self.read_gate = nn.Linear(h, 1)
        for lin in (self.path_angle, self.src_angle, self.dst_angle):
            nn.init.uniform_(lin.bias, -1.5, 1.5)
        nn.init.constant_(self.move_gate.bias, -1.0)
        nn.init.constant_(self.write_gate.bias, 1.0)
        nn.init.constant_(self.read_gate.bias, 0.0)
        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.null_logit = nn.Parameter(torch.tensor(float(null_init)))
        self.out_raw = nn.Parameter(torch.zeros(self.d_model))
        self.out_clean = nn.Parameter(torch.full((self.d_model,), float(clean_init)))

    def _verb_basis(self):
        vs = []
        for i in range(self.n_verb):
            v = self.verb_codebook[i]
            for u in vs:
                v = v - (v * u).sum() * u
            v = v * torch.rsqrt(v.pow(2).sum() + 1e-8)
            vs.append(v)
        return torch.stack(vs, dim=0)

    def _cleanup(self, q_unit, o_unit, o_val, avail, temp, nullc):
        B, K, _ = q_unit.shape
        n = o_val.size(1)
        dot = torch.bmm(q_unit, o_unit.transpose(1, 2))
        logits = (dot / temp).masked_fill(~avail.unsqueeze(1), _NEG)
        nc = nullc.view(1, 1, 1).expand(B, K, 1).to(logits.dtype)
        att = torch.softmax(torch.cat([logits, nc], dim=-1), dim=-1)[..., :n]
        return torch.bmm(att, o_val)

    def run(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model or self.d_model % 2 != 0:
            return None
        B, L, _ = tok.shape
        if L < 1:
            return None
        n_cmd = (L + 1) // 2
        n_pair = L // 2
        dev = tok.device
        dt = tok.dtype

        c = _clean(tok[:, 0::2, :])
        o = _clean(tok[:, 1::2, :])
        if key_pad is None:
            read_ok = torch.ones(B, n_cmd, dtype=torch.bool, device=dev)
            write_ok = torch.ones(B, n_pair, dtype=torch.bool, device=dev)
        else:
            live = ~key_pad.bool()
            read_ok = live[:, 0::2]
            write_ok = read_ok[:, :n_pair] & live[:, 1::2]
        rf = read_ok.to(dt)
        wf = write_ok.to(dt)

        f = self.feat_norm(F.gelu(self.feat(c)))
        Q = self._verb_basis().to(f.dtype)
        fq = f - torch.matmul(torch.matmul(f, Q.transpose(0, 1)), Q)

        ang_p = math.pi * torch.tanh(self.path_angle(fq))
        ang_s = math.pi * torch.tanh(self.src_angle(f))
        ang_d = math.pi * torch.tanh(self.dst_angle(f))
        cp, sp = torch.cos(ang_p), torch.sin(ang_p)
        cs, ss = torch.cos(ang_s), torch.sin(ang_s)
        cd, sd = torch.cos(ang_d), torch.sin(ang_d)

        move_amt = torch.sigmoid(self.move_gate(f)).squeeze(-1) * rf
        write_amt = torch.sigmoid(self.write_gate(f)).squeeze(-1)
        gate = torch.sigmoid(self.read_gate(f))

        o_masked = o * wf.unsqueeze(-1)
        o_unit = _unit(o_masked)
        o_pair = o_masked.reshape(B, n_pair, self.half, 2)
        temp = self.log_temp.exp().clamp(0.02, 2.0)
        nullc = self.null_logit.clamp(-30.0, 30.0)
        slot = torch.arange(n_pair, device=dev)

        Sr = tok.new_zeros(B, self.half)
        Si = tok.new_zeros(B, self.half)
        raw_list = []
        clean_list = []

        for i in range(n_cmd):
            cpi, spi = cp[:, i, :], sp[:, i, :]
            csi, ssi = cs[:, i, :], ss[:, i, :]
            cdi, sdi = cd[:, i, :], sd[:, i, :]

            rr = Sr * cpi + Si * spi
            ri = -Sr * spi + Si * cpi
            read_raw = torch.stack([rr, ri], dim=-1).reshape(B, self.d_model)

            xr = Sr * csi + Si * ssi
            xi = -Sr * ssi + Si * csi
            src_raw = torch.stack([xr, xi], dim=-1).reshape(B, self.d_model)

            avail = write_ok & (slot.view(1, -1) < i)
            q = torch.stack([read_raw, src_raw], dim=1)
            v = self._cleanup(_unit(q), o_unit, o_masked, avail, temp, nullc)
            raw_list.append(read_raw)
            clean_list.append(v[:, 0, :])

            vs = v[:, 1, :].reshape(B, self.half, 2)
            vr_, vi_ = vs[..., 0], vs[..., 1]
            add_r = vr_ * cdi - vi_ * sdi
            add_i = vr_ * sdi + vi_ * cdi
            sub_r = vr_ * csi - vi_ * ssi
            sub_i = vr_ * ssi + vi_ * csi
            mi = move_amt[:, i].unsqueeze(-1)
            Sr = Sr + mi * (add_r - sub_r)
            Si = Si + mi * (add_i - sub_i)

            if i < n_pair:
                op = o_pair[:, i]
                orr, oii = op[..., 0], op[..., 1]
                br = orr * cpi - oii * spi
                bi = orr * spi + oii * cpi
                wi = (write_amt[:, i] * wf[:, i]).unsqueeze(-1)
                Sr = Sr + wi * br
                Si = Si + wi * bi

            Sr = _clean(Sr).clamp(-1e4, 1e4)
            Si = _clean(Si).clamp(-1e4, 1e4)

        reads_raw = torch.stack(raw_list, dim=1) * rf.unsqueeze(-1)
        reads_clean = torch.stack(clean_list, dim=1) * rf.unsqueeze(-1)
        contrib = _clean(gate * (reads_raw * self.out_raw.view(1, 1, -1)
                                 + reads_clean * self.out_clean.view(1, 1, -1)))
        return {"reads_raw": reads_raw, "reads_clean": reads_clean, "contrib": contrib,
                "read_ok": read_ok, "write_ok": write_ok, "cmd": c, "obs": o}


def wrap(net, D, **params):
    existing = getattr(net, "_rotor_board_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params or {})
    mod = _RotorBoardMemory(int(D), int(cfg["hid"]), int(cfg["n_verb"]),
                            float(cfg["temp_init"]), float(cfg["null_init"]),
                            float(cfg["clean_init"]))
    net.rotor_board = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.dim() != 3:
            return pred, h
        if tok_emb.size(1) < 1 or pred.size(1) != tok_emb.size(1):
            return pred, h
        run = mod.run(tok_emb, key_pad)
        if run is None:
            return pred, h
        contrib = run["contrib"]
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        k = min(contrib.size(1), cmd_view.size(1))
        if k > 0:
            cmd_view[:, :k, :] = cmd_view[:, :k, :] + contrib[:, :k, :]
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "run": run, "pred": out}
        return out, h

    net.forward = _forward
    net._rotor_board_state = state
    return state


@torch.no_grad()
def _mine_path_triples(cmd, obs, valid, path_thresh, change_floor):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(_clean(cmd))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    vmask = valid.bool()

    pos = torch.arange(maxn, device=device)
    lower = pos.unsqueeze(1) > pos.unsqueeze(0)
    upper = pos.unsqueeze(1) < pos.unsqueeze(0)
    same_path = (sim > path_thresh) & vmask.unsqueeze(1)

    before = same_path & lower.unsqueeze(0)
    after = same_path & upper.unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    i_idx = torch.where(before, posf, torch.full_like(posf, -1)).amax(dim=2)
    j_idx = torch.where(after, posf, torch.full_like(posf, maxn)).amin(dim=2)

    triple_ok = (i_idx >= 0) & (j_idx < maxn) & vmask

    ic = i_idx.clamp(0, maxn - 1)
    jc = j_idx.clamp(0, maxn - 1)
    obs_i = torch.gather(obs, 1, ic.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    obs_j = torch.gather(obs, 1, jc.unsqueeze(-1).expand(B, maxn, obs.size(-1)))
    change = (obs_i - obs_j).pow(2).mean(dim=-1)
    sim_ki = torch.gather(sim, 2, ic.unsqueeze(-1)).squeeze(-1)
    sim_kj = torch.gather(sim, 2, jc.unsqueeze(-1)).squeeze(-1)
    w = sim_ki * sim_kj * change

    keep = triple_ok & (change >= change_floor) & torch.isfinite(w) & (w > 0.0)
    nz = torch.nonzero(keep, as_tuple=False)
    sel_b = nz[:, 0]
    sel_k = nz[:, 1]
    return sel_b, i_idx[sel_b, sel_k], sel_k, j_idx[sel_b, sel_k], w[sel_b, sel_k]


def _path_change_loss(cfg, batch, net, device):
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return None
    if float(cfg["trans_weight"]) <= 0.0:
        return None

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return None

    nrows = max(1, int(math.ceil(B * float(cfg["path_row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_path_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["path_change_floor"])
    )
    if r.numel() == 0:
        return None
    cap = int(cfg["path_max_examples"])
    if r.numel() > cap:
        w, order = torch.topk(w, cap)
        r = r[order]
        ti = ti[order]
        tk = tk[order]
        tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = _clean(obs[r, ti].detach())
    cmd_k = _clean(cmd[r, tk].detach())
    tgt = _clean(obs[r, tj].detach())

    pred = _clean(op(pre, cmd_k))
    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    return float(cfg["path_cos_weight"]) * cos_err + float(cfg["path_mse_weight"]) * mse_err


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
    pred_cmd = None
    if stash is not None and stash["tok"] is tok:
        run = stash["run"]
        pred_cmd = stash["pred"][:, 0::2, :]
    else:
        run = mod.run(tok, batch["key_pad"])
    if run is None:
        return 0.0

    obs = run["obs"].float()
    B, n, Dm = obs.shape
    dim = float(Dm)
    reads_clean = run["reads_clean"]
    if n < 2 or reads_clean.size(1) < n:
        total = None
        have_dup = False
    else:
        reads_clean = reads_clean[:, :n, :].float()
        reads_raw = run["reads_raw"][:, :n, :].float()
        active = run["write_ok"][:, :n]
        dev = obs.device
        total = None

        with torch.no_grad():
            osq = obs.pow(2).sum(-1)
            d2 = (osq.unsqueeze(2) + osq.unsqueeze(1)
                  - 2.0 * torch.bmm(obs, obs.transpose(1, 2))).clamp_min(0.0) / dim
            eye = torch.eye(n, dtype=torch.bool, device=dev).unsqueeze(0)
            off = (active.unsqueeze(2) & active.unsqueeze(1)) & (~eye)
            cnt = off.sum(dim=(1, 2)).clamp_min(1).to(d2.dtype)
            ref = (d2 * off).sum(dim=(1, 2)) / cnt
            dup = off & (d2 <= (float(cfg["dup_frac"]) * ref).view(B, 1, 1))

            clus = dup.sum(dim=2)
            contentful = active & (clus <= int(cfg["max_dup"]))

            idx = torch.arange(n, device=dev)
            lower = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
            none_long = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)

            earlier = dup & lower & contentful.unsqueeze(2) & contentful.unsqueeze(1)
            has_e = earlier.any(dim=2)
            last_e = torch.where(earlier, idx.view(1, 1, n), none_long).amax(dim=2)

            mined = has_e & contentful
            nz = torch.nonzero(mined, as_tuple=False)
            have_dup = nz.numel() > 0
            if have_dup:
                b_i = nz[:, 0]
                t_i = nz[:, 1]
                gap = (t_i - last_e[b_i, t_i]).float()
                cap = int(cfg["max_examples"])
                if b_i.numel() > cap:
                    _, order = torch.topk(gap, cap)
                    b_i = b_i[order]
                    t_i = t_i[order]
                    gap = gap[order]
                ex_w = torch.where(gap >= 2.0,
                                   torch.ones_like(gap),
                                   torch.full_like(gap, float(cfg["near_weight"])))
                ex_w = (ex_w / ex_w.sum().clamp_min(_EPS)).to(obs.dtype)
                seen = idx.view(1, n) <= t_i.view(-1, 1)
                cand_mask = contentful[b_i] & seen
                pos_mask = (dup[b_i, t_i] | F.one_hot(t_i, n).bool()) & cand_mask
                cand_sq = osq[b_i]

    if have_dup:
        cand = obs[b_i]
        temp = max(1e-3, float(cfg["nce_temp"]))

        def _nce(z):
            dot = torch.bmm(cand, z.unsqueeze(2)).squeeze(2)
            dz = (z.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / dim
            lg = (-dz / temp).masked_fill(~cand_mask, _NEG)
            lp = torch.log_softmax(lg, dim=1)
            return -(ex_w * torch.logsumexp(lp.masked_fill(~pos_mask, _NEG), dim=1)).sum()

        routed = reads_clean[b_i, t_i]
        truth = obs[b_i, t_i]
        total = float(cfg["read_weight"]) * _nce(routed)
        if float(cfg["raw_weight"]) > 0.0:
            total = total + float(cfg["raw_weight"]) * _nce(reads_raw[b_i, t_i])
        if float(cfg["mse_weight"]) > 0.0:
            total = total + float(cfg["mse_weight"]) * (
                ex_w * (routed - truth).pow(2).mean(dim=-1)).sum()
        if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
            total = total + float(cfg["pred_weight"]) * _nce(
                pred_cmd[:, :n, :].float()[b_i, t_i])

    path_term = _path_change_loss(cfg, batch, net, device)
    if path_term is not None:
        path_term = float(cfg["trans_weight"]) * path_term.float()
        total = path_term if total is None else total + path_term

    if total is None:
        return 0.0

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
        vals["hid"] >= 16.0,
        vals["n_verb"] >= 1.0,
        0.0 < vals["temp_init"] <= 10.0,
        -30.0 <= vals["null_init"] <= 30.0,
        0.0 <= vals["clean_init"] <= 1.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        0.0 <= vals["near_weight"] <= 1.0,
        vals["read_weight"] >= 0.0,
        vals["raw_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["trans_weight"] >= 0.0,
        0.0 < vals["path_row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["path_change_floor"] >= 0.0,
        vals["path_max_examples"] >= 1.0,
        vals["path_cos_weight"] >= 0.0,
        vals["path_mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
