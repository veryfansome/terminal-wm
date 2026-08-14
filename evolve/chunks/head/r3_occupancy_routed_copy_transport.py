import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r3_occupancy_routed_copy_transport"
DESCRIPTION = (
    "A causal slot memory over command positions that moves observation content by COPY, split "
    "across two axes so it also acts at a command whose observation has not happened yet: there "
    "are (L+1)//2 READ positions, one per command, and L//2 WRITE slots, one per completed "
    "command/observation pair. A write slot's key is a unit path signature read off its command "
    "embedding and its value is a convex blend, set by a learned mutation gate over the "
    "(command, observation) pair, of the step's own observation and the content that command "
    "retrieved from earlier slots; that value recursion is solved exactly as one unit-lower-"
    "triangular system over the write slots, so a retrieved value can itself be a retrieval of a "
    "retrieval to unbounded depth in a single pass. A read is built from the command embedding "
    "ALONE — never from an observation — so every command position, including a final command "
    "with no observation after it, issues its own query; the query subtracts the causal running "
    "mean of the command signatures seen so far, scores against the write-slot keys, adds each "
    "slot's log write-occupancy so only written slots can answer, and carries a null column so an "
    "unmatched query retrieves nothing. The retrieved content is added to the arch's own "
    "prediction at every command position through a zero-initialised per-dimension scale, so the "
    "wrapped net is the unwrapped net at initialisation. Train-time aux mines, from the batch "
    "alone, steps whose observation duplicates an earlier step's observation with at least one "
    "contentless step in between, ranks the retrieved content against the contentful observations "
    "seen so far under the eval's squared-L2 decision variable, and supervises the arch's shared "
    "transition operator, when it exposes one, toward reproducing that duplicated content from "
    "the last contentless command before the read."
)

_DEFAULTS = {
    "key_d": 64,
    "hid": 128,
    "temp_init": 0.2,
    "center_init": 2.0,
    "dup_frac": 0.05,
    "max_dup": 3,
    "max_examples": 192,
    "nce_temp": 0.5,
    "read_weight": 1.0,
    "pred_weight": 0.25,
    "mse_weight": 0.05,
    "trans_weight": 0.5,
    "cos_weight": 1.0,
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


class _CopyTransportMemory(nn.Module):
    def __init__(self, d_model, key_d, hid, temp_init, center_init):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = int(key_d)
        h = int(hid)
        self.cmd_feat = nn.Linear(self.d_model, h)
        self.obs_feat = nn.Linear(self.d_model, h)
        self.addr = nn.Linear(h, self.key_d, bias=False)
        self.mut_gate = nn.Linear(2 * h, 1)
        self.occ_gate = nn.Linear(2 * h, 1)
        self.read_gate = nn.Linear(h, 1)
        nn.init.constant_(self.mut_gate.bias, -1.0)
        nn.init.constant_(self.occ_gate.bias, 1.5)
        nn.init.constant_(self.read_gate.bias, 0.0)
        self.center_logit = nn.Parameter(torch.tensor(float(center_init)))
        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.null_logit = nn.Parameter(torch.zeros(1))
        self.out_scale = nn.Parameter(torch.zeros(self.d_model))

    def _solve_unit_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False)
        parts = []
        n = system.size(1)
        for i in range(n):
            yi = rhs[:, i, :]
            if parts:
                prev = torch.stack(parts, dim=1)
                yi = yi - torch.bmm(system[:, i:i + 1, :i], prev).squeeze(1)
            parts.append(yi / system[:, i, i].unsqueeze(-1).clamp_min(1e-6))
        return torch.stack(parts, dim=1)

    def run(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        n_read = (L + 1) // 2
        n_write = L // 2
        if n_write < 1 or n_read < 1:
            return None
        dev = tok.device
        dt = tok.dtype

        c = _clean(tok[:, 0::2, :])
        o = _clean(tok[:, 1::2, :])
        if key_pad is None:
            read_ok = torch.ones(B, n_read, dtype=torch.bool, device=dev)
            write_ok = torch.ones(B, n_write, dtype=torch.bool, device=dev)
        else:
            live = ~key_pad.bool()
            read_ok = live[:, 0::2]
            write_ok = read_ok[:, :n_write] & live[:, 1::2]
        rf = read_ok.to(dt)
        wf = write_ok.to(dt)

        cf = F.gelu(self.cmd_feat(c))
        of = F.gelu(self.obs_feat(o))
        pair = torch.cat([cf[:, :n_write, :], of], dim=-1)

        p = _unit(self.addr(cf)) * rf.unsqueeze(-1)
        run_sum = torch.cumsum(p, dim=1)
        run_cnt = torch.cumsum(rf, dim=1).clamp_min(1.0).unsqueeze(-1)
        u = _unit(p - torch.sigmoid(self.center_logit) * (run_sum / run_cnt))

        keys = p[:, :n_write, :] * wf.unsqueeze(-1)
        temp = self.log_temp.exp().clamp(0.02, 2.0)
        scores = torch.bmm(u, keys.transpose(1, 2)) / temp

        occ = torch.sigmoid(self.occ_gate(pair)).squeeze(-1) * wf
        i_idx = torch.arange(n_read, device=dev).view(n_read, 1)
        j_idx = torch.arange(n_write, device=dev).view(1, n_write)
        reachable = (i_idx > j_idx).unsqueeze(0) & write_ok.unsqueeze(1)
        logits = (scores + torch.log(occ.clamp_min(1e-6)).unsqueeze(1)).masked_fill(~reachable, _NEG)
        nullcol = self.null_logit.clamp(-30.0, 30.0).view(1, 1, 1).expand(B, n_read, 1).to(logits.dtype)
        att = torch.softmax(torch.cat([logits, nullcol], dim=-1), dim=-1)[..., :n_write]

        m = torch.sigmoid(self.mut_gate(pair))
        eye = torch.eye(n_write, device=dev, dtype=dt).unsqueeze(0).expand(B, n_write, n_write)
        system = eye - m * att[:, :n_write, :]
        rhs = (1.0 - m) * o * wf.unsqueeze(-1)
        values = _clean(self._solve_unit_lower(system, rhs)).clamp(-1e4, 1e4)
        reads = _clean(torch.bmm(att, values)) * rf.unsqueeze(-1)

        gate = torch.sigmoid(self.read_gate(cf))
        contrib = _clean(gate * reads * self.out_scale.view(1, 1, -1))
        return {"reads": reads, "contrib": contrib, "read_ok": read_ok, "write_ok": write_ok,
                "cmd": c, "obs": o}


def wrap(net, D, **params):
    existing = getattr(net, "_copy_transport_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)
    mod = _CopyTransportMemory(int(D), int(cfg["key_d"]), int(cfg["hid"]),
                               float(cfg["temp_init"]), float(cfg["center_init"]))
    net.copy_transport = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 2:
            return pred, h
        run = mod.run(tok_emb, key_pad)
        if run is None:
            return pred, h
        contrib = run["contrib"]
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        k = min(contrib.size(1), cmd_view.size(1))
        cmd_view[:, :k, :] = cmd_view[:, :k, :] + contrib[:, :k, :]
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "run": run, "pred": out}
        return out, h

    net.forward = _forward
    net._copy_transport_state = state
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
    if n < 3:
        return 0.0
    reads = run["reads"][:, :n, :].float()
    cmd = run["cmd"][:, :n, :].float()
    active = run["write_ok"][:, :n]
    dev = obs.device
    dim = float(Dm)

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
        max_dup = int(cfg["max_dup"])
        contentful = active & (clus <= max_dup)
        contentless = active & (clus > max_dup)

        idx = torch.arange(n, device=dev)
        lower = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        none_long = torch.full((1, 1, n), -1, device=dev, dtype=torch.long)

        earlier = dup & lower & contentful.unsqueeze(2) & contentful.unsqueeze(1)
        has_e = earlier.any(dim=2)
        last_e = torch.where(earlier, idx.view(1, 1, n), none_long).amax(dim=2)

        mcum = torch.cumsum(contentless.long(), dim=1)
        prev_pos = (idx.view(1, n) - 1).clamp_min(0).expand(B, n)
        hops = (torch.gather(mcum, 1, prev_pos)
                - torch.gather(mcum, 1, last_e.clamp_min(0))).clamp_min(0)

        mut_before = contentless.unsqueeze(1) & lower
        last_k = torch.where(mut_before, idx.view(1, 1, n), none_long).amax(dim=2)

        mined = has_e & contentful & (hops > 0) & (last_k > last_e)
        nz = torch.nonzero(mined, as_tuple=False)
        if nz.numel() == 0:
            return 0.0
        b_i = nz[:, 0]
        t_i = nz[:, 1]
        cap = int(cfg["max_examples"])
        if b_i.numel() > cap:
            _, order = torch.topk(hops[b_i, t_i].float(), cap)
            b_i = b_i[order]
            t_i = t_i[order]
        e_i = last_e[b_i, t_i]
        k_i = last_k[b_i, t_i]

        seen = idx.view(1, n) <= t_i.view(-1, 1)
        cand_mask = contentful[b_i] & seen
        pos_mask = (dup[b_i, t_i] | F.one_hot(t_i, n).bool()) & cand_mask
        cand_sq = osq[b_i]

    cand = obs[b_i]
    temp = max(1e-3, float(cfg["nce_temp"]))

    def _nce(z):
        dot = torch.bmm(cand, z.unsqueeze(2)).squeeze(2)
        dz = (z.pow(2).sum(-1, keepdim=True) + cand_sq - 2.0 * dot).clamp_min(0.0) / dim
        lg = (-dz / temp).masked_fill(~cand_mask, _NEG)
        lp = torch.log_softmax(lg, dim=1)
        return -(torch.logsumexp(lp.masked_fill(~pos_mask, _NEG), dim=1)).mean()

    routed = reads[b_i, t_i]
    truth = obs[b_i, t_i]
    total = float(cfg["read_weight"]) * _nce(routed)
    total = total + float(cfg["mse_weight"]) * (routed - truth).pow(2).mean()

    if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
        total = total + float(cfg["pred_weight"]) * _nce(pred_cmd[:, :n, :].float()[b_i, t_i])

    op = getattr(net, "transition_from_emb", None)
    trans_w = float(cfg["trans_weight"])
    if trans_w > 0.0 and callable(op):
        pre = obs[b_i, e_i].detach().to(tok.dtype)
        cmd_k = cmd[b_i, k_i].detach().to(tok.dtype)
        keep = truth.detach().to(tok.dtype)
        moved = _clean(op(pre, cmd_k)).float()
        gu = _unit(keep.float())
        cos_err = (1.0 - (_unit(moved) * gu).sum(dim=-1).clamp(-1.0, 1.0)).mean()
        mse_err = (moved - keep.float()).pow(2).mean()
        total = total + trans_w * (float(cfg["cos_weight"]) * cos_err
                                   + float(cfg["mse_weight"]) * mse_err)

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
        0.0 < vals["temp_init"] <= 10.0,
        -20.0 <= vals["center_init"] <= 20.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["trans_weight"] >= 0.0,
        vals["cos_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
