import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r6_backward_erasure_resolved_transport"
DESCRIPTION = (
    "A backward content resolver over command positions with dual addresses, an explicit erasure "
    "penalty, and an unbounded-depth closed form. Every command's raw embedding passes one shared "
    "GELU trunk and two unit-normalised heads giving a SOURCE address (the location the command "
    "reads from) and a DESTINATION address (the location it writes to). A learned RELAY gate, fed "
    "by the command and its paired observation, decides whether a step merely relays content "
    "(empty-output mutation) or exposes it; the location a step's content ends up at is therefore "
    "the gated blend of its destination and its source address, and that single gate is reused as "
    "the coefficient of the resolution recursion. A read at step i queries its own source address "
    "against the end-location keys of strictly earlier steps, discounted by a learned recency rate "
    "and by an ERASURE term equal to the strongest later relay that read from that same key, so a "
    "location whose content was moved away stops answering. The recursion 'the content at the "
    "location step i reads from is, at the matched step, either that step's observation or the "
    "content that step itself resolved' is one strictly-lower-triangular linear system in the "
    "content variable, solved exactly in a single unit-triangular solve, so resolution depth is "
    "unbounded rather than a fixed hop count and no sequential scan over the sequence is needed. "
    "The query is built from the command embedding alone, so the final command of an odd-length "
    "window, whose observation is the answer and is absent, still issues a full query. The "
    "resolved content enters the arch's own prediction at every command position under a sigmoid "
    "gate through two parallel paths: a zero-initialised (D,D) readout, which stays exactly zero "
    "at build time so a cold-parameter optimizer can open it, and a small non-zero per-dimension "
    "direct scale, so every parameter of the resolver measurably moves the prediction at the "
    "scored read position from the first step. Train-time aux, label-free from the batch alone: "
    "a binary calibration of "
    "the relay gate against steps whose observation falls in an oversized duplicate cluster, and, "
    "on mined reads whose observation duplicates an earlier observation with at least one such "
    "step in between, a listwise ranking of the resolved content and of the head-augmented "
    "prediction against the contentful observations seen so far under the eval's own per-dimension "
    "squared-L2 decision variable."
)

_DEFAULTS = {
    "key_d": 64,
    "addr_hidden": 256,
    "obs_hidden": 128,
    "temp_init": 0.25,
    "recency_init": 0.15,
    "erase_init": 2.0,
    "erase_sharp_init": 8.0,
    "erase_thresh_init": 0.5,
    "move_cap": 0.98,
    "direct_init": 0.02,
    "gate_bias": -2.0,
    "relay_bias": 0.0,
    "occ_bias": 1.5,
    "dup_frac": 0.05,
    "max_dup": 3,
    "max_examples": 192,
    "nce_temp": 0.5,
    "read_weight": 1.0,
    "pred_weight": 0.5,
    "relay_weight": 0.25,
    "mse_weight": 0.05,
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


def _inv_softplus(y):
    y = max(1e-4, float(y))
    return math.log(math.expm1(y)) if y < 20.0 else y


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


class _BackwardResolver(nn.Module):
    def __init__(self, d_model, key_d, addr_hidden, obs_hidden, temp_init, recency_init,
                 erase_init, erase_sharp_init, erase_thresh_init, move_cap, direct_init,
                 gate_bias, relay_bias, occ_bias):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(8, int(key_d))
        self.addr_h = max(32, int(addr_hidden))
        self.obs_h = max(16, int(obs_hidden))
        self.move_cap = min(0.999, max(0.05, float(move_cap)))

        self.addr_ln = nn.LayerNorm(self.d_model)
        self.addr_in = nn.Linear(self.d_model, self.addr_h)
        self.addr_mid = nn.Linear(self.addr_h, self.addr_h)
        self.addr_src = nn.Linear(self.addr_h, self.key_d, bias=False)
        self.addr_dst = nn.Linear(self.addr_h, self.key_d, bias=False)

        self.obs_in = nn.Linear(self.d_model, self.obs_h)
        self.relay_score = nn.Linear(self.addr_h + self.obs_h, 1)
        self.occ_score = nn.Linear(self.addr_h + self.obs_h, 1)
        nn.init.constant_(self.relay_score.bias, float(relay_bias))
        nn.init.constant_(self.occ_score.bias, float(occ_bias))

        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.recency_raw = nn.Parameter(torch.tensor(_inv_softplus(recency_init)))
        self.erase_raw = nn.Parameter(torch.tensor(_inv_softplus(erase_init)))
        self.erase_sharp = nn.Parameter(torch.tensor(float(erase_sharp_init)))
        self.erase_thresh = nn.Parameter(torch.tensor(float(erase_thresh_init)))
        self.null_logit = nn.Parameter(torch.zeros(1))

        self.read_score = nn.Linear(self.addr_h + 3, 1)
        nn.init.constant_(self.read_score.bias, float(gate_bias))
        self.read_out = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)
        self.direct_scale = nn.Parameter(
            torch.full((self.d_model,), float(direct_init)))

    def solve_unit_lower(self, system, rhs):
        if system.device.type != "mps":
            return torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        rows = []
        n = system.size(1)
        for i in range(n):
            yi = rhs[:, i, :]
            if rows:
                prev = torch.stack(rows, dim=1)
                yi = yi - torch.bmm(system[:, i:i + 1, :i], prev).squeeze(1)
            rows.append(yi)
        return torch.stack(rows, dim=1)

    def resolve(self, tok, key_pad):
        if tok.dim() != 3 or tok.size(-1) != self.d_model:
            return None
        B, L, _ = tok.shape
        n = (L + 1) // 2
        n_pair = L // 2
        if n < 2 or n_pair < 1:
            return None
        dev = tok.device
        dt = tok.dtype

        cmd = _clean(tok[:, 0::2, :])
        obs_p = _clean(tok[:, 1::2, :])
        if n_pair < n:
            obs = torch.cat([obs_p, obs_p.new_zeros(B, n - n_pair, self.d_model)], dim=1)
        else:
            obs = obs_p[:, :n, :]

        if key_pad is None:
            cmd_ok = torch.ones(B, n, dtype=torch.bool, device=dev)
            obs_ok = torch.ones(B, n, dtype=torch.bool, device=dev)
        else:
            live = ~key_pad.bool()
            cmd_ok = live[:, 0::2][:, :n]
            obs_ok = live[:, 1::2][:, :n]
            if obs_ok.size(1) < n:
                obs_ok = torch.cat(
                    [obs_ok, obs_ok.new_zeros(B, n - obs_ok.size(1))], dim=1)
        node_ok = cmd_ok & obs_ok
        node_f = node_ok.to(dt)

        a = F.gelu(self.addr_mid(F.gelu(self.addr_in(self.addr_ln(cmd)))))
        s = _unit(self.addr_src(a))
        dst = _unit(self.addr_dst(a))
        of = F.gelu(self.obs_in(obs))
        pair = torch.cat([a, of], dim=-1)

        m = self.move_cap * torch.sigmoid(self.relay_score(pair)).squeeze(-1) * node_f
        occ = torch.sigmoid(self.occ_score(pair)).squeeze(-1) * node_f
        key = _unit(m.unsqueeze(-1) * dst + (1.0 - m).unsqueeze(-1) * s)

        sim = torch.bmm(s, key.transpose(1, 2))

        idx = torch.arange(n, device=dev)
        strict = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        strict_f = strict.to(dt)

        sharpness = self.erase_sharp.clamp(0.1, 64.0)
        thresh = self.erase_thresh.clamp(-1.0, 1.0)
        taken = m.unsqueeze(-1) * torch.sigmoid(sharpness * (sim - thresh)) * strict_f
        run_max = torch.cummax(taken, dim=1).values
        away = torch.cat([run_max.new_zeros(B, 1, n), run_max[:, :-1, :]], dim=1)

        temp = self.log_temp.exp().clamp(0.02, 4.0)
        recency = F.softplus(self.recency_raw).clamp(0.0, 4.0)
        erase_w = F.softplus(self.erase_raw).clamp(0.0, 20.0)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1.0).clamp_min(0.0).unsqueeze(0).to(dt)

        logits = (sim / temp
                  + torch.log(occ.clamp_min(1e-6)).unsqueeze(1)
                  - recency * gap
                  - erase_w * away)
        allowed = strict & node_ok.unsqueeze(1) & cmd_ok.unsqueeze(2)
        logits = logits.masked_fill(~allowed, _NEG)
        nullcol = self.null_logit.clamp(-30.0, 30.0).view(1, 1, 1).expand(B, n, 1).to(logits.dtype)
        att = torch.softmax(torch.cat([logits, nullcol], dim=-1), dim=-1)[..., :n] * strict_f

        eye = torch.eye(n, device=dev, dtype=att.dtype).unsqueeze(0)
        system = eye - att * m.unsqueeze(1)
        rhs = torch.bmm(att, obs * ((1.0 - m) * node_f).unsqueeze(-1))
        content = _clean(self.solve_unit_lower(system, rhs)).clamp(-1e4, 1e4)

        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        peak = att.amax(dim=2, keepdim=True)
        mass = att.sum(dim=2, keepdim=True)
        gate = torch.sigmoid(self.read_score(torch.cat([a, rms, peak, mass], dim=-1)))
        routed = self.read_out(content) + self.direct_scale.view(1, 1, -1) * content
        contrib = _clean(gate * routed) * cmd_ok.unsqueeze(-1).to(dt)

        return {"content": content, "contrib": contrib, "att": att, "move": m,
                "obs": obs, "node_ok": node_ok, "n_pair": n_pair}


def wrap(net, D, **params):
    existing = getattr(net, "_backward_resolver_state", None)
    if existing is not None:
        return existing

    cfg = dict(_DEFAULTS)
    cfg.update(params)
    mod = _BackwardResolver(
        int(D), int(cfg["key_d"]), int(cfg["addr_hidden"]), int(cfg["obs_hidden"]),
        float(cfg["temp_init"]), float(cfg["recency_init"]), float(cfg["erase_init"]),
        float(cfg["erase_sharp_init"]), float(cfg["erase_thresh_init"]),
        float(cfg["move_cap"]), float(cfg["direct_init"]), float(cfg["gate_bias"]),
        float(cfg["relay_bias"]), float(cfg["occ_bias"]),
    )
    net.backward_resolver = mod

    state = {"cfg": cfg, "mod": mod, "step": 0, "stash": None, "D": int(D)}
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        pred, h = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if pred.dim() != 3 or pred.size(-1) != state["D"] or tok_emb.size(1) < 3:
            return pred, h
        res = mod.resolve(tok_emb, key_pad)
        if res is None:
            return pred, h
        contrib = res["contrib"]
        out = pred.clone()
        cmd_view = out[:, 0::2, :]
        k = min(contrib.size(1), cmd_view.size(1))
        cmd_view[:, :k, :] = cmd_view[:, :k, :] + contrib[:, :k, :].to(out.dtype)
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "res": res, "pred": out}
        return out, h

    net.forward = _forward
    net._backward_resolver_state = state
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
        res = stash["res"]
        pred_cmd = stash["pred"][:, 0::2, :]
    else:
        res = mod.resolve(tok, batch["key_pad"])
    if res is None:
        return 0.0

    npair = int(res["n_pair"])
    if npair < 3:
        return 0.0
    obs = res["obs"][:, :npair, :].float()
    content = res["content"][:, :npair, :].float()
    move = res["move"][:, :npair].float()
    active = res["node_ok"][:, :npair]
    B, n, Dm = obs.shape
    dim = float(Dm)
    dev = obs.device

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
        mined = has_e & contentful & (hops > 0)

    total = None
    relay_w = float(cfg["relay_weight"])
    if relay_w > 0.0:
        wf = active.to(content.dtype)
        denom = wf.sum().clamp_min(1.0)
        prob = (move / mod.move_cap).clamp(1e-4, 1.0 - 1e-4)
        lab = contentless.to(content.dtype)
        bce = -(lab * torch.log(prob) + (1.0 - lab) * torch.log(1.0 - prob))
        total = relay_w * ((bce * wf).sum() / denom)

    with torch.no_grad():
        nz = torch.nonzero(mined, as_tuple=False)
    if nz.numel() > 0:
        with torch.no_grad():
            b_i = nz[:, 0]
            t_i = nz[:, 1]
            cap = int(cfg["max_examples"])
            if b_i.numel() > cap:
                _, order = torch.topk(hops[b_i, t_i].float(), cap)
                b_i = b_i[order]
                t_i = t_i[order]
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

        routed = content[b_i, t_i]
        truth = obs[b_i, t_i]
        term = float(cfg["read_weight"]) * _nce(routed)
        term = term + float(cfg["mse_weight"]) * (routed - truth).pow(2).mean()
        if pred_cmd is not None and float(cfg["pred_weight"]) > 0.0:
            pc = pred_cmd[:, :n, :].float()
            term = term + float(cfg["pred_weight"]) * _nce(pc[b_i, t_i])
        total = term if total is None else total + term

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
        vals["key_d"] >= 8.0,
        vals["addr_hidden"] >= 32.0,
        vals["obs_hidden"] >= 16.0,
        0.0 < vals["temp_init"] <= 10.0,
        vals["recency_init"] >= 0.0,
        vals["erase_init"] >= 0.0,
        0.0 < vals["erase_sharp_init"] <= 64.0,
        -1.0 <= vals["erase_thresh_init"] <= 1.0,
        0.0 < vals["move_cap"] < 1.0,
        0.0 < abs(vals["direct_init"]) <= 1.0,
        -20.0 <= vals["gate_bias"] <= 20.0,
        -20.0 <= vals["relay_bias"] <= 20.0,
        -20.0 <= vals["occ_bias"] <= 20.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["relay_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
