import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "r7_causal_sinkhorn_chainlink_resolver"
DESCRIPTION = (
    "A chain-link resolver that addresses the window's own earlier commands instead of an abstract "
    "slot space, over command embeddings that are recentred and rescaled by a CAUSAL running "
    "estimate of the window's own command mean and variance, shrunk toward a learned global mean "
    "and scale by a fixed prior count. No fixed linear layer can subtract a window-dependent mean, "
    "so this removes the large component every command in a window shares before any key is formed; "
    "at the scored read position the causal estimate has seen the whole window. The link score "
    "between a consumer step and an earlier producer step is a learned low-rank bilinear form of "
    "those recentred embeddings (a source head and a destination head initialised as a "
    "symmetry-broken copy of each other) plus a learned recency rate, masked strictly causal. Those "
    "scores are turned into an assignment by log-domain Sinkhorn normalisation with a learned "
    "dustbin row and column, so the links compete in BOTH directions: each step takes at most one "
    "predecessor AND each earlier step is consumed by at most one successor, which is the physics "
    "of a chain of moves and is not expressible by independent per-row softmaxes. A learned relay "
    "gate, fed by the step's command features and its paired observation, decides whether a step "
    "delivers its own observation or relays what it received; the recursion 'the content a step "
    "delivers is, at its assigned predecessor, either that predecessor's observation or the content "
    "that predecessor itself resolved' is one strictly-lower-triangular linear system, solved "
    "exactly in a single unit-triangular solve, so resolution depth is unbounded and no sequential "
    "scan is needed. The query is built from the command alone, so the trailing command of an "
    "odd-length window, whose observation is the answer and is absent, still issues a full query and "
    "no content ever depends on its own step's observation. The resolved content enters the arch's "
    "own prediction at command positions under a sigmoid gate through a zero-initialised (D,D) "
    "readout plus a small non-zero per-dimension direct scale, so every resolver parameter moves the "
    "prediction at the scored read position from the first step. Train-only aux, label-free: a "
    "binary calibration of the relay gate against steps whose observation falls in an oversized "
    "duplicate cluster, a commitment term pushing each relay step's assignment row onto a single "
    "predecessor, and, on mined reads whose observation duplicates an earlier one across at least "
    "one content-free step, a listwise ranking of the resolved content and of the head-augmented "
    "prediction against the contentful observations seen so far under the eval's own per-dimension "
    "squared-L2 decision variable. Carries the forward-model consistency aux on the arch's shared "
    "transition operator with raw-observation and memory-content pre-state arms, auto-disabled on "
    "archs without that surface."
)

_DEFAULTS = {
    "key_d": 64,
    "hid": 256,
    "obs_hid": 128,
    "prior_count": 4.0,
    "sink_iters": 5,
    "dustbin_init": 1.0,
    "temp_init": 0.15,
    "recency_init": 0.05,
    "sym_init_noise": 0.02,
    "relay_bias": 0.0,
    "gate_bias": -1.0,
    "direct_init": 0.02,
    "move_cap": 0.98,
    "dup_frac": 0.05,
    "max_dup": 3,
    "max_examples": 192,
    "nce_temp": 0.5,
    "read_weight": 1.0,
    "pred_weight": 0.5,
    "relay_weight": 0.25,
    "commit_weight": 0.05,
    "mse_weight": 0.05,
    "trans_weight": 1.0,
    "trans_cos": 0.10,
    "trans_mse": 0.02,
    "trans_row_frac": 0.6,
    "trans_path_thresh": 0.60,
    "trans_change_floor": 0.25,
    "trans_max_examples": 512,
    "mem_arm_w": 0.5,
    "aux_weight": 1.0,
    "ramp_steps": 300,
}

_EPS = 1e-8
_NEG = -1e9

_MEM_ATTRS = ("cmd_proj", "obs_proj", "type_emb", "in_norm", "_positional", "_transition_reads")


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


class _ChainLinkResolver(nn.Module):
    def __init__(self, d_model, key_d, hid, obs_hid, prior_count, sink_iters, dustbin_init,
                 temp_init, recency_init, sym_init_noise, relay_bias, gate_bias, direct_init,
                 move_cap):
        super().__init__()
        self.d_model = int(d_model)
        self.key_d = max(8, int(key_d))
        self.hid = max(32, int(hid))
        self.obs_h = max(16, int(obs_hid))
        self.prior_count = max(0.0, float(prior_count))
        self.sink_iters = max(1, int(sink_iters))
        self.move_cap = min(0.999, max(0.05, float(move_cap)))

        self.global_mean = nn.Parameter(torch.zeros(self.d_model))
        self.global_scale_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0)))

        self.cmd_in = nn.Linear(self.d_model, self.hid)
        self.cmd_mid = nn.Linear(self.hid, self.hid)
        self.src_key = nn.Linear(self.hid, self.key_d, bias=False)
        self.dst_key = nn.Linear(self.hid, self.key_d, bias=False)
        with torch.no_grad():
            noise = torch.randn_like(self.src_key.weight) * float(sym_init_noise)
            self.dst_key.weight.copy_(self.src_key.weight + noise)

        self.obs_in = nn.Linear(self.d_model, self.obs_h)
        self.relay_score = nn.Linear(self.hid + self.obs_h, 1)
        nn.init.constant_(self.relay_score.bias, float(relay_bias))

        self.log_temp = nn.Parameter(torch.tensor(math.log(max(1e-3, float(temp_init)))))
        self.recency_raw = nn.Parameter(torch.tensor(_inv_softplus(recency_init)))
        self.dustbin = nn.Parameter(torch.tensor(float(dustbin_init)))

        self.read_score = nn.Linear(self.hid + 2, 1)
        nn.init.constant_(self.read_score.bias, float(gate_bias))
        self.read_out = nn.Linear(self.d_model, self.d_model)
        nn.init.zeros_(self.read_out.weight)
        nn.init.zeros_(self.read_out.bias)
        self.direct_scale = nn.Parameter(torch.full((self.d_model,), float(direct_init)))

    def causal_standardize(self, cmd, ok):
        w = ok.to(cmd.dtype).unsqueeze(-1)
        cw = cmd * w
        s1 = torch.cumsum(cw, dim=1)
        sq = cmd.pow(2).mean(dim=-1, keepdim=True) * w
        s2 = torch.cumsum(sq, dim=1)
        cnt = torch.cumsum(w, dim=1)
        kap = self.prior_count
        m0 = self.global_mean.view(1, 1, -1).to(cmd.dtype)
        s0 = F.softplus(self.global_scale_raw).clamp(1e-3, 1e3).to(cmd.dtype)
        denom = (cnt + kap).clamp_min(1e-3)
        mu = (s1 + kap * m0) / denom
        m2 = (s2 + kap * (m0.pow(2).mean(dim=-1, keepdim=True) + s0 * s0)) / denom
        var = (m2 - mu.pow(2).mean(dim=-1, keepdim=True)).clamp_min(1e-4)
        return _clean((cmd - mu) * torch.rsqrt(var)) * w

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

    def assign(self, score, allowed, dt):
        B, n, _ = score.shape
        N = n + 1
        z = self.dustbin.clamp(-30.0, 30.0).to(dt)
        masked = score.masked_fill(~allowed, _NEG)
        col_pad = z.view(1, 1, 1).expand(B, n, 1)
        top = torch.cat([masked, col_pad], dim=2)
        bot = z.view(1, 1, 1).expand(B, 1, N)
        full = torch.cat([top, bot], dim=1)

        norm = math.log(2.0 * float(n))
        log_mu = full.new_full((B, N), -norm)
        log_mu[:, n] = math.log(float(n)) - norm
        log_nu = log_mu

        u = full.new_zeros(B, N)
        v = full.new_zeros(B, N)
        for _ in range(self.sink_iters):
            v = log_nu - torch.logsumexp(full + u.unsqueeze(2), dim=1)
            u = log_mu - torch.logsumexp(full + v.unsqueeze(1), dim=2)
        plan = torch.exp(full + u.unsqueeze(2) + v.unsqueeze(1))
        att = plan[:, :n, :n] * (2.0 * float(n))
        return _clean(att) * allowed.to(dt)

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
                obs_ok = torch.cat([obs_ok, obs_ok.new_zeros(B, n - obs_ok.size(1))], dim=1)
        node_ok = cmd_ok & obs_ok
        node_f = node_ok.to(dt)

        ch = self.causal_standardize(cmd, cmd_ok)
        feat = F.gelu(self.cmd_mid(F.gelu(self.cmd_in(ch))))
        q = _unit(self.src_key(feat))
        k = _unit(self.dst_key(feat))

        sim = torch.bmm(q, k.transpose(1, 2))
        idx = torch.arange(n, device=dev)
        strict = (idx.view(n, 1) > idx.view(1, n)).unsqueeze(0)
        gap = (idx.view(n, 1) - idx.view(1, n) - 1.0).clamp_min(0.0).unsqueeze(0).to(dt)

        temp = self.log_temp.exp().clamp(0.02, 4.0).to(dt)
        recency = F.softplus(self.recency_raw).clamp(0.0, 4.0).to(dt)
        score = sim / temp - recency * gap
        allowed = strict & node_ok.unsqueeze(1) & cmd_ok.unsqueeze(2)
        att = self.assign(score, allowed, dt)

        of = F.gelu(self.obs_in(obs))
        relay = self.move_cap * torch.sigmoid(
            self.relay_score(torch.cat([feat, of], dim=-1))).squeeze(-1) * node_f

        eye = torch.eye(n, device=dev, dtype=att.dtype).unsqueeze(0)
        system = eye - att * relay.unsqueeze(1)
        rhs = torch.bmm(att, obs * ((1.0 - relay) * node_f).unsqueeze(-1))
        content = _clean(self.solve_unit_lower(system, rhs)).clamp(-1e4, 1e4)

        rms = (content.pow(2).mean(dim=-1, keepdim=True) + 1e-12).sqrt()
        mass = att.sum(dim=2, keepdim=True)
        gate = torch.sigmoid(self.read_score(torch.cat([feat, rms, mass], dim=-1)))
        routed = self.read_out(content) + self.direct_scale.view(1, 1, -1) * content
        contrib = _clean(gate * routed) * cmd_ok.unsqueeze(-1).to(dt)

        return {"content": content, "contrib": contrib, "att": att, "relay": relay,
                "obs": obs, "node_ok": node_ok, "n_pair": n_pair}


def wrap(net, D, **params):
    prev = getattr(net, "_chainlink_state", None)
    if prev is not None:
        return prev

    cfg = dict(_DEFAULTS)
    cfg.update(params)

    mod = _ChainLinkResolver(
        int(D), int(cfg["key_d"]), int(cfg["hid"]), int(cfg["obs_hid"]),
        float(cfg["prior_count"]), int(cfg["sink_iters"]), float(cfg["dustbin_init"]),
        float(cfg["temp_init"]), float(cfg["recency_init"]), float(cfg["sym_init_noise"]),
        float(cfg["relay_bias"]), float(cfg["gate_bias"]), float(cfg["direct_init"]),
        float(cfg["move_cap"]),
    )
    net.chainlink_resolver = mod

    state = {
        "cfg": cfg,
        "mod": mod,
        "step": 0,
        "stash": None,
        "D": int(D),
        "mem_ok": all(hasattr(net, a) for a in _MEM_ATTRS) and hasattr(net, "pos_scale"),
    }
    orig_forward = net.forward

    def _forward(tok_emb, types, key_pad, *extra, **kw):
        out = orig_forward(tok_emb, types, key_pad, *extra, **kw)
        state["stash"] = None
        if not (isinstance(out, tuple) and len(out) == 2):
            return out
        pred, h = out
        if not torch.is_tensor(pred) or pred.dim() != 3 or pred.size(-1) != state["D"]:
            return out
        if not torch.is_tensor(tok_emb) or tok_emb.dim() != 3 or tok_emb.size(1) < 3:
            return out
        res = mod.resolve(tok_emb, key_pad)
        if res is None:
            return out
        contrib = res["contrib"]
        new_pred = pred.clone()
        cmd_view = new_pred[:, 0::2, :]
        j = min(contrib.size(1), cmd_view.size(1))
        if j < 1:
            return out
        cmd_view[:, :j, :] = cmd_view[:, :j, :] + contrib[:, :j, :].to(new_pred.dtype)
        if torch.is_grad_enabled() and mod.training:
            state["stash"] = {"tok": tok_emb, "res": res, "pred": new_pred}
        return new_pred, h

    net.forward = _forward
    net._chainlink_state = state
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

    has_i = i_idx >= 0
    has_j = j_idx < maxn
    triple_ok = has_i & has_j & vmask

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


@torch.no_grad()
def _memory_pre(net, rows_tok, rows_types, valid, device):
    L = rows_tok.shape[1]
    t = rows_types.long().clamp(0, 1)
    x = torch.where((t == 0).unsqueeze(-1), net.cmd_proj(rows_tok), net.obs_proj(rows_tok))
    x = x + net.type_emb(t) + net.pos_scale * net._positional(L, device, x.dtype).unsqueeze(0)
    x = net.in_norm(x)
    maxn = valid.shape[1]
    x_cmd = x[:, 0::2][:, :maxn]
    obs = rows_tok[:, 1::2][:, :maxn]
    return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn)


def _transition_terms(st, net, batch, tok, valid, B, n, device):
    cfg = st["cfg"]
    if float(cfg["trans_weight"]) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["trans_row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :n]
    obs = tok[sel][:, 1::2][:, :n]
    val = valid[sel]

    r, ti, tk, tj, w = _mine_path_triples(
        cmd, obs, val, float(cfg["trans_path_thresh"]), float(cfg["trans_change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    cap = int(cfg["trans_max_examples"])
    if r.numel() > cap:
        w, order = torch.topk(w, cap)
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    cmd_k = _clean(cmd[r, tk].detach())
    tgt = _clean(obs[r, tj].detach())
    gu = _unit(tgt)

    def arm(pre):
        pred = _clean(op(pre, cmd_k))
        pu = _unit(pred)
        cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
        return float(cfg["trans_cos"]) * cos_err + float(cfg["trans_mse"]) * mse_err

    out = arm(_clean(obs[r, ti].detach()))
    mem_w = float(cfg["mem_arm_w"])
    if mem_w > 0.0 and bool(st.get("mem_ok", False)):
        reads = _memory_pre(net, tok[sel], batch["types"][sel], val, device)
        out = out + mem_w * arm(_clean(reads[r, tk].detach()))
    return float(cfg["trans_weight"]) * out


def _link_terms(st, mod, batch, tok, stash):
    cfg = st["cfg"]
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
    relay = res["relay"][:, :npair].float()
    att = res["att"][:, :npair, :npair].float()
    active = res["node_ok"][:, :npair]
    B, n, dm_i = obs.shape
    dim = float(dm_i)
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
        prob = (relay / mod.move_cap).clamp(1e-4, 1.0 - 1e-4)
        lab = contentless.to(content.dtype)
        bce = -(lab * torch.log(prob) + (1.0 - lab) * torch.log(1.0 - prob))
        total = relay_w * ((bce * wf).sum() / denom)

    commit_w = float(cfg["commit_weight"])
    if commit_w > 0.0:
        rw = (relay.detach() * contentless.to(relay.dtype))
        peak = att.amax(dim=2)
        commit = ((1.0 - peak.clamp(0.0, 1.0)) * rw).sum() / rw.sum().clamp_min(1.0)
        total = commit_w * commit if total is None else total + commit_w * commit

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
    return total


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

    parts = []
    lk = _link_terms(st, mod, batch, tok, stash)
    if torch.is_tensor(lk):
        parts.append(lk.float())
    tr = _transition_terms(st, net, batch, tok, valid, B, n, device)
    if torch.is_tensor(tr):
        parts.append(tr.float())
    if not parts:
        return 0.0

    total = parts[0]
    for extra in parts[1:]:
        total = total + extra
    out = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out).item()):
        return 0.0
    return out.to(tok.dtype)


def leak_safe(mod, params):
    p = dict(_DEFAULTS)
    extra = dict(params or {})
    for k in extra:
        if k not in _DEFAULTS:
            return False
    p.update(extra)
    try:
        vals = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    checks = [
        vals["key_d"] >= 8.0,
        vals["hid"] >= 32.0,
        vals["obs_hid"] >= 16.0,
        vals["prior_count"] >= 0.0,
        1.0 <= vals["sink_iters"] <= 64.0,
        -30.0 <= vals["dustbin_init"] <= 30.0,
        0.0 < vals["temp_init"] <= 10.0,
        vals["recency_init"] >= 0.0,
        vals["sym_init_noise"] >= 0.0,
        -20.0 <= vals["relay_bias"] <= 20.0,
        -20.0 <= vals["gate_bias"] <= 20.0,
        0.0 < abs(vals["direct_init"]) <= 1.0,
        0.0 < vals["move_cap"] < 1.0,
        0.0 < vals["dup_frac"] < 1.0,
        vals["max_dup"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["nce_temp"] > 0.0,
        vals["read_weight"] >= 0.0,
        vals["pred_weight"] >= 0.0,
        vals["relay_weight"] >= 0.0,
        vals["commit_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["trans_weight"] >= 0.0,
        vals["trans_cos"] >= 0.0,
        vals["trans_mse"] >= 0.0,
        0.0 < vals["trans_row_frac"] <= 1.0,
        -1.0 <= vals["trans_path_thresh"] < 1.0,
        vals["trans_change_floor"] >= 0.0,
        vals["trans_max_examples"] >= 1.0,
        0.0 <= vals["mem_arm_w"] <= 10.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
