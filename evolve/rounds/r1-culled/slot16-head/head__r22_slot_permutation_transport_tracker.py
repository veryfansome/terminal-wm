import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as BASE

NAME = "r22_slot_permutation_transport_tracker"
DESCRIPTION = (
    "The r18 transition-consistency aux plus a private recurrent location-slot memory with "
    "move semantics. Every command opens a slot keyed by its own embedding; a gated SOURCE "
    "read carries the content distribution out of an earlier slot into the new slot, a CLEAR "
    "distribution vacates every earlier slot the command addresses, and an observation-write "
    "gate installs a fresh content atom only when that step actually showed content. The state "
    "is a [slots x content-atoms] mass matrix, so a chain of silent moves composes soft "
    "permutations and the read command's answer is a command-addressed mixture over the "
    "observed contents. Injected as a small-init gated residual on the arch's own prediction "
    "(the D-by-D part is zero-init, only a scalar gain is live at step 0); "
    "train-time observation dropout plus a masked-chain in-window retrieval aux keep the "
    "transport path load-bearing when hop observations are absent."
)

_DEFAULTS = {
    "slot_dk": 48,
    "slot_dh": 128,
    "slot_scale_init": 8.0,
    "slot_occ_eps": 0.01,
    "slot_obs_drop": 0.2,
    "slot_move_bias": -1.0,
    "slot_obs_bias": 1.0,
    "slot_mix_init": 0.05,
    "track_weight": 0.25,
    "track_tau": 0.25,
    "track_mse": 0.10,
    "track_dup_delta": 0.05,
    "track_path_thresh": 0.60,
    "track_min_hops": 2,
    "track_max_hops": 5,
    "track_max_examples": 64,
    "track_ramp_start": 100,
    "track_ramp_steps": 400,
    "track_every": 1,
}

_EPS = 1e-8
_NEG = -1e4


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _clean(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


class _SlotTransportMemory(nn.Module):
    def __init__(self, D, dh, dk, hidden_d, scale_init, move_bias, obs_bias, mix_init):
        super().__init__()
        self.D = int(D)
        self.dh = int(dh)
        self.dk = int(dk)
        self.hidden_d = int(hidden_d)
        self.cmd_in = nn.Linear(self.D, self.dh)
        self.cmd_norm = nn.LayerNorm(self.dh)
        self.obs_in = nn.Linear(self.D, self.dh)
        self.obs_norm = nn.LayerNorm(self.dh)
        self.k_slot = nn.Linear(self.dh, self.dk, bias=False)
        self.q_read = nn.Linear(self.dh, self.dk, bias=False)
        self.q_src = nn.Linear(self.dh, self.dk, bias=False)
        self.move_gate = nn.Linear(self.dh, 1)
        self.obs_write_gate = nn.Linear(2 * self.dh, 1)
        self.log_scale = nn.Parameter(torch.full((3,), math.log(max(1e-3, float(scale_init)))))
        self.occ_coef = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.null_logit = nn.Parameter(torch.zeros(3))
        self.inject = nn.Sequential(
            nn.Linear(self.dh + self.hidden_d + 3, self.dh),
            nn.GELU(),
            nn.Linear(self.dh, 1),
        )
        self.mix_out = nn.Linear(self.D, self.D, bias=False)
        self.mix_gain = nn.Parameter(torch.full((1,), float(mix_init)))
        nn.init.zeros_(self.mix_out.weight)
        nn.init.zeros_(self.inject[2].weight)
        nn.init.zeros_(self.inject[2].bias)
        nn.init.constant_(self.move_gate.bias, float(move_bias))
        nn.init.constant_(self.obs_write_gate.bias, float(obs_bias))


def _run_slots(mem, X, O, valid_cmd, valid_obs, occ_eps):
    B, S, _ = X.shape
    C = O.size(1)
    dtype = X.dtype
    device = X.device

    f = F.gelu(mem.cmd_norm(mem.cmd_in(_clean(X))))
    k = _unit(mem.k_slot(f))
    q_read = _unit(mem.q_read(f))
    q_src = _unit(mem.q_src(f))

    vc = valid_cmd.to(dtype)
    vo = valid_obs.to(dtype)
    move = torch.sigmoid(mem.move_gate(f)).squeeze(-1) * vc

    g_obs = F.gelu(mem.obs_norm(mem.obs_in(_clean(O))))
    write = torch.sigmoid(
        mem.obs_write_gate(torch.cat([f[:, :C], g_obs], dim=-1))
    ).squeeze(-1) * vo

    scale = torch.exp(mem.log_scale.clamp(-4.0, 4.0))
    null = mem.null_logit.view(1, 3)
    eye_s = torch.eye(S, device=device, dtype=dtype)
    eye_c = torch.eye(C, device=device, dtype=dtype)

    A = X.new_zeros(B, S, C)
    live = X.new_zeros(B, S)
    reads = []
    stats = []
    for t in range(S):
        occ = A.sum(dim=-1)
        log_occ = torch.log(occ.clamp_min(0.0) + float(occ_eps))
        avail = live > 0
        sim_read = torch.einsum("bd,bsd->bs", q_read[:, t], k)
        sim_src = torch.einsum("bd,bsd->bs", q_src[:, t], k)
        sim_clear = torch.einsum("bd,bsd->bs", k[:, t], k)

        l_read = (scale[0] * sim_read + mem.occ_coef[0] * log_occ).masked_fill(~avail, _NEG)
        l_src = (scale[1] * sim_src + mem.occ_coef[1] * log_occ).masked_fill(~avail, _NEG)
        l_clear = (scale[2] * sim_clear).masked_fill(~avail, _NEG)

        w_read = torch.softmax(torch.cat([l_read, null[:, 0:1].expand(B, 1)], dim=1), dim=1)[:, :S]
        w_src = torch.softmax(torch.cat([l_src, null[:, 1:2].expand(B, 1)], dim=1), dim=1)[:, :S]
        w_clear = torch.softmax(torch.cat([l_clear, null[:, 2:3].expand(B, 1)], dim=1), dim=1)[:, :S]

        content = torch.einsum("bs,bsc->bc", w_read, A)
        moved = torch.einsum("bs,bsc->bc", w_src, A)
        reads.append(content)
        stats.append(
            torch.stack([w_read.sum(dim=-1), content.sum(dim=-1), content.amax(dim=-1)], dim=-1)
        )

        take = (move[:, t : t + 1] * w_clear).clamp(0.0, 1.0)
        A = A * (1.0 - take).unsqueeze(-1)

        row = move[:, t : t + 1] * moved
        if t < C:
            gate = write[:, t : t + 1]
            row = (1.0 - gate) * row + gate * eye_c[t].view(1, C)
        row = row * vc[:, t : t + 1]
        A = A + eye_s[t].view(1, S, 1) * row.unsqueeze(1)
        live = live + eye_s[t].view(1, S) * vc[:, t : t + 1]

    weights = torch.stack(reads, dim=1)
    mix = torch.einsum("bsc,bcd->bsd", weights, O * vo.unsqueeze(-1))
    return mix, f, torch.stack(stats, dim=1), weights


@torch.no_grad()
def _mine_chains(cmd, valid, thresh, min_hops, max_hops):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(_clean(cmd))
    sim = torch.bmm(cu, cu.transpose(1, 2))
    pos = torch.arange(maxn, device=device)
    vb = valid.bool()
    fwd = (sim > float(thresh)) & vb.unsqueeze(1) & (pos.unsqueeze(1) < pos.unsqueeze(0)).unsqueeze(0)
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    nxt = torch.where(fwd, posf, torch.full_like(posf, maxn)).amin(dim=2)

    cur = pos.view(1, maxn).expand(B, maxn).clone()
    alive = vb.clone()
    depth = torch.zeros(B, maxn, dtype=torch.long, device=device)
    path = [cur.clone()]
    hops = max(1, int(max_hops))
    for _ in range(hops):
        nx = torch.gather(nxt, 1, cur)
        step_ok = alive & (nx < maxn)
        cur = torch.where(step_ok, nx.clamp(max=maxn - 1), cur)
        depth = depth + step_ok.long()
        alive = step_ok
        path.append(cur.clone())

    stacked = torch.stack(path, dim=0)
    end = stacked.gather(0, depth.unsqueeze(0)).squeeze(0)
    ok = (depth >= int(min_hops)) & vb & torch.gather(vb, 1, end)
    score = torch.where(ok, depth, torch.full_like(depth, -1))
    best = score.argmax(dim=1)
    bidx = torch.arange(B, device=device)
    rows = torch.nonzero(score[bidx, best] >= int(min_hops), as_tuple=False).squeeze(1)
    inter = torch.zeros(B, maxn, dtype=torch.bool, device=device)
    if rows.numel() == 0:
        empty = rows.new_zeros(0)
        return rows, empty, empty, inter
    anchor = best[rows]
    sel_depth = depth[rows, anchor]
    sel_end = end[rows, anchor]
    for h in range(1, hops):
        use = h < sel_depth
        if bool(use.any().item()):
            node = stacked[h][rows, anchor]
            inter[rows[use], node[use]] = True
    return rows, sel_end, sel_depth, inter


def _chain_loss(cfg, batch):
    mem = cfg["_mem"]
    tok = batch["tok"]
    vmask = batch["cmd_mask"].bool()
    B, maxn = vmask.shape
    if maxn < 3 or B < 1:
        return 0.0
    X = tok[:, 0::2][:, :maxn]
    O = tok[:, 1::2][:, :maxn]
    rows, ends, sel_depth, inter = _mine_chains(
        X,
        vmask,
        float(cfg["track_path_thresh"]),
        int(cfg["track_min_hops"]),
        int(cfg["track_max_hops"]),
    )
    if rows.numel() < 1:
        return 0.0
    cap = int(cfg["track_max_examples"])
    if rows.numel() > cap:
        order = torch.topk(sel_depth.to(torch.float32), cap).indices
        rows = rows[order]
        ends = ends[order]
        sel_depth = sel_depth[order]

    mix, _, _, _ = _run_slots(mem, X, O, vmask, vmask & ~inter, float(cfg["slot_occ_eps"]))
    pred = mix[rows, ends]
    cand = _clean(O[rows]).detach()
    target = cand[torch.arange(rows.numel(), device=tok.device), ends]

    dist = (pred.unsqueeze(1) - cand).pow(2).mean(dim=-1)
    with torch.no_grad():
        dup = (target.unsqueeze(1) - cand).pow(2).mean(dim=-1) < float(cfg["track_dup_delta"])
        keep = vmask[rows] & ~dup
        keep[torch.arange(rows.numel(), device=tok.device), ends] = True
    logits = (-dist / float(cfg["track_tau"])).masked_fill(~keep, _NEG)

    w = sel_depth.to(pred.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()
    ce = F.cross_entropy(logits, ends, reduction="none")
    mse = (pred - target).pow(2).mean(dim=-1)
    total = (w * ce).sum() + float(cfg["track_mse"]) * (w * mse).sum()
    return total if bool(torch.isfinite(total).item()) else 0.0


def wrap(net, D, **params):
    cfg = BASE.wrap(net, D, **params)
    private = dict(_DEFAULTS)
    private.update(params)
    cfg.update(private)
    cfg["_track_step"] = 0
    hidden_d = getattr(net, "d", None)
    cfg["_track_disabled"] = not isinstance(hidden_d, int)
    if cfg["_track_disabled"]:
        return cfg

    rng = torch.random.get_rng_state()
    mem = _SlotTransportMemory(
        int(D),
        int(private["slot_dh"]),
        int(private["slot_dk"]),
        int(hidden_d),
        float(private["slot_scale_init"]),
        float(private["slot_move_bias"]),
        float(private["slot_obs_bias"]),
        float(private["slot_mix_init"]),
    )
    torch.random.set_rng_state(rng)
    net.add_module("r22_slot_transport_memory", mem)
    cfg["_mem"] = mem
    original = net.forward
    cfg["_original_forward"] = original
    obs_drop = float(private["slot_obs_drop"])
    occ_eps = float(private["slot_occ_eps"])

    def forward(tok_emb, types, key_pad):
        pred, hidden = original(tok_emb, types, key_pad)
        if types is None or tok_emb.dim() != 3 or tok_emb.size(1) < 2:
            return pred, hidden
        if hidden.dim() != 3 or hidden.size(-1) != mem.hidden_d:
            return pred, hidden
        if types.dim() != 2 or types.shape != tok_emb.shape[:2]:
            return pred, hidden
        if key_pad is None:
            live = torch.ones(tok_emb.shape[:2], dtype=torch.bool, device=tok_emb.device)
        else:
            live = ~key_pad.bool()
        X = tok_emb[:, 0::2]
        O = tok_emb[:, 1::2]
        vc = live[:, 0::2]
        vo = live[:, 1::2]
        layout = ((types[:, 0::2] == 0) | ~vc).all(dim=1) & ((types[:, 1::2] == 1) | ~vo).all(dim=1)
        if mem.training and obs_drop > 0.0:
            vo = vo & (torch.rand(vo.shape, device=vo.device) >= obs_drop)
        mix, f, stats, _ = _run_slots(mem, X, O, vc, vo, occ_eps)
        gate = torch.sigmoid(mem.inject(torch.cat([f, hidden[:, 0::2].to(f.dtype), stats], dim=-1)))
        delta = gate * (mem.mix_gain * mix + mem.mix_out(mix))
        delta = _clean(delta) * (vc & layout.unsqueeze(1)).unsqueeze(-1).to(delta.dtype)
        out = pred.clone()
        out[:, 0::2] = pred[:, 0::2] + delta.to(pred.dtype)
        return out, hidden

    net.forward = forward
    return cfg


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    base_term = BASE.aux_loss(cfg, batch, net, device)
    if cfg is None or cfg.get("_track_disabled", True):
        return base_term
    if float(cfg["track_weight"]) <= 0.0:
        return base_term
    if not BASE._interleave_layout_ok(batch):
        return base_term
    cfg["_track_step"] = int(cfg.get("_track_step", 0)) + 1
    step = cfg["_track_step"]
    if step % max(1, int(cfg["track_every"])) != 0:
        return base_term
    ramp = _smoothstep(
        (step - float(cfg["track_ramp_start"])) / max(1.0, float(cfg["track_ramp_steps"]))
    )
    if ramp <= 0.0:
        return base_term
    return base_term + float(cfg["track_weight"]) * ramp * _chain_loss(cfg, batch)


def leak_safe(mod, params):
    if not BASE.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        v = {key: float(p[key]) for key in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(x) for x in v.values()):
        return False
    return all(
        [
            v["slot_dk"] >= 8,
            v["slot_dh"] >= 16,
            v["slot_scale_init"] > 0.0,
            v["slot_occ_eps"] > 0.0,
            0.0 <= v["slot_obs_drop"] < 1.0,
            abs(v["slot_mix_init"]) <= 1.0,
            v["track_weight"] >= 0.0,
            v["track_tau"] > 0.0,
            v["track_mse"] >= 0.0,
            v["track_dup_delta"] >= 0.0,
            -1.0 <= v["track_path_thresh"] < 1.0,
            v["track_min_hops"] >= 1,
            v["track_max_hops"] >= v["track_min_hops"],
            v["track_max_examples"] >= 1,
            v["track_ramp_start"] >= 0.0,
            v["track_ramp_steps"] >= 1.0,
            v["track_every"] >= 1,
        ]
    )
