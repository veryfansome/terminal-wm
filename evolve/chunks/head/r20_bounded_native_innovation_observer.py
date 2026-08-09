"""R20 bounded innovation observer for native obs-missing imagination.

The champion forward-model auxiliary is retained. A private module additionally learns
an endpoint-only, hard-bounded correction over the native imagination-write prediction.
It is trained from masked mutation/read suffixes mined exclusively from training batches.
All private-branch inputs are detached, so its loss cannot update the shared world model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CHAMP

NAME = "r20_bounded_native_innovation_observer"
DESCRIPTION = (
    "Champion r18 transition consistency plus a private zero-init bounded innovation "
    "observer for odd [prefix,c_m,PAD,c_r] forwards. TRAIN triples supervise duplicate-"
    "masked Euclidean retrieval and an MSE anchor; shared native predictions and hidden "
    "states are detached, and ordinary even-stream forwards are exactly untouched."
)

_PRIVATE_DEFAULTS = {
    "imag_width": 128,
    "imag_row_frac": 0.6,
    "imag_path_thresh": 0.60,
    "imag_change_floor": 0.25,
    "imag_max_examples": 64,
    "imag_mut_floor": 0.10,
    "imag_mut_temp": 0.10,
    "imag_resid_rms": 0.50,
    "imag_tau": 0.25,
    "imag_mse_weight": 0.20,
    "imag_dupe_cos": 0.98,
    "imag_aux_weight": 0.05,
    "imag_ramp_start": 300,
    "imag_ramp_steps": 900,
    "imag_every": 1,
}
_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


class _BoundedInnovationObserver(nn.Module):
    def __init__(self, D, hidden_d, width, residual_rms):
        super().__init__()
        self.D = int(D)
        self.residual_rms = float(residual_rms)
        in_d = 4 * self.D + int(hidden_d)
        self.body = nn.Sequential(
            nn.Linear(in_d, int(width)),
            nn.LayerNorm(int(width)),
            nn.GELU(),
            nn.Linear(int(width), 2 * int(width)),
            nn.GELU(),
        )
        self.out = nn.Linear(2 * int(width), self.D + 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, native, cmd_m, cmd_r, prefix_obs, endpoint_h):
        feat = torch.cat([native, cmd_m, cmd_r, prefix_obs, endpoint_h], dim=-1)
        raw = self.out(self.body(feat))
        residual = torch.tanh(raw[:, : self.D])
        rms = residual.pow(2).mean(dim=-1, keepdim=True).add(_EPS).sqrt()
        residual = residual * (self.residual_rms / rms).clamp(max=1.0)
        amount = torch.sigmoid(raw[:, self.D :])
        return native + amount * residual


def _prefix_context(tok, valid, mutation_pos):
    obs = tok[:, 1::2, :]
    obs_valid = valid[:, 1::2]
    pos = 2 * torch.arange(obs.size(1), device=tok.device) + 1
    mask = obs_valid & (pos.unsqueeze(0) < mutation_pos.unsqueeze(1))
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(tok.dtype)
    return (obs * mask.unsqueeze(-1).to(tok.dtype)).sum(dim=1) / denom


def _detect_masked_endpoint(types, key_pad):
    if key_pad is None or types.dim() != 2 or key_pad.dim() != 2:
        return None
    B, L = types.shape
    if L < 3 or L % 2 == 0:
        return None
    live = ~key_pad.bool()
    cand = live[:, :-2] & ~live[:, 1:-1] & live[:, 2:]
    pos0 = torch.arange(L - 2, device=types.device)
    cand = cand & ((pos0 % 2) == 0).unsqueeze(0)
    nz = torch.nonzero(cand, as_tuple=False)
    if nz.numel() == 0:
        return None
    rows, mutation = nz[:, 0], nz[:, 1]
    read = mutation + 2
    pos = torch.arange(L, device=types.device).unsqueeze(0)
    prefix_ok = (live[rows] | (pos >= mutation.unsqueeze(1))).all(dim=1)
    tail_ok = ((~live[rows]) | (pos <= read.unsqueeze(1))).all(dim=1)
    type_ok = (
        (types[rows, mutation] == 0)
        & (types[rows, mutation + 1] == 1)
        & (types[rows, read] == 0)
    )
    keep = prefix_ok & tail_ok & type_ok
    if not bool(keep.any().item()):
        return None
    return rows[keep], mutation[keep], read[keep]


def _observer_prediction(observer, tok, valid, pred, hidden, rows, mutation, read):
    native = pred[rows, read].detach()
    cmd_m = tok[rows, mutation].detach()
    cmd_r = tok[rows, read].detach()
    context = _prefix_context(tok[rows], valid[rows], mutation).detach()
    endpoint_h = hidden[rows, read].detach()
    return observer(native, cmd_m, cmd_r, context, endpoint_h)


def _build_masked(cmd, obs, rows, mutation, read):
    cmd = cmd[rows].detach()
    obs = obs[rows].detach()
    mutation = mutation.long()
    read = read.long()
    N, maxn, D = cmd.shape
    L = 2 * int(mutation.max().item()) + 3
    pos = torch.arange(L, device=cmd.device)
    source = (pos // 2).clamp(max=maxn - 1)
    cmd_src = cmd[:, source, :]
    obs_src = obs[:, source, :]
    pair_tok = torch.where((pos % 2 == 0).view(1, L, 1), cmd_src, obs_src)
    live_prefix = pos.unsqueeze(0) < (2 * mutation).unsqueeze(1)
    tok = pair_tok * live_prefix.unsqueeze(-1).to(pair_tok.dtype)
    types = (pos % 2).long().unsqueeze(0).expand(N, -1).clone()
    key_pad = ~live_prefix
    rr = torch.arange(N, device=cmd.device)
    mpos = 2 * mutation
    rpos = mpos + 2
    tok[rr, mpos] = cmd[rr, mutation]
    tok[rr, rpos] = cmd[rr, read]
    types[rr, mpos] = 0
    types[rr, mpos + 1] = 1
    types[rr, rpos] = 0
    key_pad[rr, mpos] = False
    key_pad[rr, mpos + 1] = True
    key_pad[rr, rpos] = False
    return tok, types, key_pad, mpos, rpos


def _retrieval_loss(pred, target, weight, tau, duplicate_cos):
    distance = (pred.unsqueeze(1) - target.unsqueeze(0)).pow(2).mean(dim=-1)
    logits = -distance / float(tau)
    with torch.no_grad():
        similarity = torch.matmul(_unit(target), _unit(target).transpose(0, 1))
        eye = torch.eye(target.size(0), device=target.device, dtype=torch.bool)
        duplicate = (similarity > float(duplicate_cos)) & ~eye
    logits = logits.masked_fill(duplicate, -1e4)
    labels = torch.arange(target.size(0), device=target.device)
    per_example = F.cross_entropy(logits, labels, reduction="none")
    return (weight * per_example).sum()


def _private_loss(cfg, batch, net):
    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if B < 2 or maxn < 3:
        return 0.0
    nrows = max(1, int(math.ceil(B * float(cfg["imag_row_frac"]))))
    selected = torch.randperm(B, device=tok.device)[:nrows]
    cmd = tok[selected, 0::2, :][:, :maxn]
    obs = tok[selected, 1::2, :][:, :maxn]
    valid = cmd_mask[selected]
    rows, earlier, mutation, read, weight = CHAMP._mine_triples(
        cmd,
        obs,
        valid,
        float(cfg["imag_path_thresh"]),
        float(cfg["imag_change_floor"]),
    )
    if rows.numel() < 2:
        return 0.0
    cap = int(cfg["imag_max_examples"])
    if rows.numel() > cap:
        weight, order = torch.topk(weight, cap)
        rows = rows[order]
        earlier = earlier[order]
        mutation = mutation[order]
        read = read[order]

    with torch.no_grad():
        cmd_m = cmd[rows, mutation]
        idx0 = torch.zeros(cmd_m.size(0), dtype=torch.long, device=cmd_m.device)
        cmd_feat = net.in_norm(net.cmd_proj(cmd_m) + net.type_emb(idx0))
        mut_gate = torch.sigmoid(net.tr_mut_gate(cmd_feat)).squeeze(-1)
        soft_mut = torch.sigmoid(
            (mut_gate - 0.5) / float(cfg["imag_mut_temp"])
        )
        floor = float(cfg["imag_mut_floor"])
        weight = weight * (floor + (1.0 - floor) * soft_mut)
        weight = weight.clamp_min(0.0)
        weight = weight / weight.sum().clamp_min(_EPS)

    masked_tok, masked_types, masked_pad, mpos, rpos = _build_masked(
        cmd, obs, rows, mutation, read
    )
    original_forward = cfg["_original_forward"]
    was_training = net.training
    net.eval()
    with torch.no_grad():
        native_pred, native_hidden = original_forward(
            masked_tok, masked_types, masked_pad
        )
    if was_training:
        net.train()
    live = ~masked_pad
    local_rows = torch.arange(masked_tok.size(0), device=masked_tok.device)
    pred = _observer_prediction(
        cfg["_observer"],
        masked_tok,
        live,
        native_pred,
        native_hidden,
        local_rows,
        mpos,
        rpos,
    )
    target = obs[rows, read].detach()
    rank = _retrieval_loss(
        pred,
        target,
        weight,
        float(cfg["imag_tau"]),
        float(cfg["imag_dupe_cos"]),
    )
    mse = (weight * (pred - target).pow(2).mean(dim=-1)).sum()
    total = rank + float(cfg["imag_mse_weight"]) * mse
    return total if bool(torch.isfinite(total).item()) else 0.0


def wrap(net, D, **params):
    cfg = CHAMP.wrap(net, D, **params)
    private = dict(_PRIVATE_DEFAULTS)
    private.update(params)
    cfg.update(private)
    cfg["_observer_step"] = 0
    cfg["_imag_disabled"] = not bool(
        getattr(net, "supports_native_imagwrite", False)
    )
    if cfg["_imag_disabled"]:
        return cfg

    rng = torch.random.get_rng_state()
    observer = _BoundedInnovationObserver(
        D,
        int(getattr(net, "d")),
        int(cfg["imag_width"]),
        float(cfg["imag_resid_rms"]),
    )
    torch.random.set_rng_state(rng)
    net.add_module("r20_native_innovation_observer", observer)
    cfg["_observer"] = observer
    original_forward = net.forward
    cfg["_original_forward"] = original_forward

    def wrapped_forward(tok_emb, types, key_pad):
        if tok_emb.size(1) % 2 == 0:
            return original_forward(tok_emb, types, key_pad)
        detected = _detect_masked_endpoint(types, key_pad)
        if detected is None:
            return original_forward(tok_emb, types, key_pad)
        pred, hidden = original_forward(tok_emb, types, key_pad)
        rows, mutation, read = detected
        valid = ~key_pad.bool()
        corrected = _observer_prediction(
            observer,
            tok_emb,
            valid,
            pred,
            hidden,
            rows,
            mutation,
            read,
        )
        out = pred.clone()
        out[rows, read] = corrected
        return out, hidden

    net.forward = wrapped_forward
    return cfg


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    champion = CHAMP.aux_loss(cfg, batch, net, device)
    if cfg is None or cfg.get("_imag_disabled", True):
        return champion
    if not CHAMP._interleave_layout_ok(batch):
        return champion
    cfg["_observer_step"] = int(cfg.get("_observer_step", 0)) + 1
    step = cfg["_observer_step"]
    if step % int(cfg["imag_every"]) != 0:
        return champion
    start = int(cfg["imag_ramp_start"])
    ramp = _smoothstep(
        (step - start) / max(1.0, float(cfg["imag_ramp_steps"]))
    )
    if ramp <= 0.0 or float(cfg["imag_aux_weight"]) <= 0.0:
        return champion
    private = _private_loss(cfg, batch, net)
    return champion + float(cfg["imag_aux_weight"]) * ramp * private


def leak_safe(mod, params):
    if not CHAMP.leak_safe(mod, params):
        return False
    p = dict(_PRIVATE_DEFAULTS)
    p.update(params or {})
    try:
        vals = {k: float(p[k]) for k in _PRIVATE_DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(v) for v in vals.values()):
        return False
    return all(
        [
            vals["imag_width"] >= 16,
            0.0 < vals["imag_row_frac"] <= 1.0,
            -1.0 <= vals["imag_path_thresh"] < 1.0,
            vals["imag_change_floor"] >= 0.0,
            vals["imag_max_examples"] >= 2,
            0.0 <= vals["imag_mut_floor"] <= 1.0,
            vals["imag_mut_temp"] > 0.0,
            0.0 < vals["imag_resid_rms"] <= 2.0,
            vals["imag_tau"] > 0.0,
            vals["imag_mse_weight"] >= 0.0,
            -1.0 <= vals["imag_dupe_cos"] <= 1.0,
            vals["imag_aux_weight"] >= 0.0,
            vals["imag_ramp_start"] >= 0.0,
            vals["imag_ramp_steps"] >= 1.0,
            vals["imag_every"] >= 1.0,
        ]
    )
