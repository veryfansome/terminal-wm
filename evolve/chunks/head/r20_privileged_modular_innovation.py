"""R20 head: privileged shadow execution with modular causal innovations.

The ordinary R18 forward and its transition-consistency auxiliary are preserved.
A private student is trained in the same pass on mined mutation-to-read endpoints.
It receives two read-only executions of the R18 stack: a no-mutation counterfactual
read and the exact obs-missing [prefix, mutation, PAD, read] shadow execution.
A command-routed mixture of residual experts predicts only the innovation over the
counterfactual base.  The normal fully-observed read at the later command is used
as training-only privileged information, gated on whether it improves on the base.

At inference the private route fires only for the declared odd-length masked layout.
All student inputs are detached and all extra R18 forwards run under no_grad in
eval mode, so the imagination loss sends zero gradient to the R18 stack and consumes
no dropout RNG.  Future observations are labels only.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CHAMP


NAME = "r20_privileged_modular_innovation"
DESCRIPTION = (
    "Champion R18 head plus a read-only dual-execution imagination student: compare the "
    "champion's counterfactual [prefix, read] prediction with its exact masked "
    "[prefix, mutation, PAD, read] state, then add a mutation-command-routed mixture of "
    "independent residual experts. A truth-gated full-history prediction is privileged "
    "training-only supervision. Fully observed fitness forward and champion aux stay exact; "
    "native obs-missing suffix supplies measurement path b."
)

_DEFAULTS = {
    "imag_weight": 0.05,
    "imag_ramp_start": 300,
    "imag_ramp_len": 900,
    "imag_pairs": 64,
    "imag_every": 1,
    "imag_path_thresh": 0.60,
    "imag_tau": 0.25,
    "imag_dup_delta": 0.05,
    "imag_mse": 0.05,
    "mut_floor": 0.15,
    "width": 192,
    "att_heads": 4,
    "experts": 6,
    "router_temp": 0.50,
    "router_weight": 0.02,
    "priv_weight": 0.10,
    "priv_margin": 0.02,
    "priv_mse": 0.02,
}

_EPS = 1e-8


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


class _ModularInnovation(nn.Module):
    """Counterfactual prior plus operation-routed residual experts.

    No parameter has shape (768,768) or 64 rows, so the harness optimizer routes
    every private parameter to its ordinary AdamW group.
    """

    def __init__(self, D, hidden_d, width=192, heads=4, experts=6, router_temp=0.5):
        super().__init__()
        heads = max(1, int(heads))
        width = max(heads, int(width))
        width = width - width % heads
        self.width = width
        self.heads = heads
        self.dk = width // heads
        self.experts_n = max(2, int(experts))
        self.router_temp = float(router_temp)

        self.pair_k = nn.Linear(2 * D, width)
        self.pair_v = nn.Linear(2 * D, width)
        self.query = nn.Linear(2 * D, width)
        self.base_in = nn.Linear(D, width, bias=False)
        self.shadow_delta_in = nn.Linear(D, width, bias=False)
        self.shadow_h_in = nn.Linear(int(hidden_d), width, bias=False)
        self.mut_in = nn.Linear(D, width, bias=False)
        self.read_in = nn.Linear(D, width, bias=False)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.out_norm = nn.LayerNorm(width)
        self.router = nn.Linear(D, self.experts_n, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(width, D) for _ in range(self.experts_n)]
        )
        self.amount = nn.Linear(width, 1)

        # Exact counterfactual base at initialization. Expert symmetry is broken by
        # their input weights after the first update because routes vary by command.
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            nn.init.zeros_(expert.bias)
        nn.init.zeros_(self.amount.weight)
        nn.init.constant_(self.amount.bias, -0.5)

    def _attend(self, c_m, c_r, pair, mask):
        N, P, _ = pair.shape
        q = self.query(torch.cat([c_m, c_r], dim=-1)).view(
            N, self.heads, self.dk
        )
        k = self.pair_k(pair).view(N, P, self.heads, self.dk)
        v = self.pair_v(pair).view(N, P, self.heads, self.dk)
        score = torch.einsum("nhd,nphd->nhp", q, k) / math.sqrt(self.dk)
        score = score.masked_fill(~mask.unsqueeze(1), -1e9)
        att = torch.softmax(score, dim=-1)
        live = mask.any(dim=1, keepdim=True).unsqueeze(1)
        att = att * live.to(att.dtype)
        return torch.einsum("nhp,nphd->nhd", att, v).reshape(N, self.width)

    def forward(self, base, shadow_pred, shadow_h, c_m, c_r, pf_cmd, pf_obs, pf_mask):
        base = torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=-1e4)
        shadow_pred = torch.nan_to_num(
            shadow_pred, nan=0.0, posinf=1e4, neginf=-1e4
        )
        pair = torch.nan_to_num(
            torch.cat([pf_cmd, pf_obs], dim=-1),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        context = self._attend(c_m, c_r, pair, pf_mask)
        state = (
            self.base_in(base)
            + self.shadow_delta_in(shadow_pred - base)
            + self.shadow_h_in(shadow_h)
            + self.mut_in(c_m)
            + self.read_in(c_r)
            + context
        )
        state = self.norm(state)
        state = self.out_norm(state + self.ff(state))

        route = torch.softmax(
            self.router(F.normalize(c_m, dim=-1, eps=1e-6))
            / max(self.router_temp, 1e-4),
            dim=-1,
        )
        all_delta = torch.stack([expert(state) for expert in self.experts], dim=1)
        delta = (route.unsqueeze(-1) * all_delta).sum(dim=1)
        # Bounded innovation prevents the R19 norm explosion.
        delta = 3.0 * torch.tanh(delta / 3.0)
        amount = torch.sigmoid(self.amount(state))
        pred = base + amount * delta
        return torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4), route


@torch.no_grad()
def _mine_pairs(cmd, valid, threshold):
    """Nearest later same-path-like command for each possible intervention."""
    B, N, _ = cmd.shape
    cu = F.normalize(
        torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1
    )
    sim = torch.bmm(cu, cu.transpose(1, 2))
    pos = torch.arange(N, device=cmd.device)
    later = pos.view(1, 1, N) > pos.view(1, N, 1)
    candidate = (
        (sim > float(threshold))
        & valid.unsqueeze(1)
        & valid.unsqueeze(2)
        & later
    )
    posf = pos.view(1, 1, N).expand(B, N, N)
    j_idx = torch.where(candidate, posf, posf.new_full((), N)).amin(dim=2)
    keep = (j_idx < N) & valid
    nz = torch.nonzero(keep, as_tuple=False)
    row, mutation = nz[:, 0], nz[:, 1]
    read = j_idx[row, mutation]
    return row, mutation, read, sim[row, mutation, read]


@torch.no_grad()
def _mutation_weight(net, c_m):
    try:
        if (
            hasattr(net, "tr_mut_gate")
            and hasattr(net, "cmd_proj")
            and hasattr(net, "type_emb")
            and hasattr(net, "in_norm")
        ):
            z = torch.zeros(c_m.shape[0], dtype=torch.long, device=c_m.device)
            h = net.in_norm(net.cmd_proj(c_m) + net.type_emb(z))
            return torch.nan_to_num(
                torch.sigmoid(net.tr_mut_gate(h)).squeeze(-1), nan=0.5
            )
    except Exception:
        pass
    return torch.full(
        (c_m.shape[0],), 0.5, dtype=c_m.dtype, device=c_m.device
    )


def _l2_nce(pred, target, weight, tau, dup_delta, mse_weight):
    """Strict retrieval-geometry InfoNCE plus an absolute norm anchor."""
    n, d = pred.shape
    mse_row = (pred - target).pow(2).mean(dim=-1)
    mse = (weight * mse_row).sum()
    if n < 2:
        return mse
    p2 = pred.pow(2).sum(dim=1, keepdim=True)
    t2 = target.pow(2).sum(dim=1)
    dist = (p2 + t2.unsqueeze(0) - 2.0 * pred @ target.t()).clamp_min(0.0)
    dist = dist / float(d)
    with torch.no_grad():
        tt = (
            t2.unsqueeze(1) + t2.unsqueeze(0) - 2.0 * target @ target.t()
        ).clamp_min(0.0) / float(d)
        eye = torch.eye(n, dtype=torch.bool, device=pred.device)
        duplicate = (tt < float(dup_delta)) & ~eye
    logits = (-dist / float(tau)).masked_fill(duplicate, -1e9)
    nll = -F.log_softmax(logits, dim=1).diagonal()
    return (weight * nll).sum() + float(mse_weight) * mse


def _detect_masked_endpoint(key_pad, L, device):
    if key_pad is None or L < 3 or L % 2 == 0:
        return None, None
    valid = ~key_pad.bool()
    n_pair = L // 2
    valid_cmd = valid[:, 0::2]
    valid_obs = valid[:, 1::2]
    pattern = valid_cmd[:, :n_pair] & ~valid_obs & valid_cmd[:, 1:n_pair + 1]
    if not bool(pattern.any().item()):
        return None, None
    pos = torch.arange(n_pair, device=device)
    mutation = torch.where(
        pattern, pos.unsqueeze(0), pos.new_full((1, n_pair), -1)
    ).amax(dim=1)
    rows = torch.nonzero(mutation >= 0, as_tuple=False).squeeze(1)
    mutation = mutation[rows]
    observed = (valid_cmd[:, :n_pair] & valid_obs)[rows]
    prefix_ok = (
        observed | ~(pos.unsqueeze(0) < mutation.unsqueeze(1))
    ).all(dim=1)
    rows, mutation = rows[prefix_ok], mutation[prefix_ok]
    if rows.numel() == 0:
        return None, None
    return rows, mutation


@torch.no_grad()
def _counterfactual_base(orig_forward, net, tok, rows, mutation, read_pos):
    """Run [fully observed prefix before m, read command] through the R18 stack."""
    R = rows.numel()
    Lp = 2 * int(mutation.max().item()) + 1
    pos = torch.arange(Lp, device=tok.device)
    two_m = (2 * mutation).unsqueeze(1)
    source = torch.where(
        pos.unsqueeze(0) < two_m,
        pos.unsqueeze(0).expand(R, Lp),
        read_pos.unsqueeze(1).expand(R, Lp),
    )
    tok2 = torch.gather(
        tok[rows], 1, source.unsqueeze(-1).expand(R, Lp, tok.shape[-1])
    )
    types2 = (pos % 2).long().unsqueeze(0).expand(R, Lp)
    pad2 = pos.unsqueeze(0) > two_m
    was_training = net.training
    if was_training:
        net.eval()
    pred2, _ = orig_forward(tok2, types2, pad2)
    if was_training:
        net.train()
    ar = torch.arange(R, device=tok.device)
    return torch.nan_to_num(
        pred2[ar, 2 * mutation], nan=0.0, posinf=1e4, neginf=-1e4
    ).detach()


@torch.no_grad()
def _dual_execution(orig_forward, net, tok, row, mutation, read):
    """One concatenated no-grad forward computes counterfactual and exact masked states."""
    P = row.numel()
    Lp = 2 * int(mutation.max().item()) + 3
    pos = torch.arange(Lp, device=tok.device)
    two_m = (2 * mutation).unsqueeze(1)
    read_pos = (2 * read).unsqueeze(1)

    src_base = torch.where(
        pos.unsqueeze(0) < two_m,
        pos.unsqueeze(0).expand(P, Lp),
        read_pos.expand(P, Lp),
    )
    pad_base = pos.unsqueeze(0) > two_m

    src_shadow = torch.where(
        pos.unsqueeze(0) <= two_m,
        pos.unsqueeze(0).expand(P, Lp),
        read_pos.expand(P, Lp),
    )
    pad_shadow = ~(
        (pos.unsqueeze(0) <= two_m) | (pos.unsqueeze(0) == two_m + 2)
    )

    source = torch.cat([src_base, src_shadow], dim=0)
    tok2 = torch.gather(
        torch.cat([tok[row], tok[row]], dim=0),
        1,
        source.unsqueeze(-1).expand(2 * P, Lp, tok.shape[-1]),
    )
    types2 = (pos % 2).long().unsqueeze(0).expand(2 * P, Lp)
    pad2 = torch.cat([pad_base, pad_shadow], dim=0)

    was_training = net.training
    if was_training:
        net.eval()
    pred2, hidden2 = orig_forward(tok2, types2, pad2)
    if was_training:
        net.train()
    ar = torch.arange(P, device=tok.device)
    base = pred2[ar, 2 * mutation]
    shadow_pred = pred2[P + ar, 2 * mutation + 2]
    shadow_h = hidden2[P + ar, 2 * mutation + 2]
    return (
        torch.nan_to_num(base, nan=0.0, posinf=1e4, neginf=-1e4).detach(),
        torch.nan_to_num(
            shadow_pred, nan=0.0, posinf=1e4, neginf=-1e4
        ).detach(),
        torch.nan_to_num(shadow_h, nan=0.0, posinf=1e4, neginf=-1e4).detach(),
    )


def wrap(net, D, **params):
    """Install the private module and a branch dead on ordinary even-length streams."""
    champ_params = {
        key: value for key, value in params.items() if key in CHAMP._DEFAULTS
    }
    cfg = CHAMP.wrap(net, D, **champ_params)
    p = dict(_DEFAULTS)
    p.update({key: value for key, value in params.items() if key in _DEFAULTS})
    cfg.update(p)
    cfg["_imag_step"] = 0
    cfg["_disabled_private"] = not callable(getattr(net, "forward", None))
    if cfg["_disabled_private"]:
        return cfg

    rng = torch.get_rng_state()
    try:
        module = _ModularInnovation(
            int(D),
            int(getattr(net, "d", D)),
            width=int(cfg["width"]),
            heads=int(cfg["att_heads"]),
            experts=int(cfg["experts"]),
            router_temp=float(cfg["router_temp"]),
        )
    finally:
        torch.set_rng_state(rng)
    net.add_module("_r20_privileged_modular_innovation", module)
    cfg["_module"] = module
    original_forward = net.forward
    cfg["_orig_forward"] = original_forward
    cfg["_cache"] = None

    def wrapped_forward(tok_emb, types, key_pad):
        pred, hidden = original_forward(tok_emb, types, key_pad)
        L = tok_emb.shape[1] if tok_emb.dim() == 3 else 0
        if L % 2 == 0:
            cfg["_cache"] = (tok_emb, pred, hidden)
            return pred, hidden

        rows, mutation = _detect_masked_endpoint(key_pad, L, tok_emb.device)
        if rows is None:
            return pred, hidden
        read_pos = 2 * mutation + 2
        base = _counterfactual_base(
            original_forward, net, tok_emb, rows, mutation, read_pos
        )
        n_pair = L // 2
        pf_cmd = tok_emb[rows][:, 0::2][:, :n_pair].detach()
        pf_obs = tok_emb[rows][:, 1::2][:, :n_pair].detach()
        valid = ~key_pad.bool()
        pf_valid = (valid[:, 0::2][:, :n_pair] & valid[:, 1::2])[rows]
        pos = torch.arange(n_pair, device=tok_emb.device)
        pf_mask = pf_valid & (pos.unsqueeze(0) < mutation.unsqueeze(1))
        c_m = tok_emb[rows, 2 * mutation].detach()
        c_r = tok_emb[rows, read_pos].detach()
        shadow_pred = pred[rows, read_pos].detach()
        shadow_h = hidden[rows, read_pos].detach()
        imagined, _ = module(
            base, shadow_pred, shadow_h, c_m, c_r, pf_cmd, pf_obs, pf_mask
        )
        out = pred.clone()
        out[rows, read_pos] = imagined.to(out.dtype)
        return out, hidden

    net.forward = wrapped_forward
    return cfg


def _private_loss(cfg, batch, net):
    cache = cfg.get("_cache")
    cfg["_cache"] = None
    if cache is None or not CHAMP._interleave_layout_ok(batch):
        return 0.0
    if cfg.get("_disabled_private", True):
        return 0.0

    tok, full_pred, _ = cache
    valid = batch["cmd_mask"].bool()
    maxn = valid.shape[1]
    if maxn < 2:
        return 0.0
    cmd = tok[:, 0::2, :][:, :maxn]
    obs = tok[:, 1::2, :][:, :maxn]

    row, mutation, read, similarity = _mine_pairs(
        cmd, valid, float(cfg["imag_path_thresh"])
    )
    if row.numel() < 2:
        return 0.0
    weight = similarity * (
        float(cfg["mut_floor"]) + _mutation_weight(net, cmd[row, mutation])
    )
    weight = torch.nan_to_num(weight, nan=0.0).clamp_min(0.0)
    cap = max(2, int(cfg["imag_pairs"]))
    if row.numel() > cap:
        weight, order = torch.topk(weight, cap)
        row, mutation, read = row[order], mutation[order], read[order]

    base, shadow_pred, shadow_h = _dual_execution(
        cfg["_orig_forward"], net, tok, row, mutation, read
    )
    c_m = cmd[row, mutation].detach()
    c_r = cmd[row, read].detach()
    target = obs[row, read].detach()
    teacher = full_pred[row, 2 * read].detach()

    n_pair = maxn
    pf_cmd = cmd[row].detach()
    pf_obs = obs[row].detach()
    pos = torch.arange(n_pair, device=tok.device)
    pf_mask = valid[row] & (pos.unsqueeze(0) < mutation.unsqueeze(1))
    pred, route = cfg["_module"](
        base, shadow_pred, shadow_h, c_m, c_r, pf_cmd, pf_obs, pf_mask
    )

    norm_weight = (
        weight / weight.sum().clamp_min(_EPS)
    ).detach().to(pred.dtype)
    main = _l2_nce(
        pred,
        target,
        norm_weight,
        float(cfg["imag_tau"]),
        float(cfg["imag_dup_delta"]),
        float(cfg["imag_mse"]),
    )

    # InfoMax: sharp per-command routing, but high marginal expert use.
    rp = route.clamp_min(1e-8)
    cond_entropy = -(rp * rp.log()).sum(dim=1).mean()
    marginal = rp.mean(dim=0).clamp_min(1e-8)
    marginal_entropy = -(marginal * marginal.log()).sum()
    router_loss = cond_entropy - marginal_entropy

    # Training-only future-observation teacher. It never sees obs at read itself
    # (the R18 stack is causal), and it is used only where its squared-L2 distance to
    # the target is below the counterfactual base's by a pre-registered margin.
    with torch.no_grad():
        d_teacher = (teacher - target).pow(2).mean(dim=-1)
        d_base = (base - target).pow(2).mean(dim=-1)
        use = (
            d_teacher + float(cfg["priv_margin"]) < d_base
        ).to(pred.dtype)
    student_delta = pred - base
    teacher_delta = teacher - base
    cosine = 1.0 - F.cosine_similarity(
        student_delta, teacher_delta, dim=-1, eps=1e-6
    )
    delta_mse = (student_delta - teacher_delta).pow(2).mean(dim=-1)
    privileged = (
        use * (cosine + float(cfg["priv_mse"]) * delta_mse)
    ).sum() / use.sum().clamp_min(1.0)

    total = (
        main
        + float(cfg["router_weight"]) * router_loss
        + float(cfg["priv_weight"]) * privileged
    )
    if not bool(torch.isfinite(total).item()):
        return 0.0
    return total


def aux_loss(head_state, batch, net, device):
    """Run the R18 auxiliary first, preserving its RNG stream exactly."""
    if head_state is None:
        return 0.0
    champion = CHAMP.aux_loss(head_state, batch, net, device)
    cfg = head_state
    if cfg.get("_disabled_private", True):
        return champion
    cfg["_imag_step"] = int(cfg.get("_imag_step", 0)) + 1
    step = cfg["_imag_step"]
    if step % max(1, int(cfg["imag_every"])) != 0:
        cfg["_cache"] = None
        return champion
    ramp = _smoothstep(
        (step - float(cfg["imag_ramp_start"]))
        / max(1.0, float(cfg["imag_ramp_len"]))
    )
    if ramp <= 0.0 or float(cfg["imag_weight"]) <= 0.0:
        cfg["_cache"] = None
        return champion
    private = _private_loss(cfg, batch, net)
    if isinstance(private, float):
        return champion
    return champion + float(cfg["imag_weight"]) * ramp * private


def leak_safe(mod, params):
    """The normal path is original R18. The odd path uses only prefix, c_m and c_r.

    The masked observation value is key-padded in both the R18 and private
    executions. Later observations and the full-history teacher occur only in the
    train-only auxiliary, so they cannot affect a scored forward prediction.
    """
    params = params or {}
    champ_params = {
        key: value for key, value in params.items() if key in CHAMP._DEFAULTS
    }
    if not CHAMP.leak_safe(mod, champ_params):
        return False
    p = dict(_DEFAULTS)
    p.update({key: value for key, value in params.items() if key in _DEFAULTS})
    try:
        values = {key: float(p[key]) for key in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(value) for value in values.values()):
        return False
    checks = [
        values["imag_weight"] >= 0.0,
        values["imag_ramp_start"] >= 0.0,
        values["imag_ramp_len"] >= 1.0,
        values["imag_pairs"] >= 2.0,
        values["imag_every"] >= 1.0,
        -1.0 <= values["imag_path_thresh"] < 1.0,
        values["imag_tau"] > 0.0,
        values["imag_dup_delta"] >= 0.0,
        values["imag_mse"] >= 0.0,
        values["mut_floor"] >= 0.0,
        values["width"] >= values["att_heads"],
        values["att_heads"] >= 1.0,
        values["experts"] >= 2.0,
        values["router_temp"] > 0.0,
        values["router_weight"] >= 0.0,
        values["priv_weight"] >= 0.0,
        values["priv_margin"] >= 0.0,
        values["priv_mse"] >= 0.0,
    ]
    return all(checks)