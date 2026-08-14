TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'head': Expose wrap(net, D, **params) -> a head state or None, aux_loss(head_state, batch, net, device) -> a scalar or zero, and leak_safe(mod, params) -> bool. wrap runs BEFORE the optimizer is built, so registered readout and auxiliary parameters are optimized. aux_loss is train-time only. Two hazards a wrapper must avoid: a parent-child module cycle (hold the base net unregistered, or moving to device recurses), and forward recursion when re-pointing forward (save the original bound method first). A head that recomputes the prediction from the trunk hidden state silently bypasses any architecture whose prediction is not a per-position function of that state.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any head impl:
  wrap(net, D, **params) -> a head state, or None for no head
      Called BEFORE the optimizer is built, so any readout or auxiliary parameters it
      registers on net are optimized. Two hazards a wrapper must avoid: a parent-child
      module cycle (hold the base net unregistered, or moving to device recurses), and
      forward recursion when re-pointing net.forward (save the original bound method first).
  aux_loss(head_state, batch, net, device) -> scalar tensor or 0.0; train-time only.
  leak_safe(mod, params) -> bool; asserted before scoring.
"""

import torch

NAME_BASELINE = "baseline_passthrough"
DESCRIPTION_BASELINE = ("Arch's own Linear readout, unchanged; no aux loss. "
                        "Bit-identical to the pre-axis harness readout.")


def wrap(net, D, **params):
    return None


# A hard 0.0 (not a zero tensor) so `main + aux` is main bit-for-bit and archived
# fitnesses replay exactly.
def aux_loss(head_state, batch, net, device):
    return 0.0


def leak_safe(mod, params):
    return True


NAME = NAME_BASELINE
DESCRIPTION = DESCRIPTION_BASELINE
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-13-obspresent-imagination
  its fitness       -0.0037   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r19_obspresent_imagination_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

YOUR PARENT'S CURRENT head IMPL — r18_transition_forwardmodel_consistency (this is the code you are mutating):
--------------------------------------------------------------------------------
import math

import torch
import torch.nn as nn

NAME = "r18_transition_forwardmodel_consistency"
DESCRIPTION = (
    "Train-only forward-model consistency on the r18 latent-transition arch's shared operator: "
    "mines same-path (pre, mutating-cmd, future-read) triples and requires f(obs_pre, cmd) to "
    "reconstruct the future post-mutation observation (cosine+MSE, change-weighted). Eval forward "
    "untouched; disabled (0.0) on archs without the transition operator. Co-designed head half of "
    "the r18 transition world-model stack."
)

_DEFAULTS = {
    "row_frac": 0.6,
    "path_thresh": 0.60,
    "change_floor": 0.25,
    "max_examples": 512,
    "cos_weight": 0.10,
    "mse_weight": 0.02,
    "aux_weight": 1.0,
    "ramp_steps": 400,
}

_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _interleave_layout_ok(b):
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


def wrap(net, D, **params):
    cfg = dict(_DEFAULTS)
    cfg.update(params)
    cfg["D"] = int(D)
    cfg["_step"] = 0
    cfg["_disabled"] = not callable(getattr(net, "transition_from_emb", None))
    return cfg


@torch.no_grad()
def _mine_triples(cmd, obs, valid, path_thresh, change_floor):
    B, maxn, _ = cmd.shape
    device = cmd.device
    cu = _unit(torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4))
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
    sel_i = i_idx[sel_b, sel_k]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_i, sel_k, sel_j, sel_w


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not _interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = _smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
    if ramp <= 0.0:
        return 0.0

    tok = batch["tok"]
    cmd_mask = batch["cmd_mask"].bool()
    B, maxn = cmd_mask.shape
    if maxn < 3:
        return 0.0

    nrows = max(1, int(math.ceil(B * float(cfg["row_frac"]))))
    sel = torch.randperm(B, device=device)[:nrows]
    cmd = tok[sel][:, 0::2][:, :maxn]
    obs = tok[sel][:, 1::2][:, :maxn]
    valid = cmd_mask[sel]

    r, ti, tk, tj, w = _mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(_EPS)).detach()

    pre = obs[r, ti].detach()
    cmd_k = cmd[r, tk].detach()
    tgt = obs[r, tj].detach()
    pre = torch.nan_to_num(pre, nan=0.0, posinf=1e4, neginf=-1e4)
    cmd_k = torch.nan_to_num(cmd_k, nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(tgt, nan=0.0, posinf=1e4, neginf=-1e4)

    pred = op(pre, cmd_k)
    pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)

    pu, gu = _unit(pred), _unit(tgt)
    cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
    mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
    total = float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


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
        0.0 < vals["row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["change_floor"] >= 0.0,
        vals["max_examples"] >= 1.0,
        vals["cos_weight"] >= 0.0,
        vals["mse_weight"] >= 0.0,
        vals["aux_weight"] >= 0.0,
        vals["ramp_steps"] >= 1.0,
    ]
    return all(checks)
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r22_retrieval_composition_renderer
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r2_routedepth_tilt_readcollision_blocks   params {"alpha_max": 2.5, "ans_max_sim": 0.995, "chunk": 256, "coll_share": 0.5, "hard_frac_max": 0.75, "max_centers": 8000, "n_blocks": 2, "nb_k": 16, "proj_dim": 128, "ramp_frac": 0.3, "sim_thresh": 0.85}
  stream              ·
  head                r18_transition_forwardmodel_consistency

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r2_dualaddress_move_transport (axis head)
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

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0037 (g0-13-obspresent-imagination, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
