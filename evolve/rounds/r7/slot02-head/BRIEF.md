TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

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
  id                r4-11-flatbasin-lsam-tailavg
  its fitness       +0.0225   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r22_observation_occlusion_denoising
  optim               r4_flatbasin_lsam_tailavg   params {"avg_frac": 0.25, "beta2": 0.95, "floor_ratio": 0.12, "hold_frac": 0.3, "key_d": 64, "lr": 0.0005, "momentum": 0.95, "ns_steps": 5, "rho": 0.05, "rms_match": 0.2, "spectral_cap": 4.0, "spectral_iters": 2, "warmup_frac": 0.04, "wd": 0.0005}
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r4_registry_transport_composite

YOUR PARENT'S CURRENT head IMPL — r4_registry_transport_composite (this is the code you are mutating):
--------------------------------------------------------------------------------
from evolve.chunks.head import r2_dualaddress_move_transport as _TRANSPORT
from evolve.chunks.head import r3_rename_registry_occupancy_routing as _REGISTRY

NAME = "r4_registry_transport_composite"
DESCRIPTION = (
    "Stacks two content-routing memories on one trunk. The rename-registry memory (slots holding "
    "an address key, a content vector and a fill scalar, where a move re-addresses a slot and "
    "leaves its content untouched, with source/destination roles decided at run time by memory "
    "occupancy) wraps the arch first; the dual-address transport memory (a delta-rule "
    "outer-product store whose source and destination keys come from two extractors sharing one "
    "path-to-key projection, writing the value read at the source key into the destination key) "
    "wraps the result. Both inject their read at command positions through a per-dimension scale "
    "initialised to zero, so the composed forward equals the bare arch forward at initialisation. "
    "Both train-time auxiliaries run and are summed: the registry's bridge-mined InfoNCE plus "
    "routing entropy/load terms plus the transition-operator content-preservation term, and the "
    "transport's duplicate-observation-with-an-intervening-mutation InfoNCE on the memory read and "
    "on the final prediction. Genome params are routed by prefix: 'reg_' to the registry, 'trn_' "
    "to the transport. No auxiliary runs at eval."
)

_REG_PREFIX = "reg_"
_TRN_PREFIX = "trn_"

_TRN_OVERRIDES = {"aux_weight": 0.5}


def _split_params(params):
    reg = {}
    trn = dict(_TRN_OVERRIDES)
    unknown = []
    for k, v in (params or {}).items():
        if k.startswith(_REG_PREFIX):
            reg[k[len(_REG_PREFIX):]] = v
        elif k.startswith(_TRN_PREFIX):
            trn[k[len(_TRN_PREFIX):]] = v
        else:
            unknown.append(k)
    return reg, trn, unknown


def wrap(net, D, **params):
    existing = getattr(net, "_registry_transport_state", None)
    if existing is not None:
        return existing

    reg_p, trn_p, _ = _split_params(params)
    reg_state = _REGISTRY.wrap(net, D, **reg_p)
    trn_state = _TRANSPORT.wrap(net, D, **trn_p)

    state = {"registry": reg_state, "transport": trn_state, "D": int(D)}
    net._registry_transport_state = state
    return state


def aux_loss(head_state, batch, net, device):
    if head_state is None:
        return 0.0
    total = _REGISTRY.aux_loss(head_state.get("registry"), batch, net, device)
    total = total + _TRANSPORT.aux_loss(head_state.get("transport"), batch, net, device)
    return total


def leak_safe(mod, params):
    reg_p, trn_p, unknown = _split_params(params)
    if unknown:
        return False
    if not _REGISTRY.leak_safe(mod, reg_p):
        return False
    if not _TRANSPORT.leak_safe(mod, trn_p):
        return False
    return True
--------------------------------------------------------------------------------

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca +0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r5_rawaddress_dualpre_consistency (axis head)
import inspect
import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CH
from evolve.chunks.head import r20_dualpre_transition_consistency as H20

NAME = "r5_rawaddress_dualpre_consistency"
DESCRIPTION = (
    "The r20 dual-pre transition-consistency aux with one change: the mem-pre arm hands the "
    "arch's memory the RAW command tokens as well as the projected command features, whenever "
    "the arch's _transition_reads declares a cmd_raw parameter. An arch whose memory addresses "
    "itself from the raw coded token then produces, inside this no-grad arm, the same memory it "
    "produces at prediction time, so the shared transition operator is supervised on its actual "
    "deployment distribution rather than on a memory addressed some other way. Archs whose "
    "_transition_reads takes no cmd_raw are called exactly as r20 calls them, and the raw-obs arm, "
    "the mining pass, the RNG-draw count and every parameter are r20's."
)

_DEFAULTS = dict(H20._DEFAULTS)


def _accepts_cmd_raw(fn):
    if fn is None or not callable(fn):
        return False
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "cmd_raw" in sig.parameters


def wrap(net, D, **params):
    cfg = H20.wrap(net, D, **params)
    cfg["_mem_raw_cmd"] = _accepts_cmd_raw(getattr(net, "_transition_reads", None))
    return cfg


@torch.no_grad()
def _memory_pre(net, rows_tok, rows_types, valid, device, raw_cmd):
    L = rows_tok.shape[1]
    t = rows_types.long().clamp(0, 1)
    x = torch.where((t == 0).unsqueeze(-1), net.cmd_proj(rows_tok), net.obs_proj(rows_tok))
    x = x + net.type_emb(t) + net.pos_scale * net._positional(L, device, x.dtype).unsqueeze(0)
    x = net.in_norm(x)
    maxn = valid.shape[1]
    x_cmd = x[:, 0::2][:, :maxn]
    obs = rows_tok[:, 1::2][:, :maxn]
    if raw_cmd:
        cmd_raw = rows_tok[:, 0::2][:, :maxn]
        return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn, cmd_raw)
    return net._transition_reads(x_cmd, obs, valid, valid, maxn, maxn)


def aux_loss(head_state, batch, net, device):
    cfg = head_state
    if cfg is None or cfg.get("_disabled", True):
        return 0.0
    if float(cfg.get("aux_weight", 0.0)) <= 0.0:
        return 0.0
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return 0.0
    if not CH._interleave_layout_ok(batch):
        return 0.0

    cfg["_step"] = int(cfg.get("_step", 0)) + 1
    ramp = CH._smoothstep(cfg["_step"] / max(1.0, float(cfg["ramp_steps"])))
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

    r, ti, tk, tj, w = CH._mine_triples(
        cmd, obs, valid, float(cfg["path_thresh"]), float(cfg["change_floor"])
    )
    if r.numel() == 0:
        return 0.0
    if r.numel() > int(cfg["max_examples"]):
        w, order = torch.topk(w, int(cfg["max_examples"]))
        r = r[order]; ti = ti[order]; tk = tk[order]; tj = tj[order]

    w = w.to(cmd.dtype)
    w = (w / w.sum().clamp_min(CH._EPS)).detach()

    cmd_k = torch.nan_to_num(cmd[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    tgt = torch.nan_to_num(obs[r, tj].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    gu = CH._unit(tgt)

    def arm(pre):
        pred = torch.nan_to_num(op(pre, cmd_k), nan=0.0, posinf=1e4, neginf=-1e4)
        pu = CH._unit(pred)
        cos_err = (w * (1.0 - (pu * gu).sum(dim=-1).clamp(-1.0, 1.0))).sum()
        mse_err = (w * (pred - tgt).pow(2).mean(dim=-1)).sum()
        return float(cfg["cos_weight"]) * cos_err + float(cfg["mse_weight"]) * mse_err

    pre_raw = torch.nan_to_num(obs[r, ti].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
    total = arm(pre_raw)

    mem_w = float(cfg.get("mem_arm_w", 0.0))
    if mem_w > 0.0 and not cfg.get("_mem_disabled", True):
        reads = _memory_pre(
            net, tok[sel], batch["types"][sel], valid, device, bool(cfg.get("_mem_raw_cmd", False))
        )
        pre_mem = torch.nan_to_num(reads[r, tk].detach(), nan=0.0, posinf=1e4, neginf=-1e4)
        total = total + mem_w * arm(pre_mem)

    out_loss = float(cfg["aux_weight"]) * ramp * total
    if not bool(torch.isfinite(out_loss).item()):
        return 0.0
    return out_loss


def leak_safe(mod, params):
    return H20.leak_safe(mod, params)

--- r18_transition_forwardmodel_consistency (axis head)
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

--- routed_copy_pathchange_transport (axis head)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NAME = "routed_copy_pathchange_transport"
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
    "wrapped net is the unwrapped net at initialisation. Train-time aux supervises the arch's "
    "shared latent-transition operator, when it exposes one, from TWO disjoint mining rules over "
    "the batch alone. The PRESERVING rule mines steps whose observation duplicates an earlier "
    "step's observation with at least one contentless step in between, ranks the retrieved "
    "content against the contentful observations seen so far under the eval's squared-L2 decision "
    "variable, and asks the operator to reproduce that duplicated content from the last "
    "contentless command before the read. The CHANGING rule mines, for each command, the nearest "
    "earlier and nearest later commands whose embeddings are cosine-similar to it above a "
    "threshold (a same-path signature), keeps only triples whose two observations DIFFER by more "
    "than a floor, and asks the same operator to map the earlier observation through that command "
    "to the later one, weighted by the product of the two path similarities and the amount of "
    "change. The two rules select disjoint triples by construction — one requires the observation "
    "to be unchanged, the other requires it to change — so the operator is supervised on when a "
    "command carries content unchanged and on when it rewrites it, from the same batch."
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
    "path_weight": 1.0,
    "path_row_frac": 0.6,
    "path_thresh": 0.60,
    "path_change_floor": 0.25,
    "path_max_examples": 512,
    "path_cos_weight": 0.10,
    "path_mse_weight": 0.02,
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
    sel_i = i_idx[sel_b, sel_k]
    sel_j = j_idx[sel_b, sel_k]
    sel_w = w[sel_b, sel_k]
    return sel_b, sel_i, sel_k, sel_j, sel_w


def _path_change_loss(cfg, batch, net, device):
    op = getattr(net, "transition_from_emb", None)
    if not callable(op):
        return None
    if float(cfg["path_weight"]) <= 0.0:
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
        have_dup = nz.numel() > 0
        if have_dup:
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

    total = None

    if have_dup:
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

    path_term = _path_change_loss(cfg, batch, net, device)
    if path_term is not None:
        path_term = float(cfg["path_weight"]) * path_term.float()
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
        vals["path_weight"] >= 0.0,
        0.0 < vals["path_row_frac"] <= 1.0,
        -1.0 <= vals["path_thresh"] < 1.0,
        vals["path_change_floor"] >= 0.0,
        vals["path_max_examples"] >= 1.0,
        vals["path_cos_weight"] >= 0.0,
        vals["path_mse_weight"] >= 0.0,
    ]
    return all(checks)

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0225 (r4-11-flatbasin-lsam-tailavg, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
