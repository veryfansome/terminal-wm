"""Training closure for the world model. This module is the TRAIN half of the eval: assemble a
genome (arch / optim / batcher / objective / target / stream / head), run the fixed training loop
against a capability-pack root, and hand back the trained net. It owns nothing about scoring —
the compositional-depth measurement lives in the probe/CA modules that call `_train`.

Also owns the two seams the training path needs on the way in:
  - `_cached_encode`: the fail-closed encode gate (bench_versions + cache_meta.json + the
    embedding cache itself must all be present; a missing cache RAISES rather than silently
    re-encoding the root with a possibly-wrong encoder).
  - `_strip_target_only`: the strip seam that hides target-only aux keys from genome code.
"""

import pathlib

import torch

from realenv import seq_worldmodel as M
from evolve import genome as G
from evolve import bench_versions as BV

D = M.D


def _cached_encode(data_root, split, model, device):
    """Harness-owned wrapper around M.cached_encode (layering: the fail-closed gate consults
    evolve-side concepts — bench_versions + cache_meta.json — so realenv stays evolve-free). On a
    v3-policy root it REQUIRES the root-level cache_meta.json + perception stamp to match
    expectations and RAISES otherwise (a stamp-less/stale v3 cache can never be scored); v1/v2 roots
    pass straight through untouched (bit-identical). ALL harness cached_encode calls route here.

    The embedding cache itself is also REQUIRED: M.cached_encode falls back to encoding the split
    from scratch with `model` when emb-seq-<split>.pt is absent, which would silently re-encode a
    root with a different encoder than the one its cache_meta stamp was written for. Fail closed."""
    if BV.is_v3_policy(data_root):
        BV.require_v3_cache(data_root)
    cache = pathlib.Path(data_root) / f"emb-seq-{split}.pt"
    if not cache.exists():
        raise FileNotFoundError(
            f"missing embedding cache {cache} — refusing to re-encode {data_root} [{split}] on the "
            f"fly (M.cached_encode would silently encode with model={model!r}, which need not be the "
            f"encoder this root's cache_meta.json was stamped with). Re-run the encode step "
            f"(evolve.reencode) for this root, then score again.")
    return M.cached_encode(data_root, split, model, device)


def _strip_target_only(seqs):
    """The strip seam. The two places genome stream code receives seq dicts (stream.collate,
    stream.flatten_predictions) and the batcher's fit all receive a per-sequence shallow COPY with
    the target-only keys (exit_cls, z_delta) REMOVED — so the v3 aux channels are structurally
    invisible to genome code. Identity pass-through (returns the SAME list) when no seq carries
    those keys: v1/v2 = zero-cost, bit-identical."""
    keys = ("exit_cls", "z_delta")
    if not any(k in s for s in seqs for k in keys):
        return seqs
    return [{k: v for k, v in s.items() if k not in keys} for s in seqs]


def _train(genome, fit, device, loss_fn, seed, steps, target_mod, stream, head=None, head_p=None):
    """Train the world model with the genome's objective + arch + target + stream (+ optional head).
    Returns (net, ok); ok=False on NaN. The objective's loss compares the model's cmd-position
    prediction to target_mod.make_target(z_obs, z_prev) — e.g. the raw next obs (identity) or the
    residual z_obs - z_prev (delta). z_prev is the previous observation (strict-causal shift of tgt)."""
    torch.manual_seed(seed)
    build, aparams = G.load_arch(genome)
    net = build(**aparams)
    if getattr(target_mod, "LEARNED", False):
        # learned-target extension: the target impl provides an nn.Module (make_target/to_obs/reg)
        # whose params are registered on the net so the genome's optimizer trains them jointly.
        # The eval stays in the FIXED obs space (to_obs must reconstruct), which keeps a learned
        # target honest: collapsing the target space breaks reconstruction and is scored down.
        net.target_module = target_mod.make(D)
    head_state = None
    if head is not None:
        # head-axis extension: wrap may re-point net.forward and register readout/aux params on
        # net (trained jointly). aux_loss adds a train-only self-supervised term; passthrough is
        # a no-op returning None + 0.0. Must run BEFORE make_opt so aux params are optimized.
        head_state = head.wrap(net, D, **(head_p or {}))
    net = net.to(device)
    make_opt, bs = G.load_optim(genome)
    opt, sched = make_opt(net.parameters(), steps)
    # strip seam: the batcher and stream.collate only ever see the stripped fit (target-only
    # keys removed). Identity pass-through for v1/v2 (no such keys) -> bit-identical.
    fit_stripped = _strip_target_only(fit)
    aux_live = fit_stripped is not fit   # v3 aux channels present -> plumbing active (but dormant)
    # objective contract extension (opt-in, backward-compatible): an objective module may set
    # WANTS_CTX=True to receive a third arg `ctx` with the causal side-info aligned to pred/tgt —
    # {"cmd": command embedding per cmd-row, "prev": previous-obs per cmd-row}. Causal only (the
    # command is a model INPUT; prev is the strict-causal shift) — NO future obs, so leakage_ok
    # still holds. Objectives without the flag are called loss(pred, tgt) EXACTLY as before.
    import sys as _sys
    _obj_wants_ctx = getattr(_sys.modules.get(getattr(loss_fn, "__module__", None)), "WANTS_CTX", False)
    next_batch = G.load_batcher(genome)(fit_stripped, bs, seed)
    for step in range(1, steps + 1):
        idx = next_batch(step, steps)
        if len(idx) != bs or min(idx) < 0 or max(idx) >= len(fit_stripped):
            raise ValueError("batcher contract violation (len/bounds)")
        b = stream.collate([fit_stripped[i] for i in idx], device)
        if aux_live:
            # DORMANT aux-target plumbing: the harness-held ORIGINAL (unstripped) seq
            # dicts for this batch, indexed by the batcher's indices — the attach point for
            # multi-channel aux targets (exit_cls/z_delta). No sanctioned consumer in v3.0.
            _aux_originals = [fit[i] for i in idx]  # noqa: F841
        pred_full, _ = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = stream.extract_cmd_pred(pred_full, b)           # [B, maxn, D]
        tgt_full = b["tgt"]                                        # [B, maxn, D] = z_obs per step
        prev_full = torch.cat([torch.zeros_like(tgt_full[:, :1]), tgt_full[:, :-1]], dim=1)
        m = b["cmd_mask"]
        pred, tgt, prev = cmd_pred[m], tgt_full[m], prev_full[m]
        tmod = getattr(net, "target_module", None)
        if tmod is not None:
            _target = tmod.make_target(tgt, prev); _reg = tmod.reg()
        else:
            _target = target_mod.make_target(tgt, prev); _reg = 0.0
        if _obj_wants_ctx:
            # command-INPUT embedding per cmd-row (explicit stream method, aligned to cmd_mask) —
            # causal side-info for content-routing losses. A stream that lacks extract_cmd_input
            # fails LOUD (AttributeError) rather than silently misaligning a ctx objective.
            ctx = {"cmd": stream.extract_cmd_input(b)[m], "prev": prev}
            loss = loss_fn(pred, _target, ctx) + _reg
        else:
            loss = loss_fn(pred, _target) + _reg
        if head is not None:
            loss = loss + head.aux_loss(head_state, b, net, device)  # 0.0 for passthrough
        if not torch.isfinite(loss):
            return net, False
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
    return net, True
