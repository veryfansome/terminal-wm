# R21 causal matched-prefix prototype for native obs-missing endpoints.
#
# Ordinary interleaved sequences have even length and are returned by the original
# r18 forward byte-for-byte. On an odd [prefix,c_m,PAD,c_r] layout, this
# head retrieves the observation paired with the most c_r-similar valid prefix
# command and conservatively blends that prototype into the native prediction.
# The same code runs in the history-masked arm; with no valid prefix pair it
# returns the native prediction exactly.

import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CHAMP

NAME = "r21_causal_matched_prefix_prototype"
DESCRIPTION = (
    "Champion r18 transition consistency plus a parameter-free causal episodic "
    "read on native masked endpoints: match c_r to a strictly earlier valid "
    "(command, observation) pair, blend its raw observation into the native "
    "prediction under an RMS cap, and return native exactly when no history "
    "evidence exists. Ordinary even-stream forwards and training are untouched."
)

_DEFAULTS = {
    "prototype_match_threshold": 0.60,
    "prototype_blend": 0.50,
    "prototype_max_resid_rms": 0.50,
}
_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(
        x.pow(2).sum(dim=-1, keepdim=True).clamp_min(_EPS)
    )


def _detect_masked_endpoint(types, key_pad):
    """Find the last live cmd, masked obs, live cmd gap in each odd row.

    Earlier tokens may be live (base arm) or masked (history arm). Detection
    therefore cannot distinguish the two arms and does not use mask fractions.
    """
    if (
        key_pad is None
        or types.dim() != 2
        or key_pad.dim() != 2
        or types.shape != key_pad.shape
    ):
        return None
    B, L = types.shape
    if L < 3 or L % 2 == 0:
        return None

    live = ~key_pad.bool()
    candidate = live[:, :-2] & ~live[:, 1:-1] & live[:, 2:]
    pos0 = torch.arange(L - 2, device=types.device)
    candidate = candidate & ((pos0 % 2) == 0).unsqueeze(0)

    score = torch.where(
        candidate,
        pos0.unsqueeze(0),
        torch.full((B, L - 2), -1, device=types.device, dtype=pos0.dtype),
    )
    mutation = score.amax(dim=1)
    rows = torch.nonzero(mutation >= 0, as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return None

    mutation = mutation[rows]
    read = mutation + 2
    pos = torch.arange(L, device=types.device).unsqueeze(0)
    tail_ok = ((~live[rows]) | (pos <= read.unsqueeze(1))).all(dim=1)
    type_ok = (
        (types[rows, mutation] == 0)
        & (types[rows, mutation + 1] == 1)
        & (types[rows, read] == 0)
    )
    keep = tail_ok & type_ok
    if not bool(keep.any().item()):
        return None
    return rows[keep], mutation[keep], read[keep]


def _matched_prefix(
    tok,
    types,
    key_pad,
    rows,
    mutation,
    read,
    threshold,
):
    """Return the nearest raw prefix observation and an evidence-present mask."""
    L = tok.size(1)
    n_pairs = L // 2
    keys = tok[rows, : 2 * n_pairs : 2]
    values = tok[rows, 1 : 2 * n_pairs : 2]

    live = ~key_pad[rows].bool()
    pair_live = (
        live[:, : 2 * n_pairs : 2]
        & live[:, 1 : 2 * n_pairs : 2]
    )
    pair_type = (
        (types[rows, : 2 * n_pairs : 2] == 0)
        & (types[rows, 1 : 2 * n_pairs : 2] == 1)
    )
    command_pos = (
        2 * torch.arange(n_pairs, device=tok.device)
    ).unsqueeze(0)
    eligible = (
        pair_live
        & pair_type
        & (command_pos < mutation.unsqueeze(1))
    )

    query = tok[rows, read]
    similarity = torch.einsum(
        "npd,nd->np", _unit(keys), _unit(query)
    )
    similarity = similarity.masked_fill(~eligible, -2.0)
    best, index = similarity.max(dim=1)
    local = torch.arange(rows.numel(), device=tok.device)
    prototype = values[local, index]
    has_evidence = eligible.any(dim=1) & (best > float(threshold))
    return prototype, has_evidence


def _correct(native, prototype, blend, max_resid_rms):
    delta = float(blend) * (prototype - native)
    rms = delta.pow(2).mean(dim=-1, keepdim=True).add(_EPS).sqrt()
    delta = delta * (float(max_resid_rms) / rms).clamp(max=1.0)
    return native + delta


def wrap(net, D, **params):
    cfg = CHAMP.wrap(net, D, **params)
    private = dict(_DEFAULTS)
    private.update({k: params[k] for k in _DEFAULTS if k in params})
    cfg.update(private)

    original_forward = net.forward
    cfg["_prototype_original_forward"] = original_forward

    def wrapped_forward(tok_emb, types, key_pad):
        # All ordinary train/fitness sequences are even cmd/obs interleaves.
        if tok_emb.size(1) % 2 == 0:
            return original_forward(tok_emb, types, key_pad)

        detected = _detect_masked_endpoint(types, key_pad)
        if detected is None:
            return original_forward(tok_emb, types, key_pad)

        pred, hidden = original_forward(tok_emb, types, key_pad)
        rows, mutation, read = detected
        prototype, has_evidence = _matched_prefix(
            tok_emb,
            types,
            key_pad,
            rows,
            mutation,
            read,
            cfg["prototype_match_threshold"],
        )
        # This is the exact history-arm path: no evidence, no correction.
        if not bool(has_evidence.any().item()):
            return pred, hidden

        selected_rows = rows[has_evidence]
        selected_read = read[has_evidence]
        corrected = _correct(
            pred[selected_rows, selected_read],
            prototype[has_evidence],
            cfg["prototype_blend"],
            cfg["prototype_max_resid_rms"],
        )
        out = pred.clone()
        out[selected_rows, selected_read] = corrected
        return out, hidden

    net.forward = wrapped_forward
    return cfg


def aux_loss(head_state, batch, net, device):
    # Retain the shared transition-consistency objective exactly.
    return CHAMP.aux_loss(head_state, batch, net, device)


def leak_safe(mod, params):
    if not CHAMP.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update(
        {k: (params or {})[k] for k in _DEFAULTS if k in (params or {})}
    )
    try:
        threshold = float(p["prototype_match_threshold"])
        blend = float(p["prototype_blend"])
        max_rms = float(p["prototype_max_resid_rms"])
    except Exception:
        return False
    return (
        all(math.isfinite(x) for x in (threshold, blend, max_rms))
        and -1.0 < threshold < 1.0
        and 0.0 < blend <= 1.0
        and 0.0 < max_rms <= 2.0
    )
