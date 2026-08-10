import math

import torch

from evolve.chunks.head import r18_transition_forwardmodel_consistency as BASE

NAME = 'r21_counterfactual_history_guidance'
DESCRIPTION = (
    'The r18 transition consistency plus parameter-free counterfactual-history '
    'guidance on the frozen odd masked-endpoint layout. The same trunk predicts with '
    'the supplied prefix and with positions before c_m intervention-masked; a '
    '0.2-RMS-capped conditional-minus-no-history latent is added to the conditional '
    'endpoint. The correction is algebraically zero in the history-masked arm. '
    'Ordinary even-stream training and fitness forwards are bit-identical, with no '
    'new parameters, auxiliary loss, optimizer state, or RNG consumption.'
)

_DEFAULTS = {
    'history_guidance_gain': 1.0,
    'history_guidance_rms': 0.20,
}
_EPS = 1e-8


def _detect_masked_endpoint(types, key_pad):
    if (
        key_pad is None
        or types.dim() != 2
        or key_pad.dim() != 2
        or types.shape != key_pad.shape
    ):
        return None
    batch, length = types.shape
    if length < 3 or length % 2 == 0:
        return None

    live = ~key_pad.bool()
    pos0 = torch.arange(length - 2, device=types.device)
    candidate = live[:, :-2] & ~live[:, 1:-1] & live[:, 2:]
    candidate = candidate & ((pos0 % 2) == 0).unsqueeze(0)
    score = torch.where(
        candidate,
        pos0.unsqueeze(0),
        torch.full(
            (batch, length - 2),
            -1,
            device=types.device,
            dtype=pos0.dtype,
        ),
    )
    mutation = score.amax(dim=1)
    rows = torch.nonzero(mutation >= 0, as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return None

    mutation = mutation[rows]
    read = mutation + 2
    pos = torch.arange(length, device=types.device).unsqueeze(0)
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


def _counterfactual_pad(key_pad, rows, mutation):
    counterfactual = key_pad.bool().clone()
    pos = torch.arange(key_pad.size(1), device=key_pad.device).unsqueeze(0)
    counterfactual[rows] = counterfactual[rows] | (
        pos < mutation.unsqueeze(1)
    )
    return counterfactual


def _guided(native, no_history, gain, cap):
    delta = float(gain) * (native - no_history)
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
    rms = delta.pow(2).mean(dim=-1, keepdim=True).add(_EPS).sqrt()
    delta = delta * (float(cap) / rms).clamp(max=1.0)
    return native + delta


def wrap(net, D, **params):
    cfg = BASE.wrap(net, D, **params)
    private = dict(_DEFAULTS)
    private.update({k: params[k] for k in _DEFAULTS if k in params})
    cfg.update(private)

    original_forward = net.forward
    cfg['_history_guidance_original_forward'] = original_forward

    def wrapped_forward(tok_emb, types, key_pad):
        if tok_emb.size(1) % 2 == 0 or net.training:
            return original_forward(tok_emb, types, key_pad)

        detected = _detect_masked_endpoint(types, key_pad)
        if detected is None:
            return original_forward(tok_emb, types, key_pad)

        conditional_pred, conditional_hidden = original_forward(
            tok_emb, types, key_pad
        )
        rows, mutation, read = detected
        no_history_pad = _counterfactual_pad(key_pad, rows, mutation)
        no_history_pred, _ = original_forward(
            tok_emb, types, no_history_pad
        )

        guided = _guided(
            conditional_pred[rows, read],
            no_history_pred[rows, read],
            cfg['history_guidance_gain'],
            cfg['history_guidance_rms'],
        )
        out = conditional_pred.clone()
        out[rows, read] = guided
        return out, conditional_hidden

    net.forward = wrapped_forward
    return cfg


def aux_loss(head_state, batch, net, device):
    return BASE.aux_loss(head_state, batch, net, device)


def leak_safe(mod, params):
    if not BASE.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update({k: (params or {})[k] for k in _DEFAULTS if k in (params or {})})
    try:
        gain = float(p['history_guidance_gain'])
        cap = float(p['history_guidance_rms'])
    except Exception:
        return False
    return (
        math.isfinite(gain)
        and math.isfinite(cap)
        and 0.0 <= gain <= 4.0
        and 0.0 < cap <= 2.0
    )
