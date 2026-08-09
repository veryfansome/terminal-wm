'''R20 head: counterfactual predictive-state endpoint imagination.

The normal R18 prediction path is numerically untouched. The wrapper caches the
strictly-causal hidden belief and transition-memory read at each command. A
train-only endpoint objective applies the shared transition to its deployment-time
memory input, then renders the resulting state through the later read command.
Counterfactual history swaps force the route to use information beyond the command
suffix. Only a small explicitly scaled gradient reaches shared parameters.
'''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from evolve.chunks.head import r18_transition_forwardmodel_consistency as CHAMP


NAME = 'r20_predictive_state_counterfactual_renderer'
DESCRIPTION = (
    'R18 forward-model consistency plus a native obs-missing predictive-state route: '
    'the exact pre-mutation memory state is advanced by the shared transition and a '
    'read-command renderer predicts the endpoint; suffix-matched counterfactual history '
    'swaps force conditional history use, with a small gradient budget into shared state.'
)

_IMAG_DEFAULTS = {
    'imag_weight': 0.15,
    'imag_ramp_steps': 400,
    'imag_path_thresh': 0.60,
    'imag_max_examples': 128,
    'imag_tau': 0.25,
    'imag_dup_delta': 0.05,
    'imag_mse': 0.05,
    'cf_weight': 0.10,
    'cf_thresh': 0.72,
    'cf_margin': 0.10,
    'cf_tau': 0.25,
    'mut_floor': 0.15,
    'shared_grad': 0.05,
    'imag_width': 192,
}


def _route_grad(x, fraction):
    '''Identity in the forward pass, fraction-scaled gradient in backward.'''
    rho = float(fraction)
    return x.detach() + rho * (x - x.detach())


class _PredictiveStateRenderer(nn.Module):
    '''Factorized belief -> post-mutation state -> command-specific observation view.'''

    def __init__(self, D, hidden_d, width):
        super().__init__()
        width = int(width)
        self.post = nn.Linear(D, width, bias=False)
        self.history = nn.Linear(hidden_d, width, bias=False)
        self.payload = nn.Linear(D, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        self.read_film = nn.Linear(D, 2 * width)
        self.view_norm = nn.LayerNorm(width)
        self.out = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, D),
        )

    def forward(self, post_state, history, mutation_cmd, read_cmd):
        state = self.state_norm(
            self.post(post_state)
            + self.history(history)
            + self.payload(mutation_cmd)
        )
        gamma, shift = self.read_film(read_cmd).chunk(2, dim=-1)
        view = self.view_norm(state * (1.0 + 0.25 * torch.tanh(gamma)) + shift)
        return self.out(view)


@torch.no_grad()
def _mine_endpoints(cmd, valid, threshold):
    '''Nearest later same-path-like command for every possible mutation position.'''
    B, N, _ = cmd.shape
    unit = F.normalize(
        torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4), dim=-1
    )
    similarity = torch.bmm(unit, unit.transpose(1, 2))
    pos = torch.arange(N, device=cmd.device)
    later = pos.view(1, 1, N) > pos.view(1, N, 1)
    candidates = (
        (similarity > float(threshold))
        & valid.unsqueeze(1)
        & valid.unsqueeze(2)
        & later
    )
    sentinel = pos.new_full((1, 1, N), N)
    later_idx = torch.where(candidates, pos.view(1, 1, N), sentinel).amin(dim=2)
    keep = (later_idx < N) & valid
    nz = torch.nonzero(keep, as_tuple=False)
    row = nz[:, 0]
    mutation = nz[:, 1]
    read = later_idx[row, mutation]
    score = similarity[row, mutation, read]
    return row, mutation, read, score


def _native_endpoint(cfg, net, tok, key_pad, pred, hidden, transition_reads):
    '''Override only cmd_r in the declared odd-length missing-observation layout.'''
    if tok.shape[1] % 2 == 0 or transition_reads is None:
        return pred
    if key_pad is None:
        valid = torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device)
    else:
        valid = ~key_pad.bool()
    valid_cmd = valid[:, 0::2]
    valid_obs = valid[:, 1::2]
    n_pair = valid_obs.shape[1]
    if n_pair == 0:
        return pred

    # Missing obs after command m, immediately followed by a valid read command.
    pattern = (
        valid_cmd[:, :n_pair]
        & ~valid_obs
        & valid_cmd[:, 1:n_pair + 1]
    )
    pos = torch.arange(n_pair, device=tok.device)
    mutation = torch.where(
        pattern, pos.view(1, n_pair), pos.new_full((1, n_pair), -1)
    ).amax(dim=1)
    rows = torch.nonzero(mutation >= 0, as_tuple=False).squeeze(1)
    if rows.numel() == 0:
        return pred

    mutation = mutation[rows]
    s_pre = transition_reads[rows, mutation]
    h_m = hidden[rows, 2 * mutation]
    c_m = tok[rows, 2 * mutation]
    c_r = tok[rows, 2 * (mutation + 1)]
    post = net.transition_from_emb(s_pre, c_m)
    endpoint = cfg['_renderer'](post, h_m, c_m, c_r)
    endpoint = torch.nan_to_num(endpoint, nan=0.0, posinf=1e4, neginf=-1e4)

    out = pred.clone()
    out[rows, 2 * (mutation + 1)] = endpoint
    return out


def wrap(net, D, **params):
    '''Preserve the R18 aux, register the renderer, and install the native route.'''
    cfg = CHAMP.wrap(net, D, **params)
    for key, value in _IMAG_DEFAULTS.items():
        cfg.setdefault(key, value)
    cfg['_imag_step'] = 0

    required = (
        callable(getattr(net, 'transition_from_emb', None))
        and callable(getattr(net, '_transition_reads', None))
        and hasattr(net, 'tr_mut_gate')
        and hasattr(net, 'cmd_proj')
        and hasattr(net, 'type_emb')
        and hasattr(net, 'in_norm')
    )
    cfg['_disabled'] = bool(cfg.get('_disabled', True) or not required)
    if cfg['_disabled']:
        return cfg

    # Draw seed-dependent private parameters but restore the global RNG exactly, so
    # the harness dropout/batcher/auxiliary random stream is unchanged.
    rng = torch.get_rng_state()
    try:
        renderer = _PredictiveStateRenderer(
            int(D), int(getattr(net, 'd', D)), int(cfg['imag_width'])
        )
    finally:
        torch.set_rng_state(rng)
    net.add_module('_r20_predictive_state_renderer', renderer)
    cfg['_renderer'] = renderer
    cfg['_cache'] = None
    cfg['_tr_reads'] = None

    original_transition_reads = net._transition_reads

    def cached_transition_reads(*args, **kwargs):
        reads = original_transition_reads(*args, **kwargs)
        cfg['_tr_reads'] = reads
        return reads

    net._transition_reads = cached_transition_reads
    original_forward = net.forward

    def wrapped_forward(tok_emb, types, key_pad):
        pred, hidden = original_forward(tok_emb, types, key_pad)
        transition_reads = cfg.get('_tr_reads')
        if tok_emb.shape[1] % 2 == 0:
            # Standard fully-observed training/eval layout: return bit-identically.
            cfg['_cache'] = (tok_emb, hidden, transition_reads)
            return pred, hidden
        pred = _native_endpoint(
            cfg, net, tok_emb, key_pad, pred, hidden, transition_reads
        )
        return pred, hidden

    net.forward = wrapped_forward
    return cfg


def _imagination_loss(cfg, batch, net):
    cache = cfg.get('_cache')
    cfg['_cache'] = None
    cfg['_tr_reads'] = None
    if cache is None or not CHAMP._interleave_layout_ok(batch):
        return 0.0

    tok, hidden, transition_reads = cache
    if transition_reads is None:
        return 0.0
    valid = batch['cmd_mask'].bool()
    maxn = valid.shape[1]
    if maxn < 2:
        return 0.0
    cmd = tok[:, 0::2, :][:, :maxn]

    row, mutation, read, score = _mine_endpoints(
        cmd, valid, float(cfg['imag_path_thresh'])
    )
    if row.numel() < 2:
        return 0.0

    # The learned R18 mutation gate is a detached mining prior, never a label.
    with torch.no_grad():
        c_m_all = cmd[row, mutation]
        type_zero = torch.zeros(
            c_m_all.shape[0], dtype=torch.long, device=cmd.device
        )
        cmd_feature = net.in_norm(net.cmd_proj(c_m_all) + net.type_emb(type_zero))
        mut_prob = torch.sigmoid(net.tr_mut_gate(cmd_feature)).squeeze(-1)
        score = score * (float(cfg['mut_floor']) + mut_prob)
        score = torch.nan_to_num(score, nan=0.0).clamp_min(1e-4)

    cap = int(cfg['imag_max_examples'])
    if row.numel() > cap:
        score, order = torch.topk(score, cap)
        row = row[order]
        mutation = mutation[order]
        read = read[order]

    c_m = cmd[row, mutation].detach()
    c_r = cmd[row, read].detach()
    target = batch['tgt'][row, read].detach()
    target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)
    s_pre = transition_reads[row, mutation].detach()
    h_m = hidden[row, 2 * mutation]

    # Exact measurement-time memory distribution enters the shared transition.
    post_raw = net.transition_from_emb(s_pre, c_m)
    post_raw = torch.nan_to_num(post_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    rho = float(cfg['shared_grad'])
    post = _route_grad(post_raw, rho)
    history = _route_grad(h_m, rho)
    renderer = cfg['_renderer']
    prediction = renderer(post, history, c_m, c_r)
    prediction = torch.nan_to_num(
        prediction, nan=0.0, posinf=1e4, neginf=-1e4
    )

    n, d = prediction.shape
    pred_sq = (prediction * prediction).sum(dim=1, keepdim=True)
    target_sq = (target * target).sum(dim=1, keepdim=True)
    distance = (
        pred_sq + target_sq.t() - 2.0 * (prediction @ target.t())
    ).clamp_min(0.0) / float(d)

    with torch.no_grad():
        target_distance = (
            target_sq + target_sq.t() - 2.0 * (target @ target.t())
        ).clamp_min(0.0) / float(d)
        eye = torch.eye(n, dtype=torch.bool, device=prediction.device)
        duplicate = (
            target_distance < float(cfg['imag_dup_delta'])
        ) & ~eye

    logits = -distance / float(cfg['imag_tau'])
    logits = logits.masked_fill(duplicate, -1e4)
    labels = torch.arange(n, device=prediction.device)
    nll = F.cross_entropy(logits, labels, reduction='none')
    example_weight = (score / score.mean().clamp_min(1e-6)).detach()
    endpoint = (example_weight * nll).mean()
    endpoint = endpoint + float(cfg['imag_mse']) * F.mse_loss(
        prediction, target
    )

    # Same command suffix, wrong predictive state: a direct conditional-history test.
    with torch.no_grad():
        suffix = F.normalize(torch.cat([c_m, c_r], dim=-1), dim=-1)
        suffix_similarity = suffix @ suffix.t()
        suffix_similarity = suffix_similarity.masked_fill(eye, -2.0)
        suffix_similarity = suffix_similarity.masked_fill(
            target_distance < float(cfg['imag_dup_delta']), -2.0
        )
        best_similarity, counterfactual_idx = suffix_similarity.max(dim=1)
        use = (
            best_similarity > float(cfg['cf_thresh'])
        ).to(prediction.dtype)

    counterfactual = renderer(
        post[counterfactual_idx].detach(),
        history[counterfactual_idx].detach(),
        c_m,
        c_r,
    )
    counterfactual = torch.nan_to_num(
        counterfactual, nan=0.0, posinf=1e4, neginf=-1e4
    )
    d_true = (prediction - target).pow(2).mean(dim=-1)
    d_counterfactual = (counterfactual - target).pow(2).mean(dim=-1)
    swap = F.softplus(
        (
            d_true
            + float(cfg['cf_margin'])
            - d_counterfactual
        ) / float(cfg['cf_tau'])
    )
    swap = (use * swap).sum() / use.sum().clamp_min(1.0)
    total = endpoint + float(cfg['cf_weight']) * swap
    if not bool(torch.isfinite(total).item()):
        return 0.0
    return total


def aux_loss(head_state, batch, net, device):
    '''R18 consistency plus the single-pass endpoint predictive-state loss.'''
    cfg = head_state
    champion_loss = CHAMP.aux_loss(cfg, batch, net, device)
    if cfg is None or cfg.get('_disabled', True):
        return champion_loss
    if float(cfg.get('imag_weight', 0.0)) <= 0.0:
        return champion_loss

    cfg['_imag_step'] = int(cfg.get('_imag_step', 0)) + 1
    progress = cfg['_imag_step'] / max(1.0, float(cfg['imag_ramp_steps']))
    ramp = CHAMP._smoothstep(progress)
    if ramp <= 0.0:
        return champion_loss
    imagination = _imagination_loss(cfg, batch, net)
    if isinstance(imagination, float):
        return champion_loss
    return champion_loss + float(cfg['imag_weight']) * ramp * imagination


def leak_safe(mod, params):
    '''Validate the wrapper and explain the causal boundary.

    Fully observed forward is the original R18 forward. In the native odd layout,
    the endpoint consumes only s_pre_m and h_m, both computed at command m from
    positions <=2m, plus c_m and c_r. The missing observation value is masked and
    is never used. Future observations and mined z_r occur only as aux labels.
    '''
    if not CHAMP.leak_safe(CHAMP, params or {}):
        return False
    p = dict(_IMAG_DEFAULTS)
    p.update(params or {})
    try:
        values = {key: float(p[key]) for key in _IMAG_DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(value) for value in values.values()):
        return False
    return all([
        values['imag_weight'] >= 0.0,
        values['imag_ramp_steps'] >= 1.0,
        -1.0 <= values['imag_path_thresh'] < 1.0,
        values['imag_max_examples'] >= 2.0,
        values['imag_tau'] > 0.0,
        values['imag_dup_delta'] >= 0.0,
        values['imag_mse'] >= 0.0,
        values['cf_weight'] >= 0.0,
        -1.0 <= values['cf_thresh'] <= 1.0,
        values['cf_margin'] >= 0.0,
        values['cf_tau'] > 0.0,
        values['mut_floor'] >= 0.0,
        0.0 <= values['shared_grad'] <= 1.0,
        values['imag_width'] >= 32.0,
    ])
