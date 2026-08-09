'''R21 algebraic history-evidence innovation observer.

Keeps the champion r18 auxiliary and adds a private bounded masked-endpoint
correction. The same observer runs with or without history. Only attention values
from valid prefix pairs enter its bias-free residual, so empty history gives an
exactly zero correction without detecting an evaluation arm.
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from evolve.chunks.head import r18_transition_forwardmodel_consistency as CHAMP

NAME = 'r21_evidencegated_imagwrite_algebraic_observer_head'
DESCRIPTION = ('Champion r18 transition consistency plus a private zero-init bounded '
               'masked-endpoint observer whose only additive evidence is prefix-pair '
               'cross-attention and a raw-observation copy readout. The same module runs '
               'in both history arms and is algebraically zero for empty history. Source-'
               'free endpoint pairs train duplicate-masked L2-InfoNCE plus MSE; ordinary '
               'even-length forwards are bit-identical to the champion.')

_DEFAULTS = {
    'imag_heads': 4, 'imag_dk': 64, 'imag_dv': 64, 'imag_width': 192,
    'imag_path_thresh': 0.60, 'imag_wfloor': 0.15,
    'imag_max_examples': 64, 'imag_resid_rms': 0.50,
    'imag_tau': 0.25, 'imag_mse_weight': 0.20, 'imag_dupe_cos': 0.98,
    'imag_aux_weight': 0.05, 'imag_ramp_start': 300,
    'imag_ramp_steps': 900, 'imag_every': 1,
}
_EPS = 1e-8


def _unit(x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True).clamp_min(_EPS))


def _smoothstep(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


class _EvidenceOnlyInnovation(nn.Module):
    def __init__(self, D, hidden_d, heads, dk, dv, width, residual_rms):
        super().__init__()
        self.D, self.h, self.dk, self.dv = int(D), int(heads), int(dk), int(dv)
        self.residual_rms = float(residual_rms)
        qin = 3 * self.D + int(hidden_d)
        self.q = nn.Linear(qin, self.h * self.dk, bias=False)
        self.k = nn.Linear(2 * self.D, self.h * self.dk, bias=False)
        self.v = nn.Linear(2 * self.D, self.h * self.dv, bias=False)
        self.copy_q = nn.Linear(qin, self.dk, bias=False)
        self.copy_k = nn.Linear(2 * self.D, self.dk, bias=False)
        evidence_d, w = self.h * self.dv + self.D, int(width)
        self.mod1 = nn.Linear(qin, w)
        self.mod2 = nn.Linear(w, evidence_d, bias=False)
        self.body = nn.Sequential(nn.Linear(evidence_d, w, bias=False), nn.GELU(),
                                  nn.Linear(w, 2 * w, bias=False), nn.GELU())
        self.out = nn.Linear(2 * w, self.D + 1, bias=False)
        nn.init.zeros_(self.out.weight)

    @staticmethod
    def _attn(logits, mask, values):
        weight = torch.softmax(logits.masked_fill(~mask, -1e4), -1)
        weight = weight * mask.to(weight.dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-6)
        return torch.matmul(weight.unsqueeze(-2), values).squeeze(-2)

    def forward(self, native, cmd_m, cmd_r, endpoint_h, pairs, pair_mask):
        N, P, _ = pairs.shape
        qin = torch.cat([native, cmd_m, cmd_r, endpoint_h], -1)
        q = self.q(qin).view(N, self.h, self.dk)
        k = self.k(pairs).view(N, P, self.h, self.dk).transpose(1, 2)
        v = self.v(pairs).view(N, P, self.h, self.dv).transpose(1, 2)
        logits = torch.einsum('nhd,nhpd->nhp', q, k) / math.sqrt(self.dk)
        mask = pair_mask.unsqueeze(1)
        read = self._attn(logits, mask, v).reshape(N, self.h * self.dv)
        copy_logits = torch.einsum('nd,npd->np', self.copy_q(qin),
                                   self.copy_k(pairs)) / math.sqrt(self.dk)
        copy = self._attn(copy_logits.unsqueeze(1), mask,
                          pairs[:, :, self.D:].unsqueeze(1)).squeeze(1)
        evidence = torch.cat([read, copy], -1)
        modulation = 1.0 + torch.tanh(self.mod2(F.gelu(self.mod1(qin))))
        raw = self.out(self.body(evidence * modulation))
        residual = torch.tanh(raw[:, :self.D])
        rms = residual.pow(2).mean(-1, keepdim=True).add(_EPS).sqrt()
        residual = residual * (self.residual_rms / rms).clamp(max=1.0)
        return native + torch.sigmoid(raw[:, self.D:]) * residual


def _detect_masked_endpoint(types, key_pad):
    if key_pad is None or types.dim() != 2 or key_pad.dim() != 2:
        return None
    _, L = types.shape
    if L < 3 or L % 2 == 0:
        return None
    live = ~key_pad.bool()
    candidate = live[:, :-2] & ~live[:, 1:-1] & live[:, 2:]
    pos0 = torch.arange(L - 2, device=types.device)
    candidate &= ((pos0 % 2) == 0).unsqueeze(0)
    nz = torch.nonzero(candidate, as_tuple=False)
    if nz.numel() == 0:
        return None
    rows, mutation = nz[:, 0], nz[:, 1]
    read = mutation + 2
    pos = torch.arange(L, device=types.device).unsqueeze(0)
    tail_ok = ((~live[rows]) | (pos <= read.unsqueeze(1))).all(1)
    type_ok = ((types[rows, mutation] == 0) &
               (types[rows, mutation + 1] == 1) &
               (types[rows, read] == 0))
    keep = tail_ok & type_ok
    if not bool(keep.any().item()):
        return None
    return rows[keep], mutation[keep], read[keep]


def _prefix_pairs(tok, valid, mutation):
    obs = tok[:, 1::2, :]
    n_pair = obs.size(1)
    cmd = tok[:, 0::2, :][:, :n_pair]
    pos = torch.arange(n_pair, device=tok.device)
    mask = (valid[:, 0::2][:, :n_pair] & valid[:, 1::2][:, :n_pair] &
            (pos.unsqueeze(0) < (mutation // 2).unsqueeze(1)))
    pairs = torch.cat([cmd, obs], -1) * mask.unsqueeze(-1).to(tok.dtype)
    return pairs, mask


def _prediction(observer, tok, valid, pred, hidden, rows, mutation, read):
    pairs, mask = _prefix_pairs(tok[rows].detach(), valid[rows], mutation)
    return observer(pred[rows, read].detach(), tok[rows, mutation].detach(),
                    tok[rows, read].detach(), hidden[rows, read].detach(), pairs, mask)


@torch.no_grad()
def _mine_pairs(cmd, valid, net, threshold, wfloor):
    B, maxn, _ = cmd.shape
    clean = torch.nan_to_num(cmd, nan=0.0, posinf=1e4, neginf=-1e4)
    sim = torch.bmm(_unit(clean), _unit(clean).transpose(1, 2))
    pos = torch.arange(maxn, device=cmd.device)
    after = ((sim > threshold) & valid.bool().unsqueeze(1) &
             (pos.unsqueeze(1) < pos.unsqueeze(0)).unsqueeze(0))
    posf = pos.view(1, 1, maxn).expand(B, maxn, maxn)
    later = torch.where(after, posf, torch.full_like(posf, maxn)).amin(2)
    has = (later < maxn) & valid.bool() & (pos.unsqueeze(0) >= 1)
    gate_value = None
    try:
        modules = (net.cmd_proj, net.in_norm, net.tr_mut_gate, net.type_emb)
        if all(isinstance(x, nn.Module) for x in modules):
            zeros = torch.zeros(B, maxn, dtype=torch.long, device=cmd.device)
            xcmd = net.in_norm(net.cmd_proj(clean) + net.type_emb(zeros))
            gate_value = torch.sigmoid(net.tr_mut_gate(xcmd)).squeeze(-1)
    except Exception:
        gate_value = None
    if gate_value is None:
        gate_value = torch.zeros(B, maxn, device=cmd.device, dtype=cmd.dtype)
    lc = later.clamp(0, maxn - 1)
    similarity = torch.gather(sim, 2, lc.unsqueeze(-1)).squeeze(-1)
    weight = similarity * (float(wfloor) + gate_value)
    nz = torch.nonzero(has & torch.isfinite(weight) & (weight > 0), as_tuple=False)
    return nz[:, 0], nz[:, 1], later[nz[:, 0], nz[:, 1]], weight[nz[:, 0], nz[:, 1]]


def _masked_batch(cmd, obs, rows, mutation, read):
    cmd, obs = cmd[rows].detach(), obs[rows].detach()
    mutation, read = mutation.long(), read.long()
    N, maxn, _ = cmd.shape
    L = 2 * int(mutation.max().item()) + 3
    pos = torch.arange(L, device=cmd.device)
    source = (pos // 2).clamp(max=maxn - 1)
    pair_tok = torch.where((pos % 2 == 0).view(1, L, 1),
                           cmd[:, source], obs[:, source])
    prefix = pos.unsqueeze(0) < (2 * mutation).unsqueeze(1)
    tok = pair_tok * prefix.unsqueeze(-1).to(pair_tok.dtype)
    types = (pos % 2).long().unsqueeze(0).expand(N, -1).clone()
    key_pad = ~prefix
    rr = torch.arange(N, device=cmd.device)
    mpos, rpos = 2 * mutation, 2 * mutation + 2
    tok[rr, mpos], tok[rr, rpos] = cmd[rr, mutation], cmd[rr, read]
    types[rr, mpos], types[rr, mpos + 1], types[rr, rpos] = 0, 1, 0
    key_pad[rr, mpos], key_pad[rr, mpos + 1], key_pad[rr, rpos] = False, True, False
    return tok, types, key_pad, mpos, rpos


def _rank_loss(pred, target, weight, tau, duplicate_cos):
    distance = (pred.unsqueeze(1) - target.unsqueeze(0)).pow(2).mean(-1)
    logits = -distance / float(tau)
    with torch.no_grad():
        similarity = _unit(target) @ _unit(target).T
        eye = torch.eye(target.size(0), device=target.device, dtype=torch.bool)
        duplicate = (similarity > float(duplicate_cos)) & ~eye
    logits = logits.masked_fill(duplicate, -1e4)
    loss = F.cross_entropy(logits, torch.arange(target.size(0), device=target.device),
                           reduction='none')
    return (weight * loss).sum()


def _private_loss(cfg, batch, net):
    tok, valid = batch['tok'], batch['cmd_mask'].bool()
    B, maxn = valid.shape
    if B < 2 or maxn < 3:
        return 0.0
    cmd, obs = tok[:, 0::2, :][:, :maxn], tok[:, 1::2, :][:, :maxn]
    rows, mutation, read, weight = _mine_pairs(
        cmd, valid, net, float(cfg['imag_path_thresh']), float(cfg['imag_wfloor']))
    if rows.numel() < 2:
        return 0.0
    cap = int(cfg['imag_max_examples'])
    if rows.numel() > cap:
        weight, order = torch.topk(weight, cap)
        rows, mutation, read = rows[order], mutation[order], read[order]
    weight = (weight / weight.sum().clamp_min(_EPS)).detach().to(cmd.dtype)
    mtok, mtypes, mpad, mpos, rpos = _masked_batch(cmd, obs, rows, mutation, read)
    was_training = net.training
    net.eval()
    with torch.no_grad():
        native, hidden = cfg['_original_forward'](mtok, mtypes, mpad)
    if was_training:
        net.train()
    local = torch.arange(mtok.size(0), device=mtok.device)
    pred = _prediction(cfg['_observer'], mtok, ~mpad, native, hidden,
                       local, mpos, rpos)
    target = torch.nan_to_num(obs[rows, read].detach(), nan=0.0,
                              posinf=1e4, neginf=-1e4)
    rank = _rank_loss(pred, target, weight, float(cfg['imag_tau']),
                      float(cfg['imag_dupe_cos']))
    mse = (weight * (pred - target).pow(2).mean(-1)).sum()
    total = rank + float(cfg['imag_mse_weight']) * mse
    return total if bool(torch.isfinite(total).item()) else 0.0


def wrap(net, D, **params):
    cfg = CHAMP.wrap(net, D, **params)
    private = dict(_DEFAULTS)
    private.update(params)
    cfg.update(private)
    cfg['_observer_step'] = 0
    hidden_d = getattr(net, 'd', None)
    cfg['_imag_disabled'] = not isinstance(hidden_d, int)
    if cfg['_imag_disabled']:
        return cfg
    rng = torch.random.get_rng_state()
    observer = _EvidenceOnlyInnovation(D, hidden_d, int(cfg['imag_heads']),
        int(cfg['imag_dk']), int(cfg['imag_dv']), int(cfg['imag_width']),
        float(cfg['imag_resid_rms']))
    torch.random.set_rng_state(rng)
    net.add_module('r21_algebraic_history_evidence_observer', observer)
    cfg['_observer'] = observer
    original = net.forward
    cfg['_original_forward'] = original

    def forward(tok, types, key_pad):
        if tok.size(1) % 2 == 0:
            return original(tok, types, key_pad)
        detected = _detect_masked_endpoint(types, key_pad)
        if detected is None:
            return original(tok, types, key_pad)
        pred, hidden = original(tok, types, key_pad)
        rows, mutation, read = detected
        corrected = _prediction(observer, tok, ~key_pad.bool(), pred, hidden,
                                rows, mutation, read)
        out = pred.clone()
        out[rows, read] = corrected
        return out, hidden

    net.forward = forward
    return cfg


def aux_loss(state, batch, net, device):
    champion = CHAMP.aux_loss(state, batch, net, device)
    if state is None or state.get('_imag_disabled', True):
        return champion
    if not CHAMP._interleave_layout_ok(batch):
        return champion
    state['_observer_step'] = int(state.get('_observer_step', 0)) + 1
    step = state['_observer_step']
    if step % int(state['imag_every']) != 0:
        return champion
    ramp = _smoothstep((step - int(state['imag_ramp_start'])) /
                       max(1.0, float(state['imag_ramp_steps'])))
    if ramp <= 0 or float(state['imag_aux_weight']) <= 0:
        return champion
    return champion + float(state['imag_aux_weight']) * ramp * _private_loss(state, batch, net)


def leak_safe(mod, params):
    if not CHAMP.leak_safe(mod, params):
        return False
    p = dict(_DEFAULTS)
    p.update(params or {})
    try:
        v = {k: float(p[k]) for k in _DEFAULTS}
    except Exception:
        return False
    if any(not math.isfinite(x) for x in v.values()):
        return False
    return all([v['imag_heads'] >= 1, v['imag_dk'] >= 8, v['imag_dv'] >= 8,
                v['imag_width'] >= 16, -1 <= v['imag_path_thresh'] < 1,
                v['imag_wfloor'] >= 0, v['imag_max_examples'] >= 2,
                0 < v['imag_resid_rms'] <= 2, v['imag_tau'] > 0,
                v['imag_mse_weight'] >= 0, -1 <= v['imag_dupe_cos'] <= 1,
                v['imag_aux_weight'] >= 0, v['imag_ramp_start'] >= 0,
                v['imag_ramp_steps'] >= 1, v['imag_every'] >= 1])
