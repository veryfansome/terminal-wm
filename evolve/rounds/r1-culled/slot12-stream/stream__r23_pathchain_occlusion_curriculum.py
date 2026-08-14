import torch

from realenv import seq_worldmodel as M

NAME = "r23_pathchain_occlusion_curriculum"
DESCRIPTION = (
    "Interleave2 layout, unchanged tokens/types/targets, with a train-only OBSERVATION "
    "BLINDING curriculum on key_pad: within each trajectory the steps are grouped into "
    "reference classes by exact command string and by the filesystem paths their arguments "
    "name (a token and its parent directory), and the observations of class MIDDLES are "
    "dropped at a ramped rate while the first observation of every class is protected, so the "
    "evidence a later re-read of that path needs still exists in the prefix but only several "
    "hops further back. A small ramped uniform drop over the remaining observations is kept as "
    "background. Eval/probe collation is clean; blinding is by key_pad only, so raw token "
    "content stays available to train-only auxiliaries."
)

CUPS_LAYOUT = "interleave2"

_TARGET_P = 0.55
_BG_P = 0.08
_RAMP_START = 300
_RAMP_END = 1200
_MAX_OCC_FRAC = 0.4
_CACHE_CAP = 20000

_CLASS_CACHE = {}
_STEPS = {}


def _arg_keys(cmd):
    parts = " ".join(cmd.split()).split()
    keys = set()
    if not parts:
        return keys
    keys.add("c|" + " ".join(parts))
    for tok in parts[1:]:
        tok = tok.strip("'\"`,;()")
        if not tok or tok.startswith("-"):
            continue
        if "/" in tok:
            tok = tok.rstrip("/") or "/"
            keys.add("p|" + tok)
            head = tok.rsplit("/", 1)[0]
            if head:
                keys.add("p|" + head)
        elif "." in tok and len(tok) > 1:
            keys.add("p|" + tok)
    return keys


def _middle_candidates(cmds):
    members = {}
    for i, c in enumerate(cmds):
        for k in _arg_keys(c):
            members.setdefault(k, []).append(i)
    mid = set()
    anchor = set()
    for idxs in members.values():
        if len(idxs) < 2:
            continue
        anchor.add(idxs[0])
        if len(idxs) >= 3:
            mid.update(idxs[1:-1])
    return tuple(sorted(mid - anchor))


def _candidates_for(seq):
    cmds = seq.get("cmds")
    if not cmds:
        return ()
    key = (id(seq), seq["z_cmd"].data_ptr(), len(cmds))
    hit = _CLASS_CACHE.get(key)
    if hit is None:
        hit = _middle_candidates(cmds)
        if len(_CLASS_CACHE) > _CACHE_CAP:
            _CLASS_CACHE.clear()
        _CLASS_CACHE[key] = hit
    return hit


def _ramp():
    run = int(torch.initial_seed())
    s = _STEPS.get(run, 0) + 1
    _STEPS[run] = s
    if s <= _RAMP_START:
        return 0.0
    if s >= _RAMP_END:
        return 1.0
    x = (s - _RAMP_START) / float(_RAMP_END - _RAMP_START)
    return x * x * (3.0 - 2.0 * x)


def _blind(batch, collated, p_target, p_bg):
    kp = collated["key_pad"]
    B, L = kp.shape
    n_pair = L // 2
    if n_pair < 3 or (p_target <= 0.0 and p_bg <= 0.0):
        return collated

    live = torch.zeros(B, n_pair, dtype=torch.bool)
    prob = torch.zeros(B, n_pair)
    for bi, s in enumerate(batch):
        n = min(int(s["z_obs"].shape[0]), n_pair)
        live[bi, :n] = True
        prob[bi, :n] = p_bg
        for c in _candidates_for(s):
            if c < n:
                prob[bi, c] = p_target

    u = torch.rand(B, n_pair)
    drop = (u < prob) & live
    cap = max(1, int(_MAX_OCC_FRAC * n_pair))
    if int(drop.sum(dim=1).max()) > cap:
        rank = torch.where(drop, u, torch.full_like(u, 2.0))
        keep = torch.zeros_like(drop)
        keep.scatter_(1, rank.argsort(dim=1)[:, :cap], True)
        drop = drop & keep
    if not bool(drop.any()):
        return collated

    # Blind through key_pad only: tok keeps its raw values so train-only auxiliaries that read
    # batch["tok"] directly still mine uncorrupted observations.
    full = torch.zeros(B, L, dtype=torch.bool)
    full[:, 1:2 * n_pair:2] = drop
    collated["key_pad"] = kp | full.to(kp.device)
    return collated


def collate(batch, device):
    out = M.collate(batch, device)
    if not torch.is_grad_enabled():
        return out
    r = _ramp()
    if r <= 0.0:
        return out
    return _blind(batch, out, _TARGET_P * r, _BG_P * r)


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def flatten_predictions(net, seqs, device):
    return M.flatten_predictions(net, seqs, device)


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    if not bool((chg[:4] < 1e-4).all()):
        return False

    state = torch.get_rng_state()
    c0 = _blind(seq, M.collate(seq, device), _TARGET_P, _BG_P)
    torch.set_rng_state(state)
    c1 = _blind(seq, M.collate(seq, device), _TARGET_P, _BG_P)
    if not bool(torch.equal(c0["key_pad"], c1["key_pad"])):
        return False
    c1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0
    q0 = net(c0["tok"], c0["types"], c0["key_pad"])[0][:, 0::2].clone().cpu()
    q1 = net(c1["tok"], c1["types"], c1["key_pad"])[0][:, 0::2].cpu()
    chg2 = (q1 - q0).abs().amax(-1)[0]
    return bool((chg2[:4] < 1e-4).all())
