import hashlib
import posixpath
import shlex

import torch

NAME = "r23_pathslot_interleave"
DESCRIPTION = (
    "Interleave2 layout whose command token carries a reserved 200-dim argument-slot block: "
    "the command's source path, its parent, the destination path, the destination's parent and "
    "the directory-join destination are each encoded as a fixed pseudorandom unit code in its own "
    "disjoint coordinate block, replacing (not adding to) the native embedding coordinates there, "
    "so a downstream arch decodes exact per-slot addresses. Paths are resolved against a cwd "
    "folded from the cd commands at strictly earlier steps. Also exports slot_code/slot_valid on "
    "the batch for a slot-aware head."
)

CUPS_LAYOUT = "interleave2"

D = 768
N_SLOT = 5
SLOT_K = 40
BASIS_SEED = 20260809
CODE_SCALE = float(SLOT_K) ** 0.5

SLOT_ORDER = ("src_full", "src_dir", "dst_full", "dst_dir", "dst_join")

_SEQ_CACHE_MAX = 1024
_CODE_CACHE_MAX = 131072


def slot_index():
    g = torch.Generator().manual_seed(BASIS_SEED)
    return torch.randperm(D, generator=g)[: N_SLOT * SLOT_K].view(N_SLOT, SLOT_K).contiguous()


SLOT_INDEX = slot_index()
SLOT_FLAT = SLOT_INDEX.reshape(-1).contiguous()

_ZERO_CODE = torch.zeros(SLOT_K)
_CODE_CACHE = {}
_SEQ_CACHE = {}

_SEPARATORS = ("&&", "||", ";", "|")
_CD_VERBS = ("cd", "pushd")


def _code(path):
    if not path:
        return _ZERO_CODE
    v = _CODE_CACHE.get(path)
    if v is None:
        h = hashlib.blake2b(path.encode("utf-8", "replace"), digest_size=8).digest()
        g = torch.Generator().manual_seed(int.from_bytes(h, "big") % (2 ** 62))
        v = torch.randn(SLOT_K, generator=g)
        v = v / v.norm().clamp_min(1e-6)
        while len(_CODE_CACHE) >= _CODE_CACHE_MAX:
            _CODE_CACHE.pop(next(iter(_CODE_CACHE)))
        _CODE_CACHE[path] = v
    return v


def _segments(cmd):
    parts = [cmd]
    for sep in _SEPARATORS:
        nxt = []
        for p in parts:
            nxt.extend(p.split(sep))
        parts = nxt
    return [p.strip() for p in parts if p.strip()]


def _tokens(seg):
    try:
        return shlex.split(seg)
    except Exception:
        return seg.split()


def _resolve(cwd, raw):
    p = raw.strip().strip("\"'")
    if not p or p in (".",):
        p = cwd
    if p.startswith("~"):
        p = "/~" + p[1:]
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    p = posixpath.normpath(p)
    return p if p else "/"


def _parent(p):
    d = posixpath.dirname(p)
    return d if d else "/"


def _step_paths(cmd, cwd):
    cur = cwd
    args_abs = []
    for seg in _segments(cmd):
        toks = _tokens(seg)
        if not toks:
            continue
        verb = toks[0]
        rest = [t for t in toks[1:] if not t.startswith("-")]
        pathish = [t for t in rest if "/" in t] or rest
        cand = [a for a in (_resolve(cur, t) for t in pathish) if a]
        if verb in _CD_VERBS:
            if cand:
                cur = cand[-1]
            continue
        if cand:
            args_abs = cand
    if not args_abs:
        args_abs = [cur]
    a0 = args_abs[0]
    if len(args_abs) >= 2:
        aL = args_abs[-1]
        joined = posixpath.normpath(posixpath.join(aL, posixpath.basename(a0)))
    else:
        aL = a0
        joined = a0
    return (a0, _parent(a0), aL, _parent(aL), joined), cur


def _seq_slots(cmds, n):
    key = tuple(cmds[:n]) if len(cmds) >= n else (tuple(cmds), n)
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        return hit
    codes = torch.zeros(n, N_SLOT, SLOT_K)
    valid = torch.zeros(n, N_SLOT, dtype=torch.bool)
    cwd = "/"
    for i in range(n):
        raw = cmds[i] if i < len(cmds) else ""
        paths, cwd = _step_paths(raw if isinstance(raw, str) else "", cwd)
        for s, p in enumerate(paths):
            if p:
                codes[i, s] = _code(p)
                valid[i, s] = True
    while len(_SEQ_CACHE) >= _SEQ_CACHE_MAX:
        _SEQ_CACHE.pop(next(iter(_SEQ_CACHE)))
    _SEQ_CACHE[key] = (codes, valid)
    return codes, valid


def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    slot_code = torch.zeros(B, maxn, N_SLOT, SLOT_K)
    slot_valid = torch.zeros(B, maxn, N_SLOT, dtype=torch.bool)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        zc = s["z_cmd"][:n].clone()
        cmds = s.get("cmds") or []
        codes, valid = _seq_slots(cmds, n)
        zc[:, SLOT_FLAT] = CODE_SCALE * codes.reshape(n, -1)
        slot_code[bi, :n] = codes
        slot_valid[bi, :n] = valid
        tok[bi, 0:2 * n:2] = zc
        tok[bi, 1:2 * n:2] = s["z_obs"][:n]
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n]
        cmd_mask[bi, :n] = True
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device),
           "slot_code": slot_code.to(device), "slot_valid": slot_valid.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def verb_of(cmd):
    p = cmd.split()
    return p[0] if p else ""


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred_full, h_full = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = extract_cmd_pred(pred_full, b).cpu()
        cmd_h = h_full[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
            for t in range(n):
                preds.append(cmd_pred[bi, t])
                hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t])
                imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [verb_of(c) for c in cmds]}


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D),
            "cmds": ["ls /a", "cat /a/f1", "mv /a/f1 /b/f2", "ls /b", "cat /b/f2", "ls /a"],
            "image": "x"}]
    b0 = collate(seq, device)
    p0 = extract_cmd_pred(net(b0["tok"], b0["types"], b0["key_pad"])[0], b0).clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = extract_cmd_pred(net(b1["tok"], b1["types"], b1["key_pad"])[0], b1).cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
