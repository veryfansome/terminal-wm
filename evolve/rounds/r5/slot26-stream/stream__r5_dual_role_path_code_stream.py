import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r5_dual_role_path_code_stream"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and the observation tokens untouched, and REPLACES each "
    "COMMAND token with a role-slotted symbol code concatenated to a compressed copy of the "
    "standardized command embedding. A command is parsed into (verb, source path, destination "
    "path, redirection form); the source path's dense Rademacher identity code occupies one fixed "
    "coordinate block and the destination path's occupies a second disjoint block of equal width, "
    "so the two roles are separated by a fixed coordinate selection and a linear map can read a "
    "source key and a destination key as two different functionals of the same token. A third "
    "block holds the role-symmetric sum of the two path codes, so two commands that mention the "
    "same path in ANY roles overlap there; two narrow blocks hold the directory codes of the two "
    "paths, one holds the verb code, and eight named dimensions hold form flags (source present, "
    "explicit destination, mv, cat, ls, '>', '>>', source equals destination). Commands with no "
    "explicit destination take their source as their destination. The remaining 256 coordinates "
    "carry a fixed count-sketch of the command embedding: a frozen partition of the 768 input "
    "coordinates into 256 buckets of three, summed with frozen signs and rescaled to unit "
    "variance, which preserves embedding inner products up to the sketch error. Every coordinate "
    "of a command token is a function of that command's own string and its own embedding, so the "
    "coding is position-wise and prefix-causal."
)

D = M.D
CUPS_LAYOUT = "interleave2"

ID_SRC_LO, ID_SRC_HI = 0, 160
ID_DST_LO, ID_DST_HI = 160, 320
PSET_LO, PSET_HI = 320, 400
DIR_SRC_LO, DIR_SRC_HI = 400, 440
DIR_DST_LO, DIR_DST_HI = 440, 480
VERB_LO, VERB_HI = 480, 504
FLAG_LO, FLAG_HI = 504, 512
ZC_LO, ZC_HI = 512, 768

ID_DIM = ID_SRC_HI - ID_SRC_LO
PSET_DIM = PSET_HI - PSET_LO
DIR_DIM = DIR_SRC_HI - DIR_SRC_LO
VERB_DIM = VERB_HI - VERB_LO
FLAG_DIM = FLAG_HI - FLAG_LO
ZC_DIM = ZC_HI - ZC_LO
ZC_FANIN = D // ZC_DIM

ID_GAIN = 1.0
PSET_GAIN = 0.6
DIR_GAIN = 0.6
VERB_GAIN = 0.7
FLAG_GAIN = 1.0
ZC_GAIN = 1.0 / float(np.sqrt(ZC_FANIN))

SKETCH_SEED = 20260813

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}

_TABLE_LOCK = threading.Lock()
_TABLE_CACHE = {}

_rs = np.random.RandomState(SKETCH_SEED)
_SKETCH_PERM = _rs.permutation(D).astype(np.int64)
_SKETCH_SIGN = (_rs.randint(0, 2, size=(ZC_DIM, ZC_FANIN)).astype(np.float32) * 2.0 - 1.0)

_EMPTY_CODE = np.zeros(ZC_LO, dtype=np.float32)


def _signs(tag, text, n):
    nbytes = (n + 7) // 8
    raw = hashlib.shake_256((tag + "\x00" + text).encode("utf-8")).digest(nbytes)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n]
    return bits.astype(np.float32) * 2.0 - 1.0


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _dir_of(p):
    if not p:
        return ""
    if p == "/":
        return "/"
    idx = p.rfind("/")
    if idx < 0:
        return "."
    if idx == 0:
        return "/"
    return p[:idx]


def _parse(cmd):
    toks = cmd.split()
    if not toks:
        return "", "", "", None, False
    verb = toks[0]
    positional = []
    redirect = None
    redir_target = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t == ">" or t == ">>":
            redirect = t
            if i + 1 < len(toks):
                redir_target = toks[i + 1]
                i += 1
        elif t.startswith(">>"):
            redirect = ">>"
            redir_target = t[2:] or redir_target
        elif t.startswith(">"):
            redirect = ">"
            redir_target = t[1:] or redir_target
        elif t.startswith("-") and len(t) > 1:
            pass
        else:
            positional.append(t)
        i += 1
    src = _norm_path(positional[0]) if positional else ""
    if redir_target:
        return verb, src, _norm_path(redir_target), redirect, True
    if verb in _DST_VERBS and len(positional) >= 2:
        return verb, src, _norm_path(positional[-1]), redirect, True
    return verb, src, src, redirect, False


def _compact(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, src, dst, redirect, explicit = _parse(cmd)
    v = np.zeros(ZC_LO, dtype=np.float32)
    pset = np.zeros(PSET_DIM, dtype=np.float32)
    if src:
        v[ID_SRC_LO:ID_SRC_HI] = ID_GAIN * _signs("id", src, ID_DIM)
        v[DIR_SRC_LO:DIR_SRC_HI] = DIR_GAIN * _signs("dir", _dir_of(src), DIR_DIM)
        pset += _signs("pset", src, PSET_DIM)
    if dst:
        v[ID_DST_LO:ID_DST_HI] = ID_GAIN * _signs("id", dst, ID_DIM)
        v[DIR_DST_LO:DIR_DST_HI] = DIR_GAIN * _signs("dir", _dir_of(dst), DIR_DIM)
        pset += _signs("pset", dst, PSET_DIM)
    v[PSET_LO:PSET_HI] = PSET_GAIN * pset
    if verb:
        v[VERB_LO:VERB_HI] = VERB_GAIN * _signs("verb", verb, VERB_DIM)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        bool(src) and src == dst,
    )
    v[FLAG_LO:FLAG_HI] = FLAG_GAIN * np.array(
        [1.0 if f else -1.0 for f in flags[:FLAG_DIM]], dtype=np.float32)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = v
    return v


def _tables(device, dtype):
    key = (str(device), dtype)
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    idx = torch.from_numpy(_SKETCH_PERM).to(device=device)
    sg = torch.from_numpy(_SKETCH_SIGN).to(device=device, dtype=dtype)
    with _TABLE_LOCK:
        _TABLE_CACHE[key] = (idx, sg)
    return idx, sg


def code_cmds(cmds, z_cmd):
    if z_cmd.dim() != 2 or int(z_cmd.shape[1]) != D:
        return z_cmd.clone()
    n = int(z_cmd.shape[0])
    out = torch.zeros_like(z_cmd)
    if n == 0:
        return out
    idx, sg = _tables(z_cmd.device, z_cmd.dtype)
    gathered = z_cmd.index_select(1, idx).reshape(n, ZC_DIM, ZC_FANIN)
    out[:, ZC_LO:ZC_HI] = (gathered * sg).sum(-1) * ZC_GAIN
    m = min(n, len(cmds)) if cmds else 0
    if m > 0:
        rows = np.stack([
            _compact(c) if isinstance(c, str) and c else _EMPTY_CODE
            for c in list(cmds)[:m]
        ])
        blk = torch.from_numpy(rows).to(device=z_cmd.device, dtype=z_cmd.dtype)
        out[:m, :ZC_LO] = blk
    return out


def collate(batch, device):
    maxn = max(int(s["z_obs"].shape[0]) for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    for bi, s in enumerate(batch):
        n = int(s["z_obs"].shape[0])
        if n == 0:
            continue
        zo = s["z_obs"][:n]
        zc = code_cmds(list(s.get("cmds") or []), s["z_cmd"][:n])
        tok[bi, 0:2 * n:2] = zc
        tok[bi, 1:2 * n:2] = zo
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
        cmd_mask[bi, :n] = True
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = extract_cmd_pred(pred, b).cpu()
        cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = int(s["z_obs"].shape[0])
            for t in range(n):
                preds.append(cmd_pred[bi, t])
                hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t])
                imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [M.verb_of(c) for c in cmds]}


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
