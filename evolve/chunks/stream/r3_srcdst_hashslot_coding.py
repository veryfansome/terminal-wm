import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r3_srcdst_hashslot_coding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and the observation tokens exactly as they are, and "
    "re-codes each COMMAND token position-wise as its standardized command embedding plus a "
    "deterministic sparse symbol code. The code writes the command's source path into one fixed "
    "coordinate block, its destination path into a second disjoint block, its verb into a third, "
    "and seven form flags (source present, explicit destination, mv, cat, ls, '>', '>>') into "
    "named dimensions. A path's entries are k-sparse signed values drawn from a SHAKE-256 digest "
    "of the path string, placed at the SAME relative offsets inside whichever of the two "
    "equally-wide path blocks is used, so one path yields one pattern that a fixed coordinate "
    "selection recovers from either role, and two paths differing only in a trailing suffix yield "
    "near-orthogonal patterns. Nothing outside a command's own string and its own embedding "
    "enters that command's token."
)

D = M.D
CUPS_LAYOUT = "interleave2"

SRC_LO, SRC_HI = 0, 352
DST_LO, DST_HI = 352, 704
VERB_LO, VERB_HI = 704, 752
FLAG_LO = 752
N_FLAGS = 7

PATH_K = 24
VERB_K = 8
PATH_GAIN = 2.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}


def _sparse_block(text, tag, lo, hi, k, gain, cols):
    span = hi - lo
    raw = hashlib.shake_256((tag + "\x00" + text).encode("utf-8")).digest(3 * k)
    for j in range(k):
        a = raw[3 * j]
        b = raw[3 * j + 1]
        c = raw[3 * j + 2]
        idx = lo + (((a << 8) | b) % span)
        if idx in cols:
            continue
        cols[idx] = gain if (c & 1) else -gain


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


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
    cols = {}
    if src:
        _sparse_block(src, "path", SRC_LO, SRC_HI, PATH_K, PATH_GAIN, cols)
    if dst:
        _sparse_block(dst, "path", DST_LO, DST_HI, PATH_K, PATH_GAIN, cols)
    if verb:
        _sparse_block(verb, "verb", VERB_LO, VERB_HI, VERB_K, VERB_GAIN, cols)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
    )
    for j in range(N_FLAGS):
        cols[FLAG_LO + j] = FLAG_GAIN if flags[j] else -FLAG_GAIN
    idx = np.fromiter(cols.keys(), dtype=np.int64, count=len(cols))
    val = np.fromiter(cols.values(), dtype=np.float32, count=len(cols))
    out = (idx, val)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = out
    return out


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    if n == 0 or not cmds:
        return out
    m = min(n, len(cmds))
    rows = []
    cols = []
    vals = []
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        idx, val = _compact(c)
        if idx.size == 0:
            continue
        rows.append(np.full(idx.size, i, dtype=np.int64))
        cols.append(idx)
        vals.append(val)
    if not rows:
        return out
    r = torch.from_numpy(np.concatenate(rows)).to(out.device)
    c = torch.from_numpy(np.concatenate(cols)).to(out.device)
    v = torch.from_numpy(np.concatenate(vals)).to(device=out.device, dtype=out.dtype)
    out.index_put_((r, c), v, accumulate=True)
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
