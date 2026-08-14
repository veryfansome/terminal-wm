import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r5_dualslot_overwrite_path_coding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave and the observation tokens untouched, and rewrites each "
    "COMMAND token by OVERWRITING a fixed, command-independent set of 296 coordinates with a "
    "symbolic address code while leaving the other 472 coordinates of the standardized command "
    "embedding intact. The 296 coordinates are a deterministic pseudo-random selection out of the "
    "768, split into four disjoint fixed blocks: a 128-wide SOURCE block, a 128-wide DESTINATION "
    "block, a 32-wide VERB block and 8 named form flags (source present, explicit destination, "
    "mv, cat, ls, '>', '>>', source equals destination). A path is written as a DENSE sign vector "
    "over the whole block, with exactly half the entries positive and half negative, drawn by "
    "sorting a SHAKE-256 digest of the path string, so every path has the same code norm and zero "
    "code mean, two distinct paths give near-orthogonal codes, and one path yields the SAME "
    "pattern in the source block and in the destination block because the two blocks are read off "
    "the same generator in the same order. Because the blocks are at fixed coordinates and the "
    "code REPLACES rather than adds, a linear map that selects one block sees the address alone "
    "with no residue of the embedding at those coordinates. The parser splits a command into verb, "
    "first positional argument, and either the redirection target or the last positional argument "
    "of a two-argument mv/cp/ln/install; a command with no explicit destination writes its own "
    "argument into both blocks. Nothing outside a command's own string and its own embedding "
    "enters that command's token, so the coding is position-wise and therefore prefix-causal."
)

D = M.D
CUPS_LAYOUT = "interleave2"

PATH_M = 128
VERB_M = 32
FLAG_M = 8

PATH_GAIN = 1.5
VERB_GAIN = 1.0
FLAG_GAIN = 1.5

_TOTAL = 2 * PATH_M + VERB_M + FLAG_M

_DST_VERBS = ("mv", "cp", "ln", "install")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}


def _digest_order(seed_bytes, m):
    raw = hashlib.shake_256(seed_bytes).digest(4 * m)
    keys = [int.from_bytes(raw[4 * i:4 * i + 4], "big") for i in range(m)]
    return sorted(range(m), key=lambda i: (keys[i], i))


def _coordinate_blocks():
    order = _digest_order(b"dualslot-coordmap-v1", D)
    a = order[0:PATH_M]
    b = order[PATH_M:2 * PATH_M]
    c = order[2 * PATH_M:2 * PATH_M + VERB_M]
    f = order[2 * PATH_M + VERB_M:_TOTAL]
    return a, b, c, f


SRC_COLS, DST_COLS, VERB_COLS, FLAG_COLS = _coordinate_blocks()
_COLS_NP = np.asarray(SRC_COLS + DST_COLS + VERB_COLS + FLAG_COLS, dtype=np.int64)


def _balanced_code(text, tag, m, gain):
    order = _digest_order((tag + "\x00" + text).encode("utf-8"), m)
    v = np.full(m, -float(gain), dtype=np.float32)
    half = m // 2
    if half:
        v[np.asarray(order[:half], dtype=np.int64)] = float(gain)
    return v


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
        return "", "", "", "", False
    verb = toks[0]
    positional = []
    redirect = ""
    redir_target = ""
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


def _code_vector(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, src, dst, redirect, explicit = _parse(cmd)
    s = _balanced_code(src, "path", PATH_M, PATH_GAIN)
    d = _balanced_code(dst, "path", PATH_M, PATH_GAIN)
    v = _balanced_code(verb, "verb", VERB_M, VERB_GAIN)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        src == dst,
    )
    f = np.where(np.asarray(flags, dtype=bool), float(FLAG_GAIN), -float(FLAG_GAIN))
    out = np.concatenate([s, d, v, f.astype(np.float32)]).astype(np.float32)
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
    vecs = []
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        rows.append(i)
        vecs.append(_code_vector(c))
    if not rows:
        return out
    cols = torch.from_numpy(_COLS_NP).to(out.device)
    vals = torch.from_numpy(np.stack(vecs, axis=0)).to(device=out.device, dtype=out.dtype)
    ridx = torch.as_tensor(rows, dtype=torch.long, device=out.device)
    sel = out.index_select(0, ridx)
    sel.index_copy_(1, cols, vals)
    out.index_copy_(0, ridx, sel)
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
