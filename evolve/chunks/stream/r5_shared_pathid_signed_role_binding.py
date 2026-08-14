import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r5_shared_pathid_signed_role_binding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and leaves OBSERVATION tokens untouched, and re-codes "
    "each COMMAND token as its standardized command embedding plus a deterministic symbol code "
    "built only from that command's own string. Every path the command mentions contributes TWO "
    "components. (1) A role-AGNOSTIC identity atom: a k-sparse balanced +/- pattern drawn from a "
    "SHAKE-256 digest of the whole normalized path string, written into one shared wide "
    "coordinate block at the same coordinates with the same signs no matter whether the path is "
    "the command's source, its destination, its redirect target or the argument of a read, so two "
    "commands naming the same path have a large positive inner product in that block and two "
    "commands naming different paths have a near-zero one. Only whole-path strings are hashed; no "
    "directory or basename component is shared, so sibling paths and suffix variants stay "
    "near-orthogonal. (2) A role-SIGNED atom in a second disjoint block, drawn from a separate "
    "digest of the same path, added with a plus sign when the path is a source or a plain "
    "argument and a minus sign when it is a destination or a redirect target, so a single linear "
    "functional reports whether a command moved content INTO a named location or OUT of it, and "
    "so a destination mention and a later read of the same location cancel additively. A third "
    "block holds a sparse verb code and twelve form flags (source present, explicit destination, "
    "mv, cat, ls, cp, cd, head, '>', '>>', mutating, source==destination). Nothing outside a "
    "command's own string and its own embedding ever enters that command's token, and the code is "
    "a pure function of the string, so the coding is prefix-causal and position-independent."
)

D = M.D
CUPS_LAYOUT = "interleave2"

ID_LO, ID_HI = 0, 512
ROLE_LO, ROLE_HI = 512, 704
VERB_LO, VERB_HI = 704, 752
FLAG_LO, N_FLAGS = 752, 12

ID_K = 48
ID_GAIN = 4.5
ROLE_K = 32
ROLE_GAIN = 1.5
VERB_K = 8
VERB_GAIN = 1.0
FLAG_GAIN = 1.0

_DST_VERBS = ("mv", "cp", "ln", "install")
_REDIR_TOKENS = (">", ">>")

_ATOM_CAP = 400000
_CODE_CAP = 400000
_LOCK = threading.Lock()
_ATOM_CACHE = {}
_CODE_CACHE = {}


def _atom(text, tag, lo, hi, k, gain):
    key = (tag, text)
    hit = _ATOM_CACHE.get(key)
    if hit is not None:
        return hit
    span = hi - lo
    raw = hashlib.shake_256((tag + "\x00" + text).encode("utf-8")).digest(8 * k + 8)
    seen = set()
    cols = []
    i = 0
    while len(cols) < k and i + 1 < len(raw):
        idx = lo + (((raw[i] << 8) | raw[i + 1]) % span)
        i += 2
        if idx in seen:
            continue
        seen.add(idx)
        cols.append(idx)
    half = len(cols) // 2
    vals = [gain if j < half else -gain for j in range(len(cols))]
    out = (np.asarray(cols, dtype=np.int64), np.asarray(vals, dtype=np.float32))
    with _LOCK:
        if len(_ATOM_CACHE) >= _ATOM_CAP:
            _ATOM_CACHE.clear()
        _ATOM_CACHE[key] = out
    return out


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _is_pathish(t):
    return ("/" in t) or t.startswith("~")


def _parse(cmd):
    toks = cmd.split()
    if not toks:
        return "", "", "", None
    verb = toks[0]
    positional = []
    redirect = None
    redir_target = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in _REDIR_TOKENS:
            redirect = t
            if i + 1 < len(toks):
                redir_target = toks[i + 1]
                i += 1
        elif t.startswith(">>") and len(t) > 2:
            redirect = ">>"
            redir_target = t[2:]
        elif t.startswith(">") and len(t) > 1:
            redirect = ">"
            redir_target = t[1:]
        elif t.startswith("-") and len(t) > 1:
            pass
        else:
            positional.append(t)
        i += 1
    paths = [p for p in positional if _is_pathish(p)]
    src = _norm_path(paths[0]) if paths else ""
    dst = ""
    if redir_target:
        dst = _norm_path(redir_target)
    elif verb in _DST_VERBS and len(paths) >= 2:
        dst = _norm_path(paths[-1])
    if dst and dst == src:
        dst = ""
    return verb, src, dst, redirect


def _accumulate(acc, idx, val, scale):
    for j in range(idx.shape[0]):
        c = int(idx[j])
        acc[c] = acc.get(c, 0.0) + scale * float(val[j])


def _cmd_code(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, src, dst, redirect = _parse(cmd)
    acc = {}
    if src:
        i_id, v_id = _atom(src, "pid", ID_LO, ID_HI, ID_K, ID_GAIN)
        _accumulate(acc, i_id, v_id, 1.0)
        i_rl, v_rl = _atom(src, "prole", ROLE_LO, ROLE_HI, ROLE_K, ROLE_GAIN)
        _accumulate(acc, i_rl, v_rl, 1.0)
    if dst:
        i_id, v_id = _atom(dst, "pid", ID_LO, ID_HI, ID_K, ID_GAIN)
        _accumulate(acc, i_id, v_id, 1.0)
        i_rl, v_rl = _atom(dst, "prole", ROLE_LO, ROLE_HI, ROLE_K, ROLE_GAIN)
        _accumulate(acc, i_rl, v_rl, -1.0)
    if verb:
        i_vb, v_vb = _atom(verb, "verb", VERB_LO, VERB_HI, VERB_K, VERB_GAIN)
        _accumulate(acc, i_vb, v_vb, 1.0)
    flags = (
        bool(src),
        bool(dst),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        verb == "cp",
        verb == "cd",
        verb == "head",
        redirect == ">",
        redirect == ">>",
        bool(dst) or redirect is not None,
        bool(src) and src == dst,
    )
    for j in range(N_FLAGS):
        acc[FLAG_LO + j] = FLAG_GAIN if flags[j] else -FLAG_GAIN
    idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
    val = np.fromiter(acc.values(), dtype=np.float32, count=len(acc))
    out = (idx, val)
    with _LOCK:
        if len(_CODE_CACHE) >= _CODE_CAP:
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
    row_ids = []
    counts = []
    cols = []
    vals = []
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        idx, val = _cmd_code(c)
        if idx.size == 0:
            continue
        row_ids.append(i)
        counts.append(idx.size)
        cols.append(idx)
        vals.append(val)
    if not cols:
        return out
    r = torch.from_numpy(
        np.repeat(np.asarray(row_ids, dtype=np.int64), np.asarray(counts, dtype=np.int64))
    ).to(out.device)
    c = torch.from_numpy(np.concatenate(cols)).to(out.device)
    v = torch.from_numpy(np.concatenate(vals)).to(device=out.device, dtype=out.dtype)
    out.index_put_((r, c), v, accumulate=True)
    return out


def collate(batch, device):
    maxn = max(1, max(int(s["z_obs"].shape[0]) for s in batch))
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
