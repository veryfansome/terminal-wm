import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r6_isonorm_role_address_frame"
DESCRIPTION = (
    "Keeps the cmd/obs interleave and every OBSERVATION token untouched, and rebuilds each COMMAND "
    "token as a fixed five-block frame of constant geometry: a 192-wide SOURCE address, a 192-wide "
    "DESTINATION address, a 32-wide verb hash, a 32-wide form-flag block and a 320-wide view of the "
    "standardized command embedding. The two address blocks are index-aligned and drawn from ONE "
    "hash family, so a path yields the SAME 192-bit pattern whether it appears as a source or as a "
    "destination, and a linear map that reads the source block of one command matches a linear map "
    "that reads the destination block of an earlier one exactly when the two commands name the same "
    "location; paths that differ only by a trailing '.k' rename suffix are made near-orthogonal "
    "instead of near-identical. Every block is exactly zero-sum and exactly fixed-norm: each address "
    "code is balanced (96 positive, 96 negative), the verb hash is balanced, each of the eight form "
    "flags is written as a (+v,+v,-v,-v) quadruple, and the embedding view is a fixed 320-coordinate "
    "subset that is centred and rescaled to norm sqrt(320). The whole 768-vector therefore has mean "
    "exactly zero and squared norm exactly 768 for EVERY command, so a LayerNorm over the command "
    "token is a single fixed scalar and the address readout sees the code at one scale rather than "
    "at a scale that drifts with the sentence. The coding is a per-command pure function of the "
    "command string and its own embedding row, so it is prefix-causal and the read token is "
    "unchanged when two contents exchange their moves."
)

D = M.D
CUPS_LAYOUT = "interleave2"

SRC_LO, SRC_HI = 0, 192
DST_LO, DST_HI = 192, 384
VERB_LO, VERB_HI = 384, 416
FLAG_LO, FLAG_HI = 416, 448
RAW_LO, RAW_HI = 448, 768

ADDR_W = SRC_HI - SRC_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
RAW_W = RAW_HI - RAW_LO
CODE_W = RAW_LO
N_FLAGS = 8
FLAG_REP = FLAG_W // N_FLAGS

RAW_NORM = float(np.sqrt(RAW_W))
NORM_EPS = 1e-6

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")

_CACHE_CAP = 300000
_CACHE_LOCK = threading.Lock()
_CMD_CACHE = {}
_ADDR_CACHE = {}
_VERB_CACHE = {}
_COL_CACHE = {}


def _order(seed_bytes, m):
    raw = hashlib.shake_256(seed_bytes).digest(4 * m)
    keys = np.frombuffer(raw, dtype=">u4")
    return np.argsort(keys, kind="stable")


def _balanced(tag, text, m):
    o = _order((tag + "\x1f" + text).encode("utf-8", "replace"), m)
    v = np.full(m, -1.0, dtype=np.float32)
    v[o[: m // 2]] = 1.0
    return v


RAW_COLS_NP = np.sort(_order(b"isonorm-raw-view-v1", D)[:RAW_W]).astype(np.int64)


def _raw_cols(device):
    key = str(device)
    hit = _COL_CACHE.get(key)
    if hit is None:
        hit = torch.from_numpy(RAW_COLS_NP).to(device)
        _COL_CACHE[key] = hit
    return hit


def _addr_code(path):
    hit = _ADDR_CACHE.get(path)
    if hit is not None:
        return hit
    out = _balanced("addr", path, ADDR_W)
    with _CACHE_LOCK:
        if len(_ADDR_CACHE) >= _CACHE_CAP:
            _ADDR_CACHE.clear()
        _ADDR_CACHE[path] = out
    return out


def _verb_code(verb):
    hit = _VERB_CACHE.get(verb)
    if hit is not None:
        return hit
    out = _balanced("verb", verb, VERB_W)
    with _CACHE_LOCK:
        if len(_VERB_CACHE) >= _CACHE_CAP:
            _VERB_CACHE.clear()
        _VERB_CACHE[verb] = out
    return out


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def parse_command(cmd):
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
        if t in _REDIR_TOKENS:
            redirect = ">>" if t == ">>" else ">"
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
    src = _norm_path(positional[0]) if positional else ""
    if redir_target:
        return verb, src, _norm_path(redir_target), redirect, True
    if verb in _DST_VERBS and len(positional) >= 2:
        return verb, src, _norm_path(positional[-1]), redirect, True
    return verb, src, src, redirect, False


def _build_code(cmd):
    verb, src, dst, redirect, explicit = parse_command(cmd)
    row = np.empty(CODE_W, dtype=np.float32)
    row[SRC_LO:SRC_HI] = _addr_code(src)
    row[DST_LO:DST_HI] = _addr_code(dst)
    row[VERB_LO:VERB_HI] = _verb_code(verb)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        dst == src,
    )
    for j in range(N_FLAGS):
        v = 1.0 if flags[j] else -1.0
        base = FLAG_LO + j * FLAG_REP
        row[base + 0] = v
        row[base + 1] = v
        row[base + 2] = -v
        row[base + 3] = -v
    return row


def _command_code(cmd):
    hit = _CMD_CACHE.get(cmd)
    if hit is not None:
        return hit
    out = _build_code(cmd)
    with _CACHE_LOCK:
        if len(_CMD_CACHE) >= _CACHE_CAP:
            _CMD_CACHE.clear()
        _CMD_CACHE[cmd] = out
    return out


def code_cmds(cmds, z_cmd):
    if z_cmd.dim() != 2 or int(z_cmd.shape[1]) != D:
        return z_cmd.clone()
    n = int(z_cmd.shape[0])
    if n == 0:
        return z_cmd.clone()
    seq = list(cmds) if cmds else []
    rows = np.empty((n, CODE_W), dtype=np.float32)
    for i in range(n):
        c = seq[i] if i < len(seq) else ""
        if not isinstance(c, str):
            c = str(c)
        rows[i] = _command_code(c)
    code = torch.from_numpy(rows).to(device=z_cmd.device, dtype=z_cmd.dtype)
    view = z_cmd.index_select(1, _raw_cols(z_cmd.device))
    view = view - view.mean(dim=1, keepdim=True)
    scale = RAW_NORM / view.norm(dim=1, keepdim=True).clamp_min(NORM_EPS)
    view = view * scale
    return torch.cat([code, view], dim=1)


def collate(batch, device):
    maxn = max(int(s["z_obs"].shape[0]) for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    for bi, s in enumerate(batch):
        n = int(s["z_obs"].shape[0])
        if n <= 0:
            continue
        zo = s["z_obs"][:n]
        zc = code_cmds(list(s.get("cmds") or [])[:n], s["z_cmd"][:n])
        tok[bi, 0:2 * n:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = zo.to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo.to(tgt.dtype)
        cmd_mask[bi, :n] = True
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
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
    cmds = [
        "ls -1 /",
        "cat /tmp/w/cups/g1/a.dat",
        "cat /tmp/w/cups/etc/b.cfg",
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg.1",
        "mv /tmp/w/cups/etc/b.cfg.1 /tmp/w/cups/g1/a.dat.2",
        "cat /tmp/w/cups/g1/a.dat.2",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
