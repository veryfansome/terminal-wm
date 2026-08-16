import hashlib
import threading

import numpy as np
import torch

D = 768

NAME = "r7_cupgen_dualrail_address"
DESCRIPTION = (
    "Keeps the cmd/obs interleave and every OBSERVATION token untouched, and rewrites each "
    "COMMAND token so that a source address and a destination address occupy two disjoint, "
    "index-for-index aligned 192-coordinate blocks that OVERWRITE the standardized command "
    "embedding (the embedding survives, scaled, only above coordinate 448). Both blocks are "
    "written by ONE address function, so the token at coordinate 192+k equals the token at "
    "coordinate k exactly when the command's destination is its source, and a single linear map "
    "applied to either block returns the same vector for the same path. An address is three "
    "aligned rails rather than one hash. Rail ID is a balanced Rademacher hash of the whole "
    "normalized path and matches only on exact identity. Rail CUP is a balanced Rademacher hash "
    "of the path with its trailing '.<digits>' generation stripped, so every name a single "
    "location ever wears shares one pattern. Rail GEN is a row of a 32-point Sylvester-Hadamard "
    "matrix indexed by that generation, zero when there is none, so two generations of one cup "
    "are exactly orthogonal in those coordinates and the same 32-symbol alphabet recurs in every "
    "trajectory and every image. The three rails give a graded address geometry - exact path 1.0, "
    "same cup different generation 0.133, same generation different cup 0.067, unrelated 0.0 - so "
    "a location can be matched as an identity OR as a cup whose occupant changes over time. Every "
    "hash pattern carries exactly as many +1 as -1 entries, so an address shifts neither the "
    "token mean nor the token variance and role reads sit at a fixed operating point. Eight form "
    "flags name the occupancy semantics directly (source present, explicit destination, mv so the "
    "source is EMPTIED, plain read, ls, '>' overwrite with the source retained, '>>' append with "
    "the source retained, destination distinct from source) and a verb hash sits beside them, "
    "both in overwritten coordinates rather than superposed on the embedding. The coding is a "
    "pure per-command function, so it is prefix-causal and the read command's token is bit-"
    "identical when two contents exchange their moves."
)

SRC_LO, SRC_HI = 0, 192
DST_LO, DST_HI = 192, 384
FLAG_LO, FLAG_HI = 384, 416
VERB_LO, VERB_HI = 416, 448
KEEP_LO = 448

ADDR_W = SRC_HI - SRC_LO
ID_OFF, ID_W = 0, 96
CUP_OFF, CUP_W = 96, 64
GEN_OFF, GEN_W = 160, 32
FLAG_W = FLAG_HI - FLAG_LO
VERB_W = VERB_HI - VERB_LO
N_FLAGS = 8

ID_GAIN = 1.0
CUP_GAIN = 0.5
GEN_GAIN = 0.5
FLAG_GAIN = 0.75
VERB_GAIN = 0.75
RAW_SCALE = 0.75

GEN_ORDER = 32
MAX_GEN = GEN_ORDER - 1

CUPS_LAYOUT = "interleave2"

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")
_DIGITS = frozenset("0123456789")

_CMD_CACHE_CAP = 60000
_PATH_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CMD_CACHE = {}
_PATH_CACHE = {}
_SCALE_CACHE = {}


def _hadamard(order):
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def _build_gen_table():
    had = _hadamard(GEN_ORDER)
    table = np.zeros((GEN_ORDER, GEN_W), dtype=np.float32)
    for g in range(1, GEN_ORDER):
        table[g] = had[g] * GEN_GAIN
    return table


GEN_TABLE = _build_gen_table()


def _balanced_signs(tag, text, width, gain):
    raw = hashlib.shake_256((tag + "\x1f" + text).encode("utf-8", "replace")).digest(4 * width)
    keys = np.frombuffer(raw, dtype=">u4")
    base = np.empty(width, dtype=np.float32)
    base[: width // 2] = gain
    base[width // 2:] = -gain
    return base[np.argsort(keys, kind="stable")]


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _split_generation(path):
    i = path.rfind(".")
    if i <= 0 or i == len(path) - 1:
        return path, 0
    frag = path[i + 1:]
    if len(frag) > 2:
        return path, 0
    for ch in frag:
        if ch not in _DIGITS:
            return path, 0
    v = int(frag)
    if v <= 0:
        return path, 0
    return path[:i], (v if v < MAX_GEN else MAX_GEN)


def _path_code(path):
    hit = _PATH_CACHE.get(path)
    if hit is not None:
        return hit
    out = np.zeros(ADDR_W, dtype=np.float16)
    if path:
        stem, gen = _split_generation(path)
        out[ID_OFF:ID_OFF + ID_W] = _balanced_signs("id", path, ID_W, ID_GAIN)
        out[CUP_OFF:CUP_OFF + CUP_W] = _balanced_signs("cup", stem, CUP_W, CUP_GAIN)
        out[GEN_OFF:GEN_OFF + GEN_W] = GEN_TABLE[gen]
    with _CACHE_LOCK:
        if len(_PATH_CACHE) >= _PATH_CACHE_CAP:
            _PATH_CACHE.clear()
        _PATH_CACHE[path] = out
    return out


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


def _build_row(cmd):
    verb, src, dst, redirect, explicit = parse_command(cmd)
    row = np.zeros(D, dtype=np.float16)
    row[SRC_LO:SRC_HI] = _path_code(src)
    row[DST_LO:DST_HI] = _path_code(dst)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat" and not explicit,
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        bool(explicit) and dst != src,
    )
    fv = np.empty(FLAG_W, dtype=np.float16)
    for j in range(FLAG_W):
        fv[j] = FLAG_GAIN if flags[j % N_FLAGS] else -FLAG_GAIN
    row[FLAG_LO:FLAG_HI] = fv
    if verb:
        row[VERB_LO:VERB_HI] = _balanced_signs("verb", verb, VERB_W, VERB_GAIN)
    return row


def _command_row(cmd):
    hit = _CMD_CACHE.get(cmd)
    if hit is not None:
        return hit
    out = _build_row(cmd)
    with _CACHE_LOCK:
        if len(_CMD_CACHE) >= _CMD_CACHE_CAP:
            _CMD_CACHE.clear()
        _CMD_CACHE[cmd] = out
    return out


def _keep_scale(device, dtype):
    key = (str(device), str(dtype))
    hit = _SCALE_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.zeros(D, device=device, dtype=dtype)
    v[KEEP_LO:] = RAW_SCALE
    _SCALE_CACHE[key] = v
    return v


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    if n == 0 or not cmds:
        return out
    m = min(n, len(cmds))
    if m <= 0:
        return out
    rows = np.empty((m, D), dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        rows[i] = _command_row(c if isinstance(c, str) else str(c))
    code = torch.from_numpy(rows).to(device=out.device, dtype=out.dtype)
    scale = _keep_scale(out.device, out.dtype)
    out[:m] = out[:m] * scale.unsqueeze(0) + code
    return out


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
        zc = code_cmds(list(s.get("cmds") or []), s["z_cmd"][:n])
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
    verbs = []
    for c in cmds:
        p = c.split()
        verbs.append(p[0] if p else "")
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs, "verbs": verbs}


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    cmds = [
        "ls -1 /",
        "cat /tmp/w/cups/etc/ssh/sshd_config.cfg",
        "cat /tmp/w/cups/g72/f654.dat",
        "mv /tmp/w/cups/g72/f654.dat /tmp/w/cups/etc/ssh/sshd_config.cfg.1",
        "mv /tmp/w/cups/etc/ssh/sshd_config.cfg.1 /tmp/w/cups/etc/ssh/sshd_config.cfg.2",
        "cat /tmp/w/cups/etc/ssh/sshd_config.cfg.2",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
