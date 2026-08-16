import hashlib
import math
import threading

import numpy as np
import torch

D = 768

NAME = "r7_locationlattice_scopegated_code"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and every OBSERVATION token untouched, and rebuilds each "
    "COMMAND token as two aligned LOCATION LATTICES plus a command frame. A location lattice is one "
    "224-wide block-structured code of a single path: a wide exact-identity hash, then a narrower "
    "hash of the path's numeric-suffix-stripped stem, then a hash of its parent directory, then "
    "four replicated type bits (path present, path was created by an earlier rename because its "
    "last dot-component is all digits, path lives under the mutable workspace root, path names a "
    "file rather than a directory). The same path yields the same lattice in either role, so the "
    "source block and the destination block are aligned index-for-index and a single linear map "
    "returns the same address exactly when source and destination coincide; the family, directory "
    "and type sub-blocks are separate coordinate ranges rather than superposed, so an address map "
    "can suppress them while a step-typing map can read them. The command frame carries an "
    "undirected mention code over the distinct paths named in any role, a verb hash, and eight "
    "replicated form bits. The standardized command embedding survives, scaled, under the frame "
    "ONLY for commands that name no path under the workspace root: a command that touches a tracked "
    "file contributes no lexical channel at all, so nothing links a read of a relocated file to the "
    "origin of its content except the coded chain of moves. Pure per-command function of the command "
    "string, hence prefix-causal and identical when two contents exchange their moves."
)

SRC_LO, SRC_HI = 0, 224
DST_LO, DST_HI = 224, 448
MEN_LO, MEN_HI = 448, 704
VERB_LO, VERB_HI = 704, 736
FORM_LO, FORM_HI = 736, 768

LOC_W = SRC_HI - SRC_LO
ID_W, FAM_W, DIR_W, TYPE_W = 136, 48, 24, 16
ID_G, FAM_G, DIR_G, TYPE_G = 1.0, 0.5, 0.5, 0.5
LOC_ENERGY = ID_W * ID_G ** 2 + FAM_W * FAM_G ** 2 + DIR_W * DIR_G ** 2 + TYPE_W * TYPE_G ** 2
LOC_SCALE = math.sqrt(float(LOC_W) / LOC_ENERGY)
N_TYPE_BITS = 4

MEN_W = MEN_HI - MEN_LO
VERB_W = VERB_HI - VERB_LO
FORM_W = FORM_HI - FORM_LO
N_FORM_BITS = 8
MEN_G = 1.0
VERB_G = 1.0
FORM_G = 1.0

RES_LO = MEN_LO
RES_GAIN_OPEN = 0.6
RES_GAIN_TRACKED = 0.0
WORKSPACE_ROOT = "/tmp/"

CUPS_LAYOUT = "interleave2"

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}
_LOC_CACHE = {}
_SIGN_CACHE = {}
_MASK_CACHE = {}

_BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)] for b in range(256)],
    dtype=np.float32,
)

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")


def _signs(tag, text, width):
    key = (tag, text, width)
    hit = _SIGN_CACHE.get(key)
    if hit is not None:
        return hit
    dig = hashlib.blake2b((tag + "\x1f" + text).encode("utf-8", "replace"),
                          digest_size=width // 8).digest()
    out = _BYTE_SIGNS[np.frombuffer(dig, dtype=np.uint8)].reshape(-1)
    with _CACHE_LOCK:
        if len(_SIGN_CACHE) >= _CACHE_CAP:
            _SIGN_CACHE.clear()
        _SIGN_CACHE[key] = out
    return out


def _bit_block(bits, width, gain):
    n = len(bits)
    out = np.empty(width, dtype=np.float32)
    for j in range(width):
        out[j] = gain if bits[j % n] else -gain
    return out


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _basename(p):
    return p.rsplit("/", 1)[-1]


def _dirname(p):
    if not p:
        return ""
    head, sep, _ = p.rpartition("/")
    if not sep:
        return "."
    return head or "/"


def _stem(p):
    cur = p
    while True:
        head, sep, tail = cur.rpartition(".")
        if sep and head and tail.isdigit() and not head.endswith("/"):
            cur = head
        else:
            return cur


def _is_derived(p):
    base = _basename(p)
    head, sep, tail = base.rpartition(".")
    return bool(sep and head and tail.isdigit())


def _type_bits(p):
    if not p:
        return (False, False, False, False)
    base = _basename(_stem(p))
    named_file = "." in base[1:]
    return (True, _is_derived(p), p.startswith(WORKSPACE_ROOT), named_file)


def _location(path):
    hit = _LOC_CACHE.get(path)
    if hit is not None:
        return hit
    out = np.empty(LOC_W, dtype=np.float32)
    a = ID_W
    b = a + FAM_W
    c = b + DIR_W
    out[0:a] = ID_G * _signs("id", path, ID_W)
    out[a:b] = FAM_G * _signs("fam", _stem(path), FAM_W)
    out[b:c] = DIR_G * _signs("dir", _dirname(path), DIR_W)
    out[c:LOC_W] = _bit_block(_type_bits(path), TYPE_W, TYPE_G)
    out *= LOC_SCALE
    with _CACHE_LOCK:
        if len(_LOC_CACHE) >= _CACHE_CAP:
            _LOC_CACHE.clear()
        _LOC_CACHE[path] = out
    return out


def parse_command(cmd):
    toks = cmd.split()
    if not toks:
        return "", "", "", "", False, 0
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
        return verb, src, _norm_path(redir_target), redirect, True, len(positional)
    if verb in _DST_VERBS and len(positional) >= 2:
        return verb, src, _norm_path(positional[-1]), redirect, True, len(positional)
    return verb, src, src, redirect, False, len(positional)


def _mention(src, dst, explicit):
    named = [src]
    if explicit and dst != src:
        named.append(dst)
    named = [p for p in named if p]
    if not named:
        return _signs("men", "", MEN_W)
    acc = _signs("men", named[0], MEN_W).copy()
    for p in named[1:]:
        acc = acc + _signs("men", p, MEN_W)
    return acc * (1.0 / math.sqrt(float(len(named))))


def _build_code(cmd):
    verb, src, dst, redirect, explicit, n_pos = parse_command(cmd)
    row = np.zeros(D, dtype=np.float32)
    row[SRC_LO:SRC_HI] = _location(src)
    row[DST_LO:DST_HI] = _location(dst)
    row[MEN_LO:MEN_HI] = MEN_G * _mention(src, dst, explicit)
    row[VERB_LO:VERB_HI] = VERB_G * _signs("verb", verb, VERB_W)
    form = (
        verb == "mv",
        verb == "cat",
        verb == "ls",
        bool(explicit),
        redirect == ">",
        redirect == ">>",
        bool(explicit) and dst != src,
        n_pos >= 2,
    )
    row[FORM_LO:FORM_HI] = _bit_block(form, FORM_W, FORM_G)
    tracked = (src.startswith(WORKSPACE_ROOT) or dst.startswith(WORKSPACE_ROOT))
    res = RES_GAIN_TRACKED if tracked else RES_GAIN_OPEN
    return row, np.float32(res)


def _command_code(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    out = _build_code(cmd)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = out
    return out


def _res_mask(device, dtype):
    key = (str(device), str(dtype))
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.zeros(D, device=device, dtype=dtype)
    v[RES_LO:] = 1.0
    _MASK_CACHE[key] = v
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
    gains = np.empty(m, dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        row, g = _command_code(c if isinstance(c, str) else str(c))
        rows[i] = row
        gains[i] = g
    code = torch.from_numpy(rows).to(device=out.device, dtype=out.dtype)
    gain = torch.from_numpy(gains).to(device=out.device, dtype=out.dtype)
    mask = _res_mask(out.device, out.dtype)
    out[:m] = out[:m] * (gain.unsqueeze(1) * mask.unsqueeze(0)) + code
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
        cmd_pred = pred[:, 0::2].cpu()
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
        "cat /tmp/w/cups/g1/a.dat",
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg.1",
        "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat",
        "mv /tmp/w/cups/etc/b.cfg.1 /tmp/w/cups/etc/b.cfg.2",
        "cat /tmp/w/cups/etc/b.cfg.2",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
