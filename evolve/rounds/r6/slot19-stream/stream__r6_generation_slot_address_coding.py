import hashlib
import math
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r6_generation_slot_address_coding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave, every OBSERVATION token untouched, and the parent's additive "
    "sparse hash of source path, destination path, verb and seven form flags, but narrows the two "
    "path blocks to 312 coordinates each to free an 80-coordinate GENERATION channel that is "
    "OVERWRITTEN rather than added. A path's generation is the integer of its trailing '.<digits>' "
    "component, 0 when it has none; the channel writes the source generation into coordinates "
    "624-663 and the destination generation into coordinates 664-703 from ONE shared 16-entry "
    "codebook, index for index, so the same generation yields the same 40-vector in either role "
    "and a fixed coordinate offset turns a source read into a destination key. A codebook entry "
    "is a row of a 16-point Sylvester-Hadamard matrix laid down twice under a fixed column "
    "permutation, so two distinct generations are EXACTLY orthogonal in those 32 coordinates "
    "however many of them appear, followed by a present flag, a linear ramp and a half-period "
    "sine/cosine pair so ordinal distance stays linearly readable. Overwriting rather than adding "
    "removes the standardized embedding from those coordinates, leaving an address alphabet that "
    "is the same in every trajectory and every image instead of a fresh random pattern per path. "
    "Nothing outside a command's own string and its own embedding enters that command's token, so "
    "the coding is position-wise, prefix-causal, and identical for a read command whichever way "
    "the moves before it were routed."
)

D = M.D
CUPS_LAYOUT = "interleave2"

SRC_LO, SRC_HI = 0, 312
DST_LO, DST_HI = 312, 624
SGEN_LO, SGEN_HI = 624, 664
DGEN_LO, DGEN_HI = 664, 704
VERB_LO, VERB_HI = 704, 752
FLAG_LO = 752
N_FLAGS = 7

PATH_K = 24
VERB_K = 8
PATH_GAIN = 2.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0
GEN_GAIN = 2.0

GEN_W = 40
GEN_CODE_W = 32
GEN_ORDER = 16
MAX_GEN = GEN_ORDER - 1

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_DIGITS = frozenset("0123456789")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}


def _hadamard(order):
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def _fixed_permutation(width):
    raw = hashlib.shake_256(b"gen-column-permutation-v1").digest(4 * width)
    keys = [int.from_bytes(raw[4 * i:4 * i + 4], "big") for i in range(width)]
    return np.asarray(sorted(range(width), key=lambda i: (keys[i], i)), dtype=np.int64)


def _build_gen_table():
    had = _hadamard(GEN_ORDER)
    perm = _fixed_permutation(GEN_ORDER)
    half = GEN_CODE_W // 2
    table = np.zeros((GEN_ORDER, GEN_W), dtype=np.float32)
    for g in range(GEN_ORDER):
        row = had[(g + 1) % GEN_ORDER]
        table[g, :half] = row * GEN_GAIN
        table[g, half:GEN_CODE_W] = row[perm] * GEN_GAIN
        table[g, GEN_CODE_W] = GEN_GAIN if g > 0 else -GEN_GAIN
        table[g, GEN_CODE_W + 1] = GEN_GAIN * (float(g) / float(MAX_GEN))
        table[g, GEN_CODE_W + 2] = GEN_GAIN * math.sin(math.pi * float(g) / float(MAX_GEN))
        table[g, GEN_CODE_W + 3] = GEN_GAIN * math.cos(math.pi * float(g) / float(MAX_GEN))
    return table


GEN_TABLE = _build_gen_table()


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


def _generation_of(path):
    if not path:
        return 0
    i = path.rfind(".")
    if i <= 0 or i == len(path) - 1:
        return 0
    frag = path[i + 1:]
    if len(frag) > 3:
        return 0
    for ch in frag:
        if ch not in _DIGITS:
            return 0
    v = int(frag)
    if v <= 0:
        return 0
    return v if v < MAX_GEN else MAX_GEN


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
    out = (idx, val, _generation_of(src), _generation_of(dst))
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
    if m <= 0:
        return out
    rows = []
    cols = []
    vals = []
    gen_rows = []
    gen_block = np.zeros((m, 2 * GEN_W), dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        idx, val, sgen, dgen = _compact(c)
        if idx.size:
            rows.append(np.full(idx.size, i, dtype=np.int64))
            cols.append(idx)
            vals.append(val)
        gen_block[i, :GEN_W] = GEN_TABLE[sgen]
        gen_block[i, GEN_W:] = GEN_TABLE[dgen]
        gen_rows.append(i)
    if rows:
        r = torch.from_numpy(np.concatenate(rows)).to(out.device)
        c = torch.from_numpy(np.concatenate(cols)).to(out.device)
        v = torch.from_numpy(np.concatenate(vals)).to(device=out.device, dtype=out.dtype)
        out.index_put_((r, c), v, accumulate=True)
    if gen_rows:
        ridx = torch.as_tensor(gen_rows, dtype=torch.long, device=out.device)
        gvals = torch.from_numpy(np.ascontiguousarray(gen_block[gen_rows])).to(
            device=out.device, dtype=out.dtype)
        gcols = torch.arange(SGEN_LO, DGEN_HI, dtype=torch.long, device=out.device)
        sel = out.index_select(0, ridx)
        sel.index_copy_(1, gcols, gvals)
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
        tok[bi, 0:2 * n:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = zo.to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo.to(tgt.dtype)
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
    cmds = [
        "ls -1 /",
        "cat /tmp/w/cups/g1/a.dat",
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg.1",
        "mv /tmp/w/cups/etc/b.cfg.1 /tmp/w/cups/etc/b.cfg.2",
        "cat /tmp/w/cups/etc/b.cfg.2",
        "ls /a",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
