import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r7_bam_slot_address_link_code"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and leaves OBSERVATION tokens untouched, and re-codes "
    "each COMMAND token as its standardized command embedding plus a two-slot link code built "
    "only from that command's own string. The address space is split into two DISJOINT blocks of "
    "equal width, aligned coordinate-for-coordinate: an OUT slot and an IN slot. One path yields "
    "one dense Rademacher pattern; that SAME pattern is written into the OUT slot when the path "
    "is the command's source or its plain argument, and into the IN slot when the path is the "
    "command's destination or redirect target. A command with no explicit destination leaves the "
    "IN slot empty, so only link-creating commands occupy it. Because the two slots are disjoint "
    "and index-aligned, the swap operator J that exchanges them is an involution: matching a "
    "query's OUT slot against keys' IN slots retrieves the command that moved content INTO a named "
    "location and hands back that command's source in the OUT slot, and the same J run the other "
    "way retrieves the command that moved content OUT of a named location and hands back its "
    "destination. Both directions carry the full address energy and neither is contaminated by the "
    "other role, because a source mention and a destination mention never share a coordinate. The "
    "standardized embedding is attenuated inside the two address blocks and kept at full strength "
    "in the verb and flag blocks, so correlated command sentences stop leaking a large common-mode "
    "term into address inner products. A verb hash and sixteen replicated form flags (source "
    "present, explicit destination, mv, cat, ls, cp, cd, head, '>', '>>', link-creating, pure read, "
    "two or more paths, rm, any redirect, no path) fill the tail. Nothing outside a command's own "
    "string and its own embedding ever enters that command's token, so the coding is prefix-causal, "
    "position-independent, and identical at a read position under the native and role-swapped "
    "chains."
)

D = M.D
CUPS_LAYOUT = "interleave2"

OUT_LO, OUT_HI = 0, 320
IN_LO, IN_HI = 320, 640
VERB_LO, VERB_HI = 640, 704
FLAG_LO, FLAG_HI = 704, 768

ADDR_W = OUT_HI - OUT_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
N_FLAGS = 16

PATH_GAIN = 1.5
VERB_GAIN = 1.2
FLAG_GAIN = 1.5
RAW_ADDR = 0.4

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>")

_CACHE_CAP = 300000
_LOCK = threading.Lock()
_SIGN_CACHE = {}
_CODE_CACHE = {}
_SCALE_CACHE = {}

_BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)] for b in range(256)],
    dtype=np.float32,
)


def _signs(tag, text, width):
    key = (tag, text, width)
    hit = _SIGN_CACHE.get(key)
    if hit is not None:
        return hit
    dig = hashlib.blake2b((tag + "\x1f" + text).encode("utf-8", "replace"),
                          digest_size=width // 8).digest()
    out = _BYTE_SIGNS[np.frombuffer(dig, dtype=np.uint8)].reshape(-1)
    with _LOCK:
        if len(_SIGN_CACHE) >= _CACHE_CAP:
            _SIGN_CACHE.clear()
        _SIGN_CACHE[key] = out
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
        return "", "", "", None, 0
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
    return verb, src, dst, redirect, len(paths)


def _build_code(cmd):
    verb, src, dst, redirect, n_paths = _parse(cmd)
    row = np.zeros(D, dtype=np.float32)
    if src:
        row[OUT_LO:OUT_HI] = PATH_GAIN * _signs("addr", src, ADDR_W)
    if dst:
        row[IN_LO:IN_HI] = PATH_GAIN * _signs("addr", dst, ADDR_W)
    row[VERB_LO:VERB_HI] = VERB_GAIN * _signs("verb", verb, VERB_W)
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
        bool(src) and not dst,
        n_paths >= 2,
        verb == "rm",
        redirect is not None,
        n_paths == 0,
    )
    fv = np.empty(FLAG_W, dtype=np.float32)
    for j in range(FLAG_W):
        fv[j] = FLAG_GAIN if flags[j % N_FLAGS] else -FLAG_GAIN
    row[FLAG_LO:FLAG_HI] = fv
    return row


def _command_code(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    out = _build_code(cmd)
    with _LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = out
    return out


def _raw_scale(device, dtype):
    key = (str(device), str(dtype))
    hit = _SCALE_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.ones(D, device=device, dtype=dtype)
    v[OUT_LO:IN_HI] = RAW_ADDR
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
        rows[i] = _command_code(c if isinstance(c, str) else str(c))
    code = torch.from_numpy(rows).to(device=out.device, dtype=out.dtype)
    scale = _raw_scale(out.device, out.dtype)
    out[:m] = out[:m] * scale.unsqueeze(0) + code
    return out


def collate(batch, device):
    maxn = max(1, max(int(s["z_obs"].shape[0]) for s in batch))
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
        "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat",
        "cat /tmp/w/cups/etc/b.cfg.1",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
