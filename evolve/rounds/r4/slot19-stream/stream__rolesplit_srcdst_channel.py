import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "rolesplit_srcdst_channel"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and every observation token untouched, and rewrites a "
    "fixed 136-coordinate prefix of each COMMAND token into a role-split channel. The channel "
    "clips the standardized command embedding on those coordinates to +/-4 and attenuates it to a "
    "quarter of its amplitude, then writes, in place of the removed amplitude, a dense sign code of the command's "
    "source path in coordinates 0-63, the SAME sign code alphabet applied to the command's "
    "destination path in coordinates 64-127, and eight form bits (source present, explicit "
    "destination, mv, cat, ls, '>', '>>', destination differs from source) in coordinates "
    "128-135. A path's code is the +/-1 expansion of a SHAKE-256 digest of its normalized "
    "string, so the same path yields the identical 64-vector in whichever role it appears and "
    "two paths differing by one trailing character yield codes with expected zero overlap. "
    "Attenuating the carrier inside the channel, rather than adding on top of it, removes the "
    "common-mode component that near-identical command strings share, so a fixed linear read of "
    "either block returns the path identity with the shared command content suppressed. Nothing "
    "outside a command's own string and its own embedding enters that command's token."
)

D = M.D
CUPS_LAYOUT = "interleave2"

BLOCK = 64
SRC_LO = 0
DST_LO = SRC_LO + BLOCK
FLAG_LO = DST_LO + BLOCK
N_FLAGS = 8
CHAN_HI = FLAG_LO + N_FLAGS

CARRIER_KEEP = 0.25
CARRIER_CLIP = 4.0
PATH_GAIN = 1.0
FLAG_GAIN = 2.0

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")

_CACHE_CAP = 200000
_LOCK = threading.Lock()
_CHAN_CACHE = {}
_PATH_CACHE = {}


def _sign_code(path):
    hit = _PATH_CACHE.get(path)
    if hit is not None:
        return hit
    raw = hashlib.shake_256(("path\x00" + path).encode("utf-8")).digest(BLOCK // 8)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:BLOCK]
    code = (bits.astype(np.float32) * 2.0 - 1.0) * PATH_GAIN
    with _LOCK:
        if len(_PATH_CACHE) >= _CACHE_CAP:
            _PATH_CACHE.clear()
        _PATH_CACHE[path] = code
    return code


def _norm_path(p):
    if not p:
        return ""
    if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
        p = p[1:-1]
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


def _channel(cmd):
    hit = _CHAN_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, src, dst, redirect, explicit = _parse(cmd)
    v = np.zeros(CHAN_HI, dtype=np.float32)
    if src:
        v[SRC_LO:SRC_LO + BLOCK] = _sign_code(src)
    if dst:
        v[DST_LO:DST_LO + BLOCK] = _sign_code(dst)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        bool(dst) and dst != src,
    )
    for j in range(N_FLAGS):
        v[FLAG_LO + j] = FLAG_GAIN if flags[j] else -FLAG_GAIN
    with _LOCK:
        if len(_CHAN_CACHE) >= _CACHE_CAP:
            _CHAN_CACHE.clear()
        _CHAN_CACHE[cmd] = v
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
    chan = np.zeros((m, CHAN_HI), dtype=np.float32)
    any_row = False
    for i in range(m):
        c = cmds[i]
        if isinstance(c, str) and c:
            chan[i] = _channel(c)
            any_row = True
    if not any_row:
        return out
    add = torch.from_numpy(chan).to(device=out.device, dtype=out.dtype)
    carrier = out[:m, :CHAN_HI].clamp(-CARRIER_CLIP, CARRIER_CLIP)
    out[:m, :CHAN_HI] = CARRIER_KEEP * carrier + add
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
