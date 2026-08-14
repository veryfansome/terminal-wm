import hashlib
import threading

import numpy as np
import torch

D = 768

NAME = "r6_readwrite_slot_addressing"
DESCRIPTION = (
    "Rewrites every COMMAND token as a read-address / write-address pair plus a small symbolic "
    "residue, and deliberately starves the lexical channel that the observation-name shortcut "
    "lives in. Observation tokens and the interleave layout are untouched. A command token is "
    "five fixed contiguous regions. The first 256 coordinates hold the READ address: a centered "
    "sign code of the command's first positional path, the location the command takes content "
    "FROM. The next 256 hold the WRITE address, the same code family applied to the location the "
    "command puts content AT: the second argument of a two-argument mv/cp/ln/install, or the "
    "redirection target of '>' / '>>', and otherwise the command's own path, so a plain read both "
    "reads and establishes its own slot. A command that names no path leaves both address regions "
    "at zero. The two regions are read off ONE generator per path string, so a path yields the "
    "identical bit pattern whether it is being read from or written to, and a single linear map "
    "aligns a write key with the read key of the command that consumes it. The next 64 "
    "coordinates carry a sign code of the verb, the next 64 carry eight replicated form flags "
    "(source present, explicit destination, destination distinct from source, mv, cat, '>>', '>', "
    "an option token present). Only the final 128 coordinates keep the standardized command "
    "embedding, at 0.5 scale, so lexical near-duplicate similarity between long path strings "
    "survives at a few percent of the token's energy instead of dominating it. The coding is a "
    "pure per-command function of the command string and that command's own embedding, so it is "
    "prefix-causal and bit-identical when two contents exchange their moves."
)

SRC_LO, SRC_HI = 0, 256
DST_LO, DST_HI = 256, 512
VERB_LO, VERB_HI = 512, 576
FLAG_LO, FLAG_HI = 576, 640
RAW_LO, RAW_HI = 640, 768

ADDR_W = SRC_HI - SRC_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
N_FLAGS = 8

PATH_GAIN = 1.2
VERB_GAIN = 0.8
FLAG_GAIN = 0.8
RAW_SCALE = 0.5

CUPS_LAYOUT = "interleave2"

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}
_SIGN_CACHE = {}

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")


def _sign_code(tag, text, width):
    key = (tag, text, width)
    hit = _SIGN_CACHE.get(key)
    if hit is not None:
        return hit
    dig = hashlib.shake_256((tag + "\x1f" + text).encode("utf-8", "replace")).digest(width // 8)
    bits = np.unpackbits(np.frombuffer(dig, dtype=np.uint8))
    v = bits.astype(np.float32) * 2.0 - 1.0
    v = v - float(v.mean())
    with _CACHE_LOCK:
        if len(_SIGN_CACHE) >= _CACHE_CAP:
            _SIGN_CACHE.clear()
        _SIGN_CACHE[key] = v
    return v


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
        return "", "", "", "", False, False
    verb = toks[0]
    positional = []
    redirect = ""
    redir_target = ""
    has_opt = False
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
            has_opt = True
        else:
            positional.append(t)
        i += 1
    src = _norm_path(positional[0]) if positional else ""
    if redir_target:
        return verb, src, _norm_path(redir_target), redirect, True, has_opt
    if verb in _DST_VERBS and len(positional) >= 2:
        return verb, src, _norm_path(positional[-1]), redirect, True, has_opt
    return verb, src, src, redirect, False, has_opt


def _build_code(cmd):
    verb, src, dst, redirect, explicit, has_opt = parse_command(cmd)
    row = np.zeros(RAW_LO, dtype=np.float32)
    if src:
        row[SRC_LO:SRC_HI] = PATH_GAIN * _sign_code("addr", src, ADDR_W)
    if dst:
        row[DST_LO:DST_HI] = PATH_GAIN * _sign_code("addr", dst, ADDR_W)
    row[VERB_LO:VERB_HI] = VERB_GAIN * _sign_code("verb", verb, VERB_W)
    flags = (
        bool(src),
        bool(explicit),
        bool(explicit) and dst != src,
        verb == "mv",
        verb == "cat",
        redirect == ">>",
        redirect == ">",
        bool(has_opt),
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
    block = np.empty((m, RAW_LO), dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        block[i] = _command_code(c if isinstance(c, str) else str(c))
    code = torch.from_numpy(block).to(device=out.device, dtype=out.dtype)
    out[:m, :RAW_LO] = code
    out[:m, RAW_LO:] = out[:m, RAW_LO:] * RAW_SCALE
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
        "cat /tmp/w/cups/etc/cron.d/backup.job",
        "cat /tmp/w/cups/g35/f385.tmp",
        "mv /tmp/w/cups/etc/cron.d/backup.job /tmp/w/cups/g35/f385.tmp.1",
        "mv /tmp/w/cups/g35/f385.tmp /tmp/w/cups/etc/cron.d/backup.job.2",
        "mv /tmp/w/cups/g35/f385.tmp.1 /tmp/w/cups/etc/cron.d/backup.job.3",
        "cat /tmp/w/cups/etc/cron.d/backup.job.3",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
