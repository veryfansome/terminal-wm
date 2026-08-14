import hashlib
import threading

import numpy as np
import torch

D = 768

NAME = "r5_dualrole_address_mention_code"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and every OBSERVATION token untouched, and rewrites each "
    "COMMAND token as four fixed coordinate regions computed from that command string alone. The "
    "first two regions, equal width and aligned index-for-index, are OVERWRITTEN (the standardized "
    "embedding is zeroed there) with a dense Rademacher hash of the command's source path and the "
    "same hash of its destination path, so one path yields one identical bit pattern in either "
    "role and a single linear map applied to the two regions returns the SAME vector exactly when "
    "source and destination are the same path and near-orthogonal vectors otherwise; a command "
    "with no explicit destination takes its source as destination, and a command with no path "
    "takes a fixed null pattern. The third region carries an undirected MENTION code, a second "
    "independent hash family summed over the paths the command names in any role and renormalized, "
    "so plain token cosine between two commands measures whether they touch a common path rather "
    "than whether their sentences look alike. The fourth region carries a hash of the verb and "
    "eight replicated form flags (source present, explicit destination, mv, cat, ls, '>', '>>', "
    "destination distinct from source). The standardized command embedding survives, scaled, only "
    "under the mention and form regions. The coding is a pure per-command function, so it is "
    "prefix-causal and unchanged when two contents exchange their moves."
)

SRC_LO, SRC_HI = 0, 224
DST_LO, DST_HI = 224, 448
MEN_LO, MEN_HI = 448, 704
VERB_LO, VERB_HI = 704, 736
FLAG_LO, FLAG_HI = 736, 768

ADDR_W = SRC_HI - SRC_LO
MEN_W = MEN_HI - MEN_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
N_FLAGS = 8

RAW_SCALE = 0.6
PATH_GAIN = 1.0
MEN_GAIN = 1.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0
_INV_SQRT2 = 0.7071067811865476

CUPS_LAYOUT = "interleave2"

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}
_PATH_CACHE = {}
_SCALE_CACHE = {}

_BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)] for b in range(256)],
    dtype=np.float32,
)

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")


def _signs(text, width):
    dig = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=width // 8).digest()
    return _BYTE_SIGNS[np.frombuffer(dig, dtype=np.uint8)].reshape(-1)


def _cached_signs(tag, text, width):
    key = (tag, text)
    hit = _PATH_CACHE.get(key)
    if hit is not None:
        return hit
    out = _signs(tag + "\x1f" + text, width)
    with _CACHE_LOCK:
        if len(_PATH_CACHE) >= _CACHE_CAP:
            _PATH_CACHE.clear()
        _PATH_CACHE[key] = out
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
    row = np.zeros(D, dtype=np.float32)
    row[SRC_LO:SRC_HI] = PATH_GAIN * _cached_signs("addr", src, ADDR_W)
    row[DST_LO:DST_HI] = PATH_GAIN * _cached_signs("addr", dst, ADDR_W)
    if explicit and dst != src:
        men = (_cached_signs("mention", src, MEN_W)
               + _cached_signs("mention", dst, MEN_W)) * _INV_SQRT2
    else:
        men = _cached_signs("mention", src, MEN_W)
    row[MEN_LO:MEN_HI] = MEN_GAIN * men
    row[VERB_LO:VERB_HI] = VERB_GAIN * _cached_signs("verb", verb, VERB_W)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        bool(explicit) and dst != src,
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


def _raw_scale(device, dtype):
    key = (str(device), str(dtype))
    hit = _SCALE_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.zeros(D, device=device, dtype=dtype)
    v[MEN_LO:] = RAW_SCALE
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
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg",
        "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat",
        "mv /tmp/w/cups/etc/b.cfg /tmp/w/cups/etc/b.cfg.1",
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
