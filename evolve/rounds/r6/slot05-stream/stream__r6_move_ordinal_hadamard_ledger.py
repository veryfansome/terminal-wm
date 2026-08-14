import hashlib
import threading

import torch

from realenv import seq_worldmodel as M

NAME = "r6_move_ordinal_hadamard_ledger"
DESCRIPTION = (
    "Interleave2 layout, OBSERVATION tokens untouched, and every COMMAND token overwritten in 128 "
    "fixed coordinates with a two-role address ledger carrying TWO aligned channels per role. The "
    "IDENTITY channel is a 32-bit Rademacher hash of the full path string, written index-for-index "
    "into a READ block and a WRITE block, so one path yields one identical pattern in either role. "
    "The ORDER channel is the new mechanism: a move's destination is named by appending a numeric "
    "component to a stem, and that number is a per-window counter, so the final numeric component "
    "of a path is extracted as an ORDINAL and written as a row of a 16-point Sylvester-Hadamard "
    "matrix, again index-for-index into a READ and a WRITE block. Distinct ordinals therefore get "
    "EXACTLY orthogonal dense codes rather than merely near-orthogonal hashes, one shared linear "
    "map compares a command's read address against another command's write address, and a path "
    "with no admissible ordinal leaves the order blocks at zero instead of sharing a bucket. A "
    "numeric component is admissible only when it has no leading zero, is at most three digits, "
    "and follows a non-numeric component, which keeps versioned shared objects out of the channel. "
    "The remaining blocks hold a verb hash and ten replicated form flags. A command's token is a "
    "pure function of its own string and its own standardized embedding, so the coding is "
    "prefix-causal and the token at a read position is bit-identical under the native move chain "
    "and under a chain in which two contents exchange their moves."
)

D = M.D
CUPS_LAYOUT = "interleave2"

HASH_M = 32
ORD_M = 16
VERB_M = 12
FLAG_M = 20
N_FLAGS = 10

HASH_GAIN = 4.0
ORD_GAIN = 4.0
VERB_GAIN = 2.0
FLAG_GAIN = 2.0

CODE_M = 2 * HASH_M + 2 * ORD_M + VERB_M + FLAG_M

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")
_DIGITS = "0123456789"
_MAX_ORD_DIGITS = 3

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}
_HASH_CACHE = {}
_COL_CACHE = {}


def _column_order():
    return sorted(
        range(D),
        key=lambda i: hashlib.blake2b(b"ordinal-ledger-v1:" + str(i).encode("ascii"),
                                      digest_size=16).digest(),
    )


_ORDER = _column_order()
_SRC_HASH_COLS = _ORDER[0:HASH_M]
_DST_HASH_COLS = _ORDER[HASH_M:2 * HASH_M]
_SRC_ORD_COLS = _ORDER[2 * HASH_M:2 * HASH_M + ORD_M]
_DST_ORD_COLS = _ORDER[2 * HASH_M + ORD_M:2 * HASH_M + 2 * ORD_M]
_VERB_COLS = _ORDER[2 * HASH_M + 2 * ORD_M:2 * HASH_M + 2 * ORD_M + VERB_M]
_FLAG_COLS = _ORDER[2 * HASH_M + 2 * ORD_M + VERB_M:CODE_M]
_ALL_COLS = (_SRC_HASH_COLS + _DST_HASH_COLS + _SRC_ORD_COLS + _DST_ORD_COLS
             + _VERB_COLS + _FLAG_COLS)


def _hadamard_rows(m):
    rows = []
    for i in range(m):
        row = []
        for j in range(m):
            bits = bin(i & j).count("1")
            row.append(-1.0 if bits & 1 else 1.0)
        rows.append(row)
    return rows


_HADAMARD = _hadamard_rows(ORD_M)
_N_ORD_ROWS = ORD_M - 1
_ZERO_ORD = [0.0] * ORD_M


def _sign_bits(text, width, gain):
    digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=width // 8).digest()
    out = []
    for b in digest:
        for k in range(8):
            out.append(gain if (b >> (7 - k)) & 1 else -gain)
    return out


def _path_hash(path):
    hit = _HASH_CACHE.get(path)
    if hit is not None:
        return hit
    out = _sign_bits("path\x1f" + path, HASH_M, HASH_GAIN)
    with _CACHE_LOCK:
        if len(_HASH_CACHE) >= _CACHE_CAP:
            _HASH_CACHE.clear()
        _HASH_CACHE[path] = out
    return out


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _all_digits(s):
    if not s:
        return False
    for ch in s:
        if ch not in _DIGITS:
            return False
    return True


def path_ordinal(path):
    if not path:
        return 0
    base = path.rsplit("/", 1)[-1]
    parts = base.split(".")
    if len(parts) < 2:
        return 0
    last = parts[-1]
    if not _all_digits(last) or len(last) > _MAX_ORD_DIGITS or last[0] == "0":
        return 0
    prev = parts[-2]
    if not prev or _all_digits(prev):
        return 0
    return int(last)


def _ordinal_code(n):
    if n < 1:
        return _ZERO_ORD
    row = _HADAMARD[((n - 1) % _N_ORD_ROWS) + 1]
    return [ORD_GAIN * v for v in row]


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


def command_code_values(cmd):
    verb, src, dst, redirect, explicit = parse_command(cmd)
    src_key = src if src else "\x00nopath\x1f" + cmd
    dst_key = dst if dst else src_key
    src_ord = path_ordinal(src)
    dst_ord = path_ordinal(dst)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        src_ord >= 1,
        dst_ord >= 1,
        dst != src,
    )
    vals = []
    vals.extend(_path_hash(src_key))
    vals.extend(_path_hash(dst_key))
    vals.extend(_ordinal_code(src_ord))
    vals.extend(_ordinal_code(dst_ord))
    vals.extend(_sign_bits("verb\x1f" + verb, 16, VERB_GAIN)[:VERB_M])
    vals.extend(FLAG_GAIN if flags[j % N_FLAGS] else -FLAG_GAIN for j in range(FLAG_M))
    return vals


def _command_row(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    row = torch.tensor(command_code_values(cmd), dtype=torch.float32)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = row
    return row


def _cols(device):
    key = str(device)
    hit = _COL_CACHE.get(key)
    if hit is None:
        hit = torch.tensor(_ALL_COLS, dtype=torch.long, device=device)
        _COL_CACHE[key] = hit
    return hit


def code_cmds(cmds, z_cmd):
    z = z_cmd if torch.is_tensor(z_cmd) else torch.as_tensor(z_cmd)
    out = z.detach().clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    seq = list(cmds) if cmds is not None else []
    m = min(n, len(seq))
    if m <= 0:
        return out
    rows = [_command_row(c if isinstance(c, str) else str(c)) for c in seq[:m]]
    vals = torch.stack(rows).to(device=out.device, dtype=out.dtype)
    out[:m].index_copy_(1, _cols(out.device), vals)
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
        "cat /tmp/w/cups/g53/f149.tmp",
        "cat /tmp/w/cups/etc/ssh/sshd_config.cfg",
        "mv /tmp/w/cups/g53/f149.tmp /tmp/w/cups/etc/ssh/sshd_config.cfg.1",
        "mv /tmp/w/cups/etc/ssh/sshd_config.cfg.1 /tmp/w/cups/g53/f149.tmp.2",
        "cat /tmp/w/cups/g53/f149.tmp.2",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
