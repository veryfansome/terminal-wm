import hashlib
import threading

import numpy as np
import torch

D = 768

NAME = "r7_rename_chain_origin_address_stream"
DESCRIPTION = (
    "Interleave2 layout, OBSERVATION tokens untouched, every COMMAND token rewritten as six "
    "fixed coordinate bands plus a 0.25-scaled copy of the standardized command embedding. Four "
    "bands of equal width share ONE Rademacher address family h(path), so the same path yields "
    "the identical +/-1 pattern in whichever band it lands and a single linear map can compare "
    "any band against any other: READ carries the path of a command that has no destination "
    "(cat P, ls P), SRC and DST carry the source and destination of a command that has one, and "
    "ORIGIN carries the address of the path where the moved content ENTERED the window. ORIGIN is "
    "the only band that is not a per-command function: a rename table is carried left to right "
    "over the command strings (mv/cp/ln retarget an entry, '>' overwrites one, '>>' invalidates "
    "one) and a move's ORIGIN band is written with the fully compressed root of its source's "
    "rename chain, in the style of union-find path compression, so a depth-k chain of silent "
    "moves is readable in ONE association instead of k. The table is built strictly in prefix "
    "order, so the coding is prefix-causal, and ORIGIN is written ONLY on move verbs with two "
    "path arguments, so a command with no destination -- which is what a scored read is -- gets a "
    "token that is a pure function of its own string and is therefore bit-identical under the "
    "native move chain and under a chain in which two contents exchange their moves. A fifth band "
    "carries a role-free MENTION code, the renormalized sum of a second hash family over every "
    "path the command names in any role, so two commands touching a common path stay similar "
    "under plain cosine whatever roles they used; the sixth carries a verb hash and eight "
    "replicated form flags. A band whose role a command does not fill is left at zero, so token "
    "cosine measures shared paths rather than shared command shape."
)

CUPS_LAYOUT = "interleave2"

MEN_LO, MEN_HI = 0, 256
READ_LO, READ_HI = 256, 368
SRC_LO, SRC_HI = 368, 480
DST_LO, DST_HI = 480, 592
ORG_LO, ORG_HI = 592, 704
VERB_LO, VERB_HI = 704, 736
FLAG_LO, FLAG_HI = 736, 768

MEN_W = MEN_HI - MEN_LO
ADDR_W = READ_HI - READ_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
N_FLAGS = 8

RAW_KEEP = 0.25
ADDR_GAIN = 1.0
MEN_GAIN = 1.0
FORM_GAIN = 0.5

MOVE_VERBS = ("mv", "cp", "ln")

_PATH_CAP = 65536
_ROLE_CAP = 131072
_ROW_CAP = 16384
_LOCK = threading.Lock()
_ADDR_CACHE = {}
_MEN_CACHE = {}
_ROLE_CACHE = {}
_ROW_CACHE = {}

_BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)] for b in range(256)],
    dtype=np.float32,
)


def _signs(text, width):
    dig = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=width // 8).digest()
    return _BYTE_SIGNS[np.frombuffer(dig, dtype=np.uint8)].reshape(-1)


def _addr(path):
    hit = _ADDR_CACHE.get(path)
    if hit is not None:
        return hit
    out = _signs("addr\x1f" + path, ADDR_W)
    with _LOCK:
        if len(_ADDR_CACHE) >= _PATH_CAP:
            _ADDR_CACHE.clear()
        _ADDR_CACHE[path] = out
    return out


def _men(path):
    hit = _MEN_CACHE.get(path)
    if hit is not None:
        return hit
    out = _signs("mention\x1f" + path, MEN_W)
    with _LOCK:
        if len(_MEN_CACHE) >= _PATH_CAP:
            _MEN_CACHE.clear()
        _MEN_CACHE[path] = out
    return out


ABSENT_ADDR = np.zeros(ADDR_W, dtype=np.float32)
ABSENT_MEN = np.zeros(MEN_W, dtype=np.float32)


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
        return "", [], "", None
    verb = toks[0]
    positional = []
    redirect = ""
    target = None
    expect = False
    for t in toks[1:]:
        if expect:
            target = t
            expect = False
        elif t == ">" or t == ">>":
            redirect = t
            expect = True
        elif t.startswith(">>") and len(t) > 2:
            redirect = ">>"
            target = t[2:]
        elif t.startswith(">") and len(t) > 1:
            redirect = ">"
            target = t[1:]
        elif t.startswith("-") and len(t) > 1:
            continue
        else:
            positional.append(t)
    return verb, positional, redirect, target


def command_roles(cmd):
    hit = _ROLE_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, positional, redirect, target = parse_command(cmd)
    read_path = ""
    src = ""
    dst = ""
    is_move = False
    if verb in MOVE_VERBS and len(positional) >= 2:
        src = _norm_path(positional[0])
        dst = _norm_path(positional[-1])
        is_move = True
    elif target is not None:
        src = _norm_path(positional[0]) if positional else ""
        dst = _norm_path(target)
    else:
        read_path = _norm_path(positional[0]) if positional else ""
    out = (verb, read_path, src, dst, redirect, is_move)
    with _LOCK:
        if len(_ROLE_CACHE) >= _ROLE_CAP:
            _ROLE_CACHE.clear()
        _ROLE_CACHE[cmd] = out
    return out


def _static_row(cmd):
    hit = _ROW_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, read_path, src, dst, redirect, is_move = command_roles(cmd)
    row = np.zeros(D, dtype=np.float32)
    mentions = []
    for p in (read_path, src, dst):
        if p and p not in mentions:
            mentions.append(p)
    if mentions:
        acc = _men(mentions[0]).copy()
        for p in mentions[1:]:
            acc += _men(p)
        scale = float(np.sqrt(MEN_W)) / max(1e-6, float(np.linalg.norm(acc)))
        row[MEN_LO:MEN_HI] = MEN_GAIN * acc * scale
    else:
        row[MEN_LO:MEN_HI] = ABSENT_MEN
    row[READ_LO:READ_HI] = ADDR_GAIN * _addr(read_path) if read_path else ABSENT_ADDR
    row[SRC_LO:SRC_HI] = ADDR_GAIN * _addr(src) if src else ABSENT_ADDR
    row[DST_LO:DST_HI] = ADDR_GAIN * _addr(dst) if dst else ABSENT_ADDR
    row[ORG_LO:ORG_HI] = ABSENT_ADDR
    row[VERB_LO:VERB_HI] = FORM_GAIN * _signs("verb\x1f" + verb, VERB_W)
    flags = (
        bool(read_path),
        bool(dst),
        is_move,
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
    )
    fv = np.empty(FLAG_W, dtype=np.float32)
    for j in range(FLAG_W):
        fv[j] = FORM_GAIN if flags[j % N_FLAGS] else -FORM_GAIN
    row[FLAG_LO:FLAG_HI] = fv
    with _LOCK:
        if len(_ROW_CACHE) >= _ROW_CAP:
            _ROW_CACHE.clear()
        _ROW_CACHE[cmd] = row
    return row


def chain_origins(cmds):
    alias = {}
    out = []
    for cmd in cmds:
        verb, read_path, src, dst, redirect, is_move = command_roles(cmd)
        if is_move and src and dst:
            root = alias.get(src, src)
            out.append(root)
            alias.pop(src, None)
            alias[dst] = root
        else:
            out.append(None)
            if dst:
                if redirect == ">>":
                    alias.pop(dst, None)
                elif redirect == ">":
                    alias[dst] = (alias.get(src, src) if src else dst)
    return out


def code_cmds(cmds, z_cmd):
    z = z_cmd if torch.is_tensor(z_cmd) else torch.as_tensor(z_cmd)
    out = z.detach().clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    if n == 0:
        return out
    seq = [c if isinstance(c, str) else str(c) for c in (cmds or [])][:n]
    if len(seq) < n:
        seq = seq + [""] * (n - len(seq))
    origins = chain_origins(seq)
    rows = np.empty((n, D), dtype=np.float32)
    for i in range(n):
        rows[i] = _static_row(seq[i])
        root = origins[i]
        if root:
            rows[i, ORG_LO:ORG_HI] = ADDR_GAIN * _addr(root)
    code = torch.from_numpy(rows).to(device=out.device, dtype=out.dtype)
    return out * RAW_KEEP + code


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
        m = min(int(zc.shape[0]), n)
        tok[bi, 0:2 * m:2] = zc[:m].to(tok.dtype)
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
        "cat /tmp/w/cups/etc/ssh/sshd_config.cfg",
        "cat /tmp/w/cups/g62/f298.tmp",
        "mv /tmp/w/cups/etc/ssh/sshd_config.cfg /tmp/w/cups/g62/f298.tmp.1",
        "mv /tmp/w/cups/g62/f298.tmp.1 /tmp/w/cups/var/lib/misc/state.db.2",
        "cat /tmp/w/cups/var/lib/misc/state.db.2",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
