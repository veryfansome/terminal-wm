import hashlib

import torch

D = 768
SRC_DIM = 96
DST_DIM = 96
TAG_DIM = 16
CODE_DIM = SRC_DIM + DST_DIM + TAG_DIM
CODE_GAIN = 1.0
TAG_FEATS = 6
CACHE_MAX = 65536

NAME = "r4_pathslot_dualaddress_code"
DESCRIPTION = (
    "Overwrites three fixed, hash-chosen coordinate slots of every standardized COMMAND "
    "embedding with a purely lexical dual-address code parsed from that command string alone: "
    "a 96-bit Rademacher hash of its source path in the SOURCE slot, the same hash family "
    "applied to its destination path in the DESTINATION slot, and a 16-dim sign pattern of six "
    "structural flags (has-destination, mv, cat, ls, redirection, has-source) in the TAG slot. "
    "Commands with no source or no destination receive fixed NULL codes instead. The same path "
    "string yields the same 96-bit code wherever it appears, so a move's destination slot and a "
    "later command's source slot hold identical vectors exactly when the two paths are equal, "
    "and near-orthogonal vectors otherwise. The coding is a pure per-command function of the "
    "command string, so it is prefix-causal and unchanged when two contents exchange their "
    "moves; the remaining 560 embedding coordinates and all OBSERVATION tokens are untouched, "
    "and the token layout stays the interleaved cmd/obs stream."
)

_REDIR = (">", ">>", "1>", "2>", ">|")

_BYTE_SIGNS = tuple(
    tuple(1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)) for b in range(256)
)


def _stable_order():
    return sorted(range(D), key=lambda i: hashlib.blake2b(
        b"slot:" + str(i).encode("ascii"), digest_size=16).digest())


_ORDER = _stable_order()
_SRC_IDX = list(_ORDER[:SRC_DIM])
_DST_IDX = list(_ORDER[SRC_DIM:SRC_DIM + DST_DIM])
_TAG_IDX = list(_ORDER[SRC_DIM + DST_DIM:SRC_DIM + DST_DIM + TAG_DIM])
_IDX_CACHE = {}


def _slot_index_tensors(device):
    key = str(device)
    got = _IDX_CACHE.get(key)
    if got is None:
        got = (torch.tensor(_SRC_IDX, dtype=torch.long, device=device),
               torch.tensor(_DST_IDX, dtype=torch.long, device=device),
               torch.tensor(_TAG_IDX, dtype=torch.long, device=device))
        _IDX_CACHE[key] = got
    return got


def _sign_bits(key, nbits):
    digest = hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=nbits // 8).digest()
    out = []
    for b in digest:
        out.extend(_BYTE_SIGNS[b])
    return out


_NULL_SRC = _sign_bits("\x00null-source", SRC_DIM)
_NULL_DST = _sign_bits("\x00null-destination", DST_DIM)
_PATH_CACHE = {}
_CMD_CACHE = {}


def _path_code(path):
    got = _PATH_CACHE.get(path)
    if got is None:
        if len(_PATH_CACHE) >= CACHE_MAX:
            _PATH_CACHE.clear()
        got = _sign_bits("path\x1f" + path, SRC_DIM)
        _PATH_CACHE[path] = got
    return got


def parse_command(cmd):
    parts = cmd.split()
    if not parts:
        return "", None, None, -1
    verb = parts[0]
    ri = -1
    for i in range(1, len(parts)):
        if parts[i] in _REDIR:
            ri = i
            break
    if ri >= 0:
        left = [p for p in parts[1:ri] if p.startswith("/")]
        right = [p for p in parts[ri + 1:] if p.startswith("/")]
        return verb, (left[0] if left else None), (right[0] if right else None), ri
    paths = [p for p in parts[1:] if p.startswith("/")]
    if verb == "mv" and len(paths) >= 2:
        return verb, paths[0], paths[1], ri
    return verb, (paths[0] if paths else None), None, ri


def _command_code(cmd):
    verb, src, dst, ri = parse_command(cmd)
    vals = list(_NULL_SRC if src is None else _path_code(src))
    vals.extend(_NULL_DST if dst is None else _path_code(dst))
    feats = (
        1.0 if dst is not None else -1.0,
        1.0 if verb == "mv" else -1.0,
        1.0 if verb == "cat" else -1.0,
        1.0 if verb == "ls" else -1.0,
        1.0 if ri >= 0 else -1.0,
        1.0 if src is not None else -1.0,
    )
    vals.extend(feats[i % TAG_FEATS] for i in range(TAG_DIM))
    return torch.tensor(vals, dtype=torch.float32)


def _cached_command_code(cmd):
    got = _CMD_CACHE.get(cmd)
    if got is None:
        if len(_CMD_CACHE) >= CACHE_MAX:
            _CMD_CACHE.clear()
        got = _command_code(cmd)
        _CMD_CACHE[cmd] = got
    return got


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    n = min(len(cmds), int(out.shape[0]))
    if n <= 0:
        return out
    rowvals = [_cached_command_code(c if isinstance(c, str) else str(c)) for c in cmds[:n]]
    vals = torch.stack(rowvals).to(device=out.device, dtype=out.dtype)
    si, di, ti = _slot_index_tensors(out.device)
    rows = out[:n]
    rows[:, si] = CODE_GAIN * vals[:, :SRC_DIM]
    rows[:, di] = CODE_GAIN * vals[:, SRC_DIM:SRC_DIM + DST_DIM]
    rows[:, ti] = CODE_GAIN * vals[:, SRC_DIM + DST_DIM:]
    return out


def _coded_cmd_matrix(s, n):
    cmds = s.get("cmds") or []
    return code_cmds(list(cmds)[:n], s["z_cmd"][:n])


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
        zc = _coded_cmd_matrix(s, n)
        m = int(zc.shape[0])
        tok[bi, 0:2 * m:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = s["z_obs"][:n].to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n].to(tgt.dtype)
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
    cmds = ["ls -1 /", "cat /tmp/w/cups/g1/a.dat", "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg",
            "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat", "uname -a",
            "cat /tmp/w/cups/etc/b.cfg"]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())


CUPS_LAYOUT = "interleave2"
