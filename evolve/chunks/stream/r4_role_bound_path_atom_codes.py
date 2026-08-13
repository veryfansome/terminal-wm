import hashlib

import torch

from realenv import seq_worldmodel as M

NAME = "r4_role_bound_path_atom_codes"
DESCRIPTION = (
    "Interleave2 layout with a role-bound path-atom code written into the last 128 coordinates "
    "of every COMMAND token. Each command is parsed on its own into a verb/redirect shape and its "
    "non-flag path arguments, each argument is assigned the role src, dst or read (mv/cp take "
    "arg0/arg1; a > or >> target is the dst and the preceding argument the src; every other "
    "argument is a read), and each path contributes two deterministic blake2b-derived +/-1 "
    "Rademacher vectors, one for the full path string and one for its basename. Every atom is "
    "bundled twice: once multiplied elementwise by a fixed +/-1 role vector and once unbound, so "
    "the same path carries both a role-separated channel and a role-free identity channel. The "
    "coding is position-local, so it is prefix-causal and a command's token depends on nothing but "
    "that command's own string and its own standardized embedding. The first 640 embedding "
    "coordinates are passed through unchanged and OBSERVATION tokens are untouched."
)

D = M.D
CODE_DIMS = 128
CODE_OFF = D - CODE_DIMS

FULL_ROLE = 0.75
FULL_FREE = 0.55
BASE_ROLE = 0.22
BASE_FREE = 0.30
SHAPE_ROLE = 0.40

_ROLES = ("shape", "src", "dst", "read")
_SIGN_LIMIT = 60000
_CODE_LIMIT = 150000

_sign_cache = {}
_code_cache = {}


def _sign_vec(key):
    v = _sign_cache.get(key)
    if v is not None:
        return v
    digest = hashlib.blake2b(key.encode("utf-8", "replace"),
                             digest_size=CODE_DIMS // 8).digest()
    v = torch.tensor(
        [1.0 if (digest[i >> 3] >> (i & 7)) & 1 else -1.0 for i in range(CODE_DIMS)],
        dtype=torch.float32)
    if len(_sign_cache) >= _SIGN_LIMIT:
        _sign_cache.clear()
    _sign_cache[key] = v
    return v


def _role_vec(role):
    return _sign_vec("\x00role\x00" + role)


def _spaced(cmd):
    s = cmd.replace(">>", " \x01 ")
    s = s.replace(">", " \x02 ")
    s = s.replace("\x01", ">>").replace("\x02", ">")
    return s.replace("|", " | ")


def _parse(cmd):
    toks = _spaced(cmd).split()
    if not toks:
        return "", [], "", None
    verb = toks[0]
    pos = []
    redir_op = ""
    redir_tgt = None
    want_tgt = False
    for t in toks[1:]:
        if want_tgt:
            redir_tgt = t
            want_tgt = False
            continue
        if t == ">" or t == ">>":
            redir_op = t
            want_tgt = True
            continue
        if t == "|":
            continue
        if t.startswith("-"):
            continue
        pos.append(t)
    return verb, pos, redir_op, redir_tgt


def _basename(path):
    p = path[:-1] if len(path) > 1 and path.endswith("/") else path
    b = p.rsplit("/", 1)[-1]
    if not b or b == path:
        return None
    return b


def _slots(verb, pos, redir_tgt):
    if verb in ("mv", "cp", "ln") and len(pos) >= 2:
        return [(pos[0], "src"), (pos[1], "dst")] + [(p, "read") for p in pos[2:]]
    if redir_tgt is not None:
        out = [(redir_tgt, "dst")]
        if pos:
            out.append((pos[0], "src"))
            out += [(p, "read") for p in pos[1:]]
        return out
    return [(p, "read") for p in pos]


def _build_code(cmd):
    verb, pos, redir_op, redir_tgt = _parse(cmd)
    shape = verb + "|" + redir_op + "|" + str(len(pos)) + "|" + ("1" if redir_tgt else "0")
    vec = _role_vec("shape") * _sign_vec("\x00shape\x00" + shape) * SHAPE_ROLE
    for path, role in _slots(verb, pos, redir_tgt):
        rv = _role_vec(role)
        av = _sign_vec("\x00full\x00" + path)
        vec = vec + rv * av * FULL_ROLE + av * FULL_FREE
        base = _basename(path)
        if base is not None:
            bv = _sign_vec("\x00base\x00" + base)
            vec = vec + rv * bv * BASE_ROLE + bv * BASE_FREE
    return vec


def _cmd_code(cmd):
    key = cmd if isinstance(cmd, str) else str(cmd)
    v = _code_cache.get(key)
    if v is None:
        v = _build_code(key)
        if len(_code_cache) >= _CODE_LIMIT:
            _code_cache.clear()
        _code_cache[key] = v
    return v


def code_cmds(cmds, z_cmd):
    z = z_cmd if torch.is_tensor(z_cmd) else torch.as_tensor(z_cmd)
    out = z.detach().clone()
    n = out.shape[0]
    if n == 0:
        return out
    seq = list(cmds) if cmds is not None else []
    rows = [_cmd_code(seq[i]) if i < len(seq) else _cmd_code("") for i in range(n)]
    code = torch.stack(rows).to(device=out.device, dtype=out.dtype)
    out[:, CODE_OFF:] = code
    return out


def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
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
        n = s["z_obs"].shape[0]
        zc = code_cmds(s.get("cmds", []), s["z_cmd"])
        tok[bi, 0:2 * n:2] = zc[:n]
        tok[bi, 1:2 * n:2] = s["z_obs"][:n]
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, 0:2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n]
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
            n = s["z_obs"].shape[0]
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
    p0 = extract_cmd_pred(net(b0["tok"], b0["types"], b0["key_pad"])[0], b0).clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = extract_cmd_pred(net(b1["tok"], b1["types"], b1["key_pad"])[0], b1).cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())


CUPS_LAYOUT = "interleave2"
