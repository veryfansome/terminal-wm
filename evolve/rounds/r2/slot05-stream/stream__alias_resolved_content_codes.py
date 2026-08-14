import hashlib
import posixpath

import torch

from realenv import seq_worldmodel as M

NAME = "alias_resolved_content_codes"
DESCRIPTION = (
    "The interleave-2 layout with its geometry, observation tokens and even-index command "
    "extraction unchanged, whose COMMAND tokens additionally carry a 128-dimension content-identity "
    "code in their last coordinates. A prefix-causal symbolic table over the sequence's command "
    "strings tracks, for every filesystem path, the identity of the content currently sitting "
    "there (mv/cp/rm transport, > and >> redirection, directory invalidation, cwd from cd, an "
    "epoch bump on shell forms it cannot parse), and each step's code is a deterministic hash of "
    "that step's command text with every path argument replaced by the identity it resolves to, "
    "so two commands that read the same content carry the same code. The coding is exposed as "
    "code_cmds and collate applies exactly that function."
)

CUPS_LAYOUT = "interleave2"

D = M.D
CODE_DIMS = 128
CODE_SCALE = 1.0
_CODE_START = D - CODE_DIMS
_CACHE_MAX = 20000

_SEQ_CACHE = {}

_REDIR_TOKENS = {
    ">": ">", "1>": ">", "&>": ">",
    ">>": ">>", "1>>": ">>", "&>>": ">>",
    "2>": "2>", "2>>": "2>",
}
_OPAQUE_TOKENS = {"|", "||", "&&", ";", "&", "<", "<<"}
_OPAQUE_CHARS = ("*", "?", "$", "`", "[", "]", "{", "}", "~", "'", '"')
_READ_SRC_VERBS = {"cat", "head", "tail", "tac", "nl", "base64", "od", "xxd", "strings"}
_UNLINK_VERBS = {"rm", "unlink", "shred"}
_MUT_VERBS = {
    "mv", "cp", "rm", "ln", "tee", "touch", "truncate", "dd", "sed", "mkdir", "rmdir",
    "chmod", "chown", "install", "shred", "unlink", "gzip", "gunzip", "tar", "patch",
}


def _h(s):
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=12).hexdigest()


def _vec(key):
    vals = []
    blk = 0
    while len(vals) < CODE_DIMS:
        digest = hashlib.blake2b(
            ("%d\x1f%s" % (blk, key)).encode("utf-8", "surrogatepass"), digest_size=64
        ).digest()
        for j in range(0, 64, 2):
            vals.append((digest[j] << 8) | digest[j + 1])
        blk += 1
    v = torch.tensor(vals[:CODE_DIMS], dtype=torch.float32) / 65535.0 - 0.5
    v = v - v.mean()
    return v / v.pow(2).mean().clamp_min(1e-12).sqrt()


def _is_path(t):
    if not t or t.startswith("-"):
        return False
    return "/" in t or t in (".", "..")


def _norm(p, cwd):
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    p = posixpath.normpath(p)
    return p if p else "/"


def _resolve(state, abs_path):
    return state["loc"].get(abs_path, "p:%d:%s" % (state["epoch"], abs_path))


def _bump_dir(state, abs_path, tag):
    d = posixpath.dirname(abs_path) or "/"
    prev = _resolve(state, d)
    state["loc"][d] = "d:" + _h(prev + "\x1f" + tag + "\x1f" + abs_path)


def _split_redirs(toks):
    base = []
    redirs = []
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t in _REDIR_TOKENS:
            if i + 1 < n:
                redirs.append((_REDIR_TOKENS[t], toks[i + 1]))
                i += 2
                continue
            i += 1
            continue
        if t.startswith(">>") and len(t) > 2:
            redirs.append((">>", t[2:]))
            i += 1
            continue
        if t.startswith(">") and len(t) > 1 and not t.startswith(">&"):
            redirs.append((">", t[1:]))
            i += 1
            continue
        base.append(t)
        i += 1
    return base, redirs


def _step_key(state, cmd):
    cwd = state["cwd"]
    toks = cmd.split()
    if not toks:
        return "empty\x1fcwd=" + cwd

    opaque = False
    for t in toks:
        if t in _OPAQUE_TOKENS:
            opaque = True
            break
        for ch in _OPAQUE_CHARS:
            if ch in t:
                opaque = True
                break
        if opaque:
            break

    base, redirs = _split_redirs(toks)

    parts = ["cwd=" + cwd]
    for t in base:
        parts.append(_resolve(state, _norm(t, cwd)) if _is_path(t) else t)
    for mode, tgt in redirs:
        parts.append(mode)
        parts.append(_resolve(state, _norm(tgt, cwd)) if _is_path(tgt) else tgt)
    key = "\x1f".join(parts)

    loc = state["loc"]
    verb = posixpath.basename(base[0]) if base else ""

    if opaque:
        if redirs or verb in _MUT_VERBS:
            state["epoch"] += 1
            loc.clear()
        return key

    if redirs:
        src_id = None
        if verb in _READ_SRC_VERBS:
            args = [t for t in base[1:] if not t.startswith("-")]
            if len(args) == 1 and _is_path(args[0]) and len(args) == len(base) - 1:
                src_id = _resolve(state, _norm(args[0], cwd))
        for mode, tgt in redirs:
            if not _is_path(tgt):
                continue
            target = _norm(tgt, cwd)
            if mode == "2>":
                loc[target] = "e:" + _h(key + "\x1f" + target)
            elif mode == ">":
                loc[target] = src_id if src_id is not None else "w:" + _h(key + "\x1f" + target)
            else:
                incoming = src_id if src_id is not None else "w:" + _h(key + "\x1f" + target)
                if target in loc:
                    loc[target] = "c:" + _h(loc[target] + "\x1f+\x1f" + incoming)
                else:
                    loc[target] = incoming
            _bump_dir(state, target, "write")
        return key

    if verb == "cd":
        args = [t for t in base[1:] if not t.startswith("-")]
        if len(args) == 1 and args[0] not in ("-",):
            state["cwd"] = _norm(args[0], cwd)
        return key

    if verb in ("mv", "cp"):
        args = [t for t in base[1:] if not t.startswith("-")]
        if len(args) == 2 and _is_path(args[0]) and _is_path(args[1]):
            src = _norm(args[0], cwd)
            dst = _norm(args[1], cwd)
            loc[dst] = _resolve(state, src)
            if verb == "mv":
                loc[src] = "x:%d:%s" % (state["epoch"], src)
                _bump_dir(state, src, "unlink")
            _bump_dir(state, dst, "link")
            return key

    if verb in _UNLINK_VERBS:
        for t in base[1:]:
            if _is_path(t):
                p = _norm(t, cwd)
                loc[p] = "x:%d:%s" % (state["epoch"], p)
                _bump_dir(state, p, "unlink")
        return key

    if verb in _MUT_VERBS:
        for t in base[1:]:
            if _is_path(t):
                p = _norm(t, cwd)
                loc[p] = "w:" + _h(key + "\x1f" + p)
                _bump_dir(state, p, "write")
        return key

    return key


def _sequence_codes(cmds):
    cache_key = tuple(cmds)
    hit = _SEQ_CACHE.get(cache_key)
    if hit is not None:
        return hit
    state = {"loc": {}, "cwd": "/", "epoch": 0}
    rows = [_vec(_step_key(state, c if isinstance(c, str) else str(c))) for c in cmds]
    out = torch.stack(rows) if rows else torch.zeros(0, CODE_DIMS)
    if len(_SEQ_CACHE) >= _CACHE_MAX:
        _SEQ_CACHE.clear()
    _SEQ_CACHE[cache_key] = out
    return out


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    if cmds is None or len(cmds) == 0 or out.shape[0] == 0:
        return out
    codes = _sequence_codes(list(cmds))
    m = min(out.shape[0], codes.shape[0])
    if m > 0:
        add = codes[:m].to(device=out.device, dtype=out.dtype)
        out[:m, _CODE_START:] = out[:m, _CODE_START:] + CODE_SCALE * add
    return out


def _coded_batch(batch):
    coded = []
    for s in batch:
        cmds = s.get("cmds")
        z_cmd = s["z_cmd"]
        if cmds is None or len(cmds) != z_cmd.shape[0]:
            coded.append(s)
            continue
        s2 = dict(s)
        s2["z_cmd"] = code_cmds(cmds, z_cmd)
        coded.append(s2)
    return coded


def collate(batch, device):
    return M.collate(_coded_batch(batch), device)


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
            "cmds": ["cat /a/f0", "mv /a/f0 /b/f1", "cat /b/f1", "ls /a", "cat /a/f0",
                     "cat /b/f1"],
            "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
