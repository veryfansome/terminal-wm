import hashlib
import posixpath
import shlex

import torch

from realenv import seq_worldmodel as M

D = M.D
CODE_M = 48
FLAG_N = 16
CODE_SPAN = 2 * CODE_M + FLAG_N
SRC_OFF = D - CODE_SPAN
DST_OFF = SRC_OFF + CODE_M
FLAG_OFF = DST_OFF + CODE_M
DROP_P = 0.30

NAME = "r23_role_slot_transport_channels"
DESCRIPTION = (
    "interleave2 layout with the cmd token's tail 112 dims overwritten by "
    "[src-slot hash code (48) | dst-slot hash code (48) | role flags (16)] parsed from the raw "
    "command string; obs tokens, tgt and cmd_mask untouched. Also emits causal per-step chain "
    "annotations (slot_id / step_kind / chain_src / chain_depth) from a prefix-only simulation "
    "of the move chain. Codes are withheld on a fraction of training rows so the model keeps a "
    "code-free fallback."
)

CUPS_LAYOUT = "interleave2"

_GEN = torch.Generator()
_GEN.manual_seed(20260809)

_MUT = {"mv", "rename", "move"}
_CP = {"cp", "ln", "install", "rsync", "cpio"}
_RM = {"rm", "unlink", "rmdir", "shred"}
_READ = {
    "cat", "ls", "head", "tail", "less", "more", "stat", "file", "wc", "od", "xxd", "strings",
    "md5sum", "sha1sum", "sha256sum", "readlink", "realpath", "du", "tree", "nl", "tac", "sort",
    "uniq", "cut", "dirname", "basename", "getfacl", "lsattr",
}
_PATLAST = {"grep", "egrep", "fgrep", "zgrep", "sed", "awk", "diff", "cmp"}
_PREFIX_SKIP = {"sudo", "time", "nohup", "env", "command", "exec", "doas", "stdbuf"}

_CODE_CACHE = {}
_ANN_CACHE = {}

KIND_NONE = 0
KIND_READ = 1
KIND_MOVE = 2
KIND_COPY = 3
KIND_REMOVE = 4


def _code(s):
    v = _CODE_CACHE.get(s)
    if v is None:
        hb = hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).digest()
        g = torch.Generator()
        g.manual_seed(int.from_bytes(hb, "big") % (2 ** 62))
        v = torch.randint(0, 2, (CODE_M,), generator=g, dtype=torch.float32) * 2.0 - 1.0
        _CODE_CACHE[s] = v
    return v


def _tokenize(cmd):
    try:
        toks = shlex.split(cmd)
    except Exception:
        toks = cmd.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _PREFIX_SKIP or ("=" in t and not t.startswith("-") and "/" not in t.split("=")[0]):
            i += 1
            continue
        break
    return toks[i:]


def _operands(toks):
    ops = []
    seen_ddash = False
    for t in toks[1:]:
        if not seen_ddash and t == "--":
            seen_ddash = True
            continue
        if not seen_ddash and len(t) > 1 and t.startswith("-"):
            continue
        ops.append(t)
    return ops


def _norm(p):
    if p is None:
        return None
    p = p.strip()
    if not p:
        return None
    try:
        return posixpath.normpath(p)
    except Exception:
        return p


def _resolve_dst(raw, src):
    if raw is None:
        return None, False
    raw = raw.strip()
    if not raw:
        return None, False
    dirlike = raw.endswith("/") or raw in (".", "..") or raw.endswith("/.") or raw.endswith("/..")
    if dirlike and src:
        return _norm(posixpath.join(raw, posixpath.basename(src))), True
    return _norm(raw), False


def _roles(cmd):
    toks = _tokenize(cmd or "")
    if not toks:
        return KIND_NONE, None, None, False, 0
    verb = posixpath.basename(toks[0]).lower()
    ops = _operands(toks)
    n_ops = len(ops)
    dirlike = False
    if verb in _MUT or verb in _CP:
        if n_ops >= 2:
            src = _norm(ops[0])
            dst, dirlike = _resolve_dst(ops[-1], src)
            kind = KIND_MOVE if verb in _MUT else KIND_COPY
        elif n_ops == 1:
            src = _norm(ops[0])
            dst = src
            kind = KIND_NONE
        else:
            src = dst = None
            kind = KIND_NONE
    elif verb in _RM:
        src = _norm(ops[0]) if n_ops else None
        dst = None
        kind = KIND_REMOVE if src else KIND_NONE
    else:
        if verb in _PATLAST and n_ops >= 2:
            src = _norm(ops[-1])
        elif n_ops:
            src = _norm(ops[0])
        else:
            src = None
        dst = src
        kind = KIND_READ if (verb in _READ and src) else KIND_NONE
    return kind, src, dst, dirlike, n_ops


def _flag_vec(kind, src, dst, dirlike, n_ops):
    f = [
        1.0 if kind in (KIND_MOVE, KIND_REMOVE) else -1.0,
        1.0 if kind == KIND_COPY else -1.0,
        1.0 if src is not None else -1.0,
        1.0 if (dst is not None and dst != src) else -1.0,
        1.0 if kind == KIND_READ else -1.0,
        1.0 if dirlike else -1.0,
        1.0 if n_ops >= 2 else -1.0,
        1.0 if (src is not None and src.startswith("/")) else -1.0,
    ]
    return torch.tensor(f + f, dtype=torch.float32)


def _annotate(cmds):
    key = tuple(cmds)
    hit = _ANN_CACHE.get(key)
    if hit is not None:
        return hit
    n = len(cmds)
    src_c = torch.zeros(n, CODE_M)
    dst_c = torch.zeros(n, CODE_M)
    flags = torch.zeros(n, FLAG_N)
    slot_id = torch.full((n,), -1, dtype=torch.long)
    kinds = torch.zeros(n, dtype=torch.long)
    chain_src = torch.full((n,), -1, dtype=torch.long)
    chain_depth = torch.zeros(n, dtype=torch.long)
    owner = {}
    slots = {}
    for t in range(n):
        kind, src, dst, dirlike, n_ops = _roles(cmds[t])
        kinds[t] = kind
        flags[t] = _flag_vec(kind, src, dst, dirlike, n_ops)
        if src is not None:
            src_c[t] = _code(src)
            slot_id[t] = slots.setdefault(src, len(slots))
        if dst is not None:
            dst_c[t] = _code(dst)
        if src is not None:
            o = owner.get(src)
            if o is not None:
                chain_src[t] = o[0]
                chain_depth[t] = o[1]
        if kind == KIND_MOVE:
            o = owner.pop(src, None) if src is not None else None
            if dst is not None:
                if o is not None:
                    owner[dst] = (o[0], o[1] + 1)
                else:
                    owner.pop(dst, None)
        elif kind == KIND_COPY:
            o = owner.get(src) if src is not None else None
            if dst is not None:
                if o is not None:
                    owner[dst] = (o[0], o[1] + 1)
                else:
                    owner.pop(dst, None)
        elif kind == KIND_REMOVE:
            if src is not None:
                owner.pop(src, None)
        elif kind == KIND_READ:
            if src is not None:
                owner[src] = (t, 0)
    out = (src_c, dst_c, flags, slot_id, kinds, chain_src, chain_depth)
    if len(_ANN_CACHE) < 60000:
        _ANN_CACHE[key] = out
    return out


def collate(batch, device):
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    slot_id = torch.full((B, maxn), -1, dtype=torch.long)
    step_kind = torch.zeros(B, maxn, dtype=torch.long)
    chain_src = torch.full((B, maxn), -1, dtype=torch.long)
    chain_depth = torch.zeros(B, maxn, dtype=torch.long)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])

    # Codes are dropped only while gradients are live: every eval/probe path runs under no_grad,
    # so scoring always sees the full-code stream.
    if torch.is_grad_enabled() and DROP_P > 0.0:
        drop = torch.rand(B, generator=_GEN) < DROP_P
    else:
        drop = torch.zeros(B, dtype=torch.bool)

    for bi, s in enumerate(batch):
        n = int(s["z_obs"].shape[0])
        tok[bi, 0:2 * n:2] = s["z_cmd"][:n]
        tok[bi, 1:2 * n:2] = s["z_obs"][:n]
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, 0:2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n]
        cmd_mask[bi, :n] = True
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
        raw = s.get("cmds") or []
        cmds = [str(c) for c in raw[:n]]
        if len(cmds) < n:
            cmds = cmds + [""] * (n - len(cmds))
        sc, dc, fl, sid, kd, cs, cd = _annotate(cmds)
        slot_id[bi, :n] = sid
        step_kind[bi, :n] = kd
        chain_src[bi, :n] = cs
        chain_depth[bi, :n] = cd
        if not bool(drop[bi]):
            tok[bi, 0:2 * n:2, SRC_OFF:DST_OFF] = sc
            tok[bi, 0:2 * n:2, DST_OFF:FLAG_OFF] = dc
            tok[bi, 0:2 * n:2, FLAG_OFF:D] = fl

    out = {
        "tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
        "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device),
        "slot_id": slot_id.to(device), "step_kind": step_kind.to(device),
        "chain_src": chain_src.to(device), "chain_depth": chain_depth.to(device),
    }
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
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())
