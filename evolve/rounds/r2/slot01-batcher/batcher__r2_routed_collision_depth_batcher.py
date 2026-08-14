import math
import re

import torch

NAME = "r2_routed_collision_depth_batcher"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic file-content tracker "
    "(mv / cp / redirect move or copy content between paths, a bare read returns whatever now "
    "sits at its path) to label every read with the number of hops that routed its content "
    "there. Batches are a uniform base plus an annealed share of collision groups: distinct "
    "sequences drawn from one bucket keyed by image, normalized routed-read path shape and hop "
    "count, with bucket choice tilted toward deeper hop counts and member choice tilted toward "
    "sequences with more routed reads. Degrades to image-only blocking when no routed read is "
    "recoverable and to uniform sampling when no bucket has two members."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
_DEPTH_CAP = 4
_READ_VERBS = ("cat", "head", "tail")


def _norm_path(path):
    return _NUM.sub("#", _HEX.sub("#", path))


def _parse(cmd):
    toks = cmd.split()
    if not toks:
        return None
    verb = toks[0]
    args = []
    redir = None
    rtarget = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in ("|", "&&", "||", ";", "2>", "2>>"):
            return None
        if t in (">", ">>"):
            redir = t
            if i + 1 < len(toks):
                rtarget = toks[i + 1]
            i += 2
            continue
        if t.startswith(">>"):
            redir = ">>"
            rtarget = t[2:] or None
            i += 1
            continue
        if t.startswith(">"):
            redir = ">"
            rtarget = t[1:] or None
            i += 1
            continue
        if t.startswith("-"):
            i += 1
            continue
        args.append(t)
        i += 1
    if verb == "mv" and redir is None and len(args) == 2:
        return ("move", args[0], args[1])
    if verb == "cp" and redir is None and len(args) == 2:
        return ("copy", args[0], args[1])
    if verb in _READ_VERBS and len(args) == 1:
        if redir == ">" and rtarget:
            return ("copy", args[0], rtarget)
        if redir == ">>" and rtarget:
            return ("append", args[0], rtarget)
        if redir is None:
            return ("read", args[0], None)
    return None


def _routed_reads(cmds):
    hops = {}
    out = []
    for cmd in cmds:
        ev = _parse(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "move":
            hops[b] = hops.pop(a, 0) + 1
        elif kind == "copy":
            hops[b] = hops.get(a, 0) + 1
        elif kind == "append":
            hops[b] = max(hops.get(b, 0), hops.get(a, 0) + 1)
        else:
            d = hops.get(a, 0)
            if d >= 1:
                out.append((a, min(d, _DEPTH_CAP)))
    return out


def make_batcher(fit, bs, seed, hard_frac_max=0.5, ramp_frac=0.3, group_size=4,
                 depth_beta=0.8, size_pow=0.5):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n <= 1 or bs <= 0 or hard_frac_max <= 0.0 or int(group_size) < 2:
        return uniform_batch

    demand = [0.0] * n
    keyed = {}
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        img = str(s.get("image", "?"))
        seen = set()
        for path, d in _routed_reads(cmds):
            demand[i] += float(d)
            seen.add((img, _norm_path(path), d))
        for key in seen:
            keyed.setdefault(key, []).append(i)

    if not keyed:
        for i, s in enumerate(fit):
            keyed.setdefault((str(s.get("image", "?")), "", 0), []).append(i)

    buckets = []
    weights = []
    member_w = []
    for key in sorted(keyed):
        members = keyed[key]
        if len(members) < 2:
            continue
        buckets.append(torch.tensor(members, dtype=torch.long))
        weights.append((float(len(members)) ** float(size_pow))
                       * math.exp(float(depth_beta) * float(key[2])))
        member_w.append(torch.tensor([1.0 + demand[j] for j in members], dtype=torch.float))

    if not buckets:
        return uniform_batch

    bucket_w = torch.tensor(weights, dtype=torch.float)
    gsize = int(group_size)
    distinct = n >= bs
    max_groups = 4 * (bs // gsize + 1)

    def next_batch(step, total_steps):
        ramp = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp))
        n_hard = int(round(bs * frac))
        if n_hard > bs:
            n_hard = bs
        if n_hard < 0:
            n_hard = 0

        out = []
        used = set()
        tries = 0
        while len(out) < n_hard and tries < max_groups:
            tries += 1
            bi = int(torch.multinomial(bucket_w, 1, generator=g)[0])
            members = buckets[bi]
            k = min(gsize, int(members.numel()), n_hard - len(out))
            if k <= 0:
                break
            pick = torch.multinomial(member_w[bi], k, replacement=False, generator=g)
            for j in pick.tolist():
                v = int(members[j])
                if distinct and v in used:
                    continue
                out.append(v)
                used.add(v)
                if len(out) >= n_hard:
                    break

        need = bs - len(out)
        if need > 0:
            for v in torch.randint(0, n, (3 * need,), generator=g).tolist():
                if len(out) >= bs:
                    break
                if distinct and v in used:
                    continue
                out.append(v)
                used.add(v)
        while len(out) < bs:
            out.append(int(torch.randint(0, n, (1,), generator=g)[0]))

        order = torch.randperm(bs, generator=g).tolist()
        return [out[i] for i in order]

    return next_batch
