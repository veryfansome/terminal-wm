import math
import re

import torch

NAME = "routed_depth_sysblock_batcher"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic file-content tracker "
    "(mv / cp / redirect move or copy content between paths, a bare read returns whatever now "
    "sits at its path) to label every read with the number of hops that routed its content "
    "there. A batch is built in three annealed layers. The first layer is collision groups: "
    "distinct sequences drawn from one bucket keyed by image, normalized routed-read path shape "
    "and hop count, with bucket choice tilted toward deeper hop counts and member choice tilted "
    "toward sequences with more routed reads. The second layer fills part of the remainder from "
    "a handful of image pools resampled per batch, so the sequences that are NOT collision "
    "partners still come from a few systems instead of all of them. The third layer is uniform. "
    "Both annealed fractions ramp from zero over the same fraction of training. Degrades to "
    "image-only blocking when no routed read is recoverable and to uniform sampling when there "
    "is only one image and no bucket has two members."
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
                 depth_beta=0.8, size_pow=0.5, n_block_images=2, block_frac_max=0.75):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n <= 1 or bs <= 0:
        return uniform_batch

    image_groups = {}
    for i, s in enumerate(fit):
        image_groups.setdefault(str(s.get("image", "?")), []).append(i)
    pools = [torch.tensor(image_groups[k], dtype=torch.long) for k in sorted(image_groups)]
    pool_sizes = torch.tensor([float(p.numel()) for p in pools])
    k_imgs = max(1, min(int(n_block_images), len(pools)))
    block_on = len(pools) >= 2 and float(block_frac_max) > 0.0

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

    collide_on = bool(buckets) and float(hard_frac_max) > 0.0 and int(group_size) >= 2
    if not collide_on and not block_on:
        return uniform_batch

    bucket_w = torch.tensor(weights, dtype=torch.float) if buckets else None
    gsize = max(2, int(group_size))
    distinct = n >= bs
    max_groups = 4 * (bs // gsize + 1)

    def next_batch(step, total_steps):
        ramp = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        anneal = min(1.0, float(step) / float(ramp))

        out = []
        used = set()

        if collide_on:
            n_hard = int(round(bs * float(hard_frac_max) * anneal))
            n_hard = max(0, min(bs, n_hard))
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

        if block_on and len(out) < bs:
            remainder = bs - len(out)
            n_block = int(round(remainder * float(block_frac_max) * anneal))
            n_block = max(0, min(remainder, n_block))
            if n_block > 0:
                chosen = torch.multinomial(pool_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([pools[int(c)] for c in chosen])
                draws = pool[torch.randint(0, int(pool.numel()), (3 * n_block,), generator=g)]
                added = 0
                for v in draws.tolist():
                    if added >= n_block or len(out) >= bs:
                        break
                    if distinct and v in used:
                        continue
                    out.append(int(v))
                    used.add(int(v))
                    added += 1

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
