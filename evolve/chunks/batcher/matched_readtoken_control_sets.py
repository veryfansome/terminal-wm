import re

import torch

NAME = "matched_readtoken_control_sets"
DESCRIPTION = (
    "Matched case-control batch composition. Every training sequence is reduced to the exact "
    "destination path of its deepest mv-routed read, and sequences sharing that path form a "
    "stratum (singleton paths fall back to the suffix-stripped path family). A batch is a "
    "marginal-preserving uniform base plus matched sets: a stratum is drawn with probability "
    "proportional to its size and set_size distinct members are taken from it, so the batch "
    "carries groups of rows whose read command string is identical while their answers are "
    "different contents. Within a set, members are alternated between those whose read path "
    "family names the routed content's own origin and those where it names a different file, so "
    "each set is a minimal pair that also refutes name-keying. The matched fraction anneals up "
    "from zero over the opening fraction of training; degrades to uniform when no routed read is "
    "recoverable."
)

_READ_VERBS = ("cat", "head", "tail")
_SUFFIX = re.compile(r"(?:\.\d+)+$")
_MAX_HOPS = 64


def _routed_read(cmds):
    moves = []
    reads = []
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            moves.append((t, p[1], p[2]))
        elif len(p) == 2 and p[0] in _READ_VERBS:
            reads.append((t, p[1]))
    if not moves or not reads:
        return None
    wrote = {}
    for t, a, b in moves:
        wrote.setdefault(b, []).append((t, a))
    best = None
    for t, path in reads:
        cur, ct, depth = path, t, 0
        while depth < _MAX_HOPS:
            prior = [(j, s) for j, s in wrote.get(cur, ()) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            depth += 1
        if depth >= 1 and (best is None or depth >= best[2]):
            best = (path, cur, depth)
    return best


def _strata(fit, k):
    by_exact = {}
    origin_named = {}
    for i, s in enumerate(fit):
        got = _routed_read(s.get("cmds") or [])
        if got is None:
            continue
        read_path, origin, _depth = got
        by_exact.setdefault(read_path, []).append(i)
        origin_named[i] = 1 if _SUFFIX.sub("", read_path) == origin else 0

    groups = []
    demoted = {}
    for key in sorted(by_exact):
        members = sorted(set(by_exact[key]))
        if len(members) >= k:
            groups.append(members)
        else:
            demoted.setdefault(_SUFFIX.sub("", key), []).extend(members)
    for key in sorted(demoted):
        members = sorted(set(demoted[key]))
        if len(members) >= k:
            groups.append(members)
    return groups, origin_named


def make_batcher(fit, bs, seed, matched_frac_max=0.6, ramp_frac=0.3, set_size=2,
                 contrast_origin_named=True):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0:
        return uniform_only()

    k = max(2, int(set_size))
    if bs < k or float(matched_frac_max) <= 0.0:
        return uniform_only()

    groups, origin_named = _strata(fit, k)
    if not groups:
        return uniform_only()

    pools = []
    named_pools = []
    other_pools = []
    sizes = []
    for members in groups:
        pools.append(torch.tensor(members, dtype=torch.long))
        named_pools.append(torch.tensor(
            [i for i in members if origin_named.get(i, 0) == 1], dtype=torch.long))
        other_pools.append(torch.tensor(
            [i for i in members if origin_named.get(i, 0) == 0], dtype=torch.long))
        sizes.append(float(len(members)))
    stratum_w = torch.tensor(sizes, dtype=torch.float32)

    def shuffled(pool):
        if pool.numel() == 0:
            return pool
        return pool[torch.randperm(pool.numel(), generator=g)]

    def draw_set(si):
        take = min(k, int(pools[si].numel()))
        if not contrast_origin_named:
            return shuffled(pools[si])[:take]
        a = shuffled(named_pools[si])
        b = shuffled(other_pools[si])
        picked = []
        ia = 0
        ib = 0
        na = int(a.numel())
        nb = int(b.numel())
        while len(picked) < take and (ia < na or ib < nb):
            if ia < na:
                picked.append(int(a[ia]))
                ia += 1
            if len(picked) >= take:
                break
            if ib < nb:
                picked.append(int(b[ib]))
                ib += 1
        return torch.tensor(picked, dtype=torch.long)

    def next_batch(step, total_steps):
        span = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, max(0.0, float(step) / span))
        n_matched = int(round(bs * float(matched_frac_max) * ramp))
        n_sets = max(0, min(n_matched, bs)) // k
        n_matched = n_sets * k
        parts = []
        if bs - n_matched > 0:
            parts.append(torch.randint(0, n, (bs - n_matched,), generator=g))
        if n_sets > 0:
            si = torch.multinomial(stratum_w, n_sets, replacement=True, generator=g)
            for j in range(n_sets):
                parts.append(draw_set(int(si[j])))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
