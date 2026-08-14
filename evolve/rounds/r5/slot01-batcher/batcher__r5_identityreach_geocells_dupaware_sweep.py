import re

import torch

NAME = "r5_identityreach_geocells_dupaware_sweep"
DESCRIPTION = (
    "Resolves each training sequence's content routing by walking every read command backwards "
    "through the latest earlier write to its path -- 'mv SRC DST', 'cp SRC DST' and "
    "'cat SRC > DST' carry one content, while 'cat SRC >> DST' terminates the walk because an "
    "accumulator holds no single content -- which yields, for the sequence's deepest "
    "identity-carrying read, the move-hop count, the copy-hop count, the read path and the path "
    "the content started at. Sequences are grouped into cells by two cascades of keys, coarsened "
    "one level at a time so that whatever is left over from a level too small to be a cell is "
    "regrouped by the next key: one cascade is (image, digit-collapsed read path, move hops, copy "
    "hops) then (image, read path) then image, the other is cross-image (digit-collapsed origin "
    "path, read path, move hops, copy hops) then (origin path, read path) then read path. "
    "Separately, every step's observation embedding is quantized on a few fixed coordinates into "
    "an exact duplicate class, and the classes occurring in at least two but at most a fraction "
    "of the sequences become a per-sequence signature of avoidable duplicate observations. A "
    "batch is an annealed blocked part filled group_size at a time out of cells, plus a global "
    "part; a candidate is skipped when it repeats an index already in the batch, or when it would "
    "add more than a tolerance of duplicate observation classes the batch already holds, with the "
    "tolerance relaxed on each retry round and dropped entirely on the last. Every draw over "
    "cells and over sequences is a quasi-random additive recurrence on the normalized "
    "cumulative-weight axis instead of multinomial sampling with replacement: successive picks "
    "step by the golden-ratio conjugate modulo one, so consecutive picks land far apart while the "
    "count each item accumulates tracks its weight share with low discrepancy, and the ordering "
    "of the axis and the offset of the recurrence are redrawn once per sweep of the pool. Members "
    "inside a cell come from a reshuffling deck. Weights are exp(depth_beta * ramp * z) on the "
    "standardized routing score for sequences and size^size_pow times the cell's mean tilt for "
    "cells, annealed from flat over the opening fraction of training."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
_READ_VERBS = ("cat", "head", "tail")
_MAX_HOPS = 32
_HASH_COORDS = (7, 61, 137, 271, 439, 613)
_HASH_SCALE = 2048.0
_PHI = 0.6180339887498949


def _norm(path):
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


def _identity_reads(cmds):
    writes = {}
    reads = []
    for t, cmd in enumerate(cmds):
        ev = _parse(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "read":
            reads.append((t, a))
        elif b:
            writes.setdefault(b, []).append((t, kind, a))
    out = []
    for t, path in reads:
        cur = path
        ct = t
        mvh = 0
        cph = 0
        pure = True
        for _ in range(_MAX_HOPS):
            prior = None
            for e in writes.get(cur, ()):
                if e[0] < ct and (prior is None or e[0] > prior[0]):
                    prior = e
            if prior is None:
                break
            if prior[1] == "append":
                pure = False
                break
            ct = prior[0]
            cur = prior[2]
            if prior[1] == "move":
                mvh += 1
            else:
                cph += 1
        if pure and (mvh + cph) > 0:
            out.append((path, cur, mvh, cph))
    return out


def _scan(fit, copy_w, depth_cap):
    cap = max(1, int(depth_cap))
    deepest = []
    demand = []
    img_levels = []
    geo_levels = []
    for s in fit:
        img = str(s.get("image", "?"))
        best = None
        tot = 0.0
        for path, origin, mvh, cph in _identity_reads(s.get("cmds") or []):
            mv = min(int(mvh), cap)
            cp = min(int(cph), cap)
            d = float(mv) + float(copy_w) * float(cp)
            tot += d
            if best is None or d > best[0]:
                best = (d, _norm(path), _norm(origin), mv, cp)
        if best is None:
            deepest.append(0.0)
            demand.append(0.0)
            img_levels.append(None)
            geo_levels.append(None)
        else:
            deepest.append(best[0])
            demand.append(tot)
            rd = best[1]
            org = best[2]
            img_levels.append([(img, rd, best[3], best[4]), (img, rd), (img,)])
            geo_levels.append([(org, rd, best[3], best[4]), (org, rd), (rd,)])
    return deepest, demand, img_levels, geo_levels


def _obs_classes(fit, avoid_frac):
    counts = {}
    per = []
    for s in fit:
        zo = s.get("z_obs")
        if zo is None or zo.dim() != 2 or int(zo.shape[0]) == 0:
            per.append(set())
            continue
        cols = [c for c in _HASH_COORDS if c < int(zo.shape[1])]
        if not cols:
            per.append(set())
            continue
        q = torch.round(zo[:, cols].detach().to(torch.float64) * _HASH_SCALE)
        ids = set()
        for row in q.to(torch.int64).tolist():
            ids.add(tuple(row))
        per.append(ids)
        for h in ids:
            counts[h] = counts.get(h, 0) + 1
    limit = max(2, int(float(avoid_frac) * max(1, len(fit))))
    keep = {}
    for h, c in counts.items():
        if 2 <= c <= limit:
            keep[h] = len(keep)
    return [tuple(keep[h] for h in ids if h in keep) for ids in per]


def _cells_cascade(levels, min_cell):
    floor = max(2, int(min_cell))
    alive = [i for i, lv in enumerate(levels) if lv is not None]
    if not alive:
        return []
    depth = len(levels[alive[0]])
    out = []
    for d in range(depth):
        groups = {}
        for i in alive:
            groups.setdefault(levels[i][d], []).append(i)
        rest = []
        for k in sorted(groups):
            v = groups[k]
            if len(v) >= floor:
                out.append(torch.tensor(v, dtype=torch.long))
            else:
                rest.extend(v)
        alive = rest
        if not alive:
            break
    return out


class _WeightedSweep:
    def __init__(self, size, gen):
        self.gen = gen
        self.size = max(1, int(size))
        self.w = torch.ones(self.size, dtype=torch.float64)
        self.order = torch.randperm(self.size, generator=gen)
        self.shift = float(torch.rand((1,), generator=gen).item())
        self.tick = 0
        self.cum = torch.cumsum(self.w, 0)
        self.total = float(self.cum[-1])

    def reweight(self, w):
        v = w.to(torch.float64)
        v = torch.nan_to_num(v, nan=1.0, posinf=1.0, neginf=1e-9).clamp(1e-9, 1e9)
        self.w = v
        self._rebuild()

    def _rebuild(self):
        self.cum = torch.cumsum(self.w[self.order], 0)
        self.total = float(self.cum[-1])

    def _resweep(self):
        self.order = torch.randperm(self.size, generator=self.gen)
        self.shift = float(torch.rand((1,), generator=self.gen).item())
        self.tick = 0
        self._rebuild()

    def draw(self, m):
        k = int(m)
        if k <= 0 or self.total <= 0.0:
            return []
        base = torch.arange(k, dtype=torch.float64) + float(self.tick)
        frac = torch.remainder(base * _PHI + self.shift, 1.0)
        j = torch.searchsorted(self.cum, frac * self.total, right=False)
        j = j.clamp(max=int(self.cum.numel()) - 1)
        out = self.order[j].tolist()
        self.tick += k
        if self.tick >= self.size:
            self._resweep()
        return out


class _CellDeck:
    def __init__(self, cells, gen):
        self.cells = cells
        self.gen = gen
        self.deck = [None] * len(cells)
        self.cur = [0] * len(cells)

    def deal(self, ci, k):
        pool = self.cells[ci]
        m = int(pool.numel())
        want = min(int(k), m)
        if m == 0 or want <= 0:
            return []
        if self.deck[ci] is None or self.cur[ci] + want > m:
            self.deck[ci] = pool[torch.randperm(m, generator=self.gen)].tolist()
            self.cur[ci] = 0
        out = self.deck[ci][self.cur[ci]:self.cur[ci] + want]
        self.cur[ci] += want
        return out


def make_batcher(fit, bs, seed, hard_frac_max=0.65, ramp_frac=0.3, group_size=4,
                 geo_share=0.45, depth_beta=0.9, size_pow=0.5, copy_w=0.5,
                 demand_share=0.25, depth_cap=4, dup_tol=3, dup_frac=0.4,
                 avoid_frac=0.25, min_cell=3, ramp_buckets=32):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    if n < 2 or bs <= 0:
        def uniform_batch(step, total_steps):
            return torch.randint(0, max(1, n), (bs,), generator=g).tolist()
        return uniform_batch

    deepest, demand, img_levels, geo_levels = _scan(fit, copy_w, depth_cap)
    raw = torch.tensor([deepest[i] + float(demand_share) * demand[i] for i in range(n)],
                       dtype=torch.float64)
    ctr = raw - raw.mean()
    z = (ctr / ctr.pow(2).mean().sqrt().clamp_min(1e-6)).clamp(-4.0, 4.0)

    sig = _obs_classes(fit, avoid_frac)

    channels = []
    for levels, share in ((img_levels, max(0.0, 1.0 - float(geo_share))),
                          (geo_levels, max(0.0, float(geo_share)))):
        if share <= 0.0:
            continue
        cells = _cells_cascade(levels, min_cell)
        if not cells:
            continue
        sizes = torch.tensor([float(c.numel()) for c in cells], dtype=torch.float64)
        mz = torch.tensor([float(z[c].mean().item()) for c in cells], dtype=torch.float64)
        channels.append({"sizes": sizes, "mz": mz, "share": float(share), "cut": 1.0,
                         "sweep": _WeightedSweep(len(cells), g),
                         "deck": _CellDeck(cells, g)})
    if channels:
        tot = sum(c["share"] for c in channels)
        acc = 0.0
        for c in channels:
            acc += c["share"] / tot
            c["cut"] = acc
        channels[-1]["cut"] = 1.0

    gsweep = _WeightedSweep(n, g)
    state = {"bucket": -1}
    distinct = n >= bs
    base_tol = int(dup_tol)
    gsize = max(2, int(group_size))
    nbk = max(1, int(ramp_buckets))

    def reweight(ramp):
        e = float(depth_beta) * float(ramp)
        gsweep.reweight(torch.exp(e * z))
        for c in channels:
            c["sweep"].reweight(c["sizes"].pow(float(size_pow)) * torch.exp(e * c["mz"]))

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / float(ramp_steps))
        bucket = min(nbk - 1, int(ramp * nbk))
        if bucket != state["bucket"]:
            state["bucket"] = bucket
            reweight(ramp)

        picks = []
        used = set()
        seen = set()

        def offer(i, tol):
            if distinct and i in used:
                return
            s = sig[i]
            if tol >= 0 and s:
                budget = max(int(tol), int(float(dup_frac) * len(s)))
                coll = 0
                for c in s:
                    if c in seen:
                        coll += 1
                        if coll > budget:
                            return
            picks.append(i)
            used.add(i)
            if s:
                seen.update(s)

        n_hard = 0
        if channels and float(hard_frac_max) > 0.0:
            n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))

        tries = 0
        cap = 6 * (n_hard // gsize + 2)
        while len(picks) < n_hard and tries < cap:
            tries += 1
            u = float(torch.rand((1,), generator=g).item())
            ch = channels[-1]
            for c in channels:
                if u <= c["cut"]:
                    ch = c
                    break
            ci = ch["sweep"].draw(1)
            if not ci:
                break
            need = min(gsize, n_hard - len(picks))
            before = len(picks)
            for i in ch["deck"].deal(int(ci[0]), 3 * need):
                if len(picks) - before >= need or len(picks) >= n_hard:
                    break
                offer(int(i), base_tol)

        rounds = 0
        while len(picks) < bs and rounds < 8:
            need = bs - len(picks)
            for i in gsweep.draw(min(4 * need + 8, 4 * bs + 8)):
                if len(picks) >= bs:
                    break
                offer(int(i), base_tol + rounds)
            rounds += 1
        while len(picks) < bs:
            for i in gsweep.draw(bs):
                if len(picks) >= bs:
                    break
                offer(int(i), -1)

        out = picks[:bs]
        order = torch.randperm(bs, generator=g).tolist()
        return [out[j] for j in order]

    return next_batch
