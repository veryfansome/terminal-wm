import torch

NAME = "r7_antithetic_movepos_pairs"
DESCRIPTION = (
    "Antithetic matched-pair batching, imported from antithetic variates in Monte-Carlo "
    "estimation and from matched-pair randomized designs: a nuisance factor whose realized "
    "in-sample correlation is forced to exactly zero can supply no descent direction, whereas a "
    "factor that is merely balanced in the marginal still fluctuates by O(1/sqrt(bs)) inside each "
    "minibatch and is chased by the optimizer. Every training sequence is reduced, from its "
    "command strings alone, to its deepest mv-routed read: a 'cat P' is walked backwards through "
    "the mv edges that wrote P earlier in the same trajectory, yielding the hop count, the origin "
    "location, and the exact set of mv steps that carried the answer. Two bits record WHERE IN THE "
    "MOVE ORDER that content sits — whether the trajectory's first mv before the read belongs to "
    "the answer's own chain, and whether the last one does. Sequences are stratified on everything "
    "that is NOT that signature — hop class, whether the answer ended in the same trailing-suffix "
    "name family it started in, and the kind of location being read (cvar slot, templated system "
    "path, scratch path) — with a nested backoff that drops the last key component whenever the "
    "finer stratification cannot supply enough matched pairs. Inside a stratum only BIT-"
    "COMPLEMENTARY bags are matched: (first,last)=(1,1) against (0,0), and (1,0) against (0,1). A "
    "batch is a uniform iid base part plus pairs, each pair one draw from each side of one cell, so "
    "both position bits are exactly 50/50 in the realized batch rather than only in the marginal, "
    "and the two members of a pair differ only in which content the chain routed. Cell weight is the "
    "matched-pair count min(|A|,|B|). The paired fraction ramps from zero, so training opens on the "
    "plain uniform distribution; falls back to uniform when no matched cell can be built. All draws "
    "come from a private torch.Generator and the sequences are only read."
)

_MAX_HOPS = 64
_N_SIG = 4


def _stem(path):
    p = path
    while True:
        i = p.rfind(".")
        if i <= 0 or p[i - 1] == "/":
            break
        tail = p[i + 1:]
        if not tail or not tail.isdigit():
            break
        p = p[:i]
    return p


def _read_kind(path):
    if "/cvar/" in path:
        return 0
    parts = path.split("/")
    if len(parts) >= 2:
        parent = parts[-2]
        if len(parent) > 1 and parent[0] == "g" and parent[1:].isdigit():
            return 2
    return 1


def _profile(cmds, depth_cap):
    mv_steps = []
    wrote = {}
    for t, c in enumerate(cmds):
        if not isinstance(c, str):
            continue
        p = c.split()
        if len(p) != 3 or p[0] != "mv":
            continue
        if p[1].startswith("-") or p[2].startswith("-"):
            continue
        mv_steps.append(t)
        wrote.setdefault(p[2], []).append((t, p[1]))
    if not mv_steps:
        return None

    best = None
    for t, c in enumerate(cmds):
        if not isinstance(c, str):
            continue
        p = c.split()
        if len(p) != 2 or p[0] != "cat":
            continue
        cur, ct, hops = p[1], t, 0
        carried = []
        while hops < _MAX_HOPS:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            carried.append(ct)
            hops += 1
        if hops >= 1 and (best is None or hops > best[0]):
            best = (hops, t, p[1], cur, carried)
    if best is None:
        return None

    hops, read_t, read_path, origin, carried = best
    before = [j for j in mv_steps if j < read_t]
    if not before:
        return None
    carried_set = set(carried)
    first_bit = 1 if before[0] in carried_set else 0
    last_bit = 1 if before[-1] in carried_set else 0
    full_key = (
        min(int(hops), int(depth_cap)),
        1 if _stem(read_path) == _stem(origin) else 0,
        _read_kind(read_path),
    )
    return full_key, 2 * first_bit + last_bit


def _cells_at(profiles, level):
    strata = {}
    for i, key, sig in profiles:
        k = key[:level]
        bags = strata.get(k)
        if bags is None:
            bags = [[] for _ in range(_N_SIG)]
            strata[k] = bags
        bags[sig].append(i)
    out = []
    for k in sorted(strata):
        bags = strata[k]
        for hi in (3, 2):
            lo = 3 - hi
            if bags[hi] and bags[lo]:
                out.append((bags[hi], bags[lo]))
    return out


def _cells(fit, depth_cap, min_pairs):
    profiles = []
    for i, s in enumerate(fit):
        got = _profile(list(s.get("cmds") or ()), depth_cap)
        if got is not None:
            profiles.append((i, got[0], got[1]))
    if not profiles:
        return []
    best = []
    for level in (3, 2, 1, 0):
        cells = _cells_at(profiles, level)
        matched = sum(min(len(a), len(b)) for a, b in cells)
        if matched > sum(min(len(a), len(b)) for a, b in best):
            best = cells
        if matched >= int(min_pairs):
            return cells
    return best


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, depth_cap=4, min_pairs=64):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs < 2 or float(hard_frac_max) <= 0.0:
        return uniform_only()

    cells = _cells(fit, depth_cap, min_pairs)
    if not cells:
        return uniform_only()

    a_flat = torch.tensor([i for a, _b in cells for i in a], dtype=torch.long)
    b_flat = torch.tensor([i for _a, b in cells for i in b], dtype=torch.long)
    a_size = torch.tensor([len(a) for a, _b in cells], dtype=torch.long)
    b_size = torch.tensor([len(b) for _a, b in cells], dtype=torch.long)
    a_off = torch.cat([torch.zeros(1, dtype=torch.long), a_size.cumsum(0)[:-1]])
    b_off = torch.cat([torch.zeros(1, dtype=torch.long), b_size.cumsum(0)[:-1]])
    weight = torch.minimum(a_size, b_size).to(torch.float).clamp_min(1e-6)

    def pick(flat, off, size, sel, u):
        k = (u * size[sel].to(torch.float)).to(torch.long)
        k = torch.minimum(k.clamp_min(0), size[sel] - 1)
        return flat[off[sel] + k]

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)
        n_struct = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
        n_pairs = n_struct // 2
        n_uni = bs - 2 * n_pairs
        parts = []
        if n_uni > 0:
            parts.append(torch.randint(0, n, (n_uni,), generator=g))
        if n_pairs > 0:
            sel = torch.multinomial(weight, n_pairs, replacement=True, generator=g)
            u = torch.rand(2, n_pairs, generator=g)
            parts.append(pick(a_flat, a_off, a_size, sel, u[0]))
            parts.append(pick(b_flat, b_off, b_size, sel, u[1]))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
