import re

import torch

NAME = "r7_namecue_propensity_deepcells"
DESCRIPTION = (
    "Shortcut-propensity resampling, imported from group-balanced training under spurious "
    "correlation: when one group of examples rewards a nuisance rule, subsampling that group "
    "removes the rule's marginal payoff without discarding any structure. The nuisance here is "
    "the location NAME. Every training sequence is reduced, from command strings alone, to its "
    "deepest mv-routed read: a 'cat P' walked backwards through the mv edges that wrote P earlier "
    "in the same trajectory, yielding the hop count and the origin path the routed content was "
    "exposed at. Stripping the trailing numeric copy-suffixes off P gives the base path whose name "
    "the read location carries; a sequence is NAME-AGREEING when that base equals the routed "
    "origin, i.e. when answering 'whatever lives at the file this location is named after' is "
    "correct, and NAME-CROSSING otherwise. A batch is a uniform iid base part plus a focus part; "
    "the focus part draws cells of (hop-count class capped at four, name-agreement), spending "
    "three quarters of its slots on hop counts three and above off a fixed wheel and annealing "
    "the name-agreeing share from its natural prevalence down to a floor, so late training pays "
    "the name rule almost nothing where the chain is long. Focus slots are drawn distinct within "
    "a batch. The focused fraction ramps from zero, so the opening of training is the plain "
    "uniform distribution; falls back to uniform when no mv-routed read can be parsed."
)

_SUFFIX = re.compile(r"\.\d+$")
_DEPTH_CAP = 4
_WHEEL = (1, 2, 3, 4, 3, 4, 4, 3)
_MAX_HOPS = 64


def _strip_copy_suffix(path):
    p = path
    while True:
        q = _SUFFIX.sub("", p)
        if q == p:
            return p
        p = q


def _routed_read(cmds):
    mvs = []
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            mvs.append((t, p[1], p[2]))
    if not mvs:
        return None
    wrote = {}
    for t, src, dst in mvs:
        wrote.setdefault(dst, []).append((t, src))
    best = None
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) != 2 or p[0] != "cat":
            continue
        cur, ct, hops = p[1], t, 0
        while hops < _MAX_HOPS:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            hops += 1
        if hops >= 1 and (best is None or hops > best[0]):
            best = (hops, p[1], cur)
    return best


def _scan(fit):
    out = []
    for s in fit:
        got = _routed_read(s.get("cmds") or [])
        if got is None:
            out.append(None)
            continue
        hops, read_path, origin = got
        cls = min(int(hops), _DEPTH_CAP)
        agree = 1 if origin == _strip_copy_suffix(read_path) else 0
        out.append((cls, agree))
    return out


def make_batcher(fit, bs, seed, focus_frac_max=0.75, ramp_frac=0.3, self_named_floor=0.1):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or float(focus_frac_max) <= 0.0:
        return uniform_only()

    cells = {}
    for i, lab in enumerate(_scan(fit)):
        if lab is not None:
            cells.setdefault(lab, []).append(i)
    if not cells:
        return uniform_only()

    cells = {k: torch.tensor(v, dtype=torch.long) for k, v in sorted(cells.items())}
    labelled = torch.cat([cells[k] for k in sorted(cells)])
    n_agree = sum(int(v.numel()) for k, v in cells.items() if k[1] == 1)
    q_nat = float(n_agree) / max(1, int(labelled.numel()))
    q_floor = min(max(float(self_named_floor), 0.0), 1.0)

    by_class = {}
    for (c, a), v in sorted(cells.items()):
        by_class.setdefault(c, []).append(v)
    class_pool = {c: (v[0] if len(v) == 1 else torch.cat(v)) for c, v in by_class.items()}

    resolved = {}
    for c in range(1, _DEPTH_CAP + 1):
        for a in (0, 1):
            pool = cells.get((c, a))
            if pool is None:
                pool = cells.get((c, 1 - a))
            if pool is None:
                pool = class_pool.get(c)
            if pool is None:
                pool = labelled
            resolved[(c, a)] = pool

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)
        q = q_nat + (q_floor - q_nat) * ramp
        n_focus = max(0, min(bs, int(round(bs * float(focus_frac_max) * ramp))))
        n_base = bs - n_focus
        out = []
        used = set()
        if n_focus > 0:
            coin = torch.rand((n_focus,), generator=g).tolist()
            groups = {}
            for j in range(n_focus):
                key = (_WHEEL[(int(step) + j) % len(_WHEEL)], 1 if coin[j] < q else 0)
                groups[key] = groups.get(key, 0) + 1
            for key in sorted(groups):
                k = groups[key]
                pool = resolved[key]
                draw = pool[torch.randint(0, int(pool.numel()), (2 * k,), generator=g)].tolist()
                got = 0
                for v in draw:
                    if v not in used:
                        used.add(v)
                        out.append(v)
                        got += 1
                        if got == k:
                            break
                if got < k:
                    out.extend(draw[:k - got])
        if n_base > 0:
            out.extend(torch.randint(0, n, (n_base,), generator=g).tolist())
        batch = torch.tensor(out, dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
