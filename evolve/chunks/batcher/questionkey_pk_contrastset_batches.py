import torch

NAME = "questionkey_pk_contrastset_batches"
DESCRIPTION = (
    "PK contrast-set batching keyed on the QUESTION instead of on the system image. Each train "
    "sequence is parsed into a content-forwarding graph (`mv src dst` and `cat src > dst` are "
    "edges, `>>` is not); every read is walked backwards through the edges that wrote its path "
    "earlier, and the sequence is filed under the command string of its deepest routed read -- "
    "verbatim when that exact string has enough members, otherwise backed off to a digit-masked "
    "template that keeps the trailing move-suffix component (cat /tmp/w/cups/g#/f#.tmp.8). A batch "
    "is P distinct keys x K sequences per key, so K sequences in a batch carry the same read "
    "command, the same board shape and the same chain length with K different answers. Inside a "
    "key the K slots are a random cyclic window of a round-robin interleaving of the members by "
    "ORANK class -- the rank of the routed content among the contents exposed on its board before "
    "the board's first move -- so the group spans which exposed content turns out to be the answer "
    "rather than repeating one exposure order. Keys are drawn without replacement with weight "
    "member_count^size_pow * (1 + mean key depth)^(depth_pow * ramp); the blocked fraction and the "
    "depth exponent share one ramp, and the uniform remainder keeps coverage. Private generators "
    "only; the sequences are read, never mutated."
)

_READ_VERBS = ("cat", "head", "tail")
_N_ORANK = 4
_WALK_CAP = 64


def _edges_and_reads(cmds):
    edges = []
    reads = []
    for t, c in enumerate(cmds):
        if not isinstance(c, str):
            continue
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            if not p[1].startswith("-") and not p[2].startswith("-"):
                edges.append((t, p[1], p[2]))
        elif len(p) == 4 and p[0] in _READ_VERBS and p[2] == ">":
            if not p[1].startswith("-") and not p[3].startswith("-"):
                edges.append((t, p[1], p[3]))
        elif len(p) == 2 and p[0] in _READ_VERBS and not p[1].startswith("-"):
            reads.append((t, p[1]))
    return edges, reads


def _component(edges, root):
    adj = {}
    for _t, a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen = set()
    stack = [root]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for y in adj.get(x, ()):
            if y not in seen:
                stack.append(y)
    return seen


def _board_prefix(nodes):
    dirs = [[q for q in p.split("/") if q][:-1] for p in nodes]
    if not dirs:
        return None
    shared = dirs[0]
    for d in dirs[1:]:
        k = 0
        while k < len(shared) and k < len(d) and shared[k] == d[k]:
            k += 1
        shared = shared[:k]
        if not shared:
            return None
    return "/" + "/".join(shared) + "/"


def _scan(cmds):
    edges, reads = _edges_and_reads(cmds)
    if not edges or not reads:
        return None

    wrote = {}
    for t, a, b in edges:
        wrote.setdefault(b, []).append((t, a))

    best = None
    for t, path in reads:
        cur, ct, d = path, t, 0
        while d < _WALK_CAP:
            prior = [(j, s) for j, s in wrote.get(cur, ()) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            d += 1
        if d >= 1 and (best is None or (d, t) > (best[0], best[1])):
            best = (d, t, cur)
    if best is None:
        return None

    depth, read_t, origin = best
    comp = _component(edges, origin)
    prefix = _board_prefix(comp)
    board_t0 = min(t for t, a, b in edges if a in comp or b in comp)

    exposed = []
    for t, path in reads:
        if t >= board_t0:
            break
        on_board = path in comp or (prefix is not None and path.startswith(prefix))
        if on_board and path not in exposed:
            exposed.append(path)
    orank = exposed.index(origin) if origin in exposed else -1

    ocls = min(orank, _N_ORANK - 1) if orank >= 0 else _N_ORANK
    return cmds[read_t], _template(cmds[read_t]), depth, ocls


def _mask_digits(s):
    return "".join("#" if ch.isdigit() else ch for ch in s)


def _template(cmd):
    parts = cmd.split()
    if len(parts) != 2:
        return _mask_digits(cmd)
    comps = parts[1].split("/")
    out = []
    for k, c in enumerate(comps):
        if k < len(comps) - 1:
            out.append(_mask_digits(c))
            continue
        dots = c.split(".")
        keep = len(dots) > 1 and dots[-1].isdigit()
        out.append(".".join(
            q if (keep and j == len(dots) - 1) else _mask_digits(q)
            for j, q in enumerate(dots)))
    return parts[0] + " ~ " + "/".join(out)


def _cycle_order(rows, gen):
    buckets = [[] for _ in range(_N_ORANK + 1)]
    for i, oc in rows:
        buckets[oc].append(i)
    lanes = []
    for b in buckets:
        if not b:
            continue
        tb = torch.tensor(b, dtype=torch.long)
        lanes.append(tb[torch.randperm(tb.numel(), generator=gen)])
    width = max(int(t.numel()) for t in lanes)
    order = []
    for r in range(width):
        for lane in lanes:
            if r < int(lane.numel()):
                order.append(int(lane[r]))
    return torch.tensor(order, dtype=torch.long)


def make_batcher(fit, bs, seed, keys_per_batch_min=1, seqs_per_key=8, min_members=4,
                 hard_frac_max=0.75, ramp_frac=0.25, depth_pow=1.5, size_pow=0.5):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or float(hard_frac_max) <= 0.0:
        return uniform_only()

    floor = max(2, int(min_members))
    scanned = []
    exact_n = {}
    for i, s in enumerate(fit):
        got = _scan(s.get("cmds") or ())
        if got is None:
            continue
        scanned.append((i, got))
        exact_n[got[0]] = exact_n.get(got[0], 0) + 1

    groups = {}
    depth_sum = {}
    for i, (kx, kt, d, ocls) in scanned:
        key = kx if exact_n[kx] >= floor else "\x00" + kt
        groups.setdefault(key, []).append((i, ocls))
        depth_sum[key] = depth_sum.get(key, 0.0) + float(d)

    keys = sorted(k for k in groups if len(groups[k]) >= floor)
    if not keys:
        return uniform_only()

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x5BF03635) & 0x7FFFFFFF)
    cycles = [_cycle_order(groups[k], g_setup) for k in keys]
    base_w = torch.tensor(
        [float(len(groups[k])) ** float(size_pow) for k in keys], dtype=torch.float
    ).clamp_min(1e-6)
    key_depth = torch.tensor(
        [depth_sum[k] / float(len(groups[k])) for k in keys], dtype=torch.float)

    kpg = max(1, int(seqs_per_key))
    kmin = max(1, int(keys_per_batch_min))
    n_keys = len(keys)

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        r = min(1.0, float(step) / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * r))))

        picks = []
        got = 0
        if n_hard > 0:
            w = base_w * (1.0 + key_depth).pow(float(depth_pow) * r)
            w = torch.where(torch.isfinite(w) & (w > 0.0), w, torch.full_like(w, 1e-6))
            want = max(kmin, (n_hard + kpg - 1) // kpg)
            p_keys = max(1, min(n_keys, want))
            chosen = torch.multinomial(w, p_keys, replacement=False, generator=g)
            for c in chosen.tolist():
                need = n_hard - got
                if need <= 0:
                    break
                cyc = cycles[c]
                m = int(cyc.numel())
                take = min(kpg, m, need)
                if take <= 0:
                    continue
                o = int(torch.randint(0, m, (1,), generator=g).item())
                picks.append(cyc[(o + torch.arange(take)) % m])
                got += take

        rest = bs - got
        parts = []
        if rest > 0:
            parts.append(torch.randint(0, n, (rest,), generator=g))
        parts.extend(picks)
        batch = torch.cat(parts) if len(parts) > 1 else parts[0]
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
