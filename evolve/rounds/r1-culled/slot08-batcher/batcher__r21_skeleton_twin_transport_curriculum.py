import torch

NAME = "r21_skeleton_twin_transport_curriculum"
DESCRIPTION = (
    "Composes each batch out of SKELETON TWINS — pairs of train sequences that share the same "
    "verb skeleton on the same system but differ in their file arguments — drawn with a "
    "curriculum ramp, on top of a transport-tilted background pool that up-weights sequences "
    "whose commands read a path at the end of a multi-hop lexical move chain. The twin pair puts "
    "a structurally identical, content-different sequence in the same in-batch negative pool, so "
    "the contrastive objective can only separate them by carried content; the transport tilt "
    "raises the per-step yield of the aux heads' same-path chain miners. Degrades by a fallback "
    "cascade to plain same-image blocking and then to uniform when twins do not exist."
)

_MOVE_VERBS = frozenset((
    "mv", "cp", "ln", "install", "rsync", "rename", "mvdir",
))

_READ_VERBS = frozenset((
    "cat", "ls", "head", "tail", "less", "more", "wc", "stat", "file", "grep", "egrep",
    "fgrep", "find", "du", "readlink", "realpath", "diff", "cmp", "sort", "uniq", "nl",
    "od", "xxd", "strings", "tree", "md5sum", "sha1sum", "sha256sum", "cksum", "awk",
    "sed", "tac", "cut", "basename", "dirname",
))


def _verb_and_args(cmd):
    parts = str(cmd).split()
    if not parts:
        return "", ()
    args = tuple(p for p in parts[1:] if not p.startswith("-"))
    return parts[0], args


def _basename(path):
    q = path.strip().strip('"').strip("'")
    while len(q) > 1 and q.endswith("/"):
        q = q[:-1]
    cut = q.rfind("/")
    return q[cut + 1:] if cut >= 0 else q


def _sequence_features(cmds, read_cap):
    verbs = []
    contents = []
    chain = {}
    transport = 0.0
    for c in cmds:
        v, a = _verb_and_args(c)
        verbs.append(v)
        contents.append(a)
        names = tuple(_basename(x) for x in a if x)
        if v in _MOVE_VERBS and len(names) >= 2:
            hop = max(chain.get(names[0], 0), chain.get(names[-1], 0)) + 1
            chain[names[0]] = hop
            chain[names[-1]] = hop
        elif v in _READ_VERBS:
            for nm in names:
                hops = chain.get(nm, 0)
                if hops >= 1:
                    transport += min(float(hops), read_cap)
    return tuple(verbs), tuple(contents), transport


def _grouped(keys, contents, min_members):
    buckets = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    out = []
    for k in sorted(buckets, key=lambda z: repr(z)):
        members = buckets[k]
        if len(members) < min_members:
            continue
        if len({contents[i] for i in members}) < 2:
            continue
        out.append(members)
    return out


def make_batcher(
    fit,
    bs,
    seed,
    twin_frac_max=0.5,
    ramp_frac=0.3,
    depth_tilt=1.0,
    depth_cap=6.0,
    read_cap=4.0,
):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or bs < 1:
        return uniform_batch

    skeletons = []
    contents = []
    transports = []
    images = []
    for s in fit:
        cmds = s.get("cmds") or []
        vb, ct, tr = _sequence_features(cmds, float(read_cap))
        skeletons.append(vb)
        contents.append(ct)
        transports.append(tr)
        images.append(s.get("image", "?"))

    tilt = max(0.0, float(depth_tilt))
    cap = max(0.0, float(depth_cap))
    base_w = torch.tensor(
        [1.0 + tilt * min(t, cap) for t in transports], dtype=torch.float
    )
    log_w = base_w.log()

    levels = [
        [(images[i], skeletons[i]) for i in range(n)],
        [(skeletons[i],) for i in range(n)],
        [(images[i], tuple(sorted(skeletons[i]))) for i in range(n)],
        [(images[i],) for i in range(n)],
    ]
    groups = []
    for keys in levels:
        groups = _grouped(keys, contents, 2)
        if groups:
            break

    frac_max = min(1.0, max(0.0, float(twin_frac_max)))
    ramp_share = min(1.0, max(1e-6, float(ramp_frac)))

    if not groups or bs < 2:
        def weighted_batch(step, total_steps):
            r = min(1.0, max(0.0, step / max(1.0, ramp_share * max(1, total_steps))))
            w = torch.exp(r * log_w)
            return torch.multinomial(w, bs, replacement=True, generator=g).tolist()

        return weighted_batch

    widest = max(len(m) for m in groups)
    members = torch.zeros(len(groups), widest, dtype=torch.long)
    counts = torch.zeros(len(groups), dtype=torch.long)
    group_w = torch.zeros(len(groups), dtype=torch.float)
    for gi, m in enumerate(groups):
        members[gi, : len(m)] = torch.tensor(m, dtype=torch.long)
        counts[gi] = len(m)
        group_w[gi] = float(base_w[torch.tensor(m, dtype=torch.long)].sum())
    group_w = group_w.clamp_min(1e-6)

    cache = {"key": None, "w": None}

    def next_batch(step, total_steps):
        horizon = max(1.0, ramp_share * max(1, int(total_steps)))
        r = min(1.0, max(0.0, step / horizon))
        quant = int(round(r * 64.0))
        if cache["key"] != quant:
            cache["key"] = quant
            cache["w"] = torch.exp((quant / 64.0) * log_w)
        w = cache["w"]

        blocks = int(bs * frac_max * r) // 2
        blocks = max(0, min(blocks, bs // 2))
        n_rest = bs - 2 * blocks

        parts = []
        if n_rest > 0:
            parts.append(torch.multinomial(w, n_rest, replacement=True, generator=g))
        if blocks > 0:
            gsel = torch.multinomial(group_w, blocks, replacement=True, generator=g)
            csel = counts[gsel]
            cf = csel.to(torch.float)
            u = torch.rand(2 * blocks, generator=g)
            r1 = (u[:blocks] * cf).long().minimum(csel - 1)
            r2 = (u[blocks:] * (cf - 1.0)).long().minimum(csel - 2)
            r2 = r2 + (r2 >= r1).long()
            parts.append(members[gsel, r1])
            parts.append(members[gsel, r2])

        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
