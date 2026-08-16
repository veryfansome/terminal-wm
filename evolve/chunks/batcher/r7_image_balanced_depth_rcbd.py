import torch

NAME = "r7_image_balanced_depth_rcbd"
DESCRIPTION = (
    "Randomized complete block design over (routing depth x system image). Every training "
    "sequence is replayed through a symbolic content tracker built from its command strings "
    "alone -- 'mv A B' carries the content at A to B and destroys A, 'cat A > B' and "
    "'cat A >> B' carry a copy, a bare 'cat A' reads whatever now sits at A -- and the "
    "sequence is labelled by the hop count of its deepest routed read, capped. Depth is the "
    "TREATMENT and system image is the NUISANCE FACTOR. Each step allocates the batch across "
    "depth rows by largest-remainder rounding of a mixture between the empirical depth "
    "distribution and a monotone depth shape over the occupied rows, annealed up over the "
    "opening of training, with each row's quota capped at a fixed multiple of the visit rate "
    "uniform sampling would give it. Each row's quota is then spread across the images that occupy that "
    "row as an equal base share plus remainder units assigned fewest-options-row-first to "
    "whichever image currently holds the fewest slots in this batch, ties broken by a "
    "step-rotating offset. The result is a two-way layout in which every image appears in "
    "every batch in near-equal number and image is balanced within each depth row, so the "
    "image-specific component of every single gradient step is averaged out inside the step "
    "instead of integrated across steps. Row cells are drawn from per-cell shuffled epoch "
    "cursors, so a cell is exhausted without replacement before it reshuffles. Falls back to "
    "uniform iid sampling when the split is degenerate."
)

_MOVE = 0
_COPY = 1
_APPEND = 2
_READ = 3
_ROW_OFFSET_STRIDE = 7


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
        return (_MOVE, args[0], args[1])
    if verb == "cp" and redir is None and len(args) == 2:
        return (_COPY, args[0], args[1])
    if verb in ("cat", "head", "tail") and len(args) == 1:
        if redir == ">" and rtarget:
            return (_COPY, args[0], rtarget)
        if redir == ">>" and rtarget:
            return (_APPEND, args[0], rtarget)
        if redir is None:
            return (_READ, args[0], None)
    return None


def _routed_depth(cmds, cap):
    hops = {}
    best = 0
    for c in cmds:
        ev = _parse(c)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == _MOVE:
            hops[b] = hops.pop(a, 0) + 1
        elif kind == _COPY:
            hops[b] = hops.get(a, 0) + 1
        elif kind == _APPEND:
            prev = hops.get(b, 0)
            cur = hops.get(a, 0) + 1
            hops[b] = prev if prev > cur else cur
        else:
            d = hops.get(a, 0)
            if d > best:
                best = d
    return cap if best > cap else best


def _row_quota(p, active, bs, caps):
    q = [0] * len(p)
    frac = []
    used = 0
    for r in active:
        x = p[r] * bs
        f = int(x)
        if f > bs:
            f = bs
        q[r] = f
        used += f
        frac.append((x - f, r))
    frac.sort(key=lambda t: (-t[0], -t[1]))
    rem = bs - used
    j = 0
    while rem > 0:
        q[frac[j % len(frac)][1]] += 1
        rem -= 1
        j += 1
    while rem < 0:
        r = frac[j % len(frac)][1]
        if q[r] > 0:
            q[r] -= 1
            rem += 1
        j += 1
    surplus = 0
    for r in active:
        if q[r] > caps[r]:
            surplus += q[r] - caps[r]
            q[r] = caps[r]
    if surplus > 0:
        for r in sorted(active, reverse=True):
            if surplus <= 0:
                break
            room = caps[r] - q[r]
            if room > 0:
                take = room if room < surplus else surplus
                q[r] += take
                surplus -= take
    if surplus > 0:
        r0 = max(active, key=lambda r: caps[r])
        q[r0] += surplus
    return q


def make_batcher(fit, bs, seed, depth_cap=5, mix_max=0.75, ramp_frac=0.3, rate_cap=6.0,
                 depth_shape=1.0):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0:
        return uniform_only()

    cap = max(1, int(depth_cap))
    names = sorted({str(s.get("image", "?")) for s in fit})
    img_of = {k: j for j, k in enumerate(names)}
    n_img = len(names)
    n_row = cap + 1

    cells = [[[] for _ in range(n_img)] for _ in range(n_row)]
    for i, s in enumerate(fit):
        d = _routed_depth(s.get("cmds") or [], cap)
        cells[d][img_of[str(s.get("image", "?"))]].append(i)

    row_size = [0] * n_row
    avail = [[] for _ in range(n_row)]
    for r in range(n_row):
        for gi in range(n_img):
            k = len(cells[r][gi])
            if k > 0:
                row_size[r] += k
                avail[r].append(gi)
    active = [r for r in range(n_row) if row_size[r] > 0]
    if not active:
        return uniform_only()

    emp = [row_size[r] / float(n) for r in range(n_row)]
    shape = [0.0] * n_row
    for r in active:
        shape[r] = float(1 + r) ** float(depth_shape)
    shape_tot = sum(shape[r] for r in active)
    if shape_tot <= 0.0:
        for r in active:
            shape[r] = 1.0 / float(len(active))
    else:
        for r in active:
            shape[r] = shape[r] / shape_tot
    caps = [0] * n_row
    for r in active:
        c = int(float(rate_cap) * float(bs) * emp[r])
        if c < 1:
            c = 1
        if c > bs:
            c = bs
        caps[r] = c

    perms = [[None] * n_img for _ in range(n_row)]
    pos = [[0] * n_img for _ in range(n_row)]

    def pull(r, gi):
        lst = cells[r][gi]
        k = len(lst)
        p = perms[r][gi]
        if p is None or pos[r][gi] >= k:
            p = torch.randperm(k, generator=g).tolist()
            perms[r][gi] = p
            pos[r][gi] = 0
        v = lst[p[pos[r][gi]]]
        pos[r][gi] += 1
        return v

    fill_order = sorted(active, key=lambda r: (len(avail[r]), -r))

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * float(max(1, int(total_steps)))))
        ramp = float(step) / float(ramp_steps)
        if ramp > 1.0:
            ramp = 1.0
        lam = float(mix_max) * ramp
        p = [0.0] * n_row
        for r in active:
            p[r] = (1.0 - lam) * emp[r] + lam * shape[r]
        tot = sum(p[r] for r in active)
        if tot <= 0.0:
            for r in active:
                p[r] = shape[r]
        else:
            for r in active:
                p[r] = p[r] / tot
        q = _row_quota(p, active, bs, caps)

        load = [0] * n_img
        cnt = [[0] * n_img for _ in range(n_row)]
        for r in fill_order:
            need = q[r]
            if need <= 0:
                continue
            options = avail[r]
            k = len(options)
            base = need // k
            extra = need - base * k
            if base > 0:
                for gi in options:
                    cnt[r][gi] += base
                    load[gi] += base
            if extra > 0:
                off = (int(step) + _ROW_OFFSET_STRIDE * r) % n_img
                ranked = sorted(options, key=lambda gi: (load[gi], (gi - off) % n_img))
                for gi in ranked[:extra]:
                    cnt[r][gi] += 1
                    load[gi] += 1

        out = []
        for r in active:
            for gi in avail[r]:
                for _ in range(cnt[r][gi]):
                    out.append(pull(r, gi))
        if len(out) < bs:
            pad = torch.randint(0, n, (bs - len(out),), generator=g).tolist()
            out.extend(pad)
        out = out[:bs]
        order = torch.randperm(bs, generator=g).tolist()
        return [out[i] for i in order]

    return next_batch
