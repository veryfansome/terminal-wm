import torch

NAME = "r7_antithetic_moverrole_pairing"
DESCRIPTION = (
    "Keeps the hop-depth stratified quota and the per-batch system-image block of the parent, and "
    "replaces the independent draw inside a scheduled depth slot with an ANTITHETIC PAIR draw. The "
    "same symbolic replay that labels a sequence by the hop count of its deepest routed read is "
    "extended to carry content identity: every mv/cp/'read > path' forwards a content token along "
    "with its hop count, 'read >> path' installs an accumulator token at the destination, and each "
    "content records the board it lives on (the /tmp/w/<family>[/<group>] prefix of where it was "
    "first displaced) and the step index of its first displacement. The sequence's deepest routed "
    "read then yields a MOVER ROLE: side 0 when the content the read returns was the earliest-"
    "displaced content on its board, side 1 when some board peer was displaced first. Sequences are "
    "binned by (system image, board family, hop depth, number of board peers) and split by side; a "
    "cell holding both sides admits an antithetic pair, one sequence from each side. Each scheduled "
    "depth slot spends a fixed fraction of its quota on such pairs — a cell is sampled per pair with "
    "weight equal to the smaller side's population, pair counts are aggregated by cell so each cell "
    "costs two draws per step, and the unpaired remainder falls back to the parent's occupancy-"
    "tilted stratum draw. A batch therefore carries, for the same board geometry and the same chain "
    "length, one trajectory whose answer sits at the head of the move order and one whose answer "
    "sits behind a peer, so the shared in-batch candidate pool always contains a near-identical "
    "read command whose answer belongs to the opposite chain role. Falls back to the parent's draw "
    "whenever a cell is one-sided, and to uniform sampling when no routed read is recoverable."
)

_READ_VERBS = ("cat", "head", "tail")
_BREAK_TOKENS = ("|", "&&", "||", ";", "<", "2>", "2>>")


def _parse_event(cmd):
    toks = cmd.split()
    if not toks:
        return None
    verb = toks[0]
    args = []
    redir = None
    rtgt = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in _BREAK_TOKENS:
            return None
        if t == ">" or t == ">>":
            redir = t
            if i + 1 < len(toks):
                rtgt = toks[i + 1]
                i += 1
        elif t.startswith(">>"):
            redir = ">>"
            rtgt = t[2:] or rtgt
        elif t.startswith(">"):
            redir = ">"
            rtgt = t[1:] or rtgt
        elif t.startswith("-") and len(t) > 1:
            pass
        else:
            args.append(t)
        i += 1
    if verb == "mv" and redir is None and len(args) == 2:
        return ("move", args[0], args[1])
    if verb == "cp" and redir is None and len(args) == 2:
        return ("copy", args[0], args[1])
    if verb in _READ_VERBS and len(args) == 1:
        if redir == ">" and rtgt:
            return ("copy", args[0], rtgt)
        if redir == ">>" and rtgt:
            return ("append", args[0], rtgt)
        if redir is None:
            return ("read", args[0], None)
    return None


def _board(path):
    parts = path.split("/")
    if len(parts) >= 5 and parts[1] == "tmp" and parts[2] == "w":
        return "/".join(parts[:5]) if parts[3] != "cups" else "/".join(parts[:4])
    if len(parts) >= 4:
        return "/".join(parts[:4])
    return path


def _family(board):
    parts = board.split("/")
    return parts[3] if len(parts) >= 4 else ""


def _replay(cmds):
    occ = {}
    home = {}
    first_move = {}
    reads = []
    for t, cmd in enumerate(cmds):
        if not isinstance(cmd, str):
            continue
        ev = _parse_event(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "move":
            cur = occ.pop(a, None)
            cid, h = cur if cur is not None else (("o", a), 0)
            occ[b] = (cid, h + 1)
            home.setdefault(cid, _board(a))
            first_move.setdefault(cid, t)
        elif kind == "copy":
            cur = occ.get(a)
            cid, h = cur if cur is not None else (("o", a), 0)
            occ[b] = (cid, h + 1)
            home.setdefault(cid, _board(a))
            first_move.setdefault(cid, t)
        elif kind == "append":
            prev = occ.get(b)
            src = occ.get(a)
            hs = (src[1] if src is not None else 0) + 1
            hp = prev[1] if prev is not None else 0
            acc = ("a", b)
            occ[b] = (acc, hs if hs > hp else hp)
            home.setdefault(acc, _board(b))
            first_move.setdefault(acc, t)
        else:
            cur = occ.get(a)
            if cur is None:
                continue
            reads.append((cur[1], cur[0], a))
    return reads, home, first_move


def _quota(shares, total):
    raw = shares * float(total)
    base = torch.floor(raw)
    left = int(total) - int(base.sum().item())
    if left > 0:
        _, order = torch.sort(raw - base, descending=True, stable=True)
        base[order[:left]] += 1.0
    return [int(v) for v in base.tolist()]


def make_batcher(fit, bs, seed, curr_frac=0.75, ramp_frac=0.35, warm_frac=0.1, tau_up=0.5,
                 lam_down=0.35, n_block_images=1, depth_cap=6, min_stratum=8, demand_pow=1.0,
                 pair_frac=0.75, peers_cap=2):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or bs <= 0 or float(curr_frac) <= 0.0:
        return uniform_batch

    cap = max(1, int(depth_cap))
    pcap = max(1, int(peers_cap))
    labels = []
    member = []
    images = []
    sides = []
    cells = []
    for s in fit:
        reads, home, first_move = _replay(s.get("cmds") or [])
        deepest = 0
        load = 0.0
        pick = None
        for d, cid, path in reads:
            if d >= 1:
                c = min(int(d), cap)
                load += float(c)
                if c >= deepest:
                    deepest = c
                    pick = (cid, path)
        labels.append(deepest)
        member.append((1.0 + load) ** float(demand_pow))
        images.append(str(s.get("image", "?")))
        side = -1
        cell = ("", 0)
        if pick is not None:
            cid, path = pick
            bd = home.get(cid) or _board(path)
            own = first_move.get(cid)
            if own is not None:
                peers = [c for c in first_move if c != cid and home.get(c) == bd]
                if peers:
                    ahead = 0
                    for c in peers:
                        if first_move[c] < own:
                            ahead += 1
                    side = 0 if ahead == 0 else 1
                    cell = (_family(bd), min(len(peers), pcap))
        sides.append(side)
        cells.append(cell)

    img_names = sorted(set(images))
    img_pos = {k: j for j, k in enumerate(img_names)}

    strata = {}
    blocks = {}
    img_load = [0.0] * len(img_names)
    for i in range(n):
        d = labels[i]
        if d < 1:
            continue
        strata.setdefault(d, []).append(i)
        blocks.setdefault((img_pos[images[i]], d), []).append(i)
        img_load[img_pos[images[i]]] += 1.0

    if not strata:
        return uniform_batch

    def _pack(idxs):
        return (torch.tensor(idxs, dtype=torch.long),
                torch.tensor([member[j] for j in idxs], dtype=torch.float))

    stratum_pool = {d: _pack(v) for d, v in strata.items()}
    block_pool = {k: _pack(v) for k, v in blocks.items()}

    pair_bins = {}
    for i in range(n):
        d = labels[i]
        if d < 1 or sides[i] < 0:
            continue
        fam, pb = cells[i]
        pair_bins.setdefault((img_pos[images[i]], fam, d, pb, sides[i]), []).append(i)
        pair_bins.setdefault((-1, fam, d, pb, sides[i]), []).append(i)
    pair_pool = {k: _pack(v) for k, v in pair_bins.items()}

    cell_keys = {}
    cell_wts = {}
    for key in sorted(pair_bins, key=lambda k: (k[0], k[1], k[2], k[3], k[4])):
        im, fam, d, pb, side = key
        if side != 0:
            continue
        other = (im, fam, d, pb, 1)
        if other not in pair_bins:
            continue
        w = min(len(pair_bins[key]), len(pair_bins[other]))
        cell_keys.setdefault((im, d), []).append((fam, pb))
        cell_wts.setdefault((im, d), []).append(float(w))
    cell_choice = {k: (cell_keys[k], torch.tensor(cell_wts[k], dtype=torch.float))
                   for k in cell_keys}

    depths = sorted(stratum_pool)
    big = [d for d in depths if len(strata[d]) >= int(min_stratum)]
    d_top = float(max(big) if big else max(depths))
    d_vec = torch.tensor([float(d) for d in depths], dtype=torch.float)
    room = [len(strata[d]) for d in depths]

    img_w = torch.tensor(img_load, dtype=torch.float).clamp_min(1e-6)
    k_imgs = max(1, min(int(n_block_images), len(img_names)))
    pf = min(1.0, max(0.0, float(pair_frac)))

    def _draw(chosen, d, count):
        parts_i = []
        parts_w = []
        for j in chosen:
            hit = block_pool.get((j, d))
            if hit is not None:
                parts_i.append(hit[0])
                parts_w.append(hit[1])
        if parts_i:
            pool_i = torch.cat(parts_i) if len(parts_i) > 1 else parts_i[0]
            pool_w = torch.cat(parts_w) if len(parts_w) > 1 else parts_w[0]
        else:
            pool_i, pool_w = stratum_pool[d]
        sel = torch.multinomial(pool_w, count, replacement=True, generator=g)
        return pool_i[sel]

    def _draw_pairs(chosen, d, n_pairs):
        opt = None
        im_used = -1
        for j in chosen:
            hit = cell_choice.get((j, d))
            if hit is not None:
                opt = hit
                im_used = j
                break
        if opt is None:
            opt = cell_choice.get((-1, d))
            im_used = -1
        if opt is None:
            return None
        keys, w = opt
        picked = torch.multinomial(w, n_pairs, replacement=True, generator=g)
        counts = torch.bincount(picked, minlength=len(keys))
        parts = []
        for ci in range(len(keys)):
            m = int(counts[ci])
            if m <= 0:
                continue
            fam, pb = keys[ci]
            for side in (0, 1):
                pi, pw = pair_pool[(im_used, fam, d, pb, side)]
                sel = torch.multinomial(pw, m, replacement=True, generator=g)
                parts.append(pi[sel])
        if not parts:
            return None
        return torch.cat(parts)

    def next_batch(step, total_steps):
        total = max(1, int(total_steps))
        cur = max(1, int(step))
        ramp = min(1.0, float(cur) / max(1.0, float(ramp_frac) * total))
        warm = min(1.0, float(cur) / max(1.0, float(warm_frac) * total))
        frontier = 1.0 + (d_top - 1.0) * ramp
        n_curr = max(0, min(bs, int(round(bs * float(curr_frac) * warm))))

        parts = []
        if n_curr > 0:
            gap = d_vec - frontier
            logw = torch.where(gap > 0.0,
                               -gap / max(1e-3, float(tau_up)),
                               gap * float(lam_down))
            w = torch.exp(logw - logw.max()).clamp_min(1e-8)
            shares = w / w.sum()
            counts = _quota(shares, n_curr)
            chosen = torch.multinomial(img_w, k_imgs, replacement=False,
                                       generator=g).tolist()
            filled = 0
            for pos, d in enumerate(depths):
                c = min(counts[pos], room[pos])
                if c <= 0:
                    continue
                got = 0
                n_pairs = int(float(c) * pf // 2.0)
                if n_pairs > 0:
                    pr = _draw_pairs(chosen, d, n_pairs)
                    if pr is not None:
                        parts.append(pr)
                        got = int(pr.numel())
                rest = c - got
                if rest > 0:
                    parts.append(_draw(chosen, d, rest))
                filled += c
            n_curr = filled
        if n_curr < bs:
            parts.append(torch.randint(0, n, (bs - n_curr,), generator=g))
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
