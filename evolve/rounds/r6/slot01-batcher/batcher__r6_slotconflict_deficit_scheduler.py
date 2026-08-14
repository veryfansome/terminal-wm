import torch

NAME = "r6_slotconflict_deficit_scheduler"
DESCRIPTION = (
    "Replaces stochastic batch composition with a deterministic service scheduler. Every "
    "sequence is parsed into its mv edges and its plain reads; a read is routed when walking "
    "backwards through the moves that wrote its path earlier in the same trajectory takes at "
    "least one hop. The exact destination path string of each routed read is a SLOT KEY, and "
    "the sequences sharing a slot key form a conflict clique: inside a clique the read command "
    "is byte-identical while the content the chain delivered there is not, so a batch holding "
    "several members of one clique contains the same question with different answers and the "
    "in-batch contrastive term cannot be satisfied by a command-to-observation lookup. Cliques "
    "are served by largest-deficit-first scheduling (deficit round robin, Shreedhar & Varghese "
    "1996): each clique carries a deficit that grows by its quantum share of the exactly "
    "conserved service mass and drops by the slots it consumes, so service tracks the quantum "
    "with bounded discrepancy instead of only in expectation, and members inside a clique come "
    "off a rotating cursor over a seeded permutation. The remaining slots are apportioned over "
    "chain-depth strata by Neyman allocation, n_h proportional to N_h * sigma_h^beta with "
    "sigma_h the within-stratum spread of the deepest routed read's observation embedding, "
    "realized by largest-remainder apportionment with a carried remainder and drawn from "
    "per-stratum shuffled epochs without replacement. No multinomial and no iid draw appears "
    "anywhere in the composition."
)

_READ_VERBS = ("cat", "head", "tail")
_N_STRATA = 4


def _depth_bin(d):
    if d <= 0:
        return 0
    if d == 1:
        return 1
    if d <= 3:
        return 2
    return 3


def _routed_reads(cmds, max_hops):
    mvs = []
    reads = []
    for t, c in enumerate(cmds):
        p = c.split()
        if len(p) == 3 and p[0] == "mv":
            mvs.append((t, p[1], p[2]))
        elif len(p) == 2 and p[0] in _READ_VERBS:
            reads.append((t, p[1]))
    if not mvs or not reads:
        return 0, [], -1
    wrote = {}
    for t, a, b in mvs:
        wrote.setdefault(b, []).append((t, a))
    slots = []
    best_d = 0
    best_t = -1
    for t, path in reads:
        cur = path
        ct = t
        d = 0
        while d < max_hops:
            prior = [(j, s) for j, s in wrote.get(cur, []) if j < ct]
            if not prior:
                break
            ct, cur = max(prior)
            d += 1
        if d >= 1:
            slots.append(path)
            if d > best_d:
                best_d = d
                best_t = t
    return best_d, slots, best_t


def _feature_vector(seq, read_t):
    zo = seq.get("z_obs")
    if zo is None or int(zo.shape[0]) == 0:
        return None
    if 0 <= read_t < int(zo.shape[0]):
        return zo[read_t].detach().to(torch.float32)
    return zo.detach().to(torch.float32).mean(dim=0)


def make_batcher(fit, bs, seed, conf_frac_max=0.6, ramp_frac=0.25, block_size=4,
                 clique_gamma=0.5, neyman_beta=1.0, min_clique=2, max_clique=192,
                 max_cliques=6000, max_hops=32):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    if n < 2 or bs <= 0:
        def next_batch_degenerate(step, total_steps):
            return torch.randint(0, max(1, n), (max(0, bs),), generator=g).tolist()
        return next_batch_degenerate

    g_setup = torch.Generator().manual_seed((int(seed) * 2654435761 + 12345) & 0x7FFFFFFF)

    bin_of = [0] * n
    slot_members = {}
    slot_images = {}
    feat_sq = torch.zeros(_N_STRATA)
    feat_cnt = torch.zeros(_N_STRATA)
    feat_dim = 0
    acc = [None] * _N_STRATA

    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        depth, slots, read_t = _routed_reads(list(cmds), int(max_hops))
        h = _depth_bin(depth)
        bin_of[i] = h
        img = s.get("image", "?")
        for path in slots:
            mem = slot_members.get(path)
            if mem is None:
                slot_members[path] = [i]
                slot_images[path] = {img}
            else:
                if mem[-1] != i:
                    mem.append(i)
                slot_images[path].add(img)
        v = _feature_vector(s, read_t)
        if v is not None:
            if feat_dim == 0:
                feat_dim = int(v.numel())
            if int(v.numel()) == feat_dim:
                if acc[h] is None:
                    acc[h] = torch.zeros(feat_dim)
                acc[h] = acc[h] + v
                feat_sq[h] = feat_sq[h] + float(v.pow(2).sum())
                feat_cnt[h] = feat_cnt[h] + 1.0

    members = []
    for h in range(_N_STRATA):
        rows = [i for i in range(n) if bin_of[i] == h]
        members.append(torch.tensor(rows, dtype=torch.long))

    sigma = torch.ones(_N_STRATA)
    if feat_dim > 0:
        for h in range(_N_STRATA):
            c = float(feat_cnt[h])
            if c >= 2.0 and acc[h] is not None:
                mean_sq = float(feat_sq[h]) / c
                mean_vec = acc[h] / c
                var = max(0.0, mean_sq - float(mean_vec.pow(2).sum())) / float(feat_dim)
                sigma[h] = max(1e-6, var) ** 0.5
    sigma = sigma / sigma.mean().clamp_min(1e-12)

    sizes = torch.tensor([float(m.numel()) for m in members])
    qb = sizes * sigma.pow(float(neyman_beta))
    qb = torch.where(sizes > 0, qb, torch.zeros_like(qb))
    if float(qb.sum()) <= 0.0:
        qb = torch.where(sizes > 0, torch.ones_like(qb), torch.zeros_like(qb))
    qb = qb / qb.sum().clamp_min(1e-12)

    perms = [None] * _N_STRATA
    poss = [0] * _N_STRATA
    remainder = torch.zeros(_N_STRATA)

    def reshuffle(h):
        m = members[h]
        perms[h] = m[torch.randperm(m.numel(), generator=g)]
        poss[h] = 0

    for h in range(_N_STRATA):
        if members[h].numel() > 0:
            reshuffle(h)

    def take_from(h, m):
        out = []
        need = int(m)
        if members[h].numel() == 0 or need <= 0:
            return out
        while need > 0:
            p = perms[h]
            pos = poss[h]
            avail = int(p.numel()) - pos
            if avail <= 0:
                reshuffle(h)
                continue
            k = min(need, avail)
            out.extend(p[pos:pos + k].tolist())
            poss[h] = pos + k
            need -= k
        return out

    def apportion(total):
        counts = [0] * _N_STRATA
        if total <= 0:
            return counts
        raw = qb * float(total)
        acc_rem = remainder + raw
        base = torch.floor(acc_rem)
        acc_rem = acc_rem - base
        for h in range(_N_STRATA):
            counts[h] = int(base[h])
        for h in range(_N_STRATA):
            remainder[h] = acc_rem[h]
        live = [h for h in range(_N_STRATA) if members[h].numel() > 0]
        if not live:
            return counts
        for h in range(_N_STRATA):
            if members[h].numel() == 0 and counts[h] > 0:
                counts[live[0]] += counts[h]
                counts[h] = 0
        diff = total - sum(counts)
        while diff > 0:
            best = live[0]
            for h in live:
                if float(remainder[h]) > float(remainder[best]):
                    best = h
            counts[best] += 1
            remainder[best] = remainder[best] - 1.0
            diff -= 1
        while diff < 0:
            best = None
            for h in live:
                if counts[h] > 0 and (best is None or counts[h] > counts[best]):
                    best = h
            if best is None:
                break
            counts[best] -= 1
            remainder[best] = remainder[best] + 1.0
            diff += 1
        return counts

    blk = max(1, int(block_size))
    keys = sorted(k for k, v in slot_members.items() if len(v) >= max(2, int(min_clique)))
    cliques = []
    weights = []
    for k in keys:
        rows = slot_members[k]
        t = torch.tensor(rows, dtype=torch.long)
        if t.numel() > int(max_clique):
            t = t[torch.randperm(t.numel(), generator=g_setup)[: int(max_clique)]]
            t, _ = torch.sort(t)
        t = t[torch.randperm(t.numel(), generator=g_setup)]
        cliques.append(t)
        natural = float(bs) * float(len(rows)) / float(n)
        scarcity = max(0.05, float(blk) - natural)
        weights.append(scarcity ** float(clique_gamma) * float(len(slot_images[k])))
    if len(cliques) > int(max_cliques):
        keep = torch.randperm(len(cliques), generator=g_setup)[: int(max_cliques)]
        keep, _ = torch.sort(keep)
        cliques = [cliques[int(j)] for j in keep]
        weights = [weights[int(j)] for j in keep]

    n_cliques = len(cliques)
    if n_cliques > 0:
        qc = torch.tensor(weights, dtype=torch.float32).clamp_min(1e-8)
        qc = qc / qc.sum().clamp_min(1e-12)
        deficit = torch.zeros(n_cliques)
        jitter = torch.rand(n_cliques, generator=g_setup) * 1e-4
        cursors = [0] * n_cliques
    else:
        qc = torch.zeros(0)
        deficit = torch.zeros(0)
        jitter = torch.zeros(0)
        cursors = []

    frac_max = max(0.0, min(1.0, float(conf_frac_max)))
    ramp_denom = max(1e-6, float(ramp_frac))

    def next_batch(step, total_steps):
        picks = []
        if n_cliques > 0 and frac_max > 0.0:
            T = max(1, int(total_steps))
            ramp = min(1.0, float(step) / max(1.0, ramp_denom * T))
            budget = int(round(bs * frac_max * ramp))
            budget = max(0, min(bs, budget))
            k = min(n_cliques, budget // blk)
            if k > 0:
                _, order = torch.topk(deficit + jitter, k)
                served = 0
                for ci in order.tolist():
                    mem = cliques[ci]
                    sz = int(mem.numel())
                    m = min(blk, sz, bs - len(picks))
                    if m <= 0:
                        break
                    pos = cursors[ci]
                    for _ in range(m):
                        picks.append(int(mem[pos % sz]))
                        pos += 1
                    cursors[ci] = pos % sz
                    deficit[ci] = deficit[ci] - float(m)
                    served += m
                if served > 0:
                    deficit.add_(qc * float(served))
        n_base = bs - len(picks)
        counts = apportion(n_base)
        for h in range(_N_STRATA):
            picks.extend(take_from(h, counts[h]))
        short = bs - len(picks)
        if short > 0:
            live = [h for h in range(_N_STRATA) if members[h].numel() > 0]
            h = live[0] if live else 0
            picks.extend(take_from(h, short))
        if len(picks) > bs:
            picks = picks[:bs]
        while len(picks) < bs:
            picks.append(int(torch.randint(0, n, (1,), generator=g).item()))
        batch = torch.tensor(picks, dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
