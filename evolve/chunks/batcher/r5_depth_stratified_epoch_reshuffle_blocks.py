import math
import re

import torch

NAME = "r5_depth_stratified_epoch_reshuffle_blocks"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic file-content tracker "
    "(mv / cp / cat-redirect move or copy content between paths, a bare read returns whatever "
    "now sits at its path) and labels the sequence by the hop count of its deepest routed read, "
    "which partitions the training set into disjoint hop-count strata. Every batch reserves a "
    "uniform base share and allocates the remaining slots across strata as exact integer quotas "
    "by largest-remainder rounding of the stratum sizes tilted by exp(beta*hops), with the tilt "
    "annealed up from zero over the opening fraction of training, each quota clamped so no "
    "stratum is visited more than a fixed multiple of the per-sequence rate uniform sampling "
    "would give, and strata smaller than a floor excluded from the quota entirely. Stratum slots "
    "are filled from a per-stratum shuffled epoch cursor, so drawing inside a stratum is without "
    "replacement until that stratum is exhausted and then reshuffled. With annealed probability a "
    "run of slots is instead filled from a collision block: several distinct sequences whose "
    "deepest routed read shares one image, one digit-normalized path shape and one hop count, "
    "pre-filtered once at setup under a fixed random projection so that block members agree on "
    "the read-command embedding and disagree on the read-answer embedding. Falls back to uniform "
    "sampling when no routed read is recoverable."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
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


def _deepest_routed_read(cmds, depth_cap):
    hops = {}
    best_d = 0
    best_t = -1
    best_path = ""
    count = 0
    for t, cmd in enumerate(cmds):
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
                count += 1
                dd = min(d, depth_cap)
                if dd > best_d:
                    best_d = dd
                    best_t = t
                    best_path = a
    return best_d, best_t, best_path, count


def _refine_block(fit, members, rmat, sim_thresh, ans_max_sim):
    rows = []
    qv = []
    av = []
    for i, t in members:
        zc = fit[i].get("z_cmd")
        zo = fit[i].get("z_obs")
        if zc is None or zo is None or zc.dim() != 2 or zo.dim() != 2:
            return members
        if t >= int(zc.shape[0]) or t >= int(zo.shape[0]):
            continue
        rows.append((i, t))
        qv.append(zc[t].detach().float())
        av.append(zo[t].detach().float())
    if len(rows) < 2:
        return members
    q = torch.nn.functional.normalize(torch.stack(qv) @ rmat, dim=1, eps=1e-8)
    a = torch.nn.functional.normalize(torch.stack(av) @ rmat, dim=1, eps=1e-8)
    qsim = (q @ q[0:1].t()).view(-1).tolist()
    asim = (a @ a.t()).tolist()
    keep = []
    kept_rows = []
    for r in range(len(rows)):
        if r > 0 and qsim[r] < sim_thresh:
            continue
        dup = False
        for kr in kept_rows:
            if asim[r][kr] > ans_max_sim:
                dup = True
                break
        if dup:
            continue
        keep.append(rows[r])
        kept_rows.append(r)
    if len(keep) < 2:
        return rows
    return keep


def make_batcher(fit, bs, seed, base_frac=0.25, depth_beta=1.2, ramp_frac=0.3,
                 block_frac_max=0.5, group_size=4, max_repeat=20.0, min_stratum=16,
                 depth_cap=4, bucket_cap=32, max_filter_buckets=1200, proj_dim=128,
                 sim_thresh=0.6, ans_max_sim=0.995):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n <= 1 or bs <= 0:
        return uniform_batch

    dcap = max(1, int(depth_cap))
    depth = [0] * n
    rcount = [0] * n
    keyed = {}
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        d, t, path, c = _deepest_routed_read(cmds, dcap)
        depth[i] = d
        rcount[i] = c
        if d >= 1 and t >= 0:
            keyed.setdefault((str(s.get("image", "?")), _norm_path(path), d), []).append((i, t))

    strata = [[] for _ in range(dcap + 1)]
    for i in range(n):
        strata[depth[i]].append(i)
    sizes = [len(x) for x in strata]
    if sizes[0] == n:
        return uniform_batch

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0x7FFFFFFF)
    dim = 0
    for s in fit:
        zc = s.get("z_cmd")
        if zc is not None and zc.dim() == 2 and int(zc.shape[0]) > 0:
            dim = int(zc.shape[1])
            break
    rmat = None
    if dim > 0:
        pdim = max(8, min(int(proj_dim), dim))
        rmat = torch.randn(dim, pdim, generator=g_setup)

    blocks = [[] for _ in range(dcap + 1)]
    block_member_w = [[] for _ in range(dcap + 1)]
    n_filtered = 0
    for key in sorted(keyed):
        members = keyed[key]
        if len(members) < 2:
            continue
        cap = max(2, int(bucket_cap))
        if len(members) > cap:
            sel = torch.randperm(len(members), generator=g_setup)[:cap].tolist()
            members = [members[j] for j in sorted(sel)]
        if rmat is not None and n_filtered < int(max_filter_buckets):
            n_filtered += 1
            members = _refine_block(fit, members, rmat, float(sim_thresh), float(ans_max_sim))
        if len(members) < 2:
            continue
        d = int(key[2])
        blocks[d].append(torch.tensor([m[0] for m in members], dtype=torch.long))
        block_member_w[d].append(
            torch.tensor([1.0 + float(rcount[m[0]]) for m in members], dtype=torch.float))

    block_sel_w = [None] * (dcap + 1)
    for d in range(dcap + 1):
        if blocks[d]:
            block_sel_w[d] = torch.tensor([float(b.numel()) ** 0.5 for b in blocks[d]],
                                          dtype=torch.float)

    stratum_idx = [torch.tensor(x, dtype=torch.long) for x in strata]
    perms = [None] * (dcap + 1)
    pos = [0] * (dcap + 1)

    def pull(d):
        if sizes[d] == 0:
            return int(torch.randint(0, n, (1,), generator=g).item())
        p = perms[d]
        if p is None or pos[d] >= int(p.numel()):
            perms[d] = torch.randperm(sizes[d], generator=g)
            pos[d] = 0
            p = perms[d]
        v = int(stratum_idx[d][int(p[pos[d]])])
        pos[d] += 1
        return v

    n_base = max(0, min(bs, int(round(bs * float(base_frac)))))
    n_str = bs - n_base
    gsize = max(2, int(group_size))
    emp = torch.tensor([float(x) for x in sizes], dtype=torch.float)
    lev = torch.arange(dcap + 1, dtype=torch.float)
    floor_n = max(2, int(min_stratum))
    caps = []
    for d in range(dcap + 1):
        if sizes[d] < floor_n:
            caps.append(0)
        elif d == 0:
            caps.append(min(sizes[d], bs))
        else:
            room = int(float(max_repeat) * float(bs) * float(sizes[d]) / float(n))
            caps.append(min(sizes[d], bs, max(1, room)))

    def next_batch(step, total_steps):
        ts = max(1, int(total_steps))
        ramp = max(1, int(float(ramp_frac) * float(ts)))
        r = min(1.0, float(step) / float(ramp))

        counts = [0] * (dcap + 1)
        if n_str > 0:
            w = emp * torch.exp(lev * float(depth_beta) * r)
            tot = float(w.sum())
            if tot <= 0.0:
                counts[0] = n_str
            else:
                p = (w / tot) * float(n_str)
                pl = p.tolist()
                counts = [int(math.floor(x)) for x in pl]
                rem = n_str - sum(counts)
                order = sorted(range(dcap + 1), key=lambda d: (-(pl[d] - counts[d]), -d))
                for k in range(max(0, rem)):
                    counts[order[k % len(order)]] += 1

        surplus = 0
        for d in range(dcap + 1):
            if counts[d] > caps[d]:
                surplus += counts[d] - caps[d]
                counts[d] = caps[d]
        for d in range(dcap, -1, -1):
            if surplus <= 0:
                break
            room = caps[d] - counts[d]
            if room > 0:
                take = min(room, surplus)
                counts[d] += take
                surplus -= take

        block_frac = float(block_frac_max) * r
        out = []
        used = set()
        for d in range(dcap, -1, -1):
            need = counts[d]
            if need <= 0:
                continue
            got = 0
            attempts = 0
            limit = 4 * need + 16
            while got < need and attempts < limit:
                attempts += 1
                use_block = bool(blocks[d]) and \
                    float(torch.rand((1,), generator=g).item()) < block_frac
                if use_block:
                    bi = int(torch.multinomial(block_sel_w[d], 1, generator=g).item())
                    mem = blocks[d][bi]
                    k = min(gsize, int(mem.numel()), need - got)
                    pick = torch.multinomial(block_member_w[d][bi], k, replacement=False,
                                             generator=g)
                    for j in pick.tolist():
                        v = int(mem[j])
                        if v in used:
                            continue
                        out.append(v)
                        used.add(v)
                        got += 1
                        if got >= need:
                            break
                else:
                    v = pull(d)
                    if v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    got += 1

        tries = 0
        while len(out) < bs:
            v = int(torch.randint(0, n, (1,), generator=g).item())
            tries += 1
            if v in used and tries < 8 * bs:
                continue
            out.append(v)
            used.add(v)
        out = out[:bs]
        order = torch.randperm(bs, generator=g).tolist()
        return [out[i] for i in order]

    return next_batch
