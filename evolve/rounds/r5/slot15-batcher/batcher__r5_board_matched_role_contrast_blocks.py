import math
import re

import torch

NAME = "r5_board_matched_role_contrast_blocks"
DESCRIPTION = (
    "Replays each training sequence's command list through a symbolic content tracker that keeps "
    "content IDENTITY, not just hop counts: every read of an untouched path starts a content "
    "thread, every mv/cp/redirect carries a thread to a new path and increments its hop count, and "
    "the directories a thread touches are unioned so a trajectory splits into independent windows. "
    "For each sequence's deepest window the tracker emits two disjoint keys — a BOARD key (how many "
    "threads move, how many moves the window has, the window's maximum hop count, and the "
    "root/arity shape of the read path) holding what a role swap leaves fixed, and a ROLE key (the "
    "read thread's own hop count, its birth rank among the moving threads, and whether it is the "
    "thread that made the window's opening move) holding what a role swap permutes. "
    "Batches are a without-replacement (reshuffled) uniform base plus an annealed block share; a "
    "block is either a whole-system block over a few images or a bucket of sequences sharing one "
    "board key, from which members are drawn round-robin across DISTINCT role keys, so "
    "board-identical, role-differing sequences land in the same in-batch negative pool. Bucket "
    "choice is tilted toward deeper hop counts, larger buckets and more role-diverse buckets; "
    "member choice inside a role is tilted toward sequences with more routed reads. Degrades to "
    "the single-role bucket when a board carries only one role, to whole-system blocks when no "
    "bucket has two members, and to reshuffled uniform sampling when neither exists."
)

_HEX = re.compile(r"[0-9a-fA-F]{6,}")
_NUM = re.compile(r"\d+")
_READ_VERBS = ("cat", "head", "tail")
_DEPTH_CAP = 6
_MOVE_CAP = 16
_PATH_CAP = 12
_ROOT_KEEP = 3


def _norm(text):
    return _NUM.sub("#", _HEX.sub("#", text))


def _shape(path):
    parts = [p for p in path.split("/") if p]
    root = "/".join(_norm(p) for p in parts[:_ROOT_KEEP])
    return (root, min(len(parts), _PATH_CAP))


def _dirname(path):
    i = path.rfind("/")
    if i > 0:
        return path[:i]
    return "/"


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
            rtarget = toks[i + 1] if i + 1 < len(toks) else None
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


def _trace(cmds):
    loc = {}
    hops = []
    birth = []
    touched = []
    moves = []
    reads = []

    def start(path, base, order):
        tid = len(hops)
        hops.append(int(base))
        birth.append(int(order))
        touched.append({_dirname(path)})
        loc[path] = tid
        return tid

    for i, cmd in enumerate(cmds):
        ev = _parse(cmd)
        if ev is None:
            continue
        kind, a, b = ev
        if kind == "move":
            tid = loc.pop(a, None)
            if tid is None:
                tid = start(a, 0, i)
                loc.pop(a, None)
            loc[b] = tid
            hops[tid] += 1
            touched[tid].add(_dirname(a))
            touched[tid].add(_dirname(b))
            moves.append((i, tid))
        elif kind == "copy" or kind == "append":
            src = loc.get(a)
            base = hops[src] if src is not None else 0
            if src is not None:
                touched[src].add(_dirname(a))
            tid = loc.get(b)
            if tid is None:
                tid = start(b, base + 1, i)
            elif base + 1 > hops[tid]:
                hops[tid] = base + 1
            touched[tid].add(_dirname(a))
            touched[tid].add(_dirname(b))
            moves.append((i, tid))
        else:
            tid = loc.get(a)
            if tid is None:
                start(a, 0, i)
            else:
                reads.append((i, tid, hops[tid], a))
    return hops, birth, touched, moves, reads


def _components(touched):
    n = len(touched)
    root = list(range(n))

    def find(x):
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    owner = {}
    for t in range(n):
        for d in touched[t]:
            o = owner.get(d)
            if o is None:
                owner[d] = t
            else:
                ra, rb = find(o), find(t)
                if ra != rb:
                    if ra < rb:
                        root[rb] = ra
                    else:
                        root[ra] = rb
    groups = {}
    for t in range(n):
        groups.setdefault(find(t), []).append(t)
    return groups


def _profile(cmds):
    hops, birth, touched, moves, reads = _trace(cmds)
    if not reads:
        return None, 0.0
    groups = _components(touched)
    home = {}
    for r, members in groups.items():
        for t in members:
            home[t] = r
    demand = 0.0
    best_per_comp = {}
    for (i, t, d, path) in reads:
        if d < 1:
            continue
        demand += float(min(d, _DEPTH_CAP))
        c = home[t]
        cur = best_per_comp.get(c)
        if cur is None or (d, i) > (cur[2], cur[0]):
            best_per_comp[c] = (i, t, d, path)
    if not best_per_comp:
        return None, 0.0
    pick_c = None
    pick_r = None
    for c in sorted(best_per_comp):
        rec = best_per_comp[c]
        if pick_r is None or (rec[2], rec[0]) > (pick_r[2], pick_r[0]):
            pick_c = c
            pick_r = rec
    rt, rd, rpath = pick_r[1], pick_r[2], pick_r[3]
    members = sorted(groups[pick_c])
    comp_moves = [(i, t) for (i, t) in moves if home.get(t) == pick_c]
    movers = [t for t in members if hops[t] >= 1]
    deepest = 0
    for t in movers:
        if min(hops[t], _DEPTH_CAP) > deepest:
            deepest = min(hops[t], _DEPTH_CAP)
    board = (len(movers), min(len(comp_moves), _MOVE_CAP), deepest, _shape(rpath))
    by_birth = sorted(movers, key=lambda t: (birth[t], t))
    birth_rank = by_birth.index(rt) if rt in by_birth else len(by_birth)
    own = [k for k, (i, t) in enumerate(comp_moves) if t == rt]
    opened = 1 if (own and own[0] == 0) else 0
    role = (min(rd, _DEPTH_CAP), min(birth_rank, _MOVE_CAP), opened)
    return (board, role), demand


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, group_size=12,
                 depth_beta=0.8, size_pow=0.75, role_pow=1.5, img_share=0.25,
                 n_block_images=1):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)
    cursor = {"buf": [], "pos": 0}

    def draw_uniform():
        if cursor["pos"] >= len(cursor["buf"]):
            cursor["buf"] = torch.randperm(max(1, n), generator=g).tolist()
            cursor["pos"] = 0
        v = int(cursor["buf"][cursor["pos"]])
        cursor["pos"] += 1
        return v

    def uniform_batch(step, total_steps):
        return [draw_uniform() for _ in range(bs)]

    if n <= 1 or bs <= 0 or float(hard_frac_max) <= 0.0 or int(group_size) < 2:
        return uniform_batch

    demand = [0.0] * n
    keyed = {}
    for i in range(n):
        s = fit[i]
        prof, dm = _profile(s.get("cmds") or [])
        demand[i] = dm
        if prof is None:
            continue
        board, role = prof
        keyed.setdefault(board, {}).setdefault(role, []).append(i)

    bucket_roles = []
    bucket_role_w = []
    weights = []
    for key in sorted(keyed, key=repr):
        rolemap = keyed[key]
        size = 0
        for v in rolemap.values():
            size += len(v)
        if size < 2:
            continue
        role_keys = sorted(rolemap, key=repr)
        deepest = float(key[2])
        bucket_roles.append([torch.tensor(rolemap[rk], dtype=torch.long) for rk in role_keys])
        bucket_role_w.append([
            torch.tensor([1.0 + demand[j] for j in rolemap[rk]], dtype=torch.float)
            for rk in role_keys])
        weights.append((float(size) ** float(size_pow))
                       * math.exp(float(depth_beta) * float(deepest))
                       * (float(len(role_keys)) ** float(role_pow)))

    by_image = {}
    for i in range(n):
        by_image.setdefault(str(fit[i].get("image", "?")), []).append(i)
    img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools], dtype=torch.float)

    have_blocks = len(bucket_roles) > 0
    have_images = len(img_pools) >= 2
    if not have_blocks and not have_images:
        return uniform_batch

    bucket_w = None
    if have_blocks:
        bucket_w = torch.tensor(weights, dtype=torch.float)
        bucket_w = torch.nan_to_num(bucket_w, nan=1.0, posinf=1.0, neginf=0.0).clamp_min(1e-8)

    share = float(img_share)
    if share < 0.0:
        share = 0.0
    if share > 1.0:
        share = 1.0
    if not have_blocks:
        share = 1.0
    if not have_images:
        share = 0.0

    gsize = max(2, int(group_size))
    k_imgs = max(1, min(int(n_block_images), len(img_pools)))
    distinct = n >= bs
    block_cap = 4 * (bs // gsize + 2)
    fill_cap = 4 * bs + 16

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp_steps))
        n_hard = int(round(bs * frac))
        if n_hard < 0:
            n_hard = 0
        if n_hard > bs:
            n_hard = bs

        out = []
        used = set()
        blocks = 0
        while len(out) < n_hard and blocks < block_cap:
            blocks += 1
            k = min(gsize, n_hard - len(out))
            if k <= 0:
                break
            u = float(torch.rand((1,), generator=g).item())
            if u < share:
                chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
                pool = torch.cat([img_pools[int(c)] for c in chosen.tolist()])
                cand = pool[torch.randint(0, int(pool.numel()), (3 * k,), generator=g)].tolist()
                for v in cand:
                    v = int(v)
                    if distinct and v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    if len(out) >= n_hard:
                        break
            else:
                bi = int(torch.multinomial(bucket_w, 1, generator=g)[0])
                members = bucket_roles[bi]
                mweights = bucket_role_w[bi]
                nr = len(members)
                order = torch.randperm(nr, generator=g).tolist()
                taken = 0
                probe = 0
                limit = 3 * k + nr
                while taken < k and probe < limit:
                    ri = order[probe % nr]
                    probe += 1
                    j = int(torch.multinomial(mweights[ri], 1, generator=g)[0])
                    v = int(members[ri][j])
                    if distinct and v in used:
                        continue
                    out.append(v)
                    used.add(v)
                    taken += 1
                    if len(out) >= n_hard:
                        break

        tries = 0
        while len(out) < bs and tries < fill_cap:
            tries += 1
            v = draw_uniform()
            if distinct and v in used:
                continue
            out.append(v)
            used.add(v)
        while len(out) < bs:
            out.append(draw_uniform())

        order = torch.randperm(bs, generator=g).tolist()
        return [out[i] for i in order]

    return next_batch
