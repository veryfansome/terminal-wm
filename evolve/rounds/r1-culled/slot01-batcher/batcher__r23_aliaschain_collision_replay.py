import torch

NAME = "r23_aliaschain_collision_replay"
DESCRIPTION = (
    "Batch composition driven by per-trajectory alias-chain statistics parsed from the command "
    "strings: a trajectory's weight is how much content it carries across mutating hops (reads "
    "whose argument has already passed through k>=2 mutations), and a ramped fraction of every "
    "batch is filled with path-collision blocks — trajectories that share a rare argument token "
    "with the anchor, or (one arm in four) the anchor's system — so the in-batch negative pool "
    "holds same-path different-content rows instead of unrelated ones."
)

_MUT_VERBS = frozenset((
    "mv", "cp", "rm", "rmdir", "mkdir", "touch", "ln", "install", "rename",
    "chmod", "chown", "chgrp", "truncate", "dd", "tee", "tar", "unzip",
    "gzip", "gunzip", "sed", "patch", "shred", "mktemp", "cpio", "rsync",
))

_SEPS = frozenset((";", "&&", "||", "|"))
_SKIP = frozenset(("<", "(", ")", "{", "}", "&", ">", ">>", "2>", "2>&1"))


def _parts(cmd):
    s = str(cmd)
    for ch in (";", "|", "(", ")"):
        s = s.replace(ch, " " + ch + " ")
    return s.split()


def _is_mut(parts):
    expect_verb = True
    for p in parts:
        if p in _SEPS:
            expect_verb = True
            continue
        if p.startswith(">"):
            return True
        if expect_verb:
            if p in _MUT_VERBS:
                return True
            expect_verb = False
    return False


def _keys(parts):
    out = []
    expect_verb = True
    for p in parts:
        if p in _SEPS:
            expect_verb = True
            continue
        if p in _SKIP:
            continue
        if expect_verb:
            expect_verb = False
            continue
        if not p or p[0] == "-":
            continue
        t = p.strip("'\"`").lstrip(">")
        if not t or t in _SKIP:
            continue
        out.append(t)
        b = t.rstrip("/").rsplit("/", 1)[-1]
        if b and b != t:
            out.append(b)
    return out


def _chain_stats(cmds, carry_floor):
    reached = {}
    keyset = set()
    max_depth = 0
    carry = 0.0
    for cmd in cmds:
        parts = _parts(cmd)
        ks = _keys(parts)
        if not ks:
            continue
        keyset.update(ks)
        depth = 0
        for k in ks:
            v = reached.get(k)
            if v is not None and v > depth:
                depth = v
        if depth > max_depth:
            max_depth = depth
        mut = _is_mut(parts)
        if mut:
            nxt = depth + 1
            for k in ks:
                if reached.get(k, -1) < nxt:
                    reached[k] = nxt
        elif depth >= carry_floor:
            carry += float(depth - carry_floor + 1)
    return max_depth, carry, keyset


def make_batcher(
    fit,
    bs,
    seed,
    deep_frac_max=0.5,
    block_frac_max=0.5,
    ramp_frac=0.3,
    block_size=8,
    img_frac=0.25,
    alpha=2.0,
    score_cap=8.0,
    df_cap_frac=0.2,
    carry_floor=2,
):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def uniform_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    if n < 2 or bs < 1:
        return uniform_batch

    scores = torch.zeros(n, dtype=torch.float32)
    sigs = []
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        md, carry, keyset = _chain_stats(cmds, int(carry_floor))
        scores[i] = carry + 0.5 * float(md)
        sigs.append(keyset)

    w = (1.0 + scores.clamp(min=0.0, max=float(score_cap))) ** float(alpha)
    p_deep = w / w.sum().clamp_min(1e-12)

    df_cap = max(2, int(float(df_cap_frac) * n))
    docs = {}
    for i, keyset in enumerate(sigs):
        for k in keyset:
            docs.setdefault(k, []).append(i)

    groups = []
    seq_groups = [[] for _ in range(n)]
    for k in sorted(docs):
        members = docs[k]
        if 2 <= len(members) <= df_cap:
            gi = len(groups)
            groups.append(torch.tensor(members, dtype=torch.long))
            for m in members:
                seq_groups[m].append(gi)

    img_of = [str(s.get("image", "?")) for s in fit]
    img_members = {}
    for i, im in enumerate(img_of):
        img_members.setdefault(im, []).append(i)
    img_pools = {im: torch.tensor(v, dtype=torch.long) for im, v in img_members.items()}
    seq_img = [img_pools[img_of[i]] for i in range(n)]

    have_blocks = bool(groups) or any(p.numel() >= 2 for p in img_pools.values())
    if not have_blocks:
        block_frac_max = 0.0

    def pool_for(anchor):
        gl = seq_groups[anchor]
        take_img = (not gl) or (float(torch.rand((), generator=g)) < float(img_frac))
        if take_img:
            pool = seq_img[anchor]
            if pool.numel() >= 2:
                return pool
            if not gl:
                return None
        j = int(torch.randint(0, len(gl), (1,), generator=g))
        return groups[gl[j]]

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        ramp = min(1.0, float(step) / ramp_steps)

        n_block = int(round(bs * float(block_frac_max) * ramp))
        n_block = max(0, min(bs, n_block))
        rem = bs - n_block
        n_deep = int(round(rem * float(deep_frac_max) * ramp))
        n_deep = max(0, min(rem, n_deep))
        n_uni = bs - n_block - n_deep

        parts = []
        if n_uni > 0:
            parts.append(torch.randint(0, n, (n_uni,), generator=g))
        if n_deep > 0:
            parts.append(torch.multinomial(p_deep, n_deep, replacement=True, generator=g))

        filled = 0
        while filled < n_block:
            m = min(int(block_size), n_block - filled)
            anchor = int(torch.multinomial(p_deep, 1, replacement=True, generator=g))
            chosen = [torch.full((1,), anchor, dtype=torch.long)]
            need = m - 1
            if need > 0:
                pool = pool_for(anchor)
                if pool is not None and pool.numel() >= 1:
                    k = min(need, pool.numel())
                    order = torch.randperm(pool.numel(), generator=g)[:k]
                    chosen.append(pool[order])
                    need -= k
                if need > 0:
                    chosen.append(torch.randint(0, n, (need,), generator=g))
            parts.append(torch.cat(chosen))
            filled += m

        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
