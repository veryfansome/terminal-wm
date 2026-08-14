import torch

NAME = "r23_placefield_chainyield_replay"
DESCRIPTION = (
    "Place-field replay batcher: batches are assembled from a few PLACE CLIQUES — sets of train "
    "sequences whose commands touch the same path token (file/dir), optionally narrowed to one "
    "system — with clique size band-passed so cliques are neither singletons nor the whole corpus, "
    "and members drawn by a chain-yield priority (path revisits after a mutation, and the number of "
    "(mutation-chain, later-read) pairs the trunk task can mine). The remaining slots come from a "
    "shuffled without-replacement epoch cursor, so every sequence is covered and a batch holds "
    "distinct sequences."
)

_MUT_VERBS = frozenset({
    "mv", "cp", "rm", "rmdir", "mkdir", "touch", "ln", "install", "rename",
    "chmod", "chown", "truncate", "dd", "tee", "shred", "unlink", "sed",
    "mktemp", "cpio", "rsync",
})

_STOP = frozenset({"", ".", "..", "/", "*", "~", "-", "&&", "||", "|", ";", "&", ">", ">>", "<"})

_STRIP = "\"'`,;()[]{}"


def _verb_of(cmd):
    parts = cmd.split()
    if not parts:
        return ""
    v = parts[0]
    if "/" in v:
        v = v.rsplit("/", 1)[1]
    return v


def _place_tokens(cmd):
    out = set()
    parts = cmd.split()
    for raw in parts[1:]:
        a = raw.strip(_STRIP).rstrip("/")
        if not a or a.startswith("-") or a in _STOP:
            continue
        out.add(a)
        if "/" in a:
            head, base = a.rsplit("/", 1)
            if base and base not in _STOP:
                out.add(base)
            if head and head not in _STOP:
                out.add(head)
    return out


def _sequence_stats(cmds, max_hops):
    n = len(cmds)
    verbs = [_verb_of(c) for c in cmds]
    mut = [v in _MUT_VERBS for v in verbs]
    toks = [_place_tokens(c) for c in cmds]

    mut_prefix = [0] * (n + 1)
    for i in range(n):
        mut_prefix[i + 1] = mut_prefix[i] + (1 if mut[i] else 0)

    first_at = {}
    carry = 0
    for j in range(n):
        earliest = None
        for t in toks[j]:
            k = first_at.get(t)
            if k is not None and (earliest is None or k < earliest):
                earliest = k
        if earliest is not None and (mut_prefix[j] - mut_prefix[earliest + 1]) > 0:
            carry += 1
        for t in toks[j]:
            if t not in first_at:
                first_at[t] = j

    n_mut = sum(1 for i in range(1, n) if mut[i])
    first_mut = None
    for i in range(1, n):
        if mut[i]:
            first_mut = i
            break
    if first_mut is None:
        n_read = 0
    else:
        n_read = sum(1 for j in range(first_mut + 1, n) if not mut[j])
    chain = min(n_mut, max_hops) * n_read

    union = set()
    for t in toks:
        union |= t
    return carry, chain, union


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, n_cliques=6,
                 clique_cap=8, sys_frac=0.5, gamma_chain=0.25, pri_pow=1.0,
                 tok_pow=1.0, max_hops=3):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))
    bs = int(bs)

    pri_list = []
    tok_owners = {}
    img_ids = {}
    img_of = []
    for i, s in enumerate(fit):
        cmds = list(s.get("cmds", []) or [])
        carry, chain, union = _sequence_stats(cmds, int(max_hops))
        pri_list.append(1.0 + float(carry) + float(gamma_chain) * float(chain))
        for t in union:
            tok_owners.setdefault(t, []).append(i)
        im = s.get("image", "?")
        if im not in img_ids:
            img_ids[im] = len(img_ids)
        img_of.append(img_ids[im])

    pri = torch.tensor(pri_list, dtype=torch.float).clamp_min(1e-3)
    pri = pri.pow(float(pri_pow)).clamp_min(1e-6)
    img_of_t = torch.tensor(img_of, dtype=torch.long)

    cap = max(2, int(clique_cap))
    max_owners = max(4 * cap, 16)
    owner_lists = []
    tok_weights = []
    for t in sorted(tok_owners):
        owners = tok_owners[t]
        c = len(owners)
        if c < 2 or c >= n or c > max_owners:
            continue
        idx = torch.tensor(owners, dtype=torch.long)
        band = float(c) if c <= cap else float(cap) * float(cap) / float(c)
        owner_lists.append(idx)
        tok_weights.append(band * float(pri[idx].mean().item()))

    have_cliques = len(owner_lists) >= 1
    if have_cliques:
        tok_w = torch.tensor(tok_weights, dtype=torch.float).clamp_min(1e-6).pow(float(tok_pow))
        tok_cdf = torch.cumsum(tok_w, dim=0)
        tok_total = float(tok_cdf[-1].item())
    else:
        tok_cdf = None
        tok_total = 0.0

    img_pools = {}
    for i, k in enumerate(img_of):
        img_pools.setdefault(k, []).append(i)
    img_pool_list = [torch.tensor(v, dtype=torch.long) for k, v in sorted(img_pools.items())]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pool_list], dtype=torch.float)
    have_images = len(img_pool_list) >= 2

    state = {"perm": [], "ptr": 0}

    def _cursor():
        if state["ptr"] >= len(state["perm"]):
            state["perm"] = torch.randperm(n, generator=g).tolist()
            state["ptr"] = 0
        v = state["perm"][state["ptr"]]
        state["ptr"] += 1
        return v

    def _draw_pool(pool, k):
        k = min(int(k), int(pool.numel()))
        if k <= 0:
            return []
        w = pri[pool]
        sel = torch.multinomial(w, k, replacement=False, generator=g)
        return pool[sel].tolist()

    def _clique_pool():
        u = float(torch.rand(1, generator=g).item()) * tok_total
        ti = int(torch.searchsorted(tok_cdf, torch.tensor([u]), right=True).item())
        ti = min(ti, len(owner_lists) - 1)
        pool = owner_lists[ti]
        if pool.numel() > 2 and float(torch.rand(1, generator=g).item()) < float(sys_frac):
            anchor = int(pool[int(torch.randint(0, pool.numel(), (1,), generator=g).item())].item())
            sub = pool[img_of_t[pool] == img_of_t[anchor]]
            if sub.numel() >= 2:
                pool = sub
        return pool

    def _image_pool():
        ii = int(torch.multinomial(img_sizes, 1, generator=g).item())
        return img_pool_list[ii]

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp_steps))
        n_hard = max(0, min(bs, int(round(bs * frac))))

        chosen = []
        seen = set()
        if n_hard > 0 and (have_cliques or have_images):
            budget = max(1, int(n_cliques)) * 4 + 8
            tries = 0
            while len(chosen) < n_hard and tries < budget:
                tries += 1
                pool = _clique_pool() if have_cliques else _image_pool()
                want = min(cap, n_hard - len(chosen))
                for c in _draw_pool(pool, want):
                    if c not in seen:
                        seen.add(c)
                        chosen.append(c)
                    if len(chosen) >= n_hard:
                        break

        guard = 0
        while len(chosen) < bs and guard < 4 * (n + bs) + 8:
            guard += 1
            i = _cursor()
            if i in seen and len(seen) < n:
                continue
            seen.add(i)
            chosen.append(i)
        while len(chosen) < bs:
            chosen.append(int(torch.randint(0, n, (1,), generator=g).item()))

        batch = torch.tensor(chosen[:bs], dtype=torch.long)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
