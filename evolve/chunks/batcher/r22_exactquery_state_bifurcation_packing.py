'''R22 batcher: EXACT-QUERY STATE-BIFURCATION PACKING.

An earlier diagnostic isolated command decoding behind a history-presence gate as the
dominant failure mode. This batcher makes command decoding insufficient during ordinary causal
training: it co-packs distinct fit trajectories from the SAME image and cwd that issue the
EXACT same cat/ls command but have close-yet-distinct observation targets. Image, cwd, and
command are held fixed; only preceding trajectory/state can explain which target is right.

Mining is fit-only and read-only. For each (image, cwd, raw command) group, retain distinct-
sequence target pairs with cosine in [min_target_cos, max_target_cos]. The upper threshold
removes near-duplicate false negatives; the lower threshold keeps the pair in the
antiretrieval-ring objective's confusable regime. At training time, preserve the sysblock
uniform-to-image-blocked hard curriculum, but fill the hard block round-robin from a few of
these state-bifurcation groups. Unfilled slots use the ordinary selected-image pool.

The module changes only batch indices. It adds no tokens, targets, parameters, forward
branches, or eval behavior. Anti-collapse and causality remain those of the unchanged
objective/model. Construction and sampling use no global RNG; all sampling uses one private
torch.Generator.
'''

import collections
import posixpath
import shlex

import torch
import torch.nn.functional as F

NAME = 'r22_exactquery_state_bifurcation_packing'
DESCRIPTION = (
    'Ramped same-image hard batches packed with distinct trajectories that hold image, cwd, '
    'and exact cat/ls command fixed while their targets are close-but-distinct, forcing the '
    'unchanged contrastive objective to discriminate filesystem state from history rather '
    'than command text; uniform remainder and image-pool fallback preserve broad coverage.'
)

_QUERY_VERBS = frozenset({'cat', 'ls'})


def _tokens(cmd):
    try:
        return shlex.split(str(cmd))
    except Exception:
        return str(cmd).split()


def _advance_cwd(cmd, cwd):
    '''Track the collection environment's persistent cd state.

    cd emits an empty observation, so the cached `ok` bit is false even on success; update
    from command syntax rather than that bit. The collection policy draws valid cd targets.
    '''
    toks = _tokens(cmd)
    if not toks or toks[0] != 'cd':
        return cwd
    args = [x for x in toks[1:] if not x.startswith('-')]
    path = args[0] if args else '/root'
    if path.startswith('~'):
        path = '/root' + path[1:]
    if not path.startswith('/'):
        path = posixpath.join(cwd, path)
    return posixpath.normpath(path)


def _mine_state_groups(fit, min_cos, max_cos, max_entries):
    '''Return image pools and exact-query divergent-target pair groups.

    Each pair tensor is [E,2] of distinct fit-sequence indices. Pair identities are
    deduplicated so repeated executions of one query do not multiply its sampling weight.
    All target geometry is detached and moved to CPU during this one-time construction.
    '''
    raw = collections.defaultdict(list)
    image_members = collections.defaultdict(list)

    for i, seq in enumerate(fit):
        image = seq.get('image', '?')
        image_members[image].append(i)
        cmds = seq.get('cmds') or ()
        oks = seq.get('ok')
        if oks is None:
            oks = [True] * len(cmds)
        z_obs = seq.get('z_obs')
        cwd = '/'

        for t, cmd in enumerate(cmds):
            toks = _tokens(cmd)
            ok = bool(oks[t]) if t < len(oks) else True
            if (ok and toks and toks[0] in _QUERY_VERBS and
                    torch.is_tensor(z_obs) and t < z_obs.shape[0]):
                key = (image, cwd, str(cmd).strip())
                target = z_obs[t].detach().float().cpu()
                raw[key].append((i, target))
            cwd = _advance_cwd(cmd, cwd)

    by_image = collections.defaultdict(list)
    cap = max(2, int(max_entries))

    for key, entries in sorted(raw.items(), key=lambda kv: repr(kv[0])):
        if len(entries) < 2:
            continue
        if len(entries) > cap:
            # Deterministic coverage of a giant common query without quadratic blow-up.
            take = torch.linspace(0, len(entries) - 1, cap).round().long().unique().tolist()
            entries = [entries[j] for j in take]

        seq_ids = torch.tensor([x[0] for x in entries], dtype=torch.long)
        targets = torch.stack([x[1] for x in entries])
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1e4, neginf=-1e4)
        unit = F.normalize(targets, dim=-1)
        cosine = unit @ unit.t()

        n = len(entries)
        upper = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        keep = upper & (seq_ids[:, None] != seq_ids[None, :])
        keep = keep & torch.isfinite(cosine)
        keep = keep & (cosine >= float(min_cos)) & (cosine <= float(max_cos))
        a, b = keep.nonzero(as_tuple=True)
        if a.numel() == 0:
            continue

        pairs = torch.stack([seq_ids[a], seq_ids[b]], dim=1).sort(dim=1).values
        pairs = torch.unique(pairs, dim=0)
        if pairs.numel():
            by_image[key[0]].append(pairs)

    names = sorted(image_members)
    pools = [torch.tensor(image_members[name], dtype=torch.long) for name in names]
    sizes = torch.tensor([float(pool.numel()) for pool in pools])
    return names, pools, sizes, by_image


def make_batcher(
    fit,
    bs,
    seed,
    n_block_images=1,
    hard_frac_max=0.75,
    ramp_frac=0.30,
    max_query_groups=8,
    min_target_cos=0.20,
    max_target_cos=0.95,
    group_weight_pow=0.5,
    max_entries_per_query=256,
    pair_retry=8,
):
    n = len(fit)
    bs = int(bs)
    if n <= 0 or bs <= 0:
        raise ValueError('batcher requires non-empty fit and positive bs')
    if not (float(min_target_cos) < float(max_target_cos)):
        raise ValueError('min_target_cos must be below max_target_cos')

    g = torch.Generator().manual_seed(int(seed))
    names, pools, sizes, by_image = _mine_state_groups(
        fit,
        float(min_target_cos),
        float(max_target_cos),
        max_entries_per_query,
    )
    if not names:
        raise ValueError('fit has no image groups')

    k_images = max(1, min(int(n_block_images), len(pools)))
    group_pow = max(0.0, float(group_weight_pow))
    query_cap = max(1, int(max_query_groups))
    retries = max(1, int(pair_retry))

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        frac = float(hard_frac_max) * min(1.0, float(step) / float(ramp_steps))
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_images, replacement=False, generator=g)
            block_pool = torch.cat([pools[int(x)] for x in chosen])
            query_groups = []
            for x in chosen.tolist():
                query_groups.extend(by_image.get(names[int(x)], ()))

            hard = []
            used = set()
            if query_groups and n_hard >= 2:
                q = min(query_cap, len(query_groups), max(1, n_hard // 2))
                weights = torch.tensor([
                    float(group.shape[0]) ** group_pow for group in query_groups
                ])
                selected = torch.multinomial(weights, q, replacement=False, generator=g).tolist()

                cursor = 0
                while len(hard) + 2 <= n_hard:
                    edges = query_groups[selected[cursor % q]]
                    pair = None
                    for _ in range(retries):
                        row = edges[int(torch.randint(0, edges.shape[0], (1,), generator=g))]
                        a, b = int(row[0]), int(row[1])
                        pair = (a, b)
                        if a not in used and b not in used:
                            break
                    hard.extend(pair)
                    used.update(pair)
                    cursor += 1

            # Odd slots, sparse groups, or complete fallback use the sysblock-style image pool.
            if len(hard) < n_hard:
                fill = block_pool[torch.randint(
                    0, block_pool.numel(), (n_hard - len(hard),), generator=g
                )]
                hard.extend(fill.tolist())
            parts.append(torch.tensor(hard[:n_hard], dtype=torch.long))

        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
