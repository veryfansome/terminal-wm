import torch

NAME = "r4_distinct_blocked_transport_sampler"
DESCRIPTION = (
    "Image-blocked batch composition drawn WITHOUT replacement, so no sequence can occupy two "
    "slots of the same batch: a ramped fraction of every batch is taken from one sampled image "
    "pool and the remainder from the whole split, with the already-taken indices excluded. The "
    "draw is a weighted reservoir (Efraimidis-Spirakis exponential keys) whose per-sequence "
    "weight is a ramped power of a transport mass read off the command strings alone -- for each "
    "single-argument `cat` of a path whose current occupant arrived via `mv` or a redirect, the "
    "SQUARE of the capped number of hops that occupant has taken, summed over the sequence and "
    "normalized by the split mean."
)


def _transport_mass(cmds, hop_cap):
    depth_at = {}
    mass = 0.0
    for c in cmds:
        parts = c.split()
        if len(parts) == 3 and parts[0] == "mv":
            depth_at[parts[2]] = depth_at.pop(parts[1], 0) + 1
        elif len(parts) == 4 and parts[0] == "cat" and parts[2] in (">", ">>"):
            depth_at[parts[3]] = depth_at.get(parts[1], 0) + 1
        elif len(parts) == 2 and parts[0] == "cat":
            d = depth_at.get(parts[1])
            if d is None:
                depth_at[parts[1]] = 0
            elif d > 0:
                hops = float(min(d, hop_cap))
                mass += hops * hops
    return mass


def make_batcher(fit, bs, seed, n_block_images=1, hard_frac_max=0.75, ramp_frac=0.3,
                 chain_alpha=1.0, hop_cap=6, weight_cap=4.0):
    n = len(fit)
    bs = int(bs)
    g = torch.Generator().manual_seed(seed)

    if n < 1 or bs < 1:
        def next_batch_degenerate(step, total_steps):
            return torch.randint(0, max(1, n), (bs,), generator=g).tolist()
        return next_batch_degenerate

    mass = torch.tensor(
        [_transport_mass(list(s.get("cmds") or []), int(hop_cap)) for s in fit],
        dtype=torch.float,
    )
    ref = float(mass.mean())
    hi = float(weight_cap)
    if ref > 0.0 and hi > 1.0:
        base_w = (mass / ref).clamp(1.0 / hi, hi)
    else:
        base_w = torch.ones(n, dtype=torch.float)

    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])
    k_imgs = max(1, min(int(n_block_images), len(pools)))
    blocked_live = len(pools) >= 2 and float(hard_frac_max) > 0.0

    if n <= bs:
        def next_batch_small(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch_small

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(float(ramp_frac) * max(1, int(total_steps))))
        r = min(1.0, float(step) / float(ramp_steps))
        w = base_w.pow(float(chain_alpha) * r).clamp_min(1e-6)
        u = torch.rand(n, generator=g).clamp_min(1e-12)
        key = torch.log(u) / w

        n_hard = 0
        if blocked_live:
            n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * r))))

        taken = []
        got = 0
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            k1 = min(n_hard, int(pool.numel()))
            if k1 > 0:
                sel = pool[torch.topk(key[pool], k1).indices]
                taken.append(sel)
                got += int(sel.numel())
                key[sel] = float("-inf")
        rest = bs - got
        if rest > 0:
            taken.append(torch.topk(key, rest).indices)

        batch = torch.cat(taken)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
