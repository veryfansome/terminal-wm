import torch

NAME = "r20_depthtilted_sysblock_curriculum"
DESCRIPTION = (
    "Annealed image-blocked batches whose blocked part is additionally tilted toward "
    "sequences with long rename chains. For every train sequence the impl reads its command "
    "strings, follows two-argument `mv src dst` renames as a forwarding graph (dst inherits "
    "src's hop count, src is cleared), and records the longest hop count reached. A single ramp "
    "drives both the blocked fraction of the batch and the exponent of the per-sequence weight "
    "(1 + chain_length)^(pow * ramp), so batches start uniform and end concentrated on the "
    "long-chain tail of one or a few images. The uniform remainder is kept for coverage. All "
    "draws come from a private torch.Generator; the sequences are only read."
)


def _chain_length(cmds):
    reach = {}
    best = 0
    for c in cmds:
        if not isinstance(c, str):
            continue
        parts = c.split()
        if len(parts) != 3 or parts[0] != "mv":
            continue
        src, dst = parts[1], parts[2]
        if src.startswith("-") or dst.startswith("-"):
            continue
        hops = reach.pop(src, 0) + 1
        if hops > reach.get(dst, 0):
            reach[dst] = hops
        if reach[dst] > best:
            best = reach[dst]
    return best


def make_batcher(fit, bs, seed, n_block_images=1, hard_frac_max=0.75, ramp_frac=0.3,
                 depth_pow=2.0, depth_base=1.0):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    if n < 1 or len(pools) < 2 or hard_frac_max <= 0.0:
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    chain = torch.tensor(
        [float(_chain_length(s.get("cmds") or ())) for s in fit], dtype=torch.float
    )
    base = (float(depth_base) + chain).clamp_min(1e-3)

    k_imgs = max(1, min(int(n_block_images), len(pools)))
    pw = float(depth_pow)

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        r = min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * hard_frac_max * r))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            w = base[pool].pow(pw * r)
            w = torch.where(torch.isfinite(w) & (w > 0.0), w, torch.full_like(w, 1e-6))
            pick = torch.multinomial(w, n_hard, replacement=True, generator=g)
            parts.append(pool[pick])
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
