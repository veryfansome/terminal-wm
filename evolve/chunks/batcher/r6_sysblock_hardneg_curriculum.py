import torch

NAME = "r6_sysblock_hardneg_curriculum"
DESCRIPTION = (
    "Anneal batch composition from uniform to partially image-blocked (a few systems per "
    "batch), densifying same-system in-batch negatives for the contrastive objectives to "
    "match the eval's same-verb same-system foil geometry."
)


def make_batcher(fit, bs, seed, n_block_images=3, hard_frac_max=0.5, ramp_frac=0.4):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    if len(pools) < 2 or hard_frac_max <= 0.0:
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    k_imgs = max(1, min(int(n_block_images), len(pools)))

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
