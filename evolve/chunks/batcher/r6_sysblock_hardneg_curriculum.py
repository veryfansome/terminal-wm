"""batcher chunk: system-blocked hard-negative curriculum.

Anneal from uniform batches to batches whose hard fraction is drawn from only a few Docker
images, so the in-batch negatives of the contrastive objectives become same-system
same-verb observations — the foil type the fitness metric actually ranks against.

WHY (mechanism): the eval's same-verb foils are drawn within the inner-val pool (two
images), so most foils share the query's system; discriminating them requires path/content-
level dynamics, not system identity. Uniform batches make most training negatives
cross-image, which a system-identity shortcut separates (cheap, transfer-irrelevant).
Blocking part of the batch onto n_block_images images raises the same-system negative
density inside every loss computation. The remainder of the batch stays uniform (a hard/easy
MIX, not all-hard — the ratio, not maximal hardness, is the operative knob), and the hard
fraction ramps in over the first ramp_frac of training so early optimization is stable.

Anti-collapse: unchanged from the objective — a constant prediction still yields uniform
softmax rows (loss log(n)) under the contrastive objectives regardless of composition; this
module never touches predictions or targets.

Degenerate-data safety: with fewer than 2 distinct images in fit (or hard_frac_max <= 0)
it falls back to exact uniform sampling. All randomness flows through one private,
seed-derived torch.Generator (deterministic per seed; global RNG untouched).
"""

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

    # Group train sequence indices by image (read-only metadata; sorted for determinism).
    groups = {}
    for i, s in enumerate(fit):
        groups.setdefault(s.get("image", "?"), []).append(i)
    pools = [torch.tensor(groups[k], dtype=torch.long) for k in sorted(groups)]
    sizes = torch.tensor([float(p.numel()) for p in pools])

    if len(pools) < 2 or hard_frac_max <= 0.0:
        # Degenerate: nothing to block on — exact uniform fallback.
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    k_imgs = max(1, min(int(n_block_images), len(pools)))

    def next_batch(step, total_steps):
        # Curriculum: hard fraction ramps 0 -> hard_frac_max over the first ramp_frac of
        # training, then holds. Proxy (1000 steps, ramp_frac=0.4) finishes the ramp at 400.
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        parts = [torch.randint(0, n, (bs - n_hard,), generator=g)]
        if n_hard > 0:
            # Pick a few images (size-weighted, without replacement), then draw the hard
            # portion uniformly from the union of their sequences.
            chosen = torch.multinomial(sizes, k_imgs, replacement=False, generator=g)
            pool = torch.cat([pools[int(c)] for c in chosen])
            parts.append(pool[torch.randint(0, pool.numel(), (n_hard,), generator=g)])
        batch = torch.cat(parts)
        return batch[torch.randperm(bs, generator=g)].tolist()  # shuffle hard/easy mix

    return next_batch
