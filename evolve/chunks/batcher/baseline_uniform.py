"""Contract for any batcher impl:
    make_batcher(fit, bs, seed, **params) -> next_batch(step, total_steps) -> list[int]
      - fit: the train-split sequence dicts (read-only; may use metadata like s["image"]).
      - returns exactly bs integer indices in [0, len(fit)); called once per training step
        with step in [1, total_steps].
      - must be deterministic given seed, must not mutate fit, and must not touch the
        global torch RNG (own a private torch.Generator).
"""

import torch

NAME = "baseline_uniform"
DESCRIPTION = "Uniform iid sequence sampling; bit-identical to the pre-axis harness RNG stream."


# This impl must stay bit-identical to the pre-axis harness: same generator seeding, same
# randint call, same call order, so archived fitnesses replay exactly.
def make_batcher(fit, bs, seed):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)

    def next_batch(step, total_steps):
        return torch.randint(0, n, (bs,), generator=g).tolist()

    return next_batch
