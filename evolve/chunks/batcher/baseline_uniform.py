"""batcher chunk baseline: uniform iid sequence sampling — BIT-IDENTICAL to the historical
harness behavior (same torch.Generator seeding, same randint call, same call order), so a
genome with {"batcher": {"impl": "baseline_uniform"}} — or no batcher chunk at all — must
reproduce every archived fitness exactly. This is the plumbing check for the new axis.

Contract for any batcher impl:
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


def make_batcher(fit, bs, seed):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)  # same seeding the harness used

    def next_batch(step, total_steps):
        # identical call, identical generator state trajectory => identical index stream
        return torch.randint(0, n, (bs,), generator=g).tolist()

    return next_batch
