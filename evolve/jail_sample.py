"""Write a sample of real training trajectories for inventors to read inside their jail.

    python -m evolve.jail_sample <data-root> [<n-per-image>] [<out-path>]

The sample is drawn from the root's train split only. Training images and the two held-out
val splits are disjoint by construction (evolve/splits.py splits val by image), so a jail
carrying this file learns nothing about anything that is scored.

Records are copied verbatim. The point is that a mechanical question about the data -- does a
step carry an observation, what does a move's observation contain, where does the identity of
the moved file appear -- is answerable by looking, rather than inferred from a sibling impl.
"""
import collections
import json
import pathlib
import random
import sys

DEFAULT_OUT = "evolve/jail_data/train_sample.jsonl"
DEFAULT_PER_IMAGE = 3
BYTE_CAP = 5_000_000
SEED = 20260810


def sample(root, per_image=DEFAULT_PER_IMAGE, out_path=DEFAULT_OUT):
    src = pathlib.Path(root) / "train.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} does not exist")

    by_image = collections.defaultdict(list)
    for line in src.open():
        by_image[json.loads(line)["image"]].append(line)

    rng = random.Random(SEED)
    picked = []
    for image in sorted(by_image):
        lines = by_image[image]
        picked.extend(rng.sample(lines, min(per_image, len(lines))))
    rng.shuffle(picked)

    kept, total = [], 0
    for line in picked:
        if total + len(line) > BYTE_CAP:
            continue
        kept.append(line)
        total += len(line)

    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for line in kept:
            fh.write(line if line.endswith("\n") else line + "\n")

    images = collections.Counter(json.loads(l)["image"] for l in kept)
    steps = sum(len(json.loads(l)["steps"]) for l in kept)
    print(f"wrote {out} — {len(kept)} trajectories, {steps} steps, {total / 1e6:.2f} MB")
    print(f"images: {dict(sorted(images.items()))}")
    if len(kept) < len(picked):
        print(f"dropped {len(picked) - len(kept)} trajectories to stay under "
              f"{BYTE_CAP / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(sample(*sys.argv[1:]))
