"""Build the lane context ONCE, so a campaign loads and derives nothing per candidate.

    python -m cloud.build_context [--split inner] [--out .cache/lane-inner.pt]

Everything in here is a property of (root, split, eye, swap-seed) and NOT of the genome:

  - the encoded train and val splits, standardized in the pack root's own frame
  - the cups windows harvested from the val split, and the interleaved token layout
  - the ROLE-SWAP chains: partner draw, chain synthesis, and the re-encoded mv embeddings

That last one is the expensive part and the reason this exists. Synthesizing and encoding the
alternative chains means loading the text encoder and running it over every rewritten command; done
inside the per-candidate path it is repeated once per (genome, seed) for a result that is bit-wise
identical every time.

The artifact is written with torch's zipfile serialization so workers can memory-map it: N training
processes then share one physical copy of the tensors instead of each loading their own. Processes
rather than threads is deliberate — training seeds the global RNG and dropout draws from it, so
concurrent threads would interleave those draws and a genome+seed would stop meaning one thing.
"""
import argparse
import os
import pathlib
import sys

import torch

from evolve import cdh_probe as CDH, cups_probe as CP, harness as H
from realenv import seq_worldmodel as M


def build(root, eye, split, swap_seed, cdh_root=None):
    device = M.pick_device()

    train_full = H._cached_encode(root, "train", eye, device)
    mo, so, mc, sc = M.standardize_stats(train_full)
    M.apply_stats(train_full, mo, so, mc, sc)

    ctx = CP.load_cups_context(root, split, eye, device, stats_data=root)
    if ctx is None:
        raise SystemExit(f"no cups windows in the {split} split of {root}")

    # the one genuinely expensive, genuinely net-independent step
    swap = CP.build_swap_cache(ctx, eye, device, seed=swap_seed)

    # The cd-history windows, standardized in THIS root's frame. The net is trained here, so its
    # inputs live in this standardization frame; putting the other pack's windows in their own frame
    # would measure a net on inputs it never saw. Harvest and layout are net-independent, so they
    # belong in the shared context like everything else.
    cdh = None
    if cdh_root:
        cdh = CDH.load_cdh_context(cdh_root, split, eye, device, redir_only=True,
                                   stats_data=root)
        if cdh is None:
            raise SystemExit(f"no cd-history windows in the {split} split of {cdh_root}")

    return {"train_full": train_full, "ctx": ctx, "swap": swap, "cdh": cdh,
            "root": root, "eye": eye, "split": split, "swap_seed": swap_seed,
            "cdh_root": cdh_root}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="inner")
    ap.add_argument("--swap-seed", type=int, default=20260806)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    root = os.environ.get("TWM_CUPS_ROOT")
    cdh_root = os.environ.get("TWM_CDH_ROOT")          # optional second capability, reported only
    eye = os.environ.get("TWM_EYE", "enc_e5_ft_nocwd_hf")
    if not root:
        raise SystemExit("TWM_CUPS_ROOT is not set")

    out = pathlib.Path(a.out or f".cache/lane-{a.split}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists — delete it to rebuild")
        return 0

    print(f"building lane context: root={root} eye={eye} split={a.split}", flush=True)
    blob = build(root, eye, a.split, a.swap_seed, cdh_root=cdh_root)
    torch.save(blob, out)
    n = len(blob["ctx"]["wins"])
    ncdh = len(blob["cdh"]["wins"]) if blob.get("cdh") else 0
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) — {n} cups windows, "
          f"{ncdh} cd-history windows, "
          f"{len(blob['swap']['alts'])} role-swapped, "
          f"self-parity {blob['swap']['self_parity_cos']}")
    print("point workers at it with TWM_CONTEXT=" + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
