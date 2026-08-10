"""Build the lane context ONCE, so a campaign loads and derives nothing per candidate.

    python -m cloud.build_context [--split inner] [--out .cache/lane-inner.pt]

Everything in here is a property of (train root, frame root, pack roots, eye, split, swap-seed)
and NOT of the genome:

  - the composed and standardized training set
  - the cups windows harvested from the val split, and the interleaved token layout
  - the cd-history windows and their layout
  - the ROLE-SWAP chains: partner draw, chain synthesis, and the re-encoded mv embeddings

That last one is the expensive part and the reason this exists. Synthesizing and encoding the
alternative chains means loading the text encoder and running it over every rewritten command; done
inside the per-candidate path it is repeated once per (genome, seed) for a result that is bit-wise
identical every time.

THE TRAINING ROOT MAY BE A BLEND SPEC
  TWM_TRAIN_ROOT names the root the net trains on; unset, it is the cups pack root and this is the
  single-pack lane, unchanged. If that root holds a blend.json it is a SPEC, not a dataset: it has
  no data of its own, and the training set is composed here from the constituents' ALREADY ENCODED
  caches using the exact sequence indices the spec records. So each pack is encoded once, every
  blend arm shares bit-identical embeddings for the sequences they have in common, and no blend
  re-encodes anything. Write a spec with evolve/blend_root.py.

THE FRAME
  Standardization statistics come from ONE root and every tensor in the context is put in that one
  frame: the composed training set, the cups windows, the cd-history windows. A blend spec root has
  no caches, so it cannot supply a frame, and with two capability packs measured on one net neither
  pack can be the frame either. TWM_FRAME_ROOT names the frozen reference root that supplies the
  train statistics, and it is REQUIRED whenever a blend is in play — never inferred, never falling
  back to a constituent. A silent frame mismatch does not raise; it produces plausible numbers.
  The resolved frame is recorded in the saved context so a consumer can check what it was built in.

The artifact is written with torch's zipfile serialization so workers can memory-map it: N training
processes then share one physical copy of the tensors instead of each loading their own. Processes
rather than threads is deliberate — training seeds the global RNG and dropout draws from it, so
concurrent threads would interleave those draws and a genome+seed would stop meaning one thing.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys

import torch

from evolve import cdh_probe as CDH, cups_probe as CP, harness as H
from realenv import seq_worldmodel as M

BLEND_SPEC = "blend.json"

# A blend spec root carries the spec and the inherited summary and NOTHING else. If any of these
# appear beside it, the root has data that no spec accounts for and the two can disagree without
# anything raising.
SPEC_ROOT_FORBIDDEN = ("train.jsonl", "val.jsonl", "emb-seq-train.pt", "emb-seq-val.pt")


def load_blend_spec(train_root):
    """The blend spec at `train_root`, or None for an ordinary single-root lane."""
    p = pathlib.Path(train_root) / BLEND_SPEC
    if not p.exists():
        return None
    spec = json.loads(p.read_text())
    if spec.get("format") != "twm-blend/1":
        raise RuntimeError(f"{p}: unknown blend format {spec.get('format')!r}")
    stray = [f for f in SPEC_ROOT_FORBIDDEN if (pathlib.Path(train_root) / f).exists()]
    if stray:
        raise RuntimeError(
            f"{train_root} holds both a blend spec and its own data ({', '.join(stray)}) — a spec "
            f"root must carry no data, or its caches and its spec can disagree silently")
    if spec.get("val", {}).get("blended", True):
        raise RuntimeError(
            f"{p} declares a blended val split. The evaluation split is the frozen reference and "
            f"is never mixed; this spec cannot be resolved")
    return spec


def resolve_constituent(entry, spec_root, what):
    """Resolve one constituent root from its spec entry and verify its identity.

    The recorded path is absolute and machine-specific, so a spec written on one box names roots
    that may sit elsewhere on another. If the recorded path is absent, the same basename beside the
    spec root is tried. Either way the root's summary.json must hash to the sha the spec recorded,
    so a relocated root is accepted and a re-minted or substituted one is not."""
    want = entry["summary_sha256"]
    cands = [pathlib.Path(entry["root"]),
             pathlib.Path(spec_root).resolve().parent / pathlib.Path(entry["root"]).name]
    for c in cands:
        if (c / "summary.json").exists():
            got = hashlib.sha256((c / "summary.json").read_bytes()).hexdigest()
            if got != want:
                raise RuntimeError(
                    f"blend {what} {c}: summary.json sha {got} != the {want} recorded in the spec "
                    f"— this is not the root the spec was written against")
            return str(c)
    raise RuntimeError(
        f"blend {what} {entry['root']} not found (also looked beside the spec root). Pull the "
        f"constituent roots before resolving a blend")


def compose_train_seqs(spec, spec_root, eye, device):
    """The composed training set: the base's encoded train sequences followed, per pack in spec
    order, by exactly the pack sequences the spec's indices name, in that recorded order.

    Nothing is encoded here. Every sequence comes out of a constituent's existing cache, which is
    what makes two arms' shared sequences bit-identical."""
    base_root = resolve_constituent(spec["base"], spec_root, "base")
    seqs = list(H._cached_encode(base_root, "train", eye, device))
    if len(seqs) != spec["base"]["train_seqs"]:
        raise RuntimeError(
            f"blend base {base_root}: cache holds {len(seqs)} train seqs, the spec recorded "
            f"{spec['base']['train_seqs']}")

    for entry in spec["packs"]:
        root = resolve_constituent(entry, spec_root, "pack")
        pack = H._cached_encode(root, "train", eye, device)
        idx = entry["indices"]
        if len(pack) != entry["train_seqs"]:
            raise RuntimeError(
                f"blend pack {root}: cache holds {len(pack)} train seqs, the spec recorded "
                f"{entry['train_seqs']}")
        if not idx or len(idx) != entry["seqs_added"]:
            raise RuntimeError(
                f"blend pack {root}: {len(idx)} indices, the spec recorded "
                f"{entry['seqs_added']} added")
        if len(set(idx)) != len(idx) or idx != sorted(idx):
            raise RuntimeError(f"blend pack {root}: indices are not unique and sorted")
        if min(idx) < 0 or max(idx) >= len(pack):
            raise RuntimeError(
                f"blend pack {root}: index range [{min(idx)}, {max(idx)}] does not fit "
                f"{len(pack)} cached sequences")
        got = hashlib.sha256(json.dumps(idx).encode()).hexdigest()
        if got != entry["indices_sha256"]:
            raise RuntimeError(f"blend pack {root}: index sha {got} != recorded "
                               f"{entry['indices_sha256']}")
        seqs.extend(pack[i] for i in idx)
    return seqs


def resolve_frame_root(train_root, blend, frame_root):
    """The one root whose train statistics standardize everything in this context.

    With a blend in play the frame must be given explicitly: a spec root has no statistics of its
    own, and picking a constituent would put the net's inputs and some pack's windows in different
    frames without anything raising."""
    if blend is not None:
        if not frame_root:
            raise RuntimeError(
                "TWM_FRAME_ROOT is not set, and the training root is a BLEND SPEC. A spec root has "
                "no caches, so it cannot supply the standardization frame, and this lane measures "
                "more than one capability pack on one net, so no pack can be the frame either. "
                "Point TWM_FRAME_ROOT at the frozen reference root whose train statistics every "
                "arm is standardized in. This is never inferred: a wrong frame raises nothing and "
                "produces plausible numbers.")
    frame = frame_root or train_root
    if not (pathlib.Path(frame) / "emb-seq-train.pt").exists():
        raise RuntimeError(
            f"frame root {frame} has no emb-seq-train.pt — the frame is the train statistics of an "
            f"ENCODED root")
    return frame


def build(root, eye, split, swap_seed, cdh_root=None, train_root=None, frame_root=None):
    device = M.pick_device()
    train_root = train_root or root
    blend = load_blend_spec(train_root)
    frame = resolve_frame_root(train_root, blend, frame_root)

    if blend is not None:
        train_full = compose_train_seqs(blend, train_root, eye, device)
    else:
        train_full = H._cached_encode(train_root, "train", eye, device)

    # Statistics come off the UN-standardized train sequences of the frame root. When the frame IS
    # the training root and nothing was blended, train_full already holds exactly those sequences,
    # so reusing it is both identical and one fewer multi-GB load.
    same = (blend is None
            and os.path.realpath(frame) == os.path.realpath(train_root))
    stats_src = train_full if same else H._cached_encode(frame, "train", eye, device)
    mo, so, mc, sc = M.standardize_stats(stats_src)
    M.apply_stats(train_full, mo, so, mc, sc)

    # Every window set goes in the SAME frame as the net's training inputs. Putting a pack's
    # windows in their own frame would measure a net on inputs it never saw.
    ctx = CP.load_cups_context(root, split, eye, device, stats_data=frame)
    if ctx is None:
        raise SystemExit(f"no cups windows in the {split} split of {root}")

    # the one genuinely expensive, genuinely net-independent step
    swap = CP.build_swap_cache(ctx, eye, device, seed=swap_seed)

    cdh = None
    if cdh_root:
        cdh = CDH.load_cdh_context(cdh_root, split, eye, device, redir_only=True,
                                   stats_data=frame)
        if cdh is None:
            raise SystemExit(f"no cd-history windows in the {split} split of {cdh_root}")

    return {"train_full": train_full, "ctx": ctx, "swap": swap, "cdh": cdh,
            "root": root, "eye": eye, "split": split, "swap_seed": swap_seed,
            "cdh_root": cdh_root, "train_root": train_root, "frame_root": frame,
            # what the training set actually is, travelling with the tensors it describes
            "blend": (None if blend is None else
                      {"spec_sha256": hashlib.sha256(
                          (pathlib.Path(train_root) / BLEND_SPEC).read_bytes()).hexdigest(),
                       "provenance": blend["provenance"]})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="inner")
    ap.add_argument("--swap-seed", type=int, default=20260806)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    root = os.environ.get("TWM_CUPS_ROOT")
    cdh_root = os.environ.get("TWM_CDH_ROOT")          # the second capability pack's windows
    train_root = os.environ.get("TWM_TRAIN_ROOT")      # a blend spec, or unset for the cups root
    frame_root = os.environ.get("TWM_FRAME_ROOT")      # required when a blend is in play
    eye = os.environ.get("TWM_EYE", "enc_e5_ft_nocwd_hf")
    if not root:
        raise SystemExit("TWM_CUPS_ROOT is not set")

    out = pathlib.Path(a.out or f".cache/lane-{a.split}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} already exists — delete it to rebuild")
        return 0

    print(f"building lane context: root={root} train={train_root or root} "
          f"frame={frame_root or '(train root)'} eye={eye} split={a.split}", flush=True)
    blob = build(root, eye, a.split, a.swap_seed, cdh_root=cdh_root,
                 train_root=train_root, frame_root=frame_root)
    torch.save(blob, out)
    n = len(blob["ctx"]["wins"])
    ncdh = len(blob["cdh"]["wins"]) if blob.get("cdh") else 0
    print(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB) — {len(blob['train_full'])} training "
          f"sequences in the {blob['frame_root']} frame, {n} cups windows, "
          f"{ncdh} cd-history windows, "
          f"{len(blob['swap']['alts'])} role-swapped, "
          f"self-parity {blob['swap']['self_parity_cos']}")
    if blob["blend"]:
        print("blend: " + json.dumps(blob["blend"]["provenance"]["packs"]))
    print("point workers at it with TWM_CONTEXT=" + str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
