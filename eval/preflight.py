"""Environment preflight — run once per scoring clone, before any candidate code.

Checks the roots, the blend constituents, the frame, the ceiling table and the encoder sha, and
exits nonzero on anything wrong with the machine. Anything reached only at probe time still
raises inside the adapter.
"""
import hashlib
import json
import os
import pathlib
import sys

from eval.adapter import CEILING_TABLE, preflight


def embedding_sha(root):
    """sha256 over the encoded tensors themselves (emb-seq-train.pt, emb-seq-val.pt)."""
    h = hashlib.sha256()
    for name in ("emb-seq-train.pt", "emb-seq-val.pt"):
        h.update(name.encode())
        with open(pathlib.Path(root) / name, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def main():
    root, eye, train_root, frame_root = preflight()
    rp = pathlib.Path(root)

    tbl = json.load(open(CEILING_TABLE))
    knobs, cups = tbl.get("knobs") or {}, (json.loads((rp / "summary.json").read_text())
                                           .get("cups") or {})
    mismatch = [k for k in ("n_grid", "r_grid", "chainbias")
                if k in knobs and k in cups and str(knobs[k]) != str(cups[k])]
    if mismatch:
        raise RuntimeError(
            f"ceiling table knobs disagree with the data root on {mismatch}: table={knobs}, "
            f"root={cups}. The table decides which windows are eligible, so a mismatch redefines "
            f"the measurement rather than merely annoying it.")

    got = embedding_sha(root)
    want = os.environ.get("TWM_ROOT_SHA")
    if want and got != want:
        raise RuntimeError(
            f"encoded-root embedding sha {got} != pinned {want}. These are not the tensors this "
            f"campaign was measured against, so nothing scored here is comparable to it.")

    print(json.dumps({"ok": True, "root": root, "eye": eye, "train_root": train_root,
                      "frame_root": frame_root, "embedding_sha": got, "pinned": bool(want)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"PREFLIGHT FAILED (environment, not a candidate): {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)
