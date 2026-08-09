"""Environment preflight — run once per scoring clone, BEFORE any candidate code.

The engine treats a nonzero exit from the setup step as an infrastructure failure: it is not
archived, it does not consume budget, and it does not become some candidate's null. That is the
correct home for "the data root is missing", "the encoder is the wrong one", "the ceiling table
disagrees with the mint". Those are facts about the machine, not about the genome, and recording
them against a candidate both slanders the candidate and burns a full-eval slot.

Anything reached only at probe time still raises inside the adapter; this is the cheap front door.
"""
import hashlib
import json
import os
import pathlib
import sys

from eval.adapter import CEILING_TABLE, preflight


def embedding_sha(root):
    """sha256 over the encoded tensors themselves.

    The existing stamps cover summary.json and the encoder checkpoint — not the embeddings. But
    encoding is a GPU forward pass, and two machines that each "just re-encode" the same raw mint
    with the same pinned eye can land on different bytes, hence a different standardization frame,
    with every existing check passing. Encode once, publish, and pin THIS: it is the only stamp
    that actually asserts two boxes are measuring in the same space.
    """
    h = hashlib.sha256()
    for name in ("emb-seq-train.pt", "emb-seq-val.pt"):
        h.update(name.encode())
        with open(pathlib.Path(root) / name, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def main():
    root, eye = preflight()                     # env vars, root shards, table presence, eye sha
    rp = pathlib.Path(root)

    # The ceiling table defines the eligible slice. If it was built for a different mint, the
    # slice silently changes shape — so compare the knobs it was built at against the root's own.
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

    print(json.dumps({"ok": True, "root": root, "eye": eye,
                      "embedding_sha": got, "pinned": bool(want)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"PREFLIGHT FAILED (environment, not a candidate): {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)
