"""Structural smoke gate: the genome names real impls, every one of them imports, the
architecture constructs and has parameters, and the head wraps it. Data-free — it must not
touch the pack root. Exit nonzero records the candidate as a failure with this reason.
"""
import json
import sys

from evolve import genome as G


def main(genome_path):
    gen = json.load(open(genome_path))
    G.validate(gen)

    loaders = [("objective", G.load_objective), ("target", G.load_target),
               ("stream", G.load_stream), ("optim", G.load_optim),
               ("batcher", G.load_batcher)]
    for axis, fn in loaders:
        fn(gen)
    head, head_p = G.load_head(gen)
    build, arch_p = G.load_arch(gen)

    net = build(**arch_p)
    n_params = sum(p.numel() for p in net.parameters())
    if n_params == 0:
        raise ValueError("the built architecture has no parameters")

    head.wrap(net, __import__("realenv.seq_worldmodel", fromlist=["D"]).D, **(head_p or {}))

    print(json.dumps({"ok": True, "params": n_params,
                      "chunks": {k: v.get("impl") for k, v in gen["chunks"].items()}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f"smoke FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
