"""Structural smoke gate — runs before any GPU time is spent on a candidate.

Cheap and data-free on purpose: it must not touch the multi-GB pack root, because the engine
gives the gates a much shorter timeout than the paid eval. All it proves is that the genome names
real impls, that every one of them imports, and that the architecture actually constructs.

Exit nonzero and the candidate is recorded as a failure with this reason — which is the right
outcome: an impl that cannot be built has no number, and the reason travels to its descendants.
"""
import json
import sys

from evolve import genome as G


def main(genome_path):
    gen = json.load(open(genome_path))
    G.validate(gen)                                  # shape + every impl exists in the registry

    loaders = [("objective", G.load_objective), ("target", G.load_target),
               ("stream", G.load_stream), ("optim", G.load_optim),
               ("batcher", G.load_batcher)]
    for axis, fn in loaders:
        fn(gen)
    head, head_p = G.load_head(gen)
    build, arch_p = G.load_arch(gen)

    net = build(**arch_p)                            # must construct, not merely import
    n_params = sum(p.numel() for p in net.parameters())
    if n_params == 0:
        raise ValueError("the built architecture has no parameters")

    # wrap() runs before the optimizer is built in training, so a head that explodes here would
    # explode there — find out now, for free.
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
