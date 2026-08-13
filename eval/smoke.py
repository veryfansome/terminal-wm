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

    M = __import__("realenv.seq_worldmodel", fromlist=["D", "pick_device"])
    target_mod = G.load_target(gen)
    if getattr(target_mod, "LEARNED", False):
        net.target_module = target_mod.make(M.D)
    head.wrap(net, M.D, **(head_p or {}))

    # A chunk that names a method torch already owns shadows it, and the failure surfaces much
    # later as an inscrutable TypeError from inside .to(device). Name it here instead.
    import torch.nn as _nn
    shadowed = []
    for mod in net.modules():
        for attr in ("_apply", "forward", "to", "state_dict", "load_state_dict", "parameters",
                     "named_parameters", "modules", "children", "train", "eval", "zero_grad",
                     "register_buffer", "register_parameter", "add_module", "cuda", "cpu"):
            own = type(mod).__dict__.get(attr)
            if own is None or attr == "forward":
                continue
            if getattr(_nn.Module, attr, None) is not None and callable(own):
                shadowed.append(f"{type(mod).__name__}.{attr}")
    if shadowed:
        raise ValueError(
            f"these override methods that torch.nn.Module owns: {sorted(set(shadowed))}. "
            f"Module._apply is what .to(device) calls, and the others are used by the harness or "
            f"the optimiser, so an override with a different signature breaks at a distance. "
            f"Rename them.")

    # The move to device is the operation that reclaims a registered target module and the one
    # that calls Module._apply, so exercise it rather than assuming it works.
    net = net.to(M.pick_device())

    print(json.dumps({"ok": True, "params": n_params,
                      "chunks": {k: v.get("impl") for k, v in gen["chunks"].items()}}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f"smoke FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
