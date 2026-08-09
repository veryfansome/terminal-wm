"""The no-future-leakage guard — the one property no candidate may violate.

The whole measurement rests on the model predicting each command's next observation from what
came BEFORE it. A mechanism that can see the observation it is being asked to predict scores
beautifully and means nothing. So this runs on every candidate, before any GPU time: perturb a
later observation and assert that no earlier command's prediction moves.

It needs a BUILT net but not a TRAINED one, and no data at all, which is what makes it cheap
enough to be unconditional. Structural checks belong here rather than in a gate ceremony after
the fact — they cost every candidate the same, and a candidate that fails has no usable number.

Two independent checks:
  head.leak_safe  — the head declares whether its wrapper can see the future. A head that
                    re-points forward or adds an auxiliary task is exactly where leakage creeps
                    in, so the axis is required to answer for itself.
  stream.leakage_ok — the empirical check: build the net, perturb, measure movement.
"""
import json
import sys

from evolve import genome as G
from realenv import seq_worldmodel as M


def main(genome_path):
    gen = json.load(open(genome_path))
    G.validate(gen)
    stream = G.load_stream(gen)
    target_mod = G.load_target(gen)
    head, head_p = G.load_head(gen)

    if not head.leak_safe(head, head_p):
        raise ValueError(
            "head.leak_safe returned False — the head declares itself unable to guarantee that "
            "it cannot see the observation it is predicting")

    device = M.pick_device()
    build, arch_p = G.load_arch(gen)
    net = build(**arch_p)
    if getattr(target_mod, "LEARNED", False):
        net.target_module = target_mod.make(M.D)
    head.wrap(net, M.D, **(head_p or {}))
    net = net.to(device)

    if not stream.leakage_ok(net, device):
        raise ValueError(
            "stream.leakage_ok returned False — perturbing a later observation moved an earlier "
            "command's prediction, so this candidate can see the future it is scored on")

    print(json.dumps({"ok": True, "device": str(device)}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f"leakage guard FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
