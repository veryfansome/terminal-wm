import torch

NAME_BASELINE = "baseline_passthrough"
DESCRIPTION_BASELINE = ("Arch's own Linear readout, unchanged; no aux loss. "
                        "Bit-identical to the pre-axis harness readout.")


def wrap(net, D, **params):
    # Do nothing: leave net.forward exactly as the arch defined it, add no modules.
    # Returning None signals "no head state" to the (unused) aux path.
    return None


def aux_loss(head_state, batch, net, device):
    # No auxiliary term. Return a hard zero so `main + aux` == main bit-for-bit.
    return 0.0


def leak_safe(mod, params):
    # No aux branch, no head params, forward untouched -> trivially leak-safe.
    return True


NAME = NAME_BASELINE
DESCRIPTION = DESCRIPTION_BASELINE
