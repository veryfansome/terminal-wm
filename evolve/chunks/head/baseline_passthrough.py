"""Contract for any head impl:
  wrap(net, D, **params) -> a head state, or None for no head
      Called BEFORE the optimizer is built, so any readout or auxiliary parameters it
      registers on net are optimized. Two hazards a wrapper must avoid: a parent-child
      module cycle (hold the base net unregistered, or moving to device recurses), and
      forward recursion when re-pointing net.forward (save the original bound method first).
  aux_loss(head_state, batch, net, device) -> scalar tensor or 0.0; train-time only.
  leak_safe(mod, params) -> bool; asserted before scoring.
"""

import torch

NAME_BASELINE = "baseline_passthrough"
DESCRIPTION_BASELINE = ("Arch's own Linear readout, unchanged; no aux loss. "
                        "Bit-identical to the pre-axis harness readout.")


def wrap(net, D, **params):
    return None


# A hard 0.0 (not a zero tensor) so `main + aux` is main bit-for-bit and archived
# fitnesses replay exactly.
def aux_loss(head_state, batch, net, device):
    return 0.0


def leak_safe(mod, params):
    return True


NAME = NAME_BASELINE
DESCRIPTION = DESCRIPTION_BASELINE
