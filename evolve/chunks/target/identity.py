"""Contract for any target impl: expose two pure functions
  make_target(z_obs, z_prev) -> the tensor the model is TRAINED to predict, per cmd step;
      z_obs = true next-obs embedding [n,768], z_prev = previous obs embedding [n,768], zeros
      at the first step.
  to_obs(pred, z_prev) -> the EXACT inverse: a predicted next-obs embedding [n,768] in the fixed
      observation space, for the retrieval eval.
Both must be pure functions of their args. Or set LEARNED = True and expose make(D) -> nn.Module
with the same pair plus reg()."""

NAME = "identity"
DESCRIPTION = "Predict the next observation embedding directly (R4 default)."


def make_target(z_obs, z_prev):
    return z_obs


def to_obs(pred, z_prev):
    return pred
