"""target chunk: predict the next observation embedding directly (the R4 default).

Contract for any target impl: expose two pure functions
  make_target(z_obs, z_prev) -> the tensor the model is TRAINED to predict (per cmd step;
      z_obs = true next-obs embedding [n,768], z_prev = previous obs embedding [n,768], zeros
      at the first step). The objective's loss(pred, make_target(...)) is what's optimized.
  to_obs(pred, z_prev) -> reconstruct a predicted NEXT-OBS embedding [n,768] from the model's
      raw prediction, for the retrieval eval (which always ranks true z_obs vs foils).
Both must be pure functions of their args (z_prev is the causally-available previous observation,
so no leakage). Changing the target changes WHAT is learned, not the model or the eval metric."""

NAME = "identity"
DESCRIPTION = "Predict the next observation embedding directly (R4 default)."


def make_target(z_obs, z_prev):
    return z_obs


def to_obs(pred, z_prev):
    return pred
