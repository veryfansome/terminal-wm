"""target chunk: predict the RESIDUAL from the previous observation (z_obs - z_prev), a classic
world-model trick — model the CHANGE the command causes rather than the absolute next state. On a
shell trace many commands leave the observation similar to the previous one (same cwd banner, an
`ls` of a nearby dir), so the residual may be lower-variance and easier to predict; reconstruct
z_prev + delta for the retrieval eval."""

NAME = "delta_prev"
DESCRIPTION = "Predict z_obs - z_prev (the change); reconstruct z_prev + prediction for eval."


def make_target(z_obs, z_prev):
    return z_obs - z_prev


def to_obs(pred, z_prev):
    return z_prev + pred
