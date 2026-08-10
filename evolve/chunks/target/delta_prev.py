NAME = "delta_prev"
DESCRIPTION = "Predict z_obs - z_prev (the change); reconstruct z_prev + prediction for eval."


def make_target(z_obs, z_prev):
    return z_obs - z_prev


def to_obs(pred, z_prev):
    return z_prev + pred
