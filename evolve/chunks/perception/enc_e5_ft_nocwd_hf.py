"""cwd-OUT-render scoring module: render inherited from enc_e5_base_nocwd (`cwd=` prefix
dropped); MODEL resolves via TJ_FT_ENCODER to WHICHEVER fine-tuned checkpoint the run mounts —
provenance is the env var + the derived root's perception stamp, NOT this docstring. The eye this
root requires is the cwd-OUT-native `e5-nocwd` checkpoint (HF `veryfansome/terminal-jepa-encoders`,
subfolder `e5-nocwd`).

FAIL-CLOSED, NO DEFAULT: this module used to fall back to `/root/enc/e5-ft`, which is the
cwd-IN-tuned eye — the WRONG eye for a cwd-OUT render. That substitution is silent and corrupts
every encoded frame, so an unset TJ_FT_ENCODER now raises at import instead."""
import os
from evolve.chunks.perception.enc_e5_base_nocwd import render_obs, render_cmd, pool  # noqa: F401

MODEL = os.environ.get("TJ_FT_ENCODER")
if not MODEL:
    raise ImportError(
        "enc_e5_ft_nocwd_hf: environment variable TJ_FT_ENCODER is unset. Set it to the "
        "cwd-OUT-native encoder checkpoint — HF repo 'veryfansome/terminal-jepa-encoders', "
        "subfolder 'e5-nocwd' (download it, then point TJ_FT_ENCODER at that local directory). "
        "There is deliberately NO default: the old default '/root/enc/e5-ft' is the cwd-IN eye "
        "and substituting it silently corrupts the whole encode.")
