import os
from evolve.chunks.perception.enc_e5_base_nocwd import render_obs, render_cmd, pool  # noqa: F401

# Fail closed with no default: '/root/enc/e5-ft' is the cwd-IN-tuned encoder and this module
# renders cwd-OUT, so a fallback would silently corrupt every encoded frame.
MODEL = os.environ.get("TJ_FT_ENCODER")
if not MODEL:
    raise ImportError(
        "enc_e5_ft_nocwd_hf: environment variable TJ_FT_ENCODER is unset. Set it to the "
        "cwd-OUT-native encoder checkpoint — HF repo 'veryfansome/terminal-jepa-encoders', "
        "subfolder 'e5-nocwd' (download it, then point TJ_FT_ENCODER at that local directory). "
        "There is deliberately NO default: the old default '/root/enc/e5-ft' is the cwd-IN eye "
        "and substituting it silently corrupts the whole encode.")
