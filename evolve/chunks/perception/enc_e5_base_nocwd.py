"""ABLATION (2026-08-02, cwd-label experiment): enc_e5_base with the `cwd=…` prefix DROPPED
from the observation render — everything else (model, pool, OBS_CAP, `passage:` prefix, cmd
render) byte-identical to enc_e5_base. Tests whether the content-verb margin survives when the
model must INFER the working directory from history instead of being handed it (the JEPA
latent-state premise). Compare a champion scored on data/dockerfs3-e5 (cwd-in) vs a root
re-encoded with THIS perception (cwd-out)."""
from evolve.chunks.perception.baseline import pool, OBS_CAP
MODEL = "intfloat/e5-base-v2"


def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: exit={step.get('exit', 0)}\n{out}"      # cwd= prefix REMOVED (ablation)


def render_cmd(step):
    return "passage: " + step["cmd"]
