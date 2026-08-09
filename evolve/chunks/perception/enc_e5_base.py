"""perception TIER-2 (encoder swap = different 'eyes'): intfloat/e5-base-v2, a retrieval-tuned
768-d encoder (its training objective aligns with our retrieval eval). Uses e5's required
'passage: ' prefix + mean-pool (e5's standard usage)."""
from evolve.chunks.perception.baseline import pool, OBS_CAP
MODEL = "intfloat/e5-base-v2"
def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"
def render_cmd(step):
    return "passage: " + step["cmd"]
