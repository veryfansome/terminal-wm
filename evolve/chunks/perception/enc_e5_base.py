from evolve.chunks.perception.baseline import pool, OBS_CAP
MODEL = "intfloat/e5-base-v2"
def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"passage: cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"
def render_cmd(step):
    return "passage: " + step["cmd"]
