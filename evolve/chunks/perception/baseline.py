"""Contract for any perception impl: expose MODEL (an HF encoder name), render_obs(step) -> str,
render_cmd(step) -> str and pool(h, mask) -> [B,768]. evolve/reencode.py runs these to build a
new dataset root's caches."""
MODEL = "answerdotai/ModernBERT-base"
OBS_CAP = 1600
def render_obs(step):
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd','/')} exit={step.get('exit',0)}\n{out}"
def render_cmd(step):
    return step["cmd"]
def pool(h, mask):
    m = mask.unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1)
