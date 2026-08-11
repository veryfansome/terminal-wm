"""The stream-scoreability gate: can this genome's tokens be reproduced by the instrument?

The cups instrument does not call stream.collate. It assembles its own token tensor from the
cached embeddings, so anything collate does to command tokens is invisible to it unless the
stream also exposes it as a pure function:

    code_cmds(cmds, z_cmd) -> z_cmd'      (optional; absent means the identity)

Given the ordered command strings of a sequence and their standardized embeddings, return the
command embeddings the net is actually trained on. The instrument calls this once per arm --
the native chain for one, the role-swapped chain for the other -- so a coding stream is scored
on the tokens it trained on, in both arms.

This gate checks what the instrument assumes, by execution rather than by reading a declared
constant. A stream that writes codes into command tokens inside collate and does not expose
code_cmds has an unchanged LAYOUT, so it answers a layout question truthfully and is then
scored on tokens its net never saw.

Command tokens are compared under no_grad, the regime the instrument measures in: a stochastic
train-time augmentation is expected to be inactive there, exactly as dropout is.
"""
import torch

from realenv import seq_worldmodel as M

N_STEPS = 4
TOL = 1e-5


def _toy(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [{"z_obs": torch.randn(N_STEPS, M.D, generator=g),
             "z_cmd": torch.randn(N_STEPS, M.D, generator=g),
             "cmds": [f"mv /tmp/w/a{i}.txt /tmp/w/b{i}.txt" for i in range(N_STEPS)],
             "image": "x"}]


@torch.no_grad()
def stream_scoreable(stream, device):
    seq = _toy()
    z_cmd, z_obs = seq[0]["z_cmd"], seq[0]["z_obs"]
    try:
        b = stream.collate(seq, device)
    except Exception as e:
        return False, f"stream.collate raised on a toy batch: {type(e).__name__}: {e}"

    tok = b["tok"].detach().cpu()
    types = b["types"].detach().cpu()
    if tok.ndim != 3 or tok.shape[0] != 1 or tok.shape[2] != M.D:
        return False, (f"stream.collate returned tok shaped {tuple(tok.shape)}; the instrument "
                       f"pins [B, 2n, {M.D}]")
    if tok.shape[1] != 2 * N_STEPS:
        return False, (f"stream.collate returned {tok.shape[1]} tokens for {N_STEPS} steps; the "
                       f"instrument pins a strided [cmd, obs, ...] layout of exactly 2n")
    if not (types[0, 0::2] == 0).all() or not (types[0, 1::2] == 1).all():
        return False, ("stream.collate does not put commands at even positions and observations "
                       "at odd ones; the instrument reads the scored position by that stride")
    if not torch.allclose(tok[0, 1::2], z_obs, atol=TOL):
        return False, ("stream.collate altered the OBSERVATION tokens; the target and the "
                       "candidate bank live in that space and the instrument builds both from "
                       "the cache")

    marker = torch.arange(2 * N_STEPS, dtype=torch.float32).view(1, -1, 1).expand(1, -1, M.D)
    try:
        got = stream.extract_cmd_pred(marker.to(device), b).detach().cpu()[0, :, 0]
    except Exception as e:
        return False, f"stream.extract_cmd_pred raised: {type(e).__name__}: {e}"
    if not torch.equal(got, torch.arange(0, 2 * N_STEPS, 2, dtype=torch.float32)):
        return False, (f"stream.extract_cmd_pred reads positions {got.tolist()}; the instrument "
                       f"reads {list(range(0, 2 * N_STEPS, 2))}")

    fn = getattr(stream, "code_cmds", None)
    expected = z_cmd if fn is None else fn(list(seq[0]["cmds"]), z_cmd.clone())
    if fn is not None:
        if not torch.is_tensor(expected) or tuple(expected.shape) != tuple(z_cmd.shape):
            return False, ("stream.code_cmds must return a tensor shaped like z_cmd "
                           f"{tuple(z_cmd.shape)}, got {tuple(getattr(expected, 'shape', ()))}")
        if not torch.isfinite(expected).all():
            return False, "stream.code_cmds produced non-finite values"
    if not torch.allclose(tok[0, 0::2], expected.to(tok.dtype), atol=TOL):
        worst = (tok[0, 0::2] - expected.to(tok.dtype)).abs().max().item()
        return False, (
            "stream.collate's COMMAND tokens do not match "
            + ("its own code_cmds" if fn is not None else "the raw cached embeddings")
            + f" (max abs diff {worst:.3g}). The scoring instrument builds its tokens from the "
              "cache and reproduces the coding through code_cmds; anything collate does to a "
              "command token that code_cmds does not reproduce is present in training and "
              "absent at scoring. Expose the coding as code_cmds(cmds, z_cmd).")
    return True, ""
