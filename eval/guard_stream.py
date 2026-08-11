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


VERBS = ["ls -1 /tmp/w/cups/opt", "cat /tmp/w/cups/opt/s0.dat",
         "mv /tmp/w/cups/opt/s0.dat /tmp/w/cups/opt/s1.dat", "cd /tmp/w/cups/var",
         "cat /tmp/w/cups/opt/s1.dat", "mv /tmp/w/cups/opt/s1.dat /tmp/w/cups/var/s2.dat",
         "ls -la /tmp/w/cups/var", "cat /tmp/w/cups/var/s2.dat",
         "mv /tmp/w/cups/var/s2.dat /tmp/w/cups/opt/s0.dat", "cat /tmp/w/cups/opt/s0.dat",
         "cat /tmp/w/cups/var/s2.dat", "ls -1 /tmp/w/cups/opt"]


def _toy(seed=0):
    """Several rows of DIFFERENT lengths in one batch, with a verb mix and the pack's constant
    move observation. A single fixed-length all-mv row cannot see a coding that switches on
    length, on batch position, on the verb, or on the observation."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in (N_STEPS, 7, 11):
        cmds = VERBS[:n]
        z_obs = torch.randn(n, M.D, generator=g)
        empty = torch.randn(M.D, generator=g)
        for t, c in enumerate(cmds):
            if c.startswith("mv "):
                z_obs[t] = empty
        out.append({"z_obs": z_obs, "z_cmd": torch.randn(n, M.D, generator=g),
                    "cmds": list(cmds), "image": "x"})
    return out


@torch.no_grad()
def stream_matches_context(stream, ctx, device, n_seqs=4):
    """The decisive check, once real sequences exist: run the genome's OWN collate over them and
    compare BOTH halves of its output to what the instrument will reconstruct.

    A toy batch can only rule out the codings it happens to exercise. This asks the question on
    the data the measurement will actually use -- real lengths, real verbs, real observations.

    Everything handed to collate is a clone and every expected value is computed from a pristine
    copy BEFORE collate runs, so a collate that edits its input in place is caught rather than
    validating itself -- and cannot poison ctx['seqs'], which flatten_predictions still reads.
    The coding is also checked at two batch sizes, because training always collates at the
    genome's bs (64 for every archived genome) while a gate naturally runs at three or four."""
    fn = getattr(stream, "code_cmds", None)
    seqs = (ctx.get("seqs") or [])[:n_seqs]
    if not seqs:
        return True, ""
    pristine = [{"cmds": [st["cmd"] for st in s["steps"]],
                 "z_cmd": s["z_cmd"].detach().clone(),
                 "z_obs": s["z_obs"].detach().clone(),
                 "image": s.get("image", "x")} for s in seqs]

    want_cmd = []
    for row in pristine:
        if fn is None:
            want_cmd.append(row["z_cmd"].clone())
            continue
        try:
            w = fn(list(row["cmds"]), row["z_cmd"].clone())
        except Exception as e:
            return False, f"stream.code_cmds raised on a real sequence: {type(e).__name__}: {e}"
        if not torch.is_tensor(w) or tuple(w.shape) != tuple(row["z_cmd"].shape):
            return False, ("stream.code_cmds must return a tensor shaped like z_cmd "
                           f"{tuple(row['z_cmd'].shape)}, got "
                           f"{tuple(getattr(w, 'shape', ()))}")
        want_cmd.append(w.detach())

    for reps in (1, 2):
        batch = []
        for _ in range(reps):
            batch += [{k: (v.clone() if torch.is_tensor(v) else
                           (list(v) if isinstance(v, list) else v))
                       for k, v in row.items()} for row in pristine]
        try:
            b = stream.collate(batch, device)
        except Exception as e:
            return False, (f"stream.collate raised on real sequences at batch size {len(batch)}: "
                           f"{type(e).__name__}: {e}")
        tok = b["tok"].detach().cpu()
        for i, row in enumerate(pristine):
            n = len(row["cmds"])
            got_c = tok[i, 0:2 * n:2]
            exp_c = want_cmd[i].cpu().to(got_c.dtype)
            if got_c.shape != exp_c.shape or not torch.allclose(got_c, exp_c, atol=TOL):
                worst = ((got_c - exp_c).abs().max().item()
                         if got_c.shape == exp_c.shape else float("nan"))
                return False, (
                    f"on real sequence {i} ({n} steps, batch of {len(batch)}), collate's COMMAND "
                    f"tokens differ from what code_cmds reproduces (max abs diff {worst:.3g}). "
                    f"The instrument builds its tokens from the cache and replays the coding "
                    f"through code_cmds, so a coding that only appears at some lengths, verbs or "
                    f"batch sizes is present in training and absent at scoring.")
            got_o = tok[i, 1:2 * n:2]
            exp_o = row["z_obs"].cpu().to(got_o.dtype)
            if got_o.shape != exp_o.shape or not torch.allclose(got_o, exp_o, atol=TOL):
                return False, (
                    f"on real sequence {i}, collate altered the OBSERVATION tokens. The target "
                    f"and the forced-choice candidate bank live in that space and the instrument "
                    f"builds both from the cache, so an observation coding is unreproducible "
                    f"there by construction.")

    for i, row in enumerate(pristine):
        if not torch.equal(row["z_cmd"], seqs[i]["z_cmd"]) or \
                not torch.equal(row["z_obs"], seqs[i]["z_obs"]):
            return False, ("stream.collate mutated the sequences it was handed. The context is "
                           "shared by every later readout, so an in-place edit would leak into "
                           "the health metric and into any candidate measured after this one.")
    return True, ""


@torch.no_grad()
def stream_scoreable(stream, device):
    seq = _toy()
    z_cmd, z_obs = seq[0]["z_cmd"], seq[0]["z_obs"]
    try:
        b = stream.collate(seq, device)
    except Exception as e:
        return False, f"stream.collate raised on a toy batch: {type(e).__name__}: {e}"

    tok = b["tok"].detach().cpu()[:1]
    types = b["types"].detach().cpu()[:1]
    tok, types = tok[:, :2 * N_STEPS], types[:, :2 * N_STEPS]
    if tok.ndim != 3 or tok.shape[0] != 1 or tok.shape[2] != M.D:
        return False, (f"stream.collate returned tok shaped {tuple(tok.shape)}; the instrument "
                       f"pins [B, 2n, {M.D}]")
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
