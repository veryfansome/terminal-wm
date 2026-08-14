"""The stream axis is honored by the scoring instrument, in both arms.

Run: PYTHONPATH=$PWD .venv/bin/python tests/test_stream_axis.py
"""
import hashlib
import sys
import types

import torch

from evolve import cups_probe as CP
from eval import guard_stream as SG
from evolve.chunks.stream import baseline_interleave as BASE
from realenv import seq_worldmodel as M

D = M.D
CODE_AT = D - 1
FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILED.append(name)


def code_of(cmd):
    h = hashlib.blake2b(cmd.encode(), digest_size=8).digest()
    return float(int.from_bytes(h, "big") % 10_000) / 1000.0


def _code_cmds(cmds, z_cmd):
    z = z_cmd.clone()
    for t, c in enumerate(cmds):
        z[t, CODE_AT] = code_of(c)
    return z


def make_coding_stream(with_hook=True):
    """A stream that stamps a per-command code into one reserved coordinate inside collate --
    exactly the shape of proposal that the old declared-constant guard waved through."""
    m = types.ModuleType("coding_stream")
    m.NAME = "coding_stream"
    m.CUPS_LAYOUT = "interleave2"

    def collate(batch, device):
        b = M.collate(batch, device)
        for bi, s in enumerate(batch):
            for t, c in enumerate(s["cmds"]):
                b["tok"][bi, 2 * t, CODE_AT] = code_of(c)
        return b

    m.collate = collate
    m.extract_cmd_pred = BASE.extract_cmd_pred
    m.extract_cmd_input = BASE.extract_cmd_input
    m.flatten_predictions = BASE.flatten_predictions
    m.leakage_ok = BASE.leakage_ok
    if with_hook:
        m.code_cmds = _code_cmds
    return m


def make_ctx():
    g = torch.Generator().manual_seed(7)
    cmds = ["ls /tmp/w", "cat /tmp/w/a.txt", "cat /tmp/w/b.txt",
            "mv /tmp/w/a.txt /tmp/w/c.txt", "mv /tmp/w/b.txt /tmp/w/d.txt",
            "cat /tmp/w/c.txt"]
    seq = {"z_cmd": torch.randn(len(cmds), D, generator=g),
           "z_obs": torch.randn(len(cmds), D, generator=g),
           "steps": [{"cmd": c} for c in cmds], "cmds": cmds, "image": "x"}
    seqs = [seq]
    wins = [{"id": "w0", "si": 0, "r": 5, "N": 2, "R": 6, "depth": 1, "style": "core",
             "expose": {0: 1, 1: 2}, "routed": 0, "name": 0, "movers": [0, 1],
             "mvs": [(3, "/tmp/w/a.txt", "/tmp/w/c.txt"), (4, "/tmp/w/b.txt", "/tmp/w/d.txt")],
             "first_src": 0, "last_src": 1, "last_mover": 1, "deepest": [0]}]
    ctx = CP.build_cups_layout(wins, seqs)
    ctx["seqs"] = seqs
    return ctx, wins, seqs, cmds


def make_cache(ctx, wins, seqs):
    """A swap cache shaped like build_swap_cache's, with the swapped chain retained."""
    g = torch.Generator().manual_seed(11)
    swapped = ["mv /tmp/w/b.txt /tmp/w/c.txt", "mv /tmp/w/a.txt /tmp/w/d.txt"]
    z = torch.randn(2, D, generator=g)
    tok2 = ctx["tok"].clone()
    for kd, (t, _, _) in enumerate(wins[0]["mvs"]):
        tok2[0, 2 * t] = z[kd]
    return {"tok2": tok2, "alts": {0: 1}, "alt_marks": {0: {}}, "idxs": [0],
            "swapped_cmds": {0: swapped}, "swapped_z": {0: z},
            "name_skip": 0, "partner_mover": 1, "self_parity_cos": 1.0,
            "eye_tree_sha": None, "seed": 0}, swapped, z


def main():
    dev = torch.device("cpu")
    ctx, wins, seqs, cmds = make_ctx()
    cache, swapped, z_sw = make_cache(ctx, wins, seqs)

    print("baseline stream is unchanged by the hook")
    tok, tok2 = CP.stream_coded_toks(ctx, cache, BASE)
    check("native tok is the very same object (no clone, no drift)", tok is ctx["tok"])
    check("swap tok2 is the very same object", tok2 is cache["tok2"])
    tokN, tok2N = CP.stream_coded_toks(ctx, cache, None)
    check("a genome with no stream is identical too", tokN is ctx["tok"] and tok2N is cache["tok2"])

    print("\ncoding stream: codes land in BOTH arms, each from its OWN chain")
    cs = make_coding_stream(with_hook=True)
    tok, tok2 = CP.stream_coded_toks(ctx, cache, cs)
    native_ok = all(abs(tok[0, 2 * j, CODE_AT].item() - code_of(cmds[j])) < 1e-4
                    for j in range(6))
    check("native arm carries the native command codes", native_ok)
    mv_pos = [t for (t, _, _) in wins[0]["mvs"]]
    swap_codes = [tok2[0, 2 * t, CODE_AT].item() for t in mv_pos]
    want_swapped = [code_of(c) for c in swapped]
    want_native = [code_of(cmds[t]) for t in mv_pos]
    check("swap arm carries the SWAPPED codes (edge B: native codes in the swapped arm)",
          all(abs(a - b) < 1e-4 for a, b in zip(swap_codes, want_swapped)),
          f"got {swap_codes}, wanted {want_swapped}")
    check("swap arm codes differ from the native ones",
          all(abs(a - b) > 1e-4 for a, b in zip(swap_codes, want_native)))
    nonmv = [j for j in range(6) if j not in mv_pos]
    check("swap arm codes the unchanged commands identically to the native arm",
          all(abs(tok2[0, 2 * j, CODE_AT].item() - code_of(cmds[j])) < 1e-4 for j in nonmv))
    check("edge A: the swap arm is not left code-free",
          all(abs(c) > 1e-9 for c in swap_codes))
    check("swap arm keeps the re-encoded mv embedding outside the coded coordinate",
          torch.allclose(tok2[0, 2 * mv_pos[0], :CODE_AT], z_sw[0, :CODE_AT], atol=1e-6))
    check("observation tokens are untouched in both arms",
          torch.equal(tok[0, 1::2], ctx["tok"][0, 1::2])
          and torch.equal(tok2[0, 1::2], ctx["tok"][0, 1::2]))

    print("\nnon-causal coders are refused")
    m = make_coding_stream(with_hook=True)
    m.code_cmds = lambda cmds, z: _code_cmds([cmds[-1]] * len(cmds), z)
    try:
        CP.stream_coded_toks(ctx, cache, m)
        check("a coder reading a later command raises", False, "no error raised")
    except ValueError as e:
        check("a coder reading a later command raises", "prefix-causal" in str(e))

    print("\nthe gate: verify, don't trust the declared constant")
    ok, why = SG.stream_scoreable(BASE, dev)
    check("baseline passes", ok, why)
    ok, why = SG.stream_scoreable(make_coding_stream(with_hook=True), dev)
    check("coding stream WITH code_cmds passes", ok, why)
    ok, why = SG.stream_scoreable(make_coding_stream(with_hook=False), dev)
    check("coding stream WITHOUT code_cmds is refused (the slot05/19 hole)", not ok)
    check("  and the refusal names the fix", "code_cmds" in why, why)

    bad = make_coding_stream(with_hook=True)
    bad.code_cmds = lambda cmds, z: z
    ok, why = SG.stream_scoreable(bad, dev)
    check("code_cmds that disagrees with its own collate is refused", not ok)

    obs = make_coding_stream(with_hook=True)

    def obs_collate(batch, device):
        b = M.collate(batch, device)
        b["tok"][:, 1::2] = 0.0
        return b
    obs.collate = obs_collate
    ok, why = SG.stream_scoreable(obs, dev)
    check("a stream that edits OBSERVATION tokens is refused", not ok)

    print("\none-armed coding is refused on every route (this regression shipped once)")
    try:
        CP.stream_coded_toks(ctx, None, make_coding_stream(with_hook=True))
        check("coding stream with no swap cache raises", False, "no error")
    except ValueError as e:
        check("coding stream with no swap cache raises", "swap cache" in str(e))
    t, t2 = CP.stream_coded_toks(ctx, None, BASE)
    check("baseline with no swap cache is still the identity", t is ctx["tok"] and t2 is None)

    print("\nstream_matches_context: the decisive check, on real-shaped sequences")
    ok, why = SG.stream_matches_context(BASE, ctx, dev)
    check("baseline passes", ok, why)
    ok, why = SG.stream_matches_context(make_coding_stream(with_hook=True), ctx, dev)
    check("coding stream WITH code_cmds passes", ok, why)
    ok, why = SG.stream_matches_context(make_coding_stream(with_hook=False), ctx, dev)
    check("coding stream WITHOUT code_cmds is refused", not ok)
    check("  and it names code_cmds", "code_cmds" in why, why)

    mut = make_coding_stream(with_hook=True)

    def mutating_collate(batch, device):
        for s in batch:
            s["z_cmd"].mul_(2.0)
        return M.collate(batch, device)
    mut.collate = mutating_collate
    before = ctx["seqs"][0]["z_cmd"].clone()
    ok, why = SG.stream_matches_context(mut, ctx, dev)
    check("a collate that edits its input in place is refused", not ok)
    check("  and the shared context is left unpoisoned",
          torch.equal(ctx["seqs"][0]["z_cmd"], before))

    obs = make_coding_stream(with_hook=True)

    def obs_edit(batch, device):
        b = M.collate(batch, device)
        b["tok"][:, 1::2] += 1.0
        return b
    obs.collate = obs_edit
    ok, why = SG.stream_matches_context(obs, ctx, dev)
    check("an observation coding is refused on real sequences", not ok)

    print("\na coding that resolves the chain into the READ token is refused")
    solver = make_coding_stream(with_hook=True)

    def chain_solving_code(cmds, z_cmd):
        # what the retired stream did: simulate mv symbolically, stamp the resolved identity
        z = z_cmd.clone()
        loc = {}
        for t, c in enumerate(cmds):
            parts = c.split()
            if parts and parts[0] == "mv" and len(parts) == 3:
                loc[parts[2]] = loc.get(parts[1], parts[1])
            elif parts and parts[0] == "cat" and len(parts) == 2:
                z[t, CODE_AT] = code_of(loc.get(parts[1], parts[1]))
        return z
    solver.code_cmds = chain_solving_code
    try:
        CP.stream_coded_toks(ctx, cache, solver)
        check("a chain-resolving coding raises", False, "no error — the hole is open")
    except ValueError as e:
        check("a chain-resolving coding raises", "SCORED READ token" in str(e))
        check("  and it says what is still allowed",
              "move commands" in str(e) or "read alone" in str(e))
    ok, why = CP.stream_coded_toks(ctx, cache, make_coding_stream(with_hook=True))[0] is not None, ""
    check("a move-only coding is still allowed", ok)

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all stream-axis checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
