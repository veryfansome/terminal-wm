"""build_twins — regenerate the one-off val twins from a cups root, deterministically.

    python -m evolve.build_twins --src <encoded root> --outdir data --arms IDENT RANDPa ...

Each twin is the SAME val split with one property of the mv-destination tag changed, so a score
difference is attributable to that property and nothing else. Only commands are rewritten; no
observation mentions a cups path, so every observation embedding is unchanged by construction.

The tag matters because the pack prints each move's ordinal into its destination filename: the
trailing ".N" is that move's position in the suffixed chain, in every window. comp_ca cancels
location keying structurally but cancels move-ORDER keying only in expectation, so a printed
ordinal is a cue the metric does not remove.

What each arm does, and what it is FOR:

  IDENT       nothing changes. The pipeline null: it must reproduce the source root exactly, and
              if it does not, the harness is wrong rather than the manipulation interesting.
  RANDPa/b    a fresh random bijection of {1..k} per sequence. The tag stays a digit from the same
              alphabet and stays unique, so a code that hashes or one-hots the digit sees a
              DIFFERENT tag of the same kind. Two draws so the realization noise is visible.
  DERANGE     as RANDP but no digit may keep its value. Anti-correlated with position rather than
              neutral, so it bounds rather than estimates.
  OFFSET      every digit +1. Order preserved in the STRING; for any code that treats the tag as a
              category this is just another relabel, which is why it is not an order control.
  SHIFTSCRAM  random bijection into {2..k+1}: OFFSET's alphabet, order-neutral for k>2 and
              deterministically order-REVERSED at k=2, where rejecting only the OFFSET map leaves
              {1:3, 2:2} as the unique alternative. Measured over the scored contexts it rewrites
              1031 of 1162 tagged cups tokens (0.8873) against OFFSET's 1162 of 1162, because a
              bijection into a shifted alphabet permits per-digit fixed points; per k that is 0.8804
              at k=6 and 0.8911 at k=8. Two residual asymmetries sign OFFSET-minus-SHIFTSCRAM in
              OPPOSITE directions: the 11.27% digit fixed points bias it down, and emitting 90 novel
              ".9" tokens against OFFSET's 70 biases it up.

  NOTE ON ORDER: no pair in this set isolates order at matched alphabet AND matched coverage.
  OFFSET/SHIFTSCRAM share an alphabet but not coverage; OFFSET/DERANGE match coverage (1162/1162,
  zero fixed points) but differ in alphabet, and the alphabet effect is itself large and k-dependent.
  Read any order claim from these arms as bounded, not isolated.
  STEM        the stem of already-tagged paths is mutated, the digit untouched. Breaks the
              coreference between generations of one location while leaving the ordinal readable.
  NOSUFFIX    the digit becomes a single letter. Unique, same length, chain intact — but a parser
              looking for digits finds none, so a generation-indexed rail goes to zero. Note this
              also breaks stem coreference, so it lesions two things at once.
  MUGS        the mount point is renamed, tags untouched. A large, broad, ordinal-preserving
              perturbation: the yardstick for what any big surface change costs on its own.

Seeds are fixed so a rebuild anywhere reproduces the same roots.
"""

import argparse
import json
import pathlib
import random
import re

CUPS = re.compile(r"^(/tmp/w/cups/\S*?)\.(\d+)$")
STEM_PAT = re.compile(r"^(/tmp/w/cups/.*/)([^/]+)\.(\d+)$")
ALPHA = "abcdefghijklmnop"

SEEDS = {"RANDPa": 20260815, "RANDPb": 777001, "DERANGE": 20260815, "SHIFTSCRAM": 4242}


def _perm_random(k, rng):
    p = list(range(1, k + 1)); rng.shuffle(p)
    return {i + 1: p[i] for i in range(k)}


def _perm_derange(k, rng):
    if k == 1:
        return {1: 1}
    while True:
        p = list(range(1, k + 1)); rng.shuffle(p)
        if all(p[i] != i + 1 for i in range(k)):
            return {i + 1: p[i] for i in range(k)}


def _perm_shiftscram(k, rng):
    tgt = list(range(2, k + 2))
    while True:
        p = tgt[:]; rng.shuffle(p)
        mp = {i + 1: p[i] for i in range(k)}
        if k == 1 or any(mp[i] != i + 1 for i in mp):
            return mp


def _digits_of(steps):
    d = set()
    for s in steps:
        for t in (s.get("cmd") or "").split():
            m = CUPS.match(t)
            if m:
                d.add(int(m.group(2)))
    return d


def build(arm, src_jsonl, out_jsonl):
    rng = random.Random(SEEDS.get(arm, 0))
    n = 0
    with open(out_jsonl, "w") as fh:
        for line in open(src_jsonl):
            d = json.loads(line)
            digits = _digits_of(d["steps"])
            k = max(digits) if digits else 0
            perm = None
            if k:
                if arm in ("RANDPa", "RANDPb"):
                    perm = _perm_random(k, rng)
                elif arm == "DERANGE":
                    perm = _perm_derange(k, rng)
                elif arm == "SHIFTSCRAM":
                    perm = _perm_shiftscram(k, rng)
                elif arm == "OFFSET":
                    perm = {i: i + 1 for i in range(1, k + 1)}
                elif arm == "NOSUFFIX":
                    perm = {i: ALPHA[i - 1] for i in range(1, k + 1)}
            for s in d["steps"]:
                c = s.get("cmd")
                if not c:
                    continue
                # Split KEEPING the separators. c.split() + " ".join() silently normalized every
                # run of whitespace in every command of every arm, so IDENT was not byte-exact
                # against its source and each other arm carried that same uncontrolled edit on top
                # of the one it is supposed to isolate.
                parts = re.split(r"(\s+)", c)
                for i, tok in enumerate(parts):
                    if not tok or tok.isspace():
                        continue
                    if arm == "STEM":
                        m = STEM_PAT.match(tok)
                        if m:
                            parts[i] = f"{m.group(1)}{m.group(2)}Q.{m.group(3)}"; n += 1
                        continue
                    m = CUPS.match(tok)
                    if m and perm is not None:
                        parts[i] = f"{m.group(1)}.{perm[int(m.group(2))]}"; n += 1
                c2 = "".join(parts)
                if arm == "MUGS":
                    n += c2.count("/tmp/w/cups/")
                    c2 = c2.replace("/tmp/w/cups/", "/tmp/w/mugs/")
                s["cmd"] = c2
            fh.write(json.dumps(d) + "\n")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args(argv)
    src = pathlib.Path(a.src)
    for arm in a.arms:
        d = pathlib.Path(a.outdir) / "_twin_src" / arm
        d.mkdir(parents=True, exist_ok=True)
        n = build(arm, src / "val.jsonl", d / "val.jsonl")
        (d / "summary.json").write_bytes((src / "summary.json").read_bytes())
        print(f"  {arm:11s} {n} tokens rewritten -> {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
