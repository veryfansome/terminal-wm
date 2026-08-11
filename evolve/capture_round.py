"""Copy a round's inventor output into the repo, verbatim, as soon as the inventors finish.

    python -m evolve.capture_round <round-tag> [<jail-root>]

A hypothesis is the author's claim about what their code does. It is written into a jail under
the engine's cache, which is rebuilt with --force and is not a durable record. Waiting until
score time to keep it loses every candidate that never reaches an archive row: a guard
rejection and a dedup rejection are session-local, the engine does not archive them, and a
round that is abandoned or interrupted archives nothing at all.

So this runs right after the inventor workflow, before anything is scored, and copies each
slot's hypothesis, genome and impls under evolve/rounds/<tag>/. Nothing is rewritten or
summarised. Candidates that go on to be scored should also carry their hypothesis into the
archive row via `evolve score --rationale`, which is a separate act on a separate schedule.
"""
import pathlib
import shutil
import sys

CACHE = pathlib.Path.home() / ".cache" / "evolve-jails"
DEST = pathlib.Path("evolve") / "rounds"


def _find_root(tag, given=None):
    if given:
        p = pathlib.Path(given)
        return p if (p / tag).is_dir() else p
    hits = [d / tag for d in CACHE.glob("*") if (d / tag).is_dir()]
    if not hits:
        raise SystemExit(f"no jail root under {CACHE} holds round {tag!r} — pass one explicitly")
    if len(hits) > 1:
        raise SystemExit(f"round {tag!r} exists under several jail roots: {hits} — pass one")
    return hits[0]


def capture(tag, jail_root=None):
    src = _find_root(tag, jail_root)
    slots = sorted(d for d in src.glob("slot*") if d.is_dir())
    if not slots:
        raise SystemExit(f"{src} holds no slot directories")

    out = DEST / tag
    out.mkdir(parents=True, exist_ok=True)
    kept, missing = 0, []
    for d in slots:
        prop = d / "PROPOSAL"
        if not (prop / "hypothesis.txt").exists():
            missing.append(d.name)
            continue
        dst = out / d.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(prop.iterdir()):
            if f.is_file() and (f.suffix in (".py", ".json") or f.name == "hypothesis.txt"):
                shutil.copy(f, dst / f.name)
        if (d / "BRIEF.md").exists():
            shutil.copy(d / "BRIEF.md", dst / "BRIEF.md")
        kept += 1

    print(f"captured {kept}/{len(slots)} slots of round {tag} -> {out}")
    if missing:
        print(f"no hypothesis.txt (inventor did not finish): {missing}")
    print("these are the authors' claims, verbatim; annotate alongside, never rewrite in place")
    return 0


if __name__ == "__main__":
    sys.exit(capture(*sys.argv[1:]))
