"""blend_root — write a blend SPEC: one training set assembled from several capability packs.

    python -m evolve.blend_root --base <encoded root> \
        --pack <encoded root>:<ratio> [--pack <encoded root>:<ratio> ...] \
        --out <spec root> [--seed 0] [--allow-inert <pack root>]

A blend is a SPEC, not a dataset. This writes exactly two files and nothing else:

    <out>/blend.json    the constituents, the per-pack ratio, the sample seed, and the exact
                        sampled sequence indices
    <out>/summary.json  the base root's summary with a `blend` provenance block added

The training set is composed at load time from the constituents' already-encoded caches
(cloud/build_context.py resolves the spec); there is no mode here that materializes train.jsonl.

The blend is additive: the base is held fixed and each pack contributes
`round(ratio * |base train seqs|)` sequences, sampled without replacement with a seeded RNG and
recorded as sorted indices. The evaluation split is never blended: it is the base's val, untouched.

The provenance block records, per pack, the step roles that pack contributes and how many steps of
each landed in the mixture. `novel_roles` are the roles a pack brings that the base does not
already carry; a pack with none is refused unless it is named to --allow-inert.
"""
import argparse
import hashlib
import json
import pathlib
import random

REQUIRED_FILES = ("summary.json", "cache_meta.json", "emb-seq-train.pt", "train.jsonl")

IDENTITY_KEYS = ("bench_version", "classes_sha", "policy_sha")


def _summary(root):
    return json.loads((pathlib.Path(root) / "summary.json").read_text())


def _sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _require_encoded(root, what):
    p = pathlib.Path(root)
    if not p.is_dir():
        raise SystemExit(f"{what} {root} is not a directory")
    missing = [f for f in REQUIRED_FILES if not (p / f).exists()]
    if missing:
        raise SystemExit(
            f"{what} {root} is missing {', '.join(missing)} — a blend composes from ENCODED "
            f"caches, so every constituent must be an encoded root (build it with "
            f"evolve.reencode first)")


def _scan_train(root):
    """(n_seqs, n_steps, {role: count}) over a root's train records; steps with no meta.role are
    counted under the empty key."""
    n_seqs = n_steps = 0
    roles = {}
    with open(pathlib.Path(root) / "train.jsonl") as fh:
        for line in fh:
            seq = json.loads(line)
            n_seqs += 1
            for st in seq.get("steps", []):
                n_steps += 1
                r = (st.get("meta") or {}).get("role") or ""
                roles[r] = roles.get(r, 0) + 1
    return n_seqs, n_steps, roles


def _scan_selected(root, indices):
    """(n_steps, {role: count}) over only the sequences at `indices` (sorted, unique)."""
    want = set(indices)
    n_steps = 0
    roles = {}
    with open(pathlib.Path(root) / "train.jsonl") as fh:
        for i, line in enumerate(fh):
            if i not in want:
                continue
            for st in json.loads(line).get("steps", []):
                n_steps += 1
                r = (st.get("meta") or {}).get("role") or ""
                roles[r] = roles.get(r, 0) + 1
    return n_steps, roles


def blend(base, packs, out, seed=0, allow_inert=()):
    """Write the blend spec for `packs` = [(root, ratio), ...]; returns the provenance block."""
    _require_encoded(base, "base")
    if not packs:
        raise SystemExit("no packs given — a blend with no pack is just the base root")
    for root, ratio in packs:
        _require_encoded(root, "pack")
        if not (ratio > 0):
            raise SystemExit(f"pack {root} ratio {ratio} must be > 0")

    b_sum = _summary(base)
    if not (pathlib.Path(base) / "val.jsonl").exists():
        raise SystemExit(
            f"base {base} has no val.jsonl — the base supplies the FROZEN evaluation split, which "
            f"blending never touches, so a base without one cannot anchor a blend")

    for root, _ in packs:
        p_sum = _summary(root)
        for k in IDENTITY_KEYS:
            if b_sum.get(k) != p_sum.get(k):
                raise SystemExit(
                    f"base/pack {k} mismatch: base={b_sum.get(k)!r} {root}={p_sum.get(k)!r} — the "
                    f"pack was minted against a different bench")

    base_seqs, base_steps, base_roles = _scan_train(base)
    inert_ok = {str(r) for r in allow_inert}

    spec_packs, prov_packs = [], []
    train_steps = base_steps
    train_roles = dict(base_roles)
    for pos, (root, ratio) in enumerate(packs):
        pack_seqs, _, pack_roles = _scan_train(root)
        novel = sorted(r for r, n in pack_roles.items() if n and r not in base_roles)
        if not novel and str(root) not in inert_ok:
            raise SystemExit(
                f"pack {root} carries no step role the base lacks — it contributes nothing "
                f"distinctive to the mixture. If this is a deliberately inert control arm "
                f"(token-matched extra base material), say so with --allow-inert {root}")

        n_add = min(round(ratio * base_seqs), pack_seqs)
        if n_add <= 0:
            raise SystemExit(f"pack {root} ratio {ratio} adds 0 of {base_seqs} base seqs")
        # The per-pack RNG stream is derived from the seed and the pack's POSITION, never from a
        # path, so the draw survives the roots moving and appending a pack leaves the earlier
        # packs' samples untouched.
        rng = random.Random(f"twm-blend/{seed}/{pos}")
        idx = sorted(rng.sample(range(pack_seqs), n_add))
        steps_added, roles_added = _scan_selected(root, idx)
        train_steps += steps_added
        for r, n in roles_added.items():
            train_roles[r] = train_roles.get(r, 0) + n

        spec_packs.append({
            "root": str(root),
            "summary_sha256": _sha256_file(pathlib.Path(root) / "summary.json"),
            "ratio": round(float(ratio), 6),
            "train_seqs": pack_seqs,
            "seqs_added": n_add,
            "indices": idx,
            "indices_sha256": hashlib.sha256(json.dumps(idx).encode()).hexdigest(),
        })
        prov_packs.append({
            "root": str(root), "ratio": round(float(ratio), 6),
            "seqs_total": pack_seqs, "seqs_added": n_add,
            "steps_added": steps_added,
            "roles_added": dict(sorted(roles_added.items())),
            "novel_roles": novel,
            "inert": not novel,
        })

    for p in prov_packs:
        p["novel_role_frac_of_train_steps"] = {
            r: round(p["roles_added"].get(r, 0) / train_steps, 6) for r in p["novel_roles"]}

    prov = {
        "base": str(base), "base_seqs": base_seqs, "base_steps": base_steps,
        "packs": prov_packs,
        "sample_seed": seed,
        "train_seqs": base_seqs + sum(p["seqs_added"] for p in prov_packs),
        "train_steps": train_steps,
        "train_roles": dict(sorted(train_roles.items())),
        "val_root": str(base), "val_blended": False,
    }

    spec = {
        "format": "twm-blend/1",
        "base": {"root": str(base),
                 "summary_sha256": _sha256_file(pathlib.Path(base) / "summary.json"),
                 "train_seqs": base_seqs},
        "packs": spec_packs,
        "sample_seed": seed,
        "val": {"root": str(base), "blended": False},
        "provenance": prov,
    }

    outp = pathlib.Path(out)
    outp.mkdir(parents=True, exist_ok=True)
    for stray in ("train.jsonl", "val.jsonl", "emb-seq-train.pt", "emb-seq-val.pt"):
        if (outp / stray).exists():
            raise SystemExit(
                f"{out} already holds {stray} — a blend spec root must carry NO data of its own, "
                f"or its caches and its spec can disagree silently. Write the spec elsewhere.")
    summ = dict(b_sum)
    summ["blend"] = prov
    (outp / "summary.json").write_text(json.dumps(summ, indent=1))
    (outp / "blend.json").write_text(json.dumps(spec, indent=1))
    return prov


def _parse_pack(s):
    root, sep, ratio = s.rpartition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"--pack wants <root>:<ratio>, got {s!r}")
    try:
        return root, float(ratio)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--pack ratio {ratio!r} is not a number")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True,
                    help="the encoded base root, held fixed; its val is the frozen eval split")
    ap.add_argument("--pack", required=True, action="append", type=_parse_pack, metavar="ROOT:RATIO",
                    help="an encoded capability pack and its ratio (seqs added = "
                         "round(ratio * |base train seqs|)); repeatable")
    ap.add_argument("--out", required=True, help="output spec root (holds blend.json only)")
    ap.add_argument("--seed", type=int, default=0, help="deterministic pack-sample seed")
    ap.add_argument("--allow-inert", action="append", default=[], metavar="ROOT",
                    help="permit this pack to contribute no role the base lacks (a deliberately "
                         "inert, token-matched control arm)")
    a = ap.parse_args(argv)
    prov = blend(a.base, a.pack, a.out, seed=a.seed, allow_inert=a.allow_inert)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
