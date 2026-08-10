"""The blend lane, on synthetic encoded roots.

    PYTHONPATH=<repo> <repo>/.venv/bin/python -m pytest tests/test_blend.py -q
    PYTHONPATH=<repo> <repo>/.venv/bin/python tests/test_blend.py

pytest need not be installed: every check is a plain function with asserts and `main` runs them
all, so the file works either way.

Nothing here touches a real data root. Each fixture root is a few tiny tensors written into a temp
directory with the stamps an encoded root carries, which is enough for the composition path: the
resolver only ever reads caches and a spec.

Each sequence carries a unique tag in its `image` field, so "the composed training set is the base
followed by exactly these pack sequences in this order" is checkable by comparing tag lists.
"""
import hashlib
import json
import pathlib
import sys
import tempfile

import torch

from cloud import build_context as BC
from evolve import blend_root as BR

D = 8                       # the real roots carry 768-wide embeddings; width is irrelevant here
BENCH = {"bench_version": "dockerfs3-v3.0",
         "classes_sha": "c" * 64,
         "policy_sha": "p" * 64,
         "perception": {"impl": "enc_test", "model": "test-eye", "content_sha": "e" * 64}}


def make_root(path, tag, n_train, roles, n_val=0, bench=None):
    """A synthetic ENCODED root: stamps, raw records, and the embedding caches that match them.

    Sequence i of split s is tagged `<tag>:<s>:<i>` in its image field, and every step carries the
    first role in `roles` (cycled), which is what the spec writer counts."""
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    summ = dict(bench or BENCH)
    summ["tag"] = tag
    (p / "summary.json").write_text(json.dumps(summ, indent=1))
    (p / "cache_meta.json").write_text(json.dumps({
        "cache_format": 3,
        "bench_version": summ["bench_version"],
        "classes_sha": summ["classes_sha"],
        "policy_sha": summ["policy_sha"],
        "built_summary_sha": hashlib.sha256((p / "summary.json").read_bytes()).hexdigest()}))

    for split, n in (("train", n_train), ("val", n_val)):
        if not n:
            continue
        recs, seqs = [], []
        for i in range(n):
            image = f"{tag}:{split}:{i}"
            steps = [{"cmd": f"cat /{tag}/{i}/{t}", "output": "x", "exit": 0,
                      "meta": {"role": roles[t % len(roles)], "verb": "cat"}}
                     for t in range(2)]
            recs.append({"image": image, "steps": steps})
            # a distinct constant per sequence, so a mis-ordered composition is visible in the
            # tensors and not only in the tags
            base = float(i + 1) + (0.5 if split == "val" else 0.0)
            seqs.append({"image": image, "cmds": [s["cmd"] for s in steps],
                         "ok": [True] * len(steps),
                         "z_obs": torch.full((len(steps), D), base),
                         "z_cmd": torch.full((len(steps), D), -base)})
        with open(p / f"{split}.jsonl", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        torch.save(seqs, p / f"emb-seq-{split}.pt")
    return str(p)


def fixture(tmp):
    """A base plus two capability packs, each contributing a role the base lacks."""
    base = make_root(pathlib.Path(tmp) / "base", "base", 20, ["base_read"], n_val=6)
    cdh = make_root(pathlib.Path(tmp) / "cdh", "cdh", 12, ["cdh_read", "base_read"])
    cups = make_root(pathlib.Path(tmp) / "cups", "cups", 9, ["cups_test", "base_read"])
    return base, cdh, cups


def compose_tags(spec_root, eye="test-eye"):
    spec = BC.load_blend_spec(spec_root)
    seqs = BC.compose_train_seqs(spec, spec_root, eye, "cpu")
    return spec, seqs, [s["image"] for s in seqs]


def expect(fail_type, fn, *a, **k):
    try:
        fn(*a, **k)
    except fail_type as e:
        return e
    raise AssertionError(f"expected {fail_type.__name__}, nothing raised")


# ---------------------------------------------------------------- the composition

def test_composed_set_is_base_plus_the_sampled_pack_sequences():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, cups = fixture(tmp)
        out = str(pathlib.Path(tmp) / "blend")
        prov = BR.blend(base, [(cdh, 0.25), (cups, 0.2)], out, seed=7)

        spec, seqs, tags = compose_tags(out)
        want = [f"base:train:{i}" for i in range(20)]
        for entry in spec["packs"]:
            tag = json.loads((pathlib.Path(entry["root"]) / "summary.json").read_text())["tag"]
            want += [f"{tag}:train:{i}" for i in entry["indices"]]
        assert tags == want, f"{tags}\n!=\n{want}"

        # round(0.25 * 20) = 5 and round(0.2 * 20) = 4, capped at the pack's own size
        assert [e["seqs_added"] for e in spec["packs"]] == [5, 4]
        assert len(seqs) == 29 == prov["train_seqs"]
        # the base is held FIXED and comes first, in its own order
        assert tags[:20] == want[:20]
        # the tensors travel with the tags: sequence i of a root carries the constant i+1
        for s in seqs:
            i = int(s["image"].rsplit(":", 1)[1])
            assert torch.equal(s["z_obs"], torch.full((2, D), float(i + 1)))


def test_the_same_spec_composes_identically_twice():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, cups = fixture(tmp)
        out = str(pathlib.Path(tmp) / "blend")
        BR.blend(base, [(cdh, 0.25), (cups, 0.2)], out, seed=7)
        _, a, atags = compose_tags(out)
        _, b, btags = compose_tags(out)
        assert atags == btags
        assert all(torch.equal(x["z_obs"], y["z_obs"]) for x, y in zip(a, b))

        # and the spec itself is reproducible: same constituents, same seed, same indices
        out2 = str(pathlib.Path(tmp) / "blend2")
        BR.blend(base, [(cdh, 0.25), (cups, 0.2)], out2, seed=7)
        assert (json.loads((pathlib.Path(out) / "blend.json").read_text())["packs"]
                == json.loads((pathlib.Path(out2) / "blend.json").read_text())["packs"])

        # a different seed draws a different sample (these sizes make a collision unlikely, and
        # the point is only that the seed is live)
        out3 = str(pathlib.Path(tmp) / "blend3")
        BR.blend(base, [(cdh, 0.25), (cups, 0.2)], out3, seed=8)
        assert (json.loads((pathlib.Path(out3) / "blend.json").read_text())["packs"][0]["indices"]
                != json.loads((pathlib.Path(out) / "blend.json").read_text())["packs"][0]["indices"])


def test_indices_that_do_not_fit_the_pack_fail_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, _ = fixture(tmp)
        out = pathlib.Path(tmp) / "blend"
        BR.blend(base, [(cdh, 0.25)], str(out), seed=1)
        good = json.loads((out / "blend.json").read_text())

        def rewrite(mutate):
            spec = json.loads(json.dumps(good))
            mutate(spec)
            (out / "blend.json").write_text(json.dumps(spec))
            return expect(RuntimeError, compose_tags, str(out))

        def out_of_range(spec):
            spec["packs"][0]["indices"][-1] = 999
            spec["packs"][0]["indices_sha256"] = hashlib.sha256(
                json.dumps(spec["packs"][0]["indices"]).encode()).hexdigest()
        assert "does not fit" in str(rewrite(out_of_range))

        def wrong_count(spec):
            spec["packs"][0]["seqs_added"] = 99
        assert "recorded" in str(rewrite(wrong_count))

        def duplicated(spec):
            spec["packs"][0]["indices"][1] = spec["packs"][0]["indices"][0]
            spec["packs"][0]["indices_sha256"] = hashlib.sha256(
                json.dumps(spec["packs"][0]["indices"]).encode()).hexdigest()
        assert "unique and sorted" in str(rewrite(duplicated))

        def tampered_sha(spec):
            # a different but still valid, unique, sorted, in-range selection: it reaches the sha
            # check, which is the only thing left that can tell the two selections apart
            idx = spec["packs"][0]["indices"]
            swap = next(i for i in range(spec["packs"][0]["train_seqs"]) if i not in idx)
            spec["packs"][0]["indices"] = sorted(idx[:-1] + [swap])
        assert "index sha" in str(rewrite(tampered_sha))

        def wrong_pack_size(spec):
            spec["packs"][0]["train_seqs"] = 500
        assert "the spec recorded" in str(rewrite(wrong_pack_size))

        def substituted_root(spec):
            other = make_root(pathlib.Path(tmp) / "other", "other", 12, ["cdh_read"])
            spec["packs"][0]["root"] = other
        assert "not the root the spec was written against" in str(rewrite(substituted_root))


# ---------------------------------------------------------------- the frame

def test_a_blend_without_an_explicit_frame_fails_loudly():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, _ = fixture(tmp)
        out = str(pathlib.Path(tmp) / "blend")
        BR.blend(base, [(cdh, 0.25)], out, seed=0)
        spec = BC.load_blend_spec(out)
        assert spec is not None

        e = expect(RuntimeError, BC.resolve_frame_root, out, spec, None)
        assert "TWM_FRAME_ROOT" in str(e)
        e = expect(RuntimeError, BC.resolve_frame_root, out, spec, "")
        assert "TWM_FRAME_ROOT" in str(e)

        # never falls back to a constituent, and the frame it does take must be encoded
        assert BC.resolve_frame_root(out, spec, base) == base
        assert expect(RuntimeError, BC.resolve_frame_root, out, spec, out)

        # the single-root lane is unchanged: no spec, no frame demanded, the training root frames
        assert BC.load_blend_spec(base) is None
        assert BC.resolve_frame_root(base, None, None) == base


# ---------------------------------------------------------------- what a spec root may hold

def test_the_val_side_is_untouched_by_blending():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, cups = fixture(tmp)
        out = pathlib.Path(tmp) / "blend"
        prov = BR.blend(base, [(cdh, 0.25), (cups, 0.2)], str(out), seed=3)

        # the spec root holds a spec and a summary, and no data of any kind
        assert sorted(q.name for q in out.iterdir()) == ["blend.json", "summary.json"]
        spec = json.loads((out / "blend.json").read_text())
        assert spec["val"] == {"root": base, "blended": False}
        assert prov["val_root"] == base and prov["val_blended"] is False

        # the base's val split is byte-identical to what it was before the blend was written
        val = torch.load(pathlib.Path(base) / "emb-seq-val.pt", weights_only=False)
        assert [s["image"] for s in val] == [f"base:val:{i}" for i in range(6)]
        assert json.loads((out / "summary.json").read_text())["classes_sha"] == BENCH["classes_sha"]

        # a spec root that grew data of its own is refused: its caches could disagree with it
        (out / "emb-seq-train.pt").write_bytes(b"not a cache")
        e = expect(RuntimeError, BC.load_blend_spec, str(out))
        assert "no data" in str(e)
        (out / "emb-seq-train.pt").unlink()

        # so is a spec that claims a blended evaluation split
        spec["val"]["blended"] = True
        (out / "blend.json").write_text(json.dumps(spec))
        assert "frozen reference" in str(expect(RuntimeError, BC.load_blend_spec, str(out)))


# ---------------------------------------------------------------- what the spec writer refuses

def test_the_writer_refuses_an_incoherent_or_inert_pack():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, _ = fixture(tmp)
        out = str(pathlib.Path(tmp) / "blend")

        other_bench = dict(BENCH, classes_sha="d" * 64)
        alien = make_root(pathlib.Path(tmp) / "alien", "alien", 8, ["cdh_read"], bench=other_bench)
        e = expect(SystemExit, BR.blend, base, [(alien, 0.25)], out)
        assert "classes_sha mismatch" in str(e)

        # a pack carrying only roles the base already has contributes nothing distinctive
        twin = make_root(pathlib.Path(tmp) / "twin", "twin", 8, ["base_read"])
        e = expect(SystemExit, BR.blend, base, [(twin, 0.25)], out)
        assert "no step role the base lacks" in str(e)

        # unless it is declared as a deliberately inert control arm
        prov = BR.blend(base, [(twin, 0.25)], out, allow_inert=[twin])
        assert prov["packs"][0]["inert"] is True
        assert prov["packs"][0]["novel_roles"] == []

        # a raw (unencoded) constituent is refused at write time, not at load time
        raw = pathlib.Path(tmp) / "raw"
        raw.mkdir()
        (raw / "summary.json").write_text(json.dumps(BENCH))
        e = expect(SystemExit, BR.blend, base, [(str(raw), 0.25)],
                   str(pathlib.Path(tmp) / "blend-raw"))
        assert "ENCODED caches" in str(e)


def test_the_provenance_states_the_realized_mixture():
    with tempfile.TemporaryDirectory() as tmp:
        base, cdh, cups = fixture(tmp)
        out = str(pathlib.Path(tmp) / "blend")
        prov = BR.blend(base, [(cdh, 0.25), (cups, 0.2)], out, seed=7)

        # per pack: the roles it actually contributed, counted on the sampled sequences
        p_cdh, p_cups = prov["packs"]
        assert p_cdh["novel_roles"] == ["cdh_read"] and p_cups["novel_roles"] == ["cups_test"]
        # each fixture sequence is 2 steps alternating the pack's two roles
        assert p_cdh["roles_added"] == {"base_read": 5, "cdh_read": 5}
        assert p_cups["roles_added"] == {"base_read": 4, "cups_test": 4}
        assert prov["train_steps"] == 2 * 29
        assert prov["train_roles"] == {"base_read": 49, "cdh_read": 5, "cups_test": 4}
        assert p_cdh["novel_role_frac_of_train_steps"] == {"cdh_read": round(5 / 58, 6)}
        assert p_cups["novel_role_frac_of_train_steps"] == {"cups_test": round(4 / 58, 6)}


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main():
    for t in TESTS:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
