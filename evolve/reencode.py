"""Re-encode the raw dockerfs observations/commands with a PERCEPTION recipe (encoder model +
render + pooling) into a NEW dataset root's frozen-embedding caches, so a genome can be scored on
it via `cli score --data <root>`. This is how the `perception` (Tier-1: render/pooling) and
`encoder` (Tier-2: swap the model) chunks are evolved — each recipe is a data-side re-encode, like
the exploration chunk; the retrieval space + baselines are recomputed per root, so fitness stays
the honest content-verb margin within that space. gen-0 (`baseline` perception) reproduces the
original dockerfs cache.

  .venv/bin/python -m evolve.reencode --perception cls --src data/dockerfs --out data/dockerfs-p-cls
"""

import argparse
import hashlib
import importlib
import json
import pathlib
import shutil

import torch

from realenv.seq_worldmodel import D, pick_device
from evolve import bench_versions as BV


def load_perception(name):
    mod = importlib.import_module(f"evolve.chunks.perception.{name}")
    for fn in ("render_obs", "render_cmd", "pool"):
        if not hasattr(mod, fn):
            raise AttributeError(f"perception impl '{name}' missing {fn}")
    return mod


def _content_sha(mod):
    """sha256 of a perception impl's source file (the perception stamp's content_sha)."""
    return hashlib.sha256(pathlib.Path(mod.__file__).read_bytes()).hexdigest()


def _checkpoint_tree_sha(model_name):
    """Deterministic sha256 of a LOCAL checkpoint DIRECTORY's file tree (sorted relpath + per-file
    content hash) — so two different locally-trained encoders (DAPT/contrastive checkpoints) can NOT
    share one perception identity, and a stale/overwritten checkpoint is detectable. Returns None for
    an HF-hub model name (those are pinned by REVISION instead). Encoder-brief provenance blocker."""
    if not model_name:
        return None   # guard: Path("") is Path("."), whose .is_dir() is True → would hash the whole cwd
    p = pathlib.Path(model_name)
    if not p.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(p.rglob("*")):
        # skip encoder_meta.json: it is the provenance STAMP written AFTER this sha is computed, so hashing it
        # would make the value self-referential (never round-trips against the final on-disk tree).
        if f.is_file() and f.name != "encoder_meta.json":
            h.update(str(f.relative_to(p)).encode())
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def _write_cache_meta(src, out):
    """Root-level cache_meta.json {cache_format:3, bench_version, policy_sha, classes_sha,
    built_summary_sha} — the fail-closed format guard. Written ONLY when the SRC
    root is v3-policy; v1/v2 rebuilds write no cache_meta.json, so their byte behavior (and
    scoring) is unchanged. `built_summary_sha` is the sha256 of the OUT summary.json at build
    time — the universal staleness key that seq_worldmodel.cached_encode re-checks so EVERY
    caller (incl. the direct realenv callers) fails closed on a re-mint into an occupied path."""
    if not BV.is_v3_policy(src):
        return
    ssum = pathlib.Path(src) / "summary.json"
    js = json.loads(ssum.read_text()) if ssum.exists() else {}
    out_summary = (pathlib.Path(out) / "summary.json").read_bytes()
    cm = {"cache_format": 3, "bench_version": js.get("bench_version"),
          "policy_sha": js.get("policy_sha"), "classes_sha": js.get("classes_sha"),
          "built_summary_sha": hashlib.sha256(out_summary).hexdigest()}
    (pathlib.Path(out) / "cache_meta.json").write_text(json.dumps(cm, indent=1))


def load_perception_for_root(root):
    """Resolve the perception impl a DERIVED root was built with, from its summary.json perception
    stamp {perception:{impl,model,content_sha}} — used to render/encode the SST &
    within_traj_mut predicted texts in the root's OWN embedding space. Fail-closed on a v3-policy
    root that lacks the stamp (its SST/wtm precompute could not otherwise be render-parity-correct).
    Back-compat: a stamp-less v1/v2 root (e.g. the pre-stamp data/dockerfs-e5) falls back to the
    enc_e5_base default — the recipe every pre-stamp e5 root was in fact built with."""
    summ = pathlib.Path(root) / "summary.json"
    stamp = None
    if summ.exists():
        stamp = json.loads(summ.read_text()).get("perception") or None
    if stamp and stamp.get("impl"):
        return load_perception(stamp["impl"])
    if BV.is_v3_policy(root):
        raise ValueError(f"{root}: v3-policy root without a perception stamp — cannot resolve the "
                         f"root's render/pool recipe for SST/wtm precompute (fail-closed)")
    # stamp-less v1/v2 root: the historical default recipe (all pre-stamp e5 roots used it)
    return load_perception("enc_e5_base")


def _fertility(src, percep, tok, maxlen, sample=3000):
    """Tokenizer-fertility diagnostic (encoder-brief Stage-0 rung-0): mean subword tokens per rendered
    observation + the fraction truncated at maxlen, over a seeded train-split sample. A high-fertility
    tokenizer truncates more content — a stop/go input for the frozen trunk screen. Cheap; stamped."""
    import random
    tf = pathlib.Path(src) / "train.jsonl"
    if not tf.exists():
        return None
    steps = [s for l in open(tf) for s in json.loads(l).get("steps", [])]
    if not steps:
        return None   # degenerate root with no steps → skip (avoids a div-by-zero after embeddings are saved)
    samp = random.Random(0).sample(steps, min(sample, len(steps)))
    ids = tok([percep.render_obs(s) for s in samp], truncation=False)["input_ids"]
    lens = [len(x) for x in ids]
    n = len(lens)
    return {"n_sample": n, "mean_obs_tokens": round(sum(lens) / n, 1),
            "frac_truncated_at_maxlen": round(sum(1 for x in lens if x > maxlen) / n, 4), "maxlen": maxlen}


@torch.no_grad()
def encode_split(src_jsonl, percep, tok, model, device, bs=96, maxlen=256):
    """Length-sorted batched encode (mirrors seq_worldmodel.encode_split so `baseline` is exact).
    maxlen: the tokenizer truncation length — a perception's `MAXLEN` attr (default 256); the same
    value is passed to precompute_baselines so the render/pool/token-budget is identical across the
    eval and its baseline arms (the long-context arm sets MAXLEN=512/1024)."""
    seqs = [json.loads(l) for l in open(src_jsonl)]
    obs_texts, cmd_texts, spans = [], [], []
    for sq in seqs:
        start = len(obs_texts)
        for s in sq["steps"]:
            obs_texts.append(percep.render_obs(s))
            cmd_texts.append(percep.render_cmd(s))
        spans.append((start, len(obs_texts)))

    def enc(texts, tag):
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.zeros(len(texts), D)
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            e = tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                    truncation=True, max_length=maxlen)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            pooled = percep.pool(h, e["attention_mask"]).float().cpu()
            for k, j in enumerate(bidx):
                out[j] = pooled[k]
            if (i // bs) % 50 == 0:
                print(f"  enc {tag} {i}/{len(texts)}", flush=True)
        return out

    z_obs, z_cmd = enc(obs_texts, "obs"), enc(cmd_texts, "cmd")
    out = []
    for (a, b), sq in zip(spans, seqs):
        out.append({"z_obs": z_obs[a:b], "z_cmd": z_cmd[a:b],
                    "cmds": [s["cmd"] for s in sq["steps"]], "image": sq["image"],
                    "ok": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                           for s in sq["steps"]]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--perception", required=True)
    ap.add_argument("--src", default="data/dockerfs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    percep = load_perception(args.perception)
    model_name = getattr(percep, "MODEL", "answerdotai/ModernBERT-base")
    revision = getattr(percep, "REVISION", None)   # immutable HF commit SHA (None = latest); or a local ckpt dir
    maxlen = getattr(percep, "MAXLEN", 256)
    device = pick_device()
    print(f"perception '{args.perception}' | encoder {model_name}@{revision or 'latest'} | "
          f"maxlen {maxlen} | device {device}", flush=True)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Adaptive encode batch: memory ~ bs × maxlen (activations). The length-sorted batcher makes the
    # LONG-obs batches (rare — obs p99≈12410 chars) the OOM risk on a long-context trunk (gte@2048 is 8×
    # the 256-token footprint). Hold bs×maxlen at the 256/bs-96 reference so a max-context reencode fits
    # the same VRAM as the 256 baseline; floor 8. maxlen 256→96, 512→48, 1024→24, 2048→12.
    enc_bs = max(8, (96 * 256) // maxlen)
    if enc_bs != 96:
        print(f"adaptive encode batch: bs={enc_bs} (maxlen {maxlen}, holding bs×maxlen≈{96 * 256})", flush=True)
    for split in ("train", "val"):
        src_jsonl = pathlib.Path(args.src) / f"{split}.jsonl"
        if not src_jsonl.exists():
            # TRAIN-ONLY root tolerance (F6): the ablate raw root ships no val.jsonl — skip the
            # absent split instead of raising (required to build dockerfs3-ablate-e5).
            print(f"skip absent split '{split}' (train-only root, F6)", flush=True)
            continue
        shutil.copy(src_jsonl, outdir / f"{split}.jsonl")
        seqs = encode_split(src_jsonl, percep, tok, model, device, bs=enc_bs, maxlen=maxlen)
        torch.save(seqs, outdir / f"emb-seq-{split}.pt")
        print(f"encoded {split}: {len(seqs)} seqs -> {outdir}/emb-seq-{split}.pt", flush=True)

    # bench-version identity MUST travel with derived roots (review-B2 blocker: a missing summary
    # silently resolves as v1 and disengages v2 classes). Copy the src summary and ADD the
    # perception stamp (harmless/additive on v1/v2; the SST/wtm resolver reads it). The
    # cache_format-3 guard is written ONLY for v3-policy src roots (v1/v2 byte behavior unchanged).
    _s = pathlib.Path(args.src) / "summary.json"
    summ = json.loads(_s.read_text()) if _s.exists() else {}
    summ["perception"] = {"impl": args.perception, "model": model_name,
                          "content_sha": _content_sha(percep), "revision": revision, "maxlen": maxlen}
    ckpt_sha = _checkpoint_tree_sha(model_name)   # non-None only for a LOCAL checkpoint dir
    if ckpt_sha:
        summ["perception"]["checkpoint_tree_sha"] = ckpt_sha
    fert = _fertility(args.src, percep, tok, maxlen)   # brief Stage-0 rung-0 diagnostic (stamped)
    if fert:
        summ["perception"]["fertility"] = fert
        print(f"tokenizer fertility: mean {fert['mean_obs_tokens']} tok/obs, "
              f"{100 * fert['frac_truncated_at_maxlen']:.1f}% truncated at {maxlen} "
              f"(n={fert['n_sample']})", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summ, indent=1))
    _write_cache_meta(args.src, args.out)
    print("REENCODE DONE", flush=True)


if __name__ == "__main__":
    main()
