"""The eval adapter — the search's fitness oracle. Invoked by the evolve engine once per seed:

    python -m eval.adapter {results_dir} {genome} {seed} {split} {mode}

It trains one net on the training root — the cups pack, or a blend of capability packs when
TWM_TRAIN_ROOT names a blend spec — and writes {results_dir}/metrics.json with

    combined_score = comp_ca   (evolve/cups_ca.py)

plus a `public` block (visible to inventors) and a `private` block (recorded, never briefed).

Environment problems raise; candidate problems are written as correct:false with a reason.
"""
import copy
import json
import os
import pathlib
import sys
import traceback

import torch

from cloud import build_context as BC
from evolve import cdh_probe as CDH, cups_ca as CA, cups_probe as CP, genome as G, harness as H
from realenv import seq_worldmodel as M

REPO = pathlib.Path(__file__).resolve().parent.parent
CEILING_TABLE = REPO / "benchmarks" / "cupsA_ceiling_table.json"


def _env(name, default=None, required=False):
    v = os.environ.get(name, default)
    if required and not v:
        raise RuntimeError(
            f"{name} is not set. The eval runs in a git-less export of HEAD with no environment "
            f"injected by the engine and with the multi-GB data roots living OUTSIDE the repo, "
            f"so every root and checkpoint must arrive through the ambient environment as an "
            f"ABSOLUTE path. See evolve/EVOLVE.md.")
    return v


def _require_encoded(root, var):
    """Raise unless `root` is a complete ENCODED pack root."""
    rp = pathlib.Path(root)
    if not rp.is_dir():
        raise RuntimeError(f"{var}={root} is not a directory")
    for f in ("summary.json", "cache_meta.json", "emb-seq-train.pt", "emb-seq-val.pt",
              "train.jsonl", "val.jsonl"):
        if not (rp / f).exists():
            raise RuntimeError(
                f"{root} is missing {f} — this is an ENCODED pack root, not a raw mint. Build it "
                f"with evolve.reencode (see cloud/pack_lane.sh) before scoring anything.")


def preflight():
    """Environment checks; these raise. Returns (cups pack root, eye, training root, frame root)."""
    root = _env("TWM_CUPS_ROOT", required=True)
    eye = _env("TWM_EYE", "enc_e5_ft_nocwd_hf")
    _require_encoded(root, "TWM_CUPS_ROOT")

    train_root = _env("TWM_TRAIN_ROOT") or root
    frame_root = _env("TWM_FRAME_ROOT")
    blend = BC.load_blend_spec(train_root)
    if blend is None:
        _require_encoded(train_root, "TWM_TRAIN_ROOT")
    else:
        for entry in [blend["base"]] + blend["packs"]:
            _require_encoded(BC.resolve_constituent(entry, train_root, "constituent"),
                             "blend constituent")
    frame_root = BC.resolve_frame_root(train_root, blend, frame_root)
    _require_encoded(frame_root, "TWM_FRAME_ROOT")

    if not CEILING_TABLE.exists():
        raise RuntimeError(f"missing frozen ceiling table {CEILING_TABLE}")

    from evolve import reencode as RE
    percep = RE.load_perception(eye)
    want = _env("TWM_EYE_TREE_SHA")
    if want:
        got = RE._checkpoint_tree_sha(percep.MODEL)
        if got != want:
            raise RuntimeError(
                f"encoder tree sha {got} != pinned {want} (TJ_FT_ENCODER={percep.MODEL}). The eye "
                f"IS the frame: a different checkpoint silently re-defines every embedding.")
    return root, eye, train_root, frame_root


def write(results_dir, payload):
    d = pathlib.Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in ("combined_score", "correct") if k in payload}))


def fail(results_dir, reason, detail=""):
    write(results_dir, {"combined_score": None, "correct": False, "error": reason,
                        "text_feedback": f"{reason}: {detail}"[:600] if detail else reason,
                        "public": {}, "private": {"error": reason, "detail": detail[:4000]}})
    return 0


def main(argv):
    results_dir, genome_path, seed, split, mode = argv[1], argv[2], int(argv[3]), argv[4], argv[5]
    root, eye, train_root, frame_root = preflight()

    gen = json.load(open(genome_path))
    table = json.load(open(CEILING_TABLE))
    cells, knobs = table["cells"], table.get("knobs") or {}
    device = M.pick_device()
    steps = int(_env("TWM_STEPS") or gen.get("chunks", {}).get("optim", {}).get("steps") or 4000)

    try:
        G.validate(gen)
        loss_fn = G.load_objective(gen)
        target_mod = G.load_target(gen)
        stream = G.load_stream(gen)
        head, head_p = G.load_head(gen)
    except Exception as e:
        return fail(results_dir, "genome_invalid", f"{type(e).__name__}: {e}")

    # The scoring instrument reads predictions at strided positions of a fixed [cmd,obs,...]
    # layout; any other layout would be scored on a sequence the net never trained on, and no
    # guard can see that.
    if getattr(stream, "CUPS_LAYOUT", None) != "interleave2":
        return fail(results_dir, "stream_layout_unsupported",
                    "the scoring instrument pins a strided [cmd,obs,...] layout; this stream "
                    "declares a different one, so the measurement would not correspond to the "
                    "trained net")

    if not head.leak_safe(head, head_p):
        return fail(results_dir, "head_leak_fail",
                    "the head declares itself unsafe against the no-future-leakage contract")

    try:
        cpath = _env("TWM_CONTEXT")
        if cpath:
            blob = torch.load(cpath, map_location="cpu", weights_only=False, mmap=True)
            built = (blob["root"], blob["eye"], blob["split"],
                     blob.get("train_root", blob["root"]),
                     blob.get("frame_root", blob["root"]))
            here = (root, eye, split, train_root, frame_root)
            if built != here:
                raise RuntimeError(
                    f"context {cpath} was built for {built} but this run is {here} — a context "
                    f"from another frame would silently score in that frame")
            train_full, ctx, swap_cache = blob["train_full"], blob["ctx"], blob["swap"]
            cdh_ctx = blob.get("cdh")
        else:
            blend = BC.load_blend_spec(train_root)
            train_full = (BC.compose_train_seqs(blend, train_root, eye, device) if blend
                          else H._cached_encode(train_root, "train", eye, device))
            stats_src = (train_full if not blend and frame_root == train_root
                         else H._cached_encode(frame_root, "train", eye, device))
            mo, so, mc, sc = M.standardize_stats(stats_src)
            M.apply_stats(train_full, mo, so, mc, sc)
            ctx = CP.load_cups_context(root, split, eye, device, stats_data=frame_root)
            swap_cache = None
            cdh_root = _env("TWM_CDH_ROOT")
            cdh_ctx = (CDH.load_cdh_context(cdh_root, split, eye, device, redir_only=True,
                                            stats_data=frame_root) if cdh_root else None)
        if ctx is None:
            raise RuntimeError(f"no cups windows in the {split} split of {root}")
    except Exception as e:
        raise RuntimeError(f"pack-lane setup failed (environment, not candidate): {e}") from e

    try:
        fit, _ = M.split_train_dev(train_full, seed=seed)
        net, ok = H._train(gen, fit, device, loss_fn, seed, steps, target_mod, stream,
                           head, head_p)
        if not ok:
            return fail(results_dir, "train_diverged", "non-finite loss during training")
        if not stream.leakage_ok(net, device):
            return fail(results_dir, "leakage_fail",
                        "perturbing a later observation moved an earlier command's prediction")

        # A learned target is a REGISTERED child of the net, so tm.cpu() is undone by the next
        # net.to(device) inside the probe's forward. Take an unregistered copy instead.
        tm = getattr(net, "target_module", None)
        tmod = copy.deepcopy(tm).cpu() if tm is not None else target_mod
        ca = CA.measure_trained_net(net, ctx, tmod, device, eye, cells,
                                    ceiling_table=cells, swap_cache=swap_cache, knobs=knobs)

        flat = stream.flatten_predictions(net, H._strip_target_only(ctx["seqs"]), device)
        pred_obs = tmod.to_obs(flat["pred"], flat["prev"]) if tmod is not None else flat["pred"]
        health = M.retrieval(pred_obs, flat["true"], flat["verbs"], seed=seed)

        cdh = None
        if cdh_ctx is not None:
            cdh = {"nav": CDH.nav_probe(net, cdh_ctx, tmod, device),
                   "gate": CDH.masked_s1(net, cdh_ctx, cdh_ctx["fit_ceiling"], tmod, device)}
    except Exception as e:
        return fail(results_dir, f"exception:{type(e).__name__}",
                    f"{e}\n{traceback.format_exc()[-2000:]}")

    g = ca["guards"]
    if g["norm_ok"] is False or g["dispersion_ok"] is False:
        return fail(results_dir, "degenerate_prediction_bank",
                    f"norm_over_bank={g['norm_over_bank']}, "
                    f"angular_dispersion={g['angular_dispersion']} — the prediction bank is "
                    f"collapsed or constant, so the differential is not measuring tracking")

    pd = ca["per_depth"]
    feedback = (
        f"comp_ca {ca['comp_ca']:+.4f} over n={ca['n']} deep earnable windows "
        f"(for reference, the strongest analytic non-tracker on the same windows, "
        f"{ca['best_analytic_arm']}, sits at "
        f"{ca['analytic_band'][ca['best_analytic_arm']]:+.4f}) "
        f"(d2 n={pd['d2']['n']}, d3 n={pd['d3']['n']}, d4+ n={pd['d4plus']['n']}); "
        f"native picks {ca['native_wm']:.3f} vs chance {g['chance']:.3f}; "
        f"under role-swap the same pick is held {ca['swap_stayed']:.3f} and follows the swapped "
        f"content {ca['swap_follow']:.3f}. Next-obs retrieval health "
        f"{health['top1_sameverb']:.3f}."
        + (f" Command-history routing on the other capability pack, reported not scored: "
           f"{cdh['nav']['nav_differential_unmasked_matched']:+.3f}." if cdh else ""))

    write(results_dir, {
        "combined_score": ca["comp_ca"],
        "correct": True,
        "public": {
            "comp_ca": ca["comp_ca"],
            "analytic_band": ca["analytic_band"],
            "best_analytic_arm": ca["best_analytic_arm"],
            "n_windows": ca["n"],
            "per_depth": {k: v["comp_ca"] for k, v in pd.items()},
            "per_depth_n": {k: v["n"] for k, v in pd.items()},
            "native_wm": ca["native_wm"],
            "swap_stayed": ca["swap_stayed"],
            "swap_follow": ca["swap_follow"],
            "chance": g["chance"],
            "wm_health_top1_sameverb": health["top1_sameverb"],
            "cdh_routing": (cdh["nav"]["nav_differential_unmasked_matched"] if cdh else None),
            "steps": steps, "seed": seed, "split": split, "mode": mode,
        },
        # Keyed by seed: the engine dict-MERGES private across seeds, so an unkeyed block would
        # let the last seed silently overwrite the others.
        "private": {f"seed{seed}": {
            "guards": g,
            "slice": ca["slice"],
            "comp_ca_alt_only": ca["comp_ca_alt_only"],
            "role_swap_seed": ca["role_swap_seed"],
            "gate_report": ca["gate_report"],
            "wm_health": health,
            "cdh": cdh,
            "root": root, "eye": eye,
            "train_root": train_root, "frame_root": frame_root,
            "n_train_seqs": len(train_full),
        }},
        "text_feedback": feedback,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
