"""The eval adapter — the search's sole fitness oracle.

Invoked by the evolve engine, once per seed, as:

    python -m eval.adapter {results_dir} {genome} {seed} {split} {mode}

It trains ONE net on the cups pack root and emits {results_dir}/metrics.json:

    combined_score = comp_ca   (evolve/cups_ca.py — the compositional-depth differential)

Everything else the net can tell us rides along in `public` (what inventors get to see) and
`private` (recorded, never briefed). One net per (genome, seed): the compositional metric and
the world-model health readout both come off that same trained net, because training it twice
to measure two things off it is pure waste.

WHY THE TIER DOES NOT CHANGE THE STEP COUNT
  `mode` is reported but never shortens training. A step-reduced proxy has been measured to
  RANK-INVERT exactly the slow-converging memory and architecture mechanisms that a compositional
  objective is about — and the deepest one simply timed out.
  The cheap tier here is fewer SEEDS at full step count, which is configured in evolve.json.
  If you find yourself wanting a shorter proxy, read that lesson again first.

WHAT COUNTS AS WHOSE FAULT
  Environment problems — a missing or half-present data root, an unset or wrong-sha encoder, a
  ceiling table that disagrees with the root — RAISE before any candidate code is reached, so
  they surface as a broken run rather than being recorded as a candidate's null. Candidate
  problems — a genome that will not build, diverges, leaks, or produces a degenerate prediction
  bank — are caught and written as correct:false with the reason, because a failure that is the
  candidate's own is search signal and belongs in the archive.
"""
import copy
import json
import os
import pathlib
import sys
import traceback

import torch

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


def preflight():
    """Environment checks. These RAISE — they are never a candidate's fault."""
    root = _env("TWM_CUPS_ROOT", required=True)
    eye = _env("TWM_EYE", "enc_e5_ft_nocwd_hf")
    rp = pathlib.Path(root)
    if not rp.is_dir():
        raise RuntimeError(f"TWM_CUPS_ROOT={root} is not a directory")
    # Half-present roots are worse than absent ones: the manifests resolve and then the loader
    # dies deep inside a paid run. Check the shards up front.
    for f in ("summary.json", "cache_meta.json", "emb-seq-train.pt", "emb-seq-val.pt",
              "train.jsonl", "val.jsonl"):
        if not (rp / f).exists():
            raise RuntimeError(
                f"{root} is missing {f} — this is an ENCODED pack root, not a raw mint. Build it "
                f"with evolve.reencode (see cloud/pack_lane.sh) before scoring anything.")
    if not CEILING_TABLE.exists():
        raise RuntimeError(f"missing frozen ceiling table {CEILING_TABLE}")

    from evolve import reencode as RE
    percep = RE.load_perception(eye)          # raises if TJ_FT_ENCODER is unset
    want = _env("TWM_EYE_TREE_SHA")
    if want:
        got = RE._checkpoint_tree_sha(percep.MODEL)
        if got != want:
            raise RuntimeError(
                f"encoder tree sha {got} != pinned {want} (TJ_FT_ENCODER={percep.MODEL}). The eye "
                f"IS the frame: a different checkpoint silently re-defines every embedding.")
    return root, eye


def write(results_dir, payload):
    d = pathlib.Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in ("combined_score", "correct") if k in payload}))


def fail(results_dir, reason, detail=""):
    write(results_dir, {"combined_score": None, "correct": False, "error": reason,
                        "text_feedback": f"{reason}: {detail}"[:600] if detail else reason,
                        "public": {}, "private": {"error": reason, "detail": detail[:4000]}})
    return 0        # a candidate failure is a RESULT, not a crash


def main(argv):
    results_dir, genome_path, seed, split, mode = argv[1], argv[2], int(argv[3]), argv[4], argv[5]
    root, eye = preflight()                              # raises on environment problems

    gen = json.load(open(genome_path))
    cells = json.load(open(CEILING_TABLE))["cells"]
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

    # The cups instrument builds a fixed [cmd, obs, cmd, obs, ...] layout and reads the
    # prediction at a strided position. A stream that lays tokens out differently would be scored
    # on a sequence the net never trained on — both arms of the differential wrong, no guard able
    # to see it, and a plausible number written as a pass. Fail closed until the instrument learns
    # to ask the stream where its tokens are.
    if getattr(stream, "CUPS_LAYOUT", None) != "interleave2":
        return fail(results_dir, "stream_layout_unsupported",
                    "the scoring instrument pins a strided [cmd,obs,...] layout; this stream "
                    "declares a different one, so the measurement would not correspond to the "
                    "trained net")

    if not head.leak_safe(head, head_p):
        return fail(results_dir, "head_leak_fail",
                    "the head declares itself unsafe against the no-future-leakage contract")

    try:
        # Frame discipline: the net trains on the pack root's OWN train statistics, and the probe
        # windows are standardized in that same frame. Borrowing another root's stats here is the
        # documented way to collapse two incomparable frames into one silent number.
        #
        # TWM_CONTEXT points at a prebuilt lane context (cloud/build_context.py): the standardized
        # splits, the window layout, and the role-swap chains already synthesized and encoded. All
        # of that is a property of (root, split, eye, swap-seed) and none of it is a property of
        # the genome, so a campaign derives it once and memory-maps it into every worker instead of
        # reloading the encoder and re-encoding every alternative chain per candidate.
        cpath = _env("TWM_CONTEXT")
        if cpath:
            blob = torch.load(cpath, map_location="cpu", weights_only=False, mmap=True)
            if (blob["root"], blob["eye"], blob["split"]) != (root, eye, split):
                raise RuntimeError(
                    f"context {cpath} was built for {(blob['root'], blob['eye'], blob['split'])} "
                    f"but this run is {(root, eye, split)} — a context from another frame would "
                    f"silently score in that frame")
            train_full, ctx, swap_cache = blob["train_full"], blob["ctx"], blob["swap"]
            cdh_ctx = blob.get("cdh")
        else:
            train_full = H._cached_encode(root, "train", eye, device)
            mo, so, mc, sc = M.standardize_stats(train_full)
            M.apply_stats(train_full, mo, so, mc, sc)
            ctx = CP.load_cups_context(root, split, eye, device, stats_data=root)
            swap_cache = None
            cdh_root = _env("TWM_CDH_ROOT")
            cdh_ctx = (CDH.load_cdh_context(cdh_root, split, eye, device, redir_only=True,
                                            stats_data=root) if cdh_root else None)
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

        # A learned target is a REGISTERED child of the net, so `tm.cpu()` is undone by the next
        # `net.to(device)` inside the probe's forward. Take an unregistered copy instead.
        tm = getattr(net, "target_module", None)
        tmod = copy.deepcopy(tm).cpu() if tm is not None else target_mod
        ca = CA.measure_trained_net(net, ctx, tmod, device, eye, cells,
                                    ceiling_table=cells, swap_cache=swap_cache)

        # World-model health on the same net: plain next-observation retrieval against same-verb
        # foils on the pack's own val split. No baseline arms, no content-cell tables — this is a
        # "is this net a usable instrument at all" readout, not a competing objective.
        flat = stream.flatten_predictions(net, H._strip_target_only(ctx["seqs"]), device)
        pred_obs = tmod.to_obs(flat["pred"], flat["prev"]) if tmod is not None else flat["pred"]
        health = M.retrieval(pred_obs, flat["true"], flat["verbs"], seed=seed)

        # A SECOND capability, read off the SAME net and never scored: does this model route a read
        # through the navigation history that actually happened? Its windows were standardized in
        # this net's own frame when the lane context was built, so this measures the net on inputs
        # of the kind it was trained on. It is a transfer reading — the net trained on one pack and
        # is asked about another — and it is here so a skill's trajectory stays visible across the
        # whole search rather than only while it happens to be the objective.
        cdh = None
        if cdh_ctx is not None:
            cdh = {"nav": CDH.nav_probe(net, cdh_ctx, tmod, device),
                   "gate": CDH.masked_s1(net, cdh_ctx, cdh_ctx["fit_ceiling"], tmod, device)}
    except Exception as e:
        return fail(results_dir, f"exception:{type(e).__name__}",
                    f"{e}\n{traceback.format_exc()[-2000:]}")

    g = ca["guards"]
    if not g["norm_ok"] or not g["dispersion_ok"]:
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
            # reported, never scored — see the note at the measurement site
            "cdh_routing": (cdh["nav"]["nav_differential_unmasked_matched"] if cdh else None),
            "steps": steps, "seed": seed, "split": split, "mode": mode,
        },
        # keyed by seed: the engine dict-MERGES private across seeds, so an unkeyed block would
        # let the last seed silently overwrite the others
        "private": {f"seed{seed}": {
            "guards": g,
            "slice": ca["slice"],
            "comp_ca_alt_only": ca["comp_ca_alt_only"],
            "role_swap_seed": ca["role_swap_seed"],
            "gate_report": ca["gate_report"],
            "wm_health": health,
            "cdh": cdh,
            "root": root, "eye": eye,
        }},
        "text_feedback": feedback,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
