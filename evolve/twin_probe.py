"""twin_probe — one trained net, measured against several val roots that differ in one variable.

    TWM_CUPS_ROOT=<original> .venv/bin/python -m evolve.twin_probe \
        --genome evolve/genomes/<id>.json --seed 0 --roots <name>=<root> [...] --out <dir>

Why this exists rather than running the scoring adapter once per root: the adapter trains in
process and persists no checkpoint, so measuring N roots that way is N independent trainings and
the comparison is governed by run-to-run training noise. Measured over the archive, the run-to-run
SD of a 3-seed comp_ca mean is up to 0.037, which swamps the effect being looked for. Here the net
is trained ONCE and the same weights are measured against every root, so the training term is
identically zero and the only variance left is which of the shared scored windows flip.

Every root is measured with the standardization frame pinned to --frame (the original), so the
frame is held fixed while the val commands vary. The roots must agree on the scored window set;
that is asserted, not assumed.

This is an instrument for a one-off question. It writes a plain JSON record and touches nothing
the search reads.
"""

import argparse
import copy
import inspect
import json
import pathlib
import time

import torch

from evolve import cups_ca as CA
from evolve import cups_probe as CP
from evolve import genome as G
from evolve import harness as H
from realenv import seq_worldmodel as M

# cups_ca's own default, read from its signature. A hardcoded copy would drift silently, and a
# mismatched partner draw would make the per-window rows describe a different measurement than the
# comp_ca they sit beside — wrong with no error raised.
SWAP_SEED = inspect.signature(CA.measure_trained_net).parameters["seed"].default

REPO = pathlib.Path(__file__).resolve().parent.parent
CEILING_TABLE = REPO / "benchmarks" / "cupsA_ceiling_table.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--frame", required=True, help="root whose TRAIN stats standardize every arm")
    ap.add_argument("--train-root", default=None, help="defaults to --frame")
    ap.add_argument("--roots", nargs="+", required=True, help="name=path, measured in order")
    ap.add_argument("--split", default="inner")
    ap.add_argument("--eye", default="enc_e5_ft_nocwd_hf")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    frame = a.frame
    train_root = a.train_root or frame
    roots = [r.split("=", 1) for r in a.roots]
    device = M.pick_device()
    gen = json.load(open(a.genome))
    table = json.loads(CEILING_TABLE.read_text())
    cells, knobs = table["cells"], table.get("knobs") or {}
    steps = a.steps or int(gen.get("chunks", {}).get("optim", {}).get("steps") or 4000)

    loss_fn = G.load_objective(gen)
    target_mod = G.load_target(gen)
    stream = G.load_stream(gen)
    head, head_p = G.load_head(gen)

    train_full = H._cached_encode(train_root, "train", a.eye, device)
    mo, so, mc, sc = M.standardize_stats(H._cached_encode(frame, "train", a.eye, device))
    M.apply_stats(train_full, mo, so, mc, sc)

    contexts = []
    for name, root in roots:
        ctx = CP.load_cups_context(root, a.split, a.eye, device, stats_data=frame)
        if ctx is None:
            raise RuntimeError(f"no cups windows in {a.split} of {root}")
        contexts.append((name, root, ctx))

    # A drop is only attributable to the manipulation if every arm scores the SAME WINDOWS. The
    # harvested count agreeing is not that guarantee: eligibility depends on each window's stamped
    # cell and on having a legal role-swap partner, so the sets can differ while the counts match.
    ref_name, _, ref_ctx = contexts[0]
    ref_ids = [w["id"] for w in ref_ctx["wins"]]
    for name, root, ctx in contexts[1:]:
        ids = [w["id"] for w in ctx["wins"]]
        if ids != ref_ids:
            raise RuntimeError(
                f"harvested window IDs differ: {ref_name} vs {name} — the roots are not measuring "
                f"the same slice, so no comparison between them means anything")

    t0 = time.time()
    fit, _ = M.split_train_dev(train_full, seed=a.seed)
    net, ok = H._train(gen, fit, device, loss_fn, a.seed, steps, target_mod, stream, head, head_p)
    train_s = time.time() - t0
    if not ok:
        raise RuntimeError("non-finite loss during training")

    tm = getattr(net, "target_module", None)
    tmod = copy.deepcopy(tm).cpu() if tm is not None else target_mod

    out = {"genome": a.genome, "id": pathlib.Path(a.genome).stem, "seed": a.seed,
           "frame": frame, "train_root": train_root, "split": a.split, "eye": a.eye,
           "steps": steps, "device": str(device), "train_seconds": round(train_s, 1),
           "n_harvested_windows": len(ref_ids), "arms": {}}

    # The destination is opened before the loop and rewritten after EVERY arm, atomically. Writing
    # once at the end meant any raise inside the loop discarded the training (149-765s) and every arm
    # already measured — and the arm ordered last is the control the campaign exists to obtain.
    o = pathlib.Path(a.out); o.mkdir(parents=True, exist_ok=True)
    dest = o / f"{out['id']}.seed{a.seed}.json"
    tmp = dest.with_suffix(".part")

    def _flush():
        tmp.write_text(json.dumps(out, indent=1))
        tmp.replace(dest)

    ref_coded = None
    for ai, (name, root, ctx) in enumerate(contexts):
        # ONE cache, built here and handed to both the aggregate and the row recompute, at the seed
        # cups_ca itself uses. Previously each built its own and they agreed only by coincidence of
        # defaults; that also paid the ~8s build twice per arm.
        swap_cache = CP.build_swap_cache(ctx, a.eye, device, seed=SWAP_SEED)
        ca = CA.measure_trained_net(net, ctx, tmod, device, a.eye, cells,
                                    ceiling_table=cells, knobs=knobs, stream=stream,
                                    seed=SWAP_SEED, swap_cache=swap_cache)
        pub = ca.get("public", ca)

        # Per-window rows, recomputed the way cups_ca does internally (it computes them and returns
        # only means). Because ONE net is measured against every arm, the arms are paired at the
        # window level; comparing only the means throws that pairing away, and at a 1/89 quantum
        # knowing WHICH windows flipped is the actual evidence.
        tok, tok2 = CP.stream_coded_toks(ctx, swap_cache, stream)
        # How hard THIS arm perturbs THIS genome's own coded tokens. The arms damage different
        # rails by different amounts and a drop is only attributable once that is on the record:
        # e.g. a permuted ordinal leaves r7-26's cup-coreference rail at cosine 1.0 while an
        # alpha tag destroys it, so the two arms are not asking the same question of that genome.
        # The reference is the first arm's CODED tokens, not ctx["tok"]. ctx["tok"] is the raw
        # standardized command embedding; stream_coded_toks returns the genome's coding of it. For
        # any genome declaring code_cmds those differ, so referencing ctx["tok"] measured coding
        # against no-coding and folded the stream's own transform into a number whose whole job is
        # to isolate the ARM — wrong precisely for the stream candidates this instrument exists to
        # separate. The reference arm records null, not zero: it was never measured, and a 0.0
        # there is indistinguishable from an arm that perturbed nothing.
        if ai == 0:
            ref_coded, code_pert = tok, None
        elif tok is None or ref_coded is None or ref_coded.shape != tok.shape:
            raise RuntimeError(
                f"{name}: coded command tokens shaped "
                f"{None if tok is None else tuple(tok.shape)} against the reference arm's "
                f"{None if ref_coded is None else tuple(ref_coded.shape)}. Every arm harvests the "
                f"same window IDs, so this cannot happen unless the arms are measuring different "
                f"slices — which would invalidate every cross-arm comparison in this record.")
        else:
            d = (tok - ref_coded).reshape(-1, tok.shape[-1]).norm(dim=-1)
            nz = d[d > 0]
            code_pert = {"frac_tokens_changed": float((d > 0).float().mean()),
                         "mean_L2_over_changed": float(nz.mean()) if nz.numel() else 0.0}
        cap = CP.measure(net, ctx, tmod, device, ceiling_table=cells, tok=tok)
        alt = CP.alt_chain(net, ctx, tmod, device, a.eye, seed=SWAP_SEED, ceiling_table=cells,
                           cache=swap_cache, tok2=tok2)
        native = {r["id"]: r for r in cap["rows"]}
        swp = {r["id"]: r for r in alt["rows"]}
        W = [i for i in CA.eligible_ids(cap["rows"], cells) if i in swp]
        per_window = {str(i): [native[i]["wm"], swp[i]["stayed"]] for i in W}
        recomputed = sum(native[i]["wm"] - swp[i]["stayed"] for i in W) / len(W) if W else None
        drift = (None if recomputed is None or pub.get("comp_ca") is None
                 else abs(recomputed - pub["comp_ca"]))
        if drift is not None and drift > 1e-9:
            raise RuntimeError(
                f"{name}: per-window rows give comp_ca {recomputed} but the aggregate reports "
                f"{pub['comp_ca']} (drift {drift}) — the rows are not the rows the score came "
                f"from, so no paired analysis over them is valid")
        g = pub.get("guards") or {}
        n_scored = pub.get("n")
        out["arms"][name] = {
            "root": root,
            "comp_ca": pub.get("comp_ca"),
            "native_wm": pub.get("native_wm"),
            "swap_stayed": pub.get("swap_stayed"),
            "swap_follow": pub.get("swap_follow"),
            # the scored slice, not the harvested one: comp_ca is a mean over exactly these, so it
            # moves in steps of 1/n and a "drop of 0.02" is two windows
            "n_scored": n_scored,
            "quantum": (1.0 / n_scored) if n_scored else None,
            "n_role_swap_dropped": g.get("n_role_swap_dropped_from_W"),
            "analytic_band": pub.get("analytic_band"),
            "best_analytic_arm": pub.get("best_analytic_arm"),
            "shortcut_leaning": pub.get("shortcut_leaning"),
            "per_depth": pub.get("per_depth"),
            # id -> [native_wm, swap_stayed]; the per-window differential is native - stayed and
            # comp_ca is its mean over exactly these ids
            "per_window": per_window,
            "code_perturbation_vs_first_arm": code_pert,
            # that distance lives in coded space for a genome declaring code_cmds and in raw
            # embedding space for one that does not, so the two are not the same unit
            "code_perturbation_is_coded_space": CP._code_fn(stream) is not None,
        }
        _flush()
        print(f"  {name:9s} comp_ca {pub.get('comp_ca')}  (n={n_scored}, "
              f"1 window = {1.0/n_scored:.5f})" if n_scored else f"  {name}: n missing", flush=True)

    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
