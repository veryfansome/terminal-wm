"""cups_ca — comp_ca, the compositional-depth metric this repo's search maximizes.

comp_ca is a PAIRED, WITHIN-GENOME differential over a genome-independent window slice W:

    comp_ca = mean over i in W of ( native_hit_i  -  swap_stayed_i )

    native_hit_i  = 1[ the N-way nearest-exposure pick under the NATIVE mv chain    == routed_i ]
    swap_stayed_i = 1[ the N-way nearest-exposure pick under the ROLE-SWAP mv chain == routed_i ]

The role-swap (cups_probe.alt_chain) exchanges the move-position sets of `routed` and a seeded
partner over the SAME board, keeping the destination sequence, legality and the stamped depth
identical, and re-encoding only the mv command strings. So the two arms differ in exactly one
thing: which content the chain routes to the test location.

WHY THIS IS HARD TO FAKE
  A name-keyed non-tracker predicts the exposure at the name index. That is identical under both
  arms (only mv embeddings change), so native_hit_i == swap_stayed_i EXACTLY, per window, and the
  window contributes exactly zero. The cancellation is structural, not statistical.
  A chain-position non-tracker (first/last/deepest/eliminate) attends to mv tokens. Under the
  routed<->partner exchange it mimics a tracker on routed-marker windows and anti-mimics on the
  symmetric partner windows, so its EXPECTATION is zero — but only over an exchangeable
  population, and the scored slice is one frozen realization of a deliberately chain-biased mint.
  Measured there, a first-mover lookup scores positive. So the band of analytic arms is measured
  on every run and REPORTED beside the score — it is not subtracted, because at this slice size no
  point estimate of it is a bound in either direction. See the band note below.
  A history-ignorer or memorizer has native ~ swap, so ~zero.
  A genuine multi-hop tracker follows the chain: native picks routed, swap follows the partner, so
  the difference is positive.

WHY IT IS NOT THE CAPABILITY GATE
  The pack's honest capability gate is an ABSOLUTE quantity, `wm - ceiling_frozen`, measured
  against a frozen analytic per-cell ceiling table. comp_ca is a differential in which that
  ceiling never appears arithmetically: the table enters ONLY to define which windows are
  eligible, identically for every genome. Raising the absolute pick rate uniformly lifts both
  arms and leaves comp_ca unmoved. The gate is a measurement; this is the search signal. Keeping
  them separate is what stops the search from optimizing its own honest yardstick.

THE ONE CORRECTNESS TRAP (this module exists to avoid it)
  Do NOT compute comp_ca by subtracting cups_probe's rounded aggregate fields. `measure()` and
  `alt_chain()` round to 4 places AND aggregate over populations that do not coincide:
  measure's `earnable_core` selector applies no depth filter, while alt_chain's `earn_core`
  requires depth >= 2 and silently drops windows that had no legal role-swap partner. This module
  intersects the two arms' PER-WINDOW rows by window id, asserts the id sets match, and means the
  per-window differences unrounded.
"""
import torch
import torch.nn.functional as F

from evolve import cups_probe as CP

SLICE_N = (3, 4, 5)
SLICE_STYLE = "core"
CEILING_EARNABLE_LT = 0.99
# DEPTH_MIN must sit ABOVE the bounded-trace cap, or a scored window is solvable by a trace and
DEPTH_MIN = 3

DEPTH_BANDS = (("d2", 2, 2), ("d3", 3, 3), ("d4plus", 4, 99))

MIN_NORM_OVER_BANK = None
MIN_ANGULAR_DISPERSION = None

REPORT_NATIVE_WM_OVER_CHANCE = 0.10


BOUNDED_TRACE_CAPS = (1, 2)
ANALYTIC_ARMS = (("h_first", "h_last", "h_lastmv", "at_name", "deepest")
                 + tuple(f"trace_h{c}" for c in BOUNDED_TRACE_CAPS))


def _cell_key(row):
    return f"{row['N']},{row['depth']},{row['m']},{row['R']}"


def _native_marks(w):
    """The native chain's positional markers, in the same shape alt_chain reports for the swap."""
    return {"h_first": w["first_src"],
            "h_last": w["last_src"] if w["last_src"] is not None else w["name"],
            "h_lastmv": w["last_mover"],
            "at_name": w["name"],
            "deepest": w["deepest"]}


def _arm_hit(marks, arm, routed):
    v = marks[arm]
    if arm == "deepest":
        return (1.0 / len(v)) if v and routed in v else 0.0
    return float(v == routed)


def _trace_arm(w, cap, native):
    """A backward trace through the recorded mv pairs, capped at `cap` hops.

    It resolves a window whose depth is within the cap, and otherwise falls back to the name. On
    the NATIVE chain it therefore answers `routed`; on the SWAPPED chain the recorded commands are
    different and the same trace answers the swapped content, which is never `routed`. So a window
    it can solve contributes one to the native arm and zero to the swap arm.
    """
    if w["depth"] <= cap:
        return 1.0 if native else 0.0
    return float(w["name"] == w["routed"])


def analytic_band(win_by_id, swap, W):
    """comp_ca as each analytic non-tracker would score it, on the identical frozen slice."""
    band = {}
    for arm in ANALYTIC_ARMS:
        tot = 0.0
        for i in W:
            w, s = win_by_id[i], swap[i]
            routed = w["routed"]
            if arm.startswith("trace_h"):
                cap = int(arm[len("trace_h"):])
                tot += _trace_arm(w, cap, True) - _trace_arm(w, cap, False)
            else:
                tot += (_arm_hit(_native_marks(w), arm, routed)
                        - _arm_hit(s["alt_marks"], arm, routed))
        band[arm] = tot / len(W)
    return band


def eligible_ids(native_rows, cells):
    """W as a set of window ids, plus the per-row ceiling lookup.

    Fails loud on ANY realized cell missing from the table. cups_probe's own selectors use
    `cells.get(key, 1.0) < 0.99`, which silently DROPS an unlisted window from the earnable slice
    instead of complaining — so a table/root mismatch quietly shrinks W rather than erroring. That
    asymmetry is exactly what an unnoticed re-mint would look like, so here it raises.
    """
    missing = sorted({_cell_key(r) for r in native_rows if _cell_key(r) not in cells})
    if missing:
        raise ValueError(
            f"cups_ca: {len(missing)} realized (N,depth,m,R) cell(s) absent from the ceiling "
            f"table, e.g. {missing[:5]} — the table and the data root disagree. A mismatched "
            f"table silently redefines the eligible slice, so this is fatal, not a warning.")
    return [r["id"] for r in native_rows
            if r["N"] in SLICE_N
            and r["style"] == SLICE_STYLE
            and r["depth"] >= DEPTH_MIN
            and cells[_cell_key(r)] < CEILING_EARNABLE_LT]


def _diagnostics(pred_obs, cands, idxs):
    """Anti-degeneracy readouts over W: is the prediction bank collapsed or constant?"""
    if not idxs:
        return {"norm_over_bank": None, "angular_dispersion": None}
    pn = pred_obs[idxs]
    bank = torch.cat([cands[i] for i in idxs], dim=0)
    bank_norm = float(bank.norm(dim=-1).mean())
    norm_over_bank = float(pn.norm(dim=-1).mean()) / bank_norm if bank_norm > 0 else None
    mu = pn.mean(0, keepdim=True)
    if float(mu.norm()) == 0.0:
        dispersion = 0.0
    else:
        dispersion = 1.0 - float(F.cosine_similarity(pn, mu.expand_as(pn), dim=-1).mean())
    return {"norm_over_bank": norm_over_bank, "angular_dispersion": dispersion}


@torch.no_grad()
def assert_slice_matches_table(knobs):
    """The slice and the band must agree with the cap the ceiling table was built at.

    The eligible slice is defined by that table, and the table's ceilings encode "not solved by a
    backward trace capped at gauntlet_h hops". Two things therefore have to hold, and neither is
    checkable from the cells alone: the band must account for traces up to exactly that cap, and the
    depth floor must sit above it. A re-mint at a deeper cap would otherwise admit windows a listed
    trace arm solves, which is the one regime where such an arm scores maximally and silently.
    """
    if not knobs or "gauntlet_h" not in knobs:
        return
    cap = int(knobs["gauntlet_h"])
    if max(BOUNDED_TRACE_CAPS) != cap:
        raise ValueError(
            f"the ceiling table was built with a backward trace capped at {cap} hops, but the "
            f"analytic band accounts for caps {BOUNDED_TRACE_CAPS}. They must match, or the band "
            f"stops pricing a strategy the slice admits.")
    if DEPTH_MIN <= cap:
        raise ValueError(
            f"DEPTH_MIN={DEPTH_MIN} is not above the table's trace cap of {cap}: the slice would "
            f"admit windows that a {cap}-hop trace resolves, and such a window scores +1 for that "
            f"trace rather than contributing nothing.")


def measure_trained_net(net, ctx, target_mod, device, percep_name, cells,
                        seed=20260806, ceiling_table=None, swap_cache=None, knobs=None,
                        stream=None):
    """comp_ca for ONE trained net on ONE (root, split). Returns unrounded per-seed values.

    The scored scalar is the raw differential. `analytic_band` travels with it as a reference —
    what each analytic non-tracker scores on these same windows — but is never subtracted.

    `cells` is the flat {"N,depth,m,R": ceiling} dict (the ceiling table's ["cells"]).
    `ceiling_table` is passed through to cups_probe purely so its own reported aggregates keep
    their frozen-ceiling columns; comp_ca itself never reads a ceiling value arithmetically.
    """
    assert_slice_matches_table(knobs)
    tok, tok2 = CP.stream_coded_toks(ctx, swap_cache, stream)
    cap = CP.measure(net, ctx, target_mod, device, ceiling_table=ceiling_table, tok=tok)
    alt = CP.alt_chain(net, ctx, target_mod, device, percep_name, seed=seed,
                       ceiling_table=ceiling_table, cache=swap_cache, tok2=tok2)

    native = {r["id"]: r for r in cap["rows"]}
    swap = {r["id"]: r for r in alt["rows"]}

    _elig = eligible_ids(cap["rows"], cells)
    W = [i for i in _elig if i in swap]
    dropped = [i for i in _elig if i not in swap]
    if _elig and len(dropped) / len(_elig) > 0.35:
        raise ValueError(
            f"cups_ca: {len(dropped)}/{len(_elig)} eligible windows have no legal role-swap "
            f"partner. Too much of the slice is gone for the remainder to mean anything.")
    if not W:
        raise ValueError(
            "cups_ca: the eligible slice W is EMPTY — no window is simultaneously earnable, "
            "core-style, deep enough and role-swappable. A comp_ca over an empty slice is not a "
            "small number, it is no measurement at all.")

    if alt["partner_was_mover_frac"] not in (None, 1.0):
        raise ValueError(
            f"cups_ca: role-swap partner was a non-mover in "
            f"{1 - alt['partner_was_mover_frac']:.1%} of probed windows. The exchange-symmetry "
            f"argument this metric rests on requires BOTH sides of the swap to be movers; a "
            f"one-sided exchange lets a first/last/deepest-mover heuristic score positive.")

    for i in W:
        n_r, s_r = native[i], swap[i]
        assert (n_r["N"], n_r["depth"], n_r["m"], n_r["R"], n_r["routed"]) == \
               (s_r["N"], s_r["depth"], s_r["m"], s_r["R"], s_r["routed"]), \
               f"cups_ca: native/swap row disagreement on window {i}"

    diffs = [native[i]["wm"] - swap[i]["stayed"] for i in W]
    comp_ca = sum(diffs) / len(diffs)

    alt_only = [swap[i]["follow"] - swap[i]["stayed"] for i in W]

    per_depth = {}
    for band, lo, hi in DEPTH_BANDS:
        ids = [i for i in W if lo <= native[i]["depth"] <= hi]
        per_depth[band] = {
            "n": len(ids),
            "comp_ca": (sum(native[i]["wm"] - swap[i]["stayed"] for i in ids) / len(ids))
                       if ids else None,
        }

    idxs = [swap[i]["i"] for i in W]
    diag = _diagnostics(cap["pred_obs"], ctx["cands"], idxs)
    diag_swap = _diagnostics(alt["pred_obs"], ctx["cands"], idxs)
    for nm, pb in (("native", cap["pred_obs"]), ("swap", alt["pred_obs"])):
        if not torch.isfinite(pb[idxs]).all():
            raise ValueError(f"cups_ca: the {nm} prediction bank contains non-finite values over "
                             f"W — that is an instrument failure, not a calibration question")
    chance = sum(1.0 / native[i]["N"] for i in W) / len(W)
    native_wm = sum(native[i]["wm"] for i in W) / len(W)

    guards = {
        "self_parity_cos": alt["self_parity_cos"],
        "eye_tree_sha": alt["eye_tree_sha"],
        "slice_nonempty": True,
        "n": len(W),
        "n_role_swap_dropped_from_W": len(dropped),
        "norm_over_bank": diag["norm_over_bank"],
        "angular_dispersion": diag["angular_dispersion"],
        "norm_over_bank_swap": diag_swap["norm_over_bank"],
        "angular_dispersion_swap": diag_swap["angular_dispersion"],
        "norm_ok": (None if MIN_NORM_OVER_BANK is None else
                    all(d["norm_over_bank"] is not None
                        and d["norm_over_bank"] >= MIN_NORM_OVER_BANK
                        for d in (diag, diag_swap))),
        "dispersion_ok": (None if MIN_ANGULAR_DISPERSION is None else
                          all(d["angular_dispersion"] is not None
                              and d["angular_dispersion"] >= MIN_ANGULAR_DISPERSION
                              for d in (diag, diag_swap))),
        "native_wm": native_wm,
        "chance": chance,
        "native_wm_over_chance": native_wm - chance,
        "native_wm_clears_report_threshold":
            (native_wm - chance) >= REPORT_NATIVE_WM_OVER_CHANCE,
    }

    win_by_id = {w["id"]: w for w in ctx["wins"]}
    band = analytic_band(win_by_id, swap, W)
    best_arm = max(band, key=lambda a: band[a])

    return {
        "comp_ca": comp_ca,
        "analytic_band": band,
        "best_analytic_arm": best_arm,
        "n": len(W),
        "per_depth": per_depth,
        "comp_ca_alt_only": sum(alt_only) / len(alt_only),
        "native_wm": native_wm,
        "swap_stayed": sum(swap[i]["stayed"] for i in W) / len(W),
        "swap_follow": sum(swap[i]["follow"] for i in W) / len(W),
        "guards": guards,
        "slice": {"N_in": list(SLICE_N), "style": SLICE_STYLE, "depth_min": DEPTH_MIN,
                  "ceiling_earnable_lt": CEILING_EARNABLE_LT},
        "role_swap_seed": seed,
        "gate_report": {
            "earnable_core": cap.get("earnable_core"),
            "deep_style_core": cap.get("deep_style_core"),
        },
    }
