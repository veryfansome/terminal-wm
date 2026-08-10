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

# ---------------------------------------------------------------------------------------------
# The eligible slice W. Genome-independent by construction: every term is a property of the
# window and the frozen ceiling table, never of the net. Identical for every candidate, which is
# what makes the differential comparable across the population.
#
# Of the ceiling table's cells only fifteen are earnable, and every one of them has depth 3 or 4,
# so the ceiling filter and the depth floor largely coincide at these knobs. Both are kept explicit:
# they encode different intentions, and a re-mint can pull them apart.
# N includes 3: the table has four earnable N=3 cells in the same ceiling band and with the same
# depth profile, worth about +44% more windows at no cost. N in {4,5} is the pre-designated primary
# slice for the CAPABILITY GATE, and this is the search signal, explicitly not the gate — widening
# here leaves that pre-registration untouched. N=2 stays out: the pick is two-way, and off-diagonal
# N=2 windows have no legal role-swap partner by construction.
SLICE_N = (3, 4, 5)
SLICE_STYLE = "core"          # the model trains on core-style boards; held-out style measures
                              # style TRANSFER and is reported separately, never scored
CEILING_EARNABLE_LT = 0.99    # cells at or above this are saturated for non-trackers: a margin
                              # there is unearnable and contributes only dilution
# DEPTH_MIN must sit ABOVE the bounded-trace cap, or a scored window is solvable by a trace and
# contributes a free +1. It is pinned to the ceiling table's own gauntlet_h at measurement time
# rather than hardcoded, so a re-mint at a different cap cannot desync the slice from the band.
# It cannot go higher either: earnability needs some other mover to match routed's depth, i.e.
# R >= 2*depth + (m-2), so the frozen r_grid caps earnable depth at 4 and a depth floor of 5
# empties the slice.
DEPTH_MIN = 3

DEPTH_BANDS = (("d2", 2, 2), ("d3", 3, 3), ("d4plus", 4, 99))

# --- anti-degeneracy floors -------------------------------------------------------------------
# INHERITED, NOT MEASURED FOR THIS QUANTITY. These two numbers were calibrated for a different
# instrument (a masked-endpoint imagination differential on the base world) and are carried here
# as a starting point only. cups_ca always REPORTS the realized values, so the first measurement
# on a real net tells us whether they are set anywhere near right. Re-set them from measured data
# before treating a failure here as a statement about a candidate.
MIN_NORM_OVER_BANK = 0.5
MIN_ANGULAR_DISPERSION = 0.5

# --- capability readout, REPORTED BUT NOT ENFORCED --------------------------------------------
# "the net beats chance on W by this much" is a CAPABILITY claim, not an instrument-validity
# claim, and this search deliberately does not gate on capability. On the recorded reference run
# the pack-trained net sits within a few points of chance on this slice, so enforcing a floor here
# would null every candidate and the search could never climb out of the regime it starts in. It
# is emitted on every measurement instead, so a population that does move off chance is visible
# immediately. Revisit only with measured data in hand.
REPORT_NATIVE_WM_OVER_CHANCE = 0.10


# --- the analytic non-tracker band -------------------------------------------------------------
# The differential's null is zero only in EXPECTATION over an exchangeable population, and the
# scored slice is one frozen realization of a mint whose chains are deliberately biased. Measured
# on the real inner slice, the name-keyed and last-src arms cancel to exactly zero, as designed —
# but a FIRST-MOVER lookup, a depth-zero strategy, scores clearly positive. Restricting the swap
# partner to movers shrinks that but does not remove it, because the routed content is not
# uniformly distributed over the movers.
#
# The band is REPORTED beside the score, never folded into it. No point estimate of it bounds what
# a shortcut can earn, in either direction:
#   - On one split the largest arm is an extreme value over a few noisy directions. Measured on the
#     training split, which carries five times as many windows and is equally valid because the arms
#     never touch the model, h_first reads about +0.01 against about +0.17 on the scored split,
#     while only `deepest` holds steady across splits. Subtracting a per-split maximum would remove
#     several times the only stable effect, through an arm whose value is near zero.
#   - Nor is the larger-sample value a substitute: a committed lookup keyed on the OBSERVABLE cell,
#     fitted without ever touching a scored split, realizes more on a held-out slice than the band
#     prices it at.
# So the band exists for INTERPRETATION, and it is free: every arm is genome-independent, making the
# whole band a per-split constant that cannot reorder any candidate.
#
# WHICH ARMS HAVE TO BE IN THE BAND, AND WHY THE OTHERS DO NOT.
# The role swap re-encodes ONLY the mv command strings and splices them at the mv positions. It
# leaves untouched: the exposure observations, the previous observation, the name index, and — given
# that the partner must itself be a mover — the SET of contents that move. So three of the
# instrument's arms cancel structurally, per window, and need no accounting:
#   copy_prev  reads the previous observation, unchanged            -> identical pick both arms
#   centroid   reads the exposure bank, unchanged                   -> identical pick both arms
#   elim       is 1/|movers|, and the swap trades one mover for another -> same cardinality
# (elim cancels on cardinality alone, so it would cancel with any partner. What the mover-partner
# rule actually buys is a two-sided exchange for the POSITIONAL arms; the rail below enforces it.)
#
# The arms that do NOT cancel are the ones that read WHERE in the chain something happened, because
# that is exactly what the swap permutes: the positional markers, and the depth-bounded backward
# tracers. A tracer that solves a window natively resolves to the SWAPPED content under the swap,
# scoring one minus zero — so on any window shallow enough for it, it scores maximally. It is
# silent here only because every scored window is deeper than its cap, which is a property of the
# current slice and not a property of the metric. Listing it explicitly means a future slice change
# cannot quietly stop accounting for it.
BOUNDED_TRACE_CAPS = (1, 2)   # hop caps whose backward trace is realizable at the frozen knobs
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
    return float(w["name"] == w["routed"])           # falls back to the name, which cancels


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
def measure_trained_net(net, ctx, target_mod, device, percep_name, cells,
                        seed=20260806, ceiling_table=None, swap_cache=None):
    """comp_ca for ONE trained net on ONE (root, split). Returns unrounded per-seed values.

    The scored scalar is the raw differential. `analytic_band` travels with it as a reference —
    what each analytic non-tracker scores on these same windows — but is never subtracted.

    `cells` is the flat {"N,depth,m,R": ceiling} dict (the ceiling table's ["cells"]).
    `ceiling_table` is passed through to cups_probe purely so its own reported aggregates keep
    their frozen-ceiling columns; comp_ca itself never reads a ceiling value arithmetically.
    """
    cap = CP.measure(net, ctx, target_mod, device, ceiling_table=ceiling_table)
    # swap_cache is the prebuilt role-swap synthesis: net-independent, so a campaign builds it
    # once rather than reloading the encoder per candidate.
    alt = CP.alt_chain(net, ctx, target_mod, device, percep_name, seed=seed,
                       ceiling_table=ceiling_table, cache=swap_cache)

    native = {r["id"]: r for r in cap["rows"]}
    swap = {r["id"]: r for r in alt["rows"]}

    _elig = eligible_ids(cap["rows"], cells)
    W = [i for i in _elig if i in swap]
    dropped = [i for i in _elig if i not in swap]
    # Windows whose only non-routed mover IS the queried name have no legal role-swap partner and
    # leave the slice. Measured on the real inner slice that is about a fifth of it — sizeable, and
    # a re-mint could make it most of it, so rail on the fraction rather than on the comment being
    # right. A slice gutted this way still returns a plausible-looking number.
    if _elig and len(dropped) / len(_elig) > 0.35:
        raise ValueError(
            f"cups_ca: {len(dropped)}/{len(_elig)} eligible windows have no legal role-swap "
            f"partner. Too much of the slice is gone for the remainder to mean anything.")
    if not W:
        raise ValueError(
            "cups_ca: the eligible slice W is EMPTY — no window is simultaneously earnable, "
            "core-style, deep enough and role-swappable. A comp_ca over an empty slice is not a "
            "small number, it is no measurement at all.")

    # The exchange must be two-sided on EVERY probed window. A one-sided exchange (partner never
    # moved) is farmable by any positional heuristic — see the note in cups_probe.alt_chain. The
    # partner draw now enforces this, so this is a rail against that enforcement regressing, not
    # a condition we hope holds.
    if alt["partner_was_mover_frac"] not in (None, 1.0):
        raise ValueError(
            f"cups_ca: role-swap partner was a non-mover in "
            f"{1 - alt['partner_was_mover_frac']:.1%} of probed windows. The exchange-symmetry "
            f"argument this metric rests on requires BOTH sides of the swap to be movers; a "
            f"one-sided exchange lets a first/last/deepest-mover heuristic score positive.")

    # The two arms must be talking about the same windows. Asserted rather than assumed: at the
    # frozen knobs the only role-swap dropouts are N==2 windows, which W excludes anyway, so this
    # should be vacuous — and if it ever stops being vacuous we need to know immediately.
    for i in W:
        n_r, s_r = native[i], swap[i]
        assert (n_r["N"], n_r["depth"], n_r["m"], n_r["R"], n_r["routed"]) == \
               (s_r["N"], s_r["depth"], s_r["m"], s_r["R"], s_r["routed"]), \
               f"cups_ca: native/swap row disagreement on window {i}"

    diffs = [native[i]["wm"] - swap[i]["stayed"] for i in W]
    comp_ca = sum(diffs) / len(diffs)

    # The alt-arm-only form: follow_alt - stayed_original, computed on the IDENTICAL slice. It
    # never references the absolute pick rate, so it is fully disjoint from the capability gate;
    # its non-tracker cancellation is only expectation-zero, which is why it is not what we
    # select on. Carried as a standing cross-check — the two forms should move together, and a
    # divergence between them is a signal that something is wrong with one of the arms.
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
    # Both arms, not just the native one: comp_ca is a DIFFERENCE, so a collapsed role-swap bank
    # corrupts it exactly as badly as a collapsed native bank.
    diag = _diagnostics(cap["pred_obs"], ctx["cands"], idxs)
    diag_swap = _diagnostics(alt["pred_obs"], ctx["cands"], idxs)
    for nm, pb in (("native", cap["pred_obs"]), ("swap", alt["pred_obs"])):
        if not torch.isfinite(pb[idxs]).all():
            raise ValueError(f"cups_ca: the {nm} prediction bank contains non-finite values over "
                             f"W — that is an instrument failure, not a calibration question")
    chance = sum(1.0 / native[i]["N"] for i in W) / len(W)
    native_wm = sum(native[i]["wm"] for i in W) / len(W)

    guards = {
        # HARD: structural integrity of the measurement. alt_chain raises internally on a
        # self-parity or encoder-stamp failure, so reaching here means those passed; the values
        # are surfaced so a run is auditable without re-reading logs.
        "self_parity_cos": alt["self_parity_cos"],
        "eye_tree_sha": alt["eye_tree_sha"],
        "slice_nonempty": True,
        "n": len(W),
        "n_role_swap_dropped_from_W": len(dropped),
        "norm_over_bank": diag["norm_over_bank"],
        "angular_dispersion": diag["angular_dispersion"],
        "norm_over_bank_swap": diag_swap["norm_over_bank"],
        "angular_dispersion_swap": diag_swap["angular_dispersion"],
        "norm_ok": all(d["norm_over_bank"] is not None
                       and d["norm_over_bank"] >= MIN_NORM_OVER_BANK
                       for d in (diag, diag_swap)),
        "dispersion_ok": all(d["angular_dispersion"] is not None
                             and d["angular_dispersion"] >= MIN_ANGULAR_DISPERSION
                             for d in (diag, diag_swap)),
        # REPORTED ONLY — see REPORT_NATIVE_WM_OVER_CHANCE above.
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
        # THE scalar the search maximizes: the raw paired differential. The analytic band travels
        # beside it as a reference, never subtracted from it — see the note above.
        "comp_ca": comp_ca,
        "analytic_band": band,
        "best_analytic_arm": best_arm,
        "n": len(W),
        "per_depth": per_depth,
        "comp_ca_alt_only": sum(alt_only) / len(alt_only),   # cross-check, never selection
        "native_wm": native_wm,
        "swap_stayed": sum(swap[i]["stayed"] for i in W) / len(W),
        "swap_follow": sum(swap[i]["follow"] for i in W) / len(W),
        "guards": guards,
        "slice": {"N_in": list(SLICE_N), "style": SLICE_STYLE, "depth_min": DEPTH_MIN,
                  "ceiling_earnable_lt": CEILING_EARNABLE_LT},
        "role_swap_seed": seed,
        # the capability gate's own reading on the same net, carried for reporting. NEVER an
        # input to selection: it is the honest yardstick this search must not learn to climb.
        "gate_report": {
            "earnable_core": cap.get("earnable_core"),
            "deep_style_core": cap.get("deep_style_core"),
        },
    }
