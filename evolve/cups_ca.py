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
  A chain-position non-tracker (first/last/deepest/eliminate) attends to mv tokens, but under the
  routed<->partner position exchange it mimics a tracker on routed-marker windows and anti-mimics
  on the symmetric partner windows; drawn exchangeably its expectation is zero.
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
# At the frozen mint knobs this is already a depth>=3 slice: of the ceiling table's 100 cells only
# 15 are earnable, of which exactly 8 have N in {4,5}, and every one of those 8 has depth 3 or 4.
# DEPTH_MIN is therefore inert at these knobs; it is kept explicit so a re-mint cannot silently
# widen the slice to shallow windows an analytic heuristic can already solve.
SLICE_N = (4, 5)
SLICE_STYLE = "core"          # the model trains on core-style boards; held-out style measures
                              # style TRANSFER and is reported separately, never scored
CEILING_EARNABLE_LT = 0.99    # cells at or above this are saturated for non-trackers: a margin
                              # there is unearnable and contributes only dilution
DEPTH_MIN = 2

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


def _cell_key(row):
    return f"{row['N']},{row['depth']},{row['m']},{row['R']}"


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
                        seed=20260806, ceiling_table=None):
    """comp_ca for ONE trained net on ONE (root, split). Returns unrounded per-seed values.

    `cells` is the flat {"N,depth,m,R": ceiling} dict (the ceiling table's ["cells"]).
    `ceiling_table` is passed through to cups_probe purely so its own reported aggregates keep
    their frozen-ceiling columns; comp_ca itself never reads a ceiling value arithmetically.
    """
    cap = CP.measure(net, ctx, target_mod, device, ceiling_table=ceiling_table)
    alt = CP.alt_chain(net, ctx, target_mod, device, percep_name, seed=seed,
                       ceiling_table=ceiling_table)

    native = {r["id"]: r for r in cap["rows"]}
    swap = {r["id"]: r for r in alt["rows"]}

    W = [i for i in eligible_ids(cap["rows"], cells) if i in swap]
    dropped = [i for i in eligible_ids(cap["rows"], cells) if i not in swap]
    if not W:
        raise ValueError(
            "cups_ca: the eligible slice W is EMPTY — no window is simultaneously earnable, "
            "core-style, deep enough and role-swappable. A comp_ca over an empty slice is not a "
            "small number, it is no measurement at all.")

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
    diag = _diagnostics(cap["pred_obs"], ctx["cands"], idxs)
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
        "norm_ok": diag["norm_over_bank"] is not None
                   and diag["norm_over_bank"] >= MIN_NORM_OVER_BANK,
        "dispersion_ok": diag["angular_dispersion"] is not None
                         and diag["angular_dispersion"] >= MIN_ANGULAR_DISPERSION,
        # REPORTED ONLY — see REPORT_NATIVE_WM_OVER_CHANCE above.
        "native_wm": native_wm,
        "chance": chance,
        "native_wm_over_chance": native_wm - chance,
        "native_wm_clears_report_threshold":
            (native_wm - chance) >= REPORT_NATIVE_WM_OVER_CHANCE,
    }

    return {
        "comp_ca": comp_ca,                 # THE scalar the search maximizes
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
