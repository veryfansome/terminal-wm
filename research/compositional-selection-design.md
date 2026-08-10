# Compositional depth as the search objective

What `evolve/cups_ca.py` computes, and why it is shaped this way.

## 0. Why this objective

A single-step next-observation margin is depth-one, and so is a single-hop imagination differential. Neither rewards composing a multi-hop tracking chain, so a search built on either never pressures a genome toward one. Measurement on this capability pack bears that out: genomes selected on depth-one signals do not have the capability. To evolve toward compositional depth, the search has to score compositional depth.

## 1. The measurement

The capability pack presents, inside an ordinary shell trajectory: N contents exposed at N locations, a silent chain of `mv` commands that permutes them, and one test read. Scoring is an N-way forced choice among the window's **own** exposure observations, so chance is one over N by construction, identical for the model and for every analytic reference arm.

```
comp_ca = mean over i in W of ( native_hit_i - swap_stayed_i )

native_hit_i  = 1[ nearest-exposure pick under the NATIVE chain    == routed_i ]
swap_stayed_i = 1[ nearest-exposure pick under the ROLE-SWAP chain == routed_i ]
```

The role-swap exchanges the move-position sets of `routed` and a partner over the same board. The destination sequence, legality, and the stamped depth are all preserved; only the `mv` command strings are re-rendered, re-encoded, and spliced back at the move positions. Two guards keep that splice honest: the encoder's directory-tree hash must match the one the root was built with, and a sample of the window's **original** move commands is re-encoded and must reproduce the cached embeddings to within a tight cosine tolerance. A wrong encoder, render, or standardization frame fails loudly rather than recording noise.

**The partner must itself be a mover.** This is a requirement, not a detail. The routed content always moves, so a non-mover partner makes the exchange one-sided: routed's move positions transfer out and nothing comes back. A positional heuristic then flips from routed to the partner under the swap and scores `+1` on that window, and the windows that would cancel it — where the partner is the native marker and the swap hands that marker to routed, scoring `−1` — cannot occur, because a non-mover is never the native marker. The bias is systematic, positive, and farmable by exactly the family the metric exists to exclude. `cups_probe.build_swap_cache` draws only mover partners, and `cups_ca` refuses to return a number if any probed window had a one-sided exchange.

## 2. Why it is hard to fake

- **Name-keyed non-tracker** — predicts the exposure at the queried name. The name index is untouched by the swap, so `native_hit == swap_stayed` **exactly, per window**. Contribution: exactly zero. This cancellation is structural rather than statistical, and it is why the role-swap is preferred to a masked-endpoint content swap.
- **Previous-observation copier, exposure-centroid picker, eliminate-to-movers** — all read inputs the swap does not touch (the previous observation, the exposure bank, the mover count), so all cancel per window.
- **Depth-bounded backward tracer** — resolves any window within its hop cap. Silent here because every scored window is deeper than the cap; see §5.
- **Chain-position non-tracker** (first mover / last mover / deepest) — reads *where* in the chain something happened, which is precisely what the swap permutes. Its expectation is zero over an exchangeable population, but the scored slice is a single realization, so its realized value is not zero. §3 gives the measured size and what is done about it.
- **History-ignorer or memorizer** — native and swap agree, so approximately zero.
- **Genuine tracker** — native picks the routed content, swap follows the partner. Positive.

## 3. The analytic band is reported, never subtracted

Every analytic arm is a pure function of the command strings, so each has its own `comp_ca` on the frozen slice, computable with no model at all. The whole band is therefore **the same constant for every candidate** and cannot reorder anything. It is emitted beside every score as the reference a reader needs, and the scored scalar stays the raw differential.

It is not subtracted, and the reason is that no point estimate of it means what a subtraction would claim. Measured on all three splits under the current slice rule:

| arm | inner (n=89) | final (n=96) | train (n=466) |
|---|---|---|---|
| `at_name` | +0.0000 | +0.0000 | +0.0000 |
| `trace_h1`, `trace_h2` | +0.0000 | +0.0000 | +0.0000 |
| `deepest` | +0.0337 | +0.0417 | **+0.0365** |
| `h_lastmv` | +0.0337 | +0.1354 | +0.0558 |
| `h_last` | −0.0899 | −0.0938 | −0.0215 |
| `h_first` | **+0.1685** | +0.0312 | **+0.0064** |

Read the last two rows against the train column. `h_first` is the largest arm on the inner slice at +0.1685 and is +0.0064 on a slice five times larger — it is a sampling excursion, not an advantage a strategy actually has. `h_lastmv` behaves the same way. Only `deepest` holds steady across all three (+0.034 / +0.042 / +0.037), and it is small. So the maximum over arms on any one split is largely an extreme-value statistic over a handful of noisy directions, and subtracting it would remove several times the only real effect, through an arm whose value is near zero.

The population value is not a usable substitute either: a committed lookup keyed on the *observable* cell — the exposure count, the mover count, and the chain length, none of which require tracking — can be fitted without ever touching a scored split and still realize more on a held-out slice than the band prices it at. A fixed shortcut's realized value on a slice this size swings by more than the band itself.

Biased upward one way, under-covering the other, and no bound in either direction. Hence: report it, read scores against it, and do not fold it into the number.

`at_name` and both trace arms sitting at exactly zero on all three splits is the load-bearing observation here — it is the structural cancellation of §2 confirmed on real data, and it is what distinguishes the arms that need reporting from the arms that need nothing.

## 4. Why it is not the capability gate

The pack's honest capability measurement is `pick_rate − ceiling_frozen`: an absolute rate minus a frozen analytic per-cell ceiling. That is the yardstick — measured, reported, never optimized against.

`comp_ca` is a paired within-genome differential in which the ceiling never appears arithmetically. The table enters only to define the eligible window set, identically for every genome, exactly as a fixed mask defines a slice. They are different functionals of the same measurement, and the separation is load-bearing: raising the pick rate uniformly lifts both arms and leaves `comp_ca` unmoved, so a search climbing `comp_ca` cannot thereby climb the gate. The gate reading rides along in `private` on every measurement. It is never an input to selection.

## 5. The eligible slice W

```
W = { windows : N in {3,4,5}, style == core, ceiling[N,depth,m,R] < 0.99, depth >= 3,
                a legal mover partner exists }
```

Realized: 89 windows on the inner split, 96 on the final split, 466 on train, with depths `{3, 4}` only. Every term is a property of the window or the frozen table, never of the model.

- **`ceiling < 0.99`** keeps only cells a non-tracking strategy has not already saturated. This does most of the work and is the filter to leave alone. Every cell at depth two or below has ceiling exactly 1.0, because a two-hop backward trace resolves them — and on a differential such a window is not merely uninformative, it is actively harmful: the trace answers `routed` natively and the swapped content under the swap, scoring `+1`. Admitting depth-two windows roughly doubles the slice and takes the band from about 0.13 to about 0.45. Slice size is not worth that.
- **`depth >= 3`** sits one hop above the trace cap the ceiling table was built at, so "no scored window is solvable by a listed trace arm" holds by construction rather than by coincidence.
- **`N in {3,4,5}`** — N=3 contributes four earnable cells in the same ceiling band with the same depth profile, worth about 45% more windows at no cost. N in {4,5} is the pre-designated primary slice for the *capability gate*; this is the search signal and explicitly not the gate, so widening here leaves that pre-registration untouched. N=2 stays out: the pick would be two-way, and off-diagonal N=2 windows have no legal partner by construction.
- **`style == core`** is the board style the model trains on. Held-out style measures style *transfer*, a different question, reported separately and never scored.

About a fifth of otherwise-eligible windows leave the slice because their only non-routed mover is the queried name, so no legal partner exists. `cups_ca` refuses to return a number if that fraction exceeds a third — a slice gutted that way still produces a plausible-looking number.

**The correctness trap this design exists to avoid.** Do not compute `comp_ca` by subtracting the instrument's aggregate fields. They are rounded, *and* they aggregate over populations that do not coincide: the native arm's earnable selector applies no depth filter, while the swap arm's requires a depth floor and drops the no-partner windows. `cups_ca` intersects the two arms' **per-window rows by window id**, asserts the rows agree on every window property, and means the differences unrounded.

## 6. What the first rounds can and cannot conclude

1. **The slice is small and shared.** Every candidate is scored on the same windows. A handful of windows flipping moves the number materially, and that channel shrinks only with more windows, never with more seeds.
2. **The seeds are fixed.** Given a genome and a seed the eval is essentially deterministic, so repeated runs measure reproducibility rather than sampling. A noise floor obtained by re-running the same seeds will look far smaller than the real uncertainty and will license treating noise as signal. Derive it instead from the spread of three-seed means across *disjoint* seed triples. It is left unset in the contract until that is measured.
3. **The population starts near chance.** A guard requiring the net to clear chance by a fixed margin would fail every early candidate and leave the search unable to climb out of the regime it starts in. That quantity is therefore reported on every measurement and not enforced — it is a capability claim, and this search does not gate on capability.

Recorded pack runs made before the mover-partner requirement are not comparable and must not be used as a baseline; the split between tracking and one-sided-exchange artifact cannot be recovered from their aggregates.

## 7. Guards, and which ones bite

Enforced — a candidate failing these has no usable number:

| guard | what it catches |
|---|---|
| encoder tree hash matches the root's stamp | a wrong eye silently redefining every embedding |
| self-parity of re-encoded original commands | a wrong render or standardization frame |
| every probed window had a two-sided exchange | the farmable one-sided swap of §1 |
| eligible slice non-empty, partner-drop under a third | a measurement over almost nothing |
| every realized cell present in the ceiling table | a table/root mismatch quietly shrinking W |
| prediction-bank norm and angular dispersion, both arms | a collapsed or constant predictor |
| no future leakage; head declares itself leak-safe | seeing the observation being predicted |
| training did not diverge | non-finite loss |

Reported, not enforced: the pick rate over chance, the analytic band, and the capability gate reading.

The norm and dispersion floors are **inherited from a different instrument and have not been calibrated for this quantity.** The realized values are emitted on every measurement; re-set the thresholds from measured data before reading a failure there as a statement about a candidate.

## 7a. The three ceilings the instrument reports, and why there are three

`cups_probe.measure` emits three ceiling estimates per slice. They are not redundant, and none of
them is a candidate-facing knob.

**`arm_max`** — the best single arm's mean over the slice. Too weak to be the gate: an adversary is
not required to commit to one arm for the whole slice.

**`switch_max`** — the switching ceiling. The slice is partitioned by the *observable* cell (the
exposure count and the chain depth, both readable without tracking), the best arm's mean is taken
per cell, and the cells are mass-weighted. It dominates `arm_max` by construction.

Two corrections inside it look like over-engineering and are not:

- **Cells with fewer than `SWITCH_MIN` windows pool into their per-N marginal before switching,**
  and whatever is still too small pools into one remainder group. Without this, a singleton cell's
  best arm-mean is just that row's oracle maximum, and the "ceiling" degenerates into a per-row
  oracle — a ceiling no strategy could actually realize, which would make the margin meaningless in
  the conservative direction.
- **The arm is chosen out-of-sample.** The in-sample per-group best-mean is a maximum over ten
  correlated means and is biased upward as an estimate of the switching ceiling — around +0.035 at
  n=142, which is a large fraction of the band the gate is read against. `switch_max_xfit` splits
  each group in half by a seeded shuffle, picks the arm on one half, evaluates it on the other,
  does both directions, and weights by evaluation mass. That is what the reported margin uses;
  in-sample `switch_max` stays beside it as the conservative bound.

**`ceiling_frozen`** — the analytic per-`(N, depth, m, R)`-cell population ceiling, computed offline
from the planner at the frozen knobs and applied to the realized cell masses. When a table is
supplied this is the gate-bearing one, because the resulting margin carries the model's own
sampling noise and nothing else: no in-sample maximum bias, no cross-fit pooling gap. A realized
cell absent from the table fails loud.

The exposure-swap probe carries a related correction. Its bank is a **full-donor** bank — the
window's own routed and name exposures plus *every* donor exposure — rather than the 2×2
`{own,donor} × {routed,name}` bank it started as. The small bank forced an arm-mimic's wrong-slot
predictions onto some bank entry, spilling roughly 0.1–0.25 of its mass onto the donor-routed
column and falsifying the probe's null. With every donor exposure present, a wrong-slot prediction
lands on its own donor entry, is counted as donor-other, and the follow rate equals the strategy's
slot accuracy exactly.

## 8. The lane

One net per (genome, seed), trained on the pack root and standardized on **that root's own** train statistics. The compositional metric and the world-model health readout both come off that same net, because training twice to measure two things off it is waste.

A base-world fitness on a second root is deliberately not carried as a floor. Archived full three-seed records of one strong genome span 0.4251 / 0.4247 / 0.4262 while the entire visible band across the top thirteen genomes is 0.0019 wide — same-genome re-run noise is most of the spread, so a second lane would double the compute for a quantity that barely discriminates. The collapse check is the pack net's own next-observation retrieval, computed on the same net for free.

Frames are never pooled. The pack root is its own comparability frame; a number from another root, eye, or environment is not comparable without a measured offset.

## 9. What this mint can and cannot support

Earnability requires some other content to match or exceed the routed content's depth, which works out to `R >= 2·depth + (m−2)`. With the frozen move-count grid topping out at 8, earnable depth is capped at **4**: every earnable cell is depth 3 or 4, a depth floor of 5 empties the slice, and windows deeper than 4 all have a saturated ceiling.

The consequence, stated bluntly: a backward trace capped at four hops scores **exactly +1.0** on every non-empty slice these knobs can produce, and a three-hop trace scores about +0.7. What keeps the scored slice out of that family's reach is the two-hop cap the ceiling table was built at — a choice, not a fact about the data. No code change reaches this; it is a property of the mint.

So this is a sound objective to *search* on — the ranking signal is a paired differential on a frozen genome-independent slice, and climbing it still requires composing the chain — but it is not a basis for an external claim. The escape is a re-mint with the move-count grid extended far enough that deeper windows become earnable, putting the slice above the trace family rather than beside it.

## 10. Changing any of this

The eligible slice, the metric form, and every guard threshold are single named constants in `evolve/cups_ca.py`. Changing one changes what the search means and makes everything scored before it incomparable. That is a dated note in this file and a fresh baseline, not a tweak.
