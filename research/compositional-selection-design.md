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

## 11. Measured properties of the frozen slice and layout (2026-08-10)

These are facts about the instrument as it currently stands, recorded here because they are easy to re-derive incorrectly and because none of them belongs in a channel that reaches an inventor. Each was measured, not reasoned about.

**Earnable cells exist only at depths 3 and 4, so the depth floor selects nothing.** Of the 100 cells in the frozen ceiling table, 11 are earnable at depth 3 and 4 at depth 4; depths 1, 2, 5, 6 and 7 have none. The shallow depths are excluded by the ceiling filter before the depth floor is consulted, so `DEPTH_MIN = 3` currently removes zero windows and is a statement of intent rather than an active filter. The deep depths are empty for the reason recorded in section 9: earnability needs `R >= 2*depth + (m-2)`, and the frozen move-count grid stops at 8. A re-mint that extends the grid changes this, which is why the slice is described to inventors in terms of the mechanism rather than in terms of a depth number that is contingent on the mint.

**The scored layout has at most one command position without an observation, and it is the read.** `build_cups_layout` writes both a command and an observation for every step before the read and then the read command alone, at `2r`, with `Lmax = 2*max(r)+1`. So `valid_cmd & ~valid_obs` is true at exactly one pair per row — the read — for every window whose `r` is below the slice maximum, and at no pair at all for a window whose read sits at the maximum. In training the same condition is identically false everywhere, because `collate` clears `key_pad` for a command and its observation together. A mechanism gated on that condition is therefore untrained, and at scoring it fires once, after the value the read consumes has been taken. `eval/guard_reach.py` catches the subset of these that own parameters — measured, 2 of the 9 retracted carriers, and the discarded round-1 optim slot whose transport is trained by a synthetic auxiliary geometry and still cannot reach the scored read. It cannot catch a branch that is parameter-free, or one that shares its projections with a live path. Retirement and retraction removed this family; the gate is a partial net for the next one.

**A move's observation is present and constant.** All 23,874 move steps in the pack render to the identical string: exit 0, empty output. The observation is not missing; it carries no information about which file moved where, and that identity exists only in the command text. This is the real difficulty the objective poses, and it is distinct from the absent-observation condition above, which does not occur.

**The masked-endpoint detector is inert on the scored path and live elsewhere.** `_detect_masked_endpoint`, carried by five head impls, needs a live token after the key-padded observation slot. The cups layout never provides one, because the read is the last token, so the helper returns `None` on every scored window — verified by running all three variants against a real layout. It does fire on the cd-history probe's key-padded arm, which is reported and never scored. That asymmetry is why the mechanism family reads as plausible from its own source: it is not dead everywhere, only where the score is taken. The family is retired in `evolve/retired_impls.json` and the founders selecting it are retracted.

**A first-mover lookup outscores every measured mechanism on this slice.** The analytic band on the 89-window inner slice reads `h_first +0.1685`, `h_last -0.0899`, `h_lastmv +0.0337`, `deepest +0.0337`, and exactly `0.0` for `at_name`, `trace_h1` and `trace_h2`. The best real candidate is `+0.0300`. Name-keying cancels structurally, per window; chain-position arms cancel only in expectation, and this slice is one frozen realization. The band is reported beside every score and never subtracted, and each candidate's own differential is now also split by what the paying arms score on the same windows, in `private.shortcut_leaning`, because the per-window rows are not persisted and the split cannot be recovered afterwards.

## 12. The eval is not deterministic, and the noise floor already accounts for it (2026-08-12)

Re-measuring the nine selectable founders under identical conditions established two things that had never been tested by remeasurement.

**The same (genome, seed) does not reproduce.** Running `g0-00-baseline` twice on the same box, same code, same seeds: seed 1 gave `-0.011236` then `0.0`, seed 2 gave `+0.022472` then `-0.011236`. Every observed difference is an integer multiple of `1/89`, which identifies the mechanism — the metric picks the nearest exposure candidate by squared distance, so a floating-point difference of order `1e-7` in a GPU reduction flips a near-tied `argmin`, and a continuous perturbation becomes a discrete score change of one whole window. Roughly 1.5% of window picks moved between runs. This is a property of the instrument, not of any candidate or of any change made to the harness: the arms are structurally identical for a genome whose stream declares no `code_cmds`, which is all nine.

**The noise floor is nevertheless correct.** It was derived from the observed per-seed spread, which already contains this run-to-run component, so it does not need widening. Two independent three-seed measurements of the same nine genomes:

| quantity | value |
|---|---|
| pooled per-seed sd, both campaigns | 0.02525 |
| se of a three-seed mean | 0.01458 |
| predicted rms difference between two such means | 0.02062 |
| observed rms difference over the nine | 0.02122 |
| ratio | 1.03 |

One of nine moved by more than the floor, which is what a calibrated floor predicts. Treat `fitness.noise_floor = 0.0227` as measured rather than assumed, and treat any single ordering of the leaderboard as one realization: across the two campaigns the nominal best changed genome, and two genomes that differed by `0.026` in one campaign tied in the other.

The practical consequence for a round is that a candidate's number is a draw, not a reading. A difference smaller than the floor is not evidence, and re-measuring the same id is cheap insurance rather than duplicated work: it is the only thing that distinguishes a mechanism from a draw from the tail, and the archive keeps every measurement so the estimates combine. Two independent three-seed runs of one candidate give a six-seed mean at standard error 0.0103 rather than 0.0146.

## 2026-08-16 — the ordinal artifact in cupsF, measured

Every `mv` destination in the cups pack is named `.N`, where N is that move's position in the suffixed chain: **19,252 of 19,252** destinations across train and val. The cue is a property of the pack generator, not of any candidate, and it is total. Anyone reading a `comp_ca` number off this pack should know the cue is present and that it has been probed.

It was probed with `evolve/twin_probe.py`, which trains one net per (genome, seed) and measures those same weights against several val roots differing in one property, so arms are paired at the window level and the training term is identically zero. Nine arms over seven genomes, 19 records of 10 arms. The eligible slice, the metric form and every threshold are unchanged, so this note does not affect comparability with anything scored before it.

The arms that carry the information are NOSUFFIX (the tag becomes a letter), MUGS (the mount is renamed, the ordinal left exactly intact) and RANDPa (a per-sequence random bijection of the digits).

| | mean drop from ORIG, six live genomes | r7-26 (retracted) |
|---|---|---|
| NOSUFFIX | +0.012 | +0.9382 |
| RANDPa | +0.031 | +0.1180 |
| MUGS | +0.108 | +0.4607 |

The largest population-level effect belongs to the arm that preserves the ordinal. Across the six live genomes the ordinal-targeting arms straddle zero (+0.084, +0.082, +0.023, +0.008, +0.004, −0.127 for NOSUFFIX) and every one of them sits inside the per-seed spread of 0.034–0.081, so for those genomes the question is under-powered rather than answered.

For `r7-26` a control was pre-committed before the data existed: if MUGS — a broader, ordinal-preserving perturbation — cost at least as much as permuting the ordinal, the ordinal attribution was to be treated as dead. It cost 3.3x as much (`D_MUGS` +0.4607, cluster-exact p 6.8e-12, CI [0.352, 0.569]). **The RANDP drop is therefore not distinctively ordinal.** The retraction of `r7-26` is unaffected: it rests on the NOSUFFIX collapse, which is about the tag being a parseable digit form rather than about the digit's value.

Two consequences for practice. The pack is **not** re-minted or relabelled: that would invalidate comparability with every archived score to remove a cue no live candidate has been shown to use. Future packs should simply not print the move position, which is free at mint time. And the twin probe is worth running as a post-score audit rather than a campaign, since measuring extra roots costs no extra training.

The two remaining live genomes above 0.6, `r6-22-matchedrisk-ringband-blocks` and `r7-08-image-balanced-depth-rcbd`, were measured on the same ten roots at three seeds each, so every live genome scoring above 0.6 is now covered. Neither shows the signature: `r6-22` loses one window to NOSUFFIX (+0.0112) against +0.1985 to MUGS, and `r7-08` *gains* under both (−0.1086, −0.1011). Over all eight live genomes NOSUFFIX averages −0.003 and straddles zero, while MUGS averages +0.093 — the ordinal-preserving arm costs more than the ordinal-destroying one across the population.

What this does not settle: whether MUGS costs because of surface size or because `/tmp/w/mugs/` never appears in train (0 of 2560 lines) — that was pre-committed as unknowable from this arm set; and whether the live genomes are ordinal-free or merely under-powered, since every one of their ordinal-arm effects lies inside the per-seed spread of 0.034–0.081. Distinguishing those needs more seeds rather than more arms.
