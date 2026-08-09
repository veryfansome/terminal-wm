# Compositional depth as the search objective

**Status: implemented. This is what `evolve/cups_ca.py` computes and what the search maximizes.**
Re-derived from the predecessor project's design note of the same name, with the champion/promotion
apparatus removed and several numbers corrected against the recorded reference run.

## 0. Why this objective

The predecessor's fitness was a single-step next-observation margin, and its second axis was a
single-hop imagination differential. Both are **depth-one**. Neither rewards composing a multi-hop
tracking chain, so nothing in the search ever pressured a genome toward it — and the strongest
genome it produced does not have the capability. To evolve toward compositional depth, the search
has to score compositional depth.

## 1. The measurement

The capability pack presents, inside an ordinary shell trajectory: N contents exposed at N
locations, a silent chain of `mv` commands that permutes them, and one test read. Scoring is an
N-way forced choice among the window's **own** exposure observations, so chance is one over N by
construction and is identical for the model and for every analytic reference arm.

```
comp_ca = mean over i in W of ( native_hit_i - swap_stayed_i )

native_hit_i  = 1[ nearest-exposure pick under the NATIVE chain    == routed_i ]
swap_stayed_i = 1[ nearest-exposure pick under the ROLE-SWAP chain == routed_i ]
```

The role-swap exchanges the move-position sets of `routed` and a seeded partner over the same
board. The destination sequence, legality, and the stamped depth are all preserved; only the `mv`
command strings are re-rendered and re-encoded, then spliced back at the move positions. Two
guards make that splice honest: the encoder's directory-tree hash must match the one the root was
built with, and a sample of the window's **original** move commands is re-encoded and must
reproduce the cached embeddings to within a tight cosine tolerance. A wrong encoder, render, or
standardization frame fails loudly instead of recording noise.

## 2. Why it is hard to fake

- **Name-keyed non-tracker** — predicts the exposure at the queried name. The name index is
  untouched by the swap (only move embeddings change), so `native_hit == swap_stayed` **exactly,
  per window**. Contribution: exactly zero. This cancellation is structural, not statistical, and
  it is the reason the role-swap was chosen over a masked-endpoint content swap.
- **Chain-position non-tracker** (first mover / last mover / deepest / eliminate-the-static) —
  attends to move tokens, but under the routed↔partner exchange it mimics a tracker on
  routed-marker windows and anti-mimics on the symmetric partner windows. Drawn exchangeably, its
  expectation is zero.

  **This holds only if the partner is itself a mover, and enforcing that is load-bearing.** The
  routed content is always a mover, so if the partner is not, the exchange is one-sided: routed's
  move positions transfer to the partner and nothing comes back. A positional heuristic then flips
  from routed to the partner under the swap and scores `+1` on that window, and the windows that
  would cancel it — where the partner is the native marker and the swap hands the marker to routed,
  scoring `−1` — cannot occur, because a non-mover is never the native marker. The bias is
  systematic and positive, i.e. exactly farmable by the family this metric exists to exclude.

  The instrument inherited from the predecessor did **not** enforce this: in its recorded reference
  run the partner was a non-mover in roughly half of all probed windows. That was tolerable while
  the role-swap was a diagnostic probe; it is not tolerable now that the differential is the
  selection target. `cups_probe.alt_chain` now draws the partner from movers only, and `cups_ca`
  refuses to return a number if any probed window had a one-sided exchange.
- **History-ignorer or memorizer** — native and swap agree, so approximately zero.
- **Genuine tracker** — native picks the routed content, swap follows the partner. Positive.

## 3. Why it is not the capability gate

The pack's honest capability measurement is `pick_rate − ceiling_frozen`: an absolute rate minus a
frozen analytic per-cell ceiling. That is a conjunctive go/no-go, and it is the yardstick.

`comp_ca` is a **paired within-genome differential in which the ceiling never appears
arithmetically**. The table enters only to define the eligible window set, identically for every
genome, exactly as a fixed mask defines a slice. The two are different functionals of the same
measurement, and the separation is load-bearing: raising the pick rate uniformly lifts both arms
and leaves `comp_ca` unmoved, so a search that climbs `comp_ca` cannot thereby climb the gate. The
gate reading is carried in `private` on every measurement as a report. It is never an input to
selection.

## 4. The eligible slice W

```
W = { windows : N in {4,5}, style == core, ceiling[N,depth,m,R] < 0.99, depth >= 2,
                a legal role-swap partner exists }
```

Every term is a property of the window or the frozen table — never of the net.

- `ceiling < 0.99` keeps only cells where a non-tracking strategy is **not** already saturated. A
  margin in a saturated cell is unearnable and contributes only dilution.
- `style == core` is the board style the model trains on; held-out style measures style *transfer*
  and is reported separately, never scored.
- **`depth >= 2` is inert at the frozen knobs, and that is a finding, not an assumption.** Of the
  ceiling table's 100 cells only 15 are earnable; of those exactly 8 have N in {4,5}; and every one
  of those 8 has depth 3 or 4. So the earnable slice is *already* a depth-three-or-more slice. The
  design question of "depth at least two versus at least three" is settled by the data and does not
  need a decision. The floor is kept explicit anyway, so that a re-mint at different knobs cannot
  silently widen the slice to shallow windows an analytic heuristic already solves.

Per-depth cells `d2 / d3 / d4plus` are emitted separately so a mechanism's signal can be located; at
the frozen knobs only `d3` and `d4plus` are populated.

**The correctness trap this design exists to avoid.** Do not compute `comp_ca` by subtracting the
instrument's rounded aggregate fields. The two arms round to four places *and* aggregate over
populations that do not coincide: the native arm's earnable selector applies no depth filter, while
the swap arm's requires depth at least two and silently drops windows that had no legal partner.
`cups_ca` therefore intersects the two arms' **per-window rows by window id**, asserts the rows
agree on every window property, and means the differences unrounded.

## 5. What the reference run says — and what it does not

Recomputed from the predecessor's recorded pack run (its strongest genome, three seeds, full step
budget):

| seed | n | native pick rate | swap stayed | comp_ca *(old draw)* |
|---|---|---|---|---|
| 0 | 78 | 0.3077 | 0.2949 | +0.0128 |
| 1 | 78 | 0.3077 | 0.1923 | +0.1154 |
| 2 | 78 | 0.2821 | 0.1923 | +0.0898 |

mean +0.073, seed sd 0.053, against a capability gate reading of −0.198 on the same net.

**These comp_ca values are not valid under the current definition and must not be used as a
baseline.** They were computed from a run whose role-swap drew a non-mover partner in about half of
all probed windows (§2), which biases a positional heuristic systematically positive. How much of
that +0.073 was tracking and how much was the one-sided-exchange artifact is unknown and cannot be
recovered from the recorded aggregates — the run would have to be repeated under the corrected
draw. Treat the reference genome's comp_ca as **unmeasured**.

What survives from that run is the *structural* observation, which does not depend on the partner
draw: the gate and the differential are different functionals, so a genome can sit far below the
analytic ceiling in absolute terms and still be a candidate for a non-zero tracking differential.
The seed spread and the slice size below also survive, since they are properties of the window
population rather than of the metric.

Three consequences worth stating plainly, because they shape what the first rounds can conclude:

1. **The slice is small.** n = 78 windows, and the windows are the *same* for every candidate. The
   seed-to-seed spread above is the same order as the value itself. A three-seed mean has a
   standard error near 0.03, so differences smaller than roughly 0.06 are not distinguishable at
   this n. Widening W — dropping the core-style restriction, or admitting the earnable cells at
   lower N — is the obvious lever and is deliberately a single constant in `cups_ca.py`.
2. **The seeds are fixed.** Given a genome and a seed the eval is essentially deterministic, so
   repeated runs measure reproducibility, not sampling. A noise floor measured by re-running the
   same seeds will therefore look far smaller than the real uncertainty and will license treating
   noise as signal. The floor must instead be derived from the spread of three-seed means across
   *disjoint* seed triples. It is left unset in the contract until that is measured.
3. **The population starts near chance.** The reference genome's native pick rate on W sits a few
   points above chance. A guard requiring the net to clear chance by a fixed margin — which the
   original design proposed — would fail it, and would null every early candidate, leaving the
   search unable to climb out of the regime it starts in. That quantity is therefore **reported on
   every measurement and not enforced**. It is a capability claim, and this search does not gate on
   capability.

## 6. Guards, and which ones bite

Enforced (a candidate failing these has no usable number):

| guard | what it catches |
|---|---|
| encoder tree hash matches the root's stamp | a wrong eye silently redefining every embedding |
| self-parity of re-encoded original commands | a wrong render or standardization frame |
| eligible slice non-empty | a measurement over nothing reported as a small number |
| every realized cell present in the ceiling table | a table/root mismatch quietly shrinking W |
| prediction-bank norm and angular dispersion floors | a collapsed or constant predictor |
| no future leakage; head declares itself leak-safe | seeing the observation being predicted |
| training did not diverge | non-finite loss |

Reported, not enforced: the pick rate over chance (§5.3), and the capability gate reading.

The norm and dispersion floors are **inherited from a different instrument and have not been
calibrated for this quantity.** They are carried as a starting point, the realized values are
emitted on every measurement, and they should be re-set from measured data before any failure there
is read as a statement about a candidate.

## 7. The lane

One net per (genome, seed), trained on the pack root and standardized on **that root's own** train
statistics. Both the compositional metric and the world-model health readout come off that same
net, because training twice to measure two things off it is waste.

The predecessor measured a base-world fitness on a second root and treated it as a floor. That is
dropped, for a measured reason: its strongest genome has three archived full three-seed records
spanning 0.4251 / 0.4247 / 0.4262, while the entire top-thirteen leaderboard band is 0.0019 wide.
Same-genome re-run noise is most of the visible spread, so a base-lane floor would have doubled the
compute for a quantity that barely discriminates. The collapse check is instead the pack net's own
next-observation retrieval, computed on the same net for free.

Frames are never pooled. The pack root is its own comparability frame, and a number from another
root, eye or environment is not comparable to one from this lane without a measured offset.

## 8. Changing any of this

The eligible slice, the metric form, and every guard threshold are single named constants in
`evolve/cups_ca.py`. Changing one changes what the search means, which makes everything scored
before it incomparable. Such a change is a dated note in this file and a fresh baseline — not a
tweak.
