# insights — neutral running numbers

Numbers and setup, not conclusions. Editorializing here biases the next round; negatives carry the
same weight as wins. Distil recent archive records into this file every several generations.

## Starting facts (measured before any round, from the predecessor's recorded pack run)

Reference genome, three seeds, full step budget, on the focused pack:

| seed | eligible windows | native pick rate | swap stayed | comp_ca *(old draw — see below)* |
|---|---|---|---|---|
| 0 | 78 | 0.3077 | 0.2949 | +0.0128 |
| 1 | 78 | 0.3077 | 0.1923 | +0.1154 |
| 2 | 78 | 0.2821 | 0.1923 | +0.0898 |

- **the comp_ca column is retracted as a baseline.** That run drew a non-mover role-swap partner in
  0.529 of probed windows, which biases positional heuristics systematically positive; the split
  between tracking and artifact is unrecoverable from the recorded aggregates. The reference
  genome's comp_ca under the corrected draw is **unmeasured**.
- the same net's capability gate reading on that slice: −0.1984 (reported, never selected on)
- chance on this slice is one over N with N in {4,5}; the native pick rate sits a few points above it
- the eligible slice at the frozen knobs: 8 of 100 ceiling cells, all at depth 3 or 4; realized
  populations d3 n=54, d4 n=24
- the windows are identical across candidates and seeds; only the trained net differs

## Open, unmeasured

- **noise floor**: unset. Must come from the spread of three-seed means across *disjoint* seed
  triples, not from re-running fixed seeds (that measures reproducibility, not uncertainty).
- **anti-degeneracy thresholds**: the norm and angular-dispersion floors in `cups_ca.py` are
  inherited from a different instrument and have never been calibrated for this quantity. The
  realized values are emitted on every measurement — collect them before treating a failure there as
  a statement about a candidate.
- **starting population**: 17 genomes written, none scored on this lane. Eleven come from the
  predecessor's scoreboard (whose visible band was narrower than its own same-genome re-run spread,
  so rank carried little information); six were added for mechanism-family coverage the scoreboard
  never rewarded.
