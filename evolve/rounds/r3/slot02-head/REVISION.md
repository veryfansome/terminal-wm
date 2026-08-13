# Revised after review; review.json describes the PRE-revision code

The reviewer's fatal finding stands against the original: at the scored read the retrieval had no
query for the location being read, because the query was gated on `live_cmd & live_obs` and that
observation is withheld by construction, and because arrays were indexed by pair so a row whose
read is the last token had no entry at all. Measured on the real scored layout, perturbing only
this head's parameters moved the scored prediction by 0.000000 / 0.000000 / 0.007958.

The inventor was re-briefed once with that measurement and no prescribed fix, with withdrawal
offered as a legitimate outcome. It revised: the pair axis is split into a read axis (one entry
per command token, so the final unpaired command has one) and a write axis (one per completed
pair), and the single liveness mask is split into read_ok = live_cmd for the query and
write_ok = live_cmd & live_obs for the value.

Re-measured with the same probe that condemned it: 0.000363 / 0.075117 / 0.129069. The mechanism
now acts where the score is taken, including on the row whose read previously had no slot. Row 0
remains about two hundred times weaker than row 2, so "live" is established and "strong" is not.
