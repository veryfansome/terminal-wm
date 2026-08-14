# Revised after review; review.json describes the PRE-revision code

The reviewer's fatal finding stands against the original. The scored window is
[cmd_0, obs_0, ..., cmd_r] of length 2r+1 — odd, because the read's observation is absent rather
than padded, that observation being the answer. The head computed n = L // 2 and bailed whenever
2*n != L, so on every scored forward it returned the bare trunk. Measured against the bare
architecture: 0.08626 movement on an even layout, exactly 0.0 at the scored read.

Re-briefed once with that measurement and no prescribed fix, with withdrawal offered. It revised,
and made the design calls itself: index by command slot ((L+1)//2, matching the arch) rather than
by pair; read pairing per row from key_pad instead of inferring it from length parity; require an
observation for the content deposit but NOT for transport, since which file moved where lives only
in the command text; exclude unpaired slots from the aux statistics.

Re-measured with the same probe: 0.271 at the scored read of an odd layout, where it was 0.0.
The gate that caught it now reports head_off_at_scored_position: None.

Worth recording that the revision found a case the original defect concealed: batching windows of
different depths to a common EVEN length leaves a short row's final command unpaired while L is
even, so parity was never the right test in the first place.
