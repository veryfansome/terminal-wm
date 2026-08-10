# The cd-history probe

What `evolve/cdh_probe.py` measures, and why each guard in it exists. This capability is reported
for every candidate and never scored.

## 1. The window

The cd-history pack presents, inside an ordinary shell trajectory, a navigation block followed by
a read:

```
[ cmd_0, obs_0, ... , cd d_i, cd d_j, cd -, cat name ]
```

`cd -` returns to `d_i`, so the file the read resolves is determined by the *command* history and
by nothing else. The prediction is taken at the `cat name` command position; the read's own
observation is withheld and is the target. A window qualifies only with at least two `cd` steps of
history. The nav block is the contiguous run of `arm == 'cdhist'` `cd` steps immediately before the
read — the pack interleaves real `ls` reads (role `cdh_ctx`) between the `cd`s, and those are not
nav.

Scoring is strict-tie top-1 retrieval against same-verb foils, averaged over four rounds with a
fixed foil seed so the model and every ceiling arm are a paired comparison. Ties never count as
beating the true candidate, which is what puts a predict-the-mean model at chance by construction.

## 2. The two measurements

**`masked_s1` — the GO/NO-GO gate.** The model's content-top-1 margin over the analytic ceiling,
computed twice: with the navigation observations visible, and with them masked.

The unmasked arm is not the interesting one. `render_obs` bakes the current working directory into
a `cwd=/tmp/w/cdh/dN` token in every observation, so an unmasked read is clearable by copying the
preceding `cd -` observation — the copy-previous arm *is* the cwd oracle, and it sits in the
unmasked ceiling for exactly that reason. With the navigation observations gone, the only remaining
route to the landing directory is the `cd` command sequence itself. A positive **masked** margin is
therefore the claim: deep history-routing that can express on a masked forward. A treatment where
content is a function of the directory can pass it; a name-keyed control fails it by construction,
because retrieve-by-command already decodes its read.

**`nav_probe` — the null disambiguator.** On the same windows, the differential between the real
navigation history and a wrong one: each window's nav *command* tokens are replaced with those of a
donor matched on nav depth and redirect-ness whose landing directory differs. The read command and
the target are the window's own. A model that reads the navigation predicts the donor's landing
content and collapses; a model that ignores it is unchanged. This is measured on its own rather
than derived from any composition metric, so a null composition result stays interpretable:
routing-learned-but-disjoint and routing-not-learned are different findings.

## 3. The ceiling arms

Genome-independent, all computed on the same windows:

| arm | what it is | in which ceiling |
|---|---|---|
| retrieve-by-command | nearest train read observation by read command | both |
| copy-previous | the previous observation, i.e. the cwd oracle | unmasked only |
| cdh centroid | the mean train cdh read target | both |
| global centroid | the mean train observation | both |

The cdh centroid is there so the ceiling cannot be cleared by predicting "a cdh-ish blob": content
that is a function of the directory means few distinct targets per image, and that cluster sits far
from the global centroid. Copy-previous is dropped from the masked ceiling because the mask
neutralizes it, exactly as it neutralizes the model's own copy.

An **empty retrieve-by-command bank raises** rather than scoring. It means the probed net never
trained on data containing this pack, which is a degenerate ceiling that posts a false GO *and*
flips the permutation control to a false pass — the two readings that would otherwise cross-check
each other both fail in the same direction.

## 4. The four ways the gate could be faked, and what closes each

**The mask must cover every cdh observation in the prefix, not just the window's own nav block.**
An earlier cdh block that transited the landing directory leaves a visible `cwd=d_i` observation
the model can read instead of routing. This works because `cwd=/tmp/w/cdh/dN` is namespaced: it
appears only in cdh-arm observations, so masking those removes every landing token and no non-cdh
observation can carry one.

**Under the mask, `to_obs` must be fed a zeroed previous observation.** A z_prev-dependent target —
delta, residual, or learned, anything of the form `to_obs = z_prev + pred` — would re-inject the
very cwd token the mask just removed. The gate would then be passable by a net that does no routing
at all.

**The raw-output leak scan cannot see the render.** The scan checks that the landing path appears in
no prefix observation's `output`, and reads `stdout` and `stderr` independently of that reconciled
field so a partition bug cannot hide a leak. But a cwd-in perception re-injects the landing as a
`cwd=` token during rendering, downstream of anything in the raw record. So the probe additionally
reads the root's perception stamp, renders a probe observation through that impl, and fails loud
unless the landing is absent — the root must have been encoded with a cwd-dropping render.

**Windows preceded by any earlier cdh material are dropped.** Earlier windows' reads are a
nav-independent *elimination* channel: having seen A, B and C, the answer is D. That is not
routing. `key_pad_noprior` — a layout that masks the observations of earlier `cdh_read` steps while
leaving the navigation untouched — exists to measure that channel directly, and its reading travels
with every measurement.

For the redirected windows (`cd - >/dev/null`), the landing must appear in no observation at all,
including cdh-arm ones, which is what makes the unmasked forward leak-free and the `redir_only`
slice the honest one.

## 5. Wrong-history donors

A donor must match the window on nav depth and on redirect-ness — presence-matched — and must land
somewhere else. Matching on presence is what makes the differential about *which* history rather
than about how much history there is. Windows with no eligible donor keep their own tokens and are
counted; the matched-only differential is reported beside the pooled one so an unmatched remainder
cannot dilute the reading toward zero.

The masked arm of `nav_probe` is the command-only routing test. The unmasked arm is the valid one
under a cwd-out render, where there are no cwd observations to mask and masking the empty `cd`
observations would break the architecture's per-(command, observation) state chain.
