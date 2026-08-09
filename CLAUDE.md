# terminal-wm — Context

A shell **world model** over real Linux Docker filesystems, improved by an evolutionary search whose
objective is **compositional depth**: can the model track *which content* now sits at a location
after a silent chain of moves, on a system it has never seen?

Earlier work on this world model and its instruments is at
<https://github.com/veryfansome/jepa>. Everything below describes this repo.

## What the search is

**Parent selection and breeding a diverse set of effective solutions.** Parents are sampled by
fitness and novelty with an offspring penalty, so the search explores lineages rather than deepening
one; every scored candidate enters the archive and stays there, negatives weighted the same as wins;
each round's output is N candidates measured against their own parents.

This repo owns **no** selection machinery. The [`evolve` plugin](https://github.com/veryfansome/claudemods)
owns the archive, parent sampling, novelty dedup, isolated-export scoring, budgets, the inventor jail
and the holdout firewall. This repo supplies exactly three things: an evolvable surface, an eval
command, and a contract file. Nothing else.

Validity is a per-candidate property, enforced at score time by the gates in `eval/`. A candidate
found invalid afterwards is retracted append-only (`evolve retract`). Materializing an archived
candidate to use somewhere is an ordinary engineering act (`evolve apply`) taken outside the search,
and it never feeds back into selection.

## What is scored

`combined_score = comp_ca_margin` — one scalar, defined in `evolve/cups_ca.py`:

```
comp_ca        = mean over the eligible windows of ( native_hit - swap_stayed )
comp_ca_margin = comp_ca - the best analytic non-tracker's own comp_ca on the same windows
```

A window exposes N contents at N locations, silently moves them around in a chain, then reads one
location. `native_hit` is whether the model's next-observation prediction picks the content that
the *native* chain routed there. `swap_stayed` is whether it still picks that same content when the
chain is **role-swapped** — the move-position sets of the routed content and a partner exchanged,
over the same board, same destinations, same depth, only the move commands re-encoded.

The point is what cancels. A model that keys on the *name* being asked about predicts identically
under both chains, so its per-window difference is exactly zero — structurally, not on average, and
verified so on the real slice. A model that keys on chain position cancels only in expectation, and
the scored slice is one frozen realization where a first-mover lookup does score positive — which is
why the band of analytic non-trackers is measured on every run and subtracted. Zero means *no better
than the best depth-zero shortcut*.

**The capability gate is not the objective.** The pack also has an honest absolute measurement —
the pick rate against a frozen analytic per-cell ceiling. That is the yardstick, and the search
must never learn to climb it. `comp_ca` is a differential in which the ceiling never appears
arithmetically; the table enters only to decide which windows are eligible, identically for every
genome. The gate reading rides along in `private` as a report and is never an input to selection.

## Code map

| path | what it is |
|---|---|
| `realenv/seq_worldmodel.py` | the world model: a causal transformer over interleaved command/observation embeddings, plus the retrieval primitives. Immutable from a genome's point of view. |
| `evolve/chunks/<axis>/` | **the evolvable surface** — one impl file per mechanism, seven axes. Append-only: an archived genome must keep meaning what it meant when scored. |
| `evolve/genome.py` | resolves a genome's impl names to modules, and the structural gate |
| `evolve/harness.py` | the training closure — `_train` and the fail-closed encode. Nothing scores here. |
| `evolve/cups_probe.py` | the pack instrument: the N-way forced-foil measurement, the analytic ceiling arms, and the role-swap probe |
| `evolve/cups_ca.py` | **the objective** — `comp_ca`, built by intersecting the two arms per window |
| `eval/adapter.py` | the fitness oracle the engine calls: one net per seed, one `metrics.json` |
| `eval/smoke.py`, `eval/guard_leakage.py` | the pre-eval gates every candidate pays |
| `evolve/evolve.json` | the contract (engine-owned format) |
| `evolve/genomes/` | the starting population |
| `cloud/pack_lane.sh` | the GPU lane: pull + pin the eye, encode the root, score, ingest |
| `research/compositional-selection-design.md` | why the objective is shaped this way |

## Rules

- **Record stats, not verdicts.** Negatives are archived with the same weight as wins. A round with
  no new best is a first-class result: N candidates measured against their own parents, and that
  trade-off frontier is the deliverable.
- **The eval is sacred.** No candidate ever edits the metric, the split, or `eval/`. Deliberate
  harness maintenance happens outside a round with `EVOLVE_ALLOW_PROTECTED=1`, followed by
  `evolve doctor`.
- **Never score the `final` split for selection.** It exists for one-shot, report-only validation
  before an external claim. The engine firewalls it; don't route around it.
- **No step-reduced proxy.** The cheap tier is fewer *seeds* at full step count. A shortened proxy
  has been measured to invert the ranking of exactly the slow-converging memory and architecture
  mechanisms this objective is about. `eval/adapter.py` ignores the tier when choosing
  the step budget, structurally.
- **Scores compare only within one environment.** Remote results come back through
  `evolve ingest --env <tag>`. Measure comparability with `evolve doctor --measure-env-offset`
  before folding two environments; don't assume it.
- **Failed traits stay live.** Deprioritize, don't foreclose — a trait that failed alone can win
  recombined into a changed context.

## Doc-sync triggers

- new axis, or a change to the objective / gates / split / adapter → update this file and
  `evolve/EVOLVE.md` in the same commit
- a change to the eligible slice, the metric form, or a guard threshold → that is a change to what
  the search means; it belongs in `research/compositional-selection-design.md` with a dated note,
  and it invalidates comparability with everything scored before it
