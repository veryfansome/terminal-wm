# terminal-wm — Context

A shell **world model** over real Linux Docker filesystems, improved by an evolutionary search whose objective is **compositional depth**: can the model track *which content* now sits at a location after a silent chain of moves, on a system it has never seen?

Earlier work on this world model and its instruments is at <https://github.com/veryfansome/jepa>. Everything below describes this repo.

## What the search is

**Parent selection and breeding a diverse set of effective solutions.** Parents are sampled by fitness and novelty with an offspring penalty, so the search explores lineages rather than deepening one; every scored candidate enters the archive and stays there, negatives weighted the same as wins; each round's output is N candidates measured against their own parents.

This repo owns **no** selection machinery. The [`evolve` plugin](https://github.com/veryfansome/claudemods) owns the archive, parent sampling, novelty dedup, isolated-export scoring, budgets, the inventor jail and the holdout firewall. This repo supplies exactly three things: an evolvable surface, an eval command, and a contract file. Nothing else.

Validity is a per-candidate property, enforced at score time by the gates in `eval/`. A candidate found invalid afterwards is retracted append-only (`evolve retract`). Materializing an archived candidate to use somewhere is an ordinary engineering act (`evolve apply`) taken outside the search, and it never feeds back into selection.

## What is scored

`combined_score = comp_ca` — one scalar, defined in `evolve/cups_ca.py`:

```
comp_ca = mean over the eligible windows of ( native_hit - swap_stayed )
```

A window exposes N contents at N locations, silently moves them around in a chain, then reads one location. `native_hit` is whether the model's next-observation prediction picks the content that the *native* chain routed there. `swap_stayed` is whether it still picks that same content when the chain is **role-swapped** — the move-position sets of the routed content and a partner exchanged, over the same board, same destinations, same depth, only the move commands re-encoded.

The point is what cancels. A model that keys on the *name* being asked about predicts identically under both chains, so its per-window difference is exactly zero — structurally, not on average, and verified so on the real slice. A model that keys on chain position cancels only in expectation, and the scored slice is one frozen realization where a first-mover lookup does score positive. So the band of analytic non-trackers is measured on every run and **reported beside the score**. It is not subtracted: at this slice size the in-sample maximum is largely noise, and a committed lookup can beat it, so no point estimate of the band is a bound. Read a score against the band, don't expect the number to have the band already removed.

**The capability gate is not the objective.** The pack also has an honest absolute measurement — the pick rate against a frozen analytic per-cell ceiling. That is the yardstick, and the search must never learn to climb it. `comp_ca` is a differential in which the ceiling never appears arithmetically; the table enters only to decide which windows are eligible, identically for every genome. The gate reading rides along in `private` as a report and is never an input to selection.

## Code map

| path | what it is |
|---|---|
| `realenv/seq_worldmodel.py` | the world model: a causal transformer over interleaved command/observation embeddings, plus the retrieval primitives. Immutable from a genome's point of view. |
| `evolve/chunks/<axis>/` | **the evolvable surface** — one impl file per mechanism, seven axes. Append-only: an archived genome must keep meaning what it meant when scored. |
| `evolve/genome.py` | resolves a genome's impl names to modules, and the structural gate |
| `evolve/harness.py` | the training closure — `_train` and the fail-closed encode. Nothing scores here. |
| `evolve/cups_probe.py` | the pack instrument: the N-way forced-foil measurement, the analytic ceiling arms, and the role-swap probe |
| `evolve/cdh_probe.py` | a second capability pack — does a read route through the navigation history that actually happened. Runs only when `TWM_CDH_ROOT` points at the cd-history pack, and never enters selection. No scored campaign has mounted it, so `cdh_routing` is null in every archived record; a run that wants the reading has to set that root. |
| `evolve/cups_ca.py` | **the objective** — `comp_ca`, built by intersecting the two arms per window |
| `eval/adapter.py` | the fitness oracle the engine calls: one net per seed, one `metrics.json` |
| `eval/smoke.py`, `eval/guard_leakage.py`, `eval/guard_stream.py`, `eval/guard_reach.py` | the gates: it builds, it cannot see the future, its tokens are ones the instrument can reproduce, and the parameters it introduces can actually affect something. Which path pays which differs — `evolve score` runs all four in its guardrail phase, while the pack lane calls `eval/adapter.py` directly, so there only the stream and reachability gates (called from the adapter, pre-training) and the leakage check (post-training) run |
| `evolve/evolve.json` | the contract (engine-owned format) |
| `evolve/retired_impls.json` | impls whose mechanism is retired — kept on disk so archived genomes stay resolvable, but never offered to a new candidate. Retiring an impl and retracting the candidates selecting it are two separate acts; doing only the first leaves the mechanism reachable as a parent |
| `evolve/jail_sample.py` | writes the real trajectories an inventor reads inside its jail, from the train split only |
| `evolve/capture_round.py` | copies a round's hypotheses, genomes and impls into `evolve/rounds/` the moment the inventors finish, before anything is scored |
| `evolve/genomes/` | the starting population |
| `evolve/blend_root.py` | writes a blend **spec**: the constituent packs, their ratios, the seed and the sampled indices for a training set composed at load time |
| `cloud/build_context.py` | derives the shared lane context once — the composed training set, standardized splits, window layouts, role-swap chains — so no candidate re-pays it. Resolves a blend spec and pins the standardization frame |
| `cloud/runner.py` | runs (genome, seed) jobs concurrently against that shared context |
| `cloud/pack_lane.sh` | the GPU lane: pull + pin the eye, encode the root, measure, fold each genome into a record |
| `cloud/lane.sh` | **how to run a campaign** — provision, measure, bring results home, stop the box. `verify` gates `terminate`, so a pod is never stopped before its records are on local disk, and a failure leaves the box running and says so. Running the stages by hand is how a campaign was lost and how a box billed for eleven idle hours |
| `research/compositional-selection-design.md` | why the objective is shaped this way |

## Rules

- **Record stats, not verdicts.** Negatives are archived with the same weight as wins. A round with no new best is a first-class result: N candidates measured against their own parents, and that trade-off frontier is the deliverable.
- **The eval is sacred.** No candidate ever edits the metric, the split, or `eval/`. Deliberate harness maintenance happens outside a round with `EVOLVE_ALLOW_PROTECTED=1`, followed by `evolve doctor`.
- **Never score the `final` split for selection.** It exists for one-shot, report-only validation before an external claim. The engine firewalls it; don't route around it.
- **No step-reduced proxy.** The cheap tier is fewer *seeds* at full step count. A shortened proxy has been measured to invert the ranking of exactly the slow-converging memory and architecture mechanisms this objective is about. `eval/adapter.py` ignores the tier when choosing the step budget, structurally.
- **Scores compare only within one environment.** Remote results come back through `evolve ingest --env <tag>`. Measure comparability with `evolve doctor --measure-env-offset` before folding two environments; don't assume it.
- **Failed traits stay live.** Deprioritize, don't foreclose — a trait that failed alone can win recombined into a changed context. Retiring an impl is the exception and it is about *reachability*, not about a score: a mechanism keyed on a condition the instrument never produces cannot be measured at all, so leaving it in circulation spends slots on a question no run can answer.
- **A measurement the instrument cannot reproduce is not a measurement.** The probe builds its own tokens from the embedding cache, so anything a stream does to command tokens must also be a pure `code_cmds(cmds, z_cmd)` it can replay — on the native chain and on the role-swapped chain separately. Coding one arm inflates the differential, coding the swapped arm from native strings deflates it, and both are silent.
- **The brief states how the metric cancels, never what will score.** Location-keying cancels exactly, per window; chain-position keying does not, and on this frozen slice a first-mover lookup outscores every mechanism measured so far. That is a fact about the instrument and belongs in `jail_notes`. Whether a candidate leaned on it is detected in measurement (`private.shortcut_leaning`), not legislated in the brief — the rule that once tried to legislate it stated the opposite of the truth.

## Doc-sync triggers

- new axis, or a change to the objective / gates / split / adapter → update this file and `evolve/EVOLVE.md` in the same commit
- a change to the eligible slice, the metric form, or a guard threshold → that is a change to what the search means; it belongs in `research/compositional-selection-design.md` with a dated note, and it invalidates comparability with everything scored before it
