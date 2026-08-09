# evolve/ — this project's evolutionary search (contract + round manual)

An LLM-driven evolutionary search over a marked part of this repo (registry mode), run by the `evolve` plugin: inventor agents propose candidates, a deterministic engine decides what enters the archive and which parent seeds the next generation. This file is the project-local manual; the config in `evolve.json` is the machine-readable contract.

## The contract (fill these before anything evolves; `evolve doctor` enforces them)

- **task** — one line: what is being maximized, in plain words.
- **surface** — what a candidate may change. `markers`: only text between `EVOLVE-BLOCK-START`/`END` comment lines in the declared files. `registry`: a genome selecting one impl (+ params) per axis from `evolve/chunks/<axis>/`; impls are append-only (never edit an archived impl — genomes must keep meaning what they meant when scored).
- **eval** — commands that score a candidate. Each writes `{results_dir}/metrics.json` (or prints one JSON object) with `combined_score` (float, maximized — required), optional `public` (shown to inventors), `private` (recorded aside, never shown), `text_feedback` (fed to descendants), `correct` (false = failed candidate). Placeholders available: `{results_dir}` `{seed}` `{split}` `{mode}` `{genome}`.
- **proxy vs full** — proxy is the cheap screen every candidate gets; full (more seeds, bigger budget) is for candidates that beat their parent by more than the measured noise floor. Trust only full-budget results.
- **splits** — fitness is scored on `inner`. If you declare a `final` split (a held-out-of-held-out eval), the engine firewalls it from selection structurally: it is for one-shot, report-only validation before an external claim — never to pick parents.
- **protected** — paths no candidate may touch: the config (`evolve/evolve.json`), the archive (`evolve/archive/**`), and every eval asset you author (adapter, harnesses, holdout data). Do NOT protect the registry dir or `insights.md` (the round writes those). Enforced three ways: a tool-layer hook (disarmed until the first `evolve seed`; new files under the registry dir are always allowed), the structural guard, and isolated-export scoring (candidates run in a `.git`-less export of HEAD plus exactly the candidate — planted files never reach the eval, and eval code can't reach back to this directory). Isolation is input/discovery-level, not an OS sandbox: the eval runs candidate code with your privileges, so run evolution in a trusted environment. The eval runs in a git-less export, so an adapter that shells out to `git` won't find a repo — read commit info before scoring, not inside the eval.
- **noise floor** — measured, not asserted: `evolve doctor --measure-noise` runs the baseline k times and writes the spread. Proxy deltas under the floor are noise; don't conclude anything from them.

## How a round runs (the `/evolve:round` skill drives this)

1. Re-ground: `evolve board`.
2. `evolve sample --k N --seed <round-tag>` — parents (fitness+novelty weighted, offspring-penalized), operators, inspirations per slot. Refuses when the budget is spent. The slot table persists to `evolve/rounds/<seed>-<hash>-slots.json` (exact path printed as `slots_file`) as the round's evidence of what was assigned; use a fresh seed each round and never edit the file after dispatch.
3. Brief + isolate per slot: markers mode — `evolve brief` + a worktree per inventor; registry mode — `evolve jail --parent <id> --axis <axis> --round <tag> --slot <n> --seed <slot seed> --operator <op> [--cross-with <id>]` builds an isolated workspace whose BRIEF.md is the information-diet brief (parent genome + parent fitness + baseline + prior mechanisms as source, nothing else — asserted, not assumed), and the inventor runs with the jail as its cwd. Declare `surface.registry.inventor_files` (the harness modules an impl may import) once; add project notes (dataset roots, machine budgets) via `surface.registry.jail_notes`.
4. `evolve guard` each proposal → fix-or-re-brief on violations (bounded retries).
5. `evolve score --mode proxy` each survivor — guard, dedup, pristine clone, smoke, eval, record. Failures are recorded with reasons; they are search signal. The proxy **screens** (decides whether to spend a full run); it never concludes anything.
6. Trust only full budget: re-score keepers (`evolve rescore --id <c> --mode full`); for noisy evals, pair a parent re-run at proxy first. Child-vs-parent at full budget is the round's result.
7. Recombine winners (cross operator / stacked genome) and score the combination — epistasis is real in both directions; always test, never assume.
8. A round with no new best still closes with the measured trade-off frontier as its result.
9. Before any external claim: score that candidate once on the `final` split — report the number, never re-rank on it.

## Validity — per candidate, not per decision

The search scores, selects parents, and breeds; every round's output is N scored candidates measured against their parents. Validity is a property of a candidate, established when it is scored — not a decision taken later about which candidate is best. What the engine enforces:

- **`eval.guardrails` run against every candidate at score time.** A candidate without a PASS has no usable number — validity checks (causality, provenance, structural) belong here, so they cost every candidate the same and nothing needs a gate ceremony.
- **A post-hoc validity discovery is an append-only retraction** — `evolve retract --id <c> --reason "<why>"`. The verdict is cross-partition (a mechanism invalid on one dataset is invalid on every dataset — a regime change must not launder it), earlier scores stop counting, history stays intact, and a deliberate rescore under a fixed eval reinstates. Numeric ingests for a retracted id are refused without `--reinstate`, so a re-measurement wave can't reinstate one by accident. Verify a retraction by its EFFECT on `evolve board`, never by the write succeeding. (Remote result payloads may equivalently carry `{"retract": true, "fitness": null, "guardrail": "<reason>"}`.)
- **Retired mechanisms** — optionally record impls whose mechanism is invalid or inexpressible in `evolve/retired_impls.json` (`{"retired": {"<axis>/<impl>": {"reason": "...", ...}}}`): `evolve impls` stops listing them and briefs never offer a genome selecting one as inspiration. Retiring an impl does NOT retract its carrier records — retract those too, or they stay samplable as parents (`evolve doctor` warns).
- **Shipping is outside the engine.** `evolve apply --id <c>` materializes an archived artifact (genome or patch; `--dry` to view); what you do with it is an ordinary engineering decision, and nothing about it feeds back into selection.

## Culture

- **Record stats, not verdicts.** Negative results get archived with the same weight as wins.
- **The eval is sacred.** No candidate, ever, edits the metric, the splits, or this directory. Deliberate harness maintenance is a normal reviewed change, made outside a round (set `EVOLVE_ALLOW_PROTECTED=1` for the session doing it), followed by `evolve doctor` and a fresh `--measure-noise`.
- **Scores compare only within one environment.** Remote/offloaded results come in via `evolve ingest --env <tag>`. If the archive mixes environments (e.g. local proxy runs plus ingested remote scores), set `fitness.selection_env` to the one env selection should trust — records from other envs stay archived and reportable but are firewalled out of parent sampling and the leaderboard (like the final split). Set it to the env you actually score in; note the seed is scored in the host env, so if you pin `selection_env` to a remote box, re-baseline a candidate there or selection will be empty (`board`/`doctor` warn when that happens). Before trusting a new environment, run `evolve doctor --measure-env-offset` — it re-runs the top-scoring record (or `--ref <id>`) there and records the fitness offset vs its home env, so you *measure* comparability instead of assuming it: within the noise floor → fold the envs; beyond it → partition with `selection_env`.

---

## Project specifics (terminal-wm)

### Before anything scores
The eval needs three things that are NOT in this repo and cannot be, because the engine scores in a
git-less export of HEAD and injects no environment:

| env var | what it points at |
|---|---|
| `TWM_CUPS_ROOT` | absolute path to the **encoded** pack root (`…-nocwd`), built once by `cloud/pack_lane.sh prepare` |
| `TJ_FT_ENCODER` | absolute path to the pinned encoder checkpoint. There is deliberately no default — the old one was the wrong eye, and substituting it corrupts every embedding silently |
| `TWM_PYTHON` | absolute interpreter with torch. The clone is a bare export; do not build a venv per candidate |

Optional: `TWM_EYE_TREE_SHA` to assert the encoder's identity at preflight, `TWM_STEPS` to shorten
training for a wiring test only.

`eval/adapter.py::preflight` checks all of this **before** any candidate code runs and raises, so an
environment problem surfaces as a broken run instead of being recorded as some candidate's null.

### The tiers
`proxy` is one seed, `full` is three — both at the **same step count**. The tier never shortens
training. A step-reduced proxy has been measured to invert the ranking of exactly the
slow-converging memory and architecture mechanisms this objective is about, and the deepest one
timed out at proxy. `eval/adapter.py` ignores `{mode}` when choosing the step budget so
this cannot drift back in.

### The noise floor is deliberately unset
Do **not** fix it with `evolve doctor --measure-noise`. Given a genome and a seed this eval is
essentially deterministic, so repeated runs measure reproducibility, not uncertainty — the floor
would come back near zero and license treating noise as signal. The real uncertainty is
seed-to-seed: on the reference genome the per-seed spread is the same order as the value itself, on
a slice of fewer than a hundred windows. Derive the floor from the spread of three-seed means over
*disjoint* seed triples, then hand-write it into `fitness.noise_floor` with `noise_meta` left null
(the engine then refuses to clobber it and will not nag).

### Recombination is the point
On the next-observation margin these mechanisms were previously ranked by, the whole visible band
across the top performers was narrower than the same-genome re-run spread — rank carried almost no
information, mechanism family did. The starting population in `evolve/genomes/` is therefore chosen
for family coverage rather than for inherited rank, and it deliberately includes families that
ranking never rewarded — in particular a content-conditioned transition operator, where the widely
used alternative conditions on the command only and therefore provably cannot express a
content-dependent composition. `search.op_probs` weights `cross` above the plugin default for the
same reason.

### Reading a result
`combined_score` is `comp_ca`. Also look at `public.per_depth` (where the signal lives — a
shallow-only lift reads very differently from a flat one), `public.native_wm` against
`public.chance` (is the net off chance at all), and `public.wm_health_top1_sameverb` (is it a
working world model). `private.gate_report` carries the honest capability reading; it is a report,
never a target.

### Slot count must cover the axes
`evolve sample` assigns a slot's axis as `sorted(axes)[slot_index % len(axes)]`, and the slot index
restarts at zero every round. With seven axes, **a round with fewer than seven slots never touches
the tail of that sorted list at all** — not "less often", never. Sorted order here is:

    arch, batcher, head, objective, optim, stream, target

So `evolve sample --k 4` works arch/batcher/head/objective forever and leaves optim, stream and
target untouched. Use `--k 7` (or a multiple) when you want the whole surface worked, and if you
deliberately run a narrower round, say in the round report which axes were not offered a slot —
silent coverage gaps read as "the search tried everything and nothing helped".

### Why two registry impls are in `inventor_files`
Ten arch impls import the path-state trunk and eight head impls import the forward-model
consistency head. A jail only copies the parent's own impl for the axis being mutated plus the
per-axis baselines, so an inventor mutating a descendant would otherwise be handed source with an
import it cannot resolve or read. Those two modules are therefore granted to every jail. If a new
shared base module appears, add it here in the same commit.

### Reading a score
`combined_score` is `comp_ca_margin` — the differential minus the best analytic non-tracker measured
on the identical slice. Zero means the candidate did no better than a depth-zero positional lookup.
`public.analytic_band` prints every arm's own value and `public.best_analytic_arm` names the binding
one; if that band shifts between runs, the slice or the mint changed and nothing is comparable
across the change. `public.comp_ca` is the raw differential before the band is removed.
