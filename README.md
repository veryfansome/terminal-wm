# terminal-wm

A shell **world model** over real Linux Docker filesystems. Give it a history of shell commands and what they printed, and it predicts what the next command will print — on machines it has never seen. The model is the product. Everything else here exists to improve it, and to check honestly whether it improved.

## Capability packs

A **capability pack** has two parts: a set of real command trajectories that pose one specific question, and the instrument that grades the model's answer.

Two packs exist today.

**cups** asks about compositional depth. A trajectory shows several files at several locations, quietly moves them around, then reads one location back. To answer, the model has to track where each file went, hop by hop. This is the pack the search currently selects on.

**cdh** asks about command-history routing: when the model answers a read, does it draw on the navigation that actually happened? The instrument is built and runs alongside scoring whenever the cd-history pack is mounted, and it never feeds selection. No scored run has carried it yet.

Packs accumulate instead of replacing one another. The goal is a single model that holds several skills at once — trained on a mixture of packs and measured on all of them — rather than one model per skill. A pack that is only reported still earns its place: it is how you catch a mechanism that wins on one skill while quietly wrecking another. `evolve/blend_root.py` builds a training set from several packs at chosen ratios, and `cloud/build_context.py` assembles it at load time. The runs so far train on cups alone.

## What cups scores

The model gets a point for naming the file the real chain of moves put at a location. It loses that point if it names the same file again when the moves are rearranged so a different file ends up there. Only a model that actually follows the moves keeps the point.

This makes two cheap strategies worthless. A model that keys on *which location* was asked about answers identically in both cases, so it nets exactly zero — by construction, not on average. A model that keys on *where in the move order* a file sits also cancels, but only on average, and the scored set is one fixed sample. So a positive score is not by itself proof that anything was tracked. That is why the range such non-tracking strategies reach is measured on every run and printed beside the score. It is never subtracted from it, and a score is read against it.

There is also an absolute measure: how often the model picks correctly, against a fixed analytic ceiling. That is the honest yardstick, and it is deliberately not what the search optimizes. The ceiling never enters the scored arithmetic, so the search cannot learn to climb it.

## How it is searched

The model is assembled from seven swappable parts: objective, architecture, optimizer, target transform, batch composition, token stream layout, and readout head. Each part has a registry of implementations, and a candidate is one choice per part. LLM agents write new implementations, each sealed in a jail where the brief it is given is everything it knows. The [`evolve` plugin](https://github.com/veryfansome/claudemods) owns every decision that affects selection.

The search breeds a varied population rather than chasing a single leader. Parents are drawn by fitness and novelty, with a penalty for having many offspring, so it spreads across lineages instead of deepening one. Every scored candidate stays in the archive, and a candidate that loses counts as much as one that wins. Validity is checked per candidate when it is scored; anything later found invalid is retracted by appending, never by deleting. Each round produces N candidates, each measured against its own parents.

## Where it stands

Five scored rounds on top of the seed population: **108 live candidates**, 10 retracted, against a noise floor of **0.0227**. (A sixth round was run and culled — its candidates were unmeasurable rather than unsuccessful.)

| | score, pooled over 6 seeds | margin over the better parent |
|---|---|---|
| `r5-19-dualrole-address-mention` | +0.7472 | +0.7097 |
| `r6-28-role-unbind-provenance-chase` | +0.6030 | +0.5880 |

Each score pools two independent three-seed draws, seeds 0-2 and 3-5. Parents were measured on seeds 0-2 only, so the margin is not seed-matched. A candidate bred from two parents is held to the better of the two.

The two **differ on five of the seven parts** — architecture, head, objective, optimizer and stream — sharing only the batcher and the identity target. Both change their answer when the moves are rearranged instead of repeating the native one, which is what separates tracking from keying on something that never moved. On the confirmation draw both clear the fixed ceiling on every seed of the main slice; on a deeper slice `r6-28` clears in all three seeds and `r5-19` in two of three. One outlier would be a curiosity. Two largely independent routes to the same ability is a result.

Two lessons the rounds paid for, now built into the process:

- **One measurement is not a result.** Six candidates beat their parent by one to two noise floors on a single run. On a second, independent three-seed run, every one of the six fell back to within noise. Only the two clearing by more than twenty-five floors held. Anything that clears now gets a second run before it is reported.
- **An exploit will find the metric before a mechanism does.** An early candidate scored +0.9026 by working out the chain of moves in Python and writing the answer straight into the token being scored. It was retracted and the route closed off: the instrument now rebuilds tokens through a function that can only see one command at a time, checks that no prediction depends on the future, and requires the scored token to match in both arms to within 1e-4.

## Layout

```
realenv/seq_worldmodel.py   the world model and the retrieval primitives
evolve/chunks/<axis>/       the swappable parts — one file per implementation, append-only
evolve/cups_ca.py           the scored objective
evolve/cups_probe.py        the cups instrument (measurement + rearranged-moves probe)
evolve/cdh_probe.py         the cdh instrument — runs when its pack is mounted, never scored
evolve/blend_root.py        build a training set from several packs at chosen ratios
eval/                       the fitness oracle and the checks that run before any GPU time
cloud/lane.sh               run a campaign: provision, measure, bring results home, stop the box
cloud/build_context.py      the shared setup each campaign derives once
research/                   why the objective is shaped the way it is
```

`CLAUDE.md` is the working context; `evolve/EVOLVE.md` is the round manual.
