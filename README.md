# terminal-wm

A shell **world model** over real Linux Docker filesystems, evolved against a **compositional
depth** objective: given a history of shell commands, can the model say *which content* now sits at
a location after a silent chain of moves — on a system it has never seen?

## The question

A trajectory exposes several files at several locations, then quietly moves them around, then reads
one location back. Predicting the read correctly requires carrying each item's identity across
several hops. Knowing that *something* moved is not enough, and neither is keying on the name being
asked about — the objective is built so that both of those strategies score exactly zero.

## How it is searched

Evolutionary search over a registry of swappable mechanisms — objective, architecture, optimizer,
target transform, batch composition, token stream layout, readout head — driven by the
[`evolve` plugin](https://github.com/veryfansome/claudemods). LLM inventor agents are the mutation
operators; a deterministic engine owns everything selection-critical.

The search **accumulates and never crowns**. There is no champion, no promotion, no adoption step.
Parents are sampled by fitness and novelty with an offspring penalty; every scored candidate stays
in the archive, negatives with the same weight as wins; validity is enforced per candidate at score
time. Shipping something is an engineering decision taken outside the search.

## Status

Set up and verified; **not yet searched**. The contract holds (`evolve doctor` passes), the
starting population is written, and the pre-eval gates run. Before a first round:

1. **Prepare the pack lane** on a GPU box — `cloud/pack_lane.sh prepare` pulls the pinned encoder
   and the raw pack root, verifies the encoder's tree hash against the frozen pin, and does the
   one-time encode. The encoded root does not exist yet anywhere; nothing can be scored until it
   does.
2. **Measure the noise floor honestly.** Not with repeated identical runs — with fixed seeds the
   eval is essentially deterministic, so that measures reproducibility rather than uncertainty and
   will make noise look like signal. Derive it from the spread of three-seed means across disjoint
   seed triples. See `research/compositional-selection-design.md` §5.
3. **Score the starting population** so the archive has a diverse set of parents to breed from.

## Layout

```
realenv/seq_worldmodel.py   the world model and the retrieval primitives
evolve/chunks/<axis>/       the evolvable surface — one file per mechanism, append-only
evolve/cups_ca.py           the objective
evolve/cups_probe.py        the pack instrument (measurement + role-swap probe)
eval/                       the fitness oracle and the pre-eval gates
cloud/pack_lane.sh          the GPU lane
research/                   why the objective is shaped the way it is
```

`CLAUDE.md` is the working context; `evolve/EVOLVE.md` is the round manual.

## Provenance

Re-founded from an earlier project that built the world model, the chunk registry and the
capability-pack instruments carried here. That project's custom evolutionary machinery — a champion
pointer, a promotion predicate, drift budgets and a debt ledger — is deliberately **not** carried:
an anchor every candidate is told to beat becomes the population's objective and collapses the
diversity a search exists to maintain.
