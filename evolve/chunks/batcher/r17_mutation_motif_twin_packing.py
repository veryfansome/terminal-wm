"""batcher chunk: mutation-motif TWIN packing (pre/post counterfactual co-packing).

THE v3 LEVER this targets. The dynamical world's high-value cells are reads of a path that an
earlier rm/mv/ln/echo>/cp/mkdir/touch ALTERED, and the eval FORCES that path's own PRE-mutation
observation into the candidate set (the counterfactual twin at meta.pre_obs_step). So the single
decision the metric actually scores on a mutated cell is: rank the predicted POST-mutation content
of path P ABOVE the PRE-mutation content of the SAME P. Every registered batcher densifies a
DIFFERENT negative geometry — same-system foils (r6 sysblock), same-verb-system-subtree cliques
(r7), CROSS-system same-path variants (r16 cohorts/lattice, weakverb), loop-closure recurrence
(r8), or v2 minimal pairs (r12). NONE packs the WITHIN-trajectory mutation twin, because all were
designed on the READ-ONLY v2/v1 world where no path mutates. This batcher supplies exactly that
foil.

MECHANISM. Parse each train sequence's command strings ONCE (no meta needed — inferred from cmds
with cwd tracked through `cd`). Detect MUTATION events (rm/mv/cp/ln/mkdir/touch and any `>`/`>>`
redirect) and their affected path, and READ events (cat/head/tail/stat/ls/find/…). A sequence is
"motif-bearing" if it contains a read that references a path mutated EARLIER in the same trajectory
(a file re-read of the mutated path, or an ls/find of the directory whose contents the mutation
changed). Record the canonical mutated path(s) involved as the sequence's MOTIF KEYS.

Two facts make this pay:
  (1) WITHIN a single motif-bearing sequence, the pre-mutation AND post-mutation reads of P are
      BOTH command positions, so both land in the batch as mutually-negative target rows — the
      pre-read IS the eval's forced counterfactual twin. Merely OVERSAMPLING motif-bearing
      sequences therefore raises the in-batch density of exactly the pre/post pair the metric
      ranks.
  (2) CO-PACKING sequences that mutate+read the SAME canonical path (a path-keyed CLIQUE, drawn
      across DIFFERENT images / DIFFERENT mutation types) puts MANY pre/post/cross-mutation
      variants of one path in one batch — a bulk counterfactual collision, the same
      block->clique concentration that scored above uniform sampling for sysblock, but keyed on
      MUTATION STATE instead of system identity.

The hard fraction ramps 0 -> hard_frac_max over the first ramp_frac of training (the same warm-up
the sysblock curriculum uses); its steady state is the motif geometry. Clique members are drawn
distinct-image-first so a co-packed clique is pre-vs-post / system-variant TRUE negatives, not a
duplicated post-state (a false negative). This is complementary to the antiretrieval-ring
objective: a pre/post twin is the canonical close-but-distinct pair that lands in the ring's
pass-band and receives the repulsion hinge, while an accidental same-state duplicate is gated out by its
dupmask — so packing and objective reinforce.

HOW IT DIFFERS FROM THE SYSBLOCK BATCHER (measured +0.3922 on the read-only v2 world it was
evolved on, NOT built for mutation tracking):
sysblock densifies same-SYSTEM negatives, which on v3 does nothing for the mutated
cells whose foil is the counterfactual twin, not another system. This batcher spends the hard
budget on the mutated cells' actual foil. Retry-failed-in-changed-context: a mutation-keyed
composition was pointless on the static world and is first-class now that the world mutates.

Anti-collapse: unchanged from the objective — this module never touches predictions or targets,
only which sequence indices are drawn; a constant prediction still yields uniform softmax rows
(loss log(n)) under the contrastive objectives regardless of composition.

Degenerate-data safety: no mutation motifs at all (e.g. a v1/v2 read-only root) -> falls back to
the proven sysblock image-blocked hard/easy curriculum; < 2 images too -> exact uniform. All
randomness flows through one private seed-derived torch.Generator plus a private random.Random
(deterministic per seed; global RNG untouched; fit never mutated). Every returned batch is exactly
bs indices in [0, len(fit)).
"""

import collections
import posixpath
import random
import shlex

import torch

NAME = "r17_mutation_motif_twin_packing"
DESCRIPTION = (
    "Anneal batch composition from uniform to path-keyed MUTATION-MOTIF cliques: co-pack "
    "sequences that mutate-then-reread the SAME path (pre/post counterfactual twins, distinct-"
    "image-first) so the in-batch negatives ARE the eval's forced pre-mutation twin on the "
    "mutated cells; sysblock fallback when no mutation motifs exist."
)

_MUT_VERBS = frozenset({"rm", "mv", "cp", "ln", "mkdir", "touch", "rmdir", "install", "unlink"})
_READ_FILE = frozenset({"cat", "head", "tail", "stat", "wc", "od", "xxd", "md5sum", "sha256sum",
                        "readlink", "file", "less", "more", "grep", "sort", "nl"})
_READ_DIR = frozenset({"ls", "find", "du", "tree"})


def _resolve(p, cwd):
    """Canonicalize a path token against the tracked cwd (best-effort; no filesystem access)."""
    if not p:
        return None
    if p.startswith("~"):
        p = "/root" + p[1:]
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    return posixpath.normpath(p)


def _motif_keys(cmds):
    """Return the set of canonical MUTATED paths in `cmds` that a LATER command re-reads/re-lists
    within the same trajectory (the within-sequence mutation motif). Pure string parsing over the
    command list; cwd tracked through `cd`. Robust to odd quoting (falls back to whitespace split)."""
    cwd = "/"
    muts = []          # (step_idx, canonical_mutated_path)
    keys = set()
    for t, cmd in enumerate(cmds):
        try:
            toks = shlex.split(cmd)
        except Exception:
            toks = cmd.split()
        if not toks:
            continue
        verb = toks[0]

        # redirect writes (echo hi > P, cmd >> P) mutate the redirect target regardless of verb
        for j, tk in enumerate(toks):
            if tk in (">", ">>") and j + 1 < len(toks):
                rp = _resolve(toks[j + 1], cwd)
                if rp:
                    muts.append((t, rp))
            elif tk.startswith(">") and tk not in (">", ">>") and len(tk) > 1:
                rp = _resolve(tk.lstrip(">"), cwd)
                if rp:
                    muts.append((t, rp))

        if verb == "cd":
            args = [a for a in toks[1:] if not a.startswith("-")]
            cwd = (_resolve(args[0], cwd) if args else "/root") or cwd
            continue

        args = [a for a in toks[1:] if not a.startswith("-") and a not in (">", ">>")]

        if verb in _MUT_VERBS:
            # record every operand path as a changed location (src+dst for mv/cp/ln, targets else)
            for a in args:
                rp = _resolve(a, cwd)
                if rp:
                    muts.append((t, rp))

        if verb in _READ_FILE:
            for a in args:
                rp = _resolve(a, cwd)
                if rp is None:
                    continue
                for (tp, mp) in muts:
                    if tp < t and mp == rp:
                        keys.add(mp)
        elif verb in _READ_DIR:
            targ = _resolve(args[0], cwd) if args else cwd
            if targ is None:
                continue
            tpref = targ.rstrip("/") + "/"
            for (tp, mp) in muts:
                if tp < t and (mp == targ or posixpath.dirname(mp) == targ or mp.startswith(tpref)):
                    keys.add(mp)   # a mutation whose path lives in the listed directory
    return keys


def make_batcher(fit, bs, seed, hard_frac_max=0.5, ramp_frac=0.4, clique_k=4,
                 size_cap=24, n_block_images=1):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed ^ 0x5CA1AB1E)   # private, for distinct-image member ordering

    # ---- image groups (for the sysblock fallback + distinct-image clique draws) ----
    img_groups = collections.defaultdict(list)
    for i, s in enumerate(fit):
        img_groups[s.get("image", "?")].append(i)
    img_of = [s.get("image", "?") for s in fit]
    img_pools = [torch.tensor(img_groups[k], dtype=torch.long) for k in sorted(img_groups)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools]) if img_pools else torch.zeros(0)

    # ---- parse mutation motifs once (read-only) ----
    clique = collections.defaultdict(set)   # canonical path (+ basename alias) -> set(seq idx)
    motif_seqs = set()
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        mk = _motif_keys(cmds)
        if not mk:
            continue
        motif_seqs.add(i)
        for k in mk:
            clique[k].add(i)
            clique["base:" + posixpath.basename(k)].add(i)   # coarse alias to grow small cliques

    motif_seqs = sorted(motif_seqs)
    # packable cliques = keys with >= 2 distinct sequences; weight by capped size (bound giants)
    packable = [(k, sorted(v)) for k, v in clique.items() if len(v) >= 2]
    packable.sort(key=lambda kv: kv[0])   # deterministic order
    have_motifs = len(motif_seqs) > 0
    have_cliques = len(packable) > 0
    if have_cliques:
        cl_lists = [v for _, v in packable]
        cl_w = torch.tensor([min(float(len(v)), float(size_cap)) for v in cl_lists])
    motif_pool = torch.tensor(motif_seqs, dtype=torch.long) if have_motifs else None

    multi_img = len(img_pools) >= 2
    k_imgs = max(1, min(int(n_block_images), len(img_pools))) if multi_img else 0

    # Full degeneracy: nothing to bias on -> exact uniform (bit-identical to baseline structure).
    if hard_frac_max <= 0.0 or (not have_motifs and not multi_img) or n < 2:
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    def _draw_clique(remaining):
        """Draw up to min(clique_k, remaining) DISTINCT sequences from one size-weighted clique,
        preferring distinct images so co-packed members are pre/post / cross-system TRUE negatives
        rather than a duplicated post-mutation state (a false negative)."""
        ci = int(torch.multinomial(cl_w, 1, generator=g).item())
        members = cl_lists[ci]
        order = members[:]
        rng.shuffle(order)
        take = min(clique_k, remaining, len(order))
        picked, seen_img = [], set()
        for idx in order:                       # first pass: one per distinct image
            im = img_of[idx]
            if im not in seen_img:
                picked.append(idx); seen_img.add(im)
                if len(picked) >= take:
                    break
        if len(picked) < take:                  # second pass: fill remaining slots
            for idx in order:
                if len(picked) >= take:
                    break
                picked.append(idx)
        return picked

    def _sysblock_hard(k):
        """r6 sysblock fallback: k indices drawn from the union of a few size-weighted images."""
        chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
        pool = torch.cat([img_pools[int(c)] for c in chosen])
        return pool[torch.randint(0, pool.numel(), (k,), generator=g)].tolist()

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        idx = torch.randint(0, n, (bs - n_hard,), generator=g).tolist()   # easy remainder: uniform

        hard = []
        if n_hard > 0:
            if have_cliques:
                while len(hard) < n_hard:
                    hard.extend(_draw_clique(n_hard - len(hard)))
                hard = hard[:n_hard]
            elif have_motifs:
                # no >=2 clique but motifs exist -> oversample motif-bearing seqs (twins live
                # WITHIN each such sequence: its own pre-read is the forced counterfactual foil)
                sel = torch.randint(0, motif_pool.numel(), (n_hard,), generator=g)
                hard = motif_pool[sel].tolist()
            else:
                hard = _sysblock_hard(n_hard)   # no mutation signal (v1/v2 root) -> proven fallback
        idx.extend(hard)

        idx = idx[:bs]
        if len(idx) < bs:                       # safety pad (never expected)
            idx.extend(torch.randint(0, n, (bs - len(idx),), generator=g).tolist())
        perm = torch.randperm(bs, generator=g).tolist()
        return [idx[p] for p in perm]

    return next_batch