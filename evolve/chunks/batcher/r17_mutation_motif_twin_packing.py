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
    """Canonicalize a path token against the tracked cwd; None if empty."""
    if not p:
        return None
    if p.startswith("~"):
        p = "/root" + p[1:]
    if not p.startswith("/"):
        p = posixpath.join(cwd, p)
    return posixpath.normpath(p)


def _motif_keys(cmds):
    """Return the set of canonical mutated paths in `cmds` that a LATER command re-reads or
    re-lists within the same trajectory."""
    cwd = "/"
    muts = []
    keys = set()
    for t, cmd in enumerate(cmds):
        try:
            toks = shlex.split(cmd)
        except Exception:
            toks = cmd.split()
        if not toks:
            continue
        verb = toks[0]

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
                    keys.add(mp)
    return keys


def make_batcher(fit, bs, seed, hard_frac_max=0.5, ramp_frac=0.4, clique_k=4,
                 size_cap=24, n_block_images=1):
    n = len(fit)
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed ^ 0x5CA1AB1E)

    img_groups = collections.defaultdict(list)
    for i, s in enumerate(fit):
        img_groups[s.get("image", "?")].append(i)
    img_of = [s.get("image", "?") for s in fit]
    img_pools = [torch.tensor(img_groups[k], dtype=torch.long) for k in sorted(img_groups)]
    img_sizes = torch.tensor([float(p.numel()) for p in img_pools]) if img_pools else torch.zeros(0)

    clique = collections.defaultdict(set)
    motif_seqs = set()
    for i, s in enumerate(fit):
        cmds = s.get("cmds") or []
        mk = _motif_keys(cmds)
        if not mk:
            continue
        motif_seqs.add(i)
        for k in mk:
            clique[k].add(i)
            clique["base:" + posixpath.basename(k)].add(i)

    motif_seqs = sorted(motif_seqs)
    packable = [(k, sorted(v)) for k, v in clique.items() if len(v) >= 2]
    packable.sort(key=lambda kv: kv[0])
    have_motifs = len(motif_seqs) > 0
    have_cliques = len(packable) > 0
    if have_cliques:
        cl_lists = [v for _, v in packable]
        cl_w = torch.tensor([min(float(len(v)), float(size_cap)) for v in cl_lists])
    motif_pool = torch.tensor(motif_seqs, dtype=torch.long) if have_motifs else None

    multi_img = len(img_pools) >= 2
    k_imgs = max(1, min(int(n_block_images), len(img_pools))) if multi_img else 0

    if hard_frac_max <= 0.0 or (not have_motifs and not multi_img) or n < 2:
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    def _draw_clique(remaining):
        """Return up to min(clique_k, remaining) distinct sequence indices from one size-weighted
        clique, distinct images first."""
        ci = int(torch.multinomial(cl_w, 1, generator=g).item())
        members = cl_lists[ci]
        order = members[:]
        rng.shuffle(order)
        take = min(clique_k, remaining, len(order))
        picked, seen_img = [], set()
        for idx in order:
            im = img_of[idx]
            if im not in seen_img:
                picked.append(idx); seen_img.add(im)
                if len(picked) >= take:
                    break
        if len(picked) < take:
            for idx in order:
                if len(picked) >= take:
                    break
                picked.append(idx)
        return picked

    def _sysblock_hard(k):
        """Return k indices drawn from the union of a few size-weighted image pools."""
        chosen = torch.multinomial(img_sizes, k_imgs, replacement=False, generator=g)
        pool = torch.cat([img_pools[int(c)] for c in chosen])
        return pool[torch.randint(0, pool.numel(), (k,), generator=g)].tolist()

    def next_batch(step, total_steps):
        ramp_steps = max(1, int(ramp_frac * max(1, total_steps)))
        frac = hard_frac_max * min(1.0, step / ramp_steps)
        n_hard = max(0, min(bs, int(round(bs * frac))))

        idx = torch.randint(0, n, (bs - n_hard,), generator=g).tolist()

        hard = []
        if n_hard > 0:
            if have_cliques:
                while len(hard) < n_hard:
                    hard.extend(_draw_clique(n_hard - len(hard)))
                hard = hard[:n_hard]
            elif have_motifs:
                sel = torch.randint(0, motif_pool.numel(), (n_hard,), generator=g)
                hard = motif_pool[sel].tolist()
            else:
                hard = _sysblock_hard(n_hard)
        idx.extend(hard)

        idx = idx[:bs]
        if len(idx) < bs:
            idx.extend(torch.randint(0, n, (bs - len(idx),), generator=g).tolist())
        perm = torch.randperm(bs, generator=g).tolist()
        return [idx[p] for p in perm]

    return next_batch
