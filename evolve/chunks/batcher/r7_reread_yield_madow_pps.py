import re

import torch

D_FEAT = 768

NAME = "r7_reread_yield_madow_pps"
DESCRIPTION = (
    "Probability-proportional-to-size sampling with an exact balanced draw, imported from survey "
    "sampling. Two changes to the parent, serving one idea: give every sequence a SIZE MEASURE "
    "counting the rows that carry the transport signal, then draw each batch as an exact design "
    "rather than a lottery. The size measure is computed per sequence from its own standardized "
    "observation embeddings and its own command strings: a row is a RE-READ when its observation "
    "exactly duplicates an earlier row of the same sequence issued under a different command and "
    "its duplicate set is small — the post-chain read of a location that a move filled — and a "
    "re-read is DECOY-BEARING when at least two earlier rows of the sequence carry different "
    "content and one of them reads a path sharing the re-read's canonical basename, so the name "
    "has a former occupant standing in the same trajectory. Duplicate detection is exact rather "
    "than approximate: identical rendered observations give identical embeddings, so their "
    "distance is zero under any projection, and a variance-preserving Gaussian projection reduces "
    "the per-sequence distance matrices to a fraction of their cost while leaving that zero "
    "untouched. Inclusion probabilities are proportional to size raised to an exponent that ramps "
    "from zero, scaled so they sum to exactly the batch size and capped at one by the standard "
    "redistribute-the-excess fixpoint. The batch is then realized by Madow systematic sampling: "
    "the frame is freshly permuted, the inclusion probabilities are accumulated, and one uniform "
    "start selects the units whose cumulative intervals contain the start plus each integer. That "
    "draw is without replacement, spends no slot on a repeated sequence, holds the batch size at "
    "exactly the requested count with zero variance, and realizes each unit's inclusion "
    "probability exactly rather than in expectation over multinomial draws. The parent's annealed "
    "same-image concentration is kept as the first stage — a few system images are chosen "
    "proportional to their aggregate size and the hard share is drawn systematically inside them "
    "— and the remaining slots are drawn systematically from everything not already selected. "
    "Falls back to uniform sampling when the batch is not smaller than the split or no size "
    "measure can be built."
)

_READ_VERBS = ("cat", "head", "tail")
_BREAK_TOKENS = ("|", "&&", "||", ";", "<", "2>", "2>>")
_NUM_SUFFIX = re.compile(r"\.\d+$")
_DIGIT_RUN = re.compile(r"\d+")
_SEQ_CHUNK = 64
_CLIP_ITERS = 24
_EPS = 1e-12


def _focus_path(cmd):
    if not isinstance(cmd, str):
        return ""
    toks = cmd.split()
    if len(toks) < 2:
        return ""
    first = None
    target = None
    want = False
    for t in toks[1:]:
        if want:
            target = t
            want = False
            continue
        if t in _BREAK_TOKENS:
            continue
        if t == ">" or t == ">>":
            want = True
            continue
        if t.startswith(">>"):
            target = t[2:] or target
            continue
        if t.startswith(">"):
            target = t[1:] or target
            continue
        if t.startswith("-") and len(t) > 1:
            continue
        if first is None:
            first = t
    if target is not None:
        return target
    return first or ""


def _canon_base(path):
    if not path:
        return ""
    p = path[:-1] if len(path) > 1 and path.endswith("/") else path
    b = p.rsplit("/", 1)[-1]
    while True:
        q = _NUM_SUFFIX.sub("", b)
        if q == b:
            break
        b = q
    return _DIGIT_RUN.sub("#", b)


def _projection(dim, proj_dim, gen):
    p = max(8, min(int(proj_dim), int(dim)))
    return torch.randn(int(dim), p, generator=gen) / float(p) ** 0.5


def _sq(a, b):
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1, keepdim=True)
    return (a2 + b2.t() - 2.0 * (a @ b.t())).clamp_min(0.0) / float(D_FEAT)


def _mean_offdiag(fit, rmat, gen, stat_n):
    pool = []
    for i, s in enumerate(fit):
        zo = s.get("z_obs")
        if zo is None or int(zo.shape[0]) == 0:
            continue
        pool.append((i, int(zo.shape[0])))
    if len(pool) < 4:
        return None
    m = max(16, min(int(stat_n), 4 * len(pool)))
    picks = torch.randint(0, len(pool), (m,), generator=gen)
    vecs = []
    for k in range(int(picks.numel())):
        i, n_steps = pool[int(picks[k])]
        t = int(torch.randint(0, n_steps, (1,), generator=gen).item())
        vecs.append(fit[i]["z_obs"][t].detach().float())
    if len(vecs) < 16:
        return None
    y = torch.stack(vecs) @ rmat
    dm = _sq(y, y)
    k = int(y.shape[0])
    total = float(dm.sum().item()) - float(dm.diagonal().sum().item())
    val = total / max(1.0, float(k * (k - 1)))
    if not (val > 0.0):
        return None
    return val


def _yield_of(y, cmds, eq_tol, grp_max):
    n = int(y.shape[0])
    if n < 3:
        return 0.0, 0.0
    dm = _sq(y, y)
    eq = dm <= eq_tol
    cls_size = eq.sum(dim=1)
    ids = {}
    cid = []
    for c in cmds[:n]:
        key = c if isinstance(c, str) else str(c)
        v = ids.get(key)
        if v is None:
            v = len(ids)
            ids[key] = v
        cid.append(v)
    while len(cid) < n:
        cid.append(-1)
    cidt = torch.tensor(cid, dtype=torch.long)
    order = torch.arange(n)
    earlier = order.view(1, n) < order.view(n, 1)
    diff_cmd = cidt.view(1, n) != cidt.view(n, 1)
    reread = (eq & earlier & diff_cmd).any(dim=1) & (cls_size <= int(grp_max))
    n_reread = float(reread.sum().item())
    if n_reread <= 0.0:
        return 0.0, 0.0
    cand = earlier & (~eq)
    cand_n = cand.sum(dim=1)
    bases = [_canon_base(_focus_path(cmds[t])) if t < len(cmds) else "" for t in range(n)]
    n_decoy = 0.0
    for t in range(n):
        if not bool(reread[t].item()) or int(cand_n[t].item()) < 2:
            continue
        bt = bases[t]
        if not bt:
            continue
        for j in range(t):
            if bases[j] == bt and bool(cand[t, j].item()):
                n_decoy += 1.0
                break
    return n_reread, n_decoy


def _size_measure(fit, gen, proj_dim, stat_n, eq_frac, grp_max, w_reread, w_decoy):
    dim = 0
    for s in fit:
        z = s.get("z_obs")
        if z is not None and int(z.shape[0]) > 0:
            dim = int(z.shape[1])
            break
    if dim <= 0:
        return None
    rmat = _projection(dim, proj_dim, gen)
    scale = _mean_offdiag(fit, rmat, gen, stat_n)
    if scale is None:
        return None
    eq_tol = max(_EPS, float(eq_frac) * scale)

    sizes = torch.ones(len(fit), dtype=torch.float)
    for c0 in range(0, len(fit), _SEQ_CHUNK):
        chunk = fit[c0:c0 + _SEQ_CHUNK]
        lens = []
        blocks = []
        for s in chunk:
            zo = s.get("z_obs")
            k = 0 if zo is None else int(zo.shape[0])
            lens.append(k)
            if k > 0:
                blocks.append(zo.detach().float())
        if not blocks:
            continue
        proj = torch.cat(blocks) @ rmat
        at = 0
        for j, k in enumerate(lens):
            if k <= 0:
                continue
            y = proj[at:at + k]
            at += k
            cmds = chunk[j].get("cmds") or []
            r, d = _yield_of(y, cmds, eq_tol, grp_max)
            sizes[c0 + j] = 1.0 + float(w_reread) * r + float(w_decoy) * d
    if not bool(torch.isfinite(sizes).all()):
        return None
    return sizes.clamp_min(1e-3)


def _madow(weights, k, gen):
    m = int(weights.numel())
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    if k >= m:
        return torch.arange(m, dtype=torch.long)
    p = weights.detach().float().clamp_min(_EPS)
    total = float(p.sum().item())
    if not (total > 0.0):
        return torch.randperm(m, generator=gen)[:k]
    p = p * (float(k) / total)
    for _ in range(_CLIP_ITERS):
        over = p > 1.0
        if not bool(over.any()):
            break
        free = ~over
        excess = float((p[over] - 1.0).sum().item())
        p = torch.where(over, torch.ones_like(p), p)
        room = float(p[free].sum().item())
        if room <= _EPS or excess <= _EPS:
            break
        p = torch.where(free, p * (1.0 + excess / room), p)
    p = p.clamp(_EPS, 1.0)
    perm = torch.randperm(m, generator=gen)
    c = torch.cumsum(p[perm], dim=0)
    span = float(c[-1].item())
    if span <= float(k) - 1.0:
        return perm[:k]
    u = float(torch.rand(1, generator=gen).item()) * max(_EPS, span - float(k) + 1.0)
    targets = u + torch.arange(k, dtype=c.dtype)
    pos = torch.searchsorted(c.contiguous(), targets.contiguous())
    pos = pos.clamp(max=m - 1)
    return perm[pos]


def make_batcher(fit, bs, seed, hard_frac_max=0.75, ramp_frac=0.3, n_block_images=1,
                 gamma_max=1.5, w_reread=1.0, w_decoy=1.5, grp_max=4, eq_frac=2e-3,
                 proj_dim=96, stat_n=1536):
    n = len(fit)
    g = torch.Generator().manual_seed(int(seed))

    def uniform_only():
        def next_batch(step, total_steps):
            return torch.randint(0, n, (bs,), generator=g).tolist()
        return next_batch

    if n < 2 or bs <= 0 or bs >= n:
        return uniform_only()

    g_setup = torch.Generator().manual_seed((int(seed) ^ 0x27D4EB2F) & 0x7FFFFFFF)
    sizes = _size_measure(fit, g_setup, proj_dim, stat_n, eq_frac, grp_max, w_reread, w_decoy)
    if sizes is None:
        return uniform_only()
    log_size = sizes.log()

    by_image = {}
    for i, s in enumerate(fit):
        by_image.setdefault(str(s.get("image", "?")), []).append(i)
    img_pools = [torch.tensor(by_image[k], dtype=torch.long) for k in sorted(by_image)]
    img_mass = torch.tensor([float(sizes[p].sum().item()) for p in img_pools],
                            dtype=torch.float).clamp_min(1e-6)
    k_imgs = max(1, min(int(n_block_images), len(img_pools)))
    use_images = len(img_pools) >= 2 and float(hard_frac_max) > 0.0

    def next_batch(step, total_steps):
        total = max(1, int(total_steps))
        ramp_steps = max(1, int(float(ramp_frac) * total))
        ramp = min(1.0, float(step) / float(ramp_steps))
        n_hard = max(0, min(bs, int(round(bs * float(hard_frac_max) * ramp))))
        w = torch.exp((float(gamma_max) * ramp) * log_size)

        parts = []
        got = 0
        taken = torch.zeros(n, dtype=torch.bool)
        if use_images and n_hard >= 2:
            chosen = torch.multinomial(img_mass, k_imgs, replacement=False, generator=g)
            frame = torch.cat([img_pools[int(c)] for c in chosen.tolist()])
            if int(frame.numel()) > n_hard:
                picks = frame[_madow(w[frame], n_hard, g)]
                parts.append(picks)
                taken[picks] = True
                got = int(picks.numel())

        rest = bs - got
        if rest > 0:
            free = torch.nonzero(~taken, as_tuple=False).squeeze(1)
            if int(free.numel()) > rest:
                parts.append(free[_madow(w[free], rest, g)])
            else:
                parts.append(torch.randint(0, n, (rest,), generator=g))

        batch = torch.cat(parts)
        if int(batch.numel()) > bs:
            batch = batch[:bs]
        elif int(batch.numel()) < bs:
            batch = torch.cat([batch,
                               torch.randint(0, n, (bs - int(batch.numel()),), generator=g)])
        return batch[torch.randperm(bs, generator=g)].tolist()

    return next_batch
