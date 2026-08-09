"""R4: a SEQUENCE world model over real Docker-shell exploration trajectories.

Each trajectory is an exploration of one Linux image: identify the system (`uname`,
`cat` of a config file), then navigate/inspect (`cd`/`ls`/`cat`). We embed every command
and every resulting observation with a FROZEN encoder (ModernBERT), then train a small
CAUSAL transformer over the interleaved token stream

    cmd_0  obs_0  cmd_1  obs_1  ...  cmd_t  obs_t

whose hidden state AT each command position must predict that command's resulting
observation embedding z_obs[t] (in latent space, standardized) — the world-model bet:
preview a command's consequence BEFORE running it, using the whole exploration history
(you cannot read obs_t; it doesn't exist yet). Only the transformer is learned; perception
is frozen.

The fair test is generalization to UNSEEN SYSTEMS (held-out Docker *images* — NOT held-out
tools; inferring a never-seen tool is a read-the-man-page capability and an eventual goal).
We report on two splits:
  - dev    : seen images, unseen sequences (in-distribution reference);
  - heldout: unseen images (fedora/rocky/mariadb/httpd) — the milestone.

Against honest no-model baselines on the SAME next-observation retrieval yardstick:
  - predict-mean   (train-mean obs; chance under retrieval by construction);
  - copy-prev-obs  (z_obs[t] := z_obs[t-1]);
  - retrieve-by-cmd(nearest train observation whose COMMAND embedding matches — lexical
                    memory, no world model).
Metric: given a prediction, rank the true next observation against foils (random foils and
hard same-verb foils) — top-1 accuracy + MRR. A world model that has learned system dynamics
should beat copy and lexical retrieval on UNSEEN images, not just memorize.

This module is a LIBRARY: the model + the eval primitives (encode/standardize/split/collate/
retrieval) that the harness and the cups instruments call. Its standalone experiment CLI was
dropped in the re-founding; there is no __main__ entrypoint.
"""

import json
import pathlib
import random
from collections import defaultdict

import torch
import torch.nn as nn

D = 768
OBS_CAP = 1600  # chars of command output kept before encoding (median 37, p95 ~1k)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- rendering

def render_obs(step):
    """The OBSERVATION = what you see after running the command: resulting cwd + exit code
    + (truncated) output. cwd matters because `cd` has no stdout — its visible effect IS the
    new working directory."""
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd', '/')} exit={step.get('exit', 0)}\n{out}"


def render_cmd(step):
    return step["cmd"]


def verb_of(cmd):
    p = cmd.split()
    return p[0] if p else ""


# ---------------------------------------------------------------- encoding

@torch.no_grad()
def encode_split(path, model, tok, device, bs=96):
    """Per-sequence arrays z_obs[n,768], z_cmd[n,768] (frozen mean-pooled), + cmds/image.
    All texts across all sequences are encoded once, then regrouped."""
    seqs = [json.loads(l) for l in open(path)]
    obs_texts, cmd_texts, spans = [], [], []
    for sq in seqs:
        start = len(obs_texts)
        for s in sq["steps"]:
            obs_texts.append(render_obs(s))
            cmd_texts.append(render_cmd(s))
        spans.append((start, len(obs_texts)))

    def enc(texts, tag):
        # length-sorted batching: group similar-length texts so padding is minimal (median
        # obs ~37 chars, p95 ~1k). Compute in sorted order, scatter back to original order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = torch.zeros(len(texts), D)
        for i in range(0, len(order), bs):
            bidx = order[i:i + bs]
            e = tok([texts[j] for j in bidx], return_tensors="pt", padding=True,
                    truncation=True, max_length=256)
            e = {k: v.to(device) for k, v in e.items()}
            h = model(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1)
            pooled = ((h * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu()
            for k, j in enumerate(bidx):
                out[j] = pooled[k]
            if (i // bs) % 50 == 0:
                print(f"  enc {tag} {i}/{len(texts)}", flush=True)
        return out

    z_obs, z_cmd = enc(obs_texts, "obs"), enc(cmd_texts, "cmd")
    out = []
    for (a, b), sq in zip(spans, seqs):
        out.append({"z_obs": z_obs[a:b], "z_cmd": z_cmd[a:b],
                    "cmds": [s["cmd"] for s in sq["steps"]], "image": sq["image"],
                    # per-step success flag (exit 0 + non-empty output) — v2 class slicing
                    # (grep-miss exclusion); absent in v1 caches, consumers default all-True
                    "ok": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                           for s in sq["steps"]]})
    return out


def _v3_cache_guard(data_root):
    """Universal fail-closed staleness guard (dockerfs3 §13.2). Fires ONLY when a cache_meta.json
    sits beside the root — which is true for v3 derived roots and NOTHING else, so v1/v2 roots are
    byte-untouched. Self-contained (no evolve import; realenv stays evolve-free): requires
    cache_format==3 and that the root's current summary.json still hashes to the built_summary_sha
    recorded at encode time. A re-mint into an occupied path rewrites summary.json -> the emb-seq
    caches are stale -> this RAISES, so every caller (harness AND the direct realenv/sanity
    callers) is protected, making §13.2's 'impossible by construction' invariant actually hold."""
    import hashlib
    cm_path = pathlib.Path(data_root) / "cache_meta.json"
    if not cm_path.exists():
        return  # v1/v2 root (or a raw root) — no guard, unchanged behavior
    cm = json.loads(cm_path.read_text())
    if cm.get("cache_format") != 3:
        raise RuntimeError(f"stale/unknown cache_format in {cm_path}: {cm.get('cache_format')}")
    summ = pathlib.Path(data_root) / "summary.json"
    cur = hashlib.sha256(summ.read_bytes()).hexdigest() if summ.exists() else None
    if cur != cm.get("built_summary_sha"):
        raise RuntimeError(
            f"stale v3 cache at {data_root}: summary.json sha {cur} != built {cm.get('built_summary_sha')} "
            f"(re-mint into an occupied path; delete emb-seq-*.pt + cache_meta.json and re-encode)")


def cached_encode(data_root, split, model_name, device):
    _v3_cache_guard(data_root)
    cache = pathlib.Path(data_root) / f"emb-seq-{split}.pt"
    if cache.exists():
        print(f"  using cache {cache}", flush=True)
        return torch.load(cache, weights_only=False)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    seqs = encode_split(pathlib.Path(data_root) / f"{split}.jsonl", model, tok, device)
    torch.save(seqs, cache)
    print(f"  cached {len(seqs)} sequences -> {cache}", flush=True)
    return seqs


# ---------------------------------------------------------------- bag-of-tokens (gen twin)

def _iter_steps(data_root, split):
    for line in open(pathlib.Path(data_root) / f"{split}.jsonl"):
        for s in json.loads(line)["steps"]:
            yield s


# ---------------------------------------------------------------- data prep

def standardize_stats(seqs):
    allo = torch.cat([s["z_obs"] for s in seqs])
    allc = torch.cat([s["z_cmd"] for s in seqs])
    return (allo.mean(0, keepdim=True), allo.std(0, keepdim=True).clamp(min=1e-6),
            allc.mean(0, keepdim=True), allc.std(0, keepdim=True).clamp(min=1e-6))


def apply_stats(seqs, mo, so, mc, sc):
    for s in seqs:
        s["z_obs"] = (s["z_obs"] - mo) / so
        s["z_cmd"] = (s["z_cmd"] - mc) / sc
        if "z_obs_multi" in s:  # multi-vector stream segments live in the obs space
            s["z_obs_multi"] = (s["z_obs_multi"] - mo.unsqueeze(0)) / so.unsqueeze(0)


# ---------------------------------------------------------------- model

class SeqWorldModel(nn.Module):
    """Causal transformer over interleaved [cmd_0,obs_0,cmd_1,obs_1,...] frozen-embedding
    tokens. `aux='jepa'` -> head at cmd positions predicts z_obs[t] (latent). `aux='recon'`
    -> predicts the obs token-bag (surface reconstruction; the generative twin). Both expose
    the pre-head hidden state at cmd positions for a common downstream probe."""

    def __init__(self, aux="jepa", vsize=0, d=192, layers=4, heads=4, dropout=0.1,
                 no_history=False):
        super().__init__()
        self.aux = aux
        self.d = d
        self.no_history = no_history  # self-only attention => matched-capacity, history-free control
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)   # 0=cmd, 1=obs
        self.pos_emb = nn.Embedding(64, d)   # max 2*32 steps
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D if aux == "jepa" else vsize)

    def encode(self, tok_emb, types, key_pad):
        """tok_emb [B,L,D] frozen embeddings; types [B,L] in {0,1}; key_pad [B,L] True=pad.
        Returns hidden [B,L,d]. no_history=True masks all attention except self — same
        architecture/capacity as the full model but each token sees only itself, isolating the
        value of the exploration history from raw function-approximation capacity."""
        B, L, _ = tok_emb.shape
        pos = torch.arange(L, device=tok_emb.device)
        x = self.proj(tok_emb) + self.type_emb(types) + self.pos_emb(pos)[None]
        if self.no_history:
            mask = ~torch.eye(L, device=tok_emb.device, dtype=torch.bool)  # allow only the diagonal
        else:
            mask = torch.triu(torch.ones(L, L, device=tok_emb.device, dtype=torch.bool), 1)  # causal
        return self.tf(x, mask=mask, src_key_padding_mask=key_pad)

    def forward(self, tok_emb, types, key_pad):
        h = self.encode(tok_emb, types, key_pad)   # [B,L,d]
        return self.head(h), h                     # predictions at every position


def collate(batch, device):
    """batch: list of seq dicts. Interleave cmd/obs into token stream, pad, build masks and
    the target (standardized z_obs at each cmd position). Returns tensors on device."""
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)          # True = pad
    tgt = torch.zeros(B, maxn, D)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)     # valid cmd positions
    for bi, s in enumerate(batch):
        n = s["z_obs"].shape[0]
        for i in range(n):
            tok[bi, 2 * i] = s["z_cmd"][i]
            tok[bi, 2 * i + 1] = s["z_obs"][i]
            types[bi, 2 * i] = 0
            types[bi, 2 * i + 1] = 1
            key_pad[bi, 2 * i] = False
            key_pad[bi, 2 * i + 1] = False
            tgt[bi, i] = s["z_obs"][i]
            if bag is not None:
                bag[bi, i] = s["bag"][i]
            cmd_mask[bi, i] = True
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def cmd_hidden(net, b):
    """Hidden state (and prediction) at cmd positions: [sum_valid, .]."""
    pred, h = net(b["tok"], b["types"], b["key_pad"])
    cmd_pred = pred[:, 0::2]          # [B, maxn, .] — cmd tokens are even positions
    cmd_h = h[:, 0::2]
    m = b["cmd_mask"]
    return cmd_pred[m], cmd_h[m], b["tgt"][m], (b["bag"][m] if "bag" in b else None)


def train_model(aux, fit, device, vsize=0, steps=4000, bs=64, lr=3e-4, seed=0, no_history=False,
                jepa_loss=None):
    """jepa_loss: optional callable(pred, tgt)->scalar overriding the default MSE for the 'jepa'
    aux (used by the evolve harness to train under an evolved objective; None = MSE baseline)."""
    torch.manual_seed(seed)
    net = SeqWorldModel(aux, vsize, no_history=no_history).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    bce = nn.functional.binary_cross_entropy_with_logits
    jloss = jepa_loss if jepa_loss is not None else (lambda p, t: ((p - t) ** 2).mean())
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(fit), (bs,), generator=g).tolist()
        b = collate([fit[i] for i in idx], device)
        pred, _, tgt, bag = cmd_hidden(net, b)
        loss = jloss(pred, tgt) if aux == "jepa" else bce(pred, bag)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0:
            print(f"  [{aux}] step {step} loss {loss.item():.4f}", flush=True)
    return net


class CmdOnlyMLP(nn.Module):
    """History-FREE learned baseline: f(z_cmd) -> z_obs, no exploration context. The critical
    ablation — if the sequence world model can't beat this, the history/sequence buys nothing
    (it's just a per-command lookup, which is what a world model is supposed to transcend)."""

    def __init__(self, d=D, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                                 nn.Linear(h, d))

    def forward(self, zc):
        return self.net(zc)


def train_cmd_only(fit, device, steps=4000, bs=256, lr=3e-4, seed=0):
    zc = torch.cat([s["z_cmd"] for s in fit]); zo = torch.cat([s["z_obs"] for s in fit])
    torch.manual_seed(seed)
    net = CmdOnlyMLP().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for step in range(1, steps + 1):
        idx = torch.randint(0, zc.shape[0], (bs,), generator=g)
        loss = ((net(zc[idx].to(device)) - zo[idx].to(device)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return net


# ---------------------------------------------------------------- flat eval tensors

@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Run the model over every sequence; collect per-step predicted z_obs, hidden state,
    true z_obs, previous z_obs (copy baseline), command text, image. Order = step order."""
    net.eval()  # dropout off for inference (MPS fused attention also rejects nonzero dropout)
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::2].cpu(); cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
            for t in range(n):
                preds.append(cmd_pred[bi, t]); hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t]); imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [verb_of(c) for c in cmds]}


# ---------------------------------------------------------------- retrieval metric

def _foils_random(N, K, gen):
    return torch.randint(0, N, (N, K), generator=gen)


def _foils_sameverb(verbs, K, gen):
    """[N,K] foil indices, each row drawn (with replacement) from the SAME-verb steps."""
    by_verb = defaultdict(list)
    for j, v in enumerate(verbs):
        by_verb[v].append(j)
    pools = {v: torch.tensor(idxs) for v, idxs in by_verb.items()}
    foil = torch.empty(len(verbs), K, dtype=torch.long)
    for v, rows in by_verb.items():
        pool = pools[v]
        pick = pool[torch.randint(0, len(pool), (len(rows), K), generator=gen)]
        foil[torch.tensor(rows)] = pick
    return foil


@torch.no_grad()
def _rank_stats(pred, true, foil_idx, blk=1024):
    """Given per-step foil indices [N,K], score each candidate (true + K foils) by distance to
    pred and count foils strictly closer than the true. Returns (top1, mrr) over all N."""
    N = true.shape[0]
    top1 = 0.0; mrr = 0.0
    for s0 in range(0, N, blk):
        sl = slice(s0, min(s0 + blk, N))
        q = pred[sl].unsqueeze(1)                                    # [b,1,D]
        d_true = ((true[sl] - pred[sl]) ** 2).mean(-1, keepdim=True)  # [b,1]
        d_foil = ((true[foil_idx[sl]] - q) ** 2).mean(-1)            # [b,K]
        closer = (d_foil < d_true).sum(1)                            # strict: ties don't beat true
        top1 += (closer == 0).float().sum().item()
        mrr += (1.0 / (closer.float() + 1.0)).sum().item()
    return top1 / N, mrr / N


def _inject_forced(f_h, forced):
    """Replace the FIRST m same-verb foil slots with pre-registered forced foils (§8.1 UD-4(b)
    counterfactual injection). `forced` [N,m] holds indices INTO `true` (same index space as f_h);
    −1 = no forced foil for that slot (keep the sampled foil). Only the same-verb arm is injected;
    the random arm stays pure. Returns a fresh tensor — `f_h` is untouched."""
    out = f_h.clone()
    m = min(forced.shape[1], out.shape[1])
    seg = out[:, :m]                       # a view into out
    fv = forced[:, :m]
    valid = fv >= 0
    seg[valid] = fv[valid].to(seg.dtype)
    return out


def retrieval(pred, true, verbs, n_foils=63, rounds=4, seed=0, forced_foils=None):
    """For each step, rank the TRUE next obs against foils by distance to `pred`: random foils
    AND hard same-verb foils. top-1 acc + MRR, averaged over rounds. `pred` is any prediction
    aligned with `true` (model output, z_prev for copy, zeros for predict-mean, ...). Ties are
    not counted as beating the true candidate, so predict-mean -> ~chance by construction.

    forced_foils (default None = the exact historical code path, byte-identical): an [N,m] index
    tensor (values index into `true`; −1 = none). When given, the surviving forced indices REPLACE
    the first m of the n_foils SAMPLED same-verb foils in EVERY round, before _rank_stats (which is
    unchanged). The random-foil arm is never injected. Indices must already be in `true`'s space —
    content_retrieval owns the full→subset seam translation (§8.1)."""
    N = true.shape[0]
    gen = torch.Generator().manual_seed(seed)
    t1s = mrrs = t1r = mrrr = 0.0
    for _ in range(rounds):
        f_h = _foils_sameverb(verbs, n_foils, gen)
        if forced_foils is not None:
            f_h = _inject_forced(f_h, forced_foils)
        a, b = _rank_stats(pred, true, f_h)
        t1s += a; mrrs += b
        f_r = _foils_random(N, n_foils, gen); a, b = _rank_stats(pred, true, f_r)
        t1r += a; mrrr += b
    return {"top1_sameverb": t1s / rounds, "mrr_sameverb": mrrs / rounds,
            "top1_random": t1r / rounds, "mrr_random": mrrr / rounds}


VERBSET = ("uname", "ls", "cat", "cd")


def per_verb_breakdown(preds, true, verbs, seed, verbset=VERBSET):
    """top-1 (same-verb foils) restricted to each verb's steps. Exposes whether the world
    model's advantage is real (ls/cat: predict a listing/file's content on an unseen system)
    or trivial (cd: the observation is just `cwd=<target>`, echoable from the command)."""
    out = {}
    for v in verbset:
        idx = [i for i, vv in enumerate(verbs) if vv == v]
        if len(idx) < 20:
            continue
        ii = torch.tensor(idx); sub_true = true[ii]; sub_verbs = [v] * len(idx)
        row = {"n": len(idx)}
        for name, p in preds.items():
            row[name] = retrieval(p[ii], sub_true, sub_verbs, seed=seed)["top1_sameverb"]
        out[v] = row
    return out


def content_retrieval(pred, true, verbs, content=("ls", "cat"), seed=0, forced_foils=None):
    """Retrieval restricted to CONTENT verbs (ls/cat) — the observations that are NOT lexically
    echoable from the command (unlike cd's `cwd=<target>`). This is the honest headline: can the
    model predict a listing / file content on an unseen system? Foils are same-verb within the
    content subset.

    forced_foils (default None = the exact historical path, byte-identical): an [N,m] index tensor
    of counterfactual foils in FULL-array step positions. content_retrieval owns the §8.1 SUBSET-SEAM
    TRANSLATION: (a) row-subset forced_foils to the content rows `ii`; (b) remap each forced VALUE
    from full-array position to CONTENT-SUBSET position; (c) DROP any forced target outside the
    content subset (its embedding is absent from the subset `true`), counting it in `cf_dropped`
    (returned in the result) — a dropped slot falls back to a sampled foil in retrieval()."""
    idx = [i for i, v in enumerate(verbs) if v in content]
    ii = torch.tensor(idx)
    if forced_foils is None:
        return retrieval(pred[ii], true[ii], [verbs[i] for i in idx], seed=seed)
    full2sub = {full: sub for sub, full in enumerate(idx)}
    ff_rows = forced_foils[ii]                       # [N_sub, m] — row-subset to content rows
    sub_forced = torch.full_like(ff_rows, -1)
    cf_dropped = 0
    for r in range(ff_rows.shape[0]):
        for c in range(ff_rows.shape[1]):
            val = int(ff_rows[r, c])
            if val < 0:
                continue
            s = full2sub.get(val)
            if s is None:
                cf_dropped += 1                      # forced target outside the content subset
            else:
                sub_forced[r, c] = s
    res = retrieval(pred[ii], true[ii], [verbs[i] for i in idx], seed=seed, forced_foils=sub_forced)
    res["cf_dropped"] = cf_dropped
    return res


def latent_mse(pred, true):
    return ((pred - true) ** 2).mean().item()


def cosine(pred, true):
    a = torch.nn.functional.normalize(pred, dim=-1)
    b = torch.nn.functional.normalize(true, dim=-1)
    return (a * b).sum(-1).mean().item()


# ---------------------------------------------------------------- retrieve-by-command baseline

def retrieve_by_cmd_baseline(fit_seqs, eval_flat):
    """No-model lexical memory: predict the eval step's obs as the obs of the TRAIN step whose
    COMMAND embedding is nearest (cosine). Returns a prediction tensor aligned with
    eval_flat['true']. This is the "just memorize commands" competitor a world model must beat."""
    train_cmd = torch.cat([s["z_cmd"] for s in fit_seqs])          # [M, D]
    train_obs = torch.cat([s["z_obs"] for s in fit_seqs])          # [M, D]
    q = eval_flat["_cmd_embs"]                                     # [N, D]
    tn = torch.nn.functional.normalize(train_cmd, dim=-1)
    preds = []
    for i in range(0, q.shape[0], 256):
        qn = torch.nn.functional.normalize(q[i:i + 256], dim=-1)
        preds.append(train_obs[(qn @ tn.T).argmax(1)])
    return torch.cat(preds)


# ---------------------------------------------------------------- splits

def split_train_dev(fit_seqs, frac=0.1, seed=0):
    rng = random.Random(f"dev:{seed}")
    idx = list(range(len(fit_seqs)))
    rng.shuffle(idx)
    k = max(1, int(len(idx) * frac))
    dev = set(idx[:k])
    return ([s for i, s in enumerate(fit_seqs) if i not in dev],
            [s for i, s in enumerate(fit_seqs) if i in dev])
