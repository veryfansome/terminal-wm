"""Sequence world model over shell trajectories, and the eval primitives the harness and the
cups instruments call: encode, standardize, split, collate, retrieval.

A causal transformer runs over the interleaved token stream

    cmd_0  obs_0  cmd_1  obs_1  ...  cmd_t  obs_t

of frozen-encoder embeddings; the hidden state at each command position predicts that
command's resulting observation embedding z_obs[t] in standardized latent space.
"""

import json
import pathlib
import random
from collections import defaultdict

import torch
import torch.nn as nn

D = 768
OBS_CAP = 1600


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def render_obs(step):
    """Observation text: resulting cwd + exit code + truncated output. The cwd is part of the
    observation because `cd` has no stdout and the new working directory is its only effect."""
    out = step.get("output", "") or ""
    if len(out) > OBS_CAP:
        out = out[:OBS_CAP] + f"\n...[{len(out) - OBS_CAP} more chars]"
    return f"cwd={step.get('cwd', '/')} exit={step.get('exit', 0)}\n{out}"


def render_cmd(step):
    return step["cmd"]


def verb_of(cmd):
    p = cmd.split()
    return p[0] if p else ""


@torch.no_grad()
def encode_split(path, model, tok, device, bs=96):
    """Per-sequence arrays z_obs[n,768], z_cmd[n,768] (frozen mean-pooled), plus cmds/image/ok."""
    seqs = [json.loads(l) for l in open(path)]
    obs_texts, cmd_texts, spans = [], [], []
    for sq in seqs:
        start = len(obs_texts)
        for s in sq["steps"]:
            obs_texts.append(render_obs(s))
            cmd_texts.append(render_cmd(s))
        spans.append((start, len(obs_texts)))

    def enc(texts, tag):
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
                    "ok": [s.get("exit", 0) == 0 and bool((s.get("output") or "").strip())
                           for s in sq["steps"]]})
    return out


def _v3_cache_guard(data_root):
    """Raise if a v3 derived root's embedding caches are stale. No-op for roots with no
    cache_meta.json."""
    import hashlib
    cm_path = pathlib.Path(data_root) / "cache_meta.json"
    if not cm_path.exists():
        return
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


def _iter_steps(data_root, split):
    for line in open(pathlib.Path(data_root) / f"{split}.jsonl"):
        for s in json.loads(line)["steps"]:
            yield s


def standardize_stats(seqs):
    allo = torch.cat([s["z_obs"] for s in seqs])
    allc = torch.cat([s["z_cmd"] for s in seqs])
    return (allo.mean(0, keepdim=True), allo.std(0, keepdim=True).clamp(min=1e-6),
            allc.mean(0, keepdim=True), allc.std(0, keepdim=True).clamp(min=1e-6))


def apply_stats(seqs, mo, so, mc, sc):
    for s in seqs:
        s["z_obs"] = (s["z_obs"] - mo) / so
        s["z_cmd"] = (s["z_cmd"] - mc) / sc
        if "z_obs_multi" in s:
            s["z_obs_multi"] = (s["z_obs_multi"] - mo.unsqueeze(0)) / so.unsqueeze(0)


class SeqWorldModel(nn.Module):
    """Causal transformer over interleaved [cmd_0,obs_0,cmd_1,obs_1,...] frozen-embedding tokens.
    aux='jepa' predicts z_obs[t] at cmd positions; aux='recon' predicts the obs token-bag."""

    def __init__(self, aux="jepa", vsize=0, d=192, layers=4, heads=4, dropout=0.1,
                 no_history=False):
        super().__init__()
        self.aux = aux
        self.d = d
        self.no_history = no_history
        self.proj = nn.Linear(D, d)
        self.type_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(64, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, D if aux == "jepa" else vsize)

    def encode(self, tok_emb, types, key_pad):
        """tok_emb [B,L,D] frozen embeddings; types [B,L] in {0,1}; key_pad [B,L] True=pad.
        Returns hidden [B,L,d]. no_history=True masks all attention except self."""
        B, L, _ = tok_emb.shape
        pos = torch.arange(L, device=tok_emb.device)
        x = self.proj(tok_emb) + self.type_emb(types) + self.pos_emb(pos)[None]
        if self.no_history:
            mask = ~torch.eye(L, device=tok_emb.device, dtype=torch.bool)
        else:
            mask = torch.triu(torch.ones(L, L, device=tok_emb.device, dtype=torch.bool), 1)
        return self.tf(x, mask=mask, src_key_padding_mask=key_pad)

    def forward(self, tok_emb, types, key_pad):
        h = self.encode(tok_emb, types, key_pad)
        return self.head(h), h


def collate(batch, device):
    """Interleave each sequence's cmd/obs embeddings into a padded token stream. Returns
    tok/types/key_pad/tgt/cmd_mask (and bag, when present) as tensors on device."""
    maxn = max(s["z_obs"].shape[0] for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
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
    cmd_pred = pred[:, 0::2]
    cmd_h = h[:, 0::2]
    m = b["cmd_mask"]
    return cmd_pred[m], cmd_h[m], b["tgt"][m], (b["bag"][m] if "bag" in b else None)


def train_model(aux, fit, device, vsize=0, steps=4000, bs=64, lr=3e-4, seed=0, no_history=False,
                jepa_loss=None):
    """Train and return a SeqWorldModel. jepa_loss overrides the default MSE when given."""
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
    """History-free learned baseline: f(z_cmd) -> z_obs, with no exploration context."""

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


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    """Per-step pred / h / true / prev z_obs plus cmds, imgs and verbs, in step order."""
    net.eval()
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
    """Score each candidate (true + K foils) by distance to pred. Returns (top1, mrr) over N."""
    N = true.shape[0]
    top1 = 0.0; mrr = 0.0
    for s0 in range(0, N, blk):
        sl = slice(s0, min(s0 + blk, N))
        q = pred[sl].unsqueeze(1)
        d_true = ((true[sl] - pred[sl]) ** 2).mean(-1, keepdim=True)
        d_foil = ((true[foil_idx[sl]] - q) ** 2).mean(-1)
        # Strict inequality: a tie does not count as beating the true candidate, which is what
        # puts a constant prediction at chance instead of scoring it.
        closer = (d_foil < d_true).sum(1)
        top1 += (closer == 0).float().sum().item()
        mrr += (1.0 / (closer.float() + 1.0)).sum().item()
    return top1 / N, mrr / N


def _inject_forced(f_h, forced):
    """Copy of f_h with its first m columns replaced by `forced` [N,m] indices into `true`;
    -1 keeps the sampled foil. f_h is untouched."""
    out = f_h.clone()
    m = min(forced.shape[1], out.shape[1])
    seg = out[:, :m]
    fv = forced[:, :m]
    valid = fv >= 0
    seg[valid] = fv[valid].to(seg.dtype)
    return out


def retrieval(pred, true, verbs, n_foils=63, rounds=4, seed=0, forced_foils=None):
    """Rank the true next obs against random foils and same-verb foils by distance to `pred`.
    Returns top-1 and MRR for both foil arms, averaged over rounds. forced_foils [N,m] holds
    indices into `true` (-1 = none) that replace the first m sampled same-verb foils each round;
    the random arm is never injected."""
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
    """top-1 (same-verb foils) restricted to each verb's steps, for verbs with >= 20 steps."""
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
    """Retrieval restricted to the content verbs, with foils drawn from the same subset.
    forced_foils arrives in FULL-array positions and is translated here into content-subset
    positions; a forced target outside the subset is dropped and counted in `cf_dropped`."""
    idx = [i for i, v in enumerate(verbs) if v in content]
    ii = torch.tensor(idx)
    if forced_foils is None:
        return retrieval(pred[ii], true[ii], [verbs[i] for i in idx], seed=seed)
    full2sub = {full: sub for sub, full in enumerate(idx)}
    ff_rows = forced_foils[ii]
    sub_forced = torch.full_like(ff_rows, -1)
    cf_dropped = 0
    for r in range(ff_rows.shape[0]):
        for c in range(ff_rows.shape[1]):
            val = int(ff_rows[r, c])
            if val < 0:
                continue
            s = full2sub.get(val)
            if s is None:
                cf_dropped += 1
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


def retrieve_by_cmd_baseline(fit_seqs, eval_flat):
    """Lexical-memory baseline: the obs of the train step whose command embedding is nearest
    (cosine). Returns a prediction tensor aligned with eval_flat['true']."""
    train_cmd = torch.cat([s["z_cmd"] for s in fit_seqs])
    train_obs = torch.cat([s["z_obs"] for s in fit_seqs])
    q = eval_flat["_cmd_embs"]
    tn = torch.nn.functional.normalize(train_cmd, dim=-1)
    preds = []
    for i in range(0, q.shape[0], 256):
        qn = torch.nn.functional.normalize(q[i:i + 256], dim=-1)
        preds.append(train_obs[(qn @ tn.T).argmax(1)])
    return torch.cat(preds)


def split_train_dev(fit_seqs, frac=0.1, seed=0):
    rng = random.Random(f"dev:{seed}")
    idx = list(range(len(fit_seqs)))
    rng.shuffle(idx)
    k = max(1, int(len(idx) * frac))
    dev = set(idx[:k])
    return ([s for i, s in enumerate(fit_seqs) if i not in dev],
            [s for i, s in enumerate(fit_seqs) if i in dev])
