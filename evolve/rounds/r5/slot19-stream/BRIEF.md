TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: CROSSOVER — combine the parent with the second program below into one coherent design that keeps the best mechanism of each.

THE CONTRACT — axis 'stream': Owns the token layout. Expose collate(batch, device), extract_cmd_pred(pred_full, batch), flatten_predictions(net, seqs, device) and leakage_ok(net, device). Optionally expose extract_cmd_input(batch), which is what lets an objective opt into the causal context extension. leakage_ok is asserted before scoring and a stream that cannot demonstrate causality is rejected. If your collate writes anything into COMMAND tokens, you must also expose code_cmds(cmds, z_cmd) -> z_cmd': a pure, prefix-causal function from a sequence's ordered command strings and their standardized command embeddings to the command embeddings your net is trained on. The scoring instrument builds its own tokens from the embedding cache and reproduces your coding through code_cmds once per arm, so anything collate does to a command token that code_cmds does not reproduce would be present in training and absent at scoring; a gate checks this against a real collate and refuses the candidate before any GPU time. Prefix-causal means the value at a position may depend only on commands up to it — a scored window is a prefix of its sequence. OBSERVATION tokens are fixed: the target and the forced-choice candidate bank live in that space and the instrument builds both from the cache. One hard invariant on any coding: the token at the READ position must come out IDENTICAL under the native move chain and under a chain in which two contents exchange their moves. The read command string is the same in both, so a difference means the coding resolved the chain and handed the answer to the model at the position being scored; that is checked and refused. Coding the move commands themselves is fine — the model still has to integrate them.
The reference baseline below is authoritative — match its interface exactly, keep your module self-contained:
--------------------------------------------------------------------------------
"""Contract for any stream impl:
  collate(batch, device) -> dict with tok [B,L,D], types [B,L] in {0,1}, key_pad [B,L] bool,
      tgt [B,maxn,D] (single-vector standardized next-obs target per STEP — the target/eval space
      is FIXED across streams), cmd_mask [B,maxn] bool
  extract_cmd_pred(pred_full [B,L,D], batch) -> [B,maxn,D], the prediction at each step's cmd token
  flatten_predictions(net, seqs, device) -> dict with at least pred/prev/true/cmds/verbs, step order
  leakage_ok(net, device) -> bool, a stream-aware causality probe (corrupt obs_t, cmd_<=t frozen);
      asserted before scoring, and a stream that cannot demonstrate causality is rejected
  extract_cmd_input(batch) -> [B,maxn,D] (optional), what lets an objective opt into WANTS_CTX
"""

import torch

from realenv import seq_worldmodel as M

NAME = "baseline_interleave"
DESCRIPTION = "Single-vector cmd/obs interleave; bit-identical to the pre-axis harness plumbing."


# This impl must stay bit-identical to the pre-axis harness: collate/flatten delegate to the
# seq_worldmodel functions the harness always called, and leakage_ok uses the same seed, toy
# sequence and perturbed index, so archived fitnesses replay exactly.
def collate(batch, device):
    return M.collate(batch, device)


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def flatten_predictions(net, seqs, device):
    return M.flatten_predictions(net, seqs, device)


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0  # corrupt obs_3 (odd index 2*3+1)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

# The cups scoring instrument pins this layout; a stream declaring a different one is
# refused before any GPU time is spent (see eval/adapter.py).
CUPS_LAYOUT = "interleave2"
--------------------------------------------------------------------------------

PARENT — you are mutating this candidate.
  id                g0-14-pathstate-rssm
  its fitness       -0.0075   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r18_pathstate_latent_transition_worldmodel
  optim               r18_spectral_capped_transition_readout
  target              ·
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              ·
  head                r18_transition_forwardmodel_consistency

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

CROSSOVER PARTNER GENOME — combine your parent with this design. Its identity and its fitness are withheld by the information diet; judge it as a mechanism.
  objective           r12_antiretrieval_ring_negatives
  arch                r24_reference_closure_transport   params {"obs_info_bias": 4.0, "ref_d": 128, "ref_pd": 64, "ref_recency_init": 0.1}
  optim               r4_twotimescale_router_polyak_tail   params {"avg_frac": 0.25, "beta2": 0.95, "floor_ratio": 0.1, "hold_frac": 0.3, "key_d": 64, "lr": 0.0005, "momentum": 0.95, "ns_steps": 5, "rms_match": 0.2, "router_lr_mult": 2.0, "router_out_max": 128, "router_wd": 0.0, "spectral_cap": 4.0, "spectral_iters": 2, "warmup_frac": 0.04, "wd": 0.0005}
  target              identity
  batcher             r6_sysblock_hardneg_curriculum   params {"hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              r3_srcdst_hashslot_coding
  head                r20_dualpre_transition_consistency

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r4_pathslot_dualaddress_code (axis stream)
import hashlib

import torch

D = 768
SRC_DIM = 96
DST_DIM = 96
TAG_DIM = 16
CODE_DIM = SRC_DIM + DST_DIM + TAG_DIM
CODE_GAIN = 1.0
TAG_FEATS = 6
CACHE_MAX = 65536

NAME = "r4_pathslot_dualaddress_code"
DESCRIPTION = (
    "Overwrites three fixed, hash-chosen coordinate slots of every standardized COMMAND "
    "embedding with a purely lexical dual-address code parsed from that command string alone: "
    "a 96-bit Rademacher hash of its source path in the SOURCE slot, the same hash family "
    "applied to its destination path in the DESTINATION slot, and a 16-dim sign pattern of six "
    "structural flags (has-destination, mv, cat, ls, redirection, has-source) in the TAG slot. "
    "Commands with no source or no destination receive fixed NULL codes instead. The same path "
    "string yields the same 96-bit code wherever it appears, so a move's destination slot and a "
    "later command's source slot hold identical vectors exactly when the two paths are equal, "
    "and near-orthogonal vectors otherwise. The coding is a pure per-command function of the "
    "command string, so it is prefix-causal and unchanged when two contents exchange their "
    "moves; the remaining 560 embedding coordinates and all OBSERVATION tokens are untouched, "
    "and the token layout stays the interleaved cmd/obs stream."
)

_REDIR = (">", ">>", "1>", "2>", ">|")

_BYTE_SIGNS = tuple(
    tuple(1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)) for b in range(256)
)


def _stable_order():
    return sorted(range(D), key=lambda i: hashlib.blake2b(
        b"slot:" + str(i).encode("ascii"), digest_size=16).digest())


_ORDER = _stable_order()
_SRC_IDX = list(_ORDER[:SRC_DIM])
_DST_IDX = list(_ORDER[SRC_DIM:SRC_DIM + DST_DIM])
_TAG_IDX = list(_ORDER[SRC_DIM + DST_DIM:SRC_DIM + DST_DIM + TAG_DIM])
_IDX_CACHE = {}


def _slot_index_tensors(device):
    key = str(device)
    got = _IDX_CACHE.get(key)
    if got is None:
        got = (torch.tensor(_SRC_IDX, dtype=torch.long, device=device),
               torch.tensor(_DST_IDX, dtype=torch.long, device=device),
               torch.tensor(_TAG_IDX, dtype=torch.long, device=device))
        _IDX_CACHE[key] = got
    return got


def _sign_bits(key, nbits):
    digest = hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=nbits // 8).digest()
    out = []
    for b in digest:
        out.extend(_BYTE_SIGNS[b])
    return out


_NULL_SRC = _sign_bits("\x00null-source", SRC_DIM)
_NULL_DST = _sign_bits("\x00null-destination", DST_DIM)
_PATH_CACHE = {}
_CMD_CACHE = {}


def _path_code(path):
    got = _PATH_CACHE.get(path)
    if got is None:
        if len(_PATH_CACHE) >= CACHE_MAX:
            _PATH_CACHE.clear()
        got = _sign_bits("path\x1f" + path, SRC_DIM)
        _PATH_CACHE[path] = got
    return got


def parse_command(cmd):
    parts = cmd.split()
    if not parts:
        return "", None, None, -1
    verb = parts[0]
    ri = -1
    for i in range(1, len(parts)):
        if parts[i] in _REDIR:
            ri = i
            break
    if ri >= 0:
        left = [p for p in parts[1:ri] if p.startswith("/")]
        right = [p for p in parts[ri + 1:] if p.startswith("/")]
        return verb, (left[0] if left else None), (right[0] if right else None), ri
    paths = [p for p in parts[1:] if p.startswith("/")]
    if verb == "mv" and len(paths) >= 2:
        return verb, paths[0], paths[1], ri
    return verb, (paths[0] if paths else None), None, ri


def _command_code(cmd):
    verb, src, dst, ri = parse_command(cmd)
    vals = list(_NULL_SRC if src is None else _path_code(src))
    vals.extend(_NULL_DST if dst is None else _path_code(dst))
    feats = (
        1.0 if dst is not None else -1.0,
        1.0 if verb == "mv" else -1.0,
        1.0 if verb == "cat" else -1.0,
        1.0 if verb == "ls" else -1.0,
        1.0 if ri >= 0 else -1.0,
        1.0 if src is not None else -1.0,
    )
    vals.extend(feats[i % TAG_FEATS] for i in range(TAG_DIM))
    return torch.tensor(vals, dtype=torch.float32)


def _cached_command_code(cmd):
    got = _CMD_CACHE.get(cmd)
    if got is None:
        if len(_CMD_CACHE) >= CACHE_MAX:
            _CMD_CACHE.clear()
        got = _command_code(cmd)
        _CMD_CACHE[cmd] = got
    return got


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    n = min(len(cmds), int(out.shape[0]))
    if n <= 0:
        return out
    rowvals = [_cached_command_code(c if isinstance(c, str) else str(c)) for c in cmds[:n]]
    vals = torch.stack(rowvals).to(device=out.device, dtype=out.dtype)
    si, di, ti = _slot_index_tensors(out.device)
    rows = out[:n]
    rows[:, si] = CODE_GAIN * vals[:, :SRC_DIM]
    rows[:, di] = CODE_GAIN * vals[:, SRC_DIM:SRC_DIM + DST_DIM]
    rows[:, ti] = CODE_GAIN * vals[:, SRC_DIM + DST_DIM:]
    return out


def _coded_cmd_matrix(s, n):
    cmds = s.get("cmds") or []
    return code_cmds(list(cmds)[:n], s["z_cmd"][:n])


def collate(batch, device):
    maxn = max(int(s["z_obs"].shape[0]) for s in batch)
    L = 2 * maxn
    B = len(batch)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    tgt = torch.zeros(B, maxn, D)
    cmd_mask = torch.zeros(B, maxn, dtype=torch.bool)
    bag = None
    if "bag" in batch[0]:
        bag = torch.zeros(B, maxn, batch[0]["bag"].shape[1])
    for bi, s in enumerate(batch):
        n = int(s["z_obs"].shape[0])
        if n <= 0:
            continue
        zc = _coded_cmd_matrix(s, n)
        m = int(zc.shape[0])
        tok[bi, 0:2 * m:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = s["z_obs"][:n].to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n].to(tgt.dtype)
        cmd_mask[bi, :n] = True
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = pred[:, 0::2].cpu()
        cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = int(s["z_obs"].shape[0])
            for t in range(n):
                preds.append(cmd_pred[bi, t])
                hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t])
                imgs.append(s["image"])
    verbs = []
    for c in cmds:
        p = c.split()
        verbs.append(p[0] if p else "")
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs, "verbs": verbs}


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    cmds = ["ls -1 /", "cat /tmp/w/cups/g1/a.dat", "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg",
            "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat", "uname -a",
            "cat /tmp/w/cups/etc/b.cfg"]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())


CUPS_LAYOUT = "interleave2"

--- baseline_interleave (axis stream)
"""Contract for any stream impl:
  collate(batch, device) -> dict with tok [B,L,D], types [B,L] in {0,1}, key_pad [B,L] bool,
      tgt [B,maxn,D] (single-vector standardized next-obs target per STEP — the target/eval space
      is FIXED across streams), cmd_mask [B,maxn] bool
  extract_cmd_pred(pred_full [B,L,D], batch) -> [B,maxn,D], the prediction at each step's cmd token
  flatten_predictions(net, seqs, device) -> dict with at least pred/prev/true/cmds/verbs, step order
  leakage_ok(net, device) -> bool, a stream-aware causality probe (corrupt obs_t, cmd_<=t frozen);
      asserted before scoring, and a stream that cannot demonstrate causality is rejected
  extract_cmd_input(batch) -> [B,maxn,D] (optional), what lets an objective opt into WANTS_CTX
"""

import torch

from realenv import seq_worldmodel as M

NAME = "baseline_interleave"
DESCRIPTION = "Single-vector cmd/obs interleave; bit-identical to the pre-axis harness plumbing."


# This impl must stay bit-identical to the pre-axis harness: collate/flatten delegate to the
# seq_worldmodel functions the harness always called, and leakage_ok uses the same seed, toy
# sequence and perturbed index, so archived fitnesses replay exactly.
def collate(batch, device):
    return M.collate(batch, device)


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


def flatten_predictions(net, seqs, device):
    return M.flatten_predictions(net, seqs, device)


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, M.D), "z_cmd": torch.randn(6, M.D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = M.collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = M.collate(seq, device)
    b1["tok"][0, 7] = torch.randn(M.D, device=device) * 100.0  # corrupt obs_3 (odd index 2*3+1)
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

# The cups scoring instrument pins this layout; a stream declaring a different one is
# refused before any GPU time is spent (see eval/adapter.py).
CUPS_LAYOUT = "interleave2"

--- r3_srcdst_hashslot_coding (axis stream)
import hashlib
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r3_srcdst_hashslot_coding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and the observation tokens exactly as they are, and "
    "re-codes each COMMAND token position-wise as its standardized command embedding plus a "
    "deterministic sparse symbol code. The code writes the command's source path into one fixed "
    "coordinate block, its destination path into a second disjoint block, its verb into a third, "
    "and seven form flags (source present, explicit destination, mv, cat, ls, '>', '>>') into "
    "named dimensions. A path's entries are k-sparse signed values drawn from a SHAKE-256 digest "
    "of the path string, placed at the SAME relative offsets inside whichever of the two "
    "equally-wide path blocks is used, so one path yields one pattern that a fixed coordinate "
    "selection recovers from either role, and two paths differing only in a trailing suffix yield "
    "near-orthogonal patterns. Nothing outside a command's own string and its own embedding "
    "enters that command's token."
)

D = M.D
CUPS_LAYOUT = "interleave2"

SRC_LO, SRC_HI = 0, 352
DST_LO, DST_HI = 352, 704
VERB_LO, VERB_HI = 704, 752
FLAG_LO = 752
N_FLAGS = 7

PATH_K = 24
VERB_K = 8
PATH_GAIN = 2.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}


def _sparse_block(text, tag, lo, hi, k, gain, cols):
    span = hi - lo
    raw = hashlib.shake_256((tag + "\x00" + text).encode("utf-8")).digest(3 * k)
    for j in range(k):
        a = raw[3 * j]
        b = raw[3 * j + 1]
        c = raw[3 * j + 2]
        idx = lo + (((a << 8) | b) % span)
        if idx in cols:
            continue
        cols[idx] = gain if (c & 1) else -gain


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def _parse(cmd):
    toks = cmd.split()
    if not toks:
        return "", "", "", None, False
    verb = toks[0]
    positional = []
    redirect = None
    redir_target = None
    i = 1
    while i < len(toks):
        t = toks[i]
        if t == ">" or t == ">>":
            redirect = t
            if i + 1 < len(toks):
                redir_target = toks[i + 1]
                i += 1
        elif t.startswith(">>"):
            redirect = ">>"
            redir_target = t[2:] or redir_target
        elif t.startswith(">"):
            redirect = ">"
            redir_target = t[1:] or redir_target
        elif t.startswith("-") and len(t) > 1:
            pass
        else:
            positional.append(t)
        i += 1
    src = _norm_path(positional[0]) if positional else ""
    if redir_target:
        return verb, src, _norm_path(redir_target), redirect, True
    if verb in _DST_VERBS and len(positional) >= 2:
        return verb, src, _norm_path(positional[-1]), redirect, True
    return verb, src, src, redirect, False


def _compact(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    verb, src, dst, redirect, explicit = _parse(cmd)
    cols = {}
    if src:
        _sparse_block(src, "path", SRC_LO, SRC_HI, PATH_K, PATH_GAIN, cols)
    if dst:
        _sparse_block(dst, "path", DST_LO, DST_HI, PATH_K, PATH_GAIN, cols)
    if verb:
        _sparse_block(verb, "verb", VERB_LO, VERB_HI, VERB_K, VERB_GAIN, cols)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
    )
    for j in range(N_FLAGS):
        cols[FLAG_LO + j] = FLAG_GAIN if flags[j] else -FLAG_GAIN
    idx = np.fromiter(cols.keys(), dtype=np.int64, count=len(cols))
    val = np.fromiter(cols.values(), dtype=np.float32, count=len(cols))
    out = (idx, val)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = out
    return out


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    if n == 0 or not cmds:
        return out
    m = min(n, len(cmds))
    rows = []
    cols = []
    vals = []
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        idx, val = _compact(c)
        if idx.size == 0:
            continue
        rows.append(np.full(idx.size, i, dtype=np.int64))
        cols.append(idx)
        vals.append(val)
    if not rows:
        return out
    r = torch.from_numpy(np.concatenate(rows)).to(out.device)
    c = torch.from_numpy(np.concatenate(cols)).to(out.device)
    v = torch.from_numpy(np.concatenate(vals)).to(device=out.device, dtype=out.dtype)
    out.index_put_((r, c), v, accumulate=True)
    return out


def collate(batch, device):
    maxn = max(int(s["z_obs"].shape[0]) for s in batch)
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
        n = int(s["z_obs"].shape[0])
        if n == 0:
            continue
        zo = s["z_obs"][:n]
        zc = code_cmds(list(s.get("cmds") or []), s["z_cmd"][:n])
        tok[bi, 0:2 * n:2] = zc
        tok[bi, 1:2 * n:2] = zo
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo
        if bag is not None:
            bag[bi, :n] = s["bag"][:n]
        cmd_mask[bi, :n] = True
    out = {"tok": tok.to(device), "types": types.to(device), "key_pad": key_pad.to(device),
           "tgt": tgt.to(device), "cmd_mask": cmd_mask.to(device)}
    if bag is not None:
        out["bag"] = bag.to(device)
    return out


def extract_cmd_pred(pred_full, batch):
    return pred_full[:, 0::2]


def extract_cmd_input(batch):
    return batch["tok"][:, 0::2]


@torch.no_grad()
def flatten_predictions(net, seqs, device, bs=64):
    net.eval()
    preds, hids, trues, prevs, cmds, imgs = [], [], [], [], [], []
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        b = collate(chunk, device)
        pred, h = net(b["tok"], b["types"], b["key_pad"])
        cmd_pred = extract_cmd_pred(pred, b).cpu()
        cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = int(s["z_obs"].shape[0])
            for t in range(n):
                preds.append(cmd_pred[bi, t])
                hids.append(cmd_h[bi, t])
                trues.append(s["z_obs"][t])
                prevs.append(s["z_obs"][t - 1] if t > 0 else torch.zeros(D))
                cmds.append(s["cmds"][t])
                imgs.append(s["image"])
    return {"pred": torch.stack(preds), "h": torch.stack(hids), "true": torch.stack(trues),
            "prev": torch.stack(prevs), "cmds": cmds, "imgs": imgs,
            "verbs": [M.verb_of(c) for c in cmds]}


@torch.no_grad()
def leakage_ok(net, device):
    net.eval()
    torch.manual_seed(0)
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D),
            "cmds": ["ls /a"] * 6, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of -0.0075 (g0-14-pathstate-rssm, full budget, inner split).
The unmodified baseline scores -0.0075 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
