TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: TARGETED EDIT — make a focused change to the parent; do NOT rewrite everything. Keep what works, change one mechanism.

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
  id                r6-07-capacity-ledger-transport
  its fitness       +0.0562   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r22_exact_target_equivalence_quotient
  arch                r6_capacity_ledger_transport   params {"addr_hidden": 192, "avail_eps": 0.0001, "consume_bias": 0.0, "d": 176, "dropout": 0.1, "dst_init_noise": 0.02, "ffn_mult": 2, "gate_hidden": 96, "heads": 4, "key_d": 64, "layers": 4, "link_temp": 0.25, "presence_margin": 1.0, "presence_scale": 0.5, "read_gate_bias": -0.5, "readout_init": 0.5, "recency_init": -1.0, "take_bias": 1.0, "tr_gscale": 0.5, "tr_hidden": 192, "tr_mix_bias": -2.0, "tr_out_std": 0.01, "write_bias": 1.0}
  optim               r18_spectral_capped_transition_readout
  target              within_sequence_contrast_equalizer
  batcher             r20_depthtilted_sysblock_curriculum   params {"depth_base": 1.0, "depth_pow": 2.0, "hard_frac_max": 0.75, "n_block_images": 1, "ramp_frac": 0.3}
  stream              baseline_interleave
  head                r4_chainlink_route_transport

YOUR PARENT'S CURRENT stream IMPL — baseline_interleave (this is the code you are mutating):
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

PARENT'S EVAL FEEDBACK: comp_ca +0.0112 n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

PRIOR MECHANISMS — the engine sampled these as relevant to your slot, shown as SOURCE. No outcome is attached to any of them, and no ordering is implied. There is no instruction to beat any of them; your objective is your own parent.

--- r5_dualrole_address_mention_code (axis stream)
import hashlib
import threading

import numpy as np
import torch

D = 768

NAME = "r5_dualrole_address_mention_code"
DESCRIPTION = (
    "Keeps the cmd/obs interleave layout and every OBSERVATION token untouched, and rewrites each "
    "COMMAND token as four fixed coordinate regions computed from that command string alone. The "
    "first two regions, equal width and aligned index-for-index, are OVERWRITTEN (the standardized "
    "embedding is zeroed there) with a dense Rademacher hash of the command's source path and the "
    "same hash of its destination path, so one path yields one identical bit pattern in either "
    "role and a single linear map applied to the two regions returns the SAME vector exactly when "
    "source and destination are the same path and near-orthogonal vectors otherwise; a command "
    "with no explicit destination takes its source as destination, and a command with no path "
    "takes a fixed null pattern. The third region carries an undirected MENTION code, a second "
    "independent hash family summed over the paths the command names in any role and renormalized, "
    "so plain token cosine between two commands measures whether they touch a common path rather "
    "than whether their sentences look alike. The fourth region carries a hash of the verb and "
    "eight replicated form flags (source present, explicit destination, mv, cat, ls, '>', '>>', "
    "destination distinct from source). The standardized command embedding survives, scaled, only "
    "under the mention and form regions. The coding is a pure per-command function, so it is "
    "prefix-causal and unchanged when two contents exchange their moves."
)

SRC_LO, SRC_HI = 0, 224
DST_LO, DST_HI = 224, 448
MEN_LO, MEN_HI = 448, 704
VERB_LO, VERB_HI = 704, 736
FLAG_LO, FLAG_HI = 736, 768

ADDR_W = SRC_HI - SRC_LO
MEN_W = MEN_HI - MEN_LO
VERB_W = VERB_HI - VERB_LO
FLAG_W = FLAG_HI - FLAG_LO
N_FLAGS = 8

RAW_SCALE = 0.6
PATH_GAIN = 1.0
MEN_GAIN = 1.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0
_INV_SQRT2 = 0.7071067811865476

CUPS_LAYOUT = "interleave2"

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}
_PATH_CACHE = {}
_SCALE_CACHE = {}

_BYTE_SIGNS = np.array(
    [[1.0 if (b >> (7 - k)) & 1 else -1.0 for k in range(8)] for b in range(256)],
    dtype=np.float32,
)

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_REDIR_TOKENS = (">", ">>", "1>", "2>", ">|")


def _signs(text, width):
    dig = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=width // 8).digest()
    return _BYTE_SIGNS[np.frombuffer(dig, dtype=np.uint8)].reshape(-1)


def _cached_signs(tag, text, width):
    key = (tag, text)
    hit = _PATH_CACHE.get(key)
    if hit is not None:
        return hit
    out = _signs(tag + "\x1f" + text, width)
    with _CACHE_LOCK:
        if len(_PATH_CACHE) >= _CACHE_CAP:
            _PATH_CACHE.clear()
        _PATH_CACHE[key] = out
    return out


def _norm_path(p):
    if not p:
        return ""
    while "//" in p:
        p = p.replace("//", "/")
    if len(p) > 1 and p.endswith("/"):
        p = p.rstrip("/") or "/"
    return p


def parse_command(cmd):
    toks = cmd.split()
    if not toks:
        return "", "", "", "", False
    verb = toks[0]
    positional = []
    redirect = ""
    redir_target = ""
    i = 1
    while i < len(toks):
        t = toks[i]
        if t in _REDIR_TOKENS:
            redirect = ">>" if t == ">>" else ">"
            if i + 1 < len(toks):
                redir_target = toks[i + 1]
                i += 1
        elif t.startswith(">>") and len(t) > 2:
            redirect = ">>"
            redir_target = t[2:]
        elif t.startswith(">") and len(t) > 1:
            redirect = ">"
            redir_target = t[1:]
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


def _build_code(cmd):
    verb, src, dst, redirect, explicit = parse_command(cmd)
    row = np.zeros(D, dtype=np.float32)
    row[SRC_LO:SRC_HI] = PATH_GAIN * _cached_signs("addr", src, ADDR_W)
    row[DST_LO:DST_HI] = PATH_GAIN * _cached_signs("addr", dst, ADDR_W)
    if explicit and dst != src:
        men = (_cached_signs("mention", src, MEN_W)
               + _cached_signs("mention", dst, MEN_W)) * _INV_SQRT2
    else:
        men = _cached_signs("mention", src, MEN_W)
    row[MEN_LO:MEN_HI] = MEN_GAIN * men
    row[VERB_LO:VERB_HI] = VERB_GAIN * _cached_signs("verb", verb, VERB_W)
    flags = (
        bool(src),
        bool(explicit),
        verb == "mv",
        verb == "cat",
        verb == "ls",
        redirect == ">",
        redirect == ">>",
        bool(explicit) and dst != src,
    )
    fv = np.empty(FLAG_W, dtype=np.float32)
    for j in range(FLAG_W):
        fv[j] = FLAG_GAIN if flags[j % N_FLAGS] else -FLAG_GAIN
    row[FLAG_LO:FLAG_HI] = fv
    return row


def _command_code(cmd):
    hit = _CODE_CACHE.get(cmd)
    if hit is not None:
        return hit
    out = _build_code(cmd)
    with _CACHE_LOCK:
        if len(_CODE_CACHE) >= _CACHE_CAP:
            _CODE_CACHE.clear()
        _CODE_CACHE[cmd] = out
    return out


def _raw_scale(device, dtype):
    key = (str(device), str(dtype))
    hit = _SCALE_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.zeros(D, device=device, dtype=dtype)
    v[MEN_LO:] = RAW_SCALE
    _SCALE_CACHE[key] = v
    return v


def code_cmds(cmds, z_cmd):
    out = z_cmd.clone()
    if out.dim() != 2 or int(out.shape[1]) != D:
        return out
    n = int(out.shape[0])
    if n == 0 or not cmds:
        return out
    m = min(n, len(cmds))
    if m <= 0:
        return out
    rows = np.empty((m, D), dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        rows[i] = _command_code(c if isinstance(c, str) else str(c))
    code = torch.from_numpy(rows).to(device=out.device, dtype=out.dtype)
    scale = _raw_scale(out.device, out.dtype)
    out[:m] = out[:m] * scale.unsqueeze(0) + code
    return out


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
        zo = s["z_obs"][:n]
        zc = code_cmds(list(s.get("cmds") or []), s["z_cmd"][:n])
        tok[bi, 0:2 * n:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = zo.to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo.to(tgt.dtype)
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
    cmds = [
        "ls -1 /",
        "cat /tmp/w/cups/g1/a.dat",
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg",
        "cat /tmp/w/cups/g2/c.bin >> /tmp/w/cups/g2/acc.dat",
        "mv /tmp/w/cups/etc/b.cfg /tmp/w/cups/etc/b.cfg.1",
        "cat /tmp/w/cups/etc/b.cfg.1",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
    p1 = net(b1["tok"], b1["types"], b1["key_pad"])[0][:, 0::2].cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())

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

--- r6_generation_slot_address_coding (axis stream)
import hashlib
import math
import threading

import numpy as np
import torch

from realenv import seq_worldmodel as M

NAME = "r6_generation_slot_address_coding"
DESCRIPTION = (
    "Keeps the cmd/obs interleave, every OBSERVATION token untouched, and the parent's additive "
    "sparse hash of source path, destination path, verb and seven form flags, but narrows the two "
    "path blocks to 312 coordinates each to free an 80-coordinate GENERATION channel that is "
    "OVERWRITTEN rather than added. A path's generation is the integer of its trailing '.<digits>' "
    "component, 0 when it has none; the channel writes the source generation into coordinates "
    "624-663 and the destination generation into coordinates 664-703 from ONE shared 16-entry "
    "codebook, index for index, so the same generation yields the same 40-vector in either role "
    "and a fixed coordinate offset turns a source read into a destination key. A codebook entry "
    "is a row of a 16-point Sylvester-Hadamard matrix laid down twice under a fixed column "
    "permutation, so two distinct generations are EXACTLY orthogonal in those 32 coordinates "
    "however many of them appear, followed by a present flag, a linear ramp and a half-period "
    "sine/cosine pair so ordinal distance stays linearly readable. Overwriting rather than adding "
    "removes the standardized embedding from those coordinates, leaving an address alphabet that "
    "is the same in every trajectory and every image instead of a fresh random pattern per path. "
    "Nothing outside a command's own string and its own embedding enters that command's token, so "
    "the coding is position-wise, prefix-causal, and identical for a read command whichever way "
    "the moves before it were routed."
)

D = M.D
CUPS_LAYOUT = "interleave2"

SRC_LO, SRC_HI = 0, 312
DST_LO, DST_HI = 312, 624
SGEN_LO, SGEN_HI = 624, 664
DGEN_LO, DGEN_HI = 664, 704
VERB_LO, VERB_HI = 704, 752
FLAG_LO = 752
N_FLAGS = 7

PATH_K = 24
VERB_K = 8
PATH_GAIN = 2.0
VERB_GAIN = 1.0
FLAG_GAIN = 1.0
GEN_GAIN = 2.0

GEN_W = 40
GEN_CODE_W = 32
GEN_ORDER = 16
MAX_GEN = GEN_ORDER - 1

_DST_VERBS = ("mv", "cp", "ln", "install", "rename")
_DIGITS = frozenset("0123456789")

_CACHE_CAP = 200000
_CACHE_LOCK = threading.Lock()
_CODE_CACHE = {}


def _hadamard(order):
    h = np.ones((1, 1), dtype=np.float32)
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    return h


def _fixed_permutation(width):
    raw = hashlib.shake_256(b"gen-column-permutation-v1").digest(4 * width)
    keys = [int.from_bytes(raw[4 * i:4 * i + 4], "big") for i in range(width)]
    return np.asarray(sorted(range(width), key=lambda i: (keys[i], i)), dtype=np.int64)


def _build_gen_table():
    had = _hadamard(GEN_ORDER)
    perm = _fixed_permutation(GEN_ORDER)
    half = GEN_CODE_W // 2
    table = np.zeros((GEN_ORDER, GEN_W), dtype=np.float32)
    for g in range(GEN_ORDER):
        row = had[(g + 1) % GEN_ORDER]
        table[g, :half] = row * GEN_GAIN
        table[g, half:GEN_CODE_W] = row[perm] * GEN_GAIN
        table[g, GEN_CODE_W] = GEN_GAIN if g > 0 else -GEN_GAIN
        table[g, GEN_CODE_W + 1] = GEN_GAIN * (float(g) / float(MAX_GEN))
        table[g, GEN_CODE_W + 2] = GEN_GAIN * math.sin(math.pi * float(g) / float(MAX_GEN))
        table[g, GEN_CODE_W + 3] = GEN_GAIN * math.cos(math.pi * float(g) / float(MAX_GEN))
    return table


GEN_TABLE = _build_gen_table()


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


def _generation_of(path):
    if not path:
        return 0
    i = path.rfind(".")
    if i <= 0 or i == len(path) - 1:
        return 0
    frag = path[i + 1:]
    if len(frag) > 3:
        return 0
    for ch in frag:
        if ch not in _DIGITS:
            return 0
    v = int(frag)
    if v <= 0:
        return 0
    return v if v < MAX_GEN else MAX_GEN


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
    out = (idx, val, _generation_of(src), _generation_of(dst))
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
    if m <= 0:
        return out
    rows = []
    cols = []
    vals = []
    gen_rows = []
    gen_block = np.zeros((m, 2 * GEN_W), dtype=np.float32)
    for i in range(m):
        c = cmds[i]
        if not isinstance(c, str) or not c:
            continue
        idx, val, sgen, dgen = _compact(c)
        if idx.size:
            rows.append(np.full(idx.size, i, dtype=np.int64))
            cols.append(idx)
            vals.append(val)
        gen_block[i, :GEN_W] = GEN_TABLE[sgen]
        gen_block[i, GEN_W:] = GEN_TABLE[dgen]
        gen_rows.append(i)
    if rows:
        r = torch.from_numpy(np.concatenate(rows)).to(out.device)
        c = torch.from_numpy(np.concatenate(cols)).to(out.device)
        v = torch.from_numpy(np.concatenate(vals)).to(device=out.device, dtype=out.dtype)
        out.index_put_((r, c), v, accumulate=True)
    if gen_rows:
        ridx = torch.as_tensor(gen_rows, dtype=torch.long, device=out.device)
        gvals = torch.from_numpy(np.ascontiguousarray(gen_block[gen_rows])).to(
            device=out.device, dtype=out.dtype)
        gcols = torch.arange(SGEN_LO, DGEN_HI, dtype=torch.long, device=out.device)
        sel = out.index_select(0, ridx)
        sel.index_copy_(1, gcols, gvals)
        out.index_copy_(0, ridx, sel)
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
        tok[bi, 0:2 * n:2] = zc.to(tok.dtype)
        tok[bi, 1:2 * n:2] = zo.to(tok.dtype)
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, :2 * n] = False
        tgt[bi, :n] = zo.to(tgt.dtype)
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
    cmds = [
        "ls -1 /",
        "cat /tmp/w/cups/g1/a.dat",
        "mv /tmp/w/cups/g1/a.dat /tmp/w/cups/etc/b.cfg.1",
        "mv /tmp/w/cups/etc/b.cfg.1 /tmp/w/cups/etc/b.cfg.2",
        "cat /tmp/w/cups/etc/b.cfg.2",
        "ls /a",
    ]
    seq = [{"z_obs": torch.randn(6, D), "z_cmd": torch.randn(6, D), "cmds": cmds, "image": "x"}]
    b0 = collate(seq, device)
    p0 = net(b0["tok"], b0["types"], b0["key_pad"])[0][:, 0::2].clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=b1["tok"].device) * 100.0
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
Beat your parent's fitness of +0.0562 (r6-07-capacity-ledger-transport, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
