TASK: Maximize compositional depth in a shell world model: the paired within-genome difference between the model's next-observation pick under the native chain of silent file moves and its pick under a role-swapped chain over the same board.

OPERATOR: REWRITE — replace the mutable code wholesale with a genuinely different design. A rewrite that lands near the parent is a wasted slot.

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
  id                r6-01-slotconflict-deficit-scheduler
  its fitness       +0.0337   (this is YOUR target)
  its full genome (the diet grants you your parent's genome):
  objective           r12_antiretrieval_ring_negatives
  arch                r22_retrieval_composition_renderer
  optim               r18_spectral_capped_transition_readout
  target              identity
  batcher             r6_slotconflict_deficit_scheduler   params {"block_size": 4, "clique_gamma": 1.0, "conf_frac_max": 0.6, "neyman_beta": 1.0, "ramp_frac": 0.25}
  stream              baseline_interleave
  head                r18_transition_forwardmodel_consistency

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

PARENT'S EVAL FEEDBACK: comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].
comp_ca [withheld] n=89 (d3 60, d4+ 29); native [withheld] vs chance [withheld]; swapped: held [withheld], follows [withheld]; health [withheld].

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

--- r4_role_bound_path_atom_codes (axis stream)
import hashlib

import torch

from realenv import seq_worldmodel as M

NAME = "r4_role_bound_path_atom_codes"
DESCRIPTION = (
    "Interleave2 layout with a role-bound path-atom code written into the last 128 coordinates "
    "of every COMMAND token. Each command is parsed on its own into a verb/redirect shape and its "
    "non-flag path arguments, each argument is assigned the role src, dst or read (mv/cp take "
    "arg0/arg1; a > or >> target is the dst and the preceding argument the src; every other "
    "argument is a read), and each path contributes two deterministic blake2b-derived +/-1 "
    "Rademacher vectors, one for the full path string and one for its basename. Every atom is "
    "bundled twice: once multiplied elementwise by a fixed +/-1 role vector and once unbound, so "
    "the same path carries both a role-separated channel and a role-free identity channel. The "
    "coding is position-local, so it is prefix-causal and a command's token depends on nothing but "
    "that command's own string and its own standardized embedding. The first 640 embedding "
    "coordinates are passed through unchanged and OBSERVATION tokens are untouched."
)

D = M.D
CODE_DIMS = 128
CODE_OFF = D - CODE_DIMS

FULL_ROLE = 0.75
FULL_FREE = 0.55
BASE_ROLE = 0.22
BASE_FREE = 0.30
SHAPE_ROLE = 0.40

_ROLES = ("shape", "src", "dst", "read")
_SIGN_LIMIT = 60000
_CODE_LIMIT = 150000

_sign_cache = {}
_code_cache = {}


def _sign_vec(key):
    v = _sign_cache.get(key)
    if v is not None:
        return v
    digest = hashlib.blake2b(key.encode("utf-8", "replace"),
                             digest_size=CODE_DIMS // 8).digest()
    v = torch.tensor(
        [1.0 if (digest[i >> 3] >> (i & 7)) & 1 else -1.0 for i in range(CODE_DIMS)],
        dtype=torch.float32)
    if len(_sign_cache) >= _SIGN_LIMIT:
        _sign_cache.clear()
    _sign_cache[key] = v
    return v


def _role_vec(role):
    return _sign_vec("\x00role\x00" + role)


def _spaced(cmd):
    s = cmd.replace(">>", " \x01 ")
    s = s.replace(">", " \x02 ")
    s = s.replace("\x01", ">>").replace("\x02", ">")
    return s.replace("|", " | ")


def _parse(cmd):
    toks = _spaced(cmd).split()
    if not toks:
        return "", [], "", None
    verb = toks[0]
    pos = []
    redir_op = ""
    redir_tgt = None
    want_tgt = False
    for t in toks[1:]:
        if want_tgt:
            redir_tgt = t
            want_tgt = False
            continue
        if t == ">" or t == ">>":
            redir_op = t
            want_tgt = True
            continue
        if t == "|":
            continue
        if t.startswith("-"):
            continue
        pos.append(t)
    return verb, pos, redir_op, redir_tgt


def _basename(path):
    p = path[:-1] if len(path) > 1 and path.endswith("/") else path
    b = p.rsplit("/", 1)[-1]
    if not b or b == path:
        return None
    return b


def _slots(verb, pos, redir_tgt):
    if verb in ("mv", "cp", "ln") and len(pos) >= 2:
        return [(pos[0], "src"), (pos[1], "dst")] + [(p, "read") for p in pos[2:]]
    if redir_tgt is not None:
        out = [(redir_tgt, "dst")]
        if pos:
            out.append((pos[0], "src"))
            out += [(p, "read") for p in pos[1:]]
        return out
    return [(p, "read") for p in pos]


def _build_code(cmd):
    verb, pos, redir_op, redir_tgt = _parse(cmd)
    shape = verb + "|" + redir_op + "|" + str(len(pos)) + "|" + ("1" if redir_tgt else "0")
    vec = _role_vec("shape") * _sign_vec("\x00shape\x00" + shape) * SHAPE_ROLE
    for path, role in _slots(verb, pos, redir_tgt):
        rv = _role_vec(role)
        av = _sign_vec("\x00full\x00" + path)
        vec = vec + rv * av * FULL_ROLE + av * FULL_FREE
        base = _basename(path)
        if base is not None:
            bv = _sign_vec("\x00base\x00" + base)
            vec = vec + rv * bv * BASE_ROLE + bv * BASE_FREE
    return vec


def _cmd_code(cmd):
    key = cmd if isinstance(cmd, str) else str(cmd)
    v = _code_cache.get(key)
    if v is None:
        v = _build_code(key)
        if len(_code_cache) >= _CODE_LIMIT:
            _code_cache.clear()
        _code_cache[key] = v
    return v


def code_cmds(cmds, z_cmd):
    z = z_cmd if torch.is_tensor(z_cmd) else torch.as_tensor(z_cmd)
    out = z.detach().clone()
    n = out.shape[0]
    if n == 0:
        return out
    seq = list(cmds) if cmds is not None else []
    rows = [_cmd_code(seq[i]) if i < len(seq) else _cmd_code("") for i in range(n)]
    code = torch.stack(rows).to(device=out.device, dtype=out.dtype)
    out[:, CODE_OFF:] = code
    return out


def collate(batch, device):
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
        zc = code_cmds(s.get("cmds", []), s["z_cmd"])
        tok[bi, 0:2 * n:2] = zc[:n]
        tok[bi, 1:2 * n:2] = s["z_obs"][:n]
        types[bi, 1:2 * n:2] = 1
        key_pad[bi, 0:2 * n] = False
        tgt[bi, :n] = s["z_obs"][:n]
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
        cmd_pred = extract_cmd_pred(pred, b).cpu()
        cmd_h = h[:, 0::2].cpu()
        for bi, s in enumerate(chunk):
            n = s["z_obs"].shape[0]
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
    p0 = extract_cmd_pred(net(b0["tok"], b0["types"], b0["key_pad"])[0], b0).clone().cpu()
    b1 = collate(seq, device)
    b1["tok"][0, 7] = torch.randn(D, device=device) * 100.0
    p1 = extract_cmd_pred(net(b1["tok"], b1["types"], b1["key_pad"])[0], b1).cpu()
    chg = (p1 - p0).abs().amax(-1)[0]
    return bool((chg[:4] < 1e-4).all())


CUPS_LAYOUT = "interleave2"

STANDING RULES (every inventor, every round):
- NOVELTY OVER SAFETY — a safe tweak is a wasted slot; invent a genuinely different mechanism or a novel recombination of archived ideas. Commit to ONE best design.
- RETRY FAILED TRAITS — a design that scored low before may win in a changed context (recombined with a newer winner); if you retry one, argue what changed.
- LOOK OUTSIDE THE DOMAIN — search the literature beyond this problem's field and translate ONE concrete mechanism into code (equations, not metaphor).
- NEVER touch the eval, the metric, the splits, or any protected path — the harness re-checks structurally and a violation scores as a failed candidate.

Scoring trains one net per seed on a capability-pack data root of real shell trajectories and measures it on windows held out by IMAGE, so a mechanism only earns anything by transferring to systems it never trained on. Training is a fixed step budget on frozen encoder embeddings; a mechanism that cannot finish inside it is not ready, so profile speed as well as correctness. evolve/jail_data/train_sample.jsonl in this jail is real trajectories from the training split, verbatim: check any mechanical assumption about the data against it rather than inferring the answer from another impl's source. The observation a step carries is rendered from its exit code and output; realenv/seq_worldmodel.py collate shows how a trajectory becomes tokens. How the score cancels, which is worth understanding before you design against it: it is a PAIRED difference between the same board under the native chain of moves and under a chain in which two contents exchange their moves. A predictor keying only on WHICH LOCATION is being read sees the same read token in both arms, so it predicts identically and contributes exactly zero per window — which holds by construction while the command tokens outside the moves are the same in both arms, as they are for any stream that declares no code_cmds. Keying on WHERE IN THE MOVE ORDER a content sits does not cancel that way — it cancels only in expectation, and the scored slice is one frozen realization — so a positive number is not by itself evidence that a content was carried. What the objective asks for is the thing that survives both arms: carrying a particular content's identity through the chain of moves, so that a read returns what is actually there. You cannot run the real harness from here — write the impl so it is correct by construction, and state any performance claim as unmeasured rather than extrapolating from a miniature run, because miniature probes in this project have inverted rank in both directions.

YOUR OBJECTIVE
Beat your parent's fitness of +0.0337 (r6-01-slotconflict-deficit-scheduler, full budget, runpod-4090, inner split).
The unmodified baseline scores +0.0112 in the same measurement — a floor, not a target.
No other candidate's score is shown to you, by design: parents are sampled by fitness, so beating YOUR parent is the whole bar. Report which axes you changed (axes_changed) with your proposal.
