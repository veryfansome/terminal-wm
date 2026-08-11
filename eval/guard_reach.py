"""The reachability gate: can this candidate's parameters affect anything?

Two questions per parameter, each answered by one real backward pass.

  untrained -- no gradient reaches it from the training loss the genome actually uses
               (its objective, its target, its head's aux).
  inert     -- no gradient reaches it from the prediction at the scored read position,
               on the odd [cmd, obs, ..., cmd] layout the instrument actually builds.

Neither alone is a defect. An aux-only head parameter is inert by design and still shapes the
trunk. A parameter that is BOTH cannot influence the score and cannot learn to: it is dead
weight, and a candidate whose novel mechanism is dead scores its other changes while the archive
records the result under the mechanism's name.

Parameters are perturbed off their initial values first. This lineage's house style is to be
identical to the parent at init behind a zero-initialised readout, so a gradient taken at init
would report a live mechanism as dead.

The fixture has to be STRUCTURED, not random. Heads in this lineage mine their supervision by
cosine over command embeddings above a threshold (r18 uses 0.60), which random 768-d vectors
never clear -- with randn embeddings every head's aux is exactly 0.0 and the untrained arm is
measured with heads switched off. So embeddings are built from the command strings: a direction
per path, per parent directory and per verb, summed and normalised, which makes two commands
naming the same path similar and two unrelated commands not. Observations follow a simulated
board, so a read returns the content that is actually there and every move renders to the same
constant -- as every one of the pack's move steps does. Rows run to the real horizon; pos_emb is
Embedding(64), so 32 steps is the hard ceiling.
"""
import hashlib
import json
import sys

import torch

from evolve import genome as G
from realenv import seq_worldmodel as M

D = M.D
EPS = 0.05
SEED = 20260810
MAX_STEPS = 30


SCALE = D ** 0.5


def _dir(key):
    """A fixed direction at STANDARDIZED magnitude. Cached embeddings are standardized to roughly
    unit variance per dimension, so their norm is about sqrt(D). Unit-norm directions would keep
    every cosine intact and still starve the miners, whose second condition is a squared-distance
    floor on observations (r18: 0.25) that unit vectors miss by three orders of magnitude."""
    h = hashlib.blake2b(key.encode(), digest_size=8).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h, "big") % (2 ** 31))
    v = torch.randn(D, generator=g)
    return SCALE * v / v.norm()


def _cmd_vec(cmd):
    parts = cmd.split()
    v = 0.6 * _dir("verb:" + parts[0])
    for a in parts[1:]:
        if a.startswith("-"):
            continue
        v = v + 3.0 * _dir("path:" + a)
        if "/" in a:
            v = v + 1.0 * _dir("dir:" + a.rsplit("/", 1)[0])
    return SCALE * v / v.norm()


def _board(root, n_files, n_hops, want_steps):
    """One trajectory over a small board: expose every content, then thread a chain of moves in
    which the destination of one hop is the source of a later one, reading locations back along
    the way. Threading matters -- it is what makes commands share a path, which is the structure
    every mining head keys on and the structure the scored windows are built from."""
    slots = [f"{root}/s{i}.dat" for i in range(n_files)]
    loc = {p: f"content{i}" for i, p in enumerate(slots)}
    cmds, obs = [f"ls -1 {root}"], ["listing"]
    for p in slots:
        cmds.append(f"cat {p}")
        obs.append(loc[p])
    hop = 0
    while len(cmds) < want_steps - 2 and hop < n_hops:
        src = slots[hop % n_files]
        dst = slots[(hop + 1) % n_files]
        if src in loc:
            content = loc.pop(src)
            evicted = loc.get(dst)
            loc[dst] = content
            cmds.append(f"mv {src} {dst}")
            obs.append("empty")
            if evicted is not None and len(cmds) < want_steps - 2:
                spill = f"{root}/s{(hop + 2) % n_files}.dat"
                loc[spill] = evicted
                cmds.append(f"mv {dst} {spill}")
                obs.append("empty")
        hop += 1
        if hop % 2 == 0 and len(cmds) < want_steps - 2:
            probe = slots[hop % n_files]
            cmds.append(f"cat {probe}")
            obs.append(loc.get(probe, "missing"))
    while len(cmds) < want_steps:
        probe = slots[len(cmds) % n_files]
        cmds.append(f"cat {probe}")
        obs.append(loc.get(probe, "missing"))
    return cmds[:want_steps], obs[:want_steps]


BOARDS = [("/tmp/w/cups/opt/app/conf", 3, 9, 28),
          ("/tmp/w/cups/var/lib/misc", 5, 12, 30),
          ("/tmp/w/cvar/g0", 4, 10, 26)]


def _seqs(gen):
    out = []
    for k, (root, n_files, n_hops, n_steps) in enumerate(BOARDS):
        cmds, obs_keys = _board(root, n_files, n_hops, n_steps)
        n = len(cmds)
        z_cmd = torch.stack([_cmd_vec(c) for c in cmds])
        z_obs = torch.stack([_dir("obs:" + o) for o in obs_keys])
        z_cmd = z_cmd + 0.05 * torch.randn(n, D, generator=gen)
        out.append({"z_obs": z_obs, "z_cmd": z_cmd, "cmds": list(cmds), "image": f"img{k}"})
    return out


def _cups_layout(seqs, device):
    rs = [len(s["cmds"]) - 1 for s in seqs]
    L = 2 * max(rs) + 1
    B = len(seqs)
    tok = torch.zeros(B, L, D)
    types = torch.zeros(B, L, dtype=torch.long)
    valid = torch.zeros(B, L, dtype=torch.bool)
    rpos = torch.zeros(B, dtype=torch.long)
    for i, s in enumerate(seqs):
        r = rs[i]
        for j in range(r):
            tok[i, 2 * j] = s["z_cmd"][j]; valid[i, 2 * j] = True
            tok[i, 2 * j + 1] = s["z_obs"][j]; types[i, 2 * j + 1] = 1
            valid[i, 2 * j + 1] = True
        tok[i, 2 * r] = s["z_cmd"][r]; valid[i, 2 * r] = True
        rpos[i] = 2 * r
    return (tok.to(device), types.to(device), (~valid).to(device), rpos.to(device))


def _advance_ramps(head_state):
    """Push any warmup counter in the head's state past its ramp before the measured backward.

    Auxiliary losses in this lineage ramp in over a few hundred steps, so at step one the aux is
    damped by four or five orders of magnitude and the gradient it contributes is a rounding
    error. The advance is bounded by the head's own declared ramp rather than set to a huge
    number: a head that sizes or indexes a schedule by its counter would raise on a fabricated
    step, and the gate would then report a well-formed batch as the cause."""
    if not isinstance(head_state, dict):
        return
    ramps = [int(v) for k, v in head_state.items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)
             and "ramp" in k.lower() and 0 < float(v) < 10 ** 6]
    target = max(ramps) + 1 if ramps else 1000
    for k, v in list(head_state.items()):
        if isinstance(v, int) and not isinstance(v, bool) and "step" in k.lower():
            head_state[k] = target


def _zero_grad_names(net):
    out = []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not bool(p.grad.abs().max() > 0):
            out.append(n)
    return out


def analyze(genome_path):
    gen_cfg = json.load(open(genome_path))
    G.validate(gen_cfg)
    loss_fn = G.load_objective(gen_cfg)
    target_mod = G.load_target(gen_cfg)
    stream = G.load_stream(gen_cfg)
    head, head_p = G.load_head(gen_cfg)
    device = M.pick_device()

    torch.manual_seed(SEED)
    build, aparams = G.load_arch(gen_cfg)
    net = build(**aparams)
    arch_owned = set(dict(net.named_parameters()))
    if getattr(target_mod, "LEARNED", False):
        net.target_module = target_mod.make(D)
    head_state = head.wrap(net, D, **(head_p or {})) if head is not None else None
    net = net.to(device)
    net.train()

    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(EPS * torch.randn(p.shape, generator=g).to(p.device, p.dtype))

    seqs = _seqs(g)
    b = stream.collate(seqs, device)
    pred_full, _ = net(b["tok"], b["types"], b["key_pad"])
    cmd_pred = stream.extract_cmd_pred(pred_full, b)
    tgt_full = b["tgt"]
    prev_full = torch.cat([torch.zeros_like(tgt_full[:, :1]), tgt_full[:, :-1]], dim=1)
    m = b["cmd_mask"]
    pred, tgt, prev = cmd_pred[m], tgt_full[m], prev_full[m]
    tmod = getattr(net, "target_module", None)
    if tmod is not None:
        _t, _reg = tmod.make_target(tgt, prev), tmod.reg()
    else:
        _t, _reg = target_mod.make_target(tgt, prev), 0.0
    wants_ctx = getattr(sys.modules.get(getattr(loss_fn, "__module__", None)),
                        "WANTS_CTX", False)
    if wants_ctx:
        loss = loss_fn(pred, _t, {"cmd": stream.extract_cmd_input(b)[m], "prev": prev}) + _reg
    else:
        loss = loss_fn(pred, _t) + _reg
    aux_ran = True
    if head is not None:
        _advance_ramps(head_state)
        try:
            loss = loss + head.aux_loss(head_state, b, net, device)
        except Exception as e:
            aux_ran = f"{type(e).__name__}: {e}"
    net.zero_grad(set_to_none=True)
    loss.backward()
    untrained = set(_zero_grad_names(net))

    net.zero_grad(set_to_none=True)
    net.eval()
    tok, types, key_pad, rpos = _cups_layout(seqs, device)
    p_full, _ = net(tok, types, key_pad)
    scored = p_full[torch.arange(p_full.shape[0], device=device), rpos]
    # A random projection rather than .sum(): an architecture whose last operation subtracts the
    # mean over the feature axis has zero gradient of the SUM with respect to everything, which
    # would report the whole net inert.
    u = torch.randn(scored.shape, generator=torch.Generator().manual_seed(SEED + 1)).to(scored)
    (scored * u).sum().backward()
    inert = set(_zero_grad_names(net))

    total = sum(1 for _, p in net.named_parameters() if p.requires_grad)
    dead = sorted(untrained & inert)
    arch_inert = sorted(arch_owned & inert)
    return {"ok": not dead and not arch_inert and aux_ran is True and len(inert) < total,
            "n_params": total, "n_untrained": len(untrained), "n_inert": len(inert),
            "n_dead": len(dead), "dead": dead[:20],
            "n_arch_inert": len(arch_inert), "arch_inert": arch_inert[:20],
            "aux_loss_ran": aux_ran, "loss": float(loss.detach())}


def reachable(genome_path):
    """(ok, why) for callers that are already inside a scoring run. The engine's guardrail phase
    does not run on the pack lane -- cloud/runner.py invokes eval.adapter directly -- so the
    adapter calls this too, and the gate is paid on both paths."""
    r = analyze(genome_path)
    if r["aux_loss_ran"] is not True:
        return False, (f"head.aux_loss raised on a well-formed batch: {r['aux_loss_ran']}")
    if r["n_inert"] == r["n_params"]:
        return False, ("no parameter shows gradient from the scored position — the probe failed, "
                       "rather than the net being dead")
    if r["dead"]:
        return False, (f"{r['n_dead']} of {r['n_params']} parameters receive no gradient from the "
                       f"training loss AND none from the prediction at the scored read position: "
                       f"{r['dead'][:8]}")
    if r["arch_inert"]:
        return False, (f"{r['n_arch_inert']} ARCHITECTURE parameters cannot affect the prediction "
                       f"at the scored read position: {r['arch_inert'][:8]}")
    return True, ""


def main(genome_path):
    r = analyze(genome_path)
    print(json.dumps(r))
    if r["aux_loss_ran"] is not True:
        print(f"reachability gate FAILED: head.aux_loss raised on a well-formed batch: "
              f"{r['aux_loss_ran']}", file=sys.stderr)
        return 1
    if r["n_inert"] == r["n_params"]:
        print("reachability gate INCONCLUSIVE: no parameter shows gradient from the scored "
              "position, which means the probe itself failed rather than that the net is dead",
              file=sys.stderr)
        return 1
    if r["dead"]:
        print(f"reachability gate FAILED: {r['n_dead']} of {r['n_params']} parameters receive no "
              f"gradient from the training loss AND no gradient from the prediction at the scored "
              f"read position. They cannot influence the score and cannot learn to. "
              f"First: {r['dead'][:8]}", file=sys.stderr)
        return 1
    if r["arch_inert"]:
        print(f"reachability gate FAILED: {r['n_arch_inert']} ARCHITECTURE parameters cannot "
              f"affect the prediction at the scored read position. An arch parameter exists to "
              f"shape predictions; one that cannot is a branch whose trigger the scored layout "
              f"never presents. A head's auxiliary parameters are exempt -- these are not. "
              f"First: {r['arch_inert'][:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f"reachability gate ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
