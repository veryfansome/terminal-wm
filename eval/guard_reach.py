"""The reachability gate: can this candidate's parameters affect anything?

Two questions per parameter, each answered by one real backward pass.

  untrained -- no gradient reaches it from the training loss the genome actually uses
               (its objective, its target, its head's aux), on a batch of realistic
               command sequences.
  inert     -- no gradient reaches it from the prediction at the scored read position,
               on the odd [cmd, obs, ..., cmd] layout the instrument actually builds.

Neither alone is a defect. An aux-only head parameter is inert by design and still shapes
the trunk; a parameter can be untrained on one batch and trained on another. A parameter
that is BOTH cannot influence the score and cannot learn to: it is dead weight, and a
candidate whose novel mechanism is dead scores its other changes while the archive records
the result under the mechanism's name.

Parameters are perturbed off their initial values first. This lineage's house style is to be
identical to the parent at init behind a zero-initialised readout, so a gradient taken at
init would report a live mechanism as dead.
"""
import json
import sys

import torch

from evolve import genome as G
from realenv import seq_worldmodel as M

D = M.D
EPS = 0.05
SEED = 20260810

CMD_SETS = [
    ["ls -1 /tmp/w/cups/opt/app/conf",
     "cat /tmp/w/cups/opt/app/conf/settings.yaml",
     "cat /tmp/w/cups/var/lib/misc/state.db",
     "cat /tmp/w/cups/srv/data/index.json",
     "mv /tmp/w/cups/opt/app/conf/settings.yaml /tmp/w/cups/var/lib/misc/state.db.1",
     "mv /tmp/w/cups/var/lib/misc/state.db /tmp/w/cups/srv/data/index.json.2",
     "mv /tmp/w/cups/var/lib/misc/state.db.1 /tmp/w/cups/srv/data/index.json.3",
     "cat /tmp/w/cups/srv/data/index.json.3"],
    ["ls -la /tmp/w/cups/var/lib/misc",
     "cat /tmp/w/cups/var/log/app.log",
     "cat /tmp/w/cups/etc/app/app.conf",
     "mv /tmp/w/cups/var/log/app.log /tmp/w/cups/etc/app/app.conf.1",
     "mv /tmp/w/cups/etc/app/app.conf /tmp/w/cups/var/log/app.log.2",
     "cat /tmp/w/cups/etc/app/app.conf.1"],
    ["cat /tmp/w/cvar/g0/s0.dat",
     "cat /tmp/w/cvar/g0/s1.dat",
     "cat /tmp/w/cvar/g0/s0.dat >> /tmp/w/cvar/g0/acc.dat",
     "mv /tmp/w/cvar/g0/acc.dat /tmp/w/cvar/g0/s2.dat",
     "cat /tmp/w/cvar/g0/s2.dat"],
]


def _seqs(gen):
    out = []
    for k, cmds in enumerate(CMD_SETS):
        n = len(cmds)
        out.append({"z_obs": torch.randn(n, D, generator=gen),
                    "z_cmd": torch.randn(n, D, generator=gen),
                    "cmds": list(cmds), "image": f"img{k}"})
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
    scored.sum().backward()
    inert = set(_zero_grad_names(net))

    total = sum(1 for _, p in net.named_parameters() if p.requires_grad)
    dead = sorted(untrained & inert)
    return {"ok": not dead and aux_ran is True,
            "n_params": total, "n_untrained": len(untrained), "n_inert": len(inert),
            "n_dead": len(dead), "dead": dead[:20], "aux_loss_ran": aux_ran,
            "loss": float(loss.detach())}


def main(genome_path):
    r = analyze(genome_path)
    print(json.dumps(r))
    if r["aux_loss_ran"] is not True:
        print(f"reachability gate FAILED: head.aux_loss raised on a well-formed batch: "
              f"{r['aux_loss_ran']}", file=sys.stderr)
        return 1
    if r["dead"]:
        print(f"reachability gate FAILED: {r['n_dead']} of {r['n_params']} parameters receive no "
              f"gradient from the training loss AND no gradient from the prediction at the scored "
              f"read position. They cannot influence the score and cannot learn to. "
              f"First: {r['dead'][:8]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as e:
        print(f"reachability gate ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
