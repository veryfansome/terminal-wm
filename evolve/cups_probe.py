"""cups_probe — the cups-pack instrument (research/cups-pack-design.md v2 §2–§3).

Scoring is an N-way FORCED-FOIL classification (the v2 review's foil-pool critical): the model's
prediction at the test-read position is scored by nearest-neighbor among the window's OWN N
exposure-obs embeddings — chance = 1/N by construction, identical for the WM and every arm.

Arms (all computed, no training — the §3 ceiling, pinned by the 2026-08-06 review):
  at_name      — predict exposure[name_idx] (C1's success mode; chance 1/N in treatment by the
                 stem⊥chain construction — the transpose oracle)
  copy_prev    — nearest exposure to the previous obs embedding
  h1 / gauntlet_H / h_last / h_first — the analytic heuristic gauntlet: a backward trace of the
                 test path through the recorded mv (src,dst) pairs capped at k hops solves
                 exactly the depth<=k windows (verified realizable, review round 1), so
                 h_k = truth if depth<=k else the at_name fallback; h_last = the exposure whose
                 slot is the LAST slot-src mentioned across the mv commands; h_first = the
                 exposure whose slot is the FIRST mv's src (the depth-0 positional marker the
                 review's critical finding measured at ~0.90 on deep windows pre-decorrelation).
  elim         — the elimination oracle (§3, definition pinned 2026-08-06): eliminate exposures
                 that never moved (their first-move srcs are textually recognizable slot paths),
                 then guess uniformly over the movers — fractional per-window value 1/|movers|.
  centroid     — nearest exposure to the mean of the window's N exposure embeddings (the
                 exposure-cluster-copier failure mode).
Probes:
  exposure_swap — swap the exposure OBS with a same-N donor's; scored on the 2x2 bank
                  {own,donor} x {routed,name}: PASS = nearest is donor-routed (content attribution).
  alt_chain     — re-encode the window with a DIFFERENT valid chain over the SAME board (mv command
                  strings synthesized in the window's own path vocabulary, encoded at probe time via
                  the perception impl): a tracker's prediction follows the alternative routed answer
                  (operator attribution). Requires --encoder support in the caller.
Stratified reporting: per (N, depth) cells + pooled marginals; the gate slices are assembled by the
caller per the frozen prereg. Never selection.
"""
import json
import pathlib

import torch
import torch.nn.functional as F

from evolve import harness as H
from evolve.splits import split_val
from realenv import seq_worldmodel as M

D = M.D
SWITCH_MIN = 5      # min windows for an (N,depth) cell to switch independently (§6 open number)


def harvest_cups_windows(seqs):
    """Windows from encoded seqs (IW.load_windows attaches raw steps): one per cups_test step,
    carrying the exposure slots' step indices (by cups_slot), the mv step indices + parsed
    src/dst, the ctx steps, and the stamped metadata."""
    wins = []
    for si, s in enumerate(seqs):
        steps = s["steps"]
        for r, st in enumerate(steps):
            meta = st.get("meta", {})
            if meta.get("role") != "cups_test":
                continue
            cups = meta["cups"]
            expose, mvs, ctx = {}, [], []
            for t in range(r):
                mt = steps[t].get("meta", {})
                if mt.get("arm") != "cups":
                    continue
                if mt.get("role") == "cups_expose":
                    expose[mt["cups_slot"]] = t
                elif mt.get("role") == "cups_ctx":
                    ctx.append(t)
                elif "cups_mv" in mt:
                    parts = steps[t]["cmd"].split()
                    mvs.append((t, parts[1], parts[2]))          # (step, src, dst)
            n = cups["N"]
            assert len(expose) == n, f"window {si}:{r} missing exposures"
            # slot paths in slot-index order (from the exposure commands), then a TEXTUAL
            # chain replay — the one authority for the positional-marker fields the ceiling
            # arms need (h_first / h_last / h_lastmv / elim / deepest) AND the stamp-equality
            # asserts the design promises (review round 2: stamps were taken on faith).
            slots = [steps[expose[k]]["cmd"].split()[1] for k in range(n)]
            slot_of = {p: k for k, p in enumerate(slots)}
            loc = {k: slots[k] for k in range(n)}
            depths = {k: 0 for k in range(n)}
            last_mover = None
            for (_, msrc, mdst) in mvs:
                c = next(k for k, v in loc.items() if v == msrc)
                loc[c] = mdst
                depths[c] += 1
                last_mover = c
            movers = sorted(k for k, d in depths.items() if d > 0)
            if cups.get("movers") is not None:
                assert sorted(cups["movers"]) == movers, (
                    f"{si}:{r} mover stamp {cups['movers']} != textual {movers}")
            if cups.get("depths") is not None:
                assert {int(k): v for k, v in cups["depths"].items() if v} == \
                       {k: v for k, v in depths.items() if v}, f"{si}:{r} depth-stamp mismatch"
            assert depths[cups["routed_idx"]] == cups["depth"], f"{si}:{r} depth mismatch"
            dmax = max(d for d in depths.values() if d > 0)
            first_src = slot_of[mvs[0][1]] if mvs else None
            last_src = next((slot_of[msrc] for (_, msrc, _) in reversed(mvs)
                             if msrc in slot_of), None)
            wins.append({"id": f"{s['image']}:{si}:{r}", "si": si, "r": r,
                         "image": s["image"], "N": n, "R": cups["R"],
                         "depth": cups["depth"], "routed": cups["routed_idx"],
                         "name": cups["name_idx"], "expose": expose, "mvs": mvs, "ctx": ctx,
                         "slots": slots, "first_src": first_src, "last_src": last_src,
                         "last_mover": last_mover, "movers": movers,
                         "style": cups.get("style", "core"),
                         "deepest": sorted(k for k in movers if depths[k] == dmax)})
    return wins


def build_cups_layout(wins, seqs):
    """Native interleaved [cmd_0,obs_0,...,cmd_r] per window, pred read at the test-cmd position
    (obs_r withheld = the target). Also gathers the per-window exposure candidate matrix [N,D]
    (the forced-foil bank) and z_prev."""
    B = len(wins)
    Lmax = 2 * max(w["r"] for w in wins) + 1
    tok = torch.zeros(B, Lmax, D)
    types = torch.zeros(B, Lmax, dtype=torch.long)
    valid = torch.zeros(B, Lmax, dtype=torch.bool)
    rpos = torch.zeros(B, dtype=torch.long)
    z_prev = torch.zeros(B, D)
    cands = []                                           # per-window [N,D] exposure obs embeddings
    for i, w in enumerate(wins):
        s = seqs[w["si"]]
        r = w["r"]
        for j in range(r):
            tok[i, 2 * j] = s["z_cmd"][j]; valid[i, 2 * j] = True
            tok[i, 2 * j + 1] = s["z_obs"][j]; types[i, 2 * j + 1] = 1; valid[i, 2 * j + 1] = True
        tok[i, 2 * r] = s["z_cmd"][r]; valid[i, 2 * r] = True
        rpos[i] = 2 * r
        z_prev[i] = s["z_obs"][r - 1]
        cands.append(torch.stack([s["z_obs"][w["expose"][k]] for k in range(w["N"])]))
    return {"wins": wins, "tok": tok, "types": types, "key_pad": ~valid, "rpos": rpos,
            "z_prev": z_prev, "cands": cands}


@torch.no_grad()
def _fwd_pred(net, tok, types, key_pad, rpos, device, bs=512):
    outs = []
    net = net.to(device).eval()
    for i in range(0, tok.shape[0], bs):
        sl = slice(i, min(i + bs, tok.shape[0]))
        p, _ = net(tok[sl].to(device), types[sl].to(device), key_pad[sl].to(device))
        outs.append(p[torch.arange(p.shape[0], device=device), rpos[sl].to(device)].cpu())
    return torch.cat(outs)


def _nway(pred_obs, ctx):
    """Per-window N-way pick: nearest exposure candidate (squared-L2, the eval's metric)."""
    picks = []
    for i, c in enumerate(ctx["cands"]):
        d = ((c - pred_obs[i].unsqueeze(0)) ** 2).mean(-1)
        picks.append(int(d.argmin()))
    return picks


def measure(net, ctx, target_mod, device, gauntlet_h=2, ceiling_table=None):
    """The capability measurement: per-window N-way picks for the WM + the computed arms,
    stratified by (N, depth). Returns per-window rows + stratified aggregates."""
    wins = ctx["wins"]
    pred = _fwd_pred(net, ctx["tok"], ctx["types"], ctx["key_pad"], ctx["rpos"], device)
    pred_obs = target_mod.to_obs(pred, ctx["z_prev"]) if target_mod is not None else pred
    wm_pick = _nway(pred_obs, ctx)
    cp_pick = _nway(ctx["z_prev"], ctx)
    ARMS = ("at_name", "copy_prev", "gauntlet", "h1", "h_first", "h_last", "h_lastmv",
            "elim", "deepest", "centroid")
    rows = []
    for i, w in enumerate(wins):
        truth, name = w["routed"], w["name"]
        cent = ctx["cands"][i].mean(0)
        cent_pick = int(((ctx["cands"][i] - cent.unsqueeze(0)) ** 2).mean(-1).argmin())
        rows.append({"id": w["id"], "N": w["N"], "R": w["R"], "depth": w["depth"],
                     "style": w["style"], "pick": wm_pick[i], "m": len(w["movers"]),
                     "routed": truth, "name": name,
                     "wm": int(wm_pick[i] == truth),
                     "at_name": int(name == truth),
                     "copy_prev": int(cp_pick[i] == truth),
                     "gauntlet": 1 if w["depth"] <= gauntlet_h else int(name == truth),
                     "h1": 1 if w["depth"] <= 1 else int(name == truth),
                     "h_first": int(w["first_src"] == truth),
                     "h_last": int(w["last_src"] == truth) if w["last_src"] is not None
                               else int(name == truth),
                     "h_lastmv": int(w["last_mover"] == truth),
                     "elim": 1.0 / max(1, len(w["movers"])),
                     "deepest": (1.0 / len(w["deepest"])) if truth in w["deepest"] else 0.0,
                     "centroid": int(cent_pick == truth),
                     "chance": 1.0 / w["N"]})
    def agg(sel):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return None
        out = {k: round(sum(r[k] for r in rs) / len(rs), 4)
               for k in ("wm",) + ARMS + ("chance",)}
        out["n"] = len(rs)
        out["arm_max"] = round(max(out[a] for a in ARMS), 4)
        # the SWITCHING ceiling (review round 2: max-of-means is not sound — an adversary may
        # pick a different arm per window from OBSERVABLE features): partition the slice by
        # the observable (N, depth) cell, take the best arm-mean per cell, mass-weight.
        # Cells below SWITCH_MIN windows pool into their per-N marginal (then a remainder
        # group) before switching — review round 3: a singleton cell's best-mean IS the
        # per-row oracle max, the over-strong ceiling round 2 prohibited. Dominates arm_max
        # by construction; this is the gate-bearing ceiling.
        cells, smalls, groups = {}, {}, []
        for r_ in rs:
            cells.setdefault((r_["N"], r_["depth"]), []).append(r_)
        for key, cell in cells.items():
            (groups.append(cell) if len(cell) >= SWITCH_MIN
             else smalls.setdefault(key[0], []).append(cell))
        rest = []
        for nkey, cl in smalls.items():
            pooled = [r_ for cell in cl for r_ in cell]
            (groups.append(pooled) if len(pooled) >= SWITCH_MIN else rest.extend(pooled))
        if rest:
            groups.append(rest)
        sw = sum(len(g) * max(sum(r_[a] for r_ in g) / len(g) for a in ARMS)
                 for g in groups) / len(rs)
        out["switch_max"] = round(sw, 4)
        # CROSS-FIT switching ceiling (freeze review 2026-08-07, critical C1): the in-sample
        # per-group best-mean is max-of-means UPWARD-biased (~+0.035 at n=142), silently
        # eating the gate band. Split each group in half by a seeded shuffle; choose the arm
        # on one half, EVALUATE it on the other, both directions, eval-mass-weighted —
        # unbiased for the true switching ceiling. Gate-bearing margin uses THIS; the
        # in-sample switch_max stays reported as the conservative upper bound.
        import random as _r
        rng = _r.Random(20260807)
        xf_num = 0.0
        for g in groups:
            if len(g) < 2:
                # a lone remainder window: choose its arm out-of-sample (on the slice minus
                # the window) — never on itself
                others = [r_ for r_ in rs if r_ is not g[0]]
                best = max(ARMS, key=lambda a: sum(r_[a] for r_ in others) / len(others)) \
                    if others else "at_name"
                xf_num += g[0][best]
                continue
            gg = list(g)
            rng.shuffle(gg)
            half = len(gg) // 2
            for pick_h, eval_h in ((gg[:half], gg[half:]), (gg[half:], gg[:half])):
                best = max(ARMS, key=lambda a: sum(r_[a] for r_ in pick_h) / len(pick_h))
                xf_num += sum(r_[best] for r_ in eval_h)
        out["switch_max_xfit"] = round(xf_num / len(rs), 4)
        # the FROZEN-TABLE ceiling (freeze review C1/F1, final form): an exact analytic
        # per-(N,depth)-cell population ceiling computed offline from the planner at the
        # frozen knobs (benchmarks/cupsA_ceiling_table.json) applied to the REALIZED cell
        # masses — the gate margin then carries wm sampling noise ONLY (no in-sample
        # max-of-means bias, no cross-fit pooling gap). A realized cell missing from the
        # table fails loud. Gate-bearing when a table is supplied; the estimated ceilings
        # stay reported.
        if ceiling_table is not None:
            num = 0.0
            for r_ in rs:
                key = f"{r_['N']},{r_['depth']},{r_['m']},{r_['R']}"
                assert key in ceiling_table, f"cell {key} missing from the ceiling table"
                num += ceiling_table[key]
            out["ceiling_frozen"] = round(num / len(rs), 4)
            out["margin_frozen"] = round(out["wm"] - out["ceiling_frozen"], 4)
        out["margin"] = round(out["wm"] - out["switch_max_xfit"], 4)
        out["margin_insample"] = round(out["wm"] - out["switch_max"], 4)
        out["margin_armmax"] = round(out["wm"] - out["arm_max"], 4)
        return out
    strata = {}
    for n in sorted({w["N"] for w in wins}):
        for dbkt in ("d1", "d2", "d3plus"):
            lo, hi = {"d1": (1, 1), "d2": (2, 2), "d3plus": (3, 99)}[dbkt]
            a = agg(lambda r, n=n, lo=lo, hi=hi: r["N"] == n and lo <= r["depth"] <= hi)
            if a:
                strata[f"N{n}_{dbkt}"] = a
    return {"rows": rows,
            # the raw prediction bank, for callers computing anti-degeneracy diagnostics
            # (norm / angular dispersion) without paying a second forward. In-process only —
            # never serialize it.
            "pred_obs": pred_obs,
            "pooled": agg(lambda r: True),
            # the §3 pre-designated PRIMARY slice, verbatim: N in {4,5}, relay depth >= 2
            # (round 1: the depth>gauntlet_h narrowing was a prereg mismatch; round 2: the
            # explicit N-set replaces the open-ended >=4, and the gate ceiling is switch_max,
            # under which h_H saturates the depth<=H CELLS — the primary margin is genuinely
            # earnable only where the analytic family runs out).
            "deep": agg(lambda r: r["N"] in (4, 5) and r["depth"] >= 2),
            # the earnable-stratum aggregate the §6-F gate references (freeze review C4: the
            # instrument previously had no pooled N-in-{4,5} d>=3 slice)
            "d3plus": agg(lambda r: r["N"] in (4, 5) and r["depth"] >= 3),
            # the EARNABLE slice (freeze review, final gate design): the table cells where a
            # non-tracking strategy is NOT saturated (ceiling < 0.99 — knob-robust rule);
            # saturated cells contribute only dilution to a margin. Gate-bearing = the CORE
            # style earnable slice with the frozen-table margin. None when no table given.
            "earnable_core": agg(lambda r: ceiling_table is not None
                                 and ceiling_table.get(
                                     f"{r['N']},{r['depth']},{r['m']},{r['R']}", 1.0) < 0.99
                                 and r["N"] in (4, 5) and r["style"] == "core"),
            "earnable_all": agg(lambda r: ceiling_table is not None
                                and ceiling_table.get(
                                    f"{r['N']},{r['depth']},{r['m']},{r['R']}", 1.0) < 0.99
                                and r["N"] in (4, 5)),
            "d3plus_core": agg(lambda r: r["N"] in (4, 5) and r["depth"] >= 3
                               and r["style"] == "core"),
            # style split (census freeze): the GATE-BEARING slice is deep_style_core — the
            # model trains on core-style boards only, so held-out-style windows measure style
            # TRANSFER, reported separately (freeze review F4: mixing them into the gate
            # conflated tracking with style generalization)
            "deep_style_core": agg(lambda r: r["N"] in (4, 5) and r["depth"] >= 2
                                   and r["style"] == "core"),
            "deep_style_heldout": agg(lambda r: r["N"] in (4, 5) and r["depth"] >= 2
                                      and r["style"] == "heldout"),
            "per_N": {f"N{n}": agg(lambda r, n=n: r["N"] == n)
                      for n in sorted({w["N"] for w in wins})},
            "per_depth": {f"d{d}": agg(lambda r, d=d: r["depth"] == d)
                          for d in sorted({w["depth"] for w in wins})},
            "strata": strata, "gauntlet_h": gauntlet_h}


def exposure_swap(net, ctx, target_mod, device, seed=20260806, ceiling_table=None):
    """Probe 2 (content attribution): swap the exposure OBS embeddings with a same-N donor's;
    score on the 2x2 bank {own,donor} x {routed,name}. PASS-rate = fraction picking donor-routed."""
    wins = ctx["wins"]
    N_w = len(wins)
    perm = torch.randperm(N_w, generator=torch.Generator().manual_seed(seed)).tolist()
    donors, tok2 = [], ctx["tok"].clone()
    for i, w in enumerate(wins):
        donor = next((perm[(i + k) % N_w] for k in range(1, N_w)
                      if wins[perm[(i + k) % N_w]]["N"] == w["N"]
                      and perm[(i + k) % N_w] != i), None)
        donors.append(donor)
        if donor is None:
            continue
        dw = wins[donor]
        ds = ctx["cands"][donor]
        for k in range(w["N"]):
            tok2[i, 2 * w["expose"][k] + 1] = ds[k]
    pred = _fwd_pred(net, tok2, ctx["types"], ctx["key_pad"], ctx["rpos"], device)
    pred_obs = target_mod.to_obs(pred, ctx["z_prev"]) if target_mod is not None else pred
    cnt = {k: 0 for k in ("follow", "own_r", "own_n", "don_n", "don_o", "m")}
    dcnt = dict(cnt)                       # core deep slice (reported)
    ecnt = dict(cnt)                       # core EARNABLE slice — gate-bearing (freeze v3:
    diag = 0                               # deep_core is ~43% 2-hop-solvable; earnable is 0%)
    for i, w in enumerate(wins):
        if donors[i] is None:
            continue
        if w["routed"] == w["name"]:
            # diagonal window: the {routed, name} bank columns coincide, so argmin's first-index
            # tie-break would credit a purely name-keyed model with 'follow' (review 2026-08-06).
            # Uninformative for content-vs-name attribution — excluded, counted.
            diag += 1
            continue
        own, don = ctx["cands"][i], ctx["cands"][donors[i]]
        # FULL-DONOR bank (binding-round V4-1, dated §2 amendment): the 2x2 bank FORCED an
        # arm-mimic's wrong-slot predictions onto some bank entry, spilling ~0.1-0.25 of its
        # mass onto don[routed] and falsifying the probe null. With EVERY donor exposure in
        # the bank, a wrong-slot mimic prediction lands on its OWN donor entry (counted as
        # donor_other — the directly-measured arm-mimic mass) and the follow null equals the
        # strategy's slot accuracy exactly.
        bank = torch.cat([torch.stack([own[w["routed"]], own[w["name"]]]), don])
        d = ((bank - pred_obs[i].unsqueeze(0)) ** 2).mean(-1)
        pick = int(d.argmin())
        is_deep = w["N"] in (4, 5) and w["depth"] >= 2 and w["style"] == "core"
        is_earn = (is_deep and ceiling_table is not None and ceiling_table.get(
            f"{w['N']},{w['depth']},{len(w['movers'])},{w['R']}", 1.0) < 0.99)
        tallies = (cnt,) + ((dcnt,) if is_deep else ()) + ((ecnt,) if is_earn else ())
        for c in tallies:
            c["m"] += 1
            c["follow"] += int(pick == 2 + w["routed"])
            c["own_r"] += int(pick == 0)
            c["own_n"] += int(pick == 1)
            c["don_n"] += int(pick == 2 + w["name"])
            c["don_o"] += int(pick >= 2 and pick not in (2 + w["routed"], 2 + w["name"]))

    def _rep(c):
        m = c["m"]
        return {"n_matched": m,
                "follow_donor_routed": round(c["follow"] / m, 4) if m else None,
                "own_routed": round(c["own_r"] / m, 4) if m else None,
                "own_name": round(c["own_n"] / m, 4) if m else None,
                "donor_name": round(c["don_n"] / m, 4) if m else None,
                "donor_other": round(c["don_o"] / m, 4) if m else None}
    out = _rep(cnt)
    out["n_diagonal_skipped"] = diag
    # freeze reviews C5/F4 + re-verification R2-1: pooled depths are 2-hop-clearable and the
    # deep_core slice is still ~43% depth<=2 — the GATE reads the core EARNABLE slice (where
    # the resolver contributes 0 and the analytic arm family caps at ~0.52); deep_core and
    # pooled stay reported.
    out["deep_core"] = _rep(dcnt)
    out["earnable_core"] = _rep(ecnt)
    return out


def build_swap_cache(ctx, percep_name, device, seed=20260806, max_windows=None):
    """Synthesize and encode the role-swap chains ONCE.

    Nothing here depends on the net: the partner draw is a deterministic function of the window
    index and the seed, the alternative chain is a function of the board, and the resulting mv
    embeddings are a function of the encoder. So this is a fixed property of
    (root, split, eye, seed) and must not be recomputed per candidate — doing so reloads the
    encoder and re-encodes every alternative chain once per (genome, seed), which is the dominant
    avoidable cost in a measurement campaign.

    Returns the spliced token tensor plus the routing and marker maps that scoring needs.

    Probe 1 (operator attribution): re-encode each window with a DIFFERENT valid chain over the
    SAME board, synthesized by ROLE-SWAP (review 2026-08-06): pick a swap partner c' != routed and
    exchange the move-position sets of routed and c' in the schedule. The alternative keeps the
    exact dst sequence, is always legal (each content's moves stay in increasing positions from
    its current location), routes c' to the test location with alt-depth == the STAMPED depth
    (the rejection-sampled version drew ~53% shallow alternatives on the deep slice — a shallow
    last-hop resolver could inflate follow_alt_deep), and covers 100% of windows (no attempt-loop
    selection surface). The new `mv` strings are rendered via the perception impl, encoded
    probe-time, standardized with the cached cmd stats, spliced at the mv command positions.
    PASS = the N-way pick follows the ALTERNATIVE routed content.

    Encoder-frame guards (review 2026-08-06): the probe root's perception stamp (when present)
    must match the loaded checkpoint's tree sha, and a SELF-PARITY gate re-encodes a sample of
    the windows' ORIGINAL mv commands and asserts cosine > 0.999 against the cached standardized
    z_cmd rows — a wrong encoder, render, or stats frame fails loud instead of recording noise."""
    import random as _random
    from evolve import reencode as RE
    from transformers import AutoModel, AutoTokenizer
    percep = RE.load_perception(percep_name)
    revision = getattr(percep, "REVISION", None)
    stamp = None
    root = ctx.get("data_root")
    if root:
        summ = pathlib.Path(root) / "summary.json"
        if summ.exists():
            stamp = (json.loads(summ.read_text()).get("perception") or {}).get(
                "checkpoint_tree_sha")
    eye_tree_sha = RE._checkpoint_tree_sha(percep.MODEL) if hasattr(
        RE, "_checkpoint_tree_sha") else None
    if stamp and eye_tree_sha is not None:
        got = eye_tree_sha
        assert got == stamp, (
            f"alt_chain encoder mismatch: {percep_name} resolves to a checkpoint with tree sha "
            f"{got[:12]} but the probe root was encoded under {stamp[:12]} — set TJ_FT_ENCODER "
            f"to the root's eye")
    tokz = AutoTokenizer.from_pretrained(percep.MODEL, revision=revision)
    enc = AutoModel.from_pretrained(percep.MODEL, revision=revision).to(device).eval()
    maxlen = getattr(percep, "MAXLEN", 256)
    mc, sc = ctx["cmd_stats"]                            # the cached-embedding cmd stats frame

    def encode_cmds(cmds):
        texts = [percep.render_cmd({"cmd": c}) for c in cmds]
        e = tokz(texts, return_tensors="pt", padding=True, truncation=True, max_length=maxlen)
        e = {k: v.to(device) for k, v in e.items()}
        with torch.no_grad():
            h = enc(**e).last_hidden_state
            z = percep.pool(h, e["attention_mask"]).float().cpu()
        return (z - mc) / sc

    wins = ctx["wins"]
    rng0 = _random.Random(seed)
    idxs = (sorted(rng0.sample(range(len(wins)), max_windows))
            if max_windows and max_windows < len(wins) else list(range(len(wins))))
    # self-parity gate over a sample of ORIGINAL mv commands (positions -> cached rows)
    par = [(i, t, src, dst) for i in idxs for (t, src, dst) in wins[i]["mvs"]][:32]
    self_parity_cos = None
    if par:
        z_re = encode_cmds([f"mv {src} {dst}" for (_, _, src, dst) in par])
        z_cached = torch.stack([ctx["seqs"][wins[i]["si"]]["z_cmd"][t]
                                for (i, t, _, _) in par])
        cos = F.cosine_similarity(z_re, z_cached, dim=-1)
        self_parity_cos = float(cos.mean())
        assert float(cos.mean()) > 0.999, (
            f"alt_chain self-parity FAIL (mean cos {float(cos.mean()):.4f}): the probe-time "
            f"encode does not reproduce the cached z_cmd — wrong encoder/render/stats frame")
    tok2 = ctx["tok"].clone()
    alts, partner_mover, name_skip = {}, 0, 0
    alt_marks = {}
    for i in idxs:
        w = wins[i]
        s = ctx["seqs"][w["si"]]
        slots = w["slots"]
        mv_steps = [t for (t, _, _) in w["mvs"]]
        dsts = [d for (_, _, d) in w["mvs"]]
        tloc = s["steps"][w["r"]]["cmd"].split()[1]
        # integrity: replay the ORIGINAL chain, reproduce the stamped routing, and recover the
        # per-content move-position sets the role-swap needs
        loc = {k: slots[k] for k in range(w["N"])}
        pos = {k: [] for k in range(w["N"])}
        for kd, (_, msrc, mdst) in enumerate(w["mvs"]):
            c = next(k for k, v in loc.items() if v == msrc)
            loc[c] = mdst
            pos[c].append(kd)
        assert loc[w["routed"]] == tloc, f"chain replay mismatch on {w['id']}"
        # role-swap: partner c' != routed takes routed's positions (and vice versa).
        # c' also != name_idx (review round 2: routing the NAME-keyed content to tloc would
        # credit a purely name-keyed model with "follow"); on the diagonal (routed == name)
        # any c' != routed is safe. N==2 off-diagonal windows have no valid partner — skipped
        # and counted (the exposure_swap diagonal-exclusion pattern).
        # The partner MUST itself be a mover. This is the condition the exchange-symmetry
        # argument actually needs, and it was not enforced while alt_chain was only a
        # diagnostic probe (a recorded reference run drew a non-mover partner in roughly half
        # of all probed windows). It matters now because this differential is the SELECTION
        # target, and a one-sided exchange is farmable:
        #   routed is always a mover (the slice requires depth >= 1), so if the partner never
        #   moved, routed's move positions transfer to it and NOTHING comes back. A positional
        #   heuristic — first mover, last mover, deepest — then flips from routed to the
        #   partner under the swap, scoring native_hit = 1 and swap_stayed = 0, i.e. +1. The
        #   windows that were supposed to cancel it (where the PARTNER is the native marker
        #   and the swap hands the marker to routed, scoring -1) cannot exist, because a
        #   non-mover is never the native positional marker.
        # With both sides movers the exchange is genuinely symmetric and the -1 windows are
        # exactly as likely as the +1 windows, which is what makes the expectation zero.
        cand2 = [k for k in range(w["N"])
                 if k != w["routed"] and (k != w["name"] or w["name"] == w["routed"])
                 and len(pos[k]) > 0]
        if not cand2:
            name_skip += 1
            continue
        rng = _random.Random(seed + i)
        c2 = rng.choice(cand2)
        partner_mover += int(len(pos[c2]) > 0)          # now 1 by construction; kept as a rail
        sched2 = {}
        for kd in pos[w["routed"]]:
            sched2[kd] = c2
        for kd in pos[c2]:
            sched2[kd] = w["routed"]
        loc2 = {k: slots[k] for k in range(w["N"])}
        cmds2, srcs2 = [], []
        depth2 = {k: 0 for k in range(w["N"])}
        lastmv2 = None
        for kd, dst in enumerate(dsts):
            c = sched2.get(kd)
            if c is None:
                # an unswapped content keeps its original move: its location trajectory is
                # identical under both chains, so the original src still identifies it
                c = next(k for k, v in loc2.items() if v == w["mvs"][kd][1])
            cmds2.append(f"mv {loc2[c]} {dst}")
            srcs2.append(loc2[c])
            loc2[c] = dst
            depth2[c] += 1
            lastmv2 = c
        routed2 = next(k for k, v in loc2.items() if v == tloc)
        # The SWAPPED chain's positional markers, so a caller can evaluate the analytic
        # non-tracker arms on the same exchange the net sees. These are pure functions of the
        # command strings — no net, no embeddings — which is exactly why they are the honest
        # floor to measure a candidate against.
        slot_of2 = {p: k for k, p in enumerate(slots)}
        dmax2 = max([d for d in depth2.values() if d > 0] or [0])
        alt_marks[i] = {
            "h_first": slot_of2.get(srcs2[0]) if srcs2 else None,
            "h_last": next((slot_of2[s] for s in reversed(srcs2) if s in slot_of2), w["name"]),
            "h_lastmv": lastmv2,
            "at_name": w["name"],                      # untouched by the swap, by construction
            "deepest": sorted(k for k, d in depth2.items() if d == dmax2 and d > 0),
        }
        assert routed2 == c2 and routed2 != w["routed"], f"role-swap failed on {w['id']}"
        z = encode_cmds(cmds2)
        for kd, t in enumerate(mv_steps):
            tok2[i, 2 * t] = z[kd]
        alts[i] = routed2
    return {"tok2": tok2, "alts": alts, "alt_marks": alt_marks, "idxs": idxs,
            "name_skip": name_skip, "partner_mover": partner_mover,
            "self_parity_cos": self_parity_cos, "eye_tree_sha": eye_tree_sha, "seed": seed}


def alt_chain(net, ctx, target_mod, device, percep_name=None, seed=20260806, max_windows=None,
              ceiling_table=None, cache=None):
    """Score a trained net against the role-swapped chains.

    `cache` is the output of build_swap_cache. Pass it: it is net-independent, and rebuilding it
    per candidate re-pays the encoder load and the whole re-encode. Omitting it rebuilds inline,
    which is correct but wasteful and is kept only so a one-off probe stays a one-liner.
    """
    if cache is None:
        cache = build_swap_cache(ctx, percep_name, device, seed=seed, max_windows=max_windows)
    wins = ctx["wins"]
    tok2, alts, alt_marks = cache["tok2"], cache["alts"], cache["alt_marks"]
    idxs, name_skip, partner_mover = cache["idxs"], cache["name_skip"], cache["partner_mover"]
    self_parity_cos, eye_tree_sha = cache["self_parity_cos"], cache["eye_tree_sha"]

    pred = _fwd_pred(net, tok2, ctx["types"], ctx["key_pad"], ctx["rpos"], device)
    pred_obs = target_mod.to_obs(pred, ctx["z_prev"]) if target_mod is not None else pred
    picks = _nway(pred_obs, ctx)
    m = len(alts)
    follow = sum(1 for i, r2 in alts.items() if picks[i] == r2)
    stayed = sum(1 for i in alts if picks[i] == wins[i]["routed"])
    # PER-WINDOW rows (added for cups_ca). Every aggregate below is derivable from these; they
    # exist because the compositional differential must be intersected with measure()'s rows
    # window-by-window and meaned UNROUNDED. Subtracting the round(4) aggregates instead is
    # wrong twice over: the rounding, and the fact that the two arms' slice selectors do not
    # agree (measure's earnable_core has no depth filter; earn_core here requires depth>=2 and
    # silently drops the no-partner windows). Keyed by window id, which is stable across arms.
    rows = [{"id": wins[i]["id"], "i": i, "N": wins[i]["N"], "R": wins[i]["R"],
             "depth": wins[i]["depth"], "style": wins[i]["style"],
             "m": len(wins[i]["movers"]), "routed": wins[i]["routed"],
             "alt_routed": r2, "pick": picks[i],
             "stayed": int(picks[i] == wins[i]["routed"]),
             "follow": int(picks[i] == r2),
             "alt_marks": alt_marks[i]}
            for i, r2 in sorted(alts.items())]
    # deep = the §3 PRIMARY slice (N in {4,5}, depth >= 2); alt-depth == stamped depth by the
    # role-swap construction, so this slice's alternatives are genuinely deep. The GATE reads
    # the core-style deep slice (freeze review C5/F4); both are reported.
    deep = [i for i in alts if wins[i]["N"] in (4, 5) and wins[i]["depth"] >= 2]
    deep_core = [i for i in deep if wins[i]["style"] == "core"]
    earn_core = [i for i in deep_core if ceiling_table is not None and ceiling_table.get(
        f"{wins[i]['N']},{wins[i]['depth']},{len(wins[i]['movers'])},{wins[i]['R']}",
        1.0) < 0.99]
    return {"rows": rows, "pred_obs": pred_obs,
            "self_parity_cos": self_parity_cos, "eye_tree_sha": eye_tree_sha,
            "n_matched": m, "n_probed": len(idxs), "n_no_partner_skipped": name_skip,
            "alt_depth_equals_stamped": True,
            "partner_was_mover_frac": round(partner_mover / m, 4) if m else None,
            "follow_alt_routed": round(follow / m, 4) if m else None,
            "stayed_original": round(stayed / m, 4) if m else None,
            "follow_alt_deep": round(sum(1 for i in deep if picks[i] == alts[i]) / len(deep), 4)
                               if deep else None,
            "n_deep": len(deep),
            "follow_alt_deep_core": round(sum(1 for i in deep_core if picks[i] == alts[i])
                                          / len(deep_core), 4) if deep_core else None,
            "stayed_original_deep_core": round(sum(1 for i in deep_core
                                                   if picks[i] == wins[i]["routed"])
                                               / len(deep_core), 4) if deep_core else None,
            "n_deep_core": len(deep_core),
            "follow_alt_earnable_core": round(sum(1 for i in earn_core
                                                  if picks[i] == alts[i])
                                              / len(earn_core), 4) if earn_core else None,
            "stayed_original_earnable_core": round(sum(1 for i in earn_core
                                                       if picks[i] == wins[i]["routed"])
                                                   / len(earn_core), 4) if earn_core else None,
            "n_earnable_core": len(earn_core)}


def _load_standardized_seqs(data, split, model, device, stats_data=None):
    """The (root, split) loader: cached embeddings standardized by the stats root's TRAIN stats,
    restricted to the split's images, with the raw jsonl `steps` attached and length/shape-asserted.

    Ported from the old imag_windows.load_windows, minus its unconditional imagination-window
    harvest — the cups lane never used those windows and paying for them dragged in the whole
    base-world shell-state closure. Standardizing on the pack root's OWN train stats is the
    frozen frame discipline: never borrow another root's stats here."""
    train_full = H._cached_encode(data, "train", model, device)
    if stats_data and stats_data != data:
        mo, so, mc, sc = M.standardize_stats(H._cached_encode(stats_data, "train", model, device))
    else:
        mo, so, mc, sc = M.standardize_stats(train_full)       # train-only stats (own root)
    if split in ("inner", "final"):
        val_seqs = H._cached_encode(data, "val", model, device)
        M.apply_stats(val_seqs, mo, so, mc, sc)
        seqs = split_val(val_seqs, split)
        raw = split_val([json.loads(l) for l in open(pathlib.Path(data) / "val.jsonl")], split)
    elif split == "train":
        M.apply_stats(train_full, mo, so, mc, sc)
        seqs = train_full
        raw = [json.loads(l) for l in open(pathlib.Path(data) / "train.jsonl")]
    else:
        raise ValueError(f"split must be inner|final|train, got {split}")
    assert len(seqs) == len(raw), f"embedding/raw length mismatch {len(seqs)} vs {len(raw)}"
    for e, r in zip(seqs, raw):
        assert e["image"] == r["image"] and e["z_obs"].shape[0] == len(r["steps"])
        e["steps"] = r["steps"]
    return seqs


def load_cups_context(data, split, model, device=None, stats_data=None):
    """Windows + layout for a (root, split), standardized on the stats root's train frame."""
    device = device or M.pick_device()
    seqs = _load_standardized_seqs(data, split, model, device, stats_data=stats_data)
    wins = harvest_cups_windows(seqs)
    if not wins:
        return None
    ctx = build_cups_layout(wins, seqs)
    ctx["seqs"] = seqs
    ctx["data_root"] = data          # for alt_chain's perception-stamp guard
    # the cmd standardization frame (for probe-time alt-chain encodes): recompute exactly as
    # IW.load_windows did — the stats root's raw train cache
    stats_train = H._cached_encode(stats_data or data, "train", model, device)
    _mo, _so, mc, sc = M.standardize_stats(stats_train)
    ctx["cmd_stats"] = (mc, sc)
    return ctx
