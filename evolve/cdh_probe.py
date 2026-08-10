"""cdh_probe — the cd-history nav->read instrument.

Two measurements over ONE cdh_read window harvest + a native masked-endpoint layout:

  masked_s1 (the GO/NO-GO gate): the world model's content-top1 margin over the ceiling on the
    pack's history-routed reads, UNMASKED vs with the nav-block cwd obs MASKED. render_obs bakes the
    history-set cwd into a `cwd=/tmp/w/cdh/dN` token in every observation, so an UNMASKED read is
    clearable by a 1-step copy of the preceding `cd -` obs (copy_prev == the cwd oracle). The
    **masked** margin is the decisive gate: with the nav cwd obs gone, the only route to the landing
    dir d_i is the cd-COMMAND sequence (`cd d_i; cd d_j; cd -` -> back to d_i). A masked margin > 0
    certifies deep history-routing that can express on a masked forward. Treatment
    (content/cwd = f(dir)) can pass; the permutation/name-keyed control FAILS by construction
    (retrieve-by-command already decodes its read).

  nav_probe (the null-disambiguator): the b vs wrong-history differential on the SAME windows
    ("is history-routing learned?"), measured on its own rather than off any composition metric, so
    a null composition result stays interpretable (routing-learned-but-disjoint vs
    routing-not-learned).

A cdh_read window = [cmd_0,obs_0, ..., cd d_i, cd d_j, cd -, cat name] with pred read at the
`cat name` command position. The nav block = the contiguous arm=='cdhist' cd steps immediately
before the read; their obs are the cwd tokens the mask removes. Structure-only here; the net-based
margins run on the same full-scale trained net the fitness eval uses — never a reduced-scale proxy.

This capability is REPORTED for every candidate and never scored.
"""
import torch

from evolve.cups_probe import _fwd_pred, _load_standardized_seqs
from realenv import seq_worldmodel as M

D = M.D
K, ROUNDS = 63, 4   # foils per round / rounds, for the strict-tie top-1 retrieval metric
MIN_NAV_DEPTH = 2   # a window qualifies only with >=2 cd steps of history (cd d_i; cd d_j; cd -)


# The two helpers below are the shared retrieval-metric primitives of the direct-endpoint
# imagination instrument: _row_top1 scores a prediction by strict-tie top-1 against same-verb
# foils, and _retrieve_by_cmd is the read-command-only nearest-neighbour ceiling arm. They are
# carried here so this probe stands alone.

def _row_top1(pred, true, verbs, foil_seed=0):
    """Mean over ROUNDS of strict-tie top-1 vs K same-verb foils. foil_seed FIXED across all arms so
    the model and every ceiling arm are a PAIRED comparison."""
    gen = torch.Generator().manual_seed(foil_seed)
    acc = torch.zeros(true.shape[0])
    for _ in range(ROUNDS):
        foil = M._foils_sameverb(verbs, K, gen)
        d_true = ((true - pred) ** 2).mean(-1)
        d_foil = ((true[foil] - pred.unsqueeze(1)) ** 2).mean(-1)
        acc += ((d_foil < d_true.unsqueeze(1)).sum(1) == 0).float()
    return acc / ROUNDS


def _retrieve_by_cmd(fit_wins, fit_seqs, read_cmd_q):
    """Read-command-only retrieval arm (kept for continuity + as a floor component)."""
    keys = torch.nn.functional.normalize(
        torch.stack([fit_seqs[w["si"]]["z_cmd"][w["r"]] for w in fit_wins]), dim=-1)
    vals = torch.stack([fit_seqs[w["si"]]["z_obs"][w["r"]] for w in fit_wins])
    q = torch.nn.functional.normalize(read_cmd_q, dim=-1)
    return torch.cat([vals[(q[i:i + 512] @ keys.T).argmax(1)] for i in range(0, q.shape[0], 512)])


def harvest_cdh_windows(seqs):
    """Scan encoded seqs (from _load_standardized_seqs: per-step z_cmd/z_obs + raw `steps`) for
    cdh_read steps; return a window per read carrying the >=2-deep nav block preceding it."""
    wins = []
    for si, s in enumerate(seqs):
        steps = s["steps"]
        for r, st in enumerate(steps):
            if st.get("meta", {}).get("role") != "cdh_read":
                continue
            nav, ctx_steps = [], []                   # contiguous cdhist window body before r
            t = r - 1
            while t >= 0:
                mt = steps[t].get("meta", {})
                if mt.get("arm") == "cdhist" and mt.get("role") != "cdh_read":
                    # windows interleave real `ls` reads (role cdh_ctx) between the cds — nav
                    # (the swappable history) is the CD steps only.
                    (nav if steps[t]["cmd"].split()[0] == "cd" else ctx_steps).append(t)
                    t -= 1
                else:
                    break
            if len(nav) < MIN_NAV_DEPTH:
                continue
            wins.append({"id": f"{s['image']}:{si}:{r}", "si": si, "r": r,
                         "nav": sorted(nav), "ctx": sorted(ctx_steps),
                         "image": s["image"], "read_sig": "cdh_read",
                         "landing": steps[r].get("cwd"),   # the read's cwd = the history-set dir
                         # this window's final nav was `cd - >/dev/null` -> the landing is echoed
                         # NOWHERE (the leak-free probe set)
                         "redir": bool(steps[r].get("meta", {}).get("cdh_redir")) or any(
                             steps[t]["cmd"].endswith(">/dev/null") for t in nav)})
    return wins


def build_cdh_layout(wins, seqs):
    """Native interleaved [cmd_0,obs_0,...,cmd_r] per window (pred read at cmd_r; NO obs_r). Returns
    the ctx dict with an UNMASKED key_pad and a key_pad_masked whose nav cwd obs are key_padded."""
    B = len(wins)
    rs = [w["r"] for w in wins]
    Lmax = 2 * max(rs) + 1
    tok = torch.zeros(B, Lmax, D)
    types = torch.zeros(B, Lmax, dtype=torch.long)
    valid = torch.zeros(B, Lmax, dtype=torch.bool)
    z_r = torch.zeros(B, D)
    z_prev = torch.zeros(B, D)
    read_cmd = torch.zeros(B, D)
    rpos = torch.zeros(B, dtype=torch.long)
    masked_obs_pos = []                                 # per-window obs positions the masked pass hides
    unmasked_landing = []                               # landing tokens left visible
    for i, w in enumerate(wins):
        s = seqs[w["si"]]
        r = w["r"]
        for j in range(r):                              # steps 0..r-1: full (cmd, obs)
            tok[i, 2 * j] = s["z_cmd"][j]; valid[i, 2 * j] = True
            tok[i, 2 * j + 1] = s["z_obs"][j]; types[i, 2 * j + 1] = 1; valid[i, 2 * j + 1] = True
        tok[i, 2 * r] = s["z_cmd"][r]; valid[i, 2 * r] = True    # read cmd, obs_r withheld (target)
        rpos[i] = 2 * r
        z_r[i] = s["z_obs"][r]
        z_prev[i] = s["z_obs"][r - 1]                   # the `cd -` obs (cwd=d_i) — the copy_prev cheat
        read_cmd[i] = s["z_cmd"][r]
        # mask EVERY cdh-arm obs in the prefix, not just this window's immediate nav block: an
        # EARLIER cdh block that transited d_i would otherwise leave a visible cwd=d_i obs the model
        # could read instead of routing. cwd=/tmp/w/cdh/dN is namespaced, so it appears ONLY in
        # cdh-arm obs -> masking them removes every landing token; non-cdh obs can't carry it.
        cdh_obs = [2 * t + 1 for t in range(r)
                   if s["steps"][t].get("meta", {}).get("arm") == "cdhist"]
        masked_obs_pos.append(cdh_obs)
        landing = w.get("landing") or ""                # verify no UNMASKED prefix obs shows the landing
        # for redirected windows the landing must appear in NO obs AT ALL — including cdh-arm obs
        # (the redirected cd- prints nothing) — so the UNMASKED forward is leak-free.
        # Scan the two-channel fields too: `output` is the reconciled partition, but stdout/stderr
        # are scanned independently so a partition bug can't hide a leak. NOTE this checks the
        # RAW record; the cwd-RENDER channel is closed separately by load_cdh_context's redir_only
        # perception guard (a cwd-in render re-injects the landing as a cwd= token).
        strict = bool(w.get("redir"))
        leaked = [t for t in range(r)
                  if (strict or s["steps"][t].get("meta", {}).get("arm") != "cdhist")
                  and landing and any(landing in (s["steps"][t].get(f) or "")
                                      for f in ("output", "stdout", "stderr"))]
        unmasked_landing.append(leaked)
    key_pad = ~valid
    key_pad_masked = key_pad.clone()
    for i, poss in enumerate(masked_obs_pos):
        for p in poss:
            key_pad_masked[i, p] = True                 # drop every cdh cwd obs -> command-only routing
    # ELIMINATION-CHANNEL test: mask the obs of EARLIER cdh_READ steps in the prefix — their content
    # is exactly what a within-trajectory eliminator uses ("seen A,B,C -> answer is D").
    # If wm collapses under key_pad_noprior, the prediction rode on prior-reads elimination, NOT nav.
    key_pad_noprior = key_pad.clone()
    for i, w in enumerate(wins):
        s = seqs[w["si"]]
        for t in range(w["r"]):
            if s["steps"][t].get("meta", {}).get("role") == "cdh_read":
                key_pad_noprior[i, 2 * t + 1] = True
    verbs = [w["read_sig"] for w in wins]               # constant 'cdh_read' -> foils are the other reads
    return {"wins": wins, "tok": tok, "types": types, "key_pad": key_pad,
            "key_pad_masked": key_pad_masked, "key_pad_noprior": key_pad_noprior,
            "rpos": rpos, "z_r": z_r, "z_prev": z_prev,
            "read_cmd": read_cmd, "verbs": verbs, "masked_obs_pos": masked_obs_pos,
            "unmasked_landing": unmasked_landing}


def assert_no_leak(ctx):
    """Two guards: (1) the target obs (z_r) and every post-read token are absent from BOTH layouts —
    a defensive check against a future layout bug placing an obs at/after cmd_r; (2) the meaningful
    one: no UNMASKED non-cdh prefix obs contains the landing path, and the masked layout hides every
    cdh-arm obs — so the ONLY route to the landing dir is the cd-command history."""
    types, rpos = ctx["types"], ctx["rpos"]
    for kp in (ctx["key_pad"], ctx["key_pad_masked"]):
        for i in range(len(ctx["wins"])):
            rp = int(rpos[i])
            after = (types[i, rp:] == 1) & (~kp[i, rp:])    # any valid obs at/after read cmd?
            assert not after.any(), f"win {i}: valid obs at/after read pos -> leak"
    for i, leaked in enumerate(ctx["unmasked_landing"]):
        assert not leaked, f"win {i}: landing path visible in unmasked non-cdh obs at steps {leaked}"
    for i, poss in enumerate(ctx["masked_obs_pos"]):
        for p in poss:
            assert bool(ctx["key_pad_masked"][i, p]), f"win {i}: cdh obs {p} not masked"
            assert not bool(ctx["key_pad"][i, p]), f"win {i}: cdh obs {p} masked in the UNMASKED layout"


def _fit_ceiling(fit_seqs, ctx):
    """Genome-independent ceiling components on the cdh windows: retrieve-by-command (the `cat name`
    read command -> nearest train read obs) + the global train-obs centroid. copy_prev (== the cwd
    oracle) is added in masked_s1 from ctx z_prev so it drops out under the mask, exactly as the
    model's own copy would."""
    fit_wins = harvest_cdh_windows(fit_seqs)
    if fit_wins:
        retr_cmd = _retrieve_by_cmd(fit_wins, fit_seqs, ctx["read_cmd"])
        # the natural degenerate predictor for cdh windows: the TRAIN cdh read-target centroid —
        # much closer to the K-blob cluster than the global obs centroid, so the ceiling can't be
        # cleared by merely predicting "a cdh-ish blob".
        cdh_cent = torch.stack([fit_seqs[w["si"]]["z_obs"][w["r"]] for w in fit_wins]
                               ).mean(0, keepdim=True).expand(len(ctx["wins"]), D)
    else:                                               # no cdh windows in train -> no cmd bank
        retr_cmd = torch.zeros_like(ctx["z_r"])
        cdh_cent = torch.zeros_like(ctx["z_r"])
    allobs = torch.stack([s["z_obs"][j] for s in fit_seqs for j in range(s["z_obs"].shape[0])])
    glob = allobs.mean(0, keepdim=True).expand(len(ctx["wins"]), D)
    return {"retr_cmd": retr_cmd, "glob": glob, "cdh_cent": cdh_cent,
            "n_fit_cdh": len(fit_wins)}


def masked_s1(net, ctx, fit_ceiling, target_mod, device):
    """The gate measurement: WM content-top1 margin over the ceiling, UNMASKED vs MASKED.
    target_mod.to_obs reconstructs next-obs for retrieval (the same to_obs seam the scored eval uses).
    """
    z_r, verbs = ctx["z_r"], ctx["verbs"]
    # an empty retrieve-by-command bank is a degenerate ceiling that can post a false GO (and flip
    # the control to a false pass); the probed net MUST have trained on data containing the pack.
    # Fail loud.
    assert fit_ceiling["n_fit_cdh"] > 0, (
        "empty retrieve_by_cmd bank (no cdh windows in the TRAIN split) -> degenerate ceiling; "
        "the probed net must train on pack/blend data. Refusing to emit a false GO.")
    z0 = torch.zeros_like(ctx["z_prev"])                 # see the masked pass below

    def wm_top1(key_pad, zprev):
        pred = _fwd_pred(net, ctx["tok"], ctx["types"], key_pad, ctx["rpos"], device)
        pred_obs = target_mod.to_obs(pred, zprev) if target_mod is not None else pred
        return float(_row_top1(pred_obs, z_r, verbs).mean())

    wm_un = wm_top1(ctx["key_pad"], ctx["z_prev"])
    # under the mask, feed to_obs a ZEROED z_prev: a z_prev-dependent target (delta/residual/learned,
    # e.g. to_obs = z_prev + pred) would otherwise re-inject the very cwd token the mask removed,
    # defeating the gate — a non-routing net could post a false GO.
    wm_mk = wm_top1(ctx["key_pad_masked"], z0)
    # elimination-channel test: mask the EARLIER cdh reads' obs (the eliminator's evidence). If this
    # collapses wm toward the unmasked value while the nav is untouched, the prediction rode on
    # within-trajectory elimination, not nav-routing.
    wm_np = wm_top1(ctx["key_pad_noprior"], ctx["z_prev"]) if "key_pad_noprior" in ctx else None
    rbc = float(_row_top1(fit_ceiling["retr_cmd"], z_r, verbs).mean())
    cpy = float(_row_top1(ctx["z_prev"], z_r, verbs).mean())          # cwd oracle (unmasked only)
    glob = float(_row_top1(fit_ceiling["glob"], z_r, verbs).mean())
    cdhc = float(_row_top1(fit_ceiling["cdh_cent"], z_r, verbs).mean())   # K-blob centroid arm
    ceil_un = max(rbc, cpy, glob, cdhc)                 # unmasked: copy_prev/cwd-oracle available
    ceil_mk = max(rbc, glob, cdhc)                      # masked: prev obs gone -> copy_prev neutralized
    # coarseness diagnostic: content=f(dir) means <=K distinct targets/image, so surface how many
    # distinct landings + how separable the targets are (near-1 mean-cos == a coarse, tie-heavy pool
    # the top-1 can't finely discriminate).
    zr_n = torch.nn.functional.normalize(z_r, dim=-1)
    n = zr_n.shape[0]
    mean_cos = float((zr_n @ zr_n.t()).sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    return {"n_windows": len(ctx["wins"]), "n_fit_cdh": fit_ceiling["n_fit_cdh"],
            "n_distinct_landings": len({w["landing"] for w in ctx["wins"]}),
            "mean_target_cos": round(mean_cos, 4),
            "wm_unmasked": round(wm_un, 4), "wm_masked": round(wm_mk, 4),
            "retrieve_by_cmd": round(rbc, 4), "copy_prev_cwd_oracle": round(cpy, 4),
            "global_centroid": round(glob, 4), "cdh_centroid": round(cdhc, 4),
            "ceiling_unmasked": round(ceil_un, 4), "ceiling_masked": round(ceil_mk, 4),
            "margin_unmasked": round(wm_un - ceil_un, 4),
            "margin_masked": round(wm_mk - ceil_mk, 4),   # THE GATE: treatment > band, control <= 0
            "wm_noprior": round(wm_np, 4) if wm_np is not None else None,  # elimination-channel test
            "margin_noprior": round(wm_np - ceil_un, 4) if wm_np is not None else None}


def build_wrong_nav(ctx, seqs, seed=20260803):
    """The wrong-history arm on the cdh windows: replace each window's nav-block COMMAND tokens with
    a presence-matched (same nav depth) donor whose landing dir DIFFERS — so the command history now
    routes elsewhere while the read cmd + target z_r stay the window's own. On the MASKED layout (nav
    obs already gone) this isolates command-history routing: a genome that reads off the nav commands
    predicts the donor's landing content and its top1-vs-z_r collapses; a genome ignoring the history
    is unchanged. Differential b_masked - wrong_masked > 0 == history-routing learned."""
    wins = ctx["wins"]
    N = len(wins)
    depth = [len(w["nav"]) for w in wins]
    land = [w["landing"] for w in wins]
    red = [bool(w.get("redir")) for w in wins]      # donors must match redirect-ness too
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
    tok_wrong = ctx["tok"].clone()
    donors = []
    cursor = 0
    for i, w in enumerate(wins):
        donor = None
        for step in range(N):
            j = perm[(cursor + step) % N]
            # same presence (nav count + redirect-ness), different routing (landing)
            if depth[j] == depth[i] and land[j] != land[i] and red[j] == red[i]:
                donor = j
                cursor = (cursor + step + 1) % N
                break
        donors.append(donor)
        if donor is None:                                     # no differently-landing donor -> leave as-is
            continue
        ds = seqs[wins[donor]["si"]]
        for slot, t in enumerate(w["nav"]):
            dt = wins[donor]["nav"][slot]
            tok_wrong[i, 2 * t] = ds["z_cmd"][dt]             # swap the nav COMMAND token only
    ctx["tok_wrong"] = tok_wrong
    ctx["wrong_donors"] = donors
    ctx["n_wrong_matched"] = sum(d is not None for d in donors)
    return ctx


def nav_probe(net, ctx, target_mod, device):
    """The PRIMARY null-disambiguator: on the MASKED layout (command-only routing), the b arm (real
    nav) vs the wrong-history arm (donor nav) top1-vs-z_r differential. Fires (>0) iff the WM's read
    prediction depends on the actual navigation history — measured on its own rather than off any
    composition metric, so a null composition result stays interpretable
    (routing-learned-but-disjoint vs routing-not-learned). Requires build_wrong_nav(ctx) first."""
    assert "tok_wrong" in ctx, "call build_wrong_nav(ctx, seqs) first"
    z_r, verbs = ctx["z_r"], ctx["verbs"]
    z0 = torch.zeros_like(ctx["z_prev"])                 # masked layout -> zeroed z_prev

    def top1_rows(tok, key_pad, zprev):
        pred = _fwd_pred(net, tok, ctx["types"], key_pad, ctx["rpos"], device)
        pred_obs = target_mod.to_obs(pred, zprev) if target_mod is not None else pred
        return _row_top1(pred_obs, z_r, verbs)           # per-window [N]

    matched = torch.tensor([d is not None for d in ctx["wrong_donors"]])

    def _diff(key_pad, zprev):
        b = top1_rows(ctx["tok"], key_pad, zprev)
        w = top1_rows(ctx["tok_wrong"], key_pad, zprev)
        # matched-only differential (unmatched windows have wrong==b -> contribute 0)
        dm = float((b[matched] - w[matched]).mean()) if bool(matched.any()) else 0.0
        return float(b.mean()), float(w.mean()), dm

    b_m, w_m, dm_m = _diff(ctx["key_pad_masked"], z0)          # masked layout (cwd-in: command-only)
    # UNMASKED causal test — the valid one in cwd-OUT (no cwd obs to mask; masking the empty cd-obs
    # breaks the arch's per-(cmd,obs) state chain). Swap the nav COMMANDS to a different-landing donor:
    # if the prediction depends on the actual nav history, the differential is > 0.
    b_u, w_u, dm_u = _diff(ctx["key_pad"], ctx["z_prev"])
    return {"n_windows": len(ctx["wins"]), "n_wrong_matched": ctx["n_wrong_matched"],
            "b_masked": round(b_m, 4), "wrong_history_masked": round(w_m, 4),
            "nav_differential": round(b_m - w_m, 4),
            "nav_differential_matched": round(dm_m, 4),        # >0 == history-routing (masked layout)
            "b_unmasked": round(b_u, 4), "wrong_history_unmasked": round(w_u, 4),
            "nav_differential_unmasked": round(b_u - w_u, 4),
            "nav_differential_unmasked_matched": round(dm_u, 4)}   # the cwd-OUT causal routing test


def load_cdh_context(data, split, model, device=None, redir_only=False, stats_data=None):
    """Load the cdh windows + layout + fit ceiling for a (data_root, split). Uses the shared
    standardized-seqs loader, so the embeddings + train stats match the fitness eval byte-for-byte.
    redir_only: keep only `cd - >/dev/null` windows — the landing is echoed nowhere, so the UNMASKED
    forward is the leak-free routing probe.
    stats_data (default None = own-root stats, bit-identical): standardize the probe windows AND the
    fit ceiling bank by ANOTHER root's train statistics. This is load-bearing: the net being probed
    is trained on a different root, so its inputs live in THAT root's standardization frame, and the
    cd-history windows must be put in the same frame or the measurement is of a net fed inputs it
    never saw."""
    device = device or M.pick_device()
    seqs = _load_standardized_seqs(data, split, model, device, stats_data=stats_data)
    fit_seqs = _load_standardized_seqs(data, "train", model, device, stats_data=stats_data)
    wins = harvest_cdh_windows(seqs)
    if redir_only:
        # the raw-output leak scan cannot see the RENDER: a cwd-IN perception re-injects the landing
        # as a 'cwd=/tmp/w/cdh/dN' token in every prefix obs. Fail loud unless the root was ENCODED
        # with a cwd-dropping render (per its stamp).
        import json as _json
        import pathlib as _pl
        from evolve.reencode import load_perception as _lp
        stamp = (_json.loads((_pl.Path(data) / "summary.json").read_text()).get("perception")
                 or {})
        impl = stamp.get("impl")
        assert impl, f"redir_only: {data} carries no perception stamp — reencode it first"
        probe = _lp(impl).render_obs({"cwd": "/tmp/w/cdh/dPROBE", "exit": 0, "output": "x"})
        assert "/tmp/w/cdh/dPROBE" not in probe, (
            f"redir_only: {data} was encoded with a cwd-IN render ({impl}) — the landing leaks "
            f"through the cwd token; use a nocwd perception (e.g. enc_e5_ft_nocwd_hf)")
        wins = [w for w in wins if w.get("redir")]
        # drop windows preceded by ANY earlier cdh material in the trajectory: on a multi-read root,
        # earlier windows' reads/echoes are a nav-independent ELIMINATION channel. On one-read-per-
        # trajectory mints this drops nothing.
        def _no_prior_cdh(w):
            steps = seqs[w["si"]]["steps"]
            start = min(w["nav"] + w["ctx"] + [w["r"]])
            return not any(steps[t].get("meta", {}).get("arm") == "cdhist"
                           for t in range(start))
        wins = [w for w in wins if _no_prior_cdh(w)]
    if not wins:
        return None
    ctx = build_cdh_layout(wins, seqs)
    assert_no_leak(ctx)
    build_wrong_nav(ctx, seqs)                          # the nav->read wrong-history arm
    ctx["fit_ceiling"] = _fit_ceiling(fit_seqs, ctx)
    ctx["seqs"] = seqs
    return ctx
