"""cdh_probe — the cd-history nav->read instrument (see research/cdh-probe-design.md)."""
import torch

from evolve.cups_probe import _fwd_pred, _load_standardized_seqs
from realenv import seq_worldmodel as M

D = M.D
K, ROUNDS = 63, 4
MIN_NAV_DEPTH = 2


def _row_top1(pred, true, verbs, foil_seed=0):
    """Mean over ROUNDS of strict-tie top-1 against K same-verb foils, per row."""
    gen = torch.Generator().manual_seed(foil_seed)
    acc = torch.zeros(true.shape[0])
    for _ in range(ROUNDS):
        foil = M._foils_sameverb(verbs, K, gen)
        d_true = ((true - pred) ** 2).mean(-1)
        d_foil = ((true[foil] - pred.unsqueeze(1)) ** 2).mean(-1)
        acc += ((d_foil < d_true.unsqueeze(1)).sum(1) == 0).float()
    return acc / ROUNDS


def _retrieve_by_cmd(fit_wins, fit_seqs, read_cmd_q):
    """Read-command-only retrieval: the train read obs whose read command is nearest each query."""
    keys = torch.nn.functional.normalize(
        torch.stack([fit_seqs[w["si"]]["z_cmd"][w["r"]] for w in fit_wins]), dim=-1)
    vals = torch.stack([fit_seqs[w["si"]]["z_obs"][w["r"]] for w in fit_wins])
    q = torch.nn.functional.normalize(read_cmd_q, dim=-1)
    return torch.cat([vals[(q[i:i + 512] @ keys.T).argmax(1)] for i in range(0, q.shape[0], 512)])


def harvest_cdh_windows(seqs):
    """One window per cdh_read step, carrying the >=MIN_NAV_DEPTH nav block preceding it."""
    wins = []
    for si, s in enumerate(seqs):
        steps = s["steps"]
        for r, st in enumerate(steps):
            if st.get("meta", {}).get("role") != "cdh_read":
                continue
            nav, ctx_steps = [], []
            t = r - 1
            while t >= 0:
                mt = steps[t].get("meta", {})
                if mt.get("arm") == "cdhist" and mt.get("role") != "cdh_read":
                    (nav if steps[t]["cmd"].split()[0] == "cd" else ctx_steps).append(t)
                    t -= 1
                else:
                    break
            if len(nav) < MIN_NAV_DEPTH:
                continue
            wins.append({"id": f"{s['image']}:{si}:{r}", "si": si, "r": r,
                         "nav": sorted(nav), "ctx": sorted(ctx_steps),
                         "image": s["image"], "read_sig": "cdh_read",
                         "landing": steps[r].get("cwd"),
                         "redir": bool(steps[r].get("meta", {}).get("cdh_redir")) or any(
                             steps[t]["cmd"].endswith(">/dev/null") for t in nav)})
    return wins


def build_cdh_layout(wins, seqs):
    """Native interleaved [cmd_0,obs_0,...,cmd_r] per window (no obs_r), with the unmasked key_pad
    plus the masked and no-prior-reads variants."""
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
    masked_obs_pos = []
    unmasked_landing = []
    for i, w in enumerate(wins):
        s = seqs[w["si"]]
        r = w["r"]
        for j in range(r):
            tok[i, 2 * j] = s["z_cmd"][j]; valid[i, 2 * j] = True
            tok[i, 2 * j + 1] = s["z_obs"][j]; types[i, 2 * j + 1] = 1; valid[i, 2 * j + 1] = True
        tok[i, 2 * r] = s["z_cmd"][r]; valid[i, 2 * r] = True
        rpos[i] = 2 * r
        z_r[i] = s["z_obs"][r]
        z_prev[i] = s["z_obs"][r - 1]
        read_cmd[i] = s["z_cmd"][r]
        # Mask EVERY cdh-arm obs in the prefix, not only this window's nav block: an earlier cdh
        # block that transited the landing dir leaves a visible cwd token the model can read
        # instead of routing. cwd=/tmp/w/cdh/dN appears only in cdh-arm obs.
        cdh_obs = [2 * t + 1 for t in range(r)
                   if s["steps"][t].get("meta", {}).get("arm") == "cdhist"]
        masked_obs_pos.append(cdh_obs)
        landing = w.get("landing") or ""
        # stdout and stderr are scanned independently of the reconciled `output` field so a
        # partition bug cannot hide a leak. This sees the RAW record only, never the render.
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
            key_pad_masked[i, p] = True
    key_pad_noprior = key_pad.clone()
    for i, w in enumerate(wins):
        s = seqs[w["si"]]
        for t in range(w["r"]):
            if s["steps"][t].get("meta", {}).get("role") == "cdh_read":
                key_pad_noprior[i, 2 * t + 1] = True
    verbs = [w["read_sig"] for w in wins]
    return {"wins": wins, "tok": tok, "types": types, "key_pad": key_pad,
            "key_pad_masked": key_pad_masked, "key_pad_noprior": key_pad_noprior,
            "rpos": rpos, "z_r": z_r, "z_prev": z_prev,
            "read_cmd": read_cmd, "verbs": verbs, "masked_obs_pos": masked_obs_pos,
            "unmasked_landing": unmasked_landing}


def assert_no_leak(ctx):
    """Assert no obs at or after the read position, no landing path in an unmasked non-cdh obs,
    and every cdh-arm obs masked in the masked layout and visible in the unmasked one."""
    types, rpos = ctx["types"], ctx["rpos"]
    for kp in (ctx["key_pad"], ctx["key_pad_masked"]):
        for i in range(len(ctx["wins"])):
            rp = int(rpos[i])
            after = (types[i, rp:] == 1) & (~kp[i, rp:])
            assert not after.any(), f"win {i}: valid obs at/after read pos -> leak"
    for i, leaked in enumerate(ctx["unmasked_landing"]):
        assert not leaked, f"win {i}: landing path visible in unmasked non-cdh obs at steps {leaked}"
    for i, poss in enumerate(ctx["masked_obs_pos"]):
        for p in poss:
            assert bool(ctx["key_pad_masked"][i, p]), f"win {i}: cdh obs {p} not masked"
            assert not bool(ctx["key_pad"][i, p]), f"win {i}: cdh obs {p} masked in the UNMASKED layout"


def _fit_ceiling(fit_seqs, ctx):
    """Genome-independent ceiling banks on the cdh windows: retrieve-by-command, the train cdh
    read-target centroid, and the global train-obs centroid."""
    fit_wins = harvest_cdh_windows(fit_seqs)
    if fit_wins:
        retr_cmd = _retrieve_by_cmd(fit_wins, fit_seqs, ctx["read_cmd"])
        cdh_cent = torch.stack([fit_seqs[w["si"]]["z_obs"][w["r"]] for w in fit_wins]
                               ).mean(0, keepdim=True).expand(len(ctx["wins"]), D)
    else:
        retr_cmd = torch.zeros_like(ctx["z_r"])
        cdh_cent = torch.zeros_like(ctx["z_r"])
    allobs = torch.stack([s["z_obs"][j] for s in fit_seqs for j in range(s["z_obs"].shape[0])])
    glob = allobs.mean(0, keepdim=True).expand(len(ctx["wins"]), D)
    return {"retr_cmd": retr_cmd, "glob": glob, "cdh_cent": cdh_cent,
            "n_fit_cdh": len(fit_wins)}


def masked_s1(net, ctx, fit_ceiling, target_mod, device):
    """The gate measurement: the net's content-top1 margin over the ceiling, unmasked vs masked."""
    z_r, verbs = ctx["z_r"], ctx["verbs"]
    assert fit_ceiling["n_fit_cdh"] > 0, (
        "empty retrieve_by_cmd bank (no cdh windows in the TRAIN split) -> degenerate ceiling; "
        "the probed net must train on pack/blend data. Refusing to emit a false GO.")
    z0 = torch.zeros_like(ctx["z_prev"])

    def wm_top1(key_pad, zprev):
        pred = _fwd_pred(net, ctx["tok"], ctx["types"], key_pad, ctx["rpos"], device)
        pred_obs = target_mod.to_obs(pred, zprev) if target_mod is not None else pred
        return float(_row_top1(pred_obs, z_r, verbs).mean())

    wm_un = wm_top1(ctx["key_pad"], ctx["z_prev"])
    # Under the mask, to_obs must be fed a ZEROED z_prev: a z_prev-dependent target (delta,
    # residual, learned) would re-inject the cwd token the mask removed, and a non-routing net
    # could post a false GO.
    wm_mk = wm_top1(ctx["key_pad_masked"], z0)
    wm_np = wm_top1(ctx["key_pad_noprior"], ctx["z_prev"]) if "key_pad_noprior" in ctx else None
    rbc = float(_row_top1(fit_ceiling["retr_cmd"], z_r, verbs).mean())
    cpy = float(_row_top1(ctx["z_prev"], z_r, verbs).mean())
    glob = float(_row_top1(fit_ceiling["glob"], z_r, verbs).mean())
    cdhc = float(_row_top1(fit_ceiling["cdh_cent"], z_r, verbs).mean())
    ceil_un = max(rbc, cpy, glob, cdhc)
    ceil_mk = max(rbc, glob, cdhc)
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
            "margin_masked": round(wm_mk - ceil_mk, 4),
            "wm_noprior": round(wm_np, 4) if wm_np is not None else None,
            "margin_noprior": round(wm_np - ceil_un, 4) if wm_np is not None else None}


def build_wrong_nav(ctx, seqs, seed=20260803):
    """Attach the wrong-history arm: each window's nav COMMAND tokens replaced by those of a
    depth- and redirect-matched donor whose landing dir differs. Returns the mutated ctx."""
    wins = ctx["wins"]
    N = len(wins)
    depth = [len(w["nav"]) for w in wins]
    land = [w["landing"] for w in wins]
    red = [bool(w.get("redir")) for w in wins]
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
    tok_wrong = ctx["tok"].clone()
    donors = []
    cursor = 0
    for i, w in enumerate(wins):
        donor = None
        for step in range(N):
            j = perm[(cursor + step) % N]
            if depth[j] == depth[i] and land[j] != land[i] and red[j] == red[i]:
                donor = j
                cursor = (cursor + step + 1) % N
                break
        donors.append(donor)
        if donor is None:
            continue
        ds = seqs[wins[donor]["si"]]
        for slot, t in enumerate(w["nav"]):
            dt = wins[donor]["nav"][slot]
            tok_wrong[i, 2 * t] = ds["z_cmd"][dt]
    ctx["tok_wrong"] = tok_wrong
    ctx["wrong_donors"] = donors
    ctx["n_wrong_matched"] = sum(d is not None for d in donors)
    return ctx


def nav_probe(net, ctx, target_mod, device):
    """The real-nav vs wrong-history top1 differential on both layouts. Requires
    build_wrong_nav(ctx, seqs) first."""
    assert "tok_wrong" in ctx, "call build_wrong_nav(ctx, seqs) first"
    z_r, verbs = ctx["z_r"], ctx["verbs"]
    z0 = torch.zeros_like(ctx["z_prev"])

    def top1_rows(tok, key_pad, zprev):
        pred = _fwd_pred(net, tok, ctx["types"], key_pad, ctx["rpos"], device)
        pred_obs = target_mod.to_obs(pred, zprev) if target_mod is not None else pred
        return _row_top1(pred_obs, z_r, verbs)

    matched = torch.tensor([d is not None for d in ctx["wrong_donors"]])

    def _diff(key_pad, zprev):
        b = top1_rows(ctx["tok"], key_pad, zprev)
        w = top1_rows(ctx["tok_wrong"], key_pad, zprev)
        dm = float((b[matched] - w[matched]).mean()) if bool(matched.any()) else 0.0
        return float(b.mean()), float(w.mean()), dm

    # The masked arm zeroes z_prev for the same reason masked_s1 does.
    b_m, w_m, dm_m = _diff(ctx["key_pad_masked"], z0)
    b_u, w_u, dm_u = _diff(ctx["key_pad"], ctx["z_prev"])
    return {"n_windows": len(ctx["wins"]), "n_wrong_matched": ctx["n_wrong_matched"],
            "b_masked": round(b_m, 4), "wrong_history_masked": round(w_m, 4),
            "nav_differential": round(b_m - w_m, 4),
            "nav_differential_matched": round(dm_m, 4),
            "b_unmasked": round(b_u, 4), "wrong_history_unmasked": round(w_u, 4),
            "nav_differential_unmasked": round(b_u - w_u, 4),
            "nav_differential_unmasked_matched": round(dm_u, 4)}


def load_cdh_context(data, split, model, device=None, redir_only=False, stats_data=None):
    """Windows + layout + fit ceiling for a (data_root, split), or None if no window qualifies.

    redir_only keeps only `cd - >/dev/null` windows. stats_data selects the standardization
    frame: None means the root's own train statistics, otherwise another root's.
    """
    device = device or M.pick_device()
    seqs = _load_standardized_seqs(data, split, model, device, stats_data=stats_data)
    fit_seqs = _load_standardized_seqs(data, "train", model, device, stats_data=stats_data)
    wins = harvest_cdh_windows(seqs)
    if redir_only:
        # The raw-output leak scan cannot see the RENDER: a cwd-IN perception re-injects the
        # landing as a cwd token in every prefix obs, so the root's own render is checked here.
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
        # Windows preceded by ANY earlier cdh material are dropped: earlier windows' reads are a
        # nav-independent elimination channel.
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
    build_wrong_nav(ctx, seqs)
    ctx["fit_ceiling"] = _fit_ceiling(fit_seqs, ctx)
    ctx["seqs"] = seqs
    return ctx
