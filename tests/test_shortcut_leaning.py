"""The shortcut-leaning readout splits a candidate's own score by where a shortcut pays.

Run: PYTHONPATH=$PWD .venv/bin/python tests/test_shortcut_leaning.py
"""
import sys

from evolve.cups_ca import arm_diff, shortcut_leaning

FAILED = []


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        FAILED.append(name)


def win(wid, routed, first_src, name=9, last_src=None, last_mover=9, deepest=None, depth=4):
    # depth is what the trace arms read; the scored slice is depth 3 and 4, both above their caps
    return {"id": wid, "routed": routed, "first_src": first_src, "name": name, "depth": depth,
            "last_src": last_src, "last_mover": last_mover, "deepest": deepest or []}


def swap(wid, h_first, name=9, h_last=9, h_lastmv=9, deepest=None):
    return {"id": wid, "alt_marks": {"h_first": h_first, "h_last": h_last,
                                     "h_lastmv": h_lastmv, "at_name": name,
                                     "deepest": deepest or []}}


def main():
    # routed=0. A window "pays" the first-mover shortcut when the routed content moves first
    # natively and does NOT move first under the swap.
    W = ["pays1", "pays2", "neutral1", "neutral2", "costs1"]
    wins = {
        "pays1": win("pays1", 0, first_src=0),
        "pays2": win("pays2", 0, first_src=0),
        "neutral1": win("neutral1", 0, first_src=1),
        "neutral2": win("neutral2", 0, first_src=0),
        "costs1": win("costs1", 0, first_src=1),
    }
    sw = {
        "pays1": swap("pays1", h_first=1),
        "pays2": swap("pays2", h_first=1),
        "neutral1": swap("neutral1", h_first=1),
        "neutral2": swap("neutral2", h_first=0),
        "costs1": swap("costs1", h_first=0),
    }
    want = {"pays1": 1, "pays2": 1, "neutral1": 0, "neutral2": 0, "costs1": -1}
    for i in W:
        d = arm_diff(wins[i], sw[i], "h_first")
        check(f"arm_diff({i}) == {want[i]:+d}", d == want[i], f"got {d}")

    print("\na candidate that only wins where the shortcut pays is visible as such")
    only = {"pays1": 1.0, "pays2": 1.0, "neutral1": 0.0, "neutral2": 0.0, "costs1": 0.0}
    r = shortcut_leaning(wins, sw, W, only, "h_first")
    check("counts partition W", r["pays"]["n"] + r["neutral"]["n"] + r["costs"]["n"] == len(W))
    check("strata sizes are 2/2/1",
          (r["pays"]["n"], r["neutral"]["n"], r["costs"]["n"]) == (2, 2, 1),
          str({k: v["n"] for k, v in r.items()}))
    check("earns 1.0 where the shortcut pays", r["pays"]["mean"] == 1.0)
    check("earns 0.0 where it does not", r["neutral"]["mean"] == 0.0 and r["costs"]["mean"] == 0.0)

    print("\na candidate that tracks content earns evenly across the strata")
    even = {i: 0.6 for i in W}
    r = shortcut_leaning(wins, sw, W, even, "h_first")
    check("all three strata read the same",
          r["pays"]["mean"] == r["neutral"]["mean"] == r["costs"]["mean"] == 0.6)

    print("\nan empty stratum reports None, not zero")
    W2 = ["pays1", "pays2"]
    r = shortcut_leaning(wins, sw, W2, only, "h_first")
    check("empty strata are None", r["neutral"]["mean"] is None and r["costs"]["mean"] is None)
    check("empty strata count zero", r["neutral"]["n"] == 0 and r["costs"]["n"] == 0)

    print("\nname-keying cancels per window, so at_name has no paying stratum")
    r = shortcut_leaning(wins, sw, W, only, "at_name")
    check("at_name is entirely neutral", r["neutral"]["n"] == len(W))

    print("\ndeepest splits credit across tied movers, so its difference is fractional")
    fw = {"f": win("f", 0, first_src=9, deepest=[0, 1])}
    fs = {"f": swap("f", h_first=9, deepest=[])}
    d = arm_diff(fw["f"], fs["f"], "deepest")
    check("arm_diff is 0.5, not an integer", d == 0.5, f"got {d}")
    check("round() would have mis-binned it as neutral", round(d) == 0)
    r = shortcut_leaning(fw, fs, ["f"], {"f": 1.0}, "deepest")
    check("classified by sign, it lands in 'pays'", r["pays"]["n"] == 1,
          str({k: v["n"] for k, v in r.items()}))

    print("\nthe band is the mean of arm_diff, and the strata reconstruct the score")
    from evolve.cups_ca import analytic_band
    band = analytic_band(wins, sw, W)
    for arm in ("h_first", "at_name"):
        mean_diff = sum(arm_diff(wins[i], sw[i], arm) for i in W) / len(W)
        check(f"analytic_band[{arm}] == mean(arm_diff)", abs(band[arm] - mean_diff) < 1e-12,
              f"{band[arm]} vs {mean_diff}")
    r = shortcut_leaning(wins, sw, W, only, "h_first")
    recon = sum(v["n"] * v["mean"] for v in r.values() if v["mean"] is not None) / len(W)
    overall = sum(only[i] for i in W) / len(W)
    check("count-weighted strata reconstruct the overall differential",
          abs(recon - overall) < 1e-12, f"{recon} vs {overall}")

    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("all shortcut-leaning checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
