"""v3-policy detection + the fail-closed cache-staleness gate.

This repo scores exactly one thing (the compositional-depth metric on the cups pack root), so the
only benchmark-version machinery it keeps is the part that refuses to train on a stale or
unstamped v3 cache. The frozen class tables, the baseline-arm sets and the classes-file resolver
lived here for the base-world content-verb margin and are gone with it.
"""

import json
import pathlib


# ---------------------------------------------------------------- v3-policy scoring-side infra
# These helpers are classes-file-INDEPENDENT (dockerfs3-prereg §7): they let the reencode/
# mv_encode stampers, the harness cached-encode gate, and load_perception_for_root fail-closed on
# a v3 root, keyed only on the root's own declaration. They never touch v1/v2 roots.

def is_v3_policy(data_root):
    """True iff the root's summary.json declares a dockerfs3 (v3) bench policy. Lightweight
    detection (no classes.json load); False for a missing/unparseable summary, for v1 (no
    summary), and for dockerfs2-v2.0. This is the cheap predicate the fail-closed gates branch
    on."""
    s = pathlib.Path(data_root) / "summary.json"
    if not s.exists():
        return False
    try:
        js = json.loads(s.read_text())
    except Exception:
        return False
    return str(js.get("bench_version", "")).startswith("dockerfs3")


def require_v3_cache(data_root):
    """Fail-closed staleness gate for a v3-policy root (§13.2): the root-level cache_meta.json must
    exist and carry {cache_format: 3, bench_version, policy_sha, classes_sha} consistent with the
    root's summary.json, AND summary.json must carry the perception stamp {perception:{impl,model,
    content_sha}}. Any absence/mismatch RAISES — a v3 root scored against a v2-era or partial cache
    is impossible by construction. NEVER call this on a v1/v2 root (they pass straight through)."""
    root = pathlib.Path(data_root)
    summ = root / "summary.json"
    if not summ.exists():
        raise ValueError(f"{data_root}: v3-policy root with no summary.json (fail-closed, §13.2)")
    js = json.loads(summ.read_text())
    cm_path = root / "cache_meta.json"
    if not cm_path.exists():
        raise ValueError(f"{data_root}: v3-policy root missing cache_meta.json — refusing to load a "
                         f"stamp-less v3 cache (fail-closed, §13.2)")
    cm = json.loads(cm_path.read_text())
    if cm.get("cache_format") != 3:
        raise ValueError(f"{data_root}: cache_meta.json cache_format={cm.get('cache_format')!r} != 3 "
                         f"(fail-closed, §13.1)")
    for fld in ("bench_version", "policy_sha", "classes_sha"):
        cv, jv = cm.get(fld), js.get(fld)
        # B1: reject FALSY stamps, not only mismatched — a pre-B1 root (or a hand-edited cache)
        # carrying no policy_sha/classes_sha would otherwise pass this guard vacuously (None==None),
        # exactly the gap that let an unstamped v3 root reach scoring. A v3 root MUST pin all three.
        if not cv or not jv:
            raise ValueError(f"{data_root}: v3 {fld} is empty (cache_meta={cv!r}, summary={jv!r}) "
                             f"— a v3 root MUST carry a non-empty {fld} (fail-closed, §13.2)")
        if cv != jv:
            raise ValueError(f"{data_root}: cache_meta.json {fld}={cv!r} != summary.json "
                             f"{jv!r} — stale/mismatched v3 cache (fail-closed, §13.2)")
    if not ((js.get("perception") or {}).get("content_sha")):
        raise ValueError(f"{data_root}: v3-policy root lacking the perception stamp "
                         f"{{perception:{{impl,model,content_sha}}}} (fail-closed, §10.3/§13.1)")
    return cm
