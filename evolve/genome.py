"""Genome = a JSON-able dict selecting one implementation per chunk (+ params). The registry
resolves a chunk-impl name to code. Every axis takes the {"impl": name, "params": {...}} form;
the optional axes (target/batcher/stream/head) default to the R4 baseline impl."""

import importlib
import pathlib

CHUNKS_DIR = pathlib.Path(__file__).resolve().parent / "chunks"



def load_objective(genome):
    """Import the objective impl module named by the genome and return its `loss` callable.
    Raises a clear error if the impl is missing or malformed."""
    name = genome["chunks"]["objective"]["impl"]
    mod = importlib.import_module(f"evolve.chunks.objective.{name}")
    if not hasattr(mod, "loss"):
        raise AttributeError(f"objective impl '{name}' has no loss(pred, tgt) function")
    return mod.loss


def load_optim(genome):
    """Return (make_fn, bs). optim = {"impl": name, "bs": B} uses the optim registry (optimizer +
    LR schedule). make_fn(params, steps) -> (optimizer, scheduler_or_None)."""
    o = genome["chunks"]["optim"]
    bs = o.get("bs", 64)
    if "impl" in o:
        mod = importlib.import_module(f"evolve.chunks.optim.{o['impl']}")
        if not hasattr(mod, "make"):
            raise AttributeError(f"optim impl '{o['impl']}' has no make(params, steps)")
        p = dict(o.get("params", {}))
        return (lambda params, steps: mod.make(params, steps, **p)), bs
    raise ValueError("optim chunk must be {'impl': <name>, 'params': {...}, 'bs': B} — the legacy "
                     "{lr,wd,steps,bs} form is not supported")


def load_target(genome):
    """Return the target-chunk module. Pure impls expose make_target/to_obs (the R4 contract);
    LEARNED impls expose LEARNED=True and make(D) -> nn.Module with make_target/to_obs/reg,
    whose params the harness registers on the net (trained jointly, eval still in obs space).
    Defaults to identity when a genome has no target chunk, so existing genomes are unchanged."""
    t = genome["chunks"].get("target", {"impl": "identity"})
    mod = importlib.import_module(f"evolve.chunks.target.{t['impl']}")
    if getattr(mod, "LEARNED", False):
        if not hasattr(mod, "make"):
            raise AttributeError(f"learned target impl '{t['impl']}' has no make(D)")
        return mod
    for fn in ("make_target", "to_obs"):
        if not hasattr(mod, fn):
            raise AttributeError(f"target impl '{t['impl']}' has no {fn}")
    return mod


def load_batcher(genome):
    """Return make(fit, bs, seed) -> next_batch(step, total_steps) -> list[int] of length bs.
    Defaults to baseline_uniform (bit-identical to the historical torch.randint stream) when a
    genome has no batcher chunk, so all archived genomes are unchanged."""
    b = genome["chunks"].get("batcher", {"impl": "baseline_uniform"})
    mod = importlib.import_module(f"evolve.chunks.batcher.{b['impl']}")
    if not hasattr(mod, "make_batcher"):
        raise AttributeError(f"batcher impl '{b['impl']}' has no make_batcher(fit, bs, seed)")
    p = dict(b.get("params", {}))
    return lambda fit, bs, seed: mod.make_batcher(fit, bs, seed, **p)


def load_stream(genome):
    """Return the stream-chunk module (collate / extract_cmd_pred / flatten_predictions /
    leakage_ok) — how the (cmd, obs) step sequence is laid out as tokens. Defaults to
    baseline_interleave (bit-identical to the historical harness plumbing) when a genome has
    no stream chunk, so all archived genomes are unchanged."""
    s = genome["chunks"].get("stream", {"impl": "baseline_interleave"})
    mod = importlib.import_module(f"evolve.chunks.stream.{s['impl']}")
    for fn in ("collate", "extract_cmd_pred", "flatten_predictions", "leakage_ok"):
        if not hasattr(mod, fn):
            raise AttributeError(f"stream impl '{s['impl']}' has no {fn}")
    return mod


def load_head(genome):
    """Return (head_module, params) for the readout-head chunk (wrap / aux_loss / leak_safe).
    Defaults to baseline_passthrough (bit-identical to the historical bare-Linear readout: wrap
    is a no-op, aux_loss returns 0.0) when a genome has no head chunk, so archived genomes are
    unchanged."""
    h = genome["chunks"].get("head", {"impl": "baseline_passthrough"})
    mod = importlib.import_module(f"evolve.chunks.head.{h['impl']}")
    for fn in ("wrap", "aux_loss", "leak_safe"):
        if not hasattr(mod, fn):
            raise AttributeError(f"head impl '{h['impl']}' has no {fn}")
    return mod, dict(h.get("params", {}))


def load_arch(genome):
    """Return (build_fn, params) for the arch chunk. arch = {"impl": name, "params": {...}} uses
    the arch registry (a swappable model module). build_fn(**params) -> nn.Module with
    SeqWorldModel's I/O contract."""
    a = genome["chunks"]["arch"]
    if "impl" in a:
        mod = importlib.import_module(f"evolve.chunks.arch.{a['impl']}")
        if not hasattr(mod, "build"):
            raise AttributeError(f"arch impl '{a['impl']}' has no build(**params) function")
        return mod.build, dict(a.get("params", {}))
    raise ValueError("arch chunk must be {'impl': <name>, 'params': {...}} — the legacy "
                     "{d,layers,heads,dropout} form is not supported")


def list_impls(chunk="objective"):
    d = CHUNKS_DIR / chunk
    return sorted(p.stem for p in d.glob("*.py") if p.stem != "__init__")


def validate(genome):
    """Cheap structural check before spending a training run on a genome."""
    c = genome.get("chunks", {})
    for k in ("objective", "arch", "optim"):
        if k not in c:
            raise ValueError(f"genome missing chunk '{k}'")
    if c["objective"]["impl"] not in list_impls("objective"):
        raise ValueError(f"unknown objective impl '{c['objective']['impl']}' "
                         f"(have {list_impls('objective')})")
    a = c["arch"]
    if "impl" not in a:
        raise ValueError("arch chunk must be {'impl': <name>, 'params': {...}} — the legacy "
                         "{d,layers,heads,dropout} form is not supported")
    if a["impl"] not in list_impls("arch"):
        raise ValueError(f"unknown arch impl '{a['impl']}' (have {list_impls('arch')})")
    o = c["optim"]
    if "impl" not in o:
        raise ValueError("optim chunk must be {'impl': <name>, 'params': {...}, 'bs': B} — the "
                         "legacy {lr,wd,steps,bs} form is not supported")
    if o["impl"] not in list_impls("optim"):
        raise ValueError(f"unknown optim impl '{o['impl']}' (have {list_impls('optim')})")
    if "batcher" in c and c["batcher"]["impl"] not in list_impls("batcher"):
        raise ValueError(f"unknown batcher impl '{c['batcher']['impl']}' "
                         f"(have {list_impls('batcher')})")
    if "stream" in c and c["stream"]["impl"] not in list_impls("stream"):
        raise ValueError(f"unknown stream impl '{c['stream']['impl']}' "
                         f"(have {list_impls('stream')})")
    if "head" in c and c["head"]["impl"] not in list_impls("head"):
        raise ValueError(f"unknown head impl '{c['head']['impl']}' "
                         f"(have {list_impls('head')})")
    return True
