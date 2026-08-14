# Integration fix applied before scoring

The proposal defined `def _apply(self, x, gain)` on its `nn.Module`. `_apply` is a torch
internal: `Module._apply(fn)` is what `.to(device)` calls, so the override shadowed it and the
harness died with `TypeError: _apply() missing 1 required positional argument: 'gain'` at the
first `.to(device)`. Both the leakage and reachability gates caught it before any GPU time.

Renamed to `_rescale` at all three sites. Nothing else changed: no mechanism, no arithmetic, no
hyperparameter. The inventor could not have caught this from inside the jail, where the real
harness is not runnable, so this is integration hardening rather than co-authorship — the
mechanism being measured is exactly the one proposed.
