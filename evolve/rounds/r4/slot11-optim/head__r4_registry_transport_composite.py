from evolve.chunks.head import r2_dualaddress_move_transport as _TRANSPORT
from evolve.chunks.head import r3_rename_registry_occupancy_routing as _REGISTRY

NAME = "r4_registry_transport_composite"
DESCRIPTION = (
    "Stacks two content-routing memories on one trunk. The rename-registry memory (slots holding "
    "an address key, a content vector and a fill scalar, where a move re-addresses a slot and "
    "leaves its content untouched, with source/destination roles decided at run time by memory "
    "occupancy) wraps the arch first; the dual-address transport memory (a delta-rule "
    "outer-product store whose source and destination keys come from two extractors sharing one "
    "path-to-key projection, writing the value read at the source key into the destination key) "
    "wraps the result. Both inject their read at command positions through a per-dimension scale "
    "initialised to zero, so the composed forward equals the bare arch forward at initialisation. "
    "Both train-time auxiliaries run and are summed: the registry's bridge-mined InfoNCE plus "
    "routing entropy/load terms plus the transition-operator content-preservation term, and the "
    "transport's duplicate-observation-with-an-intervening-mutation InfoNCE on the memory read and "
    "on the final prediction. Genome params are routed by prefix: 'reg_' to the registry, 'trn_' "
    "to the transport. No auxiliary runs at eval."
)

_REG_PREFIX = "reg_"
_TRN_PREFIX = "trn_"

_TRN_OVERRIDES = {"aux_weight": 0.5}


def _split_params(params):
    reg = {}
    trn = dict(_TRN_OVERRIDES)
    unknown = []
    for k, v in (params or {}).items():
        if k.startswith(_REG_PREFIX):
            reg[k[len(_REG_PREFIX):]] = v
        elif k.startswith(_TRN_PREFIX):
            trn[k[len(_TRN_PREFIX):]] = v
        else:
            unknown.append(k)
    return reg, trn, unknown


def wrap(net, D, **params):
    existing = getattr(net, "_registry_transport_state", None)
    if existing is not None:
        return existing

    reg_p, trn_p, _ = _split_params(params)
    reg_state = _REGISTRY.wrap(net, D, **reg_p)
    trn_state = _TRANSPORT.wrap(net, D, **trn_p)

    state = {"registry": reg_state, "transport": trn_state, "D": int(D)}
    net._registry_transport_state = state
    return state


def aux_loss(head_state, batch, net, device):
    if head_state is None:
        return 0.0
    total = _REGISTRY.aux_loss(head_state.get("registry"), batch, net, device)
    total = total + _TRANSPORT.aux_loss(head_state.get("transport"), batch, net, device)
    return total


def leak_safe(mod, params):
    reg_p, trn_p, unknown = _split_params(params)
    if unknown:
        return False
    if not _REGISTRY.leak_safe(mod, reg_p):
        return False
    if not _TRANSPORT.leak_safe(mod, trn_p):
        return False
    return True
