from collections.abc import MutableMapping

import numpy as np


def cartesian_product(*arrays):
    sz = len(arrays)
    prod = np.empty([len(a) for a in arrays] + [sz], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        prod[..., i] = a
    return prod.reshape(-1, sz)


def _flat(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flat(d, parent_key, sep))


def unflatten_dict(d: MutableMapping, sep: str = ".") -> dict:
    """Inverse of :func:`flatten_dict`: rebuild a nested dict from flat keys.

    Each key is split on ``sep`` into a path; intermediate dicts are created as
    needed. ``unflatten_dict(flatten_dict(x)) == x`` for any nested mapping
    whose keys do not themselves contain ``sep``.
    """
    out: dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def output_filename(state, filename) -> str:
    return filename if filename else f"nuspacesim_run_{state.sim_time}.fits"
