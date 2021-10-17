import numpy as np


def cartesian_product(*arrays):
    sz = len(arrays)
    prod = np.empty([len(a) for a in arrays] + [sz], dtype=np.result_type(*arrays))
    for i, a in enumerate(np.ix_(*arrays)):
        prod[..., i] = a
    return prod.reshape(-1, sz)
