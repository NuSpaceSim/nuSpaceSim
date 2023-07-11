import numpy as np

import nuspacesim.simulation.eas_optical.atmospheric_models as atm

if __name__ == "__main__":
    N = int(1e6)
    lo = np.random.uniform(0, 2, N)
    hi = lo + np.random.uniform(1, 20, N)
    theta = np.random.uniform(0, np.pi / 2, N)
    atm.slant_depth_us(lo, hi, theta)
