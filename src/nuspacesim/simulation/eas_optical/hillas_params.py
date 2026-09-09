"""Hillas 1982 eq. 13 angular distribution parameters."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HillasParams:
    """Frozen container for Hillas angular-distribution constants.

    Parameters from Hillas (1982) eq. 13.
    """

    A: float = 0.777
    z_star: float = 0.59
    lam1: float = 0.478
    lam2: float = 0.380

    @property
    def u_star(self) -> float:
        return self.z_star**2

    def lam(self, z) -> np.ndarray:
        """Piecewise lambda: lam1 for z < z_star, lam2 for z >= z_star."""
        z = np.asarray(z, dtype=np.float64)
        return np.where(z < self.z_star, self.lam1, self.lam2)


HILLAS_DEFAULT = HillasParams()

# Energy-regime-aware Hillas parameters (per HILLAS_SINGLE_INTEGRAL_KERNEL.md)
HILLAS_LOW_ENERGY = HillasParams(A=0.777, z_star=0.59, lam1=0.478, lam2=0.380)
HILLAS_HIGH_ENERGY = HillasParams(A=1.318, z_star=0.37, lam1=0.413, lam2=0.380)
