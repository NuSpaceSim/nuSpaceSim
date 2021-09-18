"""

"""

from functools import cached_property
from dataclasses import dataclass
import numpy as np

earth_radius = 6371.0  # in KM
low_earth_orbit = 525.0  # in KM
atmosphere_end = 65.0  # in KM
c = 2.9979246e5
massTau = 1.77686
mean_Tau_life = 2.903e-13


@dataclass
class Fund_Constants:
    """
    Fundimental constants used on nuspacesim simulator.
    """

    earth_radius: float = earth_radius
    c: float = c
    massTau: float = massTau
    mean_Tau_life: float = mean_Tau_life
    pi: float = np.pi

    @cached_property
    def inv_mean_Tau_life(self) -> float:
        return 1.0 / self.mean_Tau_life  # [s^-1]

    def __call__(self) -> dict[str, tuple[float, str]]:
        """
        Return a dictionary representation of the data members with descriptive
        comments. This is useful for setting the FITS Header Keywords in the Simulation
        Ouput file.
        """
        return {
            "R_Earth": (self.earth_radius, "FundimentalConstants: Earth Radius"),
            "c": (self.c, "FundimentalConstants: speed of light"),
            "massTau": (self.massTau, "FundimentalConstants: massTau"),
            "uTauLife": (self.mean_Tau_life, "FundimentalConstants: mean Tau Lifetime"),
            "pi": (self.pi, "FundimentalConstants: pi"),
        }
