####
from numpy import radians, log10, sin

from typing import Union
from dataclasses import dataclass
from functools import cached_property
from nuspacesim import constants as const

__all__ = [
    "DetectorCharacteristics",
    "SimulationParameters",
    "NssConfig",
]


@dataclass
class DetectorCharacteristics:
    """Dataclass holding Detector Characteristics"""

    altitude: float = 525.0
    ra_start: float = 0.0
    dec_start: float = 0.0
    telescope_effective_area: float = 2.5
    quantum_efficiency: float = 0.2
    photo_electron_threshold: float = 10.0

    def __call__(self) -> dict[str, tuple[float, str]]:
        """
        Return a dictionary representation of the data members with descriptive
        comments. This is useful for setting the FITS Header Keywords in the Simulation
        Ouput file.
        """

        return {
            "detAlt": (self.altitude, "Detector: Altitude"),
            "raStart": (self.ra_start, "Detector: Initial Right Ascencion"),
            "decStart": (self.dec_start, "Detector: Initial Declination"),
            "telEffAr": (
                self.telescope_effective_area,
                "Detector: Telescope Effective Area",
            ),
            "quantEff": (self.quantum_efficiency, "Detector: Quantum Efficiency"),
            "phEthres": (
                self.photo_electron_threshold,
                "Detector: Photo-Electron Threshold",
            ),
        }


@dataclass
class SimulationParameters:
    """Holding simulation parameters"""

    N: int = 1000
    theta_ch_max: float = radians(3.0)
    nu_tau_energy: float = 1e8
    e_shower_frac: float = 0.5
    ang_from_limb: float = radians(7.0)
    max_azimuth_angle: float = radians(360.0)

    @cached_property
    def log_nu_tau_energy(self) -> float:
        return log10(self.nu_tau_energy)

    @cached_property
    def sin_theta_ch_max(self) -> float:
        return sin(self.theta_ch_max)

    def __call__(self) -> dict[str, tuple[Union[int, float], str]]:
        """
        Return a dictionary representation of the data members with descriptive
        comments. This is useful for setting the FITS Header Keywords in the Simulation
        Ouput file.
        """
        return {
            "N": (self.N, "Simulation: thrown neutrinos"),
            "thChMax": (self.theta_ch_max, "Simulation: Maximum Cherenkov Angle"),
            "nuTauEn": (self.nu_tau_energy, "Simulation: nutau energy (GeV)"),
            "eShwFrac": (self.e_shower_frac, "Simulation: Fraction of Etau in Shower"),
            "angLimb": (self.ang_from_limb, "Simulation: Angle From Limb"),
            "maxAzAng": (self.max_azimuth_angle, "Simulation: Maximum Azimuthal Angle"),
        }


@dataclass
class NssConfig:
    """
    Necessary Configuration Data for NuSpaceSim.
    """

    detector: DetectorCharacteristics = DetectorCharacteristics()
    simulation: SimulationParameters = SimulationParameters()
    constants: const.Fund_Constants = const.Fund_Constants()
