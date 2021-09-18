#

from astropy.table import Table as AstropyTable
import datetime
from typing import Any, Iterable
from numpy.typing import NDArray

from nuspacesim.configuration import NssConfig
from nuspacesim.nssgeometry import RegionGeom
from nuspacesim.taus import Taus
from nuspacesim.EAScherGen import EAS

__all__ = ["Simulation", "simulate"]


class Simulation(AstropyTable):
    def __init__(self, config: NssConfig):
        now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
        super().__init__(
            meta={
                **config.detector(),
                **config.simulation(),
                **config.constants(),
                "simTime": (now, "Start time of Simulation"),
            }
        )

    def __call__(self, col_names: Iterable[str], columns: Iterable[NDArray]):
        """
        Insert data into the table, with the names corresponding to the
        values in col_names.
        """
        self.add_columns(columns, names=col_names)

    def add_meta(self, name: str, value: Any, comment: str):
        """Insert a named attribute into the table metadata store."""
        self.meta[name] = (value, comment)


def simulate(config: NssConfig, verbose=False) -> Simulation:
    """
    Simulate an upward going shower.
    """

    sim = Simulation(config)
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)

    # Run simulation
    beta_tr = geom(config.simulation.N, store=sim)
    tauBeta, tauLorentz, showerEnergy, tauExitProb = tau(beta_tr, store=sim)
    altDec = eas.altDec(beta_tr, tauBeta, tauLorentz)
    numPEs, costhetaChEff = eas(beta_tr, altDec, showerEnergy, store=sim)
    mcintegral, mcintgeo, numEvPass = geom.mcintegral(
        numPEs, costhetaChEff, tauExitProb, store=sim
    )

    if verbose:
        print("MonteCarlo Integral", mcintegral)
        print("MonteCarlo Integral, GEO Only", mcintgeo)
        print("Number of Passing Events", numEvPass)

    return sim
