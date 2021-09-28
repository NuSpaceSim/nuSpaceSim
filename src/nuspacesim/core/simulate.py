# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Simulation Results class and full simulation main function."""

from astropy.table import Table as AstropyTable
import datetime
from numpy.typing import NDArray
from typing import Any, Iterable, Union

from nuspacesim.core import NssConfig
from nuspacesim.modules.geometry import RegionGeom
from nuspacesim.modules.taus import Taus
from nuspacesim.modules.eas_optical import EAS

__all__ = ["Simulation", "simulate"]


class Simulation(AstropyTable):
    r"""Results of NuSpaceSim simulation stages.

    Simulation inherits from astropy.table.Table and uses that implementation to manage
    Result data columns. This enables easy serialization of simulation results to
    output file formats.
    """

    def __init__(self, config: NssConfig):
        r"""Constructor for Simulation class instances.

        Parameters
        ----------
        config: NssConfig
            Configuration object. Used to initialize result metadata.
        """

        now = f"{datetime.datetime.now():%Y%m%d%H%M%S}"
        super().__init__(
            meta={
                **config.detector(),
                **config.simulation(),
                **config.constants(),
                "simTime": (now, "Start time of Simulation"),
            }
        )

    def __call__(self, col_names: Iterable[str], columns: Iterable[NDArray]) -> None:
        r"""Add named columns to the simulation results.

        Insert data into the result table, with the names corresponding to the
        values in col_names.

        Parameters
        ----------
        col_names: Iterable[str]
            List of column names.
        columns: Iterable[NDArray]
            List of column
        """

        self.add_columns(columns, names=col_names)

    def add_meta(self, name: str, value: Any, comment: str) -> None:
        r"""Add metadata attributes to the simulation results.

        Insert a named attribute into the table metadata store, with a descriptive
        comment.

        Parameters
        ----------
        name: str
            Attribute name keyword.
        value: Any
            Attribute scalar value.
        comment: str
            Attribute descriptive comments.
        """

        self.meta[name] = (value, comment)

    def write(self, filename: Union[str, None] = None, **kwargs) -> None:
        r"""Write the simulation results to a file.

        Uses the astropy.table.Table write method of the Simulation base class to write
        FITS file.

        Parameters
        ----------
        filename: {str, None}, optional
            The filename of the output file. If None, return default with timestamp.

        Raises
        ------
        ValueError:
            If the input format value is not one of fits or hdf5, an exception is
            raised.

        """

        if "format" not in kwargs:
            kwargs["format"] = "fits"

        if kwargs["format"] == "fits":

            filename = (
                f"nuspacesim_run_{self.meta['simTime'][0]}.fits"
                if filename is None
                else filename
            )
            super().write(filename, **kwargs)

        elif kwargs["format"] == "hdf5":

            filename = (
                f"nuspacesim_run_{self.meta['simTime'][0]}.hdf5"
                if filename is None
                else filename
            )

            if "path" not in kwargs:
                kwargs["path"] = "/"
            if "overwrite" not in kwargs:
                kwargs["overwrite"] = True
            kwargs["serialize_meta"] = True

            super().write(filename, **kwargs)

        else:
            raise ValueError(f"File output format {format} not in [fits, hdf5]!")


def simulate(config: NssConfig, verbose: bool = False) -> Simulation:
    r"""Simulate an upward going shower.

        Parameters
        ----------
    config: NssConfig
        Configuration object.
    verbose: bool, optional
        Flag enabling optional verbose output.
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

    mcint, mcintgeo, numEvPass = geom.mcintegral(
        numPEs, costhetaChEff, tauExitProb, store=sim
    )

    if verbose:
        print("Monte Carlo Integral", mcint)
        print("Monte Carlo Integral, GEO Only", mcintgeo)
        print("Number of Passing Events", numEvPass)

    return sim
