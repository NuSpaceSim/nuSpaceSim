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

r"""The main proceedure for performaing a full simulation in nuspacesim.

*********************
NuSpaceSim Simulation
*********************

.. currentmodule:: nuspacesim


.. autosummary::
   :toctree:

   simulate

"""

from .config import NssConfig
from .results_table import ResultsTable
from .simulation.geometry.region_geometry import RegionGeom
from .simulation.taus import Taus
from .simulation.eas_optical.eas import EAS

__all__ = ["simulate"]


def simulate(
    config: NssConfig,
    verbose: bool = False,
    output_file: str = None,
    to_plot: list = [],
    write_stages=False,
) -> ResultsTable:
    r"""Simulate an upward going shower.

    The main proceedure for performaing a full simulation in nuspacesim.
    Given a valid NssConfig object, :func:`simulate`, will perform the simulation as
    follows:

    #. Initialize the ResultsTable object.
    #. Initialize the appropritate :ref:`simulation modules<simulation>`.
    #. Compute array of valid beta angle trajectories: beta_tr from :class:`RegionGeom`.
    #. Compute tau interaction attributes componentwise for each element of beta_tr.

       #. tauBeta
       #. tauLorentz
       #. showerEnergy
       #. tauExitProb

    #. Compute Extensive Air Shower attributes componentwise

       #. Decay Altitude
       #. Photon Density
       #. Cherenkov Angle

    #. Compute the Monte Carlo integral for the resulting shower geometries.

    At each stage of the simulation, array results are stored as contiguous columns,
    and scalar results are stored as attributes, both in the :class:`ResultsTable`
    object.


    Parameters
    ----------
    config: NssConfig
        Configuration object.
    verbose: bool, optional
        Flag enabling verbose output.
    to_plot: list, optional
        Call the listed plotting functions as appropritate.

    Returns
    -------
    ResultsTable
        The Table of result valuse from each stage of the simulation.
    """

    def printv(*args):
        if verbose:
            print(*args)

    sim = ResultsTable(config)
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)

    class _stage_writer:
        def __call__(self, *args, **kwargs):
            sim(*args, **kwargs)
            if write_stages:
                sim.write(output_file, overwrite=True)

        def add_meta(self, *args, **kwargs):
            sim.add_meta(*args, **kwargs)
            if write_stages:
                sim.write(output_file, overwrite=True)

    stage_writer = _stage_writer()

    printv(f"Running NuSpaceSim with log(E_nu)={config.simulation.log_nu_tau_energy}")

    # Run simulation
    beta_tr = geom(config.simulation.N, store=stage_writer, plot=to_plot)
    printv(f"Threw {config.simulation.N} neutrinos. {beta_tr.size} were valid.")

    printv(f"Computing taus.")
    tauBeta, tauLorentz, showerEnergy, tauExitProb = tau(beta_tr, store=stage_writer)

    printv(f"Computing decay altitudes.")
    altDec = eas.altDec(beta_tr, tauBeta, tauLorentz, store=stage_writer)

    printv(f"Computing EAS Cherenkov light.")
    numPEs, costhetaChEff = eas(beta_tr, altDec, showerEnergy, store=stage_writer)

    printv(f"Computing Monte Carlo Integral.")
    mcint, mcintgeo, numEvPass = geom.mcintegral(
        numPEs, costhetaChEff, tauExitProb, store=stage_writer
    )

    printv("Monte Carlo Integral:", mcint)
    printv("Monte Carlo Integral, GEO Only:", mcintgeo)
    printv("Number of Passing Events:", numEvPass)

    return sim
