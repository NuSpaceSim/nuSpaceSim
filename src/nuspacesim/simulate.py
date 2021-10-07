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

import numpy as np
from .config import NssConfig
from .results_table import ResultsTable
from .simulation.geometry.region_geometry import RegionGeom
from .simulation.taus.taus import Taus
from .simulation.eas_optical.eas import EAS
from .simulation.eas_radio.radio_antenna import noise_efield_from_range, calculate_snr

__all__ = ["simulate"]


def simulate(config: NssConfig, verbose: bool = True) -> ResultsTable:
    r"""Simulate an upward going shower.
    Parameters
    ----------
    config: NssConfig
        Configuration object.
    verbose: bool, optional
        Flag enabling optional verbose output.
    """

    FreqRange = (config.detector.low_freq, config.detector.high_freq)
    sim = ResultsTable(config)
    geom = RegionGeom(config)
    tau = Taus(config)
    eas = EAS(config)

    if verbose:
        print(
            f"Running Nu Space Simulate with log(E_nu) = "
            f"{config.simulation.log_nu_tau_energy}"
        )

    # Run simulation
    betaArr, thetaArr, pathLenArr = geom(config.simulation.N, store=sim)
    if verbose:
        print(f"Threw {config.simulation.N} neutrinos. {betaArr.size} were valid.")

    if verbose:
        print(f"Computing taus.")
    tauBeta, tauLorentz, showerEnergy, tauExitProb = tau(betaArr, store=sim)
    altDec, lenDec = eas.altDec(betaArr, tauBeta, tauLorentz)

    if config.detector.method == 'Optical':
        if verbose:
            print(f"Computing EAS (Optical).")
        numPEs, costhetaChEff = eas(betaArr, altDec, lenDec, thetaArr, pathLenArr, showerEnergy, store=sim)
        if verbose:
            print(f"Computing MC Integral.")
        mcint, mcintgeo, numEvPass = geom.mcintegral(
            numPEs, costhetaChEff, tauExitProb, store=sim
        )

    if config.detector.method == 'Radio':
        if verbose:
            print(f"Computing EAS (Radio).")
        EFields, decay_h = eas(betaArr, altDec, lenDec, thetaArr, pathLenArr, showerEnergy)
        E_sigsum = EFields.sum(axis=1)
        E_noise = noise_efield_from_range(FreqRange, config.detector.altitude)
        snrs = calculate_snr(EFields, FreqRange, config.detector.altitude, config.detector.det_Nant, config.detector.det_gain)
        costhetaArr = np.cos(thetaArr)
        if verbose:
            print(f"Computing MC Integral.")
        mcint, mcintgeo, numEvPass = geom.mcintegral(snrs,
                                                        costhetaArr, tauExitProb)
    if config.detector.method == 'Both':
        if verbose:
            print(f"Computing EAS (Both).")
        npe_ef, costhetaChEff = eas(betaArr, altDec, lenDec, thetaArr, pathLenArr, showerEnergy)
        numPEs = npe_ef[0]
        EFields = npe_ef[1]
        E_noise = noise_efield_from_range(FreqRange, config.detector.altitude)
        snrs = calculate_snr(EFields, FreqRange, config.detector.altitude, config.detector.det_Nant, config.detector.det_gain)
        trigger_conds = [numPEs, snrs]
        if verbose:
            print(f"Computing MC Integral.")
        mcint, mcintgeo, numEvPass = geom.mcintegral(trigger_conds,
                                                        [costhetaChEff, np.cos(thetaArr)], tauExitProb)


    if verbose:
        print("Monte Carlo Integral:", mcint)
        print("Monte Carlo Integral, GEO Only:", mcintgeo)
        print("Number of Passing Events:", numEvPass)

    return sim
