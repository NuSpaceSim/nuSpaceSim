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

import numpy as np
from .cphotang import CphotAng
from ... import NssConfig

__all__ = ["EAS"]


class EAS:
    """
    Electromagnetic Air Shower wrapper class.

    Vectorized computation of photo-electrons and Cherenkov angles.
    """

    def __init__(self, config: NssConfig):
        self.config = config
        self.CphotAng = CphotAng()

    def altDec(self, beta, tauBeta, tauLorentz, u=None):
        """
        alt Decay
        """

        u = np.random.uniform(0.0, 1.0, len(beta)) if u is None else u

        tDec = (-1.0 * tauLorentz / self.config.constants.inv_mean_Tau_life) * np.log(u)

        lenDec = tDec * tauBeta * self.config.constants.c

        altDec = np.sqrt(
            self.config.constants.earth_radius ** 2
            + lenDec ** 2
            + 2.0 * self.config.constants.earth_radius * lenDec * np.sin(beta)
        )

        altDec -= self.config.constants.earth_radius

        return altDec

    def __call__(self, beta, altDec, showerEnergy, store=None) -> tuple:
        """
        Electromagnetic Air Shower operation.
        """

        # Mask out-of-bounds events. Do not pass to CphotAng. Instead use
        # Default values for dphots and thetaCh
        mask = (altDec < 0.0) | (altDec > 20.0)
        mask |= beta > np.radians(25.0)
        mask = ~mask

        # phots and theta arrays with default 0 and 1.5 values.
        dphots = np.full(beta.shape, np.nan)
        thetaCh = np.full(beta.shape, np.nan)

        # Run CphotAng on in-bounds events
        dphots[mask], thetaCh[mask] = self.CphotAng(beta[mask], altDec[mask])

        numPEs = (
            dphots
            * showerEnergy
            * self.config.detector.telescope_effective_area
            * self.config.detector.quantum_efficiency
        )

        enhanceFactor = numPEs / self.config.detector.photo_electron_threshold
        logenhanceFactor = np.empty_like(enhanceFactor)
        efMask = enhanceFactor > 2.0
        logenhanceFactor[efMask] = np.log(enhanceFactor[efMask])
        logenhanceFactor[~efMask] = 0.5

        hwfm = np.sqrt(2.0 * logenhanceFactor)
        thetaChEnh = np.multiply(thetaCh, hwfm)
        thetaChEff = np.where(thetaChEnh >= thetaCh, thetaChEnh, thetaCh)

        costhetaChEff = np.cos(np.radians(thetaChEff))

        if store is not None:
            store(
                ["altDec", "numPEs", "costhetaChEff"], [altDec, numPEs, costhetaChEff]
            )

        return numPEs, costhetaChEff
