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

"""
cphotang

Cherenkov photon density and angle determination class.
"""
import numpy as np
from numpy.polynomial import Polynomial

# from scipy import interpolate
import dask
import dask.bag as db
from dask.distributed import Client

# import multiprocessing
# from joblib import Parallel, delayed
from .zsteps import zsteps as cppzsteps

__all__ = ["CphotAng"]


class CphotAng:
    def __init__(self):
        """
        CphotAng: Cherenkov photon density and angle determination class.

        Iterative summation of cherenkov radiation reimplemented in numpy and
        C++.
        """
        self.dtype = np.float32
        self.wave1 = np.array(
            [
                200.0,
                225.0,
                250.0,
                275.0,
                300.0,
                325.0,
                350.0,
                375.0,
                400.0,
                425.0,
                450.0,
                475.0,
                500.0,
                525.0,
                550.0,
                575.0,
                600.0,
                625.0,
                650.0,
                675.0,
                700.0,
                725.0,
                750.0,
                775.0,
                800.0,
                825.0,
                850.0,
                875.0,
                900.0,
            ],
            dtype=self.dtype,
        )

        self.OzZeta = np.array(
            [5.35, 10.2, 14.75, 19.15, 23.55, 28.1, 32.8, 37.7, 42.85, 48.25, 100.0],
            dtype=self.dtype,
        )

        self.OzDepth = np.array(
            [15.0, 9.0, 10.0, 31.0, 71.0, 87.2, 57.0, 29.4, 10.9, 3.2, 1.3],
            dtype=self.dtype,
        )

        self.OzDsum = np.array(
            [310.0, 301.0, 291.0, 260.0, 189.0, 101.8, 44.8, 15.4, 4.5, 1.3, 0.1],
            dtype=self.dtype,
        )

        self.aOD55 = np.array(
            [
                0.250,
                0.136,
                0.086,
                0.065,
                0.055,
                0.049,
                0.045,
                0.042,
                0.038,
                0.035,
                0.032,
                0.029,
                0.026,
                0.023,
                0.020,
                0.017,
                0.015,
                0.012,
                0.010,
                0.007,
                0.006,
                0.004,
                0.003,
                0.003,
                0.002,
                0.002,
                0.001,
                0.001,
                0.001,
                0.001,
            ],
            dtype=self.dtype,
        )

        # self.aOD55_interp = interpolate.interp1d(np.arange(30)+1, self.aOD55,
        #                                          kind='linear',
        #                                          bounds_error=False,
        #                                          fill_value=0,
        #                                          assume_sorted=True)

        # self.OzDsum_interp = interpolate.interp1d(self.OzZeta, self.OzDsum,
        #                                           kind='linear',
        #                                           bounds_error=False,
        #                                           fill_value=0.1,
        #                                           assume_sorted=True)

        self.dL = self.dtype(0.1)  # parin(1) step size km
        self.orbit_height = self.dtype(525.0)  # parin(2) orbit height km
        self.hist_bin_size = self.dtype(4.0)  # parin(3) bin size histogram km
        # parin(5) record time dispersion at
        self.time_disp_rec_point = self.dtype(0.5)
        # this radial point (km)
        self.logE = self.dtype(17.0)  # parin(8) log(E) E in eV

        # c  parameters for 1/Beta fit vs wavelength
        # c     5th order polynomial
        aP = Polynomial(
            np.array(
                [
                    -1.2971,
                    0.22046e-01,
                    -0.19505e-04,
                    0.94394e-08,
                    -0.21938e-11,
                    0.19390e-15,
                ],
                dtype=self.dtype,
            )
        )

        self.wmean = self.wave1[:-1] + self.dtype(12.5)
        tBetinv = aP(self.wmean)
        self.aBetaF = np.reciprocal(tBetinv, dtype=self.dtype)
        self.aBetaF /= self.dtype(0.158)
        # self.aBeta55 = self.dtype(0.158)

        # c            Ozone Trans = exp(-kappa dx)
        # c               where dx=ozone slant depth in atm-cm
        # c               and kappa = 110.5 x wave^(-44.21) in atm-cm^-1
        self.Okappa = np.log10(self.wmean, dtype=self.dtype)
        self.Okappa *= self.dtype(44.21)
        self.Okappa = self.dtype(110.5) - self.Okappa
        self.Okappa = np.power(10.0, self.Okappa, dtype=self.dtype)
        self.Okappa *= self.dtype(-1e-3)

        # c
        # c calc OD/km difference
        # c
        # self.dfaOD55 = np.diff(self.aOD55[::-1], append=0)
        tmp = [self.dtype(self.aOD55[i] - self.aOD55[i + 1]) for i in range(29)]
        tmp.append(self.dtype(0))
        self.dfaOD55 = np.array(tmp, dtype=self.dtype)
        # np.append(self.dfaOD55, 0)

        self.alpha = np.reciprocal(self.dtype(137.04))
        self.pi = self.dtype(3.1415926)

        self.PYieldCoeff = (
            self.dtype(2e12)
            * self.dL
            * self.pi
            * self.alpha
            * (np.reciprocal(self.wave1)[:-1] - np.reciprocal(self.wave1)[1:])
        )

        self.zmax = self.orbit_height
        self.zMaxZ = self.dtype(65.0)
        self.RadE = self.dtype(6378.14)

        self.Eshow = self.dtype(10.0 ** (self.logE - 9.0))
        Zair = self.dtype(7.4)
        ecrit = self.dtype(0.710 / (Zair + 0.96))

        self.beta = self.dtype(np.log(np.float64(self.Eshow) / np.float64(ecrit)))

        # c     Calc ang spread ala Hillas
        self.Ieang = int(np.log10(self.Eshow)) - 2 + 3
        eang = np.arange(self.dtype(1.0), self.Ieang + self.dtype(2))
        self.ehill = np.power(10.0, eang, dtype=self.dtype)

    def theta_view(self, ThetProp):
        """
        Compute theta view from initial betas
        """
        # ThetProp = np.radians(betaE)
        ThetView = self.RadE / (self.RadE + self.zmax)
        ThetView *= np.cos(ThetProp, dtype=self.dtype)
        ThetView = np.arcsin(ThetView, dtype=self.dtype)
        return ThetView

    def grammage(self, z):
        """
        # c     Calculate Grammage
        """
        X = np.empty_like(z, dtype=self.dtype)
        mask1 = z < 11
        mask2 = np.logical_and(z >= 11, z < 25)
        mask3 = z >= 25
        X[mask1] = np.power(((z[mask1] - 44.34) / -11.861), (1 / 0.19))
        X[mask2] = np.exp(
            np.divide(z[mask2] - 45.5, -6.34, dtype=self.dtype), dtype=self.dtype
        )
        X[mask3] = np.exp(
            np.subtract(
                13.841,
                np.sqrt(28.920 + 3.344 * z[mask3], dtype=self.dtype),
                dtype=self.dtype,
            )
        )

        rho = np.empty_like(z, dtype=self.dtype)
        rho[mask1] = (
            self.dtype(-1.0e-5)
            * (1 / 0.19)
            / (-11.861)
            * ((z[mask1] - 44.34) / -11.861) ** ((1.0 / 0.19) - 1.0)
        )
        rho[mask2] = np.multiply(
            -1e-5 * np.reciprocal(-6.34), X[mask2], dtype=self.dtype
        )
        rho[mask3] = np.multiply(
            np.divide(
                0.5e-5 * 3.344,
                np.sqrt(28.920 + 3.344 * z[mask3], dtype=self.dtype),
                dtype=self.dtype,
            ),
            X[mask3],
            dtype=self.dtype,
        )
        return X, rho

    def ozone_losses(self, z):
        """
        Calculate ozone losses from altitudes (z) in km.
        """
        msk1 = z < 5.35
        TotZon = np.empty_like(z)
        TotZon[msk1] = self.dtype(310) + (
            (self.dtype(5.35) - z[msk1]) / self.dtype(5.35)
        ) * self.dtype(15)
        msk2 = z >= 100
        TotZon[msk2] = self.dtype(0.1)

        msk3 = np.logical_and(~msk1, ~msk2)
        idxs = np.searchsorted(self.OzZeta, z[msk3])
        TotZon[msk3] = (
            self.OzDsum[idxs]
            + (
                (self.OzZeta[idxs] - z[msk3])
                / (self.OzZeta[idxs] - self.OzZeta[idxs - 1])
            )
            * self.OzDepth[idxs]
        )
        return TotZon

    def theta_prop(self, z, sinThetView):
        """
        theta propagation.
        """
        # aa = np.sin(ThetView, dtype=self.dtype)
        tp = (self.RadE + self.zmax) / (self.RadE + z)
        return np.arccos(sinThetView * tp, dtype=self.dtype)

    # def delta_z(self, z, ThetProp):
    #     '''
    #     Change in z.
    #     '''
    #     Rad = z + self.RadE
    #     return np.sqrt((Rad*Rad) + (self.dL*self.dL) -
    #                    self.dtype(2)*Rad*self.dL*np.cos(
    #                        (self.pi/self.dtype(2)) +
    #         ThetProp, dtype=self.dtype),
    #         dtype=self.dtype) - Rad

    def zsteps(self, z, sinThetView):
        """
        Compute all mid-bin z steps and corresponding delz values
        """
        # zsave = []
        # delzs = []
        # while (z <= self.zMaxZ):
        #     # c  correct ThetProp for starting altitude
        #     ThetProp = self.theta_prop(z, sinThetView)
        #     delz = self.delta_z(z, ThetProp)
        #     delzs.append(delz)
        #     zsave.append(z+delz/self.dtype(2.))
        #     z += delz

        # zsave = np.array(zsave)
        # delzs = np.array(delzs)
        return cppzsteps(
            z, sinThetView, self.RadE, self.zMaxZ, self.zmax, self.dL, self.pi
        )

        # return zsave, delzs

    def slant_depth(self, alt, sinThetView):
        """Determine Rayleigh and Ozone slant depth."""

        zsave, delzs = self.zsteps(alt, sinThetView)
        gramz, rhos = self.grammage(zsave)

        delgram_vals = rhos * self.dL * self.dtype(1e5)
        gramsum = np.cumsum(delgram_vals)
        delgram = np.cumsum(delgram_vals[::-1])[::-1]

        TotZons = self.ozone_losses(np.insert(zsave, 0, alt))
        ZonZ_vals = (TotZons[:-1] - TotZons[1:]) / delzs * self.dL
        ZonZ = np.cumsum(ZonZ_vals[::-1])[::-1]

        ThetPrpA = self.theta_prop(zsave, sinThetView)

        return zsave, delgram, gramsum, gramz, ZonZ, ThetPrpA

    def aerosol_model(self, z, ThetPrpA):
        """
        Put in aerosol model based on 550 nm Elterman results.

        Use scipy linear interpolation function initialized in constructor.

        z values above 30 (km altitude) return OD filled to 0, this should then
        return aTrans = 1, but in the future masking may be used to further
        optimze for performance by avoiding this computation.
        """

        aTrans = np.ones((*z.shape, *self.aBetaF.shape), dtype=self.dtype)

        tmpOD = (
            self.aOD55[np.int32(z[z < 30])]
            - (z[z < 30] - self.dtype(np.int32(z[z < 30])))
            * self.dfaOD55[np.int32(z[z < 30])]
        )

        aODepth = -np.outer(tmpOD, self.aBetaF)
        costhet = np.cos(self.pi / 2 - ThetPrpA[z < 30], dtype=self.dtype)

        aTrans[z < 30, :] = np.exp((aODepth / costhet[:, None]), dtype=self.dtype)

        return aTrans

    def tracklen(self, E0, eCthres, s):
        """Return tracklength and Tfrac."""
        t1 = np.subtract(np.multiply(0.89, E0, dtype=self.dtype), 1.2, dtype=self.dtype)
        t2 = np.divide(t1, (E0 + eCthres), dtype=self.dtype)
        t3 = np.power(t2, s, dtype=self.dtype)
        t4 = np.power(1 + self.dtype(1e-4 * s * eCthres), 2, dtype=self.dtype)
        return np.divide(t3, t4, dtype=self.dtype)

    def sphoton_yeild(self, thetaC, RN, delgram, ZonZ, z, ThetPrpA):

        # c      Calculate Light Yield
        PYield = np.sin(thetaC, dtype=self.dtype)
        PYield = np.power(PYield, 2, dtype=self.dtype)
        PYield = PYield[..., None] * self.PYieldCoeff[None, :]

        # c      Calculate Losses due to Rayleigh Scattering
        TrRayl = np.divide(-delgram, 2974, dtype=self.dtype)
        TrRb = np.divide(400, self.wmean, dtype=self.dtype)
        TrRb = np.power(TrRb, 4, dtype=self.dtype)
        TrRayl = TrRayl[..., None] * TrRb[None, :]
        TrRayl = np.exp(TrRayl, dtype=self.dtype)

        # c        Calculate Ozone Losses
        # c          Ozone atten parameter given by R. McPeters
        TrOz = np.exp(ZonZ[:, None] * self.Okappa[None, :], dtype=self.dtype)

        # c put in aerosol model based on 550 nm
        # c     Elterman results
        aTrans = self.aerosol_model(z, ThetPrpA)

        # # Scaled Photon Yield
        SPYield = PYield * TrRayl * TrOz * aTrans * RN[..., None]

        return SPYield

    def photon_sum(self, SPYield, DistStep, thetaC, e2hill, eCthres, Tfrac, E0, s):
        """
        Sum photons (Gaisser-Hillas)
        """
        sigval = np.divide(
            SPYield,
            np.power(1e3 * self.hist_bin_size, 2, dtype=self.dtype),
            dtype=self.dtype,
        )

        # c   set limits by distance to det
        # c     and Cherenkov Angle
        CradLim = DistStep * np.tan(thetaC, dtype=self.dtype)
        jlim = np.floor(CradLim) + 1
        max_jlim = np.amax(jlim)
        jstep = np.arange(max_jlim)
        jjstep = np.broadcast_to(jstep, (*CradLim.shape, *jstep.shape))
        jmask = jjstep < jlim[..., None]

        athetaj = jjstep[:, 1:] - 0.5
        athetaj = np.arctan2(athetaj, DistStep[:, None], dtype=self.dtype)
        athetaj = 2.0 * (1.0 - np.cos(athetaj, dtype=self.dtype))

        sthetaj = np.arctan2(jjstep, DistStep[:, None], dtype=self.dtype)
        sthetaj = 2.0 * (1.0 - np.cos(sthetaj, dtype=self.dtype))

        # c     Calc ang spread ala Hillas
        ehillave = np.where(
            eCthres[..., None] >= self.ehill[:-1][None, :],
            (eCthres[..., None] + self.ehill[1:][None, :]) / self.dtype(2),
            self.dtype(5) * self.ehill[:-1][None, :],
        )

        tlen = np.where(
            eCthres[..., None] >= self.ehill[None, :],
            Tfrac[..., None],
            self.tracklen(E0[..., None], self.ehill[None, :], s[:, None]),
        )

        deltrack = tlen[..., :-1] - tlen[..., 1:]
        deltrack[deltrack < 0] = self.dtype(0.0)

        vhill = ehillave / e2hill[..., None]

        wave = self.dtype(
            0.0054 * ehillave * (1 + vhill) / (1 + 13 * vhill + 8.3 * vhill ** 2)
        )

        poweha = np.power(ehillave / 21.0, 2, dtype=self.dtype)

        uhill = np.einsum("zj,ze->zje", athetaj, poweha, dtype=self.dtype)
        uhill /= wave[..., None, :]
        ubin = np.einsum("zj,ze->zje", sthetaj, poweha, dtype=self.dtype)
        ubin /= wave[..., None, :]
        ubin = ubin[..., 1:, :] - ubin[..., :-1, :]
        ubin[ubin < 0] = self.dtype(0)

        # z0hill = self.dtype(0.59)
        # ahill = self.dtype(0.777)

        xhill = np.sqrt(uhill, dtype=self.dtype) - self.dtype(0.59)

        svtrm = np.where(xhill < 0, self.dtype(0.478), self.dtype(0.380))
        svtrm = np.exp(-xhill / svtrm)
        svtrm *= ubin * deltrack[..., None, :] * self.dtype(0.777)
        svtrm[~jmask[..., 1:]] = self.dtype(0)

        photsum = np.einsum("zje,zw->", svtrm, sigval, dtype=self.dtype)
        photsum *= np.power(1e3 * self.hist_bin_size, 2, dtype=self.dtype)

        return photsum

    def valid_arrays(self, zsave, delgram, gramsum, gramz, ZonZ, ThetPrpA):
        """
        Return data arrays with invalid values removed
        """
        mask = zsave <= self.zmax

        AirN = np.empty_like(zsave, dtype=self.dtype)
        AirN[mask] = 1.0 + 0.000296 * (gramz[mask] / 1032.9414) * (
            273.2 / (204.0 + 0.091 * gramz[mask])
        )

        mask &= (AirN != 1) & (AirN != 0)

        # c  do greissen param
        t = np.zeros_like(zsave, dtype=self.dtype)
        t[mask] = gramsum[mask] / self.dtype(36.66)

        s = np.zeros_like(zsave, dtype=self.dtype)
        s[mask] = self.dtype(3) * t[mask] / (t[mask] + self.dtype(2) * self.beta)

        RN = np.zeros_like(zsave, dtype=self.dtype)
        RN[mask] = (
            self.dtype(0.31)
            / np.sqrt(self.beta, dtype=self.dtype)
            * np.exp(
                t[mask] * (1 - self.dtype(3 / 2) * np.log(s[mask], dtype=self.dtype)),
                dtype=self.dtype,
            )
        )
        RN[RN < 0] = self.dtype(0)

        mask &= ~((RN < 1) & (s > 1))

        e2hill = np.zeros_like(zsave, dtype=self.dtype)
        e2hill[mask] = self.dtype(1150) + self.dtype(454) * np.log(
            s[mask], dtype=self.dtype
        )
        mask &= ~(e2hill <= 0)

        # final mask set for loop

        zs = zsave[mask]
        delgram = delgram[mask]
        ZonZ = ZonZ[mask]
        ThetPrpA = ThetPrpA[mask]
        AirN = AirN[mask]
        s = s[mask]
        RN = RN[mask]
        e2hill = e2hill[mask]

        return zs, delgram, ZonZ, ThetPrpA, AirN, s, RN, e2hill

    def e0(self, shape, s):
        """not sure what E0 is?"""
        E0 = np.full(shape, 26.0, dtype=self.dtype)
        E0[s >= 0.4] = 44.0 - 17.0 * (s[(s >= 0.4)] - 1.46) ** 2
        return E0

    def cherenkov_threshold_angle(self, AirN):
        """Calc Cherenkov Threshold and Cherenkov angle."""
        eCthres = np.reciprocal(np.power(AirN, 2))
        eCthres = np.sqrt(1.0 - eCthres, dtype=self.dtype)
        eCthres = np.divide(self.dtype(0.511), eCthres)
        # c  Calculate Cerenkov Angle
        thetaC = np.arccos(np.reciprocal(AirN), dtype=self.dtype)
        return eCthres, thetaC

    def distance_to_detector(self, ThetView, ThetPrpA, zs):
        """Distance to detector."""
        AngE = self.pi / (2) - ThetView - ThetPrpA
        DistStep = np.sin(AngE, dtype=self.dtype)
        DistStep /= np.sin(ThetView, dtype=self.dtype)
        DistStep *= self.RadE + zs
        return DistStep

    def cher_ang_sig_i(self, taphotstep, taphotsum, thetaC, AveCangI):
        """ """
        nAcnt = np.count_nonzero(taphotstep * thetaC, axis=-1)
        CangsigI = taphotstep / taphotsum
        CangsigI *= np.power(thetaC - AveCangI, 2, dtype=self.dtype)
        CangsigI = np.sum(CangsigI, axis=-1, dtype=self.dtype)
        CangsigI = np.sqrt(CangsigI * nAcnt / (nAcnt - 1), dtype=self.dtype)
        return CangsigI

    def cherenkov_area(self, AveCangI, DistStep, izRNmax):
        CherArea = np.tan(AveCangI, dtype=self.dtype) * self.dtype(1e3)
        CherArea *= DistStep[izRNmax]
        CherArea = self.pi * np.power(CherArea, 2, dtype=self.dtype)
        return CherArea

    def run(self, betaE, alt):
        """Main body of simulation code."""

        # Should we just skip these with a mask in valid_arrays?
        betaE = self.dtype(
            np.radians(self.dtype(1)) if betaE < np.radians(1.0) else betaE
        )

        ThetView = self.theta_view(betaE)
        sinThetView = np.sin(ThetView, dtype=self.dtype)
        #
        # Shower
        #

        zs, delgram, ZonZ, ThetPrpA, AirN, s, RN, e2hill = self.valid_arrays(
            *self.slant_depth(alt, sinThetView)
        )

        izRNmax = np.argmax(RN, axis=-1)
        E0 = self.e0(zs.shape, s)

        # c  Calc Cherenkov Threshold
        eCthres, thetaC = self.cherenkov_threshold_angle(AirN)

        Tfrac = self.tracklen(E0, eCthres, s)

        # c
        # c    Determine geometry
        # c

        # distance to detector
        DistStep = self.distance_to_detector(ThetView, ThetPrpA, zs)

        # Scaled Photon Yield
        SPYield = self.sphoton_yeild(thetaC, RN, delgram, ZonZ, zs, ThetPrpA)

        # Total photons
        photsum = self.photon_sum(
            SPYield, DistStep, thetaC, e2hill, eCthres, Tfrac, E0, s
        )

        taphotstep = np.sum(SPYield, axis=-1, dtype=self.dtype) * Tfrac

        taphotsum = np.sum(taphotstep, axis=-1, dtype=self.dtype)

        AveCangI = np.sum(taphotstep * thetaC, axis=-1, dtype=self.dtype) / taphotsum

        CangsigI = self.cher_ang_sig_i(taphotstep, taphotsum, thetaC, AveCangI)

        CherArea = self.cherenkov_area(AveCangI, DistStep, izRNmax)

        photonDen = self.dtype(0.5) * photsum / CherArea
        Cang = np.degrees(AveCangI + CangsigI)

        return photonDen, Cang

    def __call__(self, betaE, alt, client_input=None):
        """
        Iterate over the list of events and return the result as pair of
        numpy arrays.
        """

        if client_input is not None:
            client = Client(client_input)

        #######################
        b = db.from_sequence(zip(betaE, alt), partition_size=100)
        Dphots, Cang = zip(*b.map(lambda x: self.run(*x)).compute())
        return np.asarray(Dphots), np.array(Cang)
