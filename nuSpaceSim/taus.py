import importlib_resources
import h5py
import numpy as np
from scipy import interpolate


def extract_nutau_data(filename, lognuenergy):
    """
    Extract RenoNuTau tables into params
    """

    f = h5py.File(filename, "r")
    pegrp = f["pexitdata"]
    pegrppedset = pegrp["logPexit"]
    pegrpbdset = pegrp["BetaRad"]
    pegrplnedset = pegrp["logNuEnergy"]

    pegrppearr = np.array(pegrppedset)
    pegrpbarr = np.array(pegrpbdset)
    pegrplnearr = np.array(pegrplnedset)

    # testring = 'TauEdist_grp_e{:02.0f}_{:02.0f}'.format(np.floor(lognuenergy)
    # ,(lognuenergy - np.floor(lognuenergy))*100)
    # if lognuenergy >= 11.0:
    # Need some kind of exception here
    # elif:

    # lognuebin = float(closestNumber(np.rint(lognuenergy*100),25))/100.
    # If we want to do closest bin rather than histogram bins
    q = int((lognuenergy * 100) / 25.0)
    lognuebin = float((q * 25) / 100.0)

    #print (q, lognuebin)
    #testring = "TauEdist_grp_e{:02.0f}_{:02.0f}".format(
    #    np.floor(lognuebin), (lognuenergy - np.floor(lognuebin)) * 100
    #)
    testring = "TauEdist_grp_e{:02.0f}_{:02.0f}".format(
      np.floor(lognuebin), (lognuebin - np.floor(lognuebin)) * 100
    )

    #print(testring)

    tegrp = f[testring]
    tegrpcdfdset = tegrp["TauEDistCDF"]
    tegrpbdset = tegrp["BetaRad"]
    tegrptauedset = tegrp["TauEFrac"]

    tegrpcdfarr = np.array(tegrpcdfdset)
    tegrpbarr = np.array(tegrpbdset)

    teBetaLowBnds = (tegrpbarr[1:] + tegrpbarr[:-1]) / 2
    teBetaUppBnds = np.append(teBetaLowBnds, tegrpbarr[-1])
    teBetaLowBnds = np.insert(teBetaLowBnds, 0, 0.0, axis=0)

    tegrptauearr = np.array(tegrptauedset)
    f.close()

    return (
        pegrppearr,
        pegrpbarr,
        pegrplnearr,
        tegrpcdfarr,
        tegrpbarr,
        teBetaLowBnds,
        teBetaUppBnds,
        tegrptauearr,
    )


class Taus(object):
    """
    Describe Tau Module HERE!
    """

    def __init__(self, config):
        """
        Intialize the Taus object.
        """
        self.config = config

        ref = importlib_resources.files(
            'nuSpaceSim.DataLibraries.RenoNu2TauTables') / 'nu2taudata_v_2.hdf5'
        with importlib_resources.as_file(ref) as path:
            self.pearr, self.pebarr, self.pelnearr, self.tecdfarr, \
                self.tebarr, self.betaLowBnds, self.betaUppBnds, \
                self.tetauefracarr = extract_nutau_data(path,
                                                        config.logNuTauEnergy)

        self.peb_rbf = np.tile(self.pebarr, self.pelnearr.shape)
        self.pelne_rbf = np.repeat(self.pelnearr, self.pebarr.shape, 0)

        # Interpolating function to be used to find P_exit
        self.pexitfunc = interpolate.Rbf(self.peb_rbf,
                                         self.pelne_rbf,
                                         self.pearr,
                                         function="cubic",
                                         smooth=0)

        # Array of tecdf interpolation functions
        self.tauEFracInterps = [
            interpolate.interp1d(self.tecdfarr[:, betaind], self.tetauefracarr)
            for betaind in range(self.tecdfarr.shape[1])
        ]

        self.nuTauEnergy = self.config.nuTauEnergy

    def tau_exit_prob(self, betaArr):
        """
        Tau Exit Probability
        """
        brad = np.radians(betaArr)  # * (self.config.fundcon.pi / 180.0)

        logtauexitprob = self.pexitfunc(
            brad, np.full(brad.shape, self.config.logNuTauEnergy))

        tauexitprob = 10**logtauexitprob

        return tauexitprob

    def tau_energy(self, betaArr, u=None):
        """
        Tau energies interpolated from teCDF for given beta index.
        """
        u = np.random.rand(betaArr.shape[0]) if u is None else u

        # fast interpolation selection with masking
        betaIdxs = np.searchsorted(np.degrees(self.betaUppBnds), betaArr)

        tauEF = np.empty_like(betaArr)

        for i in range(self.tecdfarr.shape[1]):
            mask = betaIdxs == i
            tauEF[mask] = self.tauEFracInterps[i](u[mask])

        return tauEF * self.nuTauEnergy

    def __call__(self, betaArr):
        """
        Perform main operation for Taus module.

        Returns:

        """

        tauExitProb = self.tau_exit_prob(betaArr)
        tauEnergy = self.tau_energy(betaArr)

        # in units of 100 PeV
        showerEnergy = self.config.eShowFrac * tauEnergy / 1.0e8

        tauLorentz = tauEnergy / self.config.fundcon.massTau

        tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz**2))

        return tauBeta, tauLorentz, showerEnergy, tauExitProb
