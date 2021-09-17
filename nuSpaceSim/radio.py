import numpy as np
import os as os
import importlib_resources
import nuSpaceSim.ionosphere.ionospheric_dispersion as iono
from scipy.interpolate import interp1d

class RadioEFieldParams(object):
    """
    Parametrization of ZHAireS upgoing air showers to obtain e field at a detector
    """
    def __init__(self, freqRange):
        """
        freq range is a 2 float tuple (in MHz) 
        """
        param_dir = str(importlib_resources.files("nuSpaceSim.zhaires_nss.params"))
        fname = param_dir + "/ZHAireS_params_10MHz_bins.npz"
        f = np.load(fname)
        self.lowFreq = int(freqRange[0])
        self.highFreq = int(freqRange[1])
        self.zeniths = f['zenith']
        self.heights = f['height']
        self.ps = f['params']
        f.close()

    def __call__(
            self, 
            zenith: np.ndarray, 
            viewAngle: np.ndarray,
            h: np.ndarray
            ) -> np.ndarray:
        """
        evaluate peak e field using this parameterization
        based on the zenith angle and the decay altitude of the tau.

        Parameters:
        zenith: np.ndarray
            zenith angle of the shower (in degrees)
        viewAngle:
            observer's view angle off of cherenkov peak (in degrees)
        h: np.ndarray
            decay height of the tau (in km)
        
        Returns:
        Efield: np.ndarray
            voltage at the detector for each shower
        """
      
        zenith_diffs = np.abs(np.subtract.outer(zenith, self.zeniths))
        h_diffs = np.abs(np.subtract.outer(h, self.heights))
        j = np.argmin(zenith_diffs + h_diffs, axis = 1)
        origView = viewAngle

        params = self.ps[j]

        fcenter = params[:,:,0]
        cut = np.logical_and(fcenter >= self.lowFreq, fcenter <= self.highFreq)
        nrow = np.sum(cut[0])
        ncol = cut.shape[0]

        E0 = params[:,:, 1][cut].reshape(ncol, nrow)
        peak = params[:,:, 2][cut].reshape(ncol, nrow)
        w = params[:,:, 3][cut].reshape(ncol, nrow)
        E1 = params[:,:, 4][cut].reshape(ncol, nrow)
        w2 = params[:,:, 5][cut].reshape(ncol, nrow)
       
        viewAngle = (peak.T + viewAngle).T
        Efield = E0 * np.exp(-( (viewAngle-peak )**2)/(2.*w*w)) + np.abs(E1) * np.exp(-viewAngle**2/(2.*w2*w2))/2. #this factor of 2 is to make up for how i made this fits in the first place
        return Efield

class IonosphereParams(object):
    """
    Parametrization of ionospheric dispersion degradation of overall SNR as a function of TEC
    """
    def __init__(self, freqRange, TECerr):
        """
        freq range is a 2 float tuple (in MHz) 
        TECerr is 
        """
        self.lowFreq = int(freqRange[0])
        self.highFreq = int(freqRange[1])
        self.TECerr = TECerr
        param_dir = str(importlib_resources.files("nuSpaceSim.ionosphere.tecparams"))
        fname = param_dir + "/f_{}_{}_TECerr_{}.npz".format(self.lowFreq, self.highFreq, self.TECerr)
        if not os.path.exists(fname):
            iono.generate_TEC_files(self.lowFreq,self.highFreq,self.TECerr)
        f = np.load(fname, allow_pickle=True)
        self.TECs = f['TEC']
        self.scaling = f['scaling']
        f.close()
    def __call__(self, TEC, EFields):
        """
        TEC is the slant depth TEC in TECU
        """
        if TEC == 0:
            return iono.perfect_TEC_disperse(self.lowFreq*1.e6, self.highFreq*1.e6, TEC)/100.
        ind = np.argmin(np.abs(self.TECs - TEC))
        TECerr = np.random.uniform(-self.TECerr, self.TECerr, EFields.shape)
        scale = self.scaling[ind](self.TECs[ind] + TECerr)
        return scale/100.
        

