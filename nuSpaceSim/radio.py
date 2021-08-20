import numpy as np
import os as os
#from nuSpaceSim.zhaires_nss.wfm_processor import wfmProcessor
#import scipy.interpolate as interp
#from supersmoother import SuperSmoother

class RadioEFieldParams(object):
    """
    Parametrization of ZHAireS upgoing air showers to obtain e field at a detector
    """
    def __init__(self, freqRange):
        """
        freq range is a 2 float tuple (in MHz) 
        """
        param_dir = "/home/abl/nuspacesim/nuSpaceSim/zhaires_nss/params/"
        fname = param_dir + "ZHAireS_params_10MHz_bins.npz"
        f = np.load(fname)
        self.lowFreq = int(freqRange[0])
        self.highFreq = int(freqRange[1])
        self.zeniths = f['zenith']
        self.heights = f['height']
        self.ps = f['params']
        f.close()
        #fname = param_dir + "zhs_z_h_viewangle_efield_spline.npz"
        #f2 = np.load(fname, allow_pickle=True)
        #self.freqs = f2['freq']
        #self.interps = f2['interps']
        #f2.close()
        #fname = param_dir + "zhaires_smooth.npz"
        #f2 = np.load(fname, allow_pickle=True)
        #self.smooths = f2['params']
        #f2.close()

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
        E0 = params[:,:, 1] * cut
        peak = params[:,:, 2]
        w = params[:,:, 3]
        E1 = params[:,:, 4] * cut
        w2 = params[:,:, 5]
       
        viewAngle = (peak.T + viewAngle).T
        Efield = np.sum(E0 * np.exp(-( (viewAngle-peak )**2)/(2.*w*w)) + np.abs(E1) * np.exp(-viewAngle**2/(2.*w2*w2)), axis=1)/2. #this factor of 2 is to make up for how i made this fits in the first place

        #factor of ~5-10 difference in this (plus its really slow)
        #cut2 = np.logical_and(self.freqs >= self.lowFreq, self.freqs <= self.highFreq)
        #Efield2 = np.zeros(Efield.shape)
        #apply_vectorized = np.vectorize(lambda f, x, y, z: f(x,y,z), otypes=[object])
        #for i in range(h.shape[0]):
        #    Efield2[i] = np.abs(np.sum(apply_vectorized(self.interps[cut2], zenith[i], h[i], origView[i])))
        #Efield2[np.isnan(Efield2)] = 0.

        #smooth
        #smoothers = self.smooths[j]
        #cut2 = np.logical_and(smoothers[:,:,0] >= self.lowFreq, smoothers[:,:,0] <= self.highFreq)
        #Efield2 = np.zeros(Efield.shape)
        #apply_vectorized = np.vectorize(lambda f, x: f.predict(x), otypes=[object])
        #for i in range(h.shape[0]):
        #    Efield2[i] = np.sum(apply_vectorized(smoothers[:,:,1][cut2], origView[i]))
        return Efield
        

