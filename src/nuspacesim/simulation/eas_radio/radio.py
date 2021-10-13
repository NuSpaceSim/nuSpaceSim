import numpy as np
import astropy.io.misc.hdf5 as hf
import astropy.table

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


class RadioEFieldParams(object):
    """
    Parametrization of ZHAireS upgoing air showers to obtain e field at a detector
    """

    def __init__(self, freqRange):
        """
        freq range is a 2 float tuple (in MHz)
        """
        param_dir = str(files("nuspacesim.data.radio_params"))
        fname = param_dir + "/waveform_params.hdf5"
        f = hf.read_table_hdf5(fname)
        self.lowFreq = int(freqRange[0])
        self.highFreq = int(freqRange[1])
        self.zeniths = np.array(f["zenith"])
        self.heights = np.array(f["height"])
        self.ps = np.array(f["params"])

    def __call__(
        self, zenith: np.ndarray, viewAngle: np.ndarray, h: np.ndarray
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
        j = np.argmin(zenith_diffs + h_diffs, axis=1)

        params = self.ps[j]

        fcenter = params[:, :, 0]
        cut = np.logical_and(fcenter >= self.lowFreq, fcenter <= self.highFreq)
        nrow = np.sum(cut[0])
        ncol = cut.shape[0]

        E0 = params[:, :, 1][cut].reshape(ncol, nrow)
        peak = params[:, :, 2][cut].reshape(ncol, nrow)
        w = params[:, :, 3][cut].reshape(ncol, nrow)
        E1 = params[:, :, 4][cut].reshape(ncol, nrow)
        w2 = params[:, :, 5][cut].reshape(ncol, nrow)

        viewAngle = (peak.T + viewAngle).T
        Efield = (
            E0 * np.exp(-((viewAngle - peak) ** 2) / (2.0 * w * w))
            + np.abs(E1) * np.exp(-(viewAngle ** 2) / (2.0 * w2 * w2)) / 2.0
        )  # this factor of 2 is to make up for how i made this fits in the first place
        return Efield


class IonosphereParams(object):
    """
    Parametrization of ionospheric dispersion degradation of overall SNR as a function of TEC
    """

    def __init__(self, freqRange, TECerr, TEC):
        """
        freq range is a 2 float tuple (in MHz)
        TECerr is
        """
        self.lowFreq = int(freqRange[0])
        self.highFreq = int(freqRange[1])
        self.TECerr = TECerr
        self.params_exist = True
        param_dir = str(files("nuspacesim.data.radio_params"))
        fname = param_dir + "/ionosphere_params.hdf5"
        f = hf.read_table_hdf5(fname)
        freqs = np.array(f['freqs']).astype(str)
        tecs = np.array(f['TEC'])
        params = np.array(f['params'])
        desired_freqs = 'f_{}_{}'.format(int(self.lowFreq), int(self.highFreq))
        desiredParams = (self.lowFreq, self.highFreq, self.TECerr)
        if desired_freqs not in freqs:
            self.params_exist = False
        if TEC not in tecs:
            self.params_exist = False
        if TECerr > 10.:
            self.params_exist = False
        if not self.params_exist:
            print(
                "",
                "***** WARNING *****",
                "The only supported parameters for ionospheric dispersion are:",
                "frequency ranges of 30-80 MHz, 30-300 MHz, 300-1000 MHz",
                "TEC values of 1, 5, 10, 50, 100, 150",
                "and TECerr values of < 10.",
                "Arbitrary values of these parameters will eventually be enabled, but not yet !!",
                "No scaling will be applied",
                "",
                sep="\n",
            )
        else:
            cut = np.logical_and(freqs == desired_freqs, tecs == TEC)
            self.TEC = tecs[cut][0]
            self.params = params[cut][0]

    def __call__(self, EFields):
        """
        TEC is the slant depth TEC in TECU
        """
        if not self.params_exist:
            return 1.0
        # if TEC == 0:
        #    return perfect_TEC_disperse(self.lowFreq*1.e6, self.highFreq*1.e6, TEC)/100.
        TECerr = np.random.uniform(-self.TECerr, self.TECerr, EFields.shape)
        #i fit all of these with gaussians
        scale = self.params[0] * np.exp(-(self.TEC+TECerr - self.params[1])**2/(2.*self.params[2]**2)) + self.params[3]
        return scale / 100.0
