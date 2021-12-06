import astropy.io.misc.hdf5 as hf
import numpy as np

from ...config import NssConfig
from ...utils import decorators

__all__ = ["EASRadio", "RadioEFieldParams", "IonosphereParams"]

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files


class EASRadio:
    """
    Extensive Air Shower for radio emission
    """

    def __init__(self, config: NssConfig):
        self.config = config

    def decay_to_detector_dist(self, beta, altDec, detectAlt, lenDec, viewAngle):
        Re = self.config.constants.earth_radius
        r1 = detectAlt + Re
        r2 = altDec + Re
        exit = np.pi / 2.0 - beta
        r2squared = r2 ** 2
        thetaE = (Re ** 2 + (Re + altDec) ** 2 - lenDec ** 2) / (2 * Re * (Re + altDec))
        thetaE[thetaE < 0] = 0
        thetaE[thetaE > 1] = 1
        thetaE = np.arccos(thetaE)
        thetaRel = exit - thetaE + viewAngle
        cosexit = np.cos(thetaRel)
        return (
            np.sqrt(r2squared * cosexit * cosexit - r2squared + r1 * r1) - r2 * cosexit
        )

    def get_decay_view(self, exitView, losDist, lenDec):
        """
        get view angle from shower to detector
        """
        # sin^2 of our decay view angle
        s2phi = (losDist * np.sin(exitView)) ** 2.0 / (
            lenDec * lenDec
            + losDist * losDist
            - 2 * losDist * lenDec * np.cos(exitView)
        )
        ang = np.arcsin(np.sqrt(s2phi))
        return ang

    @decorators.nss_result_store("EFields")
    def __call__(
        self, beta, altDec, lenDec, theta, pathLen, showerEnergy, *args, **kwargs
    ):
        """
        EAS radio output from ZHAires lookup tables
        """
        FreqRange = (self.config.detector.low_freq, self.config.detector.high_freq)
        radioParams = RadioEFieldParams(FreqRange)
        mask = (altDec < 0.0) | (
            altDec > 10.0
        )  # TODO set a reasonable cut for max shower height
        mask = ~mask

        viewAngles = np.zeros_like(theta)
        viewAngles[mask] = self.get_decay_view(theta[mask], pathLen[mask], lenDec[mask])

        # rudimentary distance scaling TODO investigate that this actually works with zhaires
        nssDist = self.decay_to_detector_dist(
            beta[mask],
            altDec[mask],
            self.config.detector.altitude,
            lenDec[mask],
            viewAngles[mask],
        )
        zhairesDist = self.decay_to_detector_dist(
            beta[mask], altDec[mask], 525.0, lenDec[mask], viewAngles[mask]
        )

        EFields = np.zeros_like(beta)
        EFields = radioParams(
            np.degrees(np.pi / 2.0 - beta), np.rad2deg(viewAngles), altDec
        )
        EFields = (EFields.T * mask).T

        # scale by the energy of the shower (all zhaires files are for 10^18 eV shower)
        # shower energy is in units of 100 PeV, we want in GeV
        EFields[mask] = (EFields[mask].T * showerEnergy[mask] / 10.0).T
        distScale = zhairesDist / nssDist
        EFields[mask] = (EFields[mask].T * distScale).T
        # no ionosphere if the detector is below 90km
        if self.config.detector.altitude > 90.0:
            if self.config.simulation.model_ionosphere:
                if self.config.simulation.TEC < 0:
                    print(
                        "TEC should be positive!! continuing without ionospheric dispersion"
                    )
                else:
                    ionosphere = IonosphereParams(
                        FreqRange,
                        self.config.simulation.TECerr,
                        self.config.simulation.TEC,
                    )
                    ionosphereScaling = ionosphere(EFields[mask])
                    EFields[mask] *= ionosphereScaling

        # radio emission from EAS has two components, geomagnetic and Askaryan
        # Askaryan is ~20% of full strength geomagnetic
        # Askaryan and geomagnetic components can have any phase w/r/t one another
        # geomagnetic is only full strength when perpendicular to Earth B-field
        # here i apply a scaling for vxB for an orbit close to equatorial

        Re = self.config.constants.earth_radius
        B_angle = np.ones(altDec[mask].shape)
        B_angle *= np.pi / 2.0 - np.arccos(
            (lenDec[mask] ** 2.0 + (altDec[mask] + Re) ** 2.0 - Re ** 2.0)
            / (2.0 * lenDec[mask] * (altDec[mask] + Re))
        )
        bounds = np.radians(30.0)
        B_angle += np.random.uniform(-1.0 * bounds, bounds, altDec[mask].shape)
        B_angle = np.abs(np.sin(B_angle))
        askaryan_phase = np.sin(
            np.random.uniform(0.0, -2.0 * np.pi, altDec[mask].shape)
        )
        EFields[mask] = (1.0 / 6.0 * EFields[mask].T * askaryan_phase).T + (
            5.0 / 6.0 * EFields[mask].T * B_angle
        ).T

        return EFields


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
        freqs = np.array(f["freqs"]).astype(str)
        tecs = np.array(f["TEC"])
        params = np.array(f["params"])
        desired_freqs = "f_{}_{}".format(int(self.lowFreq), int(self.highFreq))
        if desired_freqs not in freqs:
            self.params_exist = False
        if TEC not in tecs:
            self.params_exist = False
        if TECerr > 10.0:
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
        # i fit all of these with gaussians
        scale = (
            self.params[0]
            * np.exp(
                -((self.TEC + TECerr - self.params[1]) ** 2)
                / (2.0 * self.params[2] ** 2)
            )
            + self.params[3]
        )
        return scale / 100.0
