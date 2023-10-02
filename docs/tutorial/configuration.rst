.. configuration:

##################
Configuration File
##################

.. toctree::
   :hidden:

Simulation configuration is governed by an XML file. The File structure is
separated into 2 primary sections, the DetectorCharacteristics and the
SimulationParameters. Both XML sections map 1-to-1 to the nuspacesim.config
dataclass objects of the same names.



DetectorCharacteristics
***********************

This is a dataclass holding the Detector Characteristics for a given simulation.
The member attributes are as follows:

  * **method**: Type of Detector, default = Optical
  * **altitude**: Altitude from sea level in km
  * **lat_start**: Latitude (Degrees | Radians)
  * **long_start**: Longitude (Degrees | Radians)
  * **telescope_effective_area**: Effective area of the detector scope. Default = 2.5 m^2
  * **quantum_efficiency**: Quantum Efficiency of the detector telescope. Default = 0.2
  * **photo_electron_threshold**: Threshold Number of Photo electrons. Default = 10
  * **low_freq**: Low end for radio band in MHz: Default = 30
  * **high_freq**: High end of radio band in MHz: Default = 300
  * **det_SNR_thres**: SNR threshold for radio triggering: Default = 5
  * **det_Nant**: Number of radio antennas: Default = 10
  * **det_gain**: Antenna gain in dB: Default = 1.8


SimulationParameters
********************

This is a dataclass holding the Detector Characteristics for a given simulation.
The member attributes are as follows:

  * **N**: Number of thrown Trajectories
  * **theta_ch_max**: Maximum Cherenkov Angle in radians. Default = π/60 radians (3 deg)
  * **spectrum**: Distribution from which to draw nu_tau energies. See `Spectrum Classes`_.
  * **e_shower_frac**: Fraction of ETau in Shower. Default = 0.5
  * **ang_from_limb**: Angle From Limb. Default = π/25.714 radians (7 degrees)
  * **max_azimuth_angle**: Maximum Azimuthal Angle. Default = 2π radians (360 degrees)
  * **model_ionosphere**: Model ionosphere for radio propagation?. Default = 0 (false)
  * **TEC**: Total Electron Content for ionospheric propagation. Default = 10
  * **TECerr**: Error for TEC reconstruction. Default = 0.1

Spectrum Classes
****************

Two Neutrino energy spectra type classes are implemented, with another having only stub
support. These configurations determine the energies of the neutrinos thrown in the 
simulation.

MonoSpectrum
============
Neutrino energies are a scalar constant.

  * **log_nu_tau_energy**: Log base 10 energy of the tau neutrinos in GeV.

PowerSpectrum
=============
Neutrino energies are drawn from a modified power law distribution


  * **index**: Power Law Log Energy of the tau neutrinos in GeV
  * **lower_bound**: Lower Bound Log nu_tau Energy GeV.
  * **upper_bound**: Upper Bound Log nu_tau Energy GeV.
