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
Module contains functions for parsing and interacting with XML configuration files.
"""

from dataclasses import dataclass, field

import lxml.etree as ET
import numpy as np
from astropy import units as u

from .. import constants as const
from ..config import (
    DetectorCharacteristics,
    FileSpectrum,
    MonoSpectrum,
    NssConfig,
    PowerSpectrum,
    SimulationParameters,
)
from . import config_xml_schema

__all__ = [
    "is_valid_xml",
    "parse_detector_chars",
    "parse_simulation_params",
    "parseXML",
    "config_from_xml",
    "create_xml",
]


def is_valid_xml(xmlfile: str) -> bool:
    r"""Check that the given xml file is valid.

    Confirm that the given xml document is valid by validating it with the XMLSchema.

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.

    Returns
    -------
    bool
        Whether the file is valid.
    """

    xmlschema_doc = ET.parse(config_xml_schema.xsd)
    xmlschema = ET.XMLSchema(xmlschema_doc)
    return xmlschema.validate(ET.parse(xmlfile))


@dataclass
class BaseUnits:
    """
    Class that defines the base units used in the rest of the
    program and a function to automatically convert to these units
    """

    # Additional unit definitions
    mHz = u.def_unit("mHz", u.Hz / 1000)
    kHz = u.def_unit("kHz", u.Hz * 1000)
    MHz = u.def_unit("MHz", kHz * 1000)
    GHz = u.def_unit("GHz", MHz * 1000)
    month = u.def_unit("month", u.day * 30)

    # List of the base units
    energy_base: u.Quantity = u.eV
    time_base: u.Quantity = u.second
    distance_base: u.Quantity = u.km
    angle_base: u.Quantity = u.rad
    frequency_base: u.Quantity = MHz
    area_base: u.Quantity = u.m * u.m

    # Declare the allowed units
    energy_units: list = field(default_factory=list)
    time_units: list = field(default_factory=list)
    distance_units: list = field(default_factory=list)
    angle_units: list = field(default_factory=list)
    frequency_units: list = field(default_factory=list)
    area_units: list = field(default_factory=list)

    def __post_init__(self):
        """Function to define the units that are allowed as input"""
        self.energy_units = ["eV", "keV", "MeV", "GeV", "TeV", "PeV", "EeV", "J"]
        self.time_units = ["sec", "min", "hour", "day", "month", "year"]
        self.distance_units = ["mm", "cm", "m", "km", "lyr", "pc"]
        self.angle_units = ["arcsec", "arcmin", "deg", "rad", "Degrees", "Radians"]
        self.frequency_units = ["mHz", "Hz", "kHz", "MHz", "GHz"]
        self.area_units = ["Sq.Meters"]

    def unit_conversion(self, value: float, unit: str) -> float:
        """
        Function to convert the given value and unit to the corresponding
        base unit
        """
        if unit in self.energy_units:
            quantity = u.Quantity(value, unit)
            return quantity.to(self.energy_base).value
        if unit in self.time_units:
            if unit == "sec":
                quantity = u.Quantity(value, u.s)
            else:
                quantity = u.Quantity(value, unit)
            return quantity.to(self.time_base).value
        if unit in self.distance_units:
            quantity = u.Quantity(value, unit)
            return quantity.to(self.distance_base).value
        if unit in self.angle_units:
            if unit == "Degrees":
                unit = "deg"
            if unit == "Radians":
                unit = "rad"
            quantity = u.Quantity(value, unit)
            return quantity.to(self.angle_base).value
        if unit in self.frequency_units:
            quantity = u.Quantity(value, unit)
            return quantity.to(self.frequency_base).value
        if unit in self.area_units:
            if unit == "Sq.Meters":
                return float(value)
        raise Exception(
            f"\
            Unit {unit} has not been found!\n\
            For energies use: {self.energy_units}\n\
            For times use: {self.time_units}\n\
            For distances use: {self.distance_units}\n\
            For angles use: {self.angle_units}\n\
            For frequencies use: {self.frequency_units}\n\
            For areas use: {self.area_units}\n\
            "
        )


def check_unit(node, units: BaseUnits):
    if "Unit" in node.attrib:
        return units.unit_conversion(node.text, node.attrib["Unit"])
    return str(node.text)


def parse_detector_chars(xmlfile: str) -> DetectorCharacteristics:
    r"""Parse the XML file into a DetectorCharacteristics object and if possible apply
    unit conversions

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.

    Returns
    -------
    DetectorCharacteristics
        The detector characteristics object.
    """
    units = BaseUnits()
    detchar: dict[str, str] = {}
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    eldetchar = root.find("DetectorCharacteristics")
    detchar["Method"] = eldetchar.attrib["Method"]

    """Define a preset value for the unneeded parameters"""
    detchar["SunMoonCuts"] = False
    detchar["SunAngleBelowHorizonCut"] = 0
    detchar["MoonAngleBelowHorizonCut"] = 0
    detchar["MoonMinPhaseAngleCut"] = 0

    for node in tree.find("./DetectorCharacteristics"):

        if node.tag == "PhotoElectronThreshold":
            detchar[node.tag] = str(node.attrib["Preset"])
            if node.attrib["Preset"] == "true":
                detchar["NPE"] = str(node.find("NPE").text)

        elif node.tag == "SunMoonCuts":
            try:
                detchar[node.tag] = True
                detchar["SunAngleBelowHorizonCut"] = check_unit(
                    node.find("SunAngleBelowHorizonCut"), units
                )
                detchar["MoonAngleBelowHorizonCut"] = check_unit(
                    node.find("MoonAngleBelowHorizonCut"), units
                )
                detchar["MoonMinPhaseAngleCut"] = check_unit(
                    node.find("MoonMinPhaseAngleCut"), units
                )

            except AttributeError:
                raise Exception(
                    "Please provide cut values for: "
                    + '"SunAngleBelowHorizonCut", "SunAngleBelowHorizonCut" and "MoonMinPhaseAngleCut" '
                    + "If only a subset are needed provide values for those and use default values of (0, 0, 0) "
                    + "for the other two."
                )

        else:
            detchar[node.tag] = check_unit(node, units)

    return DetectorCharacteristics(
        method=detchar["Method"],
        altitude=detchar["DetectorAltitude"],
        detlat=detchar["InitialDetectorLatitude"],
        detlong=detchar["InitialDetectorLongitude"],
        telescope_effective_area=detchar["TelescopeEffectiveArea"],
        quantum_efficiency=float(detchar["QuantumEfficiency"]),
        photo_electron_threshold=float(detchar["NPE"]),
        low_freq=detchar["LowFrequency"],
        high_freq=detchar["HighFrequency"],
        det_SNR_thres=float(detchar["SNRThreshold"]),
        det_Nant=int(detchar["NAntennas"]),
        det_gain=float(detchar["AntennaGain"]),
        sun_moon_cuts=detchar["SunMoonCuts"],
        sun_alt_cut=detchar["SunAngleBelowHorizonCut"],
        moon_alt_cut=detchar["MoonAngleBelowHorizonCut"],
        MoonMinPhaseAngleCut=detchar["MoonMinPhaseAngleCut"],
    )


def parse_simulation_params(xmlfile: str) -> SimulationParameters:
    r"""Parse the XML file into a SimulationParameters object.

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.

    Returns
    -------
    SimulationParameters
        The Simulation Parameters object.
    """
    units = BaseUnits()
    simparams = {}
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    elsimparams = root.find("SimulationParameters")
    simparams[elsimparams.tag] = elsimparams.attrib["DetectionMode"]

    """Define a preset value for the unneeded parameters"""
    simparams["SourceRightAscension"] = 0
    simparams["SourceDeclination"] = 0
    simparams["SourceDate"] = "0000-00-00T00:00:00"
    simparams["SourceDateFormat"] = "isot"
    simparams["ObservationPeriod"] = 0

    for node in tree.find("./SimulationParameters"):
        if node.tag == "TauShowerType":
            simparams[node.tag] = node.attrib["Preset"]
            if node.attrib["Preset"] == "true":
                simparams["FracETauInShower"] = str(node.find("FracETauInShower").text)

        elif node.tag == "NuTauEnergySpecType":
            for spectrum_type in node:
                if "MonoSpectrum" == spectrum_type.tag:
                    simparams["Spectrum"] = MonoSpectrum(
                        log_nu_tau_energy=float(spectrum_type.find("LogNuEnergy").text),
                    )
                if "PowerSpectrum" == spectrum_type.tag:
                    simparams["Spectrum"] = PowerSpectrum(
                        index=float(spectrum_type.find("PowerLawIndex").text),
                        lower_bound=float(spectrum_type.find("LowerBound").text),
                        upper_bound=float(spectrum_type.find("UpperBound").text),
                    )
                if "FileSpectrum" == spectrum_type.tag:
                    simparams["Spectrum"] = FileSpectrum(
                        path=str(node.spectrum_type("FilePath").text)
                    )

        elif node.tag == "ToOSourceParameters":
            if simparams["SimulationParameters"] == "ToO":
                try:
                    simparams["SourceRightAscension"] = check_unit(
                        node.find("SourceRightAscension"), units
                    )
                    simparams["SourceDeclination"] = check_unit(
                        node.find("SourceDeclination"), units
                    )
                    simparams["SourceDate"] = node.find("SourceDate").text
                    simparams["SourceDateFormat"] = node.find("SourceDate").attrib[
                        "Format"
                    ]
                    simparams["ObservationPeriod"] = check_unit(
                        node.find("ObservationPeriod"), units
                    )

                except AttributeError:
                    raise Exception(
                        '\
                    Please provide values for: \
                    "SourceRightAscension", "SourceRightAscension",\
                    "SourceDate" and "ObservationPeriod"'
                    )
            else:
                simparams["SourceRightAscension"] = 0
                simparams["SourceDeclination"] = 0
                simparams["SourceDate"] = "2022-05-02T00:00:00"
                simparams["SourceDateFormat"] = "isot"
                simparams["ObservationPeriod"] = 0

        else:
            simparams[node.tag] = check_unit(node, units)

    return SimulationParameters(
        N=int(simparams["NumTrajs"]),
        det_mode=simparams["SimulationParameters"],
        source_RA=simparams["SourceRightAscension"],
        source_DEC=simparams["SourceDeclination"],
        source_date=simparams["SourceDate"],
        source_date_format=simparams["SourceDateFormat"],
        source_obst=simparams["ObservationPeriod"],
        theta_ch_max=simparams["MaximumCherenkovAngle"],
        spectrum=simparams["Spectrum"],
        e_shower_frac=float(simparams["FracETauInShower"]),
        ang_from_limb=simparams["AngleFromLimb"],
        max_azimuth_angle=simparams["AzimuthalAngle"],
        model_ionosphere=bool(int(simparams["ModelIonosphere"])),
        TEC=float(simparams["TEC"]),
        TECerr=np.abs(float(simparams["TECerr"])),
        tau_table_version=simparams["TauTableVer"],
    )


def parseXML(xmlfile: str) -> tuple:
    r"""Parse the XML file into a pair of configuration objects.

    If the xml file is valid, parse the file into a DetectorCharacteristics and
    SimulationParameters tuple.

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.

    Returns
    -------
    tuple:
        Tuple of [DetectorCharacteristics, SimulationParameters] objects.

    Raises
    ------
    RuntimeError
        If the xml file is invalid, an exception is raised.
    """

    if is_valid_xml(xmlfile):
        return parse_detector_chars(xmlfile), parse_simulation_params(xmlfile)
    else:
        raise RuntimeError("Invalid XML File!")


def config_from_xml(
    xmlfile: str, fundcon: const.Fund_Constants = const.Fund_Constants()
) -> NssConfig:
    r"""Parse the XML file into an NssConfig object.

    If the xml file is valid, parse the file into a DetectorCharacteristics and
    SimulationParameters tuple.

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.
    fundcon: Fund_Constants
        A fundimental constants object. Defaults to default constructed object.

    Returns
    -------
    NssConfig:
        Parsed NssConfig object.
    """

    d, s = parseXML(xmlfile)
    return NssConfig(d, s, fundcon)


def create_xml(filename: str, config: NssConfig = NssConfig()) -> None:
    r"""Create an XML configuration file.

    Parameters
    ----------
    filename: str
        The name of the output xml file.
    config: NssConfig
        A NssConfig object from which to build the XML file.
    """

    nuspacesimparams = ET.Element("NuSpaceSimParams")

    detchar = ET.SubElement(nuspacesimparams, "DetectorCharacteristics")
    detchar.set("Type", "Satellite")
    detchar.set("Method", "Both")

    qeff = ET.SubElement(detchar, "QuantumEfficiency")
    qeff.text = str(config.detector.quantum_efficiency)

    telaeff = ET.SubElement(detchar, "TelescopeEffectiveArea")
    telaeff.set("Unit", "Sq.Meters")
    telaeff.text = str(config.detector.telescope_effective_area)

    pethres = ET.SubElement(detchar, "PhotoElectronThreshold")
    pethres.set("Preset", "true")

    detalt = ET.SubElement(detchar, "DetectorAltitude")
    detalt.set("Unit", "km")
    detalt.text = str(config.detector.altitude)

    detlat = ET.SubElement(detchar, "InitialDetectorLatitude")
    detlat.set("Unit", "Degrees")
    detlat.text = str(config.detector.detlat)

    detlong = ET.SubElement(detchar, "InitialDetectorLongitude")
    detlong.set("Unit", "Degrees")
    detlong.text = str(config.detector.detlong)

    npe = ET.SubElement(pethres, "NPE")
    npe.text = str(config.detector.photo_electron_threshold)

    detlow_freq = ET.SubElement(detchar, "LowFrequency")
    detlow_freq.set("Unit", "MHz")
    detlow_freq.text = str(config.detector.low_freq)

    dethigh_freq = ET.SubElement(detchar, "HighFrequency")
    dethigh_freq.set("Unit", "MHz")
    dethigh_freq.text = str(config.detector.high_freq)

    detSNRthres = ET.SubElement(detchar, "SNRThreshold")
    detSNRthres.text = str(config.detector.det_SNR_thres)

    detNant = ET.SubElement(detchar, "NAntennas")
    detNant.text = str(config.detector.det_Nant)

    detGain = ET.SubElement(detchar, "AntennaGain")
    detGain.text = str(config.detector.det_gain)

    detSunMoon = ET.SubElement(detchar, "SunMoonCuts")
    dethigh_freq = ET.SubElement(detSunMoon, "SunAngleBelowHorizonCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.sun_alt_cut)

    dethigh_freq = ET.SubElement(detSunMoon, "MoonAngleBelowHorizonCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.moon_alt_cut)

    dethigh_freq = ET.SubElement(detSunMoon, "MoonMinPhaseAngleCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.MoonMinPhaseAngleCut)

    simparams = ET.SubElement(nuspacesimparams, "SimulationParameters")
    simparams.set("DetectionMode", "Diffuse")

    cherangmax = ET.SubElement(simparams, "MaximumCherenkovAngle")
    cherangmax.set("Unit", "Radians")
    cherangmax.text = str(config.simulation.theta_ch_max)

    limbang = ET.SubElement(simparams, "AngleFromLimb")
    limbang.set("Unit", "Radians")
    limbang.text = str(config.simulation.ang_from_limb)

    eshowtype = ET.SubElement(simparams, "TauShowerType")
    eshowtype.set("Preset", "true")

    fraceshow = ET.SubElement(eshowtype, "FracETauInShower")
    fraceshow.text = str(config.simulation.e_shower_frac)

    nutauspectype = ET.SubElement(simparams, "NuTauEnergySpecType")

    if isinstance(config.simulation.spectrum, MonoSpectrum):
        mono = ET.SubElement(nutauspectype, "MonoSpectrum")
        nutauen = ET.SubElement(mono, "LogNuEnergy")
        nutauen.text = str(config.simulation.spectrum.log_nu_tau_energy)

    if isinstance(config.simulation.spectrum, PowerSpectrum):
        power = ET.SubElement(nutauspectype, "PowerSpectrum")
        sp1 = ET.SubElement(power, "PowerLawIndex")
        sp2 = ET.SubElement(power, "LowerBound")
        sp3 = ET.SubElement(power, "UpperBound")
        sp1.text = str(config.simulation.spectrum.index)
        sp2.text = str(config.simulation.spectrum.lower_bound)
        sp3.text = str(config.simulation.spectrum.upper_bound)

    if isinstance(config.simulation.spectrum, FileSpectrum):
        filespec = ET.SubElement(nutauspectype, "FileSpectrum")
        sp1 = ET.SubElement(filespec, "FilePath")
        sp1.text = str(config.simulation.spectrum.path)

    azimuthang = ET.SubElement(simparams, "AzimuthalAngle")
    azimuthang.set("Unit", "Radians")
    azimuthang.text = str(config.simulation.max_azimuth_angle)

    numtrajs = ET.SubElement(simparams, "NumTrajs")
    numtrajs.text = str(config.simulation.N)

    ionosphere = ET.SubElement(simparams, "ModelIonosphere")
    ionosphere.text = str(config.simulation.model_ionosphere)

    tec = ET.SubElement(simparams, "TEC")
    tec.text = str(config.simulation.TEC)

    tecerr = ET.SubElement(simparams, "TECerr")
    tecerr.text = str(config.simulation.TECerr)

    TauTableVer = ET.SubElement(simparams, "TauTableVer")
    TauTableVer.text = str(config.simulation.tau_table_version)

    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(nuspacesimparams)

    tree = ET.ElementTree(nuspacesimparams)
    tree.write(filename, encoding="utf-8", xml_declaration=True, method="xml")


def create_xml_too(filename: str, config: NssConfig = NssConfig()) -> None:
    r"""Create an XML configuration file.

    Parameters
    ----------
    filename: str
        The name of the output xml file.
    config: NssConfig
        A NssConfig object from which to build the XML file.
    """

    nuspacesimparams = ET.Element("NuSpaceSimParams")

    detchar = ET.SubElement(nuspacesimparams, "DetectorCharacteristics")
    detchar.set("Type", "Satellite")
    detchar.set("Method", "Both")

    qeff = ET.SubElement(detchar, "QuantumEfficiency")
    qeff.text = str(config.detector.quantum_efficiency)

    telaeff = ET.SubElement(detchar, "TelescopeEffectiveArea")
    telaeff.set("Unit", "Sq.Meters")
    telaeff.text = str(config.detector.telescope_effective_area)

    pethres = ET.SubElement(detchar, "PhotoElectronThreshold")
    pethres.set("Preset", "true")

    detalt = ET.SubElement(detchar, "DetectorAltitude")
    detalt.set("Unit", "km")
    detalt.text = str(config.detector.altitude)

    detra = ET.SubElement(detchar, "InitialDetectorLatitude")
    detra.set("Unit", "Degrees")
    detra.text = str(config.detector.detlat)

    detdec = ET.SubElement(detchar, "InitialDetectorLongitude")
    detdec.set("Unit", "Degrees")
    detdec.text = str(config.detector.detlong)

    npe = ET.SubElement(pethres, "NPE")
    npe.text = str(config.detector.photo_electron_threshold)

    detlow_freq = ET.SubElement(detchar, "LowFrequency")
    detlow_freq.set("Unit", "MHz")
    detlow_freq.text = str(config.detector.low_freq)

    dethigh_freq = ET.SubElement(detchar, "HighFrequency")
    dethigh_freq.set("Unit", "MHz")
    dethigh_freq.text = str(config.detector.high_freq)

    detSNRthres = ET.SubElement(detchar, "SNRThreshold")
    detSNRthres.text = str(config.detector.det_SNR_thres)

    detNant = ET.SubElement(detchar, "NAntennas")
    detNant.text = str(config.detector.det_Nant)

    detGain = ET.SubElement(detchar, "AntennaGain")
    detGain.text = str(config.detector.det_gain)

    detSunMoon = ET.SubElement(detchar, "SunMoonCuts")
    dethigh_freq = ET.SubElement(detSunMoon, "SunAngleBelowHorizonCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.sun_alt_cut)

    dethigh_freq = ET.SubElement(detSunMoon, "MoonAngleBelowHorizonCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.moon_alt_cut)

    dethigh_freq = ET.SubElement(detSunMoon, "MoonMinPhaseAngleCut")
    dethigh_freq.set("Unit", "Degrees")
    dethigh_freq.text = str(config.detector.MoonMinPhaseAngleCut)

    simparams = ET.SubElement(nuspacesimparams, "SimulationParameters")
    simparams.set("DetectionMode", "ToO")

    tooparams = ET.SubElement(simparams, "ToOSourceParameters")
    source_ra = ET.SubElement(tooparams, "SourceRightAscension")
    source_ra.set("Unit", "Degrees")
    source_ra.text = str(config.simulation.source_RA)

    source_dec = ET.SubElement(tooparams, "SourceDeclination")
    source_dec.set("Unit", "Degrees")
    source_dec.text = str(config.simulation.source_DEC)

    source_date = ET.SubElement(tooparams, "SourceDate")
    source_date.set("Format", config.simulation.source_date_format)
    source_date.text = str(config.simulation.source_date)

    source_period = ET.SubElement(tooparams, "ObservationPeriod")
    source_period.set("Unit", "sec")
    source_period.text = str(config.simulation.source_obst)

    cherangmax = ET.SubElement(simparams, "MaximumCherenkovAngle")
    cherangmax.set("Unit", "Radians")
    cherangmax.text = str(config.simulation.theta_ch_max)

    limbang = ET.SubElement(simparams, "AngleFromLimb")
    limbang.set("Unit", "Radians")
    limbang.text = str(config.simulation.ang_from_limb)

    eshowtype = ET.SubElement(simparams, "TauShowerType")
    eshowtype.set("Preset", "true")

    fraceshow = ET.SubElement(eshowtype, "FracETauInShower")
    fraceshow.text = str(config.simulation.e_shower_frac)

    nutauspectype = ET.SubElement(simparams, "NuTauEnergySpecType")

    if isinstance(config.simulation.spectrum, MonoSpectrum):
        mono = ET.SubElement(nutauspectype, "MonoSpectrum")
        nutauen = ET.SubElement(mono, "LogNuEnergy")
        nutauen.text = str(config.simulation.spectrum.log_nu_tau_energy)

    if isinstance(config.simulation.spectrum, PowerSpectrum):
        power = ET.SubElement(nutauspectype, "PowerSpectrum")
        sp1 = ET.SubElement(power, "PowerLawIndex")
        sp2 = ET.SubElement(power, "LowerBound")
        sp3 = ET.SubElement(power, "UpperBound")
        sp1.text = str(config.simulation.spectrum.index)
        sp2.text = str(config.simulation.spectrum.lower_bound)
        sp3.text = str(config.simulation.spectrum.upper_bound)

    if isinstance(config.simulation.spectrum, FileSpectrum):
        filespec = ET.SubElement(nutauspectype, "FileSpectrum")
        sp1 = ET.SubElement(filespec, "FilePath")
        sp1.text = str(config.simulation.spectrum.path)

    azimuthang = ET.SubElement(simparams, "AzimuthalAngle")
    azimuthang.set("Unit", "Radians")
    azimuthang.text = str(config.simulation.max_azimuth_angle)

    numtrajs = ET.SubElement(simparams, "NumTrajs")
    numtrajs.text = str(config.simulation.N)

    ionosphere = ET.SubElement(simparams, "ModelIonosphere")
    ionosphere.text = str(config.simulation.model_ionosphere)

    tec = ET.SubElement(simparams, "TEC")
    tec.text = str(config.simulation.TEC)

    tecerr = ET.SubElement(simparams, "TECerr")
    tecerr.text = str(config.simulation.TECerr)

    TauTableVer = ET.SubElement(simparams, "TauTableVer")
    TauTableVer.text = str(config.simulation.tau_table_version)

    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(nuspacesimparams)

    tree = ET.ElementTree(nuspacesimparams)
    tree.write(filename, encoding="utf-8", xml_declaration=True, method="xml")
