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

import lxml.etree as ET
import numpy as np

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


def parse_detector_chars(xmlfile: str) -> DetectorCharacteristics:
    r"""Parse the XML file into a DetectorCharacteristics object.

    Parameters
    ----------
    xmlfile: str
        The input configuration xml file.

    Returns
    -------
    DetectorCharacteristics
        The detector characteristics object.
    """

    detchar: dict[str, str] = {}
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    eldetchar = root.find("DetectorCharacteristics")
    detchar["Method"] = eldetchar.attrib["Method"]
    for node in tree.find("./DetectorCharacteristics"):
        if node.tag == "PhotoElectronThreshold":
            detchar[node.tag] = str(node.attrib["Preset"])
            if node.attrib["Preset"] == "true":
                detchar["NPE"] = str(node.find("NPE").text)
        else:
            detchar[node.tag] = str(node.text)

        # Convert Degrees to Radians
        if "Unit" in node.attrib:
            if node.tag in [
                "InitialDetectorRightAscension",
                "InitialDetectorDeclination",
            ]:
                x = float(node.text)
                detchar[node.tag] = (
                    x if node.attrib["Unit"] == "Degrees" else np.radians(x)
                )

                np.degrees(float(node.text))
            elif node.attrib["Unit"] == "Degrees":
                detchar[node.tag] = np.radians(float(node.text))

    return DetectorCharacteristics(
        method=detchar["Method"],
        altitude=float(detchar["DetectorAltitude"]),
        ra_start=float(detchar["InitialDetectorRightAscension"]),
        dec_start=float(detchar["InitialDetectorDeclination"]),
        telescope_effective_area=float(detchar["TelescopeEffectiveArea"]),
        quantum_efficiency=float(detchar["QuantumEfficiency"]),
        photo_electron_threshold=float(detchar["NPE"]),
        low_freq=float(detchar["LowFrequency"]),
        high_freq=float(detchar["HighFrequency"]),
        det_SNR_thres=float(detchar["SNRThreshold"]),
        det_Nant=int(detchar["NAntennas"]),
        det_gain=float(detchar["AntennaGain"]),
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

    simparams = {}
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    elsimparams = root.find("SimulationParameters")
    simparams[elsimparams.tag] = elsimparams.attrib["DetectionMode"]

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
        else:
            simparams[node.tag] = str(node.text)

        # Convert Degrees to Radians
        if "Unit" in node.attrib:
            if node.attrib["Unit"] == "Degrees":
                simparams[node.tag] = np.radians(float(node.text))

    return SimulationParameters(
        N=int(simparams["NumTrajs"]),
        theta_ch_max=float(simparams["MaximumCherenkovAngle"]),
        spectrum=simparams["Spectrum"],
        e_shower_frac=float(simparams["FracETauInShower"]),
        ang_from_limb=float(simparams["AngleFromLimb"]),
        max_azimuth_angle=float(simparams["AzimuthalAngle"]),
        model_ionosphere=bool(int(simparams["ModelIonosphere"])),
        TEC=float(simparams["TEC"]),
        TECerr=np.abs(float(simparams["TECerr"])),
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

    detra = ET.SubElement(detchar, "InitialDetectorRightAscension")
    detra.set("Unit", "Degrees")
    detra.text = str(config.detector.ra_start)

    detdec = ET.SubElement(detchar, "InitialDetectorDeclination")
    detdec.set("Unit", "Degrees")
    detdec.text = str(config.detector.dec_start)

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
