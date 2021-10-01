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
    NssConfig,
    DetectorCharacteristics,
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
    if eldetchar.attrib["Method"] == "Optical":
        for node in tree.find("./DetectorCharacteristics"):
            if node.tag == "PhotoElectronThreshold":
                detchar[node.tag] = str(node.attrib["Preset"])
                if node.attrib["Preset"] == "true":
                    detchar["NPE"] = str(node.find("NPE").text)
            else:
                detchar[node.tag] = str(node.text)

            # Convert Degrees to Radians
            if "Unit" in node.attrib:
                if node.attrib["Unit"] == "Degrees":
                    detchar[node.tag] = np.radians(float(node.text))

    return DetectorCharacteristics(
        altitude=float(detchar["DetectorAltitude"]),
        ra_start=float(detchar["InitialDetectorRightAscension"]),
        dec_start=float(detchar["InitialDetectorDeclination"]),
        telescope_effective_area=float(detchar["TelescopeEffectiveArea"]),
        quantum_efficiency=float(detchar["QuantumEfficiency"]),
        photo_electron_threshold=float(detchar["NPE"]),
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

    simparams: dict[str, str] = {}
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
            simparams[node.tag] = node.attrib["SpectrumType"]
            if node.attrib["SpectrumType"] == "Mono":
                simparams["NuTauEnergy"] = str(node.find("NuTauEnergy").text)
        else:
            simparams[node.tag] = str(node.text)

        # Convert Degrees to Radians
        if "Unit" in node.attrib:
            if node.attrib["Unit"] == "Degrees":
                simparams[node.tag] = np.radians(float(node.text))

    return SimulationParameters(
        N=int(simparams["NumTrajs"]),
        theta_ch_max=float(simparams["MaximumCherenkovAngle"]),
        nu_tau_energy=float(simparams["NuTauEnergy"]),
        e_shower_frac=float(simparams["FracETauInShower"]),
        ang_from_limb=float(simparams["AngleFromLimb"]),
        max_azimuth_angle=float(simparams["AzimuthalAngle"]),
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
    detchar.set("Method", "Optical")

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
    detra.set("Unit", "Radians")
    detra.text = str(config.detector.ra_start)

    detdec = ET.SubElement(detchar, "InitialDetectorDeclination")
    detdec.set("Unit", "Radians")
    detdec.text = str(config.detector.dec_start)

    npe = ET.SubElement(pethres, "NPE")
    npe.text = str(config.detector.photo_electron_threshold)

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
    nutauspectype.set("SpectrumType", "Mono")

    nutauen = ET.SubElement(nutauspectype, "NuTauEnergy")
    nutauen.text = str(config.simulation.nu_tau_energy)

    azimuthang = ET.SubElement(simparams, "AzimuthalAngle")
    azimuthang.set("Unit", "Radians")
    azimuthang.text = str(config.simulation.max_azimuth_angle)

    numtrajs = ET.SubElement(simparams, "NumTrajs")
    numtrajs.text = str(config.simulation.N)

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
