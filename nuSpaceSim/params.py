####
import xml.etree.ElementTree as ET
import numpy as np


class NssConfig:

    def __init__(self, configfile="sample_input_file.xml"):
        self.detchar, self.simparams = parseXML(configfile)
        self.fundcon = Fund_Constants()
        self.EarthRadius = self.fundcon.earth_radius
        self.detectAlt = float(self.detchar["DetectorAltitude"])
        self.raStart = float(self.detchar["InitialDetectorRightAscension"])
        self.decStart = float(self.detchar["InitialDetectorDeclination"])
        self.detAeff = float(self.detchar["TelescopeEffectiveArea"])
        self.detQeff = float(self.detchar["QuantumEfficiency"])
        self.detPEthres = 0 if "NPE" not in self.detchar else float(self.detchar["NPE"])
        self.logNuTauEnergy = float(self.simparams["NuTauEnergy"])
        self.nuTauEnergy = 10**self.logNuTauEnergy
        self.eShowFrac = float(self.simparams["FracETauInShower"])
        self.AngFrLimb = self.fundcon.pi * \
            (float(self.simparams["AngleFromLimb"]) / 180.0)
        self.thetaChMax = self.fundcon.pi * (
            float(self.simparams["MaximumCherenkovAngle"]) / 180.0)
        self.sinthetaChMax = np.sin(self.thetaChMax)
        self.maxaziang = self.fundcon.pi * \
            (float(self.simparams["AzimuthalAngle"]) / 180.0)
        self.N = int(self.simparams["NumTrajs"])
        self.method = self.detchar["Method"]
        #this is all radio stuff
        lf = 30 if "LowFrequency" not in self.detchar else float(self.detchar["LowFrequency"])
        hf = 1000 if "HighFrequency" not in self.detchar else float(self.detchar["HighFrequency"])
        self.detSNRthres = 2. if "SNRThreshold" not in self.detchar else float(self.detchar["SNRThreshold"])
        self.detNant = 1 if "NAntennas" not in self.detchar else int(self.detchar["NAntennas"])
        self.detGain = 10. if "AntennaGain" not in self.detchar else float(self.detchar["AntennaGain"])
        self.detFreqRange = (lf, hf)
        self.modelIonosphere = True if "ModelIonosphere" not in self.simparams else bool(int(self.simparams["ModelIonosphere"]))
        self.TEC = 10. if "TEC" not in self.simparams else float(self.simparams["TEC"])
        self.TECerr = 0.1 if "TECErr" not in self.simparams else np.abs(float(self.simparams["TECErr"]))



class Fund_Constants:
    """
    Fundimental constants used on nuSpaceSim simulator.
    """

    def __init__(
        self,
        pi=np.pi,
        earth_radius=6371.0,
        c=2.9979246e5,
        massTau=1.77686,
        mean_Tau_life=2.903e-13,
    ):
        self.pi = pi
        self.earth_radius = earth_radius  # [km]
        self.c = c  # [km/s]
        self.massTau = massTau  # [GeV/c^2]
        self.mean_Tau_life = mean_Tau_life  # [s]
        self.inv_mean_Tau_life = 1.0 / (mean_Tau_life)  # [s^-1]


def parseXML(xmlfile):
    fdetchar = {}
    fsimparams = {}
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    eldetchar = root.find("DetectorCharacteristics")
    fdetchar["Method"] = eldetchar.attrib["Method"]
    for node in tree.find("./DetectorCharacteristics"):
        if node.tag == "PhotoElectronThreshold":
            fdetchar[node.tag] = node.attrib["Preset"]
            if node.attrib["Preset"] == "True":
                fdetchar["NPE"] = node.find("NPE").text
        else:
            fdetchar[node.tag] = node.text

    elsimparams = root.find("SimulationParameters")
    fsimparams[elsimparams.tag] = elsimparams.attrib["DetectionMode"]

    for node in tree.find("./SimulationParameters"):
        if node.tag == "TauShowerType":
            fsimparams[node.tag] = node.attrib["Preset"]
            if node.attrib["Preset"] == "True":
                fsimparams["FracETauInShower"] = node.find(
                    "FracETauInShower").text
        elif node.tag == "NuTauEnergySpecType":
            fsimparams[node.tag] = node.attrib["SpectrumType"]
            if node.attrib["SpectrumType"] == "Mono":
                fsimparams["NuTauEnergy"] = node.find("NuTauEnergy").text
        else:
            fsimparams[node.tag] = node.text
    return (fdetchar, fsimparams)
