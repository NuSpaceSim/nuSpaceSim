import lxml.etree as ET

def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def main():
    nuspacesimparams = ET.Element('NuSpaceSimParams')

    detchar = ET.SubElement(nuspacesimparams, 'DetectorCharacteristics')
    detchar.set('Type', 'Satellite')
    detchar.set('Method', 'Optical')
    
    #detmethod = ET.SubElement(simparams, 'DetectionMethod')
    #detmethod.set('Method', 'Optical')
    
    #detchar = ET.SubElement(detmethod, 'DetectorCharacteristics')

    qeff = ET.SubElement(detchar, 'QuantumEfficiency')
    qeff.text = '0.2'

    telaeff = ET.SubElement(detchar, 'TelescopeEffectiveArea')
    telaeff.set('Unit', 'Sq.Meters')
    telaeff.text = '2.5'

    pethres = ET.SubElement(detchar, 'PhotoElectronThreshold')
    pethres.set('Preset', 'True')
    #pethres.text = '10'

    detalt = ET.SubElement(detchar, 'DetectorAltitude')
    detalt.set('Unit', 'km')
    detalt.text = '525.0'

    detra = ET.SubElement(detchar, 'InitialDetectorRightAscension')
    detra.set('Unit', 'Degrees')
    detra.text = '0.0'

    detdec = ET.SubElement(detchar, 'InitialDetectorDeclination')
    detdec.set('Unit', 'Degrees')
    detdec.text = '0.0'

    npe = ET.SubElement(pethres, 'NPE')
    npe.text = '10'

    simparams = ET.SubElement(nuspacesimparams, 'SimulationParameters')
    simparams.set('DetectionMode', 'Diffuse')

    cherangmax = ET.SubElement(simparams, 'MaximumCherenkovAngle')
    cherangmax.set('Unit', 'Degrees')
    cherangmax.text = '3.0'

    limbang = ET.SubElement(simparams, 'AngleFromLimb')
    limbang.set('Unit', 'Degrees')
    limbang.text = '7.0'

    eshowtype = ET.SubElement(simparams, 'TauShowerType')
    eshowtype.set('Preset', 'True')
    #eshowtype.text = '0.5'

    fraceshow = ET.SubElement(eshowtype, 'FracETauInShower')
    fraceshow.text = '0.5'

    nutauspectype = ET.SubElement(simparams, 'NuTauEnergySpecType')
    nutauspectype.set('SpectrumType', 'Mono')

    nutauen = ET.SubElement(nutauspectype, 'NuTauEnergy')
    nutauen.text = '8.0'

    azimuthang = ET.SubElement(simparams, 'AzimuthalAngle')
    azimuthang.set('Unit', 'Degrees')
    azimuthang.text = '360.0'

    numtrajs = ET.SubElement(simparams, 'NumTrajs')
    numtrajs.text = '10'

    #detloc = ET.SubElement(simparams, 'DetectorLocation')
    
    indent(nuspacesimparams)

    tree = ET.ElementTree(nuspacesimparams)
    tree.write('sample_input_file.xml', encoding='utf-8', xml_declaration=True, method="xml")

if __name__ == "__main__":
    main ()
