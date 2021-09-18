from io import StringIO

xsd = StringIO(
    """\
<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">
<xs:complexType name="AngleType">
    <xs:simpleContent>
      <xs:extension base="xs:decimal">
        <xs:attribute name="Unit">
          <xs:simpleType>
            <xs:restriction base="xs:string">
              <xs:enumeration value="Degrees"/>
              <xs:enumeration value="Radians"/>
            </xs:restriction>
          </xs:simpleType>
        </xs:attribute>
      </xs:extension>
    </xs:simpleContent>
</xs:complexType>

<xs:complexType name="AreaType">
    <xs:simpleContent>
      <xs:extension base="xs:decimal">
        <xs:attribute name="Unit">
          <xs:simpleType>
            <xs:restriction base="xs:string">
              <xs:enumeration value="Sq.Meters"/>
            </xs:restriction>
          </xs:simpleType>
        </xs:attribute>
      </xs:extension>
    </xs:simpleContent>
</xs:complexType>

<xs:complexType name="DistType">
    <xs:simpleContent>
      <xs:extension base="xs:decimal">
        <xs:attribute name="Unit">
          <xs:simpleType>
            <xs:restriction base="xs:string">
              <xs:enumeration value="km"/>
            </xs:restriction>
          </xs:simpleType>
        </xs:attribute>
      </xs:extension>
    </xs:simpleContent>
</xs:complexType>

<xs:complexType name="PETType">
  <xs:sequence>
    <xs:element name="NPE" type="xs:integer"/>
  </xs:sequence>
  <xs:attribute name="Preset" type="xs:boolean"/>
</xs:complexType>

<xs:complexType name="TSType">
  <xs:sequence>
    <xs:element name="FracETauInShower" type="xs:decimal"/>
  </xs:sequence>
  <xs:attribute name="Preset" type="xs:boolean"/>
</xs:complexType>

<xs:complexType name="NTESType">
  <xs:sequence>
    <xs:element name="NuTauEnergy" type="xs:decimal"/>
  </xs:sequence>
  <xs:attribute name="SpectrumType" type="xs:string"/>
</xs:complexType>

<xs:element name="NuSpaceSimParams">
  <xs:complexType>
    <xs:sequence>
      <xs:element name="DetectorCharacteristics">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="QuantumEfficiency" type="xs:decimal"/>
            <xs:element name="TelescopeEffectiveArea" type="AreaType"/>
            <xs:element name="PhotoElectronThreshold" type="PETType"/>
            <xs:element name="DetectorAltitude" type="DistType"/>
            <xs:element name="InitialDetectorRightAscension" type="AngleType"/>
            <xs:element name="InitialDetectorDeclination" type="AngleType"/>
          </xs:sequence>
          <xs:attribute name="Type" type="xs:string"/>
          <xs:attribute name="Method" type="xs:string"/>
        </xs:complexType>
      </xs:element>
      <xs:element name="SimulationParameters">
        <xs:complexType>
          <xs:sequence>
            <xs:element name="MaximumCherenkovAngle" type="AngleType"/>
            <xs:element name="AngleFromLimb" type="AngleType"/>
            <xs:element name="TauShowerType" type="TSType"/>
            <xs:element name="NuTauEnergySpecType" type="NTESType"/>
            <xs:element name="AzimuthalAngle" type="AngleType"/>
            <xs:element name="NumTrajs" type="xs:decimal"/>
          </xs:sequence>
          <xs:attribute name="DetectionMode" type="xs:string"/>
          <xs:attribute name="Method" type="xs:string"/>
        </xs:complexType>
      </xs:element>
    </xs:sequence>
  </xs:complexType>
</xs:element>
</xs:schema>
        """
)
