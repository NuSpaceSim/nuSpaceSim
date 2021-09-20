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

"""XMLSchema string file. For validating XML configuration files."""

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
    <xs:element name="NPE" type="xs:decimal"/>
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
