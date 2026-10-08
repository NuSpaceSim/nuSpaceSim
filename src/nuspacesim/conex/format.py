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

r"""The nuSpaceSim CONEX output format: an explicit, fixed-shape schema.

A CONEX file holds two TTrees (CONEX output version 2.51 branch names):

* ``Header`` -- one entry describing the run.
* ``Shower`` -- one entry per shower, with scalar summary branches and
  longitudinal *profile* branches.

In CONEX proper the profile branches are variable length (``X[nX]/F``). Here
every shower is sampled on the same fixed set of ``n_nodes`` longitudinal nodes
(the EAS optical quadrature grid), so each profile branch is a *fixed* array
leaf ``X[n_nodes]/F`` and the whole format is a NumPy structured dtype:
:func:`shower_dtype` is the template, :func:`empty_showers` instantiates it with
every field at its documented fill value, and the writer only overwrites the
fields nuSpaceSim actually computes. The ``n*`` count branches are kept (always
``n_nodes``) so readers written against CONEX, which loop to ``nX``, still work.

Every field below states its type, unit, and the value written when nuSpaceSim
has no counterpart (``fill``). Change the format here, nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "Field",
    "HEADER_FIELDS",
    "SHOWER_FIELDS",
    "PROFILE",
    "header_dtype",
    "shower_dtype",
    "empty_header",
    "empty_showers",
]

PROFILE = "profile"  # shape marker: one value per longitudinal node
N_LAMBDA = 31  # CONEX hadronic cross-section table length


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str
    unit: str
    doc: str
    fill: float | int = 0
    shape: tuple | str = ()


nan = float("nan")

HEADER_FIELDS = (
    Field("Seed1", "i4", "", "Random seed (none used)", -1),
    Field(
        "Particle", "i4", "", "Primary particle code (100 = proton; no tau code)", 100
    ),
    Field("Alpha", "f8", "", "Energy spectrum slope (unset)", nan),
    Field("lgEmin", "f8", "log10(eV)", "Minimum primary energy (unset)", nan),
    Field("lgEmax", "f8", "log10(eV)", "Maximum primary energy (unset)", nan),
    Field("zMin", "f8", "deg", "Minimum zenith angle", 90),
    Field("zMax", "f8", "deg", "Maximum zenith angle", 132),
    Field("SvnRevision", "i1", "", "CONEX revision (unset)", 0),
    Field("Version", "f4", "", "CONEX version emulated", 64),
    Field("OutputVersion", "f4", "", "CONEX output format version", 2.51),
    Field("HEModel", "i4", "", "High-energy hadronic model (none)", -1),
    Field("LEModel", "i4", "", "Low-energy hadronic model (none)", -1),
    Field("HiLowEgy", "f4", "GeV", "HE/LE model transition energy (unset)", nan),
    Field("hadCut", "f4", "", "Hadron MC/CE switch threshold (unset)", nan),
    Field("emCut", "f4", "", "EM MC/CE switch threshold (unset)", nan),
    Field("hadThr", "f4", "GeV", "Hadron threshold (unset)", nan),
    Field("muThr", "f4", "GeV", "Muon threshold (unset)", nan),
    Field("emThr", "f4", "GeV", "EM threshold (unset)", nan),
    Field("haCut", "f4", "GeV", "Hadron energy cut (unset)", nan),
    Field("muCut", "f4", "GeV", "Muon energy cut (unset)", nan),
    Field("elCut", "f4", "GeV", "Electron energy cut (unset)", nan),
    Field("gaCut", "f4", "GeV", "Photon energy cut (unset)", nan),
    Field(
        "lambdaLgE", "f8", "log10(eV)", "Cross-section table energies", nan, (N_LAMBDA,)
    ),
    Field("lambdaProton", "f8", "g/cm2", "Proton interaction length", nan, (N_LAMBDA,)),
    Field("lambdaPion", "f8", "g/cm2", "Pion interaction length", nan, (N_LAMBDA,)),
    Field("lambdaHelium", "f8", "g/cm2", "Helium interaction length", nan, (N_LAMBDA,)),
    Field(
        "lambdaNitrogen", "f8", "g/cm2", "Nitrogen interaction length", nan, (N_LAMBDA,)
    ),
    Field("lambdaIron", "f8", "g/cm2", "Iron interaction length", nan, (N_LAMBDA,)),
)

_PROFILE_COUNTS = ("nX", "nN", "nH", "nD", "ndEdX", "nMu", "nGamma", "nElectrons")
_PROFILE_COUNTS += ("nHadrons", "ndMu")

SHOWER_FIELDS = (
    Field("lgE", "f4", "log10(eV)", "Shower energy"),
    Field("zenith", "f4", "deg", "Zenith angle of the axis (90 + emergence angle)"),
    Field("azimuth", "f4", "deg", "Azimuth of the axis (uniform random)"),
    Field("Seed2", "i4", "", "Random seed (none used)", -1),
    Field("Seed3", "i4", "", "Random seed (none used)", -1),
    Field("Xfirst", "f4", "g/cm2", "Slant depth from ground to the tau decay point"),
    Field("Hfirst", "f4", "m", "Altitude of the tau decay point"),
    Field("XfirstIn", "f4", "", "Inverse first-interaction length (fixed)", 0.5),
    Field("altitude", "f8", "m", "Observation altitude (unset)", nan),
    Field("X0", "f4", "g/cm2", "Gaisser-Hillas X0", nan),
    Field("Xmax", "f4", "g/cm2", "Gaisser-Hillas Xmax", nan),
    Field("Nmax", "f4", "", "Gaisser-Hillas Nmax", nan),
    Field("p1", "f4", "g/cm2", "GH lambda(X) = p1 + p2 X + p3 X^2: constant", nan),
    Field("p2", "f4", "", "GH lambda(X): linear coefficient", nan),
    Field("p3", "f4", "cm2/g", "GH lambda(X): quadratic coefficient", nan),
    Field("chi2", "f4", "", "Gaisser-Hillas fit quality (x1e5, CONEX scaling)", nan),
    Field("Xmx", "f4", "g/cm2", "Depth of the profile node with the most particles"),
    Field("Nmx", "f4", "", "Particle count at Xmx"),
    Field("XmxdEdX", "f4", "g/cm2", "Depth of the dE/dX maximum (taken = Xmx)"),
    Field("dEdXmx", "f4", "GeV cm2/g", "dE/dX at Xmx"),
    Field("cpuTime", "f4", "s", "CPU time (unset)", nan),
    Field("X", "f4", "g/cm2", "Slant depth from ground at each node", shape=PROFILE),
    Field("N", "f4", "", "Charged particles at each node", shape=PROFILE),
    Field("H", "f4", "m", "Altitude of each node", shape=PROFILE),
    Field("D", "f4", "m", "Distance from ground along the axis", shape=PROFILE),
    Field(
        "dEdX", "f4", "GeV cm2/g", "Energy deposit (age-parameterized)", shape=PROFILE
    ),
    Field("Mu", "f4", "", "Muons (not simulated)", shape=PROFILE),
    Field("Gamma", "f4", "", "Photons (not simulated)", shape=PROFILE),
    Field("Electrons", "f4", "", "Electrons (= N)", shape=PROFILE),
    Field("Hadrons", "f4", "", "Hadrons (not simulated)", shape=PROFILE),
    Field("dMu", "f4", "", "Muon production (not simulated)", shape=PROFILE),
    *(
        Field(c, "i4", "", f"Length of {c[1:]} (always n_nodes)")
        for c in _PROFILE_COUNTS
    ),
    Field("EGround", "f4", "GeV", "Energy reaching ground (unset)", nan, (3,)),
)


def _dtype(fields, n_nodes=None) -> np.dtype:
    spec = []
    for f in fields:
        shape = (n_nodes,) if f.shape == PROFILE else f.shape
        spec.append((f.name, f.dtype, shape) if shape else (f.name, f.dtype))
    return np.dtype(spec)


def header_dtype() -> np.dtype:
    """Structured dtype of one ``Header`` entry."""
    return _dtype(HEADER_FIELDS)


def shower_dtype(n_nodes: int) -> np.dtype:
    """Structured dtype of one ``Shower`` entry with ``n_nodes`` profile nodes."""
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes}")
    return _dtype(SHOWER_FIELDS, n_nodes)


def _filled(fields, dtype, n, n_nodes) -> np.ndarray:
    out = np.zeros(n, dtype)
    for f in fields:
        out[f.name] = f.fill
    if n_nodes is not None:
        for c in _PROFILE_COUNTS:
            out[c] = n_nodes
    return out


def empty_header() -> np.ndarray:
    """A 1-entry ``Header`` array with every field at its fill value."""
    return _filled(HEADER_FIELDS, header_dtype(), 1, None)


def empty_showers(n: int, n_nodes: int) -> np.ndarray:
    """``n`` ``Shower`` entries with every field at its fill value."""
    return _filled(SHOWER_FIELDS, shower_dtype(n_nodes), n, n_nodes)
