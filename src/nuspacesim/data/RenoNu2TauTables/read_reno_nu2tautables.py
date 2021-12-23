import math

import h5py
import numpy as np


def extract_pexit_data(filename):
    infile = open(filename, "r")
    data = [line.split() for line in infile]
    b = [(math.pi * float(lne[0]) / 180.0) for lne in data]
    le = [math.log10(float(lne[1])) for lne in data]
    p = [math.log10(float(lne[-1])) for lne in data]
    infile.close()
    return b, le, p


def extra_taudist_data(filename):
    bdeg = np.array([1.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 17.0, 20.0, 25.0])
    infile = open(filename, "r")
    data = [line.split() for line in infile]
    brad = math.pi * bdeg / 180.0
    z = np.array([float(lne[0]) for lne in data])
    for lne in data:
        del lne[0]
    cv = np.array(data, float)
    infile.close()
    return z, brad, cv


def main():
    f = h5py.File("RenoNu2TauTables/nu2taudata.hdf5", "w")
    pexitgrp = f.create_group("pexitdata")
    blist, lelist, plist = extract_pexit_data("RenoNu2TauTables/multi-efix.26")
    beta = np.array(blist)
    logenergy = np.array(lelist)
    pexitval = np.array(plist)
    buniq = np.unique(beta)
    leuniq = np.unique(logenergy)
    pexitarr = pexitval.reshape((leuniq.size, buniq.size))

    pexitgrp.create_dataset("BetaRad", data=buniq, dtype="f")
    pexitgrp.create_dataset("logNuEnergy", data=leuniq, dtype="f")
    pexitgrp.create_dataset("logPexit", data=pexitarr, dtype="f")

    for lognuenergy in np.arange(7.0, 11.0, 0.25):
        mygrpstring = "TauEdist_grp_e{:02.0f}_{:02.0f}".format(
            math.floor(lognuenergy), (lognuenergy - math.floor(lognuenergy)) * 100
        )
        tedistgrp = f.create_group(mygrpstring)

        myfilestring = (
            "RenoNu2TauTables/nu2tau-angleC-e{:02.0f}-{:02.0f}smx.dat".format(
                math.floor(lognuenergy), (lognuenergy - math.floor(lognuenergy)) * 100
            )
        )
        tauEfrac, tdbeta, cdfvalues = extra_taudist_data(myfilestring)

        tedistgrp.create_dataset("TauEFrac", data=tauEfrac, dtype="f")
        tedistgrp.create_dataset("BetaRad", data=tdbeta, dtype="f")
        tedistgrp.create_dataset("TauEDistCDF", data=cdfvalues, dtype="f")


if __name__ == "__main__":
    main()
