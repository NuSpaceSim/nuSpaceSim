# Requires nuspacesim and tabulate
#
# pip install nuspacesim tabulate
#
# Usage:
#
# python jfki_result_summary directory/with/simulation/files/


import os
import sys

from astropy.table import Table as AstropyTable
from astropy.units import Quantity
from tabulate import tabulate

import nuspacesim as nss

if __name__ == "__main__":

    results = list()

    path = os.path.abspath(sys.argv[1])
    for filename in os.listdir(path):

        _, extension = os.path.splitext(filename)
        if extension != ".fits":
            continue

        filepath = os.path.join(path, filename)
        # r = nss.ResultsTable.read(filepath)
        r = AstropyTable.read(filepath)

#        energy = r.meta["SPECPARA"]
        energy = r.meta["Config simulation spectrum log_nu_energy"]
        mci = r.meta["OMCINT"]
        gf = r.meta["OMCINTGO"]
        npe = r.meta["ONEVPASS"]
#        betae = r.meta["Config simulation angle_from_limb"] * 180./3.1415926
#        thchmax = r.meta["Config simulation max_cherenkov_angle"] * 180./3.1415926

        betae = Quantity(r.meta["Config simulation angle_from_limb"]).value 
        thchmax = Quantity(r.meta["Config simulation max_cherenkov_angle"]).value 
        radmci = r.meta["RMCINT"]
        radgf = r.meta["RMCINTGO"]
        nradevt = r.meta["RNEVPASS"]

        t = tuple([energy, betae, thchmax, mci, gf, npe, radmci, radgf, nradevt])
#        t = tuple([energy,  mci, gf, npe, radmci, radgf, nradevt])


        results.append(t)

    results.sort()

    print(
        tabulate(
            results,
            headers=[
                "log_e_nu",
                "betaE (deg)",
                "thchmax (deg)",
                "Opt Monte Carlo Integral",
                "Opt Geometry Factor",
                "Opt Passing Events",
                "Radio Monte Carlo Integral",
                "Radio Geometry Factor",
                "Radio Passing Events",
            ],
        )
    )
