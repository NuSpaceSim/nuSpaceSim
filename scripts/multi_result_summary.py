# Requires nuspacesim and tabulate
#
# pip install nuspacesim tabulate
#
# Usage:
#
# python multi_result_summary directory/with/simulation/files/


import os
import sys

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
        r = nss.ResultsTable.read(filepath)

        energy = r.meta["SPECPARA"]
        mci = r.meta["OMCINT"]
        gf = r.meta["OMCINTGO"]
        npe = r.meta["ONEVPASS"]

        t = tuple([energy, mci, gf, npe])

        results.append(t)

    results.sort()

    print(
        tabulate(
            results,
            headers=[
                "log_e_nu",
                "Monte Carlo Integral",
                "Geometry Factor",
                "Passing Events",
            ],
        )
    )
