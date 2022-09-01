import numpy as np
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from nuspacesim.simulation.eas_composite.comp_eas_utils import bin_nmax_xmax
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py


f = "../conex_7_50_runs/test_runs/log_17_eV_1shwr_5_earthemergence_360_azimuthal_eposlhc_1692126373_100.root"

shwr_data = ReadConex(f, shower_header_name=1)
depths = shwr_data.get_depths()[0]
alts = shwr_data.get_alts()[0]
mus = shwr_data.get_muons()[0]
char = shwr_data.get_charged()[0]
el = shwr_data.get_elec_pos()[0]
had = shwr_data.get_hadrons()[0]
gam = shwr_data.get_gamma()[0]

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(4, 4), dpi=200)
ax.plot(depths, mus, label="muons")
ax.plot(depths, char, label="charged")
ax.plot(depths, had, label="hadrons")
ax.set(ylabel="$N$", xlabel="X (g/cm^2) \n {}".format(f.split("/")[-1]))
ax.set_ylim(bottom=1)
ax.set_yscale("log")
ax.legend()
