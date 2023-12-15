.. _simulation:

#########################################
Neutrino Simulation with `nuspacesim run`
#########################################

The simulation program of nuspacesim is available on the the command line from the
`run` command of the nuspacesim application. This simulator will throw a given
number of neutrino trajectories through the atmosphere and compute simulation
variables determined by the physics modules.

Configuration File
******************

Simulation configuration is governed by a TOML file.

Results File
************

The simulation results are written to an output file in FITs format.

Useful Plots from Simulation Results
************************************

Plots of simulation data can be generated at runtime or after the fact with `nuspacesim show-plot`
