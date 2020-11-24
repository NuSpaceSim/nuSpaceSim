```
      ___           ___
     /\  \         /\  \
     \:\  \        \:\  \
      \:\  \        \:\  \
  _____\:\  \   ___  \:\  \
 /::::::::\__\ /\  \  \:\__\
 \:\~~\~~\/__/ \:\  \ /:/  /
  \:\  \        \:\  /:/  /
   \:\  \        \:\/:/  /
    \:\__\        \::/  /
     \/__/         \/__/
      ___           ___         ___           ___           ___
     /\__\         /\  \       /\  \         /\__\         /\__\
    /:/ _/_       /::\  \     /::\  \       /:/  /        /:/ _/_
   /:/ /\  \     /:/\:\__\   /:/\:\  \     /:/  /        /:/ /\__\
  /:/ /::\  \   /:/ /:/  /  /:/ /::\  \   /:/  /  ___   /:/ /:/ _/_
 /:/_/:/\:\__\ /:/_/:/  /  /:/_/:/\:\__\ /:/__/  /\__\ /:/_/:/ /\__\
 \:\/:/ /:/  / \:\/:/  /   \:\/:/  \/__/ \:\  \ /:/  / \:\/:/ /:/  /
  \::/ /:/  /   \::/__/     \::/__/       \:\  /:/  /   \::/_/:/  /
   \/_/:/  /     \:\  \      \:\  \        \:\/:/  /     \:\/:/  /
     /:/  /       \:\__\      \:\__\        \::/  /       \::/  /
     \/__/         \/__/       \/__/         \/__/         \/__/
      ___                       ___
     /\__\                     /\  \
    /:/ _/_       ___         |::\  \
   /:/ /\  \     /\__\        |:|:\  \
  /:/ /::\  \   /:/__/      __|:|\:\  \
 /:/_/:/\:\__\ /::\  \     /::::|_\:\__\
 \:\/:/ /:/  / \/\:\  \__  \:\~~\  \/__/
  \::/ /:/  /   ~~\:\/\__\  \:\  \
   \/_/:/  /       \::/  /   \:\  \
     /:/  /        /:/  /     \:\__\
     \/__/         \/__/       \/__/

```

# nuSpaceSim

%% This repository contains python code for a preliminary version of the nuspacesim master scheduler. This code calculates the tau neutrino acceptance for the Optical Cherenkov technique. Also included in the repository is python code for creating the input xml file (create_xml.py), a sample input xml file (sample_input_file.xml), a directory including the tau propagation files from Reno et al. 2019, and python code to read the tau propagation files and store the necessary arrays in an hdf5 file.

%% ## Requirements

%% -- master_loop_skeleton.py -- To run this code, you will need:

%% * python3 
%% * The "nssgeometry" module installed. You can find the repository and instructions for building it at https://github.com/NuSpaceSim/nssgeometry.git . Once installed, you must update the "sys.path.append" statement in the master_loop_skeleton_traj_only.py code to point at the user's local nssgeometry repository.
%% * Detection parameters are read from an input xml file (sample included in this repository). The "create_xml.py" script creates an xml file using the format expected by the master loop script.

%% -- create_xml.py -- To run this code, you will need:

%% * The lxml module installed.

%% ## Download and run

%% 1. git clone https://github.com/NuSpaceSim/MasterLoopTrajOnly.git
%% 2. python master_loop_skeleton.py
