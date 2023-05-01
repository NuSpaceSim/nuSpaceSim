"""
generating composite showers using the profiles themselves from conex, not just the GH
100 PeV or 10^17 eV for 5 degree earth emergence angles
"""

import numpy as np


import h5py
from comp_eas_utils import numpy_argmax_reduceat
from nuspacesim.simulation.eas_composite.x_to_z_lookup import depth_to_alt_lookup_v2
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter

from scipy.optimize import fsolve


try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


class ConexCompositeShowers:
    """
    generating composite showers using the profiles themselves from conex,
    not just the GH for 10^17 eV for 5 degree earth emergence angles
    """

    def __init__(
        self,
        shower_comps: list,
        init_pid: list,
        beta: int = 5,
        shwr_per_file: int = 1000,
        tau_table_start: int = 100,
        energy_pev: int = 100,
    ):
        if (beta != 5) or (energy_pev != 100):
            print("Note, we currently only have conex data for beta = 5deg, e=100PeV")

        with as_file(
            files("nuspacesim.data.pythia_tau_decays") / "tau_100_PeV.h5"
        ) as path:
            data = h5py.File(path, "r")
            tau_decays = np.array(data.get("tau_data"))

        self.tau_tables = tau_decays[tau_table_start:, :]
        # 11 is electron
        # 22 is gamma
        # +/- 211 and +/- 321 treated as the same. kaons and pions.
        self.nshowers = shwr_per_file
        self.pid = init_pid
        self.showers = shower_comps

    def tau_daughter_energies(self, particle_id):
        r"""Isolate energy contributions given a specific pid

        Used to scale Nmax of shower components

        """

        if (particle_id == 211) or (particle_id == 321):
            # if either the initiating particle is a kaon or poin,
            # both energies will be used
            mask = (
                (self.tau_tables[:, 2] == 211) | (self.tau_tables[:, 2] == -211)
            ) | ((self.tau_tables[:, 2] == 321) | (self.tau_tables[:, 2] == -321))
        else:
            mask = np.abs(self.tau_tables[:, 2]) == particle_id

        # each row has [event_num, decay_code, daughter_pid, mother_pid, energy ]
        # here we only want the event num, decay code, and energy scaling.
        energies = self.tau_tables[mask][:, [0, 1, -1]]
        return energies

    def scale_energies(self, energies, single_shower):
        scaled_shower = single_shower * energies[:, -1][: self.nshowers][:, np.newaxis]
        labeled = np.concatenate(
            (energies[:, :2][: self.nshowers], scaled_shower), axis=1
        )

        return labeled

    def composite(self, single_showers):

        # make composite showers
        single_showers = single_showers[single_showers[:, 0].argsort()]
        grps, idx, num_showers_in_evt = np.unique(
            single_showers[:, 0], return_index=True, return_counts=True, axis=0
        )
        unique_decay_codes = np.take(single_showers[:, 1], idx)
        composite_showers = np.column_stack(
            (
                grps,
                unique_decay_codes,
                np.add.reduceat(single_showers[:, 2:], idx),
            )
        )
        return composite_showers

    def __call__(self):
        stacked_unsummed = []
        for p, init in enumerate(self.pid):
            s = self.showers[p]
            energy = self.tau_daughter_energies(particle_id=init)
            scaled_s = self.scale_energies(energies=energy, single_shower=s)
            stacked_unsummed.append(scaled_s)

        stacked_unsummed = np.concatenate(stacked_unsummed, axis=0)
        print(stacked_unsummed.shape)
        return self.composite(stacked_unsummed)


# #%% Code example
# =============================================================================
# tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
# tup_folder = "C:/Users/144/Desktop/g_drive/Research/NASA/Work/conex2r7_50-runs"
# read in pythia decays
# import matplotlib.pyplot as plt
# import os
# from scipy.optimize import curve_fit
# # we can read in the showers with different primaries
# elec_init = ReadConex(
#     os.path.join(
#         tup_folder,
#         "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_2033993834_11.root",
#     )
# )
# pion_init = ReadConex(
#     os.path.join(
#         tup_folder,
#         "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_730702871_211.root",
#     )
# )
# gamma_init = ReadConex(
#     os.path.join(
#         tup_folder,
#         "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
#     )
# )
# # we can get the charged compoenent
# elec_charged = elec_init.get_charged()
# gamma_charged = gamma_init.get_charged()
# pion_charged = pion_init.get_charged()
# depths = elec_init.get_depths()
#
# # note, once can also generate compoosites using any component, e.g. electron component
# # elec_elec = elec_init.get_elec()
# # gamma_elec = gamma_init.get_elec()
# # pion_elec = pion_init.get_elec()
#
# pids = [11, 22, 211]
# init = [elec_charged, gamma_charged, pion_charged]
# gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
# comp_charged = gen_comp()
#
# fig, ax = plt.subplots(
#     nrows=1, ncols=1, dpi=300, figsize=(8, 5), sharey=True, sharex=True
# )
# # ax = ax.ravel()
# ax.plot(
#     depths[0, :].T,
#     np.log10(comp_charged[:, 2:].T),
#     color="tab:blue",
#     alpha=0.2,
#     label="Charged, Scaled, Summed",
# )
#
# =============================================================================
