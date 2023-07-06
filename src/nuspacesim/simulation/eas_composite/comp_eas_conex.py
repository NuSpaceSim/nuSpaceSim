"""
generating composite showers using the profiles themselves from conex, not just the GH
100 PeV or 10^17 eV for 5 degree earth emergence angles
"""

import numpy as np
import h5py
from nuspacesim.simulation.eas_composite.comp_eas_utils import (
    numpy_argmax_reduceat,
    get_decay_channel,
)

from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from scipy.optimize import fsolve
from scipy.signal import argrelextrema

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


class ConexCompositeShowers:
    """
    Generating composite showers using the profiles themselves from conex,
    not just the GH for 10^17 eV for 5 degree earth emergence angles
    """

    def __init__(
        self,
        shower_comps: list,
        init_pid: list,
        beta: int = 5,
        shwr_per_file: int = 1000,
        tau_table_start: int = 0,
        log_e: int = 17,
    ):
        if (beta != 5) or (log_e != 17):
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

        # calculate branching ratios
        decay_channels, shwrs_perchannel = np.unique(
            self.tau_tables[:, 1].astype("int"), return_counts=True
        )
        most_common_sort = np.flip(shwrs_perchannel.argsort())
        decay_channels = decay_channels[most_common_sort]
        shwrs_perchannel = shwrs_perchannel[most_common_sort]
        branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)
        decay_labels = [get_decay_channel(x) for x in decay_channels]
        # print(decay_labels, 100 * branch_percent)

    def tau_daughter_energies(self, particle_id):
        r"""Isolate energy contributions given a specific pid
        which are used to scale Nmax of shower components

        Parameters
        ----------
        particle_id: int

        Returns
        -------
        energies: float
            scaling energies from all the pythia tables.
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
        r"""Scale a shower component with pythia

        Parameters
        ----------
        energies: float
            scaling energy to scale the charge component
        single_shower: array
            CONEX Profile

        Returns
        -------
        label: array
            scaled and labeled showers based on the shower number and decay
        """

        scaled_shower = (
            single_shower * energies[:, -1][: single_shower.shape[0]][:, np.newaxis]
        )
        labeled = np.concatenate(
            (energies[:, :2][: single_shower.shape[0]], scaled_shower), axis=1
        )

        return labeled

    def composite(self, single_showers):
        r"""Composite shower based on the tagged single showers with decay channels

        Parameters
        ----------
        single_showers: array
            single showers with decay code and event number

        Returns
        -------
        composite_showers: array
            each row is summed bin-by-bin with charged shower component contribution
            from all the showers
        """
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

    def __call__(
        self,
        n_comps=None,
        channel=None,
        return_table=False,
        no_subshwrs=False,
        shower_seed=False,
    ):

        if n_comps == None and channel == None:
            r"""
            this summing method goes in order of the decay table, and is based on
            needing to use up all shower profiles uniquely.
            this runs into the problem that you get more chances for a single electron
            shower + electron neutrino shower since you have more chances of forming it.

            we resolve it by shifting to a more table-centric shower generation below,
            where
            we sample events randomly and then randomly draw a corresponding shower
            to fullfill the requirements to make that decay event turned composite shower.

            """

            stacked_unsummed = []
            for p, init in enumerate(self.pid):
                s = self.showers[p]  # take a single particle shower initiated run
                # take scaling energies for a specific daughter particle
                energy = self.tau_daughter_energies(particle_id=init)
                # scale the shower by the energy
                scaled_s = self.scale_energies(energies=energy, single_shower=s)
                stacked_unsummed.append(scaled_s)

            stacked_unsummed = np.concatenate(stacked_unsummed, axis=0)
            # print(stacked_unsummed.shape)
            return self.composite(stacked_unsummed)

        elif n_comps is not None or channel is not None:
            # trim the table to the desired number of composites,
            # in this case no need to oversample

            print("> Randomly sampling tau decay tables")
            #!!! tables, oversample decays, by adding another tag, pseudo decay tag.

            # get rows with daughter particles we care about
            wanted_pids = [211, 321, -211, -321, 11, 22]
            trimmed_table = np.isin(self.tau_tables[:, 2], wanted_pids)
            trimmed_table = self.tau_tables[trimmed_table]
            if channel is not None:

                if isinstance(channel, list):
                    print("> Limiting Decay Channels to", channel)
                    # print(" ", get_decay_channel(channel))
                    trimmed_table = trimmed_table[np.isin(trimmed_table[:, 1], channel)]

                else:

                    print("> Limiting Decay Channel to", channel)
                    # print(" ", get_decay_channel(channel))
                    trimmed_table = trimmed_table[trimmed_table[:, 1] == channel]

                # print(trimmed_table.shape)
            evt_nums = list(sorted(set(trimmed_table[:, 0])))
            resampled_evts = np.random.choice(evt_nums, size=n_comps)
            # loop through resample events, and construct a "new" table with resamples
            # to make sure n_comps is met
            new_table = []
            for pseudo_evtnum, e in enumerate(resampled_evts, start=1):
                sampled_event = trimmed_table[trimmed_table[:, 0] == e]
                # pseudo_tags = pseudo_evtnum * np.ones(sampled_event.shape[0])
                tagged_sampled_event = np.insert(
                    sampled_event, 0, pseudo_evtnum, axis=1
                )
                new_table.append(tagged_sampled_event)
            new_table = np.concatenate(new_table)

            stacked_unsummed = []
            showers_used_idx = []

            for p, init in enumerate(self.pid):
                # print(init)
                if (init == 211) or (init == 321):  # pion kaon same
                    mask = (np.abs(new_table[:, 3]) == 321) | (
                        np.abs(new_table[:, 3]) == 211
                    )
                else:
                    mask = np.abs(new_table[:, 3]) == init
                # this is all the energies associated with a given daughter paticle
                # [pseudo event number, decay tag, scaling energy]
                energies = new_table[mask][:, [0, 2, -1]]

                # oversample shower profiles
                original_showers = self.showers[p]  # single showers

                if no_subshwrs is True:
                    no_subshwr_idx = []  # index of composite showers with subshowers
                    for i, s in enumerate(original_showers):
                        num_of_extrema = len(argrelextrema(np.log10(s), np.greater)[0])
                        if num_of_extrema <= 2:
                            # sub_showers = False
                            # ax.plot(depths[0, :], s[2:], lw=1, color="tab:blue", alpha=0.2)
                            no_subshwr_idx.append(i)
                        else:
                            # sub_showers = True
                            # ax.plot(depths[0, :], s[2:], lw=1, alpha=0.25, zorder=12)
                            pass
                    no_subshwr_idx = np.array(no_subshwr_idx)
                    original_showers = original_showers[no_subshwr_idx]

                rand = np.random.randint(
                    low=0, high=original_showers.shape[0], size=energies.shape[0]
                )
                if shower_seed is True:
                    showers_used_idx.append(rand)

                sampled_showers = np.take(original_showers, rand, axis=0)
                # print(energies.shape)
                # print(self.showers[p].shape[0])
                # print(sampled_showers.shape)
                scaled_s = self.scale_energies(
                    energies=energies, single_shower=sampled_showers
                )
                stacked_unsummed.append(scaled_s)

            stacked_unsummed = np.concatenate(stacked_unsummed, axis=0)

        if return_table is True:
            print("> Returning flattened, unsummed, randomly sampled table.")
            return self.composite(stacked_unsummed), new_table
        else:
            return self.composite(stacked_unsummed)

        # approach, number of showers, sample the 10k events uniformly
        # based on those return the events, can have double count events
        # based on the needs of the tau tables,
        # daughter energies are pulled from the showers based on how many is needed
        # by the number of tau samples
        # make sure that the tau samples dont' have consecutive


#%% Code example
# tup_folder = "/home/fabg/g_drive/Research/NASA/Work/conex2r7_50-runs/"
# # tup_folder = "C:/Users/144/Desktop/g_drive/Research/NASA/Work/conex2r7_50-runs"
# # read in pythia decays
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
# #%%
# # note, once can also generate compoosites using any component, e.g. electron component
# # elec_elec = elec_init.get_elec()
# # gamma_elec = gamma_init.get_elec()
# # pion_elec = pion_init.get_elec()

# pids = [11, 22, 211]
# init = [elec_charged, gamma_charged, pion_charged]
# gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
# comp_charged = gen_comp(n_comps=5000)  #!!! todo resample table based on composites
# #%%
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
