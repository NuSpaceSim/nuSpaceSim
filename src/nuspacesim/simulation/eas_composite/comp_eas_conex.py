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
from nuspacesim.simulation.eas_composite.comp_eas_utils import (
    decay_channel_filter,
    slant_depth_to_alt,
)
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


class ConexCompositeShowers:
    """
    Generating composite showers using the profiles themselves from CONEX,
    not just the GH for 10^17 eV for 5 degree earth emergence angles

    Parameters
    ----------
    shower_comps : list of arrays
        shower components, created with the charged components of
    init_pid : list
        PID number of the primary initiating particles. See PDG tables.
    beta : int, optional
        Earth emergence angle in degrees. The default is 5.
    shwr_per_file : int, optional
        the number of shower per CONEX file enetered in the shower_comps .
        The default is 1000.
    tau_table_start : int, optional
        Where to start sampling the tau tables. Flag is irrelevant when the
        ConexCompositeShowers class is called with the n_showers flag.
        The default is 0.
    log_e : int, optional
        log10 of the Energy of the primary particles. The default is 17.


    Returns
    -------
    A shower generator class that can be called given additional calls.

    Examples
    -------
    >>> elec_init = ReadConex(
        os.path.join(
            tup_folder,
            "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_2033993834_11.root",
        )
    )
    >>> pion_init = ReadConex(
        os.path.join(
            tup_folder,
            "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_730702871_211.root",
        )
    )
    >>> gamma_init = ReadConex(
        os.path.join(
            tup_folder,
            "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
        )
    )
    # we can get the charged compoenent
    >>> elec_charged = elec_init.get_charged()
    >>> gamma_charged = gamma_init.get_charged()
    >>> pion_charged = pion_init.get_charged()
    >>> depths = elec_init.get_depths()
    # note, once can also generate compoosites using any component,
    # e.g. electron component
    # elec_elec = elec_init.get_elec()
    # gamma_elec = gamma_init.get_elec()
    # pion_elec = pion_init.get_elec()
    >>> pids = [11, 22, 211]
    >>> init = [elec_charged, gamma_charged, pion_charged]
    >>> gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
    >>> comp_charged = gen_comp(n_comps=5000)

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
        n_comps: int = None,
        channel: int = None,
        return_table: bool = False,
        no_subshwrs: bool = False,
        shower_seed: list = False,
    ):
        """
        Call the CONEX Composite EAS generator

        Parameters
        ----------
        n_comps : int, optional
            Number of composite showers to generate. Will oversample if the number of
            showers in the libraries entered can't cover the number of shwoers asked for
            If left None,  goes in order of the decay table, and is based on needing
            to use up all shower profiles uniquely.
            This runs into the problem that you get more chances for a single electron
            shower + electron neutrino shower since this decay channel only needs
            one shower.
            This is quirk is left so that one can directly compare with GH profiles.
            The default is None.
        channel : int, optional
            If not None, restricts the composite shower generation
            strictly to the decay channels given. The default is None.
        return_table : bool, optional
            Return the (pseudo/oversampled) tau decay table used to generate/ group
            showers before summing.
            The default is False.
        no_subshwrs : bool, optional
            If True, will remove composite EASs with showers after generation.
            The default is False.
        shower_seed : list, optional
            Feature in progress, will return a list of the showers used from the CONEX
            .root ntuple.
            The default is False.

        Returns
        -------
        comp_showers
            composite showers. make sure you have the corresponding slant depths
        tau_tables
            optionally, return the (oversampled) tau tables

        """

        if n_comps == None and channel == None:
            r"""
            since the user does not indicate the desired number of showers,
            this summing method goes in order of the decay table, and is based on
            needing to use up all shower profiles uniquely.
            this runs into the problem that you get more chances for a single electron
            shower + electron neutrino shower since you have more chances of forming it.

            we resolve it by shifting to a more table-centric shower generation below,
            where the user does input the desired number of showers.
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


# %% Code example
"""
Example on how to generate composite EAS from CONEX Profiles
"""

# tup_folder = "/home/fabg/gdrive_umd/Research/NASA/Work/conex2r7_50-runs/"
# note, this can be found in the google drive
# https://drive.google.com/drive/u/2/folders/1wbvLxs71s1LmU88tWqwPfMP5RxCUT7OA
# in the nuSpaceSim/CONEXresults/conex2r7_50-runs/1000_evts_0km_start


# read in pythia decays


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

# # note, once can also generate compoosites using any component, e.g. electron component
# # elec_elec = elec_init.get_elec()
# # gamma_elec = gamma_init.get_elec()
# # pion_elec = pion_init.get_elec()

# pids = [11, 22, 211]
# init = [elec_charged, gamma_charged, pion_charged]
# gen_comp = ConexCompositeShowers(shower_comps=init, init_pid=pids)
# comp_charged = gen_comp(n_comps=5000)  #!!! todo resample table based on composites

# ##=============================================================================
# ## %%  diagnostics plots
# ##=============================================================================


# #  segregate by decay channel and see most common one
# decay_channels, shwrs_perchannel = np.unique(
#     comp_charged[:, 1].astype("int"), return_counts=True
# )
# most_common_sort = np.flip(shwrs_perchannel.argsort())
# decay_channels = decay_channels[most_common_sort]

# shwrs_perchannel = shwrs_perchannel[most_common_sort]
# branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# # sum channels that contributed less than 3 percent to the decay
# other_mask = branch_percent < 0.02
# other_category = np.sum(shwrs_perchannel[other_mask])
# decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
# decay_labels.append("other")

# fig, ax = plt.subplots(nrows=1, ncols=1, dpi=300, figsize=(5, 4))
# ax.pie(
#     np.append(shwrs_perchannel[~other_mask], other_category),
#     labels=decay_labels,
#     autopct="%1.0f%%",
#     # rotatelabels=True,
#     startangle=75,
#     pctdistance=0.8,
#     radius=1.0,
#     # labeldistance=None,
#     textprops={"fontsize": 10},
# )
# ax.text(
#     -0.2,
#     0.88,
#     r"${{\rm{:.0f}\:Composites}}$"
#     "\n"
#     r"${{\rm 100\:PeV}}$"
#     "\n"
#     r"${{\rm \beta\:=\:5\:\degree}}$".format(comp_charged.shape[0]),
#     transform=ax.transAxes,
# )

# # plt.savefig(
# #     "../../../../../g_drive/Research/NASA/composite_branching_ratio.png",
# #     dpi=300,
# #     bbox_inches="tight",
# #     pad_inches=0.05,
# # )

# # filter out composites with subshowers

# #!!! how to add stochastic process in the future?

# subshwr_idx = []  # index of composite showers with subshowers
# fig, ax = plt.subplots(nrows=1, ncols=1, dpi=200, figsize=(4, 3))
# for i, s in enumerate(comp_charged):
#     num_of_extrema = len(argrelextrema(np.log10(s), np.greater)[0])
#     if num_of_extrema <= 2:
#         # sub_showers = False
#         ax.plot(depths[0, :], np.log10(s[2:]), lw=1, color="tab:blue", alpha=0.2)

#     else:
#         # sub_showers = True
#         ax.plot(
#             depths[0, :], np.log10(s[2:]), lw=1, color="tab:red", alpha=0.25, zorder=12
#         )
#         subshwr_idx.append(i)

# ax.set(ylim=(0, 8), xlabel=r"${\rm slant\:depth\:(g \: cm^{-2})}$", ylabel=r"${\rm N}$")
# ax_twin = ax.twiny()
# ax_twin.plot(depths[0, :], np.log10(s[2:]), alpha=0)

# ax_twin.set_xticklabels(
#     list(
#         np.round(
#             slant_depth_to_alt(
#                 earth_emergence_ang=5, slant_depths=ax.get_xticks(), alt_stop=200
#             ),
#             1,
#         ).astype("str")
#     )
# )
# ax_twin.set(xlabel=r"${\rm altitude\:(km)}$")

# # plot eas with sub shower vs no sub shower branching ratio
# fig, ax = plt.subplots(nrows=2, ncols=1, dpi=300, figsize=(10, 5))
# ax[0].pie(
#     np.append(shwrs_perchannel[~other_mask], other_category),
#     labels=decay_labels,
#     autopct="%1.0f%%",
#     # rotatelabels=True,
#     startangle=75,
#     pctdistance=0.8,
#     radius=1.2,
#     # labeldistance=None,
# )
# # ax.legend(
# #     title=r"100 PeV Composites, $\beta = 5 \degree$",
# #     ncol=2,
# #     bbox_to_anchor=(0.08, 1.1),
# #     fontsize=5,
# # )

# ax[0].text(
#     -0.3,
#     0.98,
#     r"${{\rm {:.0f}\:Composites}}$"
#     "\n"
#     r"${{\rm 100\:PeV}}$"
#     "\n"
#     r"${{\rm \beta\:=\:5\:\degree}}$".format(comp_charged.shape[0]),
#     transform=ax[0].transAxes,
# )

# subshwr_idx = np.array(subshwr_idx)
# comp_sub = comp_charged[subshwr_idx]

# decay_channels, shwrs_perchannel = np.unique(
#     comp_sub[:, 1].astype("int"), return_counts=True
# )
# most_common_sort = np.flip(shwrs_perchannel.argsort())
# decay_channels = decay_channels[most_common_sort]

# shwrs_perchannel = shwrs_perchannel[most_common_sort]
# branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)

# # sum channels that contributed less than 3 percent to the decay
# other_mask = branch_percent < 0.01
# other_category = np.sum(shwrs_perchannel[other_mask])
# decay_labels = [get_decay_channel(x) for x in decay_channels[~other_mask]]
# decay_labels.append("other")

# ax[1].pie(
#     np.append(shwrs_perchannel[~other_mask], other_category),
#     labels=decay_labels,
#     autopct="%1.0f%%",
#     # rotatelabels=True,
#     startangle=0,
#     pctdistance=0.8,
#     # explode=[0, 0, 0, 0, 0, 0, 0.1],
#     radius=1.2,
#     # labeldistance=None,
# )

# ax[1].text(
#     -0.3,
#     0.98,
#     r"${{\rm{:.0f}\:Composites, with\:sub-showers}}$" "\n".format(comp_sub.shape[0]),
#     transform=ax[1].transAxes,
# )
