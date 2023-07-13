import h5py
import numpy as np
from nuspacesim.simulation.eas_composite.comp_eas_utils import get_decay_channel
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
from scipy.optimize import fsolve
from scipy.signal import argrelextrema
from scipy.stats import exponnorm

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


class CompositeEAS:
    """
    Generating composite showers using the profiles themselves from conex,
    not just the GH for 10^17 eV for 5 degree earth emergence angles
    """

    def __init__(
        self,
        beta: int = 5,
        log_e: int = 17,
        start_elevation: int = 0,
        tau_table_start: int = 0,
    ):
        if (beta != 5) or (log_e != 17):
            print("Note, we currently only have conex data for beta = 5deg, e=100PeV")

        with as_file(
            files("nuspacesim.data.pythia_tau_decays") / "tau_100_PeV.h5"
        ) as path:
            data = h5py.File(path, "r")
            tau_decays = np.array(data.get("tau_data"))

        self.tau_tables = tau_decays[tau_table_start:, :]

        self.lepton_decay = [300001, 300002]
        self.had_pionkaon_1bod = [200011, 210001]
        # fmt: off
        self.had_pi0 = [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111,
        500131, 500311, 501211, 501212, 510301, 510121, 510211, 510111, 510112, 600411,
        600231,
        ]
        self.had_no_pi0 = [310001, 311001, 310011, 311002, 311003, 400031, 410021,
        410011, 410012, 410013, 410014, 501031, 501032, 510031, 600051,
        ]
        # fmt: on

        # load params extracted from fluctuate_hadronic.py
        with as_file(
            files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
        ) as path:
            data = h5py.File(path, "r")

            self.leptonic = np.array(data["leptonic"])
            self.one_body_kpi = np.array(data["one_body_kpi"])
            self.with_pi0 = np.array(data["with_pi0"])
            self.no_pi0 = np.array(data["no_pi0"])

            self.mean_leptonic = np.array(data["mean_leptonic"])
            self.mean_one_body_kpi = np.array(data["mean_one_body_kpi"])
            self.mean_with_pi0 = np.array(data["mean_with_pi0"])
            self.mean_no_pi0 = np.array(data["mean_no_pi0"])

            self.depths = np.array(data["slant_depth"])

        # 11 is electron
        # 22 is gamma
        # +/- 211 and +/- 321 treated as the same. kaons and pions.

        # calculate branching ratios
        # decay_channels, shwrs_perchannel = np.unique(
        #     self.tau_tables[:, 1].astype("int"), return_counts=True
        # )
        # most_common_sort = np.flip(shwrs_perchannel.argsort())
        # decay_channels = decay_channels[most_common_sort]
        # shwrs_perchannel = shwrs_perchannel[most_common_sort]
        # branch_percent = shwrs_perchannel / np.sum(shwrs_perchannel)
        # decay_labels = [get_decay_channel(x) for x in decay_channels]

    def __call__(
        self,
        n_comps: int = 1000,
        # channel=None,
        # return_table=False,
        # shower_seed=False,
    ):
        # get rows with daughter particles we care about
        wanted_pids = [211, 321, -211, -321, 11, 22]
        trimmed_table = np.isin(self.tau_tables[:, 2], wanted_pids)
        trimmed_table = self.tau_tables[trimmed_table]

        evt_nums = list(sorted(set(trimmed_table[:, 0])))
        resampled_evts = np.random.choice(evt_nums, size=n_comps)
        # loop through resample events, and construct a "new" table with resamples
        # to make sure n_comps is met
        new_table = []

        showers = []

        for pseudo_evtnum, enum in enumerate(resampled_evts, start=1):
            sampled_event = trimmed_table[trimmed_table[:, 0] == enum]
            # pseudo_tags = pseudo_evtnum * np.ones(sampled_event  .shape[0])
            channel = sampled_event[:, 1][0]
            tagged_sampled_event = np.insert(sampled_event, 0, pseudo_evtnum, axis=1)
            new_table.append(tagged_sampled_event)

            # print(enum)
            # print(channel)

            if np.isin(channel, self.lepton_decay):
                lamb = self.leptonic[0]
                sig = self.leptonic[1]
                mu = self.leptonic[2]
                right_trunc = self.leptonic[3]  # right truncation of the distribution

                mean = self.mean_leptonic

            elif np.isin(channel, self.had_pionkaon_1bod):
                lamb = self.one_body_kpi[0]
                sig = self.one_body_kpi[1]
                mu = self.one_body_kpi[2]
                right_trunc = self.one_body_kpi[3]

                mean = self.mean_one_body_kpi

            elif np.isin(channel, self.had_pi0):
                lamb = self.with_pi0[0]
                sig = self.with_pi0[1]
                mu = self.with_pi0[2]
                right_trunc = self.with_pi0[3]

                mean = self.mean_with_pi0

            elif np.isin(channel, self.had_no_pi0):
                lamb = self.no_pi0[0]
                sig = self.no_pi0[1]
                mu = self.no_pi0[2]
                right_trunc = self.no_pi0[3]

                mean = self.mean_no_pi0

            # while is not good, not sure how to approach other way
            k = 1 / (lamb * sig)  # shape paremeter
            while len(showers) != n_comps:
                r = exponnorm.rvs(k, loc=mu, scale=sig)
                # print(r)
                if (r > 0) and (r <= right_trunc):
                    showers.append(np.expand_dims(r * mean, axis=1))

        new_table = np.concatenate(new_table)

        return np.hstack(showers).T, self.depths


#%%
import matplotlib.pyplot as plt

gen = CompositeEAS(beta=5, log_e=17)
comps, depth = gen(n_comps=1000)
#%%
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    dpi=300,
    figsize=(4, 4),
    sharey=True,
)
ax.plot(depth, comps.T, color="grey", alpha=0.5)
ax.set(yscale="log", ylim=(100, 3e8))
