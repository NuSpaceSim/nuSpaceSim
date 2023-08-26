import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py
from scipy.stats import exponnorm
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
from nuspacesim.simulation.eas_composite.comp_eas_utils import mean_shower
import os
from nuspacesim.simulation.eas_composite.conex_interface import ReadConex
import matplotlib.lines as mlines

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):
    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x**2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x**2))))

    return particles


class CompositeShowers:
    """
    Main light-weight composite EAS generator made by sampling full CONEX showers.
    Light-weight version of ConexCompositeShowers, with slight quirks.
    Uses 5th order polynomial interpolation to piece together the bulk and tail of
    the showers.
    Modelled using conex2r7_50, using primary particles gamma, electron, and pion
    (which are treated identically as kaon).

    Parameters
    ----------
    beta : int, optional
        Earth emergence angle in degrees. The default is 5.
    log_e : int, optional
        Primary energu of the constituent showers. The default is 17.

    Returns
    -------
    A shower generator class that can be called given additional calls.

    Examples
    -------
    >>>  synth_generator = CompositeShowers(beta=5, log_e=17)
    >>> eas, depth = synth_generator(n_comps=20, return_depth=True)

    """

    def __init__(
        self,
        beta: int = 5,
        log_e: int = 17,
    ):
        self.emer_ang = beta
        self.prim_log_energy = log_e

        self.lepton_decay = [300001, 300002]
        self.had_pionkaon_1bod = [200011, 210001]
        # fmt: off
        self.had_pi0 = [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111, 
                   500131,500311, 501211, 501212, 510301, 510121, 510211, 510111, 
                   510112, 600411, 600231,
                   ]
        self.had_no_pi0 = [310001, 311001, 310011, 311002, 311003, 400031, 410021, 
                           410011, 410012, 410013, 410014, 501031, 501032, 510031, 
                           600051,
                      ]
        # fmt: on

        groupings = ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]

        # read the relevant tables
        with as_file(
            files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
        ) as path:
            self.nmaxdata = h5py.File(path, "r")

        with as_file(
            files("nuspacesim.data.pythia_tau_decays") / "tau_100_PeV.h5"
        ) as path:
            data = h5py.File(path, "r")
            tau_decays = np.array(data.get("tau_data"))

            self.tau_tables = tau_decays

        with as_file(
            files("nuspacesim.data.eas_reco.rms_params") / "xmax_rms_params.h5"
        ) as path:
            self.xmaxdata = h5py.File(path, "r")

        with as_file(
            files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
        ) as path:
            data = h5py.File(path, "r")

            self.slantdepth = np.array(data["slant_depth"])

        with as_file(
            files("nuspacesim.data.eas_reco") / "mean_shwr_bulk_gh_params.h5"
        ) as path:
            bulk_gh_data = h5py.File(path, "r")

            # each key has the order [nmax, xmax, x0, p1, p2, p3]
            self.leptonic_gh = np.array(bulk_gh_data["leptonic"])
            self.one_body_gh = np.array(bulk_gh_data["one_body_kpi"])
            self.with_pi0_gh = np.array(bulk_gh_data["with_pi0"])
            self.no_pi0_gh = np.array(bulk_gh_data["no_pi0"])

        with as_file(
            files("nuspacesim.data.eas_reco.elongation_rates") / "1000_evts.h5"
        ) as path:
            with h5py.File(path, "r") as f:
                elong_charged = np.array(f["charged"])

        with as_file(
            files("nuspacesim.data.eas_reco.energy_scaling") / "1000_evts.h5"
        ) as path:
            with h5py.File(path, "r") as f:
                energy_charged = np.array(f["charged"])

        if beta not in elong_charged[:, 0]:
            print("Earth Emergence Angle not yet supported")
            print("Avaialable value of beta are")
            print(elong_charged[:, 0])
            raise ValueError
        else:
            base_energy = 17
            elong_params = elong_charged[:, 1:][elong_charged[:, 0] == beta]
            elong_slope = elong_params[0][0]
            elong_intercept = elong_params[0][2]

            mean_xmax = (elong_slope * log_e) + elong_intercept
            ref_mean_xmax = (elong_slope * base_energy) + elong_intercept

            self.elongation_scaling = mean_xmax / ref_mean_xmax

            energy_params = energy_charged[:, 1:][energy_charged[:, 0] == beta]
            energy_slope = energy_params[0][0]
            energy_intercept = energy_params[0][2]

            log_mean_nmax = (energy_slope * log_e) + energy_intercept
            log_ref_mean_nmax = (energy_slope * base_energy) + energy_intercept

            self.energy_scaling = 10**log_mean_nmax / 10**log_ref_mean_nmax

            # print()

    def nmax_sampler(self, n_showers: int, reco_type: str):
        """
        Take values stored in "nuspacesim.data.eas_reco.rms_params" to get scaling
        values for the mean 100 PeV shower.

        Parameters
        ----------
        n_showers : int
            number of showers to make.
        reco_type : str
            grouping. ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]

        Returns
        -------
        nmaxmult : float
            nmax multipliers

        """
        nmax_params = np.array(self.nmaxdata[reco_type])

        lamb = nmax_params[0]
        sig = nmax_params[1]
        mu = nmax_params[2]
        right_trunc = nmax_params[3]

        # pull from the nmax distribution
        nmaxmult = []
        while len(nmaxmult) != n_showers:
            r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
            # print(r)
            if (r > 0) and (r <= right_trunc):
                nmaxmult.append(r)
        return nmaxmult

    def xmax_sampler(self, n_showers: int, reco_type: str):
        """
        Take values stored in "nuspacesim.data.eas_reco.rms_params" to get Xmax scaling
        values for the mean 100 PeV shower.

        Parameters
        ----------
        n_showers : int
            number of showers to make.
        reco_type : str
            grouping. ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]

        Returns
        -------
        xmaxmult : float
            xmax multipliers

        """
        xmax_params = np.array(self.xmaxdata[reco_type])

        lamb = xmax_params[0]
        sig = xmax_params[1]
        mu = xmax_params[2]
        left_trunc = xmax_params[3]
        right_trunc = xmax_params[4]
        # pull from the xmax distribution
        xmaxmult = []
        while len(xmaxmult) != n_showers:
            r = exponnorm.rvs(1 / (lamb * sig), loc=mu, scale=sig)
            # print(r)
            if (r >= left_trunc) and (r <= right_trunc):
                xmaxmult.append(r)
        return xmaxmult

    def get_mean(self, reco_type):
        return np.array(self.nmaxdata["mean_" + reco_type])

    def synthetic_eas(self, mean, nmax_mult, xmax_mult, gh_params):
        """
        Generate syntehtic Composite EAS by piecing together the bulk and shower tails.

        Parameters
        ----------
        mean : float
            shower mean
        nmax_mult :
            multiplier for nmaxs
        xmax_mult :
            multiplier for xmaxs, same length as nmax
        gh_params :
            GH parameters describing the bulk of the mean. this is varied by values
            in nmax_mult and xmax_mult

        Returns
        -------
        synthetic_composite_eas :
            composite EAS showers pieced together by interpolation.

        """
        nmax, xmax, x0, p1, p2, p3 = list(gh_params)

        # if the energy is not 100 PeV and earth emergence angle is not  5 degrees
        # xmax = xmax * self.elongation_scaling
        # nmax = nmax * self.energy_scaling

        synthetic_composite_eas = []
        for n, x in zip(nmax_mult, xmax_mult):
            # we need this for the tail
            vertically_shifted = mean * self.energy_scaling * n
            # print(self.elongation_scaling, self.energy_scaling)
            bulk_varied = modified_gh(
                self.slantdepth,
                nmax * self.energy_scaling * n,
                xmax * self.elongation_scaling * x,
                x0,
                p1,
                p2,
                p3,
            )
            # we peiece together the shifted bulk and scaled tail
            # the range for spline is dictated by the store Xmax gh fit for each channel

            if self.prim_log_energy <= 15:
                s_gram = xmax * 1.1 * self.elongation_scaling
                e_gram = xmax * 2.2 * self.elongation_scaling
            else:
                s_gram = xmax * 1.8 * self.elongation_scaling
                e_gram = xmax * 2.3 * self.elongation_scaling

            s_spline = np.argmin(np.abs(self.slantdepth - s_gram))
            e_spline = np.argmin(np.abs(self.slantdepth - e_gram))

            depth_tail = self.slantdepth[e_spline:]
            shwr_tail = vertically_shifted[e_spline:]
            depth_bulk = self.slantdepth[:s_spline]
            shwr_bulk = bulk_varied[:s_spline]  # from theory

            # synthetic shower with gap that we piece together
            shower_synth = np.concatenate((shwr_bulk, shwr_tail))
            depth_synth = np.concatenate((depth_bulk, depth_tail))

            spline_synthetic = interpolate.interp1d(depth_synth, shower_synth, kind=5)

            synthetic_composite_eas.append(spline_synthetic(self.slantdepth))

        return synthetic_composite_eas

    def __call__(
        self,
        n_comps: int = 1000,
        channel: int = None,
        return_table: bool = False,
        return_depth: bool = False,
    ):
        """
        Call the synthetic generator

        Parameters
        ----------
        n_comps : int, optional
            number of composite EASs to generate. follows the (modified) branching ratios,
            granted that we only use, gamma, pion/kaon, and electron primary showers.
            The default is 1000.
        channel : int, optional
            limit generation to a specific channel. see decay codes.
            The default is None.
        return_table : bool, optional
            return the tau decay tables from pythia, which would be used if the genreation
            was done via CONEX
            The default is False.
        return_depth : bool, optional
            return the slant depth for the showers The default is False.


        Returns
        -------
        call : float
            the showers with the first column being the shower number,
            the second column being the corresponding event number in the PYTHIA
            decay tables.
            the third column being the decay channel code.

        """
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

        # sample_index = np.where(np.in1d(trimmed_table[:, 0], resampled_evts))[0]

        # loop through resample events, and construct a "new" table with resamples
        # to make sure n_comps is met
        new_table = []
        sampled_channels = []

        out = np.zeros((n_comps, len(self.slantdepth) + 3))

        for pseudo_evtnum, e in enumerate(resampled_evts, start=1):
            sampled_event = trimmed_table[trimmed_table[:, 0] == e]
            decay_channel = sampled_event[:, 1][0]
            # pseudo_tags = pseudo_evtnum * np.ones(sampled_event.shape[0])
            tagged_sampled_event = np.insert(sampled_event, 0, pseudo_evtnum, axis=1)

            sampled_channels.append(decay_channel)
            new_table.append(tagged_sampled_event)

            out[pseudo_evtnum - 1, 0] = pseudo_evtnum
            out[pseudo_evtnum - 1, 1] = e
            out[pseudo_evtnum - 1, 2] = decay_channel

        new_table = np.concatenate(new_table)
        lep_loc = np.isin(sampled_channels, self.lepton_decay)
        pk1_loc = np.isin(sampled_channels, self.had_pionkaon_1bod)
        wpi_loc = np.isin(sampled_channels, self.had_pi0)
        npi_loc = np.isin(sampled_channels, self.had_no_pi0)

        #  groupings = ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]
        lep_nmax = self.nmax_sampler(sum(lep_loc), "leptonic")
        pk1_nmax = self.nmax_sampler(sum(pk1_loc), "one_body_kpi")
        wpi_nmax = self.nmax_sampler(sum(wpi_loc), "with_pi0")
        npi_nmax = self.nmax_sampler(sum(npi_loc), "no_pi0")

        lep_xmax = self.xmax_sampler(sum(lep_loc), "leptonic")
        pk1_xmax = self.xmax_sampler(sum(pk1_loc), "one_body_kpi")
        wpi_xmax = self.xmax_sampler(sum(wpi_loc), "with_pi0")
        npi_xmax = self.xmax_sampler(sum(npi_loc), "no_pi0")

        lep = self.synthetic_eas(
            self.get_mean("leptonic"), lep_nmax, lep_xmax, self.leptonic_gh
        )
        pk1 = self.synthetic_eas(
            self.get_mean("one_body_kpi"), pk1_nmax, pk1_xmax, self.one_body_gh
        )
        wpi = self.synthetic_eas(
            self.get_mean("with_pi0"), wpi_nmax, wpi_xmax, self.with_pi0_gh
        )
        npi = self.synthetic_eas(
            self.get_mean("no_pi0"), npi_nmax, npi_xmax, self.no_pi0_gh
        )
        # print(np.shape(lep))
        # print(np.shape(pk1))
        # print(np.shape(wpi))
        # print(np.shape(npi))

        # fill the output matrix
        if np.shape(lep)[0] != 0:
            out[:, 3:][lep_loc] = lep
        if np.shape(pk1)[0] != 0:
            out[:, 3:][pk1_loc] = pk1
        if np.shape(wpi)[0] != 0:
            out[:, 3:][wpi_loc] = wpi
        if np.shape(npi)[0] != 0:
            out[:, 3:][npi_loc] = npi

        call = []

        call.append(out)

        if return_table is True:
            print("> Returning flattened, randomly sampled table.")
            call.append(new_table)
        if return_depth is True:
            call.append(self.slantdepth)

        return call


# %% plot of full energy scan
# import matplotlib as mpl

# plt.rcParams.update(
#     {
#         "font.family": "serif",
#         "mathtext.fontset": "cm",
#         "xtick.labelsize": 8,
#         "ytick.labelsize": 8,
#         "font.size": 10,
#         "xtick.direction": "in",
#         "ytick.direction": "in",
#         "ytick.right": True,
#         "xtick.top": True,
#     }
# )
# energy_scan = np.linspace(15, 20, 20)
# norm = mpl.colors.Normalize(vmin=15, vmax=20)
# sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
# cmap = plt.cm.get_cmap("viridis_r")(np.linspace(0, 1, 20))

# fig, ax = plt.subplots(nrows=1, ncols=2, dpi=300, figsize=(7, 3.5))

# for n, e in enumerate(energy_scan):
#     synth_generator = CompositeShowers(beta=5, log_e=e)
#     eas, depth = synth_generator(n_comps=20, return_depth=True)
#     ax[0].plot(depth.T, eas[:, 3:].T, alpha=0.5, color=cmap[n])
# cbar_ax = ax[0].inset_axes([0, 1.03, 1, 0.05])
# fig.colorbar(
#     sm, cax=cbar_ax, orientation="horizontal", label=r"${\rm Primary\:Energy\:(eV)}$"
# )
# cbar_ax.xaxis.set_ticks_position("top")
# cbar_ax.xaxis.set_label_position("top")
# ax[0].set(
#     yscale="log",
#     ylim=(100, 1e11),
#     xlim=(0, 5000),
#     ylabel=r"$N$",
#     xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
# )
# ax[0].text(
#     0.95,
#     0.95,
#     r"${\rm \beta=5\degree}$",
#     transform=ax[0].transAxes,
#     ha="right",
#     va="top",
# )

# angle_scan = [1.0, 5.0, 15.0, 20.0, 25.0, 35.0]
# norm = mpl.colors.Normalize(vmin=1, vmax=35)
# sm = plt.cm.ScalarMappable(cmap="magma_r", norm=norm)
# cmap = plt.cm.get_cmap("magma_r")(np.linspace(0, 1, 35))
# for n, b in enumerate(angle_scan):
#     synth_generator = CompositeShowers(beta=b, log_e=17)
#     eas, depth = synth_generator(n_comps=20, return_depth=True)
#     ax[1].plot(depth.T, eas[:, 3:].T, alpha=0.5, color=cmap[int(b - 1)])

# cbar_ax = ax[1].inset_axes([0, 1.03, 1, 0.05])
# fig.colorbar(
#     sm,
#     cax=cbar_ax,
#     orientation="horizontal",
#     label=r"${\rm Earth\:Emergence\:Angle(\degree)}$",
# )
# cbar_ax.xaxis.set_ticks_position("top")
# cbar_ax.xaxis.set_label_position("top")
# ax[1].set(
#     yscale="log",
#     ylim=(100, 1e8),
#     xlim=(0, 5000),
#     ylabel=r"$N$",
#     xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
# )

# ax[1].text(
#     0.95,
#     0.95,
#     r"${\rm Primary\:Energy = 10^{17}\:eV }$",
#     transform=ax[1].transAxes,
#     ha="right",
#     va="top",
# )


# plt.savefig(
#     "../../../../../gdrive_umd/Research/NASA/energy_and_angle_scan.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )

# plt.show()


# ax[0].set_ylim(bottom=100)
