"""
analysis scripts used to write the CompositeShowers class in comp_eas.py
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import h5py
from scipy.stats import exponnorm
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):
    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x**2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x**2))))

    return particles


def pwr_law(x, a, b):
    # power law
    return a * x**b


# %%
def reco_params(n_showers, reco_type):
    """
    Pull the shower mean and sample from the  variability PDFs.

    Lepton decay codes are  [300001, 300002]
    pion or kaon 1 body  [200011, 210001]
    hadronic with pi0
    [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111, 500131,
    500311, 501211, 501212, 510301, 510121, 510211, 510111, 510112, 600411,
    600231,
    ]
    hadronic without pi0
    [310001, 311001, 310011, 311002, 311003, 400031, 410021, 410011, 410012,
    410013, 410014, 501031, 501032, 510031, 600051,]

    Parameters
    ----------
    n_showers : int
        number of showers to be made

    Returns
    -------
    mean_leptonic : array
        mean shower for leptonic decay channel
    nmaxmult :array
        length of n_showers, nmax multipliers based on the variability for a given
        energy level; i.e. 100 PeV
    xmaxmult : array
        length of n_showers, xmax multipliers

    """

    groupings = ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]
    if reco_type not in groupings:
        print("Grouping does not exist, must be in")
        print(groupings)
        raise ValueError

    with as_file(
        files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
    ) as path:
        nmaxdata = h5py.File(path, "r")

        nmax_params = np.array(nmaxdata[reco_type])
        mean = np.array(nmaxdata["mean_" + reco_type])

    nlamb = nmax_params[0]
    nsig = nmax_params[1]
    nmu = nmax_params[2]
    nright_trunc = nmax_params[3]

    with as_file(
        files("nuspacesim.data.eas_reco.rms_params") / "xmax_rms_params.h5"
    ) as path:
        xmaxdata = h5py.File(path, "r")
        xmax_params = np.array(xmaxdata[reco_type])

    xlamb = xmax_params[0]
    xsig = xmax_params[1]
    xmu = xmax_params[2]
    xleft_trunc = xmax_params[3]
    xright_trunc = xmax_params[4]

    # pull from the nmax distribution
    nmaxmult = []
    while len(nmaxmult) != n_showers:  #!!! edit this for more showers
        r = exponnorm.rvs(1 / (nlamb * nsig), loc=nmu, scale=nsig)
        # print(r)
        if (r > 0) and (r <= nright_trunc):
            nmaxmult.append(r)

    # pull from the xmax distribution
    xmaxmult = []
    while len(xmaxmult) != n_showers:
        r = exponnorm.rvs(1 / (xlamb * xsig), loc=xmu, scale=xsig)
        # print(r)
        if (r >= xleft_trunc) and (r <= xright_trunc):
            xmaxmult.append(r)

    return mean, nmaxmult, xmaxmult


def reco_slt_depth():
    with as_file(
        files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
    ) as path:
        data = h5py.File(path, "r")

        slantdepth = np.array(data["slant_depth"])

    return slantdepth


lep_m, lep_nmax, lep_xmax = reco_params(1000, reco_type="leptonic")
kpi_m, kpi_nmax, kpi_xmax = reco_params(1000, reco_type="one_body_kpi")
pi0_m, pi0_nmax, pi0_xmax = reco_params(1000, reco_type="with_pi0")
npi_m, npi_nmax, npi_xmax = reco_params(1000, reco_type="no_pi0")
slantdepth = reco_slt_depth()
nmax_multipliers = [lep_nmax, kpi_nmax, pi0_nmax, npi_nmax]
xmax_multipliers = [lep_xmax, kpi_xmax, pi0_xmax, npi_xmax]

decay_labels = [
    r"${\rm leptonic\:decay}$",
    r"${\rm  1\:body\:K,\:\pi^{+/-}}$",
    r"${\rm  hadronic\:with\:\pi_0}$",
    r"${\rm  hadronic\:no\:\pi_0}$",
]

with as_file(files("nuspacesim.data.eas_reco") / "mean_shwr_bulk_gh_params.h5") as path:
    bulk_gh_data = h5py.File(path, "r")

    # each key has the order [nmax, xmax, x0, p1, p2, p3]
    leptonic_gh = np.array(bulk_gh_data["leptonic"])
    one_body_gh = np.array(bulk_gh_data["one_body_kpi"])
    with_pi0_gh = np.array(bulk_gh_data["with_pi0"])
    no_pi0_gh = np.array(bulk_gh_data["no_pi0"])


# =============================================================================
# # reconstruct nmax xmax correlation
# =============================================================================
fig, ax = plt.subplots(
    nrows=2, ncols=2, dpi=300, figsize=(5, 5), sharey=True, sharex=True
)
ax = ax.ravel()
plt.subplots_adjust(wspace=0, hspace=0)
for i, n in enumerate([lep_m, kpi_m, pi0_m, npi_m]):
    # ax.scatter(x, np.log10(nmaxs_perchan[i]), s=1, color=cmap[i], alpha=0.5)
    cts = ax[i].hist2d(
        slantdepth[np.argmax(n)] * np.array(xmax_multipliers[i]),
        np.log10(np.max(n) * np.array(nmax_multipliers[i])),
        bins=(50, 50),
        range=[[590, 1100], [6.3, 8]],
        cmap="RdPu",
    )
    ax[i].text(
        0.95, 0.95, decay_labels[i], transform=ax[i].transAxes, ha="right", va="top"
    )


ax[i].set(ylim=(6.3, 8))
fig.text(0.5, 0.05, r"${\rm shower} \: X_{\rm max} {\rm (g\:cm^{-2})}$", ha="center")
fig.text(
    0.02,
    0.5,
    r"$\log_{10}\: {\rm shower}\:{N_{\rm max}}$",
    va="center",
    rotation="vertical",
)
cbar_ax = ax[0].inset_axes([0.00, 1.1, 2, 0.05])
cbar = fig.colorbar(cts[3], cax=cbar_ax, pad=-1, orientation="horizontal")
cbar_ax.set_title(
    r"${\rm Number\: of\: Synthetic\: Showers\:(1000\:per\:grouping)}$", size=8
)

plt.savefig(
    "../../../../../gdrive_umd/Research/NASA/synthetic_correlation.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
# %%

# =============================================================================

# fit the mean showers to describe the bulk of the shower and be able
# to vary the bulk of the shower

# =============================================================================
gh_params = []
for m in [lep_m, kpi_m, pi0_m, npi_m]:
    print(m)
    params, pcov = curve_fit(
        modified_gh,
        slantdepth,
        m,
        p0=[np.max(m), slantdepth[np.argmax(m)], 0, 70, 0, 0],
        bounds=(
            [0, 0, -1e-6, -np.inf, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        ),
    )

    nmax = params[0]
    xmax = params[1]
    x0 = params[2]
    p1 = params[3]
    p2 = params[4]
    p3 = params[5]
    print(params)
    gh_params.append(params)

keys = ["leptonic", "one_body_kpi", "with_pi0", "no_pi0"]
fname = "mean_shwr_bulk_gh_params"
with as_file(files("nuspacesim.data.eas_reco") / f"{fname}.h5") as path:
    print(path)
    with h5py.File(path, "w") as f:
        for i, gh in enumerate(gh_params):
            f.create_dataset(
                keys[i],
                data=gh,
                dtype="f",
            )


# %% shower reconstruction demo for each decay channel
showers = []
noxmaxshowers = []

cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 7))[1:]

fig, ax = plt.subplots(4, 1, dpi=300, figsize=(4.3, 11), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0, wspace=0)
ax = ax.ravel()

for i, m in enumerate([lep_m, kpi_m, pi0_m, npi_m]):
    # if i != 0:  # flag if you want to just plot a specific decay channel
    #     continue
    print(decay_labels[i])

    # fit the mean
    for d in range(1):
        # fit the bulk of the mean shower
        params, pcov = curve_fit(
            modified_gh,
            slantdepth,
            m,
            p0=[np.max(m), slantdepth[np.argmax(m)], 0, 70, 0, 0],
            bounds=(
                [0, 0, -1e-6, -np.inf, -np.inf, -np.inf],
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
            ),
        )

        nmax = params[0]
        xmax = params[1]
        x0 = params[2]
        p1 = params[3]
        p2 = params[4]
        p3 = params[5]

        scaled_m = m * nmax_multipliers[i][0]

        # scale the Xmax of the mean

        shifted_xmax = xmax * xmax_multipliers[i][0]

        # the theory produces the variability in the bulk
        theory = modified_gh(slantdepth, np.max(scaled_m), shifted_xmax, x0, p1, p2, p3)
        # we peiece together the shifted bulk and scaled tail
        # the range for spline is dictated by the store Xmax gh fit for each channel
        s_gram = xmax * 2
        e_gram = xmax * 2.6
        s_spline = np.argmin(np.abs(slantdepth - s_gram))
        e_spline = np.argmin(np.abs(slantdepth - e_gram))

        depth_tail = slantdepth[e_spline:]
        shwr_tail = scaled_m[e_spline:]
        depth_bulk = slantdepth[:s_spline]
        shwr_bulk = theory[:s_spline]  # from theory

        # shower_synth = np.concatenate(
        #     (shwr_bulk, np.array([(shwr_bulk[-1] + shwr_tail[0]) / 2]), shwr_tail)
        # )
        # depth_synth = np.concatenate(
        #     (depth_bulk, np.array([(s_gram + e_gram) / 2]), depth_tail)
        # )
        # ax.scatter(
        #     1750, (shwr_bulk[-1] + shwr_tail[0]) / 2, c="red", s=1, label="interp point"
        # )
        shower_synth = np.concatenate((shwr_bulk, shwr_tail))
        depth_synth = np.concatenate((depth_bulk, depth_tail))

        # =============================================================================
        #         # two-part spline, with power law middle
        # =============================================================================

        # # power law
        # s_pwrlaw = xmax * 1.75
        # e_pwrlaw = xmax * 2.5
        # pregap_shwr = theory[np.argmin(np.abs(slantdepth - s_pwrlaw )) : s_spline]
        # pregap_depth = slantdepth[np.argmin(np.abs(slantdepth - s_pwrlaw )): s_spline]

        # postgap_shwr = scaled_m[e_spline - 1 : e_spline]
        # postgap_depth = slantdepth[e_spline - 1 : e_spline]

        # shower_pwrlaw_synth = np.concatenate((pregap_shwr, postgap_shwr))
        # depth_pwrlaw_synth = np.concatenate((pregap_depth, postgap_depth))
        # ax.scatter(
        #     depth_pwrlaw_synth,
        #     shower_pwrlaw_synth,
        #     # color="tab:grey",
        #     s=1,
        #     label=r"{} ${{\rm mean,\:scaled}}$".format(decay_labels[i]),
        #     zorder=4,
        #     alpha=0.6,
        #     # marker="s",
        # )

        # plaw_params, _ = curve_fit(
        #     pwr_law,
        #     depth_pwrlaw_synth,
        #     shower_pwrlaw_synth,
        #     # p0=[np.max(m), slantdepth[np.argmax(m)], 0, 70, 0, 0],
        #     # bounds=(
        #     #     [0, 0, -1e-6, -np.inf, -np.inf, -np.inf],
        #     #     [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        #     # ),
        # )

        # pwrlaw_depth = slantdepth[s_spline : e_spline - 1]
        # pwrlaw_theory = pwr_law(pwrlaw_depth, plaw_params)

        # ax.plot(
        #     pwrlaw_depth,
        #     pwrlaw_theory,
        #     color="tab:grey",
        #     label=r"{} ${{\rm mean,\:scaled}}$".format(decay_labels[i]),
        #     zorder=4,
        #     alpha=0.6,
        #     # marker="s",
        # )

        # spline = CubicSpline(depth_synth, shower_synth)
        spline = interpolate.interp1d(depth_synth, shower_synth, kind=5)

        showers.append(spline(slantdepth))
        noxmaxshowers.append(scaled_m)
        # ax.axvline(slantdepth[e_spline], ls=":", color="tab:red", alpha=0.5)
        # ax.axvline(slantdepth[s_spline], ls=":", color="tab:red", alpha=0.5)

        ax[i].axvspan(
            slantdepth[s_spline], slantdepth[e_spline], facecolor="grey", alpha=0.5
        )

        ax[i].plot(
            slantdepth,
            m,
            color="k",
            # s=1,
            # label=r"{} ${{\rm mean,\:scaled}}$".format(decay_labels[i]),
            label=r"${{\rm decay\:grouping\:mean}}$",
            zorder=4,
            # alpha=0.6,
            # marker="s",
        )
        ax[i].plot(
            slantdepth,
            scaled_m,
            color="g",
            lw=3,
            # label=r"{} ${{\rm mean,\:scaled}}$".format(decay_labels[i]),
            label=r"${{\rm scaled\:mean}}$",
            # zorder=4,
            ls=":",
            alpha=0.8,
            # marker="s",
        )
        ax[i].plot(
            slantdepth,
            modified_gh(slantdepth, np.max(scaled_m), xmax, x0, p1, p2, p3),
            c="tab:grey",
            alpha=0.8,
            ls="--",
            label=r"${\rm GH\:fit\:to\:scaled\:mean}$",
        )

        ax[i].plot(
            slantdepth,
            theory,
            c="r",
            alpha=0.8,
            ls="--",
            label=r"${\rm GH\:fit,\:X_{max}\:fluctuated}$",
        )

        ax[i].plot(
            slantdepth,
            spline(slantdepth),
            label=r"${\rm synthetic}$",
            c="tab:blue",
            alpha=0.5,
            lw=5,
        )

        ax[i].set(
            xlim=(0, 3000),
            yscale="log",
            # xscale="log",
            ylim=(1000, 8e7),
            # ylabel=r"$N$",
            # xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
        )
        ax[i].text(
            0.30,
            0.30,
            decay_labels[i],
            transform=ax[i].transAxes,
            ha="center",
            va="top",
        )


ax[0].legend(
    # title=r"${\rm Decay\:Channel\:|\:mean\:X_{\rm max}(g\:cm^{-2})}$",
    # loc="lower center",
    fontsize=8,
    title_fontsize=8,
    # bbox_to_anchor=(0.5, 1),
    # ncol=2,
)
ax[1].text(
    0.50,
    0.10,
    r"${\rm interpolated}$",
    transform=ax[1].transAxes,
    ha="center",
    va="bottom",
    rotation=90,
    color="w",
)
fig.text(0.5, 0.09, r"${\rm slant\:depth\:(g\:cm^{-2})}$", ha="center")
fig.text(0.02, 0.5, r"$N$", va="center", rotation="vertical")

plt.savefig(
    "../../../../../gdrive_umd/Research/NASA/synthetic_reco.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

# %%

fig, ax = plt.subplots(1, 2, dpi=300, figsize=(6, 3), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0, hspace=0)
for i, s in enumerate(showers):
    ax[0].plot(slantdepth, s, alpha=0.5)
    ax[1].plot(slantdepth, noxmaxshowers[i], alpha=0.5)
    # ax.plot(slantdepth, spline(slantdepth), label="spline")
    # ax.scatter(
    #     depth_synth, shower_synth, color="tab:grey", s=3, label="synthetic shower"
    # )
ax[0].set(
    xlim=(0, 2000),
    yscale="log",
    ylim=(100, 8e7),
    ylabel=r"$N$",
    xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
)
ax[1].set(xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$")
# ax.legend()

# %% attempt to use 1d interpolation with bulk and tail with average value as anchor

# for i, m in enumerate(mean_perchan):
#     if i == 0:
#         # fmt: off
#         params, pcov = curve_fit(
#             modified_gh, depths[0, :], m, p0=[np.max(m), mean_xmaxs[i], 0, 70, 0, 0],
#             bounds=([0,0,-1e-6,-np.inf,-np.inf, -np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf, np.inf])
#         )
#         # fmt: on
#         nmax = params[0]
#         xmax = 600
#         x0 = params[2]
#         p1 = params[3]
#         p2 = params[4]
#         p3 = params[5]
#         theory = modified_gh(x, nmax, xmax, x0, p1, p2, p3)

#         ax.scatter(depths[0, :], m, s=0.4, color=cmap[i], label="mean shower")
#         ax.axvspan(min_xmaxs[i], max_xmaxs[i], alpha=0.2, color=cmap[i])
#         # ax.axvline(mean_xmaxs[i], ls=":", color=cmap[i])

#         s_gram = xmax * 1.75
#         e_gram = xmax * 2.75
#         end_spline = np.argmin(np.abs(x - e_gram))
#         ax.axvline(x[end_spline], ls=":", color=cmap[i])

#         depth_tail = x[end_spline:]
#         shwr_tail = m[end_spline:]

#         start_spline = np.argmin(np.abs(x - s_gram))
#         ax.axvline(x[start_spline], ls=":", color=cmap[i])

#         depth_bulk = x[:start_spline]
#         shwr_bulk = theory[:start_spline]

#         # shower_synth = np.concatenate(
#         #     (shwr_bulk, np.array([(shwr_bulk[-1] + shwr_tail[0]) / 2]), shwr_tail)
#         # )
#         # depth_synth = np.concatenate(
#         #     (depth_bulk, np.array([(s_gram + e_gram) / 2]), depth_tail)
#         # )
#         shower_synth = np.concatenate((shwr_bulk, shwr_tail))
#         depth_synth = np.concatenate((depth_bulk, depth_tail))

#         # spline = CubicSpline(depth_synth, shower_synth)
#         spline = interpolate.interp1d(depth_synth, shower_synth, kind=5)
#         # ax.scatter(
#         #     1750, (shwr_bulk[-1] + shwr_tail[0]) / 2, c="red", s=1, label="interp point"
#         # )

#         ax.plot(
#             depths[0, :],
#             modified_gh(depths[0, :], nmax, xmax, x0, p1, p2, p3),
#             c="k",
#             alpha=0.8,
#             label="fit",
#         )

#         ax.plot(x, spline(x), label="spline")
#         ax.scatter(depth_synth, shower_synth, color="tab:green", s=1)

#         # ax.plot(
#         #     x,
#         #     shwr_groups[i][:, 2:].T,
#         #     lw=1,
#         #     color=cmap[ci],
#         #     alpha=0.2,
#         #     zorder=1,
#         # )

#         # xmax_idx = np.argmax(m)
#         # past_bulk_y = m[xmax_idx:]

#         # past_bulk_x = x[xmax_idx:]
#         # past_bulk_theory = theory[xmax_idx:]
#         # tail_diff = (
#         #     np.abs(past_bulk_y - past_bulk_theory)
#         #     / (0.5 * (past_bulk_y + past_bulk_theory))
#         # ) * 100
#         # s = np.argwhere(tail_diff >= 5)[0]
#         # strt_spline = past_bulk_x[s]
#         # e = np.argwhere(tail_diff >= 10)[0]
#         # end_spline = past_bulk_x[e]

#         # shwr_bulk = theory[: int(xmax_idx + s)]
#         # depth_bulk = x[: int(xmax_idx + s)]

#         # shwr_tail = m[int(xmax_idx + e) :]
#         # depth_tail = x[int(xmax_idx + e) :]

#         # ax.axvline(strt_spline, ls=":", c="grey", label="cutoff")
#         # ax.axvline(end_spline, ls=":", c="grey")

#         # ax.catter(depth_tail, shwr_tail, color="tab:green")

#         # ax.plot(
#         #     depths[0, :],
#         #     modified_gh(
#         #         depths[0, :],
#         #         params[0],
#         #         700,
#         #         params[2],
#         #         params[3],
#         #         params[4],
#         #         params[5],
#         #     ),
#         #     c="green",
#         # )


# ax.set(
#     xlim=(0, 3000),
#     yscale="log",
#     ylim=(100, 8e7),
# )
# ax.legend()
