import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import h5py
from nuspacesim.simulation.eas_composite.comp_eas_utils import decay_channel_filter
from nuspacesim.simulation.eas_composite.comp_eas_conex import ConexCompositeShowers
from matplotlib.lines import Line2D
from scipy.signal import argrelextrema
from scipy.stats import poisson
from scipy.stats import skewnorm
import scipy.special as sse
from scipy import stats
from scipy.stats import exponnorm
import matplotlib
from scipy import interpolate
from scipy.interpolate import splrep, BSpline

try:
    from importlib.resources import as_file, files
except ImportError:
    from importlib_resources import as_file, files


def modified_gh(x, n_max, x_max, x_0, p1, p2, p3):

    particles = (
        n_max
        * np.nan_to_num(
            ((x - x_0) / (x_max - x_0))
            ** ((x_max - x_0) / (p1 + p2 * x + p3 * (x ** 2)))
        )
    ) * (np.exp((x_max - x) / (p1 + p2 * x + p3 * (x ** 2))))

    return particles


def pwr_law(x, a, b):
    # power law
    return a * x ** b


with as_file(
    files("nuspacesim.data.eas_reco.rms_params") / "nmax_rms_params.h5"
) as path:
    data = h5py.File(path, "r")

    nmax_leptonic = np.array(data["leptonic"])
    nmax_one_body_kpi = np.array(data["one_body_kpi"])
    nmax_with_pi0 = np.array(data["with_pi0"])
    nmax_no_pi0 = np.array(data["no_pi0"])

    mean_leptonic = np.array(data["mean_leptonic"])
    mean_one_body_kpi = np.array(data["mean_one_body_kpi"])
    mean_with_pi0 = np.array(data["mean_with_pi0"])
    mean_no_pi0 = np.array(data["mean_no_pi0"])

    slantdepth = np.array(data["slant_depth"])

with as_file(
    files("nuspacesim.data.eas_reco.rms_params") / "xmax_rms_params.h5"
) as path:
    data = h5py.File(path, "r")

    xmax_leptonic = np.array(data["leptonic"])
    xmax_one_body_kpi = np.array(data["one_body_kpi"])
    xmax_with_pi0 = np.array(data["with_pi0"])
    xmax_no_pi0 = np.array(data["no_pi0"])

    # mean_leptonic = np.array(data["mean_leptonic"])
    # mean_one_body_kpi = np.array(data["mean_one_body_kpi"])
    # mean_with_pi0 = np.array(data["mean_with_pi0"])
    # mean_no_pi0 = np.array(data["mean_no_pi0"])

means = [mean_leptonic, mean_one_body_kpi, mean_with_pi0, mean_no_pi0]
nmax_params = [nmax_leptonic, nmax_one_body_kpi, nmax_with_pi0, nmax_no_pi0]
xmax_params = [xmax_leptonic, xmax_one_body_kpi, xmax_with_pi0, xmax_no_pi0]

decay_labels = [
    r"${\rm leptonic\:decay}$",
    r"${\rm  1\:body\:K,\:\pi^{+/-}}$",
    r"${\rm  hadronic\:with\:\pi_0}$",
    r"${\rm  hadronic\:no\:\pi_0}$",
]

showers = []
noxmaxshowers = []
for i, m in enumerate(means):
    if i != 0:
        continue
    print(decay_labels[i])
    fig, ax = plt.subplots(1, 1, dpi=300, figsize=(4, 3.5), sharex=True, sharey=True)
    # fit the mean
    for d in range(100):
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

        # scale the Nmax of the mean
        nlamb = nmax_params[i][0]
        nsig = nmax_params[i][1]
        nmu = nmax_params[i][2]
        nright_trunc = nmax_params[i][3]
        nmult = []
        while len(nmult) != 1:
            r = exponnorm.rvs(1 / (nlamb * nsig), loc=nmu, scale=nsig)
            # print(r)
            if (r > 0) and (r <= nright_trunc):
                nmult.append(r * m)

        scaled_m = nmult[0]

        # scale the Xmax of the mean
        xlamb = xmax_params[i][0]
        xsig = xmax_params[i][1]
        xmu = xmax_params[i][2]
        xleft_trunc = xmax_params[i][3]
        xright_trunc = xmax_params[i][4]
        print("sampling xmax")
        xmult = []
        while len(xmult) != 1:
            r = exponnorm.rvs(1 / (xlamb * xsig), loc=xmu, scale=xsig)
            # print(r)
            if (r >= xleft_trunc) and (r <= xright_trunc):
                xmult.append(r)

        shifted_xmax = xmult[0] * xmax

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
        ax.axvline(slantdepth[e_spline], ls=":", color="tab:red", alpha=0.5)
        ax.axvline(slantdepth[s_spline], ls=":", color="tab:red", alpha=0.5)

        ax.scatter(
            slantdepth,
            scaled_m,
            color="tab:grey",
            s=1,
            label=r"{} ${{\rm mean,\:scaled}}$".format(decay_labels[i]),
            zorder=4,
            alpha=0.6,
            marker="s",
        )
        ax.plot(
            slantdepth,
            modified_gh(slantdepth, np.max(scaled_m), xmax, x0, p1, p2, p3),
            c="tab:grey",
            alpha=0.8,
            ls="--",
            label=r"${\rm GH\:fit\:to\:scaled\:mean}$",
        )

        # ax.scatter(
        #     depth_synth,
        #     shower_synth,
        #     color="tab:grey",
        #     s=1,
        #     label=r"{} ${{\rm mean\:scaled}}$".format(decay_labels[i]),
        #     zorder=4,
        # )

        ax.plot(
            slantdepth,
            theory,
            c="k",
            alpha=0.8,
            ls="--",
            label=r"${\rm GH\:fit,\:X_{max}\:fluctuated}$",
        )

        ax.scatter(
            slantdepth, spline(slantdepth), label=r"${\rm spline}$", c="tab:red", s=1
        )

        ax.set(
            xlim=(0, 3000),
            # yscale="log",
            # ylim=(100, 8e7),
            ylabel=r"$N$",
            xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
        )
        ax.legend(
            # title=r"${\rm Decay\:Channel\:|\:mean\:X_{\rm max}(g\:cm^{-2})}$",
            loc="lower center",
            fontsize=8,
            title_fontsize=8,
            bbox_to_anchor=(0.5, 1),
            ncol=2,
        )
#%%

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
    xlim=(0, 2700),
    # yscale="log",
    # ylim=(100, 8e7),
    ylabel=r"$N$",
    xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
)
ax[1].set(xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$")
# ax.legend()


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
