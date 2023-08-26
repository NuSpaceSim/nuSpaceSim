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

with as_file(files("nuspacesim.data.eas_reco") / "mean_shwr_bulk_gh_params.h5") as path:
    bulk_gh_data = h5py.File(path, "r")

    # each key has the order [nmax, xmax, x0, p1, p2, p3]
    leptonic_gh = np.array(bulk_gh_data["leptonic"])
    one_body_gh = np.array(bulk_gh_data["one_body_kpi"])
    with_pi0_gh = np.array(bulk_gh_data["with_pi0"])
    no_pi0_gh = np.array(bulk_gh_data["no_pi0"])


def synthetic_eas(mean, nmax_mult, xmax_mult, gh_params):
    nmax, xmax, x0, p1, p2, p3 = list(gh_params)

    synthetic_composite_eas = []
    for n, x in zip(nmax_mult, xmax_mult):
        vertically_shifted = mean * n  # we need this for the tail

        bulk_varied = modified_gh(slantdepth, nmax * n, xmax * x, x0, p1, p2, p3)
        # we peiece together the shifted bulk and scaled tail
        # the range for spline is dictated by the store Xmax gh fit for each channel
        s_gram = xmax * 1.8
        e_gram = xmax * 2.3
        s_spline = np.argmin(np.abs(slantdepth - s_gram))
        e_spline = np.argmin(np.abs(slantdepth - e_gram))

        depth_tail = slantdepth[e_spline:]
        shwr_tail = vertically_shifted[e_spline:]
        depth_bulk = slantdepth[:s_spline]
        shwr_bulk = bulk_varied[:s_spline]  # from theory

        # synthetic shower with gap that we piece together
        shower_synth = np.concatenate((shwr_bulk, shwr_tail))
        depth_synth = np.concatenate((depth_bulk, depth_tail))

        spline_synthetic = interpolate.interp1d(depth_synth, shower_synth, kind=5)

        synthetic_composite_eas.append(spline_synthetic(slantdepth))

    return synthetic_composite_eas


# %% generate composite via reduced EAS method

lep_eas = synthetic_eas(
    lep_m, nmax_mult=lep_nmax, xmax_mult=lep_xmax, gh_params=leptonic_gh
)
kpi_eas = synthetic_eas(
    kpi_m, nmax_mult=kpi_nmax, xmax_mult=kpi_xmax, gh_params=one_body_gh
)
pi0_eas = synthetic_eas(
    pi0_m, nmax_mult=pi0_nmax, xmax_mult=pi0_xmax, gh_params=with_pi0_gh
)
npi_eas = synthetic_eas(
    npi_m, nmax_mult=npi_nmax, xmax_mult=pi0_xmax, gh_params=no_pi0_gh
)

# %% generate them the long way via CONEX method
tup_folder = "/home/fabg/gdrive_umd/Research/NASA/Work/conex2r7_50-runs/"
elec_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_2033993834_11.root",
    )
)
pion_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_730702871_211.root",
    )
)
gamma_init = ReadConex(
    os.path.join(
        tup_folder,
        "log_17_eV_1000shwrs_5_degearthemergence_eposlhc_1722203790_22.root",
    )
)
elec_charged = elec_init.get_charged()
gamma_charged = gamma_init.get_charged()
pion_charged = pion_init.get_charged()
depths = elec_init.get_depths()
init = [elec_charged, gamma_charged, pion_charged]
pids = [11, 22, 211]

lepton_decay = [300001, 300002]
had_pionkaon_1bod = [200011, 210001]
# fmt: off
had_pi0 = [300111, 310101, 400211, 410111, 410101, 410201, 401111, 400111, 500131,
           500311, 501211, 501212, 510301, 510121, 510211, 510111, 510112, 600411,
           600231,
           ]
had_no_pi0 = [310001, 311001, 310011, 311002, 311003, 400031, 410021, 410011, 410012,
              410013, 410014, 501031, 501032, 510031, 600051,
              ]
# fmt: on
# initialize the EAS bin by bin generator
generator = ConexCompositeShowers(shower_comps=init, init_pid=pids, tau_table_start=0)

lep_con_eas = generator(
    n_comps=1000, channel=lepton_decay, return_table=False, no_subshwrs=True
)
kpi_con_eas = generator(
    n_comps=1000, channel=had_pionkaon_1bod, return_table=False, no_subshwrs=True
)
pi0_con_eas = generator(
    n_comps=1000, channel=had_pi0, return_table=False, no_subshwrs=True
)
npi_con_eas = generator(
    n_comps=1000, channel=had_no_pi0, return_table=False, no_subshwrs=True
)

# %%

from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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


cmap = plt.cm.get_cmap("inferno")(np.linspace(0, 1, 7))[1:]
fig, ax = plt.subplots(
    nrows=1, ncols=2, dpi=300, figsize=(7, 3.5), sharex=True, sharey=True
)
plt.subplots_adjust(wspace=0, hspace=0)

inset0 = ax[0].inset_axes([0.5, 1.15, 0.6, 0.6])
inset1 = ax[1].inset_axes([0.3, 1.15, 0.6, 0.6])

m_lep_conex, _ = mean_shower(lep_con_eas[:, 2:])
m_kpi_conex, _ = mean_shower(kpi_con_eas[:, 2:])
m_pi0_conex, _ = mean_shower(pi0_con_eas[:, 2:])
m_npi_conex, _ = mean_shower(npi_con_eas[:, 2:])

m_lep, _ = mean_shower(lep_eas)
m_kpi, _ = mean_shower(kpi_eas)
m_pi0, _ = mean_shower(pi0_eas)
m_npi, _ = mean_shower(npi_eas)


ax[0].plot(slantdepth.T, np.array(lep_con_eas[:, 2:]).T, color=cmap[0], alpha=0.2)
ax[0].plot(slantdepth.T, np.array(kpi_con_eas[:, 2:]).T, color=cmap[1], alpha=0.2)
ax[0].plot(slantdepth.T, np.array(pi0_con_eas[:, 2:]).T, color=cmap[2], alpha=0.2)
ax[0].plot(slantdepth.T, np.array(npi_con_eas[:, 2:]).T, color=cmap[3], alpha=0.2)
inset0.plot(slantdepth.T, np.array(lep_con_eas[:, 2:]).T, color=cmap[0], alpha=0.2)
inset0.plot(slantdepth.T, np.array(kpi_con_eas[:, 2:]).T, color=cmap[1], alpha=0.2)
inset0.plot(slantdepth.T, np.array(pi0_con_eas[:, 2:]).T, color=cmap[2], alpha=0.2)
inset0.plot(slantdepth.T, np.array(npi_con_eas[:, 2:]).T, color=cmap[3], alpha=0.2)

ax[1].plot(slantdepth.T, np.array(lep_eas).T, color=cmap[0], alpha=0.2)
ax[1].plot(slantdepth.T, np.array(kpi_eas).T, color=cmap[1], alpha=0.2)
ax[1].plot(slantdepth.T, np.array(pi0_eas).T, color=cmap[2], alpha=0.2)
ax[1].plot(slantdepth.T, np.array(npi_eas).T, color=cmap[3], alpha=0.2)

inset1.plot(slantdepth.T, np.array(lep_eas).T, color=cmap[0], alpha=0.2)
inset1.plot(slantdepth.T, np.array(kpi_eas).T, color=cmap[1], alpha=0.2)
inset1.plot(slantdepth.T, np.array(pi0_eas).T, color=cmap[2], alpha=0.2)
inset1.plot(slantdepth.T, np.array(npi_eas).T, color=cmap[3], alpha=0.2)

ax[0].set(
    # xlim=(0, 2300),
    yscale="log",
    ylim=(100, 1e8),
    ylabel=r"$N$",
    xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
)
ax[1].set(xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$")

inset0.set(
    xlim=(-100, 3000),
    yscale="log",
    ylim=(1000, 8e7),
    ylabel=r"$N$",
    xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
)
inset1.set(
    xlim=(-100, 3000),
    yscale="log",
    ylim=(1000, 8e7),
    ylabel=r"$N$",
    xlabel=r"${\rm slant\:depth\:(g\:cm^{-2})}$",
)

mark_inset(ax[0], inset0, loc1=4, loc2=4, zorder=6)
mark_inset(ax[1], inset1, loc1=4, loc2=4, zorder=6)

decay_labels = [
    r"${\rm leptonic\:decay}$",
    r"${\rm  1\:body\:K,\:\pi^{+/-}}$",
    r"${\rm  hadronic\:with\:\pi_0}$",
    r"${\rm  hadronic\:no\:\pi_0}$",
]

lep = mlines.Line2D([], [], color=cmap[0], ls="-", lw=2, label=decay_labels[0])
kpi = mlines.Line2D([], [], color=cmap[1], ls="-", lw=2, label=decay_labels[1])
pi0 = mlines.Line2D([], [], color=cmap[2], ls="-", lw=2, label=decay_labels[2])
npi = mlines.Line2D([], [], color=cmap[3], ls="-", lw=2, label=decay_labels[3])

leg = fig.legend(
    # title="$\mathrm{SFE} \: (f_{*})$",
    loc="lower center",
    handles=[lep, kpi, pi0, npi],
    bbox_to_anchor=(0.16, 0.90),
    ncol=1,
    edgecolor="k",
)
# !!! todo: reco mean / mean
ax[0].text(
    0.5,
    0.95,
    r"${\rm Composite\:EAS\:CONEX}$",
    transform=ax[0].transAxes,
    ha="center",
    va="top",
    zorder=7,
)
ax[1].text(
    0.5,
    0.95,
    r"${\rm Reconstruction}$",
    transform=ax[1].transAxes,
    ha="center",
    va="top",
    zorder=7,
)
inset0.grid(ls="--", which="both")
inset1.grid(ls="--", which="both")
ax[0].grid(ls="--", which="both")
ax[1].grid(ls="--", which="both")


plt.savefig(
    "../../../../../gdrive_umd/Research/NASA/conex_vs_pdf.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()
