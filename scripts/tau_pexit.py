import nuspacesim as nss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.interpolate import interp1d

from importlib.resources import as_file, files


def tau_exit_compare_10_75():
    # grid of pexit table
    with as_file(
        files("nuspacesim.data.RenoNu2TauTables") / "reno_nu2tau_pexit.hdf5"
    ) as reno_file:
        renogrid = nss.utils.grid.NssGrid.read(reno_file, path="/", format="hdf5")
        reno_sgrid = renogrid[-1, :]
    with as_file(
        files("nuspacesim.data.nupyprop_tables") / "nu2tau_pexit.hdf5"
    ) as file:
        g = nss.utils.grid.NssGrid.read(file, path="pexit_regen", format="hdf5")
        sg = nss.utils.interp.grid_slice_interp(
            g, 10.75, g.axis_names.index("log_e_nu")
        )

    reno_pexit_interp = interp1d(reno_sgrid.axes[0], reno_sgrid.data)
    log_pexit_interp = interp1d(sg.axes[0], np.log10(sg.data))

    betas = np.radians(np.linspace(1, 35, 1000))

    # fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    fig, (ax1, ax2) = plt.subplots(
        2, 2, sharex=True, figsize=(10, 8), constrained_layout=True
    )
    fig.suptitle("Tau P_exit for log_e_nu = 10.75")

    ax1[0].plot(
        np.degrees(betas), 10 ** log_pexit_interp(betas), c="r", label="nupyprop 10.75"
    )
    ax1[0].plot(
        np.degrees(betas),
        10 ** reno_pexit_interp(betas),
        c="b",
        label="10**Reno-efix ~10.75",
    )
    ax1[0].scatter(
        np.degrees(sg.axes[0]),
        10 ** np.log10(sg.data),
        30,
        "r",
        "o",
        label="nupyprop table values",
    )
    ax1[0].scatter(
        np.degrees(reno_sgrid.axes[0]),
        10 ** reno_sgrid.data,
        30,
        "b",
        "+",
        label="10**Reno-efix table values",
    )
    ax1[0].grid(True)
    ax1[0].legend()
    ax1[0].set_title("P_exit Comparison")
    ax1[0].set_xlabel("Beta_tr Degrees")
    ax1[0].set_ylabel("exit probability")

    ax2[0].plot(
        np.degrees(betas), log_pexit_interp(betas), c="r", label="nupyprop 10.75"
    )
    ax2[0].plot(
        np.degrees(betas), reno_pexit_interp(betas), c="b", label="Reno-efix ~10.75"
    )
    ax2[0].scatter(
        np.degrees(sg.axes[0]),
        np.log10(sg.data),
        30,
        "r",
        "o",
        label="log(Nupyprop 10.75) Table Values",
    )
    ax2[0].scatter(
        np.degrees(reno_sgrid.axes[0]),
        reno_sgrid.data,
        30,
        "b",
        "+",
        label="Reno-efix ~10.75 Table Values",
    )
    ax2[0].grid(True)
    ax2[0].legend()
    ax2[0].set_title("Log Comparison")
    ax2[0].set_xlabel("Beta_tr Degrees")
    ax2[0].set_ylabel("Log exit probability")

    ax1[1].plot(
        np.degrees(betas),
        np.abs(10 ** log_pexit_interp(betas) - 10 ** reno_pexit_interp(betas)),
        c="g",
        label="nupyprop - 10**Reno",
    )
    ax1[1].grid(True)
    ax1[1].legend()
    ax1[1].set_title("Difference")
    ax1[1].set_xlabel("Beta_tr Degrees")
    ax1[1].set_ylabel("NuPyProp - Reno")

    ax2[1].plot(
        np.degrees(betas),
        np.abs(10 ** log_pexit_interp(betas) / 10 ** reno_pexit_interp(betas)),
        c="g",
        label="nupyprop / 10**Reno",
    )
    ax2[1].grid(True)
    ax2[1].legend()
    ax2[1].set_title("Ratio")
    ax2[1].set_xlabel("Beta_tr Degrees")
    ax2[1].set_ylabel("NuPyProp / Reno")

    plt.show()


def tau_pexit_compare():
    with as_file(
        files("nuspacesim.data.RenoNu2TauTables") / "reno_nu2tau_pexit.hdf5"
    ) as reno_file:
        renogrid = nss.utils.grid.NssGrid.read(reno_file, path="/", format="hdf5")
    with as_file(
        files("nuspacesim.data.nupyprop_tables") / "nu2tau_pexit.hdf5"
    ) as file:
        g = nss.utils.grid.NssGrid.read(file, path="pexit_regen", format="hdf5")

    betas = np.radians(np.linspace(1, 25, 1000))

    reno = np.empty((16, 1000))
    nupy = np.empty((16, 1000))

    for i, log_e_nu in enumerate(renogrid.axes[0]):

        lenu = np.around(log_e_nu, 2)

        reno_sgrid = renogrid[i, :]
        reno_pexit_interp = interp1d(reno_sgrid.axes[0], reno_sgrid.data)
        reno[i, :] = 10 ** reno_pexit_interp(betas)
        sg = nss.utils.interp.grid_slice_interp(g, lenu, g.axis_names.index("log_e_nu"))
        pexit_interp = interp1d(sg.axes[0], np.log10(sg.data))
        nupy[i, :] = 10 ** pexit_interp(betas)

    logenus = np.around(renogrid.axes[0], 2)

    return np.degrees(betas), logenus, nupy, reno


def spinning_surface_plot(betas, logenus, nupy, reno):

    X, Y = np.meshgrid(betas, logenus)

    fig = plt.figure(figsize=(16, 15), constrained_layout=True)

    def figapp(p, ax, title, zlabel):
        ax.set_title(title)
        ax.set_xlabel("Beta Degrees")
        ax.set_ylabel("Log E nu")
        ax.set_zlabel(zlabel)
        fig.colorbar(p, ax=ax, shrink=0.75, aspect=10)

    def init():
        ax = fig.add_subplot(221, projection="3d")
        p = ax.plot_surface(X, Y, np.log10(nupy), cmap="jet")
        figapp(p, ax, "Pexit log(NuPyProp)", "p_exit")

        ax = fig.add_subplot(222, projection="3d")
        p = ax.plot_surface(X, Y, np.log10(reno), cmap="jet")
        figapp(p, ax, "Pexit Reno V2", "p_exit")

        ax = fig.add_subplot(223, projection="3d")
        p = ax.plot_surface(X, Y, nupy / reno, cmap="jet")
        figapp(p, ax, "Pexit Ratio (log(NuPyProp) / Reno v2)", "p_exit ratio")

        ax = fig.add_subplot(224, projection="3d")
        p = ax.plot_surface(X, Y, np.log10(nupy / reno), cmap="jet")
        figapp(p, ax, "Log Pexit Ratio. (log(NuPyProp) / Reno v2)", "log(p_exit ratio)")

        return (fig,)

    def animate(i):
        for x in fig.get_axes()[0:-1:2]:
            x.view_init(elev=30.0, azim=i)
        return (fig,)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=360, interval=100, blit=True, repeat=True
    )
    anim.save(
        "tau_pexit_compare_nupyprop_reno.mp4", fps=10, extra_args=["-vcodec", "libx264"]
    )


def diff_3d(betas, logenus, nupy, reno):

    X, Y = np.meshgrid(betas, logenus)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)

    def figapp(p, ax, title, zlabel):
        ax.set_title(title)
        ax.set_xlabel("Beta Degrees")
        ax.set_ylabel("Log E nu")
        ax.set_zlabel(zlabel)
        fig.colorbar(p, ax=ax, shrink=0.75, aspect=10)

    def init():
        ax = fig.add_subplot(111, projection="3d")
        p = ax.plot_surface(X, Y, nupy - reno, cmap="jet")
        figapp(p, ax, "Pexit log(NuPyProp) - Pexit Reno V2", "p_exit difference")

        return (fig,)

    fig = init()
    plt.show()


if __name__ == "__main__":
    tau_exit_compare_10_75()
    # v = tau_pexit_compare()
    # spinning_surface_plot(*v)
