import numpy as np
from matplotlib import pyplot as plt


class PlotWrapper:
    """
    The PlotWrapper class produces figures that are uniformly formatted for nuspacesim as set up in sample_plot_config.ini
    """

    def __init__(
        self,
        plot_kwargs={
            "save_as": "pdf",
            "pop_up": True,
            "save_to_file": True,
            "default_color": 0,
            "default_colormap": "viridis",
            "filename": "nuspacesim_run",
        },
        rows=1,
        cols=1,
        size=(8, 7),
        title=None,
    ):
        """
        initialize figure
        rows = number of rows of plots
        cols = number of cols of plots
        default is 1 for single plot, but can be changed to add subplots for making a multiplot
        """
        self.params = {
            "save_to_file": plot_kwargs["save_to_file"],
            "save_as": plot_kwargs["save_as"],
            "pop_up": plot_kwargs["pop_up"],
            "default_colors": [
                "C{}".format(plot_kwargs["default_color"]),
                "C{}".format(plot_kwargs["default_color"] + 1),
                "C{}".format(plot_kwargs["default_color"] + 2),
            ],
            "default_colormap": plot_kwargs["default_colormap"],
            "filename": plot_kwargs["filename"],
        }
        # initialize figure as subplots
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, figsize=size)
        self.fig.suptitle(title)

    def make_labels(
        self,
        ax,
        xlabel,
        ylabel,
        clabel=None,
        im=None,
        logx=False,
        logy=False,
        logx_scale=False,
        logy_scale=False,
    ):

        xl = "$\\log_{10}$" + f"({xlabel})" if logx else xlabel
        yl = "$\\log_{10}$" + f"({ylabel})" if logy else ylabel
        xs = "log" if logx_scale else "linear"
        ys = "log" if logy_scale else "linear"

        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_xscale(xs)
        ax.set_yscale(ys)
        if clabel is not None:
            cbar = self.fig.colorbar(im, ax=ax, pad=0.0)
            cbar.set_label(clabel)

    def get_profile(self, x, y, nbins, useStd=True):

        if sum(np.isnan(y)) > 0:
            x = x[~np.isnan(y)]
            y = y[~np.isnan(y)]
        n, _ = np.histogram(x, bins=nbins)
        sy, _ = np.histogram(x, bins=nbins, weights=y)
        sy2, _ = np.histogram(x, bins=nbins, weights=y * y)
        mean = sy / n
        std = np.sqrt(sy2 / n - mean * mean)
        if not useStd:
            std /= np.sqrt(n)
        bincenter = (_[1:] + _[:-1]) / 2
        binwidth = bincenter - _[1:]

        return bincenter, mean, std, binwidth

    def hexbin(
        self,
        ax,
        x,
        y,
        gs=25,
        logx=False,
        logy=False,
        logx_scale=False,
        logy_scale=False,
    ):

        xf = np.log10 if logx else lambda q: q
        yf = np.log10 if logy else lambda q: q

        xs = "log" if logx_scale else "linear"
        ys = "log" if logy_scale else "linear"

        xmask = x > 0 if logx else np.full(x.shape, True)
        ymask = y > 0 if logy else np.full(y.shape, True)
        m = xmask & ymask

        im = ax.hexbin(
            x=xf(x[m]),
            y=yf(y[m]),
            gridsize=gs,
            mincnt=1,
            xscale=xs,
            yscale=ys,
            cmap=self.params["default_colormap"],
            edgecolors="none",
        )
        return im

    def close(self, plot_name, save_to_file=True, pop_up=True):
        self.fig.tight_layout()
        if save_to_file:
            self.fig.savefig(
                fname=self.params["filename"]
                + "_"
                + plot_name
                + "."
                + self.params["save_as"],
                bbox_inches="tight",
            )
        if pop_up:
            plt.show()
        plt.close(self.fig)
