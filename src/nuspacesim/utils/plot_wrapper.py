import configparser
from os.path import exists

from matplotlib import pyplot as plt


def read_plot_config(configfile):
    if configfile is None:
        return [], {}
    if not exists(configfile):
        return [], {}
    plot_list = []
    cfg = configparser.ConfigParser()
    cfg.read(configfile)
    plot_kwargs = {
        "title": cfg["General"]["title"],
        "rows": cfg["General"].getint("rows"),
        "columns": cfg["General"].getint("columns"),
        "figsize": eval(cfg["General"]["figsize"]),
        "save_as": cfg["General"]["save_as"],
        "pop_up": cfg["General"].getboolean("pop_up"),
        "save_to_file": cfg["General"].getboolean("save_to_file"),
        "default_color": cfg["General"].getint("default_color"),
        "default_colormap": cfg["General"].get("default_colormap"),
        # "output_path": cfg["General"]["output_path"],
    }
    for sec in cfg.sections()[1:]:
        for key in cfg[sec]:
            try:
                if cfg[sec].getboolean(key):
                    plot_list.append(key)
            except Exception as e:
                print(e, "Config file contains non-valid option")
    return plot_list, plot_kwargs


class PlotWrapper:
    """
    The PlotWrapper class produces figures that are uniformly formatted for nuspacesim as set up in sample_plot_config.ini
    """

    def __init__(
        self,
        to_plot=[],
        rows=None,
        cols=None,
        figsize=None,
        title=None,
        save_as=None,
        pop_up=None,
        save_to_file=None,
        default_color=None,
        default_colormap=None,
        filename=None,
        output_path=None,
        plotconfig=None,
    ):
        """
        initialize figure

        rows = number of rows of plots
        cols = number of cols of plots
        default is 1 for single plot, but can be changed to add subplots for making a multiplot
        """

        cfg_list, cfg_args = read_plot_config(plotconfig)

        cfg_args.setdefault("rows", 1)
        cfg_args.setdefault("cols", 1)
        cfg_args.setdefault("figsize", (8, 7))
        cfg_args.setdefault("title", "nuspacesim_run")
        cfg_args.setdefault("save_as", "pdf")
        cfg_args.setdefault("pop_up", True)
        cfg_args.setdefault("save_to_file", False)
        cfg_args.setdefault("default_color", 0)
        cfg_args.setdefault("default_colormap", "viridis")
        cfg_args.setdefault("filename", "NuSpaceSim")
        cfg_args.setdefault("output_path", ".")

        self.to_plot = list(set(list(to_plot) + cfg_list))
        self.rows = rows if rows else cfg_args["rows"]
        self.cols = cols if cols else cfg_args["cols"]
        self.figsize = figsize if figsize else cfg_args["figsize"]
        self.title = title if title else cfg_args["title"]
        self.save_as = save_as if save_as else cfg_args["save_as"]
        self.pop_up = pop_up if pop_up is not None else cfg_args["pop_up"]
        self.save_to_file = (
            save_to_file if save_to_file is not None else cfg_args["save_to_file"]
        )
        self.filename = filename if filename else cfg_args["filename"]
        self.output_path = output_path if output_path else cfg_args["output_path"]
        self.default_colormap = (
            default_colormap if default_colormap else cfg_args["default_colormap"]
        )
        self.default_color = [
            f"C{i+(default_color if default_color else cfg_args['default_color'])}"
            for i in range(3)
        ]

    def artist_params(self):

        return {
            "rows": self.rows,
            "cols": self.cols,
            "figsize": self.figsize,
            "title": self.title,
            "cmap": self.default_colormap,
            "color": self.default_color,
        }

    def init_fig(self, append_title=None):
        # initialize figure as subplots
        fig, ax = plt.subplots(nrows=self.rows, ncols=self.cols, figsize=self.figsize)
        fig.suptitle(f"{self.title}\n{append_title}")
        fig.tight_layout()
        return fig, ax

    def __call__(self, args, values, plot_fs, **kwargs):
        def do_plot(p):
            fig, ax = self.init_fig(p.__name__)
            title = p(args, values, fig, ax, **self.artist_params(), **kwargs)
            self.close(fig, title if title else p.__name__)

        # plot_fs_prs = map(lambda p: (p.__name__, p), plot_fs)
        to_plot_now = filter(lambda p: p.__name__ in self.to_plot, plot_fs)
        for p in to_plot_now:
            do_plot(p)

    def close(self, fig, plot_name):
        if self.save_to_file:
            fig.savefig(
                fname=f"{self.filename}_{plot_name}.{self.save_as}",
                bbox_inches="tight",
            )
        if self.pop_up:
            plt.show()
        plt.close(fig)
