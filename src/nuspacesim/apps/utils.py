import configparser

from ..config import Simulation

__all__ = ["parse_spectra_options", "parse_cloud_options", "read_plot_config"]


def parse_spectra_options(monospectrum, powerspectrum):
    if monospectrum and powerspectrum:
        raise RuntimeError("Only one of --monospectrum or --powerspectrum may be used.")
    if monospectrum:
        return Simulation.MonoSpectrum(log_nu_energy=monospectrum)
    if powerspectrum:
        return Simulation.PowerSpectrum(
            index=powerspectrum[0],
            lower_bound=powerspectrum[1],
            upper_bound=powerspectrum[2],
        )


def parse_cloud_options(nocloud, monocloud, pressuremapcloud):
    if sum([1 if x else 0 for x in (nocloud, monocloud, pressuremapcloud)]) > 1:
        raise RuntimeError(
            "Only one of --nocloud, --monocloud or --pressuremapcloud may be used."
        )
    if nocloud:
        return Simulation.NoCloud()
    if monocloud:
        return Simulation.MonoCloud(altitude=monocloud)
    if pressuremapcloud:
        return Simulation.PressureMapCloud(month=pressuremapcloud.month)


def read_plot_config(registry, plotall, plotconfig, plot):
    if plotall:
        return list(registry)
    elif plotconfig:
        plot_list = []
        cfg = configparser.ConfigParser()
        cfg.read(plotconfig)
        for sec in cfg.sections()[1:]:
            for key in cfg[sec]:
                try:
                    if cfg[sec].getboolean(key):
                        plot_list.append(key)
                except Exception as e:
                    print(e, "Config file contains non-valid option")
        return plot_list
    else:
        return plot
