import configparser

from ..config import Detector, Simulation

__all__ = [
    "parse_diffuse_integ_method_options",
    "parse_target_integ_method_options",
    "parse_sun_moon_cuts",
    "parse_spectra_options",
    "parse_cloud_options",
    "parse_source_info",
    "read_plot_config",
]


def parse_diffuse_integ_method_options(montecarlo, targetapprox, cubature):
    if sum([1 if x else 0 for x in (montecarlo, targetapprox, cubature)]) > 1:
        raise RuntimeError(
            "Only one of --montecarlo, --targetapprox or --cubature may be used."
        )
    if targetapprox:
        raise RuntimeError("--targetapprox not valid for diffuse calculations.")
    if montecarlo:
        return Simulation.MonteCarlo(num_events_per_time_bin=montecarlo)
    if cubature:
        return Simulation.Cubature(
            deg=cubature[0],
            num_events_per_node=cubature[1],
        )


def parse_target_integ_method_options(montecarlo, targetapprox, cubature):
    if sum([1 if x else 0 for x in (montecarlo, targetapprox, cubature)]) > 1:
        raise RuntimeError(
            "Only one of --montecarlo, --targetapprox or --cubature may be used."
        )
    if montecarlo:
        return Simulation.MonteCarlo(num_events_per_time_bin=montecarlo)
    if targetapprox:
        return Simulation.TargetApprox()
    if cubature:
        return Simulation.Cubature(
            deg=cubature[0],
            num_events_per_node=cubature[1],
        )


def parse_sun_moon_cuts(sunmooncutparameters):
    return Detector.SunMoonCuts(
        sun_alt_cut=sunmooncutparameters[0],
        moon_alt_cut=sunmooncutparameters[1],
        moon_min_phase_angle_cut=sunmooncutparameters[2],
    )


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


def parse_source_info(sourceinfo):
    return Simulation.SinglePointSource(
        source_RA=sourceinfo[0],
        source_DEC=sourceinfo[1],
        source_date=sourceinfo[2],
        source_date_format=sourceinfo[3],
        source_obst=sourceinfo[4],
    )


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
