# The Clear BSD License
#
# Copyright (c) 2021 Alexander Reustle and the NuSpaceSim Team
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
r"""The main proceedure for performaing a full simulation in nuspacesim.

*********************
NuSpaceSim Simulation
*********************

.. currentmodule:: nuspacesim


.. autosummary::
   :toctree:

   compute

"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from astropy.table import Table as AstropyTable
from numpy.typing import ArrayLike
from rich.console import Console

from . import results_table
from .config import NssConfig
from .simulation.atmosphere.clouds import CloudTopHeight
from .simulation.eas_optical.eas import EAS
from .simulation.eas_radio.radio import EASRadio
from .simulation.eas_radio.radio_antenna import calculate_snr
from .simulation.geometry.region_geometry import RegionGeom, RegionGeomToO
from .simulation.eas_optical.shower_properties import path_length_tau_atm
# from .simulation.geometry.too import *
from .simulation.spectra.spectra import Spectra
from .simulation.taus.taus import Taus
from .chasm_geom import detector_coordinates_and_tr_azimuth


__all__ = ["compute"]


def compute(
    config: NssConfig,
    verbose: bool = False,
    output_file: str | None = None,
    to_plot: list = [],
    write_stages=False,
) -> AstropyTable:
    r"""Simulate an upward going shower.

    The main proceedure for performaing a full simulation in nuspacesim.
    Given a valid NssConfig object, :func:`compute`, will perform the simulation as
    follows:

    #. Initialize the AstropyTable object.
    #. Initialize the appropritate :ref:`simulation modules<simulation>`.
    #. Compute array of valid beta angle trajectories: beta_tr from :class:`RegionGeom`.
    #. Compute tau interaction attributes componentwise for each element of beta_tr.

       #. tauBeta
       #. tauLorentz
       #. showerEnergy
       #. tauExitProb

    #. Compute Extensive Air Shower attributes componentwise

       #. Decay Altitude
       #. Photon Density
       #. Cherenkov Angle

    #. Compute the Monte Carlo integral for the resulting shower geometries.

    At each stage of the simulation, array results are stored as contiguous columns,
    and scalar results are stored as attributes, both in the :class:`AstropyTable`
    object.


    Parameters
    ----------
    config: NssConfig
        Configuration object.
    verbose: bool, optional
        Flag enabling verbose output.
    output_file: str, optional
        Name of file to write intermediate stages
    to_plot: list, optional
        Call the listed plotting functions as appropritate.
    write_stages: bool, optional
        Enable writing intermediate results to the output_file.

    Returns
    -------
    AstropyTable
        The Table of result values from each stage of the simulation.
    """

    console = Console(width=80, log_path=False)

    freqRange = (
        config.detector.radio.low_frequency,
        config.detector.radio.high_frequency,
    )

    def logv(*args):
        """optionally print descriptive messages."""
        if verbose:
            console.log(*args)

    if verbose:
        console.rule("[bold blue] NuSpaceSim")

    def mc_logv(mcint, mcintgeo, numEvPass, mcunc, method):
        logv(f"\t[blue]Monte Carlo Integral [/][magenta][{method}][/]:", mcint)
        logv(
            f"\t[blue]Monte Carlo Integral, GEO Only [/][magenta][{method}][/]:",
            mcintgeo,
        )
        logv(f"\t[blue]Number of Passing Events [/][magenta][{method}][/]:", numEvPass)
        logv(f"\t[blue]Stat uncert of MC Integral [/][magenta][{method}][/]:", mcunc)

    sim = results_table.init(config)
    geom = RegionGeom(config)
    cloud = CloudTopHeight(config)
    spec = Spectra(config)
    tau = Taus(config)
    eas = EAS(config)
    eas_radio = EASRadio(config)

    geom = (
        RegionGeomToO(config)
        if config.simulation.mode == "Target"
        else RegionGeom(config)
    )

    class StagedWriter:
        """Optionally write intermediate values to file"""

        def __call__(
            self,
            col_names: Iterable[str],
            columns: Iterable[ArrayLike],
            *args,
            **kwargs,
        ):
            sim.add_columns(columns, names=col_names, *args, **kwargs)
            if write_stages:
                sim.write(output_file, format="fits", overwrite=True)

        def add_meta(self, name: str, value: Any, comment: str):
            sim.meta[name] = (value, comment)
            if write_stages:
                sim.write(output_file, format="fits", overwrite=True)

    sw = StagedWriter()

    logv(f"Running NuSpaceSim with Energy Spectrum ({config.simulation.spectrum})")

    logv("Computing [green] Geometries.[/]")
    beta_tr, thetaArr, pathLenArr, *_ = geom(
        config.simulation.thrown_events, store=sw, plot=to_plot
    )
    thrown_color = "[blue]" if beta_tr.size else "[red]"
    logv(
        f"\t{thrown_color}Threw {config.simulation.thrown_events} neutrinos.\
        {beta_tr.size} were valid.[/]"
    )

    # Avoid Exceptions and return a (valid) empty sim object
    if beta_tr.size == 0:
        console.log(
            "\t[red] WARNING: No valid events thrown! Exiting early! Check geometry![/]"
        )
        return sim

    init_lat, init_long = geom.find_lat_long_along_traj(np.zeros_like(beta_tr))
    sw(
        ("init_lat", "init_lon"),
        (init_lat, init_long),
    )

    logv("Computing [green] Energy Spectra.[/]")

    log_e_nu, mc_spec_norm, spec_weights_sum = spec(
        beta_tr.shape[0], store=sw, plot=to_plot
    )

    logv("Computing [green] Taus.[/]")
    tauBeta, tauLorentz, tauEnergy, showerEnergy, tauExitProb = tau(
        beta_tr, log_e_nu, store=sw, plot=to_plot
    )

    logv("Computing [green] Decay Altitudes.[/]")
    altDec, lenDec = eas.altDec(beta_tr, tauBeta, tauLorentz, store=sw)

    """
    #CHASM STUFF
    #Need to figure out how to send azimuth, detcoords to cphotang function
    #Need to check that coord system of detcoords is the same as azimTrSubN (should be correct)
    zenith=np.pi/2-beta_tr

    from scipy.spatial.transform import Rotation as R

    #azimuth=


    thetaTrSubV=geom.thetas()
    phiTrSubV=geom.phis()
    costhetaNSubV=geom.valid_costhetaNSubV()
    det_azi=geom.valid_aziAngVSubN()
    det_elev=geom.valid_elevAngVSubN()
    thetaNSubV=np.arccos(costhetaNSubV)
    Pvecx=np.sin(thetaTrSubV)*np.cos(phiTrSubV)
    Pvecy=np.sin(thetaTrSubV)*np.sin(phiTrSubV)
    Pvecz=np.cos(thetaTrSubV)

    P=np.vstack((Pvecx,Pvecy,Pvecz)).T
    n_up_rot_axis = np.array([0, 1, 0])
    rot_vectors = n_up_rot_axis * thetaNSubV[:, np.newaxis]  # Shape (N, 3)
    theta_rotation = R.from_rotvec(rot_vectors)

    z_rot_axis = np.array([0, 0, 1])  # z-axis
    z_rot_vectors = z_rot_axis * (det_azi)[:, np.newaxis]  # Shape (N, 3)
    z_rotation = R.from_rotvec(z_rot_vectors)

    Pn = theta_rotation.apply(P)
    tr_n=z_rotation.apply(Pn)
    azimuth=np.arctan2(tr_n[:,1],tr_n[:,0])


    path_len=geom.pathLens()
    #path_len2=path_length_tau_atm(altitude,geom.valid_elevAngVSubN()) This is the same as path_len
    x = path_len * np.cos(det_elev) * np.cos(det_azi)  # East
    y = path_len * np.cos(det_elev) * np.sin(det_azi)  # North
    z = path_len * np.sin(det_elev)                    # Up
    detcoords=np.vstack((x,y,z)).T
    print(path_len,np.degrees(beta_tr))
    #print('Zen, zenith, beta,detazi,detzen',np.degrees(zen),np.degrees(zenith),np.degrees(beta_tr),np.degrees(det_azi),np.degrees(det_zen))
    #print(det_coord)
    for i in range(0, len(beta_tr)):
        sim = ch.ShowerSimulation()
        sim.add(ch.UpwardAxis(zenith[i], azimuth[i], curved=True))
        sim.add(ch.UserShower(np.array(X[i]), np.array(N[i])))
        sim.add(ch.SphericalCounters(detcoords, np.sqrt(1/np.pi)))
        sim.add(ch.Yield(270, 1000, N_bins=100))
        r_coordinates, total_propagation_times, photon_arrival_phZenith, photons_on_plane, wlbins, wlhist, dist_counter = run_chasm(sim, orig)
    """

    detcoords, azimuth = detector_coordinates_and_tr_azimuth(
        geom.thetas(), geom.phis(), geom.valid_costhetaNSubV(), geom.valid_aziAngVSubN(), geom.valid_elevAngVSubN(), geom.pathLens()

    )
    # if config.detector.method == "Optical" or config.detector.method == "Both":
    if config.detector.optical.enable:
        logv("Computing [green] EAS Optical Cherenkov light.[/]")

        numPEs, costhetaChEff = eas(
            beta_tr,
            altDec,
            showerEnergy,
            init_lat,
            init_long,
            detcoords,
            azimuth,
            cloudf=cloud,
            store=sw,
            plot=to_plot,
        )

        logv("Computing [green] Optical Monte Carlo Integral.[/]")
        mcint, mcintgeo, passEV, mcunc = geom.mcintegral(
            numPEs,
            costhetaChEff,
            tauExitProb,
            config.detector.optical.photo_electron_threshold,
            mc_spec_norm,
            spec_weights_sum,
            lenDec=lenDec,
            method="Optical",
            store=sw,
        )

        sw.add_meta("OMCINT", mcint, "Optical MonteCarlo Integral")
        sw.add_meta("OMCINTGO", mcintgeo, "Optical MonteCarlo Integral, GEO Only")
        sw.add_meta("ONEVPASS", passEV, "Optical Number of Passing Events")
        sw.add_meta("OMCINTUN", mcunc, "Stat unc of MonteCarlo Integral")

        mc_logv(mcint, mcintgeo, passEV, mcunc, "Optical")

    if config.detector.radio.enable:
        logv("Computing [green] EAS Radio signal.[/]")

        eFields = eas_radio(
            beta_tr, altDec, lenDec, thetaArr, pathLenArr, showerEnergy, store=sw
        )

        snrs = calculate_snr(
            eFields,
            freqRange,
            config.detector.initial_position.altitude,
            config.detector.radio.nantennas,
            config.detector.radio.gain,
        )

        logv("Computing [green] Radio Monte Carlo Integral.[/]")
        mcint, mcintgeo, passEV, mcunc = geom.mcintegral(
            snrs,
            np.cos(config.simulation.max_cherenkov_angle),
            tauExitProb,
            config.detector.radio.snr_threshold,
            mc_spec_norm,
            spec_weights_sum,
            lenDec=lenDec,
            method="Radio",
            store=sw,
        )

        sw.add_meta("RMCINT", mcint, "Radio MonteCarlo Integral")
        sw.add_meta("RMCINTGO", mcintgeo, "Radio MonteCarlo Integral, GEO Only")
        sw.add_meta("RNEVPASS", passEV, "Radio Number of Passing Events")
        sw.add_meta("RMCINTUN", mcunc, "Stat unc of MonteCarlo Integral")

        mc_logv(mcint, mcintgeo, passEV, mcunc, "Radio")

    logv("\n :sparkles: [cyan]Done[/] :sparkles:")

    return sim
