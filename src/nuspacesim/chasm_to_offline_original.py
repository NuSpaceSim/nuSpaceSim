#!@Python_EXECUTABLE@

import sys
import uproot
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from ROOT import TFile, TTree, TH1D, vector
try:
    import CHASM as ch
except ImportError as e:
    print("Module CHASM could not be imported. It is most likely not installed. \n \
          It can be installed through pip: python -m pip install CHASM-NuSpacesim. \n The detailed error message is: \n",e)
    sys.exit()


def get_vectors(zenith, azimuth, altitude, grid_angle, npoints):

    # get distance along axis corresponding to detector altitude
    r = sim.ingredients['axis'].h_to_axis_R_LOC(altitude*1.e3, zenith)
    grid_width = r*np.tan(np.radians(grid_angle))
    orig = [0, 0, r]
    rcoord_count = np.linspace(0, grid_width, npoints)

    # Radial random angle
    ang = 2*np.pi*np.random.rand(npoints)
    xall = rcoord_count*np.cos(ang)
    yall = rcoord_count*np.sin(ang)
    zall = np.full_like(xall, r)

    # Square grid
    # x = np.linspace(-grid_width, grid_width, n_side)
    # y = np.linspace(-grid_width, grid_width, n_side)
    # xx, yy = np.meshgrid(x,y)
    # zz = np.full_like(xx, r) #convert altitude to m

    # vecs = np.vstack((xx.flatten(),yy.flatten(),zz.flatten())).T
    vecs = np.vstack((xall.flatten(), yall.flatten(), zall.flatten())).T

    theta_rot_axis = np.array([0, 1, 0])
    theta_rotation = R.from_rotvec(theta_rot_axis * zenith)

    z_rot_axis = np.array([0, 0, 1])
    z_rotation = R.from_rotvec(z_rot_axis * np.pi/2)

    phi_rot_axis = np.array([0, 0, 1])
    phi_rot = R.from_rotvec(phi_rot_axis*azimuth)

    vecs = z_rotation.apply(vecs)
    vecs = theta_rotation.apply(vecs)

    tel_vecs = phi_rot.apply(vecs)
    orig = z_rotation.apply(orig)
    orig = theta_rotation.apply(orig)
    orig = phi_rot.apply(orig)
    return tel_vecs, orig


def run_chasm(sim, orig):
    sig = sim.run(mesh=False, att=True)
    del sim
    phot = np.array(sig.photons)
    phot_lambda = phot.sum(axis=2).sum(axis=0)
    phot_plane = phot.sum(axis=1)
    counters_vec = sig.counters.vectors
    dist_counter = np.sqrt(((counters_vec-orig)**2).sum(axis=-1))/1000  # km
    source_vec = sig.source_points
    to_orig = orig-source_vec
    travel_vec = sig.counters.travel_vectors(source_vec)
    dist_travel = np.sqrt((travel_vec**2).sum(axis=-1))
    plane_vec = travel_vec-to_orig
    plane_vec = plane_vec.reshape([-1, 3])
    r_coord = np.sqrt((plane_vec**2).sum(axis=-1))

    anglesin = 180/np.pi*np.arcsin(r_coord/dist_travel.flatten())
    times = np.array(sig.times)
    times = times-np.min(times)

    r_coordinates = (r_coord/1000).flatten()
    total_propagation_times = times.flatten()
    photon_arrival_phZenith = anglesin.flatten()
    photons_on_plane = phot_plane.flatten()
    wlbins = sig.wavelengths
    wlhist = phot_lambda.flatten()
    return r_coordinates, total_propagation_times, photon_arrival_phZenith, photons_on_plane, wlbins, wlhist, dist_counter


def weighted_percentile(data, percents, weights=None):
    ''' percents in units of 1%
            weights specifies the frequency (count) of data.
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 1.*w.cumsum()/w.sum()*100
    y = np.interp(percents, p, d)
    return y


def theta_D_from_theta_EE(det_alt, theta_EE):
    """
    det_alt : float
        detector altitude in km
    theta_EE : float
        earth emergence angle in rad

    Returns
    -------
    float
        detector viewing angle in degrees
    """
    theta_EE = theta_EE * (np.pi / 180)
    Re = 6371
    theta_D = (180 / np.pi) * np.arcsin((Re / (Re + det_alt))
                                        * np.sin(np.pi / 2 + theta_EE))

    return theta_D


def h_to_axis_R_LOC(h, theta) -> np.ndarray:
    earth_radius = 6371e3
    ground_level = 0
    """Return the length along the shower axis from the point of Earth
	emergence to the height above the surface specified

	Parameters:
	h: array of heights (m above ground level)
	theta: polar angle of shower axis (radians)

    returns: r (m) (same size as h), an array of distances along the shower
    axis_sp.
    """
    cos_EM = np.cos(np.pi-theta)
    R = earth_radius + ground_level
    r_CoE = h + R  # distance from the center of the earth to the specified height
    rs = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
    # rs -= rs[0]
    # rs[0] = 1.
    return rs


TH1D.AddDirectory(False)

parser = argparse.ArgumentParser()
parser.add_argument("-nss", "--nuspacesim", dest="nssfile",
                    default=False, help="NuSpaceSim Conex-like input")
parser.add_argument("-p", "--profile", dest="profile", default=False,
                    help="Text file with slant depth (g/cm2) and N charged particles")
parser.add_argument("-gh", "--GaisserHillas", dest="gh", type=float, nargs=4,
                    default=False, help="Gaisser Hilas parameters as in -gh X N X0 Lambda")
parser.add_argument("-o", "--outputfile", dest="outfile",
                    default='chasm_to_offline.root', help="Output file name")
parser.add_argument("-e", "--showerenergy", dest="senergy",
                    type=float, default=False, help="Shower Energy in log10(E/eV)")
parser.add_argument("-z", "--zenith", dest="theta", type=float, default=False,
                    help="Zenith emergence angle in degrees (89deg is very horizontal, 0 deg is upward vertical)")
parser.add_argument("-a", "--azimuth", dest="phi", type=float,
                    default=0, help="Azimuth angle in degrees")
parser.add_argument("-hf", "--heightfirst", dest="hfirst",
                    type=float, default=0, help="Height of first interaction in km")
parser.add_argument("-alt", "--altitude", dest="alt", type=float,
                    default=33, help="Detector altitude in km (default 33km)")
parser.add_argument("-n", "--npoints", dest="n", type=int, default=2500,
                    help="Number of sampled points on detection plane (default 2500)")

args = parser.parse_args()
if [args.nssfile, args.profile, args.gh].count(False) != 2:
    print('Use only one input flag. Use either a nuspacesim input, a text profile input, or a Gaisser Hillas input')
    exit()
if args.nssfile:
    try:
        with uproot.open(args.nssfile) as file:
            print('Reading NuSpaceSim file')

            tshower = file["Shower"]
            theader = file["Header"]

            zen90 = tshower["zenith"].array()
            phi = np.array(np.radians(tshower["azimuth"].array()))
            Hfirst = tshower["Hfirst"].array()/1000
            theta = np.array(np.radians(180-zen90))
            nEvents = np.size(zen90)
            lgenergy = tshower["lgE"].array()
            X = tshower["X"].array()
            N = tshower["N"].array()
    except:
        print('Could not read the file {}'.format(args.nssfile))
        exit()

else:
    nEvents = 1
    if args.senergy:
        if args.senergy < 30 and args.senergy > 9:
            lgenergy = [args.senergy]
        else:
            print('Shower energy must be in log10(E/eV)')
            exit()
    else:
        print('Introduce shower energy in log10(E/eV) using -e')
        exit()
    if args.theta:
        if args.theta <= 90 and args.theta >= 0:
            theta = [np.radians(args.theta)]
        else:
            print(
                'Shower must be upward. Zenith angle must be between 0(vertical) and 90(horizontal) degrees')
            exit()
    else:
        print('Introduce zenith angle between 0(vertical) and 90 (horizontal) degrees using -z')
        exit()
    phi = [args.phi]
    Hfirst = [args.hfirst]
    if args.profile:
        try:
            X, N = np.loadtxt(args.profile)
            X = [X]
            N = [N]
            print('Reading profile from text file')
        except:
            print('Could not read the file {}'.format(args.profile))
            exit()
    else:
        Xgh = args.gh[0]
        Ngh = args.gh[1]
        X0gh = args.gh[2]
        Lgh = args.gh[3]


# add shower axis
zenith = theta
azimuth = phi

# add grid of detectors
n_side = 30  # For square grid
npoints = args.n  # For radial
detector_grid_alt = args.alt  # km
grid_angle = 5  # degrees
file_name = args.outfile
filename = file_name
eng = np.zeros(1, dtype=float)
zen = np.zeros(1, dtype=float)
azi = np.zeros(1, dtype=float)
startdist = np.zeros(1, dtype=float)
file = TFile(filename, 'recreate')
tcher_ph = TTree("cherPhProp", "Cherenkov Photon Properties")
tshower = TTree("showerProp", "Shower Properties")
tshower.Branch("energy", eng, 'eng/D')
tshower.Branch("zenith", zen, 'zen/D')
tshower.Branch("azimuth", azi, 'azi/D')
tshower.Branch("startdist", startdist, 'startdist/D')


vec_time = vector("TH1D")()
vec_phZenith = vector("TH1D")()

histo_w = TH1D()
histo_r = TH1D()
histo_t_off = TH1D()
histo_phZenith_off = TH1D()

tcher_ph.Branch("wavelength", 'TH1D', histo_w)
tcher_ph.Branch("distance", 'TH1D', histo_r)
tcher_ph.Branch("time_offset", 'TH1D', histo_t_off)
tcher_ph.Branch("phZenith_offset", 'TH1D', histo_phZenith_off)
tcher_ph.Branch("time_dist", vec_time)
tcher_ph.Branch("phZenith_dist", vec_phZenith)


for i in range(nEvents):
    sim = ch.ShowerSimulation()
    sim.add(ch.UpwardAxis(zenith[i], azimuth[i], curved=True))
    tel_vecs, orig = get_vectors(
        zenith[i], azimuth[i], detector_grid_alt, grid_angle, npoints)
    if args.gh:
        print('Generating Gaisser Hillas shower')
        sim.add(ch.GHShower(Xgh, Ngh, X0gh, Lgh))
    else:
        sim.add(ch.UserShower(np.array(X[i]), np.array(N[i])))
    sim.add(ch.SphericalCounters(tel_vecs, np.sqrt(1/np.pi)))
    sim.add(ch.Yield(270, 1000, N_bins=100))
    print('Running CHASM simulation shower {}'.format(i))

    r_coordinates, total_propagation_times, photon_arrival_phZenith, photons_on_plane, wlbins, wlhist, dist_counter = run_chasm(
        sim, orig)

    energy = 10**lgenergy[i]
    azimdeg = np.degrees(azimuth[i])
    theta_D = theta_D_from_theta_EE(detector_grid_alt, 90-np.degrees(theta[i]))
    L_particle = Hfirst[i]  # meters

    photons_on_plane[photons_on_plane < 0] = 0

    bins, start, stop = 100, 0, 4  # For time
    time_bins = np.logspace(float(start), float(stop), num=int(bins))
    bins, start, stop = 100, -3, 1  # For Zenith
    angle_bins = np.logspace(float(start), float(stop), num=int(bins))
    spatial_bins = 100
    pct_cap = 100  # 30% recommended in easchersim, but 50 used
    r_percentiles = weighted_percentile(
        r_coordinates, pct_cap, weights=photons_on_plane)
    r_bounds = weighted_percentile(
        r_coordinates, np.array([1, 99.7]), weights=photons_on_plane)
    r_bins = np.linspace(np.min(r_coordinates)-1e-7,
                         r_percentiles, spatial_bins)
    ncounter_rbin = np.zeros(spatial_bins-1)

    time_bounds, zenith_bounds = [], []
    offset_propagation_times = total_propagation_times
    offset_arrival_phZenith = photon_arrival_phZenith
    for j in range(0, len(r_bins) - 1):
        # Find coordinates which are inside each spatial bin
        r_lower, r_upper = r_bins[j], r_bins[j + 1]
        inbin = (dist_counter <= r_upper) & (dist_counter > r_lower)
        ncounter_rbin[j] = int(np.sum(inbin))
        in_range = (r_coordinates >= r_lower) & (r_coordinates <= r_upper)
        in_range_time, in_range_zenith, in_range_photons = total_propagation_times[
            in_range], photon_arrival_phZenith[in_range], photons_on_plane[in_range]
        # Calculate the 1% and 99% weighted percentiles of the arrival time and arrival angle inside the bin
        time_percentiles = weighted_percentile(
            in_range_time, np.array([1, 99]), weights=in_range_photons)
        zenith_percentiles = weighted_percentile(
            in_range_zenith, np.array([1, 99]), weights=in_range_photons)

        time_bounds.append(time_percentiles)
        zenith_bounds.append(zenith_percentiles)

    time_offset = interpolate.interp1d(
        r_bins[:-1], np.array(time_bounds)[:, 0], kind='cubic', bounds_error=False, fill_value=0)
    phZenith_offset = interpolate.interp1d(
        r_bins[:-1], np.array(zenith_bounds)[:, 0], kind='cubic', bounds_error=False, fill_value=0)

    offset_propagation_times = total_propagation_times - \
        time_offset(r_coordinates).reshape(total_propagation_times.shape)
    onephotonmask = (photons_on_plane >= 1)
    offset_propagation_times -= np.min(offset_propagation_times[onephotonmask])

    offset_arrival_phZenith = offset_arrival_phZenith - \
        phZenith_offset(r_coordinates).reshape(photon_arrival_phZenith.shape)

    # Calculating the 1D histogram for the spatial counts and the 2D histogram for the arrival times and arrival angles
    spatial_counts = np.histogram(
        r_coordinates, r_bins, weights=photons_on_plane)[0]
    time_counts = np.histogram2d(r_coordinates, offset_propagation_times, bins=(
        r_bins, time_bins), weights=photons_on_plane)[0]
    angle_counts = np.histogram2d(r_coordinates, offset_arrival_phZenith, bins=(
        r_bins, angle_bins), weights=photons_on_plane)[0]
    spatial_counts = spatial_counts / ncounter_rbin

    # create branches for tshower

    eng[0] = energy
    zen[0] = theta_D
    azi[0] = azimdeg
    startdist[0] = L_particle

    tshower.Fill()

    # vec_time = vector("TH1D")()
    # vec_phZenith = vector("TH1D")()

    # define histograms

    num_wl_bin = len(wlbins) - 1
    num_dist_bin = len(r_bins) - 1
    num_time_bin = len(time_bins) - 1
    num_ang_bin = len(angle_bins) - 1

    # histo_w = TH1D()
    # histo_r = TH1D()
    # histo_t_off = TH1D()
    # histo_phZenith_off = TH1D()

    # histo_t=[TH1D()for i in range(num_dist_bin)]
    # histo_phZenith=[TH1D()for i in range(num_dist_bin)]

    # define histograms
    histo_w = TH1D("wl", "wavelength", num_wl_bin, wlbins)
    histo_r = TH1D("r", "distance", num_dist_bin, r_bins)
    histo_t_off = TH1D("t_off", "time offset",
                       num_dist_bin, r_bins)
    histo_phZenith_off = TH1D("ang_off", "angle offset",
                         num_dist_bin, r_bins)
    histo_t = [TH1D("t_dist_" + str(i), "time_dist_" + str(i),
                    num_time_bin, time_bins) for i in range(num_dist_bin)]
    histo_phZenith = [TH1D("ang_dist_" + str(i), "angle_dist_" + str(i),
                      num_ang_bin, angle_bins) for i in range(num_dist_bin)]
    # fill histograms
    for wl_bin, counts in enumerate(wlhist):
        histo_w.SetBinContent(wl_bin + 1, counts)
    for r_bin, counts in enumerate(spatial_counts):
        histo_r.SetBinContent(r_bin + 1, counts)
        histo_t_off.SetBinContent(
            r_bin + 1, np.array(time_bounds)[:, 0][r_bin])
        histo_phZenith_off.SetBinContent(
            r_bin + 1, np.array(zenith_bounds)[:, 0][r_bin])
        for t_bin, counts_t in enumerate(time_counts[r_bin]):
            histo_t[r_bin].SetBinContent(t_bin + 1, counts_t)
        for ang_bin, counts_ang in enumerate(angle_counts[r_bin]):
            histo_phZenith[r_bin].SetBinContent(ang_bin + 1, counts_ang)

    # set branch for histograms
    tcher_ph.SetBranchAddress("wavelength", histo_w)
    tcher_ph.SetBranchAddress("distance", histo_r)
    tcher_ph.SetBranchAddress("phZenith_offset", histo_phZenith_off)
    tcher_ph.SetBranchAddress("time_offset", histo_t_off)
    vec_time.assign(histo_t)
    vec_phZenith.assign(histo_phZenith)

    tcher_ph.Fill()
    # , vec_phZenith, vec_time
    del histo_w, histo_r, histo_t, histo_phZenith, histo_t_off, histo_phZenith_off
tshower.Write("", TFile.kOverwrite)
tcher_ph.Write("", TFile.kOverwrite)
print('Writing output to {}'.format(filename))


file.Close()
