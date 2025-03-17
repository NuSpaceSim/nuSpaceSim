#!@Python_EXECUTABLE@

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from astropy.table import QTable, Table, Column
from astropy import units as u

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
    ang_step=np.arange(0,355,10)
    nperiods=(npoints // ang_step.size) + 1
    # Radial random angle
    ang = np.tile(np.radians(ang_step),nperiods)[:npoints]
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
    source_vec = sig.source_points    #Vectors to axis points
    costheta=sig.cos_theta # This returns the cosine of the angle between the z-axis and the vector from the axis to the counter
    times = np.array(sig.times)
    times = times-np.min(times)
    phot_plane[phot_plane < 0] = 0

    total_propagation_times = times
    photons_on_plane = phot_plane
    wlbins = sig.wavelengths
    wlhist = phot_lambda
    return source_vec, total_propagation_times, costheta, photons_on_plane, wlbins, wlhist

def signal_to_astropy(sig: ShowerSignal) -> QTable:
    '''This function outputs the data in a shower signal object to an astropy table.
    '''
    column_list = []
    column_list.append(Column(sig.source_points, name='source points',unit=u.m))
    column_list.append(Column(sig.charged_particles, name='charged particles',unit=u.ct))
    column_list.append(Column(sig.depths, name='depths',unit=u.g/u.cm**2))
    for i in range(sig.photons.shape[0]):
        column_list.append(Column(sig.photons[i].T, name=f'counter {i} photons',unit=u.ct))
        column_list.append(Column(sig.times[i].T, name=f'counter {i} arrival times',unit=u.nanosecond))
        column_list.append(Column(sig.cos_theta[i].T, name=f'counter {i} cos zenith',unit=u.rad))
    return QTable(column_list)

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

zen90 = tshower["zenith"].array()
phi = np.array(np.radians(tshower["azimuth"].array()))
Hfirst = tshower["Hfirst"].array()/1000
Xfirst = tshower["Xfirst"].array()
theta = np.array(np.radians(180-zen90))
nEvents = np.size(zen90)
lgenergy = tshower["lgE"].array()
X = tshower["X"].array()
N = tshower["N"].array()



# add shower axis
zenith = theta
azimuth = phi

# add grid of detectors
n_side = 30  # For square grid
npoints = args.n  # For radial
detector_grid_alt = args.alt  # km
grid_angle = 5  # degrees
filename = args.outfile
eng = np.zeros(1, dtype=float)
zen = np.zeros(1, dtype=float)
azi = np.zeros(1, dtype=float)
startdist = np.zeros(1, dtype=float)
X0 = np.zeros(1, dtype=float)



for i in range(nEvents):
    sim = ch.ShowerSimulation()
    sim.add(ch.UpwardAxis(zenith[i], azimuth[i], curved=True))
    tel_vecs, orig = get_vectors(
        zenith[i], azimuth[i], detector_grid_alt, grid_angle, npoints)

    sim.add(ch.UserShower(np.array(X[i]), np.array(N[i])))
    sim.add(ch.SphericalCounters(tel_vecs, np.sqrt(1/np.pi)))
    sim.add(ch.Yield(270, 1000, N_bins=100))
    print('Running CHASM simulation shower {}'.format(i))

    plane_vec, total_propagation_times, photon_arrival_phZenith, photons_on_plane, wlbins, wlhist, dist_counter = run_chasm(
        sim, orig)
    r_coord = np.sqrt((plane_vec**2).sum(axis=-1))
    r_coordinates = (r_coord/1000).flatten()

    #rotate plane_vec to have it in z=0

    theta_rot_axis = np.array([0,1,0])
    theta_rotation = R.from_rotvec(theta_rot_axis * -zenith[i])

    z_rot_axis = np.array([0,0,1])
    z_rotation = R.from_rotvec(z_rot_axis * -np.pi/2)

    phi_rot_axis=np.array([0,0,1])
    phi_rot=R.from_rotvec(phi_rot_axis*-azimuth[i])

    plane_vec_rot=phi_rot.apply(plane_vec)
    plane_vec_rot=theta_rotation.apply(plane_vec_rot)
    plane_vec_rot=z_rotation.apply(plane_vec_rot)

    x_plane=plane_vec_rot[:,0]
    y_plane=plane_vec_rot[:,1]

    azimuth_angles = 0*(180/np.pi)*np.arctan2(y_plane, x_plane)
    azi_below = azimuth_angles<-180
    azi_above = azimuth_angles>180
    azimuth_angles[azi_below] = 360+azimuth_angles[azi_below]
    azimuth_angles[azi_above] = -360+azimuth_angles[azi_above]

    energy = 10**lgenergy[i]
    azimdeg = np.degrees(azimuth[i])
    theta_D = theta_D_from_theta_EE(detector_grid_alt, 90-np.degrees(theta[i]))
    L_particle = Hfirst[i]  # meters
    X0_First=Xfirst[i]

    photons_on_plane[photons_on_plane < 0] = 0

    bins, start, stop = 100, 0, 4  # For time
    time_bins = np.logspace(float(start), float(stop), num=int(bins))
    bins, start, stop = 100, -3, 1  # For Zenith
    phZenith_bins = np.logspace(float(start), float(stop), num=int(bins))
    bins, start, stop = 1440, -180, 180  # For Azimuth
    bin_size = (float(stop) - float(start)) / float(bins)
    phAzi_bins = np.arange(float(start), float(stop), bin_size)
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
    phZenith_counts = np.histogram2d(r_coordinates, offset_arrival_phZenith, bins=(
        r_bins, phZenith_bins), weights=photons_on_plane)[0]
    phAzi_counts = np.histogram2d(r_coordinates.flatten(), azimuth_angles.flatten(), bins=(
        r_bins, phAzi_bins), weights=photons_on_plane.flatten())[0]

    spatial_counts = spatial_counts / ncounter_rbin

    # create branches for tshower

    eng[0] = energy
    zen[0] = theta_D
    azi[0] = azimdeg
    startdist[0] = L_particle
    X0[0]=X0_First
    tshower.Fill()

    # vec_time = vector("TH1D")()
    # vec_phZenith = vector("TH1D")()

    # define histograms

    num_wl_bin = len(wlbins) - 1
    num_dist_bin = len(r_bins) - 1
    num_time_bin = len(time_bins) - 1
    num_phZenith_bin = len(phZenith_bins) - 1
    num_phAzi_bin = len(phAzi_bins) - 1


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
    histo_phZenith_off = TH1D("phZenith_off", "phZenith offset",
                         num_dist_bin, r_bins)
    histo_t = [TH1D("t_dist_" + str(i), "time_dist_" + str(i),
                    num_time_bin, time_bins) for i in range(num_dist_bin)]
    histo_phZenith = [TH1D("phZenith_dist_" + str(i), "phZenith_dist_" + str(i),
                      num_phZenith_bin, phZenith_bins) for i in range(num_dist_bin)]
    histo_phAzi = [TH1D("phAzi_dist_" + str(i), "phAzi_dist_" + str(i),
                      num_phAzi_bin, phAzi_bins) for i in range(num_dist_bin)]
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
        for phZenith_bin, counts_phZenith in enumerate(phZenith_counts[r_bin]):
            histo_phZenith[r_bin].SetBinContent(phZenith_bin + 1, counts_phZenith)
        for phAzi_bin, counts_phAzi in enumerate(phAzi_counts[r_bin]):
            histo_phAzi[r_bin].SetBinContent(phAzi_bin + 1, counts_phAzi)

    # set branch for histograms
    tcher_ph.SetBranchAddress("wavelength", histo_w)
    tcher_ph.SetBranchAddress("distance", histo_r)
    tcher_ph.SetBranchAddress("phZenith_offset", histo_phZenith_off)
    tcher_ph.SetBranchAddress("time_offset", histo_t_off)
    vec_time.assign(histo_t)
    vec_phZenith.assign(histo_phZenith)
    vec_phAzi.assign(histo_phAzi)


    tcher_ph.Fill()
    # , vec_phZenith, vec_time
    del histo_w, histo_r, histo_t, histo_phZenith, histo_phAzi, histo_t_off, histo_phZenith_off
tshower.Write("", TFile.kOverwrite)
tcher_ph.Write("", TFile.kOverwrite)
print('Output written to {}'.format(filename))


file.Close()
