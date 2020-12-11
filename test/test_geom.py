import numpy as np
from nuSpaceSim import nssgeometry


def test_heaviside():
    assert nssgeometry.heaviside(0.0) == 1.0
    assert nssgeometry.heaviside(-1.0) == 0.0
    assert nssgeometry.heaviside(1.0) == 1.0


def test_geom_params():
    EarthRadius = 6371.0
    DetRA = 0.0
    DetDec = 0.0
    DetAlt = 525.0
    thetaChMax = np.pi*(3.0/180.0)
    delaziang = 2.0*np.pi
    delalpha = np.pi*(7.0/180.0)
    det_geom_A = nssgeometry.Geom_params(
        EarthRadius, DetAlt, DetRA, DetDec, delalpha, thetaChMax, delaziang,
        np.pi)
    assert det_geom_A.gt_det_ra() == DetRA

    newra = 30.0

    det_geom_B = nssgeometry.Geom_params(detra=newra)

    assert int(round(det_geom_B.get_det_ra())) == int(
        round(newra*(np.pi/180.0)))

    for i in range(2):
        det_geom_A.gen_traj()  # This event is not in the stack
        det_geom_A.get_local_event()

    numTrajs = 100000000

    # All events made in here are in the stack
    det_geom_A.run_geo_dmc_from_num_traj_hdf5(numTrajs)
    det_geom_A.print_geom_factor()

    # All events made in here are in the stack
    det_geom_A.run_geo_dmc_from_num_traj_nparray(numTrajs)
    det_geom_A.print_geom_factor()

    uranarray = np.random.rand(numTrajs, 4)

    det_geom_A.run_geo_dmc_from_ran_array_hdf5(uranarray)
    det_geom_A.print_geom_factor()

    det_geom_A.run_geo_dmc_from_ran_array_nparray(uranarray)
    det_geom_A.print_geom_factor()

    assert det_geom_A.evArray[numTrajs-1][0] == det_geom_A.localevent.thetaS
