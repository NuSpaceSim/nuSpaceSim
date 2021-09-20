import numpy as np
#import math
import pytest
#from datetime import datetime
#import random
#import faulthandler; faulthandler.enable()

import nssgeometry

def test_heaviside():
    assert nssgeometry.heaviside(0.0) == 1.0
    assert nssgeometry.heaviside(-1.0) == 0.0
    assert nssgeometry.heaviside(1.0) == 1.0

def test_geom_params():
    #random.seed(datetime.now())
    
    EarthRadius = 6371.0
    DetRA = 0.0
    DetDec = 0.0
    DetAlt = 525.0
    thetaChMax = np.pi*(3.0/180.0)
    delaziang = 2.0*np.pi
    #delaziang = np.pi/36.0
    delalpha = np.pi*(7.0/180.0)
    #delalpha = np.pi*(5.0/180.0)
    
    det_geom_A = nssgeometry.Geom_params(EarthRadius, DetAlt, DetRA, DetDec, delalpha, thetaChMax, delaziang, np.pi)
    
    assert det_geom_A.get_det_ra() == DetRA
    #assert det_geom_A.get_det_dec() == DetDec

    newra = 30.0

    det_geom_B = nssgeometry.Geom_params(detra=newra)

    assert int(round(det_geom_B.get_det_ra())) == int(round(newra*(np.pi/180.0)))
    
    #det_geom_A.gen_traj()
    #det_geom_A.get_event()

    #det_geom_B.gen_traj()
    #det_geom_B.get_event()

    for i in range(2):
        det_geom_A.gen_traj() # This event is not in the stack
        det_geom_A.get_local_event()

    numTrajs = 100000000

    #det_geom_A.run_geo_dmc_from_num_traj(numTrajs) # All events made in here are in the stack
    det_geom_A.run_geo_dmc_from_num_traj_hdf5(numTrajs) # All events made in here are in the stack
    det_geom_A.print_geom_factor()

    det_geom_A.run_geo_dmc_from_num_traj_nparray(numTrajs) # All events made in here are in the stack
    det_geom_A.print_geom_factor()

    uranarray = np.random.rand(numTrajs,4);
    
    #det_geom_A.run_geo_dmc_from_ran_array(uranarray)
    det_geom_A.run_geo_dmc_from_ran_array_hdf5(uranarray)
    det_geom_A.print_geom_factor()

    det_geom_A.run_geo_dmc_from_ran_array_nparray(uranarray)
    det_geom_A.print_geom_factor()

    testarray = det_geom_A.evArray # This statement is for the numpy array implementation

    # This block is for the numpy array implementation

    #for i in range(5):
     #   det_geom_A.print_event_from_array(i)

    assert det_geom_A.evArray[numTrajs-1][0] == det_geom_A.localevent.thetaS
    #assert testarray[numTrajs-1][0] == det_geom_A.localevent.thetaS

    #det_geom_A.print_geom_factor()
    
    

