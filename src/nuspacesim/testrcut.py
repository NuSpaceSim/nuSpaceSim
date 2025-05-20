import numpy as np
from .simulation.eas_optical.shower_properties import slant_depth_trig_approx
import matplotlib.pyplot as plt
from .augermc import *
from scipy.optimize import brentq

def min_distances_to_lines(xcorecentered, ycorecentered, zcorecentered, xaxis, yaxis, zaxis, LL):
    """
    Calculate the minimum distance from a target point LL to a set of lines defined by
    coordinates (xcorecentered, ycorecentered, zcorecentered) and directions (xaxis, yaxis, zaxis).
    
    Parameters:
    xcorecentered, ycorecentered, zcorecentered : array-like, shape (n,)
        Coordinates of n starting points (e.g., in meters).
    xaxis, yaxis, zaxis : array-like, shape (n,)
        Direction vector components of n lines.
    LL : array-like, shape (3,)
        Target point [x0, y0, z0] (e.g., in meters).
    
    Returns:
    distances : ndarray, shape (n,) in km
        Minimum distances from LL to each line.
    """

    xcorecentered = np.asarray(xcorecentered)
    ycorecentered = np.asarray(ycorecentered)
    zcorecentered = np.asarray(zcorecentered)
    xaxis = np.asarray(xaxis)
    yaxis = np.asarray(yaxis)
    zaxis = np.asarray(zaxis)
    LL = np.asarray(LL)

    # Convert split coordinates and directions to (n, 3) arrays
    points = np.stack((xcorecentered, ycorecentered, zcorecentered), axis=1)  # Shape: (n, 3)
    directions = np.stack((xaxis, yaxis, zaxis), axis=1)  # Shape: (n, 3)
    target_point = np.asarray(LL)  # Shape: (3,)
    
    # Check for zero-magnitude direction vectors
    dir_norms = np.linalg.norm(directions, axis=1)  # Shape: (n,)
    if np.any(dir_norms == 0):
        raise ValueError("Direction vectors cannot have zero magnitude.")
    
    # Vector from each starting point to target point: LL - Pi
    vectors_to_target = target_point - points  # Shape: (n, 3)
    
    # Cross product: (LL - Pi) × Di
    cross_products = np.cross(vectors_to_target, directions)  # Shape: (n, 3)
    
    # Magnitude of cross product divided by direction norm gives distance
    distances = np.linalg.norm(cross_products, axis=1) / dir_norms  # Shape: (n,)
    
    return distances

def Rcutoff(lgE):

    p1 =  4.86267e+05
    p2 = -6.72442e+04
    p3 =  2.31169e+03

    return p1 + p2 * lgE + p3 * lgE * lgE	

def test_rcut(xground,yground,zground,xangle,yangle,zangle,LL,lgE):
    min_dist=min_distances_to_lines(xground,yground,zground,xangle,yangle,zangle,LL)
    rcut=Rcutoff(lgE)*1.01
    weird=(min_dist>rcut)
    print(~weird)
    print('Analysis min dist and rcut ',np.size(weird),np.sum(weird),np.sum(~weird))
    print(min_dist[weird],rcut[weird])
    return weird

def find_trajectory_points_to_height(groundecef, vcoord, zend, s_max=1e7):
    """
    For each vcoord, find the ECEF point along the ray starting at groundecef
    that reaches height zend.

    Returns:
        coords_out: (N, 3) ECEF coordinates at height zend
        path_lengths: (N,) distance along each ray
    """
    s_min=zend
    N = vcoord.shape[0]
    coords_out = np.zeros((N, 3))
    path_lengths = np.zeros(N)

    for i in range(N):
        def height_diff(s):
            coord = groundecef + s * vcoord[i]
            _, _, h = ecef_to_latlong(coord)
            return h - zend
        
        try:
            # Find s where h = zend
            s_root = brentq(height_diff, s_min, s_max)
            coords_out[i] = groundecef + s_root * vcoord[i]
            path_lengths[i] = s_root
        except ValueError:
            # No root found in [s_min, s_max]
            coords_out[i] = np.nan
            path_lengths[i] = np.nan

    return coords_out, path_lengths


