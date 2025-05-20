import numpy as np
from scipy.spatial.transform import Rotation as R

def detector_coordinates_and_tr_azimuth(thetaTrSubV,phiTrSubV,costhetaNSubV,det_azi,det_elev, path_len):
    

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
    azimuth=np.arctan2(tr_n[:,1],tr_n[:,0])%(2*np.pi) #change to between 0 and 2pi. Does it matter that azim definition doesn't match between NSS and CHASM??
    #path_len2=path_length_tau_atm(altitude,geom.valid_elevAngVSubN()) This is the same as path_len
    x = path_len * np.cos(det_elev) * np.cos(det_azi)  # East
    y = path_len * np.cos(det_elev) * np.sin(det_azi)  # North
    z = path_len * np.sin(det_elev)                    # Up
    detcoords=1000*np.vstack((x,y,z)).T
    return detcoords, azimuth

def point_to_line_distances(detcoords, betaE, azimuth):
    """
    Calculate the shortest distance from each point in detcoords to the corresponding line
    defined by betaE (emergence angle) and azimuth.

    Parameters:
    - detcoords: numpy array of shape (N, 3), points in 3D space (x, y, z).
    - betaE: numpy array of shape (N,), emergence angles in radians.
    - azimuth: numpy array of shape (N,), azimuth angles in radians.

    Returns:
    - distances: numpy array of shape (N,), distances from each point to its corresponding line.
    """
    # Ensure inputs are numpy arrays
    detcoords = np.asarray(detcoords)
    betaE = np.asarray(betaE)
    azimuth = np.asarray(azimuth)

    # Check shapes
    N = detcoords.shape[0]
    if detcoords.shape != (N, 3) or betaE.shape != (N,) or azimuth.shape != (N,):
        raise ValueError("detcoords must be (N, 3), betaE and azimuth must be (N,)")

    # Compute direction vectors of the lines
    # Direction vector: (cos(betaE) * cos(azimuth), cos(betaE) * sin(azimuth), sin(betaE))
    dx = np.cos(betaE) * np.cos(azimuth)
    dy = np.cos(betaE) * np.sin(azimuth)
    dz = np.sin(betaE)

    # Stack into direction vectors of shape (N, 3)
    direction = np.stack([dx, dy, dz], axis=-1)  # Shape: (N, 3)

    # Normalize the direction vectors (though not strictly necessary since norm will be computed)
    # norm_direction = np.linalg.norm(direction, axis=1, keepdims=True)
    # direction = direction / norm_direction

    # Compute the cross product: detcoords × direction
    # detcoords: (N, 3), direction: (N, 3)
    cross_product = np.cross(detcoords, direction)  # Shape: (N, 3)

    # Compute the magnitude of the cross product
    cross_magnitude = np.linalg.norm(cross_product, axis=1)  # Shape: (N,)

    # Compute the magnitude of the direction vector
    direction_magnitude = np.linalg.norm(direction, axis=1)  # Shape: (N,)

    # Compute the distance
    distances = cross_magnitude / direction_magnitude

    #print('DISTANCES',distances, np.shape(distances))

    return distances

def angle_at_decay(detcoord,altDec,beta_tr,azimuth):
    xdec = 1000*altDec*np.cos(beta_tr) * np.cos(azimuth)
    ydec = 1000*altDec*np.cos(beta_tr) * np.sin(azimuth)
    zdec = 1000*altDec*np.sin(beta_tr)
    decaycoord=np.vstack((xdec,ydec,zdec)).T
    decaytodetector=detcoord-decaycoord
    cosangle = np.sum(decaytodetector * decaycoord, axis=1) / (np.linalg.norm(decaytodetector, axis=1) * np.linalg.norm(decaycoord, axis=1))
    #print('COSANGLE',cosangle, np.shape(cosangle))
    return np.arccos(cosangle)