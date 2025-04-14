import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy.polynomial import Polynomial
import scipy.integrate

telangle=np.radians(15.5) #Increase FoV to 31x31 to remove any edge effects
exacttelangle=np.radians(15)

massTau = 1.77686  # GeV/c^2
mean_Tau_life = 2.903e-13  # seconds
inv_mean_Tau_life=1/mean_Tau_life
earth_radius = 6371.0e3  # in meters
c = 2.9979246e8

def utm_to_geodetic(easting, northing, zone, band, height=None):
    """
    Converts UTM coordinates (easting, northing, zone, band) to geodetic (lat, lon, height).
    Vectorized for multiple inputs.
    
    Parameters:
    -----------
    easting : array-like
        UTM easting in meters.
    northing : array-like
        UTM northing in meters.
    zone : array-like
        UTM zone number (1–60).
    band : array-like
        UTM band letter (C–X, excluding I and O).
    height : array-like, optional
        Height in meters (passed through if provided).
    
    Returns:
    --------
    lat : ndarray
        Latitude in degrees.
    lon : ndarray
        Longitude in degrees.
    height : ndarray
        Height in meters (if provided, else None).
    """
    # WGS84 constants
    a = 6378137.0  # semi-major axis (m)
    b = 6356752.314245  # semi-minor axis (m)
    f = 1 - b / a
    e2 = 2 * f - f**2  # first eccentricity squared
    e_prime2 = e2 / (1 - e2)  # second eccentricity squared
    k0 = 0.9996  # UTM scale factor
    
    # Ensure inputs are arrays
    easting = np.atleast_1d(easting)
    northing = np.atleast_1d(northing)
    zone = np.atleast_1d(zone)
    band = np.atleast_1d(band)
    if height is not None:
        height = np.atleast_1d(height)
    
    # Central meridian for the zone
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)  # in radians
    
    # Define UTM band letters and their latitude ranges
    zone_letters = np.array(list("CDEFGHJKLMNPQRSTUVWX"))
    band_min_lat = np.arange(-80, 80, 8)  # C: -80°, D: -72°, ..., W: +72°
    
    # Determine if southern hemisphere (bands C–M, lat < 0)
    band_is_southern = np.isin(band, zone_letters[:10])  # C to M are southern (up to 0°)
    
    # Adjust northing for southern hemisphere
    northing_adj = northing.copy()
    northing_adj[band_is_southern] -= 10000000  # Remove false northing for southern bands
    
    # Footpoint latitude (initial approximation)
    M = northing_adj / k0  # Meridian distance
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    phi1 = mu + (3*e1/2 - 27*e1**3/32) * np.sin(2*mu) + \
           (21*e1**2/16 - 55*e1**4/32) * np.sin(4*mu) + \
           (151*e1**3/96) * np.sin(6*mu)  # Footpoint latitude (radians)
    
    # Compute projection constants
    N1 = a / np.sqrt(1 - e2 * np.sin(phi1)**2)
    T1 = np.tan(phi1)**2
    C1 = e_prime2 * np.cos(phi1)**2
    R1 = a * (1 - e2) / (1 - e2 * np.sin(phi1)**2)**1.5
    D = (easting - 500000) / (N1 * k0)  # Normalized easting
    
    # Latitude (phi)
    lat_rad = phi1 - (N1 * np.tan(phi1) / R1) * \
              (D**2 / 2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime2) * D**4 / 24 + \
               (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime2 - 3*C1**2) * D**6 / 720)
    lat = np.degrees(lat_rad)
    
    # Longitude (lambda)
    lon_rad = lon0 + (D - (1 + 2*T1 + C1) * D**3 / 6 + \
                      (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime2 + 24*T1**2) * D**5 / 120) / np.cos(phi1)
    lon = np.degrees(lon_rad)
    
    # Return results
    if height is None:
        return lat, lon
    return lat, lon, height

# Los Leones (LL)
LL = np.array([459208.3, 6071871.5, 1416.2])
LLlat,LLlong, LLheight=utm_to_geodetic(LL[0], LL[1], 19, 'H', LL[2])
LLang=np.radians(330-360)
LLphi = LLang+np.radians([15.01, 44.89, 75.00, 104.97, 134.99, 164.92]) #angle for perfect planes 15.045
LLelev = np.radians([15.65, 15.92, 15.90, 16.07, 15.95, 15.82])
LLphitot=[np.min(LLphi)-telangle,np.max(LLphi)+telangle]
LLthetatot=[np.min(LLelev)-telangle,np.max(LLelev)+telangle]

# Los Morados (LM)
LM = np.array([498903.7, 6094570.2, 1416.4])
LMlat,LMlong, LMheight=utm_to_geodetic(LM[0], LM[1], 19, 'H', LM[2])
LMang=np.radians(60)
LMphi = LMang+np.radians([14.86, 45.12, 75.04, 105.01, 134.79, 165.02])
LMelev = np.radians([15.96, 15.87, 15.81, 15.89, 15.97, 16.05])
LMphitot=[np.min(LMphi)-telangle,np.max(LMphi)+telangle]
LMthetatot=[np.min(LMelev)-telangle,np.max(LMelev)+telangle]

# Loma Amarilla (LA)
LA = np.array([480743.1, 6134058.4, 1476.7])
LAlat,LAlong, LAheight=utm_to_geodetic(LA[0], LA[1], 19, 'H', LA[2])
LAang=np.radians(188-360)
LAphi = LAang+np.radians([14.67, 44.98, 75.05, 105.37, 134.85, 164.91])
LAelev = np.radians([16.34, 16.10, 16.13, 15.79, 16.03, 15.75])
LAphitot=[np.min(LAphi)-telangle,np.max(LAphi)+telangle]
LAthetatot=[np.min(LAelev)-telangle,np.max(LAelev)+telangle]

# Coihueco (CO)
CO = np.array([445343.8, 6114140.0, 1712.3])
COlat,COlong, COheight=utm_to_geodetic(CO[0], CO[1], 19, 'H', CO[2])
COang=np.radians(243.0219-360)
COphi = COang+np.radians([14.88, 44.92, 74.93, 105.04, 134.82, 164.98])
COelev = np.radians([16.03, 16.14, 16.03, 16.20, 16.03, 16.12])
COphitot=[np.min(COphi)-telangle,np.max(COphi)+telangle]
COthetatot=[np.min(COelev)-telangle,np.max(COelev)+telangle]

#Central phi and elevation
telphi=[np.mean(LLphitot),np.mean(LLphitot)-LLphitot[0],np.mean(LMphitot),np.mean(LMphitot)-LMphitot[0],np.mean(LAphitot),np.mean(LAphitot)-LAphitot[0],np.mean(COphitot),np.mean(COphitot)-COphitot[0]]
teltheta=[np.mean(LLthetatot),np.mean(LLthetatot)-LLthetatot[0],np.mean(LMthetatot),np.mean(LMthetatot)-LMthetatot[0],np.mean(LAthetatot),np.mean(LAthetatot)-LAthetatot[0],np.mean(COthetatot),np.mean(COthetatot)-COthetatot[0]]
# Extract easting and northing values
easting_values = [LL[0], LM[0], LA[0], CO[0]]
northing_values = [LL[1], LM[1], LA[1], CO[1]]

# Calculate the mean easting and northing
mean_easting = np.mean(easting_values)   # 0471049.725
mean_northing = np.mean(northing_values) # 6103660.025
mean_lat=np.mean([LLlat,LMlat,LAlat,COlat])#-35.209444890061114
mean_long=np.mean([LLlong,LMlong,LAlong,COlong])#-69.31811672662049
# In Lat Long this is 35.209657 S 69.318078W
telang=[LLang,LMang,LAang,COang]
h=1416

def latlongtoECEF(lat,long,height): #in degrees, height in m
    lat=np.radians(lat)
    long=np.radians(long)
    a=6378137  #earth major axis WGS84
    b=6356752.314245 #minor axis
    e2=1-b**2/a**2
    N=a/np.sqrt(1-e2/(1+1/(np.tan(lat))**2)) #https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    x=(N+height)*np.cos(lat)*np.cos(long)
    y=(N+height)*np.cos(lat)*np.sin(long)
    z=(b**2/a**2*N+height)*np.sin(lat)
    coords=np.column_stack((x,y,z))
    return coords
def ecef_to_latlong(ecefcoords):
    a = 6378137.0  # semi-major axis in meters
    b = 6356752.314245  # semi-minor axis in meters
    e2 = 1 - (b**2 / a**2)  # first eccentricity squared
    # Ensure xyz is a 2D array of shape (n, 3)
    ecefcoords = np.atleast_2d(ecefcoords)
    x, y, z = ecefcoords[:, 0], ecefcoords[:, 1], ecefcoords[:, 2]
    
    # Longitude (vectorized)
    lon = np.degrees(np.arctan2(y, x))
    
    # Distance from Z-axis
    p = np.sqrt(x**2 + y**2)
    
    # Initial approximation for latitude
    theta = np.arctan2(z * a, p * b)  # Reduced latitude angle
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Refine latitude (Bowring’s method)
    num = z + (e2 * b * sin_theta**3) / (1 - e2)
    den = p - e2 * a * cos_theta**3
    lat = np.degrees(np.arctan2(num, den))
    
    # Radius of curvature in the prime vertical
    sin_lat = np.sin(np.radians(lat))
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    
    # Height (vectorized handling of poles)
    h = np.where(p > 1e-6, p / np.cos(np.radians(lat)) - N, np.sign(z) * z - b)
    
    # Stack results into (n, 3) array
    return lat, lon, h


origin_19H_lat=utm_to_geodetic(0, 0, 19, 'H', 0)
origin_19H=latlongtoECEF(origin_19H_lat[0], origin_19H_lat[1], origin_19H_lat[2])

LLecef=latlongtoECEF(LLlat,LLlong,LLheight)
LMecef=latlongtoECEF(LMlat,LMlong,LMheight)
LAecef=latlongtoECEF(LAlat,LAlong,LAheight)
COecef=latlongtoECEF(COlat,COlong,COheight)
#centerecef=latlongtoECEF(mean_lat,mean_long,h)
telposecef=np.array([LLecef,LMecef,LAecef,COecef])
telheight=[LLheight,LMheight,LAheight,COheight]
tellat=[LLlat,LMlat,LAlat,COlat]
"""
#First we calculate center using UTM (northing easting coords), then calculate ECF.
#Not this way because we want center to be on earth surface. Here its under the Earth
#The result of this is same lat long, but 1341m height instead of 1416
xECEF=[LLecef[0],LMecef[0],LAecef[0],COecef[0]]
yECEF=[LLecef[1],LMecef[1],LAecef[1],COecef[1]]
zECEF=[LLecef[2],LMecef[2],LAecef[2],COecef[2]]
meanECEF=[np.mean(xECEF),np.mean(yECEF),np.mean(zECEF)]"""
def ecef_to_enu_matrix(ecef_origin,lat=None,lon=None):
    """
    Generate the ECEF-to-ENU rotation matrix based on an ECEF origin (WGS84).
    
    Parameters:
        ecef_origin (array-like): ECEF coordinates of the origin, shape (1, 3) or (3,).
    
    Returns:
        R (ndarray): 3x3 rotation matrix from ECEF to ENU.
    """
    # WGS84 parameters
    a = 6378137.0
    b = 6356752.314245
    
    # Ensure ecef_origin is 2D with shape (n, 3)
    ecef_origin = np.atleast_2d(ecef_origin)
    
    # Compute lat/lon if not provided
    if lat is None or lon is None:
        lat, lon, _ = ecef_to_latlong(ecef_origin)  # Returns 3 arrays, each (n,)
        # Check if ecef_origin resulted in multiple lat/lon values
        if lat.size > 1:
            raise ValueError("ecef_origin must represent a single point, got multiple coordinates")
        if lon.size > 1:
            raise ValueError("ecef_origin must represent a single point, got multiple coordinates")
        lat = lat[0]  # Take first element to get scalar
        lon = lon[0]  # Take first element to get scalar
    else:
        # Convert to arrays and check lengths
        lat_array = np.atleast_1d(lat)
        lon_array = np.atleast_1d(lon)
        if lat_array.size > 1:
            raise ValueError("lat must be a scalar or single-element array, got multiple elements")
        if lon_array.size > 1:
            raise ValueError("lon must be a scalar or single-element array, got multiple elements")
        lat = lat_array[0] if lat_array.size == 1 else lat  # Extract scalar
        lon = lon_array[0] if lon_array.size == 1 else lon  # Extract scalar
    # Convert to radians for matrix
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # ECEF-to-ENU rotation matrix
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    return R

def eceftoenu(ecef_origin, ecef_coords, lat=None, lon=None):
    """
    Convert ECEF coordinates to ENU coordinates relative to an ECEF origin.
    
    Parameters:
        ecef_coords (array-like): ECEF coordinates, shape (n, 3).
        ecef_origin (array-like): ECEF origin, shape (1, 3) or (3,).
    
    Returns:
        enu_coords (ndarray): ENU coordinates, shape (n, 3).
    """
    ecef_coords = np.atleast_2d(ecef_coords)
    R = ecef_to_enu_matrix(ecef_origin,lat,lon)
    delta = ecef_coords - ecef_origin  # Displacement from origin
    return delta @ R.T

def enutoecef(ecef_origin, enu_coords, lat=None, lon=None):
    """
    Convert ENU coordinates to ECEF coordinates given an ECEF origin.
    
    Parameters:
        enu_coords (array-like): ENU coordinates, shape (n, 3).
        ecef_origin (array-like): ECEF origin, shape (1, 3) or (3,).
    
    Returns:
        ecef_coords (ndarray): ECEF coordinates, shape (n, 3).
    """
    enu_coords = np.atleast_2d(enu_coords)
    R = ecef_to_enu_matrix(ecef_origin,lat,lon)
    return (enu_coords @ R) + ecef_origin  # R is ECEF-to-ENU, so R.T inverse is applied via R

def eceftoenu_vector(ecef_origin, ecef_vectors, lat=None, lon=None):
    """
    Convert ECEF vectors to ENU vectors (no origin displacement).
    
    Parameters:
        ecef_vectors (array-like): ECEF vectors, shape (n, 3).
        ecef_origin (array-like): ECEF origin for orientation, shape (1, 3) or (3,).
    
    Returns:
        enu_vectors (ndarray): ENU vectors, shape (n, 3).
    """
    ecef_vectors = np.atleast_2d(ecef_vectors)
    R = ecef_to_enu_matrix(ecef_origin,lat,lon)
    return ecef_vectors @ R.T

def enutoecef_vector(ecef_origin, enu_vectors, lat=None, lon=None):
    """
    Convert ENU vectors to ECEF vectors (no origin displacement).
    
    Parameters:
        enu_vectors (array-like): ENU vectors, shape (n, 3).
        ecef_origin (array-like): ECEF origin for orientation, shape (1, 3) or (3,).
    
    Returns:
        ecef_vectors (ndarray): ECEF vectors, shape (n, 3).
    """
    enu_vectors = np.atleast_2d(enu_vectors)
    R = ecef_to_enu_matrix(ecef_origin,lat,lon)
    return enu_vectors @ R

def geodetic_to_utm(lat, lon,height):
    """
    Converts geodetic coordinates (lat, lon) to UTM (easting, northing, zone, hemisphere).
    Vectorized for multiple inputs.
    """
    # WGS84 constants
    b=6356752.314245
    a=6378137.0
    f=1-b/a
    e2 = 2*f - f**2  # first eccentricity squared
    k0 = 0.9996  # UTM scale factor
    
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)
    height = np.atleast_1d(height)

    # Convert lat/lon to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # UTM zone number
    zone = np.floor((lon + 180) / 6) + 1
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)  # Central meridian of zone
    
    # Compute projection constants
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    T = np.tan(lat_rad)**2
    C = e2 / (1 - e2) * np.cos(lat_rad)**2
    A = (lon_rad - lon0) * np.cos(lat_rad)
    
    # Meridian distance with higher-order terms
    e4 = e2**2
    e6 = e2**3
    M = a * (
        (1 - e2/4 - 3*e4/64 - 5*e6/256) * lat_rad
        - (3*e2/8 + 3*e4/32 + 45*e6/1024) * np.sin(2*lat_rad)
        + (15*e4/256 + 45*e6/1024) * np.sin(4*lat_rad)
        - (35*e6/3072) * np.sin(6*lat_rad)
    )
    
    # Compute UTM Easting and Northing
    easting = (k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * e2) * A**5 / 120) + 500000)
    northing = (k0 * (M + N * np.tan(lat_rad) * (A**2 / 2 + (5 - T + 9 * C + 4 * C**2) * A**4 / 24 + (61 - 58 * T + T**2 + 600 * C - 330 * e2) * A**6 / 720)))
    
    # Adjust for southern hemisphere
    northing[lat < 0] += 10000000  # False northing
    
    # UTM zone letters
    zone_letters = np.array(list("CDEFGHJKLMNPQRSTUVWX"))
    idx = np.clip(((lat + 80) // 8).astype(int), 0, len(zone_letters) - 1)
    zone_letter = zone_letters[idx]
    return np.array(easting), np.array(northing), height, np.array(zone.astype(int)), zone_letter


def ecef_to_utm(ecef_coords):
    """
    Convert ECEF coordinates to UTM coordinates.
    
    Parameters:
        ecef_coords (array-like): ECEF coordinates, shape (n, 3).
    
    Returns:
        utm_coords (ndarray): UTM coordinates, shape (n, 5) with columns [easting, northing, height, zone_number, zone_letter].
    """
    ecef_coords = np.atleast_2d(ecef_coords)
    lat, lon, height = ecef_to_latlong(ecef_coords)
    return geodetic_to_utm(lat, lon,height)

def WGS84ellipse_scaledh(coord,height=h):
    a2=(6378137+height)**2  #earth major axis WGS84
    b2=(6356752.314245+height)**2 #minor axis
    return coord[:,0]**2/a2+coord[:,1]**2/a2+coord[:,2]**2/b2  #=1 is in ellipsoid, <1 inside >1 outside

#There's always a 40m difference between radius+h at given latitude, and distance from point to WGS84 center
#This difference is 0 at lat=0 and at lat=90, and looks like it max is around 45deg (malargue is at lat=35)

# Create the center array
nametags=['Los Leones','Los Morados', 'Loma Amarilla', 'Coihueco']
centerseaecef=latlongtoECEF(mean_lat,mean_long,0)
centerelevatedecef=latlongtoECEF(mean_lat,mean_long,h)
LLenu=eceftoenu(centerseaecef,LLecef)
LMenu=eceftoenu(centerseaecef,LMecef)
LAenu=eceftoenu(centerseaecef,LAecef)
COenu=eceftoenu(centerseaecef,COecef)

telposenu=np.array([LLenu,LMenu,LAenu,COenu])
telposenuhelevated=telposenu-[0,0,h]

teldistenu=np.linalg.norm(telposenuhelevated,axis=1) #Distance of telescopes to center of array at h

num_ang=10000
ang=np.linspace(0,np.pi,num_ang)
#print(CO)


def radiusatlat(lat):
    lat = np.radians(lat)
    b = 6356752.314245
    a = 6378137.0
    return np.sqrt(a**2 * np.cos(lat)**2 + b**2 * np.sin(lat)**2)

earth_radius_centerlat=radiusatlat(mean_lat)

def Rcutoff(lgE):

    p1 =  4.86267e+05
    p2 = -6.72442e+04
    p3 =  2.31169e+03

    return p1 + p2 * lgE + p3 * lgE * lgE	

def roundcalcradius(E, extraradius=1.01):  #This uses only the sides of the telescope (not the forward direction) since that's the limitant
                         #distance, as the telescopes are pointing inwards, towards the center.
    rEnergy=Rcutoff(E)*extraradius
    maxdist=0
    for i in range(4):
        centerenu_respect_tel=eceftoenu(telposecef[i],centerelevatedecef)
        side1=[(rEnergy*np.cos(telang[i])),(rEnergy*np.sin(telang[i])),0]
        side2=[(rEnergy*np.cos(telang[i]+np.pi)),(rEnergy*np.sin(telang[i]+np.pi)),0]
        maxcandidate=np.max([np.linalg.norm(side1-centerenu_respect_tel),np.linalg.norm(side2-centerenu_respect_tel)])
        maxdist=np.max([maxdist,maxcandidate])
    return maxdist

def calcradius(E,num_ang,extraradius=1.01,plotfig=False):
    rEnergy=Rcutoff(E)*extraradius

    #LL
    xLLcirc=rEnergy*np.cos(ang+LLang)+LLenu[0]
    yLLcirc=rEnergy*np.sin(ang+LLang)+LLenu[1]
    xLLline = np.linspace(xLLcirc[-1], xLLcirc[0], num_ang)
    yLLline = np.linspace(yLLcirc[-1], yLLcirc[0], num_ang)
    xLL=np.concatenate((xLLcirc,xLLline))
    yLL=np.concatenate((yLLcirc,yLLline))
    if plotfig==True: 
        plt.plot(xLL,yLL,color='red')

    # LM
    xLMcirc=rEnergy*np.cos(ang+LMang)+LMenu[0]
    yLMcirc=rEnergy*np.sin(ang+LMang)+LMenu[1]
    xLMline = np.linspace(xLMcirc[-1], xLMcirc[0], num_ang)
    yLMline = np.linspace(yLMcirc[-1], yLMcirc[0], num_ang)
    xLM=np.concatenate((xLMcirc,xLMline))
    yLM=np.concatenate((yLMcirc,yLMline))
    if plotfig==True: 
        plt.plot(xLM,yLM,color='purple')

    # LA
    xLAcirc=rEnergy*np.cos(ang+LAang)+LAenu[0]
    yLAcirc=rEnergy*np.sin(ang+LAang)+LAenu[1]
    xLAline = np.linspace(xLAcirc[-1], xLAcirc[0], num_ang)
    yLAline = np.linspace(yLAcirc[-1], yLAcirc[0], num_ang)
    xLA=np.concatenate((xLAcirc,xLAline))
    yLA=np.concatenate((yLAcirc,yLAline))
    if plotfig==True: 
        plt.plot(xLA,yLA,color='yellow')

    # CO
    xCOcirc=rEnergy*np.cos(ang+COang)+COenu[0]
    yCOcirc=rEnergy*np.sin(ang+COang)+COenu[1]
    xCOline = np.linspace(xCOcirc[-1], xCOcirc[0], num_ang)
    yCOline = np.linspace(yCOcirc[-1], yCOcirc[0], num_ang)
    xCO=np.concatenate((xCOcirc,xCOline))
    yCO=np.concatenate((yCOcirc,yCOline))
    if plotfig==True: 
        plt.plot(xCO,yCO,color='blue')

    dLL = np.sqrt(xLL**2 + yLL**2)
    dLM = np.sqrt(xLM**2 + yLM**2)
    dLA = np.sqrt(xLA**2 + yLA**2)
    dCO = np.sqrt(xCO**2 + yCO**2)
    d_stack = np.stack((dLL, dLM, dLA, dCO))
    maxd = np.argmax(d_stack)
    n2=int(num_ang*2)
    if np.floor(maxd/n2)==0:
        xmax=xLL[maxd%n2]
        ymax=yLL[maxd%n2]
    elif np.floor(maxd/n2)==1:
        xmax=xLM[maxd%n2]
        ymax=yLM[maxd%n2]
    elif np.floor(maxd/n2)==2:
        xmax=xLA[maxd%n2]
        ymax=yLA[maxd%n2]
    else:
        xmax=xCO[maxd%n2]
        ymax=yCO[maxd%n2]
    r=np.sqrt(xmax**2+ymax**2) #this r is bugged im not sure why
    if plotfig==True: 
        angtot=np.linspace(0,2*np.pi,num_ang)
        plt.plot(np.max(d_stack)*np.cos(angtot),np.max(d_stack)*np.sin(angtot),color='black',linewidth=2.5,label='Sphere to throw')
        plt.gca().set_aspect('equal')
 
        plt.show()
    return np.max(d_stack)

#Take random point in hemisphere (surface). Returns points in the ground and vectors pointing up
def gen_points(n,r,maxang=np.radians(90)):
    #33% more because we remove 1/4 bcs of FoV. 10% more to make sure we reach the intended n with maxang
    ninit=int((1+1/3)*4*n/np.sin(maxang)) #
    xr=np.random.normal(size=(ninit))
    yr=np.random.normal(size=(ninit))
    zr=np.random.normal(size=(ninit))
    norm=1/np.sqrt((xr**2+yr**2+zr**2))

    x=xr*r*norm
    y=yr*r*norm
    z=zr*r*norm

    xv=np.random.normal(size=(ninit))
    yv=np.random.normal(size=(ninit))
    zv=np.random.normal(size=(ninit))
    norm=1/np.sqrt((xv**2+yv**2+zv**2))
    print('Empezar ',ninit)
    x2=xv*r*norm
    y2=yv*r*norm
    z2=zv*r*norm
    biggerz=(z>z2)
    coord1 = np.column_stack((x, y, z))
    coord2 = np.column_stack((x2,y2,z2))
    coord1[biggerz],coord2[biggerz]=coord2[biggerz],coord1[biggerz] #make coord1 always the lowest point, so that vector is pointing up always

    coord1ecef=enutoecef(centerelevatedecef,coord1,mean_lat,mean_long)
    coord2ecef=enutoecef(centerelevatedecef,coord2,mean_lat,mean_long)
    inearth1=WGS84ellipse_scaledh(coord1ecef)
    inearth2=WGS84ellipse_scaledh(coord2ecef)
    #round earth gives ~0.3% more events from horizontal showers at 10**19 eV
    maskfov=~((inearth1<1)&(inearth2<1))  #invalid trajectory since its not crossing the field of view
    print('En FoV ', maskfov.sum())
    coordecef=coord1ecef[maskfov]
    vcoordecef=(coord2ecef[maskfov]-coordecef)/ np.linalg.norm((coord2ecef[maskfov]-coordecef), axis=1, keepdims=True)

    groundecef, vcoordecefground, beta, azimuth= ground_xy(coordecef,vcoordecef)

    mask=(beta<=maxang)#&(beta>=np.radians(10))  #CUIDADO CON ESTO
    return groundecef[mask,:][0:n,:], vcoordecefground[mask,:][0:n,:],beta[mask][0:n],azimuth[mask][0:n]#coordg[mask,:][0:n,:], vcoordg[mask,:][0:n,:]

def ground_xy(coord,vcoord,height=h):   #Now included inside gen_points
    #c2t**2+c1t+c0=0 
    b2=(6356752.314245+h)**2
    a2 = (6378137.0+h)**2
    c2=vcoord[:,0]**2/a2+vcoord[:,1]**2/a2+vcoord[:,2]**2/b2
    c1=2*(vcoord[:,0]*coord[:,0]/a2+vcoord[:,1]*coord[:,1]/a2+vcoord[:,2]*coord[:,2]/b2)
    c0=coord[:,0]**2/a2+coord[:,1]**2/a2+coord[:,2]**2/b2-1
    D=c1**2-4*c2*c0
    mask=(D>0)
    coord=coord[mask]
    vcoord=vcoord[mask]
    #always take bigger t because traj is upward -> starting point is lower and moves in correct direction
    t=(-c1[mask]+np.sqrt(c1[mask]**2-4*c2[mask]*c0[mask]))/(2*c2[mask])
    groundecef=coord+t[:,np.newaxis]*vcoord

    #emergence angle calculation
    normal=np.column_stack((groundecef[:,0]/a2,groundecef[:,1]/a2,groundecef[:,2]/b2))
    normal=normal/np.linalg.norm(normal, axis=1, keepdims=True)
    east_vector=np.zeros_like(normal)
    east_vector[:,0]=-normal[:,1]
    east_vector[:,1]=normal[:,0]
    east_vector = east_vector / np.linalg.norm(east_vector, axis=1, keepdims=True)
    dotprod= np.sum(vcoord*normal,axis=1)
    vtangent=vcoord-dotprod[:,np.newaxis]*normal
    vtangent = vtangent / np.linalg.norm(vtangent, axis=1, keepdims=True)

    cos_theta = np.clip(np.sum(east_vector * vtangent, axis=1), -1.0, 1.0)
    theta = np.arccos(cos_theta)    
    cross_prod = np.cross(east_vector, vtangent)
    sign = np.sign(np.sum(cross_prod* normal,axis=1))
    azimuth = np.where(sign >= 0, theta, 2*np.pi - theta)
    beta=np.pi/2-np.arccos(dotprod)
    print('Cortan Tierra, ',beta.size)
    return groundecef,vcoord, beta, azimuth

def gen_eye_vectors(telphi, teltheta):  #Vector that points to the center of FoV of the telescope (~15deg elevation, center of azimuth)
    LLvector=[np.cos(teltheta[0])*np.cos(telphi[0]),np.cos(teltheta[0])*np.sin(telphi[0]),np.sin(teltheta[0])]
    LMvector=[np.cos(teltheta[2])*np.cos(telphi[2]),np.cos(teltheta[2])*np.sin(telphi[2]),np.sin(teltheta[2])]
    LAvector=[np.cos(teltheta[4])*np.cos(telphi[4]),np.cos(teltheta[4])*np.sin(telphi[4]),np.sin(teltheta[4])]
    COvector=[np.cos(teltheta[6])*np.cos(telphi[6]),np.cos(teltheta[6])*np.sin(telphi[6]),np.sin(teltheta[6])]
    eyevector=np.vstack((LLvector,LMvector,LAvector,COvector))
    return eyevector#/np.linalg.norm(eyevector, axis=1, keepdims=True)

def trajectory_inside_tel_sphere(lgE,coordecef,vcoordecef,ntels=telposecef.shape[0],telphi=telphi,teltheta=teltheta,radiusfactor=1.01):
    r=Rcutoff(lgE)*radiusfactor
    eyevector=gen_eye_vectors(telphi,teltheta)
    identifier=np.ones(coordecef[:,0].size)
    int1=[[]]
    int2=[[]]
    code=[2,3,5,7]
    
    for i in range(ntels):#telpos.shape[0]
        coordenu=eceftoenu(telposecef[i,:],coordecef)
        vcoordenu=eceftoenu_vector(telposecef[i,:],vcoordecef)
        a=vcoordenu[:,0]**2+vcoordenu[:,1]**2+vcoordenu[:,2]**2  #https://paulbourke.net/geometry/circlesphere/index.html#linesphere
        b=2*(vcoordenu[:,0]*coordenu[:,0]+vcoordenu[:,1]*coordenu[:,1]+vcoordenu[:,2]*coordenu[:,2])
        c=coordenu[:,0]**2+coordenu[:,1]**2+coordenu[:,2]**2-r**2
        D=b**2-4*a*c
        sign=np.sign(D)
        mask1=(sign>=0)
        print(np.size(sign),'Start')
        print(mask1.sum(),'Inside Sphere')

        inplane=np.full_like(a,False,dtype='bool')
        #CHECK INTERSECTION WITH "GROUND" PLANE
        rotangle=teltheta[2*i]-teltheta[2*i+1]
        rotaxis = np.array([-eyevector[i,1],eyevector[i,0] , 0])
        theta_rotation = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * rotangle)
        coordi = theta_rotation.apply(coordenu[mask1])
        vcoordi = theta_rotation.apply(vcoordenu[mask1])
        alpha=-coordi[:,2]/vcoordi[:,2]
        x0=coordi[:,0]+alpha*vcoordi[:,0]
        y0=coordi[:,1]+alpha*vcoordi[:,1]
        c_ground=np.column_stack((x0,y0))
        dist=np.linalg.norm(c_ground,axis=1)
        dotprod=np.dot(c_ground,eyevector[i,0:2])
        inplane=(dist<=r[mask1]) & (dotprod>=0)
        index = np.arange(len(a))[mask1][inplane]
        identifier[index]=identifier[index]*code[i]
        mask1[mask1]=~inplane # UNCHECK
        print(inplane.sum(),'Inside Plane, remaining: ',mask1.sum())

        """Plots ground
        plt.plot(x0[inplane],y0[inplane])
        plt.gca().set_aspect('equal')
        angle=np.linspace(0,2*np.pi)
        plt.plot(r*np.cos(angle)+telpos[i,0],r*np.sin(angle)+telpos[i,1])
        plt.plot(telpos[i,0],telpos[i,1],'+')
        plt.show()
        """

        #CALCULATE INTERSECTION POINT WITH SPHERE
        sqrtD=np.sqrt(D[mask1])
        thetatelsup=teltheta[2*i]+telangle
        thetatelinf=teltheta[2*i]-telangle
        u1=((-b[mask1]-sqrtD)*(1/a[mask1])*(1/2))[:,np.newaxis]
        intvec1=(coordenu[mask1]+u1*vcoordenu[mask1])/r[mask1,np.newaxis]
        cosdphi1=np.dot(intvec1[:,0:2],eyevector[i,0:2])/np.linalg.norm(intvec1[:,0:2],axis=1)/np.linalg.norm(eyevector[i,0:2])   #azimuth angle difference with center of telescope
        theta1=np.arccos(np.sqrt(intvec1[:,0]**2+intvec1[:,1]**2))*np.sign(intvec1[:,2]) #elevation angle of intersection vector 
        insphere1=(cosdphi1>=np.cos(exacttelangle*5+telangle)) &  (theta1<=thetatelsup) &  (theta1>=thetatelinf)   #exactangle*5+telangle bcs the extra 0.5 deg only for the last telescope (want to increase fov by 0.5, not 0.5*6)

        index = np.arange(len(a))[mask1][insphere1]
        identifier[index]=identifier[index]*code[i]
        mask1[mask1]=~insphere1 # UNCHECK
        print(insphere1.sum(),'Sphere1, remaining: ',mask1.sum())

        #Intersection of second point (if first one wasn't inside) (this is the highest point)
        u2=((-b[mask1]+sqrtD[~insphere1])*(1/a[mask1])*(1/2))[:,np.newaxis]
        intvec2=(coordenu[mask1]+u2*vcoordenu[mask1])/r[mask1,np.newaxis]
        cosdphi2=np.dot(intvec2[:,0:2],eyevector[i,0:2])/np.linalg.norm(intvec2[:,0:2],axis=1)/np.linalg.norm(eyevector[i,0:2])   #azimuth angle difference with center of telescope
        theta2=np.arccos(np.sqrt(intvec2[:,0]**2+intvec2[:,1]**2))*np.sign(intvec2[:,2]) #elevation angle of intersection vector
        insphere2=(cosdphi2>=np.cos(exacttelangle*5+telangle)) &  (theta2<=thetatelsup) &  (theta2>=thetatelinf)
        index = np.arange(len(a))[mask1][insphere2]
        identifier[index]=identifier[index]*code[i]

        mask1[mask1]=~insphere2 # UNCHECK
        print(insphere2.sum(),'Sphere2, remaining: ',mask1.sum())
        

        #CHECK INTERSECTION WITH BACKSIDE PLANE 1
        
        phiangle1=telphi[2*i]-telphi[2*i+1]
        rotaxis = np.array([0,0,1])
        phi_rotation = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * -phiangle1)
        coordi = phi_rotation.apply(coordenu[mask1])
        vcoordi = phi_rotation.apply(vcoordenu[mask1])
        alpha=(-coordi[:,1])/vcoordi[:,1]
        x0=coordi[:,0]+alpha*vcoordi[:,0]
        z0=coordi[:,2]+alpha*vcoordi[:,2]
        y0=np.full_like(x0,0)
        v_intersec1=np.column_stack((x0,y0,z0))
        dist=np.linalg.norm(v_intersec1,axis=1)
        vreference=np.array([np.cos(teltheta[2*i]),0,np.sin(teltheta[2*i])])
        dotprod=np.dot(v_intersec1/np.linalg.norm(v_intersec1, axis=1, keepdims=True) ,vreference)
        inback1=(dist<=r[mask1]) & (dotprod>=np.cos(telangle)) 
        index = np.arange(len(a))[mask1][inback1]
        identifier[index]=identifier[index]*code[i]

        mask1[mask1]=~inback1
        print(inback1.sum(),'Back Plane1, remaining: ',mask1.sum())
        #CHECK INTERSECTION WITH BACKSIDE PLANE 2
        
        phiangle2=telphi[2*i]+telphi[2*i+1]
        rotaxis = np.array([0,0,1])
        phi_rotation = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * (-phiangle2+np.pi))
        coordi = phi_rotation.apply(coordenu[mask1])
        vcoordi = phi_rotation.apply(vcoordenu[mask1])

        alpha=(-coordi[:,1])/vcoordi[:,1]
        x0=coordi[:,0]+alpha*vcoordi[:,0]
        z0=coordi[:,2]+alpha*vcoordi[:,2]
        y0=np.full_like(x0,0)
        v_intersec2=np.column_stack((x0,y0,z0))
        dist=np.linalg.norm(v_intersec2,axis=1)
        vreference=np.array([-np.cos(teltheta[2*i]),0,np.sin(teltheta[2*i])])
        dotprod=np.dot(v_intersec2/np.linalg.norm(v_intersec2, axis=1, keepdims=True) ,vreference)
        inback2=(dist<=r[mask1]) & (dotprod>=np.cos(telangle))
        index = np.arange(len(a))[mask1][inback2]
        identifier[index]=identifier[index]*code[i]
        mask1[mask1]=~inback2
        print(inback2.sum(),'Back Plane2. End. Leftover: ',mask1.sum())
        print(f'Total Inside FoV for telescope {i} = ',np.count_nonzero(identifier%code[i]==0))



        inside=(identifier%code[i]==0)
        intfactor1=((-b[inside]-np.sqrt(D[inside]))*(1/a[inside])*(1/2))[:,np.newaxis]
        intfactor2=((-b[inside]+np.sqrt(D[inside]))*(1/a[inside])*(1/2))[:,np.newaxis]
        int1=np.append(int1,coordenu[inside]+intfactor1*vcoordenu[inside])
        int2=np.append(int2,coordenu[inside]+intfactor2*vcoordenu[inside])
    int1=int1.reshape(-1,3)
    int2=int2.reshape(-1,3)
    return identifier, int1,int2


_polyrho = Polynomial(
    [-1.00867666e-07, 2.39812768e-06, 9.91786255e-05, -3.14065045e-04, -6.30927456e-04,
     1.70053229e-03, 2.61087236e-03, -5.69630760e-03, -2.12098836e-03, 5.68074214e-03,
     6.54893281e-04, -1.98622752e-03, ],
    domain=[0.0, 100.0],
)
# fmt: on
def atmdensity(z):
    """
    Density (g/cm^3) parameterized from altitude (z) values

    Computation is an (11) degree polynomial fit to equation (2)
    in https://arxiv.org/pdf/2011.09869.pdf
    Fit performed using numpy.Polynomial.fit
    """

    p = np.where(z < 100, _polyrho(z), np.reciprocal(1e9))
    return p
def slant_depth_integrand(z, theta_tr, rho=atmdensity, earth_radius=(earth_radius_centerlat/1000)):
    """
    Integrand for computing slant_depth from input altitude z.
    Computation from equation (3) in https://arxiv.org/pdf/2011.09869.pdf
    """
    #In km
    theta_tr = np.asarray(theta_tr) 
    i = earth_radius**2 * np.cos(theta_tr) ** 2
    j = z**2
    k = 2 * z * earth_radius

    ijk = i + j + k

    return 1e5 * rho(z) * ((z + earth_radius) / np.sqrt(ijk))
def calculate_grammage_end(lgE,step=5):
        grammage=np.linspace(0.1,4000,4000)
        Eshow=10**(lgE-9)
        t = grammage / 36.66
        Zair = 7.4
        ecrit = 0.710 / (Zair + 0.96)
        greisen_beta = np.log(np.float64(Eshow) / np.float64(ecrit))
        s = 3 * t / (t + 2 * greisen_beta)

        RN=0.31 / np.sqrt(greisen_beta) * np.exp( t * (1 - 3 / 2 * np.log(s)))
        RN[RN < 0] = 0

        cumRN=np.cumsum(RN)
        step=step/100
        values=cumRN[-1]*np.arange(step,1,step)
        indexes = [np.argmax(cumRN > v) for v in values]

        return grammage[indexes]

def slant_depth(
    z_lo,     #z must be in km
    z_hi,
    beta,
    earth_radius=earth_radius_centerlat/1000,   #in km
    func=slant_depth_integrand,
    epsabs=1e-2,
    epsrel=1e-2,
):
    """
    Slant-depth in g/cm^2 from equation (3) in https://arxiv.org/pdf/2011.09869.pdf

    Parameters
    ----------
    z_lo : float
        Starting altitude for slant depth track.
    z_hi : float
        Stopping altitude for slant depth track.
    beta: float, array_like
        Trajectory angle in radians between the track and ground.
    theta_tr: float, array_like
        Trajectory angle in radians between the track and earth zenith.
    earth_radius: float
        Radius of a spherical earth. Default from nuspacesim.constants
    func: callable
        The integrand for slant_depth. If None, defaults to `slant_depth_integrand()`.

    Returns
    -------
    x_sd: ndarray
        slant_depth g/cm^2
    err: (float) numerical error.

    """
    theta_tr=np.pi/2-beta
    theta_tr = np.asarray(theta_tr)

    def f(x):
        y = np.multiply.outer(z_hi - z_lo, x).T + z_lo
        return (func(y, theta_tr=theta_tr, earth_radius=earth_radius) * (z_hi - z_lo)).T

    return scipy.integrate.quad(f, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel)
    #return scipy.integrate.nquad(f, [0.0, 1.0], **kwargs)

def path_length_tau_atm(z, beta_tr, Re=earth_radius_centerlat, xp=np):  #in m
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Resinb = Re * xp.sin(beta_tr)
    return xp.sqrt(Resinb**2 + (Re + z) ** 2 - Re**2) - Resinb

def altitude_along_path_length(s, beta_tr, Re=earth_radius_centerlat, xp=np): #in m
    """Derived by solving for z in path_length_tau_atm."""
    return xp.sqrt(s**2 + 2.0 * s * Re * xp.sin(beta_tr) + Re**2) - Re

def altitude_from_ecef(coordsecef):
    a = 6378137.0  # Semi-major axis in meters
    b = 6356752.314245  # Semi-minor axis in meters
    mag = np.linalg.norm(coordsecef, axis=1)
    direction = coordsecef / mag[:, np.newaxis]  
    A = (direction[:, 0]**2 + direction[:, 1]**2) / a**2 + direction[:, 2]**2 / b**2
    t = 1 / np.sqrt(A)  
    seaprojectiondecayecef = t[:, np.newaxis] * direction

    return np.linalg.norm(coordsecef-seaprojectiondecayecef, axis=1)

def decay(groundecef,vecef, lgE):
    #get decay altitude
    tauEnergy=10**(lgE-9)
    tauLorentz = tauEnergy / massTau
    tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz**2))
    #lgE = np.log10(data["tauEnergy"]) + 9
    u = np.random.uniform(0, 1, len(groundecef[:,0]))
    tDec = (-1.0 * tauLorentz /inv_mean_Tau_life) * np.log(u)

    lenDec = tDec * tauBeta * c

    decayecef=groundecef+lenDec[:,np.newaxis]*vecef

    a = 6378137.0  # Semi-major axis in meters
    b = 6356752.314245  # Semi-minor axis in meters

    return decayecef,altitude_from_ecef(decayecef)/1000#, lenDec



def decay_inside_fov(lgE,groundecef,vecef,beta,decayecef,altdec, id,fullint1,fullint2,ntels=telposecef.shape[0],radiusfactor=1.01,
 minshowerpct=1,step=0.5, diststep=10, telphi=telphi,teltheta=teltheta,telangle=telangle):
    r=Rcutoff(lgE)*radiusfactor
    code=[2,3,5,7]
    eyevector=gen_eye_vectors(telphi,teltheta)
    for i in range(ntels):#telpos.shape[0]
        teli=(id%code[i]==0)
        energy=lgE[teli]
        decayecefi=decayecef[teli]
        vecefi=vecef[teli]
        betai=beta[teli]
        altdeci=altdec[teli]
        decayenui=eceftoenu(telposecef[i,:],decayecefi)
        venui=eceftoenu_vector(telposecef[i,:],vecefi)
        intpoint1=fullint1[0:teli.sum(),:]
        intpoint2=fullint2[0:teli.sum(),:]

        fullint1=fullint1[teli.sum():,:]
        fullint2=fullint2[teli.sum():,:]


        #OUTSIDE UP SPHERE
        maxh=intpoint2[:,2].copy()
        telmaxangle=teltheta[2*i]+teltheta[2*i+1]
        endfov=np.sin(telmaxangle)*r[teli]
        maxh[maxh>endfov]=endfov[maxh>endfov]
        outside_up=(decayenui[:,2]>maxh) #Decay after going through FoV -> DISCARD
        index = np.arange(len(id))[teli][outside_up] #for outside

        id[index]=id[index]/code[i]
        print(teli.sum(), 'Start')
        print(outside_up.sum(), 'Decay Above FoV, remaining: ',teli.sum()-outside_up.sum())
    
        #INSIDE SPHERE
        inside=(decayenui[:,2]<maxh) & (decayenui[:,2]>intpoint1[:,2])
        intpoint2insideecef=enutoecef(telposecef[i,:],intpoint2[inside,:])
        altintpoint2=altitude_from_ecef(intpoint2insideecef)/1000
        densitydec=atmdensity(altdeci[inside]) #height in km
        density2=atmdensity(altintpoint2)     #height in km
        densityaverageinside=(densitydec+density2)/2
        distinside=np.linalg.norm(intpoint2[inside,:]-decayenui[inside,:],axis=1)
        gramminside=(distinside*100)*densityaverageinside #distance to cm
        accepted=(gramminside>-np.inf)  #initialize mask
        index = np.arange(teli.sum())[inside][accepted] #for outside
        acceptedofi=np.full_like(inside,False,dtype='bool')
        acceptedofi[index]=True


        gramm2=np.zeros(accepted.sum())
        distaccept=distinside[accepted]
        thetatelsup=teltheta[2*i]+telangle
        thetatelinf=teltheta[2*i]-telangle

        minpctinsidefov=np.full_like(distaccept,False,dtype='bool')

        for j in range(accepted.sum()):#accepted.sum()
            distintervals=np.arange(0,distaccept[j]+diststep,diststep)
            intervalscoords = decayenui[acceptedofi,:][j,:][np.newaxis, :] + venui[acceptedofi,:][j,:][np.newaxis, :] * distintervals[:, np.newaxis]            
            intervalsecef=enutoecef(telposecef[i,:],intervalscoords)
            altintervals=altitude_from_ecef(intervalsecef)/1000

            rhointervals=atmdensity(altintervals)
            rhointaverage=(rhointervals[:-1]+rhointervals[1:])/2
            grammageintervals=100*diststep*rhointaverage #one 100 to change to cm, the other because 100m per dist step. #We consider the max density in each interval
            cumgrammageintervals=np.cumsum(grammageintervals)
            gramm2[j]=np.sum(grammageintervals)   #This grammage is very good approximation of slant_depth calculated one, and always overestimates it
            coordintervals=intervalscoords[:-1,:]
            vectorintervals=coordintervals/np.linalg.norm(coordintervals,axis=1,keepdims=True)
            cosdphi=np.dot(vectorintervals[:,0:2],eyevector[i,0:2])   #azimuth angle difference with center of telescope
            theta=np.arccos(np.sqrt(vectorintervals[:,0]**2+vectorintervals[:,1]**2))*np.sign(vectorintervals[:,2]) #elevation angle of intersection vector 
            infov=(cosdphi>=np.cos(exacttelangle*5+telangle)) &  (theta<=thetatelsup) &  (theta>=thetatelinf)
            if np.sum(infov)<=1:
                continue
            enterfovgramm=(cumgrammageintervals[infov])[0]
            exitfovgramm=(cumgrammageintervals[infov])[-1]
            gramsteps=calculate_grammage_end(energy[acceptedofi][j],step)
            stepsinfov=np.sum((gramsteps>enterfovgramm)&(gramsteps<exitfovgramm))
            minpctinsidefov[j]=(stepsinfov>=np.ceil(minshowerpct/step))



        print(inside.sum(), 'Decay inside sphere.')
        accepted[accepted]=minpctinsidefov
        index = np.arange(len(id))[teli][inside][~accepted] #for outside
        id[index]=id[index]/code[i]

        print(minpctinsidefov.sum(), f' develop at least {minshowerpct}% inside Field of View. Rest are rejected.')
        #ASSIGN NEW IDS, DO OUTSIDE DOWN SPHERE



        
        #OUTSIDE DOWN SPHERE
        outside_down=(decayenui[:,2]<intpoint1[:,2])

        intpoint1ecef=enutoecef(telposecef[i,:],intpoint1[outside_down,:])
        altintpoint1=altitude_from_ecef(intpoint1ecef)/1000

        distout=np.linalg.norm(decayenui[outside_down,:]-intpoint1[outside_down,:],axis=1)
        densitydec=atmdensity(altdeci[outside_down]) #height in km
        density1=atmdensity(altintpoint1)     #height in km
        densityaverageout=(densitydec+density1)/2
        grammout=(distout*100)*densityaverageout #distance to cm
        accepted=np.full_like(grammout,True,dtype='bool')
        index = np.arange(teli.sum())[outside_down][accepted] #for outside
        acceptedofi=np.full_like(outside_down,False,dtype='bool')
        acceptedofi[index]=True        
        gramm2=np.zeros(accepted.sum())
        distaccept=np.linalg.norm(intpoint2[acceptedofi,:]-intpoint1[acceptedofi,:],axis=1)
        thetatelsup=teltheta[2*i]+telangle
        thetatelinf=teltheta[2*i]-telangle
        minpctinsidefov=np.full_like(distaccept,False,dtype='bool')    

        distout=np.linalg.norm(decayenui[outside_down,:]-intpoint1[outside_down,:],axis=1)
        

        for j in range(accepted.sum()):#accepted.sum()

            startgramm=slant_depth(altdeci[acceptedofi][j],altintpoint1[accepted][j],betai[acceptedofi][j])[0]

            distintervals=np.arange(distout[accepted][j],distaccept[j]+distout[accepted][j]+diststep,diststep)

            intervalscoords = decayenui[acceptedofi,:][j,:][np.newaxis, :] + venui[acceptedofi,:][j,:][np.newaxis, :] * distintervals[:, np.newaxis]            
            intervalsecef=enutoecef(telposecef[i,:],intervalscoords)
            altintervals=altitude_from_ecef(intervalsecef)/1000


            rhointervals=atmdensity(altintervals)
            rhointaverage=(rhointervals[:-1]+rhointervals[1:])/2
            grammageintervals=100*diststep*rhointaverage #one 100 to change to cm, the other because 100m per dist step. #We consider the max density in each interval
            cumgrammageintervals=startgramm+np.cumsum(grammageintervals)
            gramsteps=calculate_grammage_end(energy[acceptedofi][j],step)

            shower=(cumgrammageintervals<gramsteps[-1])
            coordintervals=intervalscoords[:-1,:][shower,:]
            vectorintervals=coordintervals/np.linalg.norm(coordintervals,axis=1,keepdims=True)
            cosdphi=np.dot(vectorintervals[:,0:2],eyevector[i,0:2])   #azimuth angle difference with center of telescope
            theta=np.arccos(np.sqrt(vectorintervals[:,0]**2+vectorintervals[:,1]**2))*np.sign(vectorintervals[:,2]) #elevation angle of intersection vector 
            infov=(cosdphi>=np.cos(exacttelangle*5+telangle)) &  (theta<=thetatelsup) &  (theta>=thetatelinf)
            if np.sum(infov)<=1:
                continue
            enterfovgramm=(cumgrammageintervals[shower][infov])[0]
            exitfovgramm=(cumgrammageintervals[shower][infov])[-1]
            stepsinfov=np.sum((gramsteps>enterfovgramm)&(gramsteps<exitfovgramm))
            minpctinsidefov[j]=(stepsinfov>=np.ceil(minshowerpct/step))

        print(outside_down.sum(), ' Decay before entering sphere.')
        print(minpctinsidefov.sum(), f' have at least {minshowerpct}% inside Field of View. Rest are rejected.')
       
        accepted[accepted]=minpctinsidefov

        index = np.arange(len(id))[teli][outside_down][~accepted] #for outside
        id[index]=id[index]/code[i]
        print('TOTAL NUMBER OF EVENTS ACCEPTED: \n',(id%code[i]==0).sum())

    return id