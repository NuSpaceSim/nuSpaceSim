import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.coordinates as coordin
from scipy.stats import norm
from scipy.spatial.transform import Rotation as R
from numpy.polynomial import Polynomial
import scipy.integrate
import matplotlib.cm as cm
import timeit

telangle=np.radians(15.5) #Increase FoV to 31x31 to remove any edge effects
exacttelangle=np.radians(15)

massTau = 1.77686  # GeV/c^2
mean_Tau_life = 2.903e-13  # seconds
inv_mean_Tau_life=1/mean_Tau_life
earth_radius = 6371.0e3  # in meters
c = 2.9979246e8

# Los Leones (LL)
LL = np.array([459208.3, 6071871.5, 1416.2])
LLlat=-35.495875577412704
LLlong=-69.44975612807455
LLheight=1416.2
LLang=np.radians(330-360)
LLphi = LLang+np.radians([15.01, 44.89, 75.00, 104.97, 134.99, 164.92]) #angle for perfect planes 15.045
LLelev = np.radians([15.65, 15.92, 15.90, 16.07, 15.95, 15.82])
LLphitot=[np.min(LLphi)-telangle,np.max(LLphi)+telangle]
LLthetatot=[np.min(LLelev)-telangle,np.max(LLelev)+telangle]

# Los Morados (LM)
LM = np.array([498903.7, 6094570.2, 1416.4])
LMlat=-35.29203965311884
LMlong=-69.01206472177884
LMheight=1416.4
LMang=np.radians(60)
LMphi = LMang+np.radians([14.86, 45.12, 75.04, 105.01, 134.79, 165.02])
LMelev = np.radians([15.96, 15.87, 15.81, 15.89, 15.97, 16.05])
LMphitot=[np.min(LMphi)-telangle,np.max(LMphi)+telangle]
LMthetatot=[np.min(LMelev)-telangle,np.max(LMelev)+telangle]

# Loma Amarilla (LA)
LA = np.array([480743.1, 6134058.4, 1476.7])
LAlat=-34.93578323621537
LAlong=-69.2108670167868
LAheight=1476.7
LAang=np.radians(188-360)
LAphi = LAang+np.radians([14.67, 44.98, 75.05, 105.37, 134.85, 164.91])
LAelev = np.radians([16.34, 16.10, 16.13, 15.79, 16.03, 15.75])
LAphitot=[np.min(LAphi)-telangle,np.max(LAphi)+telangle]
LAthetatot=[np.min(LAelev)-telangle,np.max(LAelev)+telangle]

# Coihueco (CO)
CO = np.array([445343.8, 6114140.0, 1712.3])
COlat=-35.114090954807736
COlong=-69.59979995461899
COheight=1712.3
COang=np.radians(243.0219-360)
COphi = COang+np.radians([14.88, 44.92, 74.93, 105.04, 134.82, 164.98])
COelev = np.radians([16.03, 16.14, 16.03, 16.20, 16.03, 16.12])
COphitot=[np.min(COphi)-telangle,np.max(COphi)+telangle]
COthetatot=[np.min(COelev)-telangle,np.max(COelev)+telangle]
##print(np.degrees(LLphitot),np.degrees(LMphitot),np.degrees(LAphitot),np.degrees(COphitot))

#Central phi and elevation
telphi=[np.mean(LLphitot),np.mean(LLphitot)-LLphitot[0],np.mean(LMphitot),np.mean(LMphitot)-LMphitot[0],np.mean(LAphitot),np.mean(LAphitot)-LAphitot[0],np.mean(COphitot),np.mean(COphitot)-COphitot[0]]
teltheta=[np.mean(LLthetatot),np.mean(LLthetatot)-LLthetatot[0],np.mean(LMthetatot),np.mean(LMthetatot)-LMthetatot[0],np.mean(LAthetatot),np.mean(LAthetatot)-LAthetatot[0],np.mean(COthetatot),np.mean(COthetatot)-COthetatot[0]]
#print(np.degrees(telphi),np.degrees(teltheta))
# Extract easting and northing values
easting_values = [LL[0], LM[0], LA[0], CO[0]]
northing_values = [LL[1], LM[1], LA[1], CO[1]]

# Calculate the mean easting and northing
mean_easting = np.mean(easting_values)   # 0471049.725
mean_northing = np.mean(northing_values) # 6103660.025
mean_lat=np.mean([LLlat,LMlat,LAlat,COlat])#-35.20965731232134
mean_long=np.mean([LLlong,LMlong,LAlong,COlong])#-69.31807837556353
# In Lat Long this is 35.209657 S 69.318078W
telang=[LLang,LMang,LAang,COang]
h=1416

def latlongtoECEF(lat,long,height=h): #in degrees, height in m
    lat=np.radians(lat)
    long=np.radians(long)
    a=6378137+h  #earth major axis WGS84
    b=6356752.314245+h #minor axis
    f=1-b/a
    e2=1-b**2/a**2
    N=a/np.sqrt(1-e2/(1+1/(np.tan(lat))**2)) #https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    x=(N+height)*np.cos(lat)*np.cos(long)
    y=(N+height)*np.cos(lat)*np.sin(long)
    z=(b**2/a**2*N+height)*np.sin(lat)
    coords=np.column_stack((x,y,z)).flatten()
    return coords
LLecef=latlongtoECEF(LLlat,LLlong,LLheight)
LMecef=latlongtoECEF(LMlat,LMlong,LMheight)
LAecef=latlongtoECEF(LAlat,LAlong,LAheight)
COecef=latlongtoECEF(COlat,COlong,COheight)
centerecef=latlongtoECEF(mean_lat,mean_long,h)
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

def radiusatlat(lat,height=h):
    lat=np.radians(lat)
    b=6356752.314245+h
    a=6378137.0+h
    f=1-b/a
    return a*(1-f*np.sin(lat)**2)
earth_radius_centerlat=radiusatlat(mean_lat,h)
def ecef_to_latlonh(ecefcoords):
    """
    Convert ECEF (X, Y, Z) coordinates to geodetic latitude, longitude, and height
    using Bowring's one-iteration method.
    
    Parameters:
        X, Y, Z : float
            ECEF coordinates in meters.
        a : float, optional
            Semi-major axis of the WGS84 ellipsoid (default: 6378137.0 m).
        b : float, optional
            Semi-minor axis of the WGS84 ellipsoid (default: 6356752.314245 m).
    
    Returns:
        lat : float
            Geodetic latitude in degrees.
        lon : float
            Longitude in degrees.
        h : float
            Height above the ellipsoid in meters.
    """
    ecefcoords = np.atleast_2d(ecefcoords)
    X=ecefcoords[:,0]
    Y=ecefcoords[:,1]
    Z=ecefcoords[:,2]
    a=6378137.0+h
    b=6356752.314245+h
    # Compute eccentricities
    e2 = 1 - (b**2 / a**2)  # First eccentricity squared
    ep2 = (a**2 / b**2) - 1  # Second eccentricity squared
    
    # Compute longitude directly
    lon = np.arctan2(Y, X)
    
    # Compute initial parameters
    p = np.sqrt(X**2 + Y**2)  # Distance from Z-axis
    T = Z * (a / b)
    
    # Initial latitude approximation (Bowring's formula)
    phi0 = np.arctan2(Z + ep2 * b * np.sin(np.arctan2(T, p))**3, p - e2 * a * np.cos(np.arctan2(T, p))**3)
    
    # Compute the radius of curvature in the prime vertical
    N = a / np.sqrt(1 - e2 * np.sin(phi0)**2)
    
    # Compute height
    height = (p / np.cos(phi0)) - N+h
    
    # Convert to degrees
    lat = np.degrees(phi0)
    lon = np.degrees(lon)
    if ecefcoords.shape[0] == 1:
        return lat[0], lon[0], height[0]
    return lat, lon, height
def ecef_to_latlonh_grok(ecefcoords):
    """
    Convert ECEF (X, Y, Z) coordinates to geodetic latitude, longitude, and height
    using Bowring's method with iteration for precisi`on.
    
    Parameters:
        ecefcoords : ndarray
            ECEF coordinates (X, Y, Z) in meters.
    
    Returns:
        lat : float or ndarray
            Geodetic latitude in degrees.
        lon : float or ndarray
            Longitude in degrees.
        height : float or ndarray
            Height above the scaled ellipsoid in meters.
    """
    ecefcoords = np.atleast_2d(ecefcoords)
    X, Y, Z = ecefcoords[:, 0], ecefcoords[:, 1], ecefcoords[:, 2]
    
    # Scaled ellipsoid parameters
    a = 6378137.0 + h  # Semi-major axis
    b = 6356752.314245 + h  # Semi-minor axis
    
    # Eccentricities
    e2 = 1 - (b**2 / a**2)  # First eccentricity squared
    ep2 = (a**2 / b**2) - 1  # Second eccentricity squared
    
    # Longitude
    lon = np.arctan2(Y, X)
    
    # Distance from Z-axis
    p = np.sqrt(X**2 + Y**2)
    
    # Initial latitude approximation
    theta = np.arctan2(Z * a, p * b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(Z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    
    # Iterate for precision
    for _ in range(2):  # Two iterations typically suffice
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        height = p / np.cos(lat) - N
        lat_new = np.arctan2(Z + ep2 * b * np.sin(lat)**3, p - e2 * a * np.cos(lat)**3)
        if np.allclose(lat, lat_new, rtol=1e-10):
            break
        lat = lat_new
    
    # Convert to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    
    if ecefcoords.shape[0] == 1:
        return lat[0], lon[0], height[0]
    print('CACA',height,height[height>1417])
    return lat, lon, height+h

def eceftoenumatrix(ecef_origin,lat=None,lon=None,h=None):
    """
    Convert ECEF coordinates to ENU coordinates with respect to a local origin.
    
    Parameters:
        ecef_point (tuple): ECEF coordinates of the point (X, Y, Z) in meters.
        ecef_origin (tuple): ECEF coordinates of the origin (X0, Y0, Z0) in meters.
    
    Returns:
        enu_coords (tuple): ENU coordinates (east, north, up) in meters.
    """
    # Unpack input coordinates
    if lat==None and lon==None and h==None:
        lat,lon,h=ecef_to_latlonh(ecef_origin)
    # Compute the transformation matrix from ECEF to ENU
    sin_lat = np.sin(np.radians(lat))
    cos_lat = np.cos(np.radians(lat))
    sin_lon = np.sin(np.radians(lon))
    cos_lon = np.cos(np.radians(lon))

    # Rotation matrix for ENU
    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])

    return R

def eceftoenu(origin,coords,lat=None,lon=None,h=None):  #Change from ECEF coordinates to local coords at origin
    mat=eceftoenumatrix(origin,lat=None,lon=None,h=None)
    return(coords-origin)@mat.T
def enutoecef(origin,coords,lat=None,lon=None,h=None): #change from local coordinates to ECEF. origin is ENU's origin coordinates in ECEF
    mat=np.linalg.inv(eceftoenumatrix(origin,lat=None,lon=None,h=None))
    return coords@mat.T +origin
def eceftoenu_vector(origin,coords,lat=None,lon=None,h=None):  #Change from ECEF coordinates to local coords at origin
    mat=eceftoenumatrix(origin,lat=None,lon=None,h=None)
    return(coords)@mat.T
def enutoecef_vector(origin,coords,lat=None,lon=None,h=None): #change from local coordinates to ECEF. origin is ENU's origin coordinates in ECEF
    mat=np.linalg.inv(eceftoenumatrix(origin,lat=None,lon=None,h=None))
    return coords@mat.T
#print(radiusatlat(COlat,COheight))
#telposenutest=eceftoenu(centerecef,telposecef,mean_lat,mean_long,h)
#print(telposecef,telposenutest,enutoecef(centerecef,telposenutest,mean_lat,mean_long,h))
#exit()
def WGS84ellipse(coord):
    a2=(6378137+h)**2  #earth major axis WGS84
    b2=(6356752.314245+h)**2 #minor axis
    return coord[:,0]**2/a2+coord[:,1]**2/a2+coord[:,2]**2/b2  #=1 is in ellipsoid, <1 inside >1 outside


def geodetic_to_utm(lat, lon,height):
    """
    Converts geodetic coordinates (lat, lon) to UTM (easting, northing, zone, hemisphere).
    Vectorized for multiple inputs.
    """
    # WGS84 constants
    b=6356752.314245+h
    a=6378137.0+h
    f=1-b/a
    e2 = 2*f - f**2  # first eccentricity squared
    k0 = 0.9996  # UTM scale factor
    
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

def ecef_to_utm(groundecef,height=h):
    """
    Converts ECEF coordinates to UTM (easting, northing, zone, hemisphere).
    Assumes all points are on Earth's surface.
    Handles multiple coordinates using vectorized operations.
    """
    lat, lon, height = ecef_to_latlonh_grok(groundecef)
    return geodetic_to_utm(lat, lon,height)

#There's always a 40m difference between radius+h at given latitude, and distance from point to WGS84 center
#This difference is 0 at lat=0 and at lat=90, and looks like it max is around 45deg (malargue is at lat=35)

# Create the center array
nametags=['Los Leones','Los Morados', 'Loma Amarilla', 'Coihueco']
center = [mean_easting, mean_northing,h]
LL=LL-center
LM=LM-center
LA=LA-center
CO=CO-center
telpos=np.array([LL,LM,LA,CO])
telposenu=eceftoenu(centerecef,telposecef)
teldistenu=np.linalg.norm(telpos,axis=1)

LLecefC=LLecef-centerecef
LMecefC=LMecef-centerecef
LAecefC=LAecef-centerecef
COecefC=COecef-centerecef
telposecefC=np.array([LLecefC,LMecefC,LAecefC,COecefC])

num_ang=10000
ang=np.linspace(0,np.pi,num_ang)
#print(CO)
def Rcutoff(lgE):

    p1 =  4.86267e+05
    p2 = -6.72442e+04
    p3 =  2.31169e+03

    return p1 + p2 * lgE + p3 * lgE * lgE	

def roundcalcradius(E):  #This uses only the sides of the telescope (not the forward direction) since that's the limitant
                         #distance, as the telescopes are pointing inwards, towards the center.
    rEnergy=Rcutoff(E)
    maxdist=0
    for i in range(4):
        centerenu=eceftoenu(telposecef[i],centerecef)
        side1=[(rEnergy*np.cos(telang[i])),(rEnergy*np.sin(telang[i])),0]
        side2=[(rEnergy*np.cos(telang[i]+np.pi)),(rEnergy*np.sin(telang[i]+np.pi)),0]
        maxcandidate=np.max([np.linalg.norm(side1-centerenu),np.linalg.norm(side2-centerenu)])
        maxdist=np.max([maxdist,maxcandidate])
    return maxdist
print(roundcalcradius(20))
exit()
def calcradius(E,num_ang,plotfig=False):
    rEnergy=Rcutoff(E)

    #LL
    xLLcirc=rEnergy*np.cos(ang+LLang)+LL[0]
    yLLcirc=rEnergy*np.sin(ang+LLang)+LL[1]
    xLLline = np.linspace(xLLcirc[-1], xLLcirc[0], num_ang)
    yLLline = np.linspace(yLLcirc[-1], yLLcirc[0], num_ang)
    xLL=np.concatenate((xLLcirc,xLLline))
    yLL=np.concatenate((yLLcirc,yLLline))
    if plotfig==True: 
        plt.plot(xLL,yLL,color='red')

    # LM
    xLMcirc=rEnergy*np.cos(ang+LMang)+LM[0]
    yLMcirc=rEnergy*np.sin(ang+LMang)+LM[1]
    xLMline = np.linspace(xLMcirc[-1], xLMcirc[0], num_ang)
    yLMline = np.linspace(yLMcirc[-1], yLMcirc[0], num_ang)
    xLM=np.concatenate((xLMcirc,xLMline))
    yLM=np.concatenate((yLMcirc,yLMline))
    if plotfig==True: 
        plt.plot(xLM,yLM,color='purple')

    # LA
    xLAcirc=rEnergy*np.cos(ang+LAang)+LA[0]
    yLAcirc=rEnergy*np.sin(ang+LAang)+LA[1]
    xLAline = np.linspace(xLAcirc[-1], xLAcirc[0], num_ang)
    yLAline = np.linspace(yLAcirc[-1], yLAcirc[0], num_ang)
    xLA=np.concatenate((xLAcirc,xLAline))
    yLA=np.concatenate((yLAcirc,yLAline))
    if plotfig==True: 
        plt.plot(xLA,yLA,color='yellow')

    # CO
    xCOcirc=rEnergy*np.cos(ang+COang)+CO[0]
    yCOcirc=rEnergy*np.sin(ang+COang)+CO[1]
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
    coord1ecef=enutoecef(centerecef,coord1,mean_lat,mean_long,h)
    coord2ecef=enutoecef(centerecef,coord2,mean_lat,mean_long,h)
    inearth1=WGS84ellipse(coord1ecef)
    inearth2=WGS84ellipse(coord2ecef)
    #round earth gives ~0.3% more events from horizontal showers at 10**19 eV
    maskfov=~((inearth1<1)&(inearth2<1))  #invalid trajectory since its not crossing the field of view
    print('En FoV ', maskfov.sum())
    coordecef=coord1ecef[maskfov]
    vcoordecef=(coord2ecef[maskfov]-coordecef)/ np.linalg.norm((coord2ecef[maskfov]-coordecef), axis=1, keepdims=True)

    groundecef, vcoordecefground, beta, azimuth= ground_xy(coordecef,vcoordecef)

    #coordg,vcoordg=ground_xy(coord,vcoord)
    #beta=np.arctan(vcoordg[:,2]/np.sqrt(vcoordg[:,0]**2+vcoordg[:,1]**2))
    mask=(beta<=maxang)#&(beta>=np.radians(10))  #CUIDADO CON ESTO
    print(mask.size,mask.sum())
    return groundecef[mask,:][0:n,:], vcoordecefground[mask,:][0:n,:],beta[mask][0:n],azimuth[mask][0:n]#coordg[mask,:][0:n,:], vcoordg[mask,:][0:n,:]

def ground_xy(coord,vcoord,height=h):   #Now included inside gen_points
    #c2t**2+c1t+c0=0 
    b2=(6356752.314245+h)**2  #+h because earth surface is defined 1416m above WGS84
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

def trajectory_inside_tel_sphere(lgE,coordecef,vcoordecef,telpos=telpos,telphi=telphi,teltheta=teltheta,extraradius=1.01,ntels=telpos.shape[0]):
    r=Rcutoff(lgE)*extraradius
    eyevector=gen_eye_vectors(telphi,teltheta)
    #eyevector=enutoecef_vector(centerecef,eyevector)
    ##print(eyevector[0,2])
    identifier=np.ones(coordecef[:,0].size)
    int1=[[]]
    int2=[[]]
    code=[2,3,5,7]
    color=['r','purple','yellow','blue']
    #f, axs=plt.subplots(2,2, sharex=True, sharey=True,figsize=(36,14),dpi=100)
    
    for i in range(ntels):#telpos.shape[0]
        #eyevector[i,:]=eceftoenu_vector(telposecef[i,:],eyevector[i,:])   #could remove this maybe
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
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect("equal")
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="r")
        v_intnorm=c_ground/r

        ax.scatter(v_intnorm[inplane][:,0],v_intnorm[inplane][:,1],np.zeros_like(inplane)[inplane],color='cyan',alpha=0.2,s=0.01,zorder=5,label='Ground Plane')
        ax.scatter(intvec2[insphere2][:,0],intvec2[insphere2][:,1],intvec2[insphere2][:,2],color='yellow',alpha=0.2,s=0.11,label='Sphere2')
        ax.scatter(intvec1[insphere1][:,0],intvec1[insphere1][:,1],intvec1[insphere1][:,2],color='green',alpha=0.2,s=0.11,label='Sphere1')

        phi_rotationplot = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * phiangle1)
        v_intersec1plot=phi_rotationplot.apply(v_intersec1)
        phi_rotationplot = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * (phiangle2-np.pi))
        v_intersec2plot=phi_rotationplot.apply(v_intersec2)

        ax.scatter(v_intersec1plot[inback1][:,0]/r,v_intersec1plot[inback1][:,1]/r,v_intersec1plot[inback1][:,2]/r,color='magenta',alpha=0.8,s=0.01,label='Back Plane')
        ax.scatter(v_intersec2plot[inback2][:,0]/r,v_intersec2plot[inback2][:,1]/r,v_intersec2plot[inback2][:,2]/r,color='magenta',marker='o',alpha=0.8,s=0.01)


        ax.legend(markerscale=32.)
        ax.view_init(azim=-25, elev=25)       
        #plt.figure()
        #phi2=np.arccos(cosdphi2[insphere2])*-np.sign(np.dot(intvec2[insphere2][:,0:2],[-eyevector[i,1],eyevector[i,0]]))
        #plt.hist(np.degrees(phi2),bins=40)
        plt.show()
        """


        inside=(identifier%code[i]==0)
        intfactor1=((-b[inside]-np.sqrt(D[inside]))*(1/a[inside])*(1/2))[:,np.newaxis]
        intfactor2=((-b[inside]+np.sqrt(D[inside]))*(1/a[inside])*(1/2))[:,np.newaxis]
        int1=np.append(int1,coordenu[inside]+intfactor1*vcoordenu[inside])
        int2=np.append(int2,coordenu[inside]+intfactor2*vcoordenu[inside])
    int1=int1.reshape(-1,3)
    int2=int2.reshape(-1,3)
    return identifier, int1,int2




def trajectory_inside_tel_sphere_plots(lgE,coordecef,vcoordecef,telpos=telpos,telphi=telphi,teltheta=teltheta,extraradius=1.01,ntels=telpos.shape[0]):
    r=Rcutoff(lgE)*extraradius
    eyevector=gen_eye_vectors(telphi,teltheta)
    #eyevector=enutoecef_vector(centerecef,eyevector)
    ##print(eyevector[0,2])
    identifier=np.ones(coordecef[:,0].size)
    int1=[[]]
    int2=[[]]
    code=[2,3,5,7]
    color=['r','purple','yellow','blue']
    #f, axs=plt.subplots(2,2, sharex=True, sharey=True,figsize=(36,14),dpi=100)
    
    for i in range(ntels):#telpos.shape[0]
        #eyevector[i,:]=eceftoenu_vector(telposecef[i,:],eyevector[i,:])   #could remove this maybe
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
        inplane=(dist<=r) & (dotprod>=0)
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
        intvec1=(coordenu[mask1]+u1*vcoordenu[mask1])/r
        cosdphi1=np.dot(intvec1[:,0:2],eyevector[i,0:2])/np.linalg.norm(intvec1[:,0:2],axis=1)/np.linalg.norm(eyevector[i,0:2])   #azimuth angle difference with center of telescope
        theta1=np.arccos(np.sqrt(intvec1[:,0]**2+intvec1[:,1]**2))*np.sign(intvec1[:,2]) #elevation angle of intersection vector 
        insphere1=(cosdphi1>=np.cos(exacttelangle*5+telangle)) &  (theta1<=thetatelsup) &  (theta1>=thetatelinf)   #exactangle*5+telangle bcs the extra 0.5 deg only for the last telescope (want to increase fov by 0.5, not 0.5*6)

        index = np.arange(len(a))[mask1][insphere1]
        identifier[index]=identifier[index]*code[i]
        mask1[mask1]=~insphere1 # UNCHECK
        print(insphere1.sum(),'Sphere1, remaining: ',mask1.sum())

        #Intersection of second point (if first one wasn't inside) (this is the highest point)
        u2=((-b[mask1]+sqrtD[~insphere1])*(1/a[mask1])*(1/2))[:,np.newaxis]
        intvec2=(coordenu[mask1]+u2*vcoordenu[mask1])/r
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
        inback1=(dist<=r) & (dotprod>=np.cos(telangle)) 
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
        inback2=(dist<=r) & (dotprod>=np.cos(telangle))
        index = np.arange(len(a))[mask1][inback2]
        identifier[index]=identifier[index]*code[i]
        mask1[mask1]=~inback2
        print(inback2.sum(),'Back Plane2. End. Leftover: ',mask1.sum())
        print(f'Total Inside FoV for telescope {i} = ',np.count_nonzero(identifier%code[i]==0))

        # vectest=np.array([np.cos(phiangle2),np.sin(phiangle2),0])
        # vectest=vectest/np.linalg.norm(vectest)
        # vectestrot=phi_rotation.apply(vectest)
        # lintest=np.column_stack((np.linspace(-1e5,1e5,100),np.linspace(-1e5,1e5,100),np.zeros(100)))
        # lintestrot=R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * -(phiangle2)+np.pi).apply(lintest-telpos[i,:])+telpos[i,:]

        phi_rotationplot = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * (phiangle2-np.pi))
        v_intersec2plot=phi_rotationplot.apply(v_intersec2)
        phi_rotationplot = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * (phiangle1-np.pi))
        v_intersec1plot=phi_rotationplot.apply(v_intersec1)

        ax.plot_wireframe(x, y, z, color="r")
        v_intnorm=c_ground/r
        ax.scatter(v_intnorm[inplane][:,0],v_intnorm[inplane][:,1],np.zeros_like(inplane)[inplane],color='cyan',alpha=0.2,s=0.01,zorder=5,label='Ground Plane')
        ax.scatter(intvec2[insphere2][:,0],intvec2[insphere2][:,1],intvec2[insphere2][:,2],color='yellow',alpha=0.2,s=0.01,label='Sphere2')
        ax.scatter(intvec1[insphere1][:,0],intvec1[insphere1][:,1],intvec1[insphere1][:,2],color='green',alpha=0.2,s=0.01,label='Sphere1')
        
        ax.scatter(v_intersec1plot[inback1][:,0]/r,v_intersec1plot[inback1][:,1]/r,v_intersec1plot[inback1][:,2]/r,color='magenta',alpha=0.8,s=0.01,label='Back Plane')
        ax.scatter(v_intersec2plot[inback2][:,0]/r,v_intersec2plot[inback2][:,1]/r,v_intersec2plot[inback2][:,2]/r,color='magenta',marker='o',alpha=0.8,s=0.01)

        mask1[mask1]=~inback2
        #print(mask1.sum(),'Back Plane2')

        #CALCULATE INTERSECTION POINT WITH SPHERE
        # draw sphere
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect("equal")
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)


        #print(intvec1[insphere1][:,2])
        #ax.scatter(intvec1[~insphere1][:,0],intvec1[~insphere1][:,1],intvec1[~insphere1][:,2],color='k',alpha=0.5,s=0.01,zorder=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        #Intersection of second point (if first one wasn't inside)

        #print(mask1.sum(),'Sphere2')

        #dcheck1=np.linalg.norm(coord[mask1]+u1*vcoord[mask1]-telpos[i,:],axis=1)
        #dcheck2=np.linalg.norm(coord[mask1]+u2*vcoord[mask1]-telpos[i,:],axis=1)
        #   

        ax.legend(markerscale=32.)
        ax.view_init(azim=-25, elev=25)       
        plt.show()

        ##print(np.sum(masksph),np.sum(inplane),np.sum(mask2),'\n',np.size(masksph),np.size(inplane),np.size(mask2)) #check that masks work 

        #plot intersection in plane
        #plt.scatter(c_ground[~inplane][:,0],c_ground[~inplane][:,1],color='k',s=0.03,zorder=1,alpha=0.3)
        #plt.scatter(c_ground[inplane][:,0],c_ground[inplane][:,1],color=color[i],s=0.03,zorder=10,alpha=0.3)

        #c_intersec=check_intersec_plane(coordi,vcoordi,rot_angle,origin[i,:],eyevector[i],r,identifier[sign>=0],code[i])
        #conecoords=[1/2,1/3,np.sqrt(2)/2]    #Las coordenadas del vector que quiero rotar
        #Vector perpendicular en el suelo de la forma (-y,x) para rotar el zenith hacia abajo
        
        #
        # plt.figure()
        # plt.quiver(xt,yt, r*vectest[0], r*vectest[1],angles='xy', scale_units='xy', scale=1, color=color[i],alpha=0.5)

        # plt.plot(lintest[:,0],lintest[:,1],'.-',color=color[i],alpha=0.4)
        # plt.plot(lintestrot[:,0],lintestrot[:,1],'--',color=color[i])

        # plt.quiver(xt,yt, r*vectestrot[0], r*vectestrot[1],angles='xy', scale_units='xy', scale=1, color=color[i])
        # plt.savefig('test.png')

        #Plot of all 4 telescopes backplane
        """plt.subplot(2,2,i+1)
        plt.gca().set_aspect('equal')
        plt.xlim([-50000,50000])
        plt.ylim([-1000,25000])
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.title(f'x-z proyection of intersections in backplane for {nametags[i]}')
        plt.plot(r*np.cos(np.linspace(0,2*np.pi)),r*np.sin(np.linspace(0,2*np.pi)),'k-',label='Telescope FoV radius',alpha=1,linewidth=2)
        plt.scatter(v_intersec1[~inback1][:,0],v_intersec1[~inback1][:,2],color='k',label='Outside right',s=0.5,alpha=0.5,zorder=1)
        plt.scatter(v_intersec2[~inback2][:,0],v_intersec2[~inback2][:,2],color='red',label='Outside left',s=0.5,alpha=0.5,zorder=1)
        plt.scatter(v_intersec1[inback1][:,0],v_intersec1[inback1][:,2],color='cyan',label='Inside right',s=0.5,zorder=10)
        plt.scatter(v_intersec2[inback2][:,0],v_intersec2[inback2][:,2],color='green',label='Inside left',s=0.5,zorder=10)
        plt.legend(loc='upper center',markerscale=15.)
        plt.grid()"""
        
        """plt.subplot(2,2,1)
        plt.gca().set_aspect('equal')
        plt.title(f'x-z proyection of intersections in backplane for Los Leones, telangle={np.degrees(telangle)}')
        plt.xlim([-50000,50000])
        plt.ylim([-5000,35000])
        plt.xlabel('x (m)')
        plt.ylabel('z (m)')
        plt.plot(r*np.cos(np.linspace(0,2*np.pi)),r*np.sin(np.linspace(0,2*np.pi)),'k-',label='Telescope FoV radius',alpha=1,linewidth=2)
        plt.scatter(v_intersec1[~inback1][:,0],v_intersec1[~inback1][:,2],color='k',label='Outside right',s=0.5,alpha=0.5)
        plt.legend(loc='upper center',markerscale=15.)
        plt.subplot(2,2,2)
        plt.scatter(v_intersec1[inback1][:,0],v_intersec1[inback1][:,2],color='cyan',label='Inside right',s=0.5)
        plt.legend(loc='upper center',markerscale=15.)
        plt.subplot(2,2,3)
        plt.scatter(v_intersec2[~inback2][:,0],v_intersec2[~inback2][:,2],color='red',label='Outside left',s=0.5,alpha=0.5)
        plt.legend(loc='upper center',markerscale=15.)
        plt.subplot(2,2,4)
        plt.scatter(v_intersec2[inback2][:,0],v_intersec2[inback2][:,2],color='green',label='Inside left',s=0.5)


        plt.legend(loc='upper center',markerscale=15.)
        plt.grid()"""
        #identifier[sign>=0][inplane]=identifier[sign>=0][inplane]*code[i]

        
    #plt.savefig('xzproyection_backplaneintersectest.png')
    mask=np.full(identifier.shape,True)
    mask[identifier==1]=False
    return identifier,mask
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
        # #print(grammage[indexes])
        # plt.plot(grammage,RN)
        # for i in range(values.size):
        #     plt.axvline(x=grammage[indexes[i]],linestyle='--',color='k',label=f'{step*100}% steps'if i==0 else '',alpha=0.5)
        # plt.xlabel('Grammage g/cm2')
        # plt.ylabel('Profile (number of particles in shower)')
        # plt.title(f'Shower intervals logE={lgE}')
        # plt.grid()
        # plt.legend()
        # plt.xlim([0,1700])
        # plt.show()


        # g1=np.argmax(cumRN > stoppercentage/100*cumRN[-1])
        # g99=np.argmax(cumRN < (100-stoppercentage)/100*cumRN[-1])
        # RNmax=np.max(RN)
        # RNargmax=np.argmax(RN)
        # RNratio=np.array(RN/RNmax)
        # pos99=int(np.argmax(RNratio[RNargmax:]<stoppercentage/100)+RNargmax)
        # pos1=int(np.argmax(RNratio[0:RNargmax]>stoppercentage/100))
        # #print('Hola',grammage[g1],grammage[g99],grammage[pos1],grammage[pos99])
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

#Test for grammage aproximation
"""zin=3
zfin=12
angulo=np.radians(1)
gramm=slant_depth(zin,zfin,angulo)

densitydec=atmdensity(zin) #height in km
density2=atmdensity(zfin)     #height in km
densityaverageinside=(densitydec+density2)/2
distinside=(zfin-zin)/np.sin(angulo)*1000 #to m

gramminside=(distinside*100)*densityaverageinside #distance to cm
print(gramm[0],gramminside)"""
def path_length_tau_atm(z, beta_tr, Re=earth_radius_centerlat, xp=np):  #in m
    """
    From https://arxiv.org/pdf/1902.11287.pdf Eqn (11)
    """
    Resinb = Re * xp.sin(beta_tr)
    return xp.sqrt(Resinb**2 + (Re + z) ** 2 - Re**2) - Resinb

def altitude_along_path_length(s, beta_tr, Re=earth_radius_centerlat, xp=np): #in m
    """Derived by solving for z in path_length_tau_atm."""
    return xp.sqrt(s**2 + 2.0 * s * Re * xp.sin(beta_tr) + Re**2) - Re

def decay(groundecef,vecef,beta, lgE,earth_radius=earth_radius_centerlat, height=h):
    #get decay altitude
    tauEnergy=10**(lgE-9)
    tauLorentz = tauEnergy / massTau
    tauBeta = np.sqrt(1.0 - np.reciprocal(tauLorentz**2))
    #lgE = np.log10(data["tauEnergy"]) + 9
    u = np.random.uniform(0, 1, len(groundecef[:,0]))
    tDec = (-1.0 * tauLorentz /inv_mean_Tau_life) * np.log(u)

    lenDec = tDec * tauBeta * c

    decayecef=groundecef+lenDec[:,np.newaxis]*vecef

    #slant_depth(0,Zfirst[i],np.pi / 2 - beta[i])


    altDec = np.sqrt(
        earth_radius**2
        + lenDec**2
        + 2.0 *earth_radius * lenDec * np.sin(beta)
    )+height-earth_radius

    #print(decaycoord[:,2],altDec)
    #decaycoords=
    return decayecef,altDec/1000 #, lenDec

def decay_inside_fov(lgE,groundecef,vecef,beta,decayecef, id,fullint1,fullint2,height=h,extraradius=1.01,
 minshowerpct=1,ntels=telpos.shape[0],step=0.5, diststep=10, telphi=telphi,teltheta=teltheta,telangle=telangle):
    r=Rcutoff(lgE)*extraradius
    code=[2,3,5,7]
    #gramsteps=calculate_grammage_end(lgE,step)
    eyevector=gen_eye_vectors(telphi,teltheta)
    #latground=np.arctan(groundecef[:,2]/np.sqrt(groundecef[:,0]**2+groundecef[:,1]**2)) #spherical earth approx
    for i in range(ntels):#telpos.shape[0]
        teli=(id%code[i]==0)
        energy=lgE[teli]
        decayecefi=decayecef[teli]
        vecefi=vecef[teli]
        betai=beta[teli]
        decayenui=eceftoenu(telposecef[i,:],decayecefi)
        groundenui=eceftoenu(telposecef[i,:],groundecef[teli])
        venui=eceftoenu_vector(telposecef[i,:],vecefi)
        dist_to_decay=np.linalg.norm(decayenui-groundenui,axis=1)
        intpoint1=fullint1[0:teli.sum(),:]
        intpoint2=fullint2[0:teli.sum(),:]

        fullint1=fullint1[teli.sum():,:]
        fullint2=fullint2[teli.sum():,:]

        #PLOTS
        # fig = plt.figure(dpi=300)

        # ax = fig.add_subplot(projection='3d')
        # ax.set_aspect("equal")
        # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        # x = r*np.cos(u)*np.sin(v)
        # y = r*np.sin(u)*np.sin(v)
        # z = r*np.cos(v)
        # ax.plot_wireframe(x, y, z, color="r",linewidth=0.2)
        

        #OUTSIDE UP SPHERE
        #betacheck=np.arctan(vcoordi[:,2]/np.sqrt(vcoordi[:,0]**2+vcoordi[:,1]**2))
        #distdecay=np.linalg.norm(decayenui,axis=1)/1000
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
        heightdec=height + altitude_along_path_length(dist_to_decay[inside],betai[inside],radiusatlat(tellat[i],h))
        int2elev=np.arctan(intpoint2[inside,2]/np.sqrt(intpoint2[inside,0]**2+intpoint2[inside,1]**2))
        heightint2=height + altitude_along_path_length(np.linalg.norm(intpoint2[inside,:],axis=1),int2elev,radiusatlat(tellat[i],h))
        densitydec=atmdensity(heightdec/1000) #height in km
        density2=atmdensity(heightint2/1000)     #height in km
        densityaverageinside=(densitydec+density2)/2
        distinside=np.linalg.norm(intpoint2[inside,:]-decayenui[inside,:],axis=1)
        gramminside=(distinside*100)*densityaverageinside #distance to cm
        #accepted=(gramminside>gramsteps[int(np.ceil(minshowerpct/step))-1])
        accepted=(gramminside>-np.inf)
        index = np.arange(teli.sum())[inside][accepted] #for outside
        acceptedofi=np.full_like(inside,False,dtype='bool')
        acceptedofi[index]=True

        #ax.scatter(decaycoordi[inside,0],decaycoordi[inside,1],decaycoordi[inside,2],'o',color='g',s=1)

        gramm2=np.zeros(accepted.sum())
        #gramm3=np.zeros(accepted.sum())
            #CHECK IF AT LEAST x% INSIDE FoV
        distaccept=distinside[accepted]
        thetatelsup=teltheta[2*i]+telangle
        thetatelinf=teltheta[2*i]-telangle
        #beta=np.arctan(vcoordi[acceptedofi,2]/np.sqrt(vcoordi[acceptedofi,0]**2+vcoordi[acceptedofi,1]**2))

        #variables for plots
        distenter=np.linalg.norm(decayenui[acceptedofi,:]-intpoint1[acceptedofi,:],axis=1)
        xt=telpos[i,0]
        yt=telpos[i,1]
        zt=telpos[i,2]

        #plt.figure(dpi=200)
        distdecayaccept=dist_to_decay[acceptedofi]
        minpctinsidefov=np.full_like(distaccept,False,dtype='bool')

        for j in range(accepted.sum()):#accepted.sum()
            distintervals=np.arange(0,distaccept[j]+diststep,diststep)
            #hintervals=decaycoordi[acceptedofi,2][j]+altitude_along_path_length(distintervals,beta[j])  
            hintervals=height+altitude_along_path_length(distdecayaccept[j]+distintervals,betai[acceptedofi][j],radiusatlat(tellat[i],h)) #height above sea level for atm, so we calculate how far up it goes with respect to ground (elevated ground) and then add ground's altitude a.s.l.
            #hintervals2=decayenui[acceptedofi,2][j]+distintervals*np.sin(betai[acceptedofi][j])

            rhointervals=atmdensity(hintervals/1000)
            rhointaverage=(rhointervals[:-1]+rhointervals[1:])/2
            grammageintervals=100*diststep*rhointaverage #one 100 to change to cm, the other because 100m per dist step. #We consider the max density in each interval
            cumgrammageintervals=np.cumsum(grammageintervals)
            gramm2[j]=np.sum(grammageintervals)   #This grammage is very good approximation of slant_depth calculated one, and always overestimates it
            #gramm3[j]=slant_depth(decaycoordi[acceptedofi,2][j]/1000,intpoint2[acceptedofi,2][j]/1000,beta[j])[0]
            coordintervals=decayenui[acceptedofi,:][j][np.newaxis,:]+distintervals[:-1,np.newaxis]*venui[acceptedofi,:][j][np.newaxis,:]
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
#            if minpctinsidefov[j]==True:
        """        
            #PLOTS
            if minpctinsidefov[j]==True:
                phiangle1=telphi[2*i]-telphi[2*i+1]
                rotaxis = np.array([0,0,1])
                phi_rotation = R.from_rotvec(rotaxis/np.linalg.norm(rotaxis) * -phiangle1)
                vecplot=decaycoordi[acceptedofi,:][j]-telpos[i,:]
                vecplot = phi_rotation.apply(vecplot)
                dirplot=phi_rotation.apply(vcoordi[acceptedofi,:][j])
                dirplot=40000*dirplot/np.linalg.norm(dirplot)
                distplot=np.linalg.norm(vecplot)
                #plt.plot([vecplot[0],dirplot[0]],[vecplot[2],dirplot[2]],'-',linewidth=0.01,alpha=0.5,zorder=1)
                plt.quiver(vecplot[0],vecplot[2],dirplot[0],dirplot[2],angles='xy', scale_units='xy', scale=1,width=0.0015,color='g',alpha=0.5,zorder=1)
                plt.scatter(vecplot[0],vecplot[2],s=0.5,color='b',zorder=10)
        
        plt.axhline(y=r/2,color='k',linestyle='--',label='Maximum FoV height')
        angplot=np.linspace(0,np.pi)
        plt.plot(np.cos(angplot)*r,np.sin(angplot)*r,label='Sphere radius')
        plt.xlabel('Telescope plane (m)')
        plt.ylabel('z (m)')
        plt.grid()

        #plt.ylim([-1000,1.1*r/2])
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1))
        plt.title(f'Decay positions on z-telescope plane of "inside" trajec, n={minpctinsidefov.sum()}')
        plt.show()

        """




        print(inside.sum(), 'Decay inside sphere.\n',accepted.sum(),f'develop more than {minshowerpct}% inside, accepted. Rest rejected')        
        accepted[accepted]=minpctinsidefov
        index = np.arange(len(id))[teli][inside][~accepted] #for outside
        id[index]=id[index]/code[i]

        print(minpctinsidefov.sum(), f' have at least {minshowerpct}% inside Field of View. Rest are rejected.')
        #ASSIGN NEW IDS, DO OUTSIDE DOWN SPHERE



        
        #OUTSIDE DOWN SPHERE
        outside_down=(decayenui[:,2]<intpoint1[:,2])


        heightdec=height + altitude_along_path_length(dist_to_decay[outside_down],betai[outside_down],radiusatlat(tellat[i],h))
        distout=np.linalg.norm(decayenui[outside_down,:]-intpoint1[outside_down,:],axis=1)
        heightint1=height + altitude_along_path_length(dist_to_decay[outside_down]+distout,betai[outside_down],radiusatlat(tellat[i],h))
        densitydec=atmdensity(heightdec/1000) #height in km
        density1=atmdensity(heightint1/1000)     #height in km
        densityaverageout=(densitydec+density1)/2
        grammout=(distout*100)*densityaverageout #distance to cm
        #accepted=~(grammout>gramsteps[int(np.floor((100-minshowerpct)/step))-1])
        accepted=np.full_like(grammout,True,dtype='bool')
        index = np.arange(teli.sum())[outside_down][accepted] #for outside
        acceptedofi=np.full_like(outside_down,False,dtype='bool')
        acceptedofi[index]=True        
        gramm2=np.zeros(accepted.sum())
        distaccept=np.linalg.norm(intpoint2[acceptedofi,:]-intpoint1[acceptedofi,:],axis=1)
        thetatelsup=teltheta[2*i]+telangle
        thetatelinf=teltheta[2*i]-telangle
        minpctinsidefov=np.full_like(distaccept,False,dtype='bool')    
        #beta=np.arctan(vcoordi[acceptedofi,2]/np.sqrt(vcoordi[acceptedofi,0]**2+vcoordi[acceptedofi,1]**2))
        for j in range(accepted.sum()):#accepted.sum()
            startgramm=slant_depth(heightdec[accepted][j]/1000,heightint1[accepted][j]/1000,betai[acceptedofi][j])[0]

            """ Dont use this because although its more exact since it uses more exact earth radius, this one should be faster, simpler and underestimates it by a small amount anyway
            distintervals=np.arange(0,distout[accepted][j],diststep)
            hintervals=h+altitude_along_path_length(dist_to_decay[acceptedofi][j]+distintervals,betai[acceptedofi][j],radiusatlat(tellat[i],h)) #height above sea level for atm, so we calculate how far up it goes with respect to ground (elevated ground) and then add ground's altitude a.s.l.
            hfinal=h+altitude_along_path_length(dist_to_decay[acceptedofi][j]+distout[accepted][j],betai[acceptedofi][j],radiusatlat(tellat[i],h))
            heightoldmeth=intpoint1[acceptedofi,2][j]-groundenui[acceptedofi,2][j]
            print(np.degrees(betai[acceptedofi][j]),heightdec[accepted][j],heightint1[accepted][j],hintervals[-2],hintervals[-1],hfinal,heightoldmeth)
            print('old gramm,', startgramm)
            rhointervals=atmdensity(hintervals/1000)
            rhointaverage=(rhointervals[:-1]+rhointervals[1:])/2
            grammageintervals=100*diststep*rhointaverage #one 100 to change to cm, the other because 100m per dist step. #We consider the max density in each interval
            grammout[j]=np.sum(grammageintervals)+100*(distout[accepted][j]-distintervals[-1])*rhointervals[-1]
            print('new gramm',grammout[j])"""

            distintervals=np.arange(0,distaccept[j]+diststep,diststep)
            #hintervals=decaycoordi[acceptedofi,2][j]+altitude_along_path_length(distintervals,beta[j])  
            hintervals=height+altitude_along_path_length(dist_to_decay[acceptedofi][j]+distout[accepted][j]+distintervals,betai[acceptedofi][j],radiusatlat(tellat[i],h)) #height above sea level for atm, so we calculate how far up it goes with respect to ground (elevated ground) and then add ground's altitude a.s.l.
            rhointervals=atmdensity(hintervals/1000)
            rhointaverage=(rhointervals[:-1]+rhointervals[1:])/2
            grammageintervals=100*diststep*rhointaverage #one 100 to change to cm, the other because 100m per dist step. #We consider the max density in each interval
            cumgrammageintervals=startgramm+np.cumsum(grammageintervals)
            gramsteps=calculate_grammage_end(energy[acceptedofi][j],step)
            shower=(cumgrammageintervals<gramsteps[-1])
            #gramm2[j]=np.sum(grammageintervals)   #This grammage is very good approximation of slant_depth calculated one, and always overestimates it
            #gramm3[j]=slant_depth(decaycoordi[acceptedofi,2][j]/1000,intpoint2[acceptedofi,2][j]/1000,beta[j])[0]
            coordintervals=decayenui[acceptedofi,:][j][np.newaxis,:]+(distout[accepted][j]+distintervals[:-1][shower,np.newaxis])*venui[acceptedofi,:][j][np.newaxis,:]
            vectorintervals=coordintervals/np.linalg.norm(coordintervals,axis=1,keepdims=True)
            cosdphi=np.dot(vectorintervals[:,0:2],eyevector[i,0:2])   #azimuth angle difference with center of telescope
            theta=np.arccos(np.sqrt(vectorintervals[:,0]**2+vectorintervals[:,1]**2))*np.sign(vectorintervals[:,2]) #elevation angle of intersection vector 
            infov=(cosdphi>=np.cos(exacttelangle*5+telangle)) &  (theta<=thetatelsup) &  (theta>=thetatelinf)
            if np.sum(infov)<=1:
                #minpctinsidefov[j]=False  not necessary, its already False by default
                continue
            enterfovgramm=(cumgrammageintervals[shower][infov])[0]
            exitfovgramm=(cumgrammageintervals[shower][infov])[-1]
            stepsinfov=np.sum((gramsteps>enterfovgramm)&(gramsteps<exitfovgramm))
            minpctinsidefov[j]=(stepsinfov>=np.ceil(minshowerpct/step))



 
 
            #     c=np.random.rand(3,)
            #     distout=np.linalg.norm(decaycoordi[acceptedofi,:][j]-intpoint1[acceptedofi,:][j])

            #     plotintervalsin=np.linspace(0,distaccept[j]+distout,100)
            #     plotintervals1=np.linspace(-8000,0,50)
            #     vecintervalsplot=decaycoordi[acceptedofi,:][j][np.newaxis,:]+plotintervals1[:-1,np.newaxis]*vcoordi[acceptedofi,:][j][np.newaxis,:]-telpos[i,:]
            #     ax.scatter(decaycoordi[acceptedofi,0][j]-xt,decaycoordi[acceptedofi,1][j]-yt,decaycoordi[acceptedofi,2][j]-zt,'+',color='g',s=0.2)
            #     ax.plot(vecintervalsplot[:,0],vecintervalsplot[:,1],vecintervalsplot[:,2],alpha=0.6,color='grey',linewidth=0.8)
            #     vecintervalsplot=decaycoordi[acceptedofi,:][j][np.newaxis,:]+plotintervalsin[:-1,np.newaxis]*vcoordi[acceptedofi,:][j][np.newaxis,:]-telpos[i,:]

            #     ax.plot(vecintervalsplot[:,0],vecintervalsplot[:,1],vecintervalsplot[:,2],color=c,linewidth=0.8)


        print(outside_down.sum(), ' Decay before entering sphere. \n',(accepted).sum(), f' Decay close enough to sphere to have {minshowerpct}% of shower inside')
        print(minpctinsidefov.sum(), f' have at least {minshowerpct}% inside Field of View. Rest are rejected.')
       
        #decaycoordi
        accepted[accepted]=minpctinsidefov

        index = np.arange(len(id))[teli][outside_down][~accepted] #for outside
        id[index]=id[index]/code[i]
        print('TOTAL NUMBER OF EVENTS ACCEPTED: \n',(id%code[i]==0).sum())



        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # lims=[-30000,30000]
        # ax.set_xlim(lims)
        # ax.set_ylim(lims)
        # ax.set_zlim(lims)
        # ax.set_title('Decays before entering sphere')

        # ax.legend(markerscale=32.)
        # ax.view_init(azim=-25, elev=25)       
        # plt.show()
    return id
#NEXT STEP -> CHECK FoV event by event -> Check angles  
"""
lgE=18
n=int(1e7)
maxangle=np.radians(90)
radius=roundcalcradius(lgE)
groundecef, vecef,beta=gen_points(n,radius,maxang=maxangle)
groundenu=eceftoenu(centerecef,groundecef)
#venu=eceftoenu_vector( centerecef,vecef)
id,int1,int2=trajectory_inside_tel_sphere(lgE,groundecef,vecef,ntels=1)
decayecef,altDec=decay(groundecef,vecef, lgE,earth_radius=earth_radius_centerlat)

idfinal=decay_inside_fov(lgE,groundecef,vecef,beta,decayecef, id,int1,int2,ntels=1,diststep=200)
distdecay=np.linalg.norm(decayecef-groundecef,axis=1)/1000

"""


""" PLOTS
r=radius
dist_to_center=np.linalg.norm(groundenu,axis=1)
xproj=dist_to_center/np.sqrt(1+groundenu[:,1]**2/groundenu[:,0]**2)*np.sign(groundenu[:,0])
yproj=xproj*groundenu[:,1]/groundenu[:,0]

n_rings=8
r_interm=np.sqrt(np.linspace(0,r**2,n_rings))
r_interm = np.append(r_interm, np.inf)
f, axs=plt.subplots(2,4,sharex=False,sharey=False, figsize=(30,18),dpi=50)
cosplot=np.linspace(0,np.pi/2)
for i in range(n_rings):
    if i==n_rings-1:
        maski=[]
        plt.subplot(2,4,i+1)
        maski=(dist_to_center<r_interm[i+1])&(dist_to_center>=r_interm[i])
        bars, bins, _ =plt.hist(beta[maski],bins=50,density=True,label='Histogram')
        print(bins)
        plt.plot(cosplot,bars[15]/(np.cos(bins[15]+(bins[1]-bins[0])/2)*(1-np.sin(bins[15]+(bins[1]-bins[0])/2)))*np.cos(cosplot)*(1-np.sin(cosplot)),label='cos(ang)(1-sin(ang)) function')
        plt.title(f'Emergence Angle for {r_interm[i]/r:.2f}R<r<{r_interm[i+1]/r:.2f}R')
        plt.xlabel('Angle (rad)')
        #plt.ylim([0,2])
        plt.legend()
    else:
        maski=[]
        plt.subplot(2,4,i+1)
        maski=(dist_to_center<r_interm[i+1])&(dist_to_center>=r_interm[i])
        bars, _, _ =plt.hist(beta[maski],bins=50,density=True,label='Histogram')
        plt.plot(cosplot,2*np.cos(cosplot)*np.sin(cosplot),label='cos(ang)sin(ang) function')
        plt.title(f'Emergence Angle for {r_interm[i]/r:.2f}R<r<{r_interm[i+1]/r:.2f}R')
        plt.xlabel('Angle (rad)')
        #plt.ylim([0,2])

        plt.legend()
#plt.show()
f, axs=plt.subplots(2,5,sharex=False,sharey=False, figsize=(30,18),dpi=50)

for i in range(n_rings):
    if i==n_rings-1:
        maski=[]
        plt.subplot(2,4,i+1)
        maski=(dist_to_center<r_interm[i+1])&(dist_to_center>=r_interm[i])
        #bars, _, _ =plt.hist(np.cos(beta[maski])**2,bins=50,density=True,label='Histogram')
        #plt.plot(np.cos(cosplot)**2,2*np.cos(cosplot)*(1-np.sin(cosplot)),label='cos(ang)(1-sin(ang)) function')
        plt.title(f'Histogram of cos**2 (angle) for {r_interm[i]/r:.2f}R<r<{r_interm[i+1]/r:.2f}R')
        plt.xlabel('Cos^2(ang)')
        #plt.ylim([0,2])
        plt.legend()
    else:
        maski=[]
        plt.subplot(2,5,i+1)
        maski=(dist_to_center<r_interm[i+1])&(dist_to_center>=r_interm[i])
        bars, _, _ =plt.hist(np.cos(beta[maski])**2,bins=50,density=True,label='Histogram')
        #plt.plot(np.cos(cosplot)**2,2*np.cos(cosplot)*np.sin(cosplot),label='cos(ang)sin(ang) function')
        plt.title(f'Histogram of cos**2 (angle) for {r_interm[i]/r:.2f}R<r<{r_interm[i+1]/r:.2f}R')
        plt.xlabel('Cos^2(ang)')
        #plt.ylim([0,2])

        plt.legend()
#plt.show()

print('Total fuera ',np.sum((dist_to_center>r)))
print('Total dentro ',np.sum((dist_to_center<=r)))
plt.figure(figsize=(12,12))
r_bins = np.linspace(-r*3,r*3, 70)
ang=np.linspace(0,2*np.pi,1000)
plt.plot(r*np.cos(ang),r*np.sin(ang),'r-',label='Radius of sphere')
plt.hist2d(xproj, yproj, bins=(r_bins, r_bins), cmap=plt.cm.jet)#norm=mpl.colors.LogNorm(),
plt.xlim([-r*3,r*3])
plt.ylim([-r*3,r*3])
plt.legend()
#plt.plot(xin,yin,'g.',alpha=0.8,markersize=0.5)

plt.plot(LL[0],LL[1],color='red',marker='v',markersize=10)
plt.plot(LM[0],LM[1],color='purple',marker='>',markersize=10)
plt.plot(LA[0],LA[1],color='yellow',marker='^',markersize=10)
plt.plot(CO[0],CO[1],color='blue',marker='<',markersize=10)
plt.gca().set_aspect('equal')
plt.colorbar()
#plt.show()

dist_inplane=np.sqrt(groundenu[:,0]**2+groundenu[:,1]**2)
dist_inplane=dist_inplane[dist_inplane<200000]
print(np.max(dist_inplane))
plt.figure()
bars,bins, patches=plt.hist(dist_inplane,bins=70)
plt.xlabel('Distance from center (m)')
plt.ylabel('Core positions')
plt.title('Core position distance to center')
plt.grid()

plt.figure()
binsmean=((bins[1]-bins[0])/2+bins)[:-1]
areaofbin=np.pi*(bins[1:]**2-bins[:-1]**2)
barsbyarea=bars/areaofbin
plt.bar(binsmean,barsbyarea,width=(bins[1]-bins[0]),align='center')
plt.xlabel('Distance from center (m)')
plt.ylabel('Core positions')
plt.title('Core position distance to center normalised by area')
plt.grid()
plt.show()
earthrad=radiusatlat(mean_lat,h)
maxdist=np.sqrt((earthrad+radius)**2-earthrad**2)
print(maxdist)


#exit()"""
"""
#plt.figure(figsize=(16,12),dpi=100)
#plt.gca().set_aspect('equal')

#print(idnew[idnew!=1].size)
#mask=(identifier%2==0)&(beta>np.radians(28))&(distdecay>55)
#"""
#print('porfa',mask.sum())
"""
def plot_angle_vs_dist_hist(beta,distdecay, id=[0]):
    plt.figure(figsize=(12,15),dpi=100)
    if np.size(id)!=1:
        betam=beta[(id%2==0)]
        distdecaym=distdecay[(id%2==0)]
    binsize=40
    binsang=np.linspace(0,np.degrees(maxangle),binsize)
    binsdist=np.logspace(np.log10(0.1),np.log10(10000),binsize)
    #binsdist=np.linspace(0,5000,binsize)
    #mask=(id%2==0)&(beta>np.radians(28))&(distdecay>55)
    #print('Total raros',np.sum(mask))
    #a,int1,int2=trajectory_inside_tel_sphere(lgE,coord[mask,:],vcoord[mask,:],ntels=1)
    #print(coord[mask,:][0,:],vcoord[mask,:][0,:],dcoord[mask,:][0,:],int2[0,:])

    #print(a,np.size(a))

    plt.hist(np.degrees(beta),bins=100)
    plt.grid()
    #plt.hist(distdecay,bins=binsdist)
    #plt.xscale('log')
    plt.title('Emergence angle histogram')
    plt.show()
    plt.figure(figsize=(12,15),dpi=100)

    ntrigg=np.sum((id%2==0))
    ntot=np.size(id)
    height, xedges,yedges=np.histogram2d(np.degrees(betam), distdecaym, bins=(binsang,binsdist))
    htotal, xedgest,yedgest=np.histogram2d(np.degrees(beta), distdecay, bins=(binsang,binsdist))

    htotal[htotal<1]=1
    effic=height/htotal
    effic[htotal<2]=0
    XX, YY = np.meshgrid(xedges, yedges,indexing='ij')
    #height[height<1]=1
    plt.pcolormesh(XX,YY, height,cmap=plt.cm.jet)
    plt.colorbar(label='Number of in-FoV events')
    plt.yscale('log')
#   plt.xticks(np.arange(0,maxdist+1,step=maxdist/5))
    plt.xlabel('angle (degrees)')
    plt.ylabel('Distance to decay, km')
    #plt.ticklabel_format(axis='both', style='sci',scilimits=(0,0))
    plt.title(f'Distance to decay vs emergence angle histogram, trigger events,n={ntrigg}, maxang={np.degrees(maxangle)}')
    plt.show()
plot_angle_vs_dist_hist(beta,distdecay,idfinal)
"""

"""
#print('\n',idnew,idnew[idnew!=1],np.size(idnew),np.size(idnew[idnew!=1]))
#plt.xlim([-100000,100000])
#plt.ylim([-100000,100000])
#plt.title(f'Trajectories that intersect ground of telescope, n={num_points}')
#plt.savefig('testintersecground.png')
#PLOT FOR EACH ENERGY BIN THE % OF EVENTS THAT ARE INSIDE SPHERE OF 1, 2, 3, 4 TELESCOPES

plt.figure(figsize=(16,32),dpi=100)
mpl.rcParams.update({'font.size': 12})
lgEintervals=np.linspace(16.5,20.5,15)
lgEintervals=np.insert(lgEintervals,0,2*lgEintervals[0]-lgEintervals[1])
print(lgEintervals)
maxang=np.radians(20)
num_points=int(4e5)
colors = cm.rainbow(np.linspace(0, 1, 4))
for i in range(lgEintervals.size-1):
    rcut=Rcutoff(lgEintervals[i+1])
    radius=calcradius(lgEintervals[i+1],num_points,plotfig=False)
    coord,vcoord=gen_points(num_points,radius,maxang)
    identifier,int1,int2=trajectory_inside_tel_sphere(lgEintervals[i+1],coord,vcoord)
    id,__=decay_inside_fov(lgEintervals[i+1],coord,vcoord,identifier,int1,int2)
    id=id[id!=1]
    sum1=np.sum(id==2)+np.sum(id==3)+np.sum(id==5)+np.sum(id==7)
    sum2=sum1+np.sum(id==14)+np.sum(id==21)+np.sum(id==35)+np.sum(id==6)+np.sum(id==10)+np.sum(id==15)
    sum3=sum2+np.sum(id==42)+np.sum(id==70)+np.sum(id==105)+np.sum(id==30)
    sum4=sum3+np.sum(id==210)
    plt.bar((lgEintervals[i]+lgEintervals[i+1])/2,100*sum4/num_points,width=lgEintervals[1]-lgEintervals[0],align='edge',color=colors[3],edgecolor='k',label="4 telescopes" if i == 0 else "")
    plt.bar((lgEintervals[i]+lgEintervals[i+1])/2,100*sum3/num_points,width=lgEintervals[1]-lgEintervals[0],align='edge',color=colors[2],edgecolor='k',label="3 telescopes" if i == 0 else "")
    plt.bar((lgEintervals[i]+lgEintervals[i+1])/2,100*sum2/num_points,width=lgEintervals[1]-lgEintervals[0],align='edge',color=colors[1],edgecolor='k',label="2 telescopes" if i == 0 else "")
    plt.bar((lgEintervals[i]+lgEintervals[i+1])/2,100*sum1/num_points,width=lgEintervals[1]-lgEintervals[0],align='edge',color=colors[0],edgecolor='k',label="1 telescopes" if i == 0 else "")
    plt.title(f'% of events inside telescope sphere as a function of energy')
    plt.xlabel('logE')
    plt.ylabel(f'% of events inside telescope sphere')
    #handles, labels = axs[0].get_legend_handles_labels()
    plt.legend()
    plt.grid(visible=True)
plt.show()

"""

# plt.savefig('numtelescopesvsenergy.png')
""" #AND INCLUDE VOLUMES RATIO
plt.figure(figsize=(16,12),dpi=200)
mpl.rcParams.update({'font.size': 22})
lgEintervals=np.linspace(15.5,17.5,50)
colors = cm.rainbow(np.linspace(0, 1, 4))
for i in range(lgEintervals.size):
    plt.grid('True')
    rcut=Rcutoff(lgEintervals[i])
    radius=calcradius(lgEintervals[i],num_points)
    coord,vcoord=gen_points(num_points,radius)
    identifier,mask=trajectory_inside_tel_sphere(lgEintervals[i],coord,vcoord)
    percinside=100*np.sum(mask)/num_points
    id=identifier[mask]
    sum1=np.sum(id==2)+np.sum(id==3)+np.sum(id==5)+np.sum(id==-1)
    sum2=sum1+np.sum(id==-2)+np.sum(id==-3)+np.sum(id==-5)+np.sum(id==6)+np.sum(id==10)+np.sum(id==15)
    sum3=sum2+np.sum(id==-6)+np.sum(id==-10)+np.sum(id==-15)+np.sum(id==30)
    sum4=sum3+np.sum(id==-30)
    plt.bar(lgEintervals[i],sum4/num_points,width=lgEintervals[1]-lgEintervals[0],align='center',color=colors[3],edgecolor='k',label="Ratio events" if i == 0 else "")
    plt.bar(lgEintervals[i],sum3/num_points,width=lgEintervals[1]-lgEintervals[0],align='center',color=colors[3],edgecolor='k')
    plt.bar(lgEintervals[i],sum2/num_points,width=lgEintervals[1]-lgEintervals[0],align='center',color=colors[3],edgecolor='k')
    plt.bar(lgEintervals[i],sum1/num_points,width=lgEintervals[1]-lgEintervals[0],align='center',color=colors[3],edgecolor='k')
    plt.title(f'ratio of events inside telescope sphere as a function of energy')
    plt.xlabel('logE')
    plt.ylabel(f'ratio of events inside telescope sphere')
    telvolume=4/3*np.pi*rcut**3*4
    throwvolume=4/3*np.pi*radius**3
    plt.bar(lgEintervals[i],telvolume/throwvolume,width=lgEintervals[1]-lgEintervals[0],align='center',color='g',edgecolor='k',label='Ratio Volumes'if i == 0 else "")
    plt.title('Ratio between telescope spheres volume and thrown volume')
    plt.ylabel('telescope volume / thrown volume')
plt.legend()
plt.grid()
plt.savefig('numtelescopesvsenergy_andvolumeratio.png')"""

"""
lgE=17
radius=calcradius(lgE,num_points)
n=int(1e5)
coord,vcoord=gen_points(n,radius)

a,b=decay(coord,vcoord,lgE)
eyevectors=gen_eye_vectors(telphi,teltheta)
origin=np.vstack((LL,LM,LA,CO))
#Plot of telescopes FoV area and their center vector
energy=np.arange(17,20.6,1)
rfinal=np.zeros_like(energy)
plt.figure()
f, axs=plt.subplots(2,2, sharex=True, sharey=True,figsize=(28,12),dpi=300)


for i in range(energy.size):
    plt.subplot(2,2,i+1)
    rfinal[i]=calcradius(energy[i],num_points,plotfig=True)
    reye=Rcutoff(energy[i])
    plt.quiver(origin[:,0],origin[:,1], reye*eyevectors[:,0], reye*eyevectors[:,1],angles='xy', scale_units='xy', scale=1, color=['r','purple','yellow','blue'])
    plt.title(f'Energy=10^{energy[i]} eV')
    
    lims=rfinal[i]
    plt.gca().set_aspect('equal')
    plt.legend(loc='upper left')
    plt.xlim([-lims,lims])
    plt.ylim([-lims,lims])
    plt.plot(0,0,'kx',markersize=15)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('telescopespheres+vectors.png')
"""

"""
def cone(xd,yd,elev,phi,lgE,angle=15,npoints=10000): #a,b,c >0 
    rmax=Rcutoff(lgE)
    c=1 #Think its useless
    a=c*np.tan(np.radians(angle))
    x= np.random.uniform(-rmax, rmax, n)
    y= np.random.uniform(-rmax, rmax, n)
    z= np.random.uniform(0, rmax, n)
    d=np.sqrt(x**2+y**2+z**2)
    mask=(z>=c*np.sqrt(((x)/a)**2+((y)/a)**2)) & (d<=rmax)
    conecoords=np.column_stack((x[mask],y[mask],z[mask]))
    theta_rot_axis = np.array([0, 1, 0])
    theta_rotation = R.from_rotvec(theta_rot_axis * (np.pi/2-elev))

    phi_rot_axis = np.array([0, 0, 1])
    phi_rot = R.from_rotvec(phi_rot_axis*phi)

    conecoords = theta_rotation.apply(conecoords)
    conecoords = phi_rot.apply(conecoords)
    conecoords[:,0]=conecoords[:,0]+xd
    conecoords[:,1]=conecoords[:,1]+yd

    return conecoords

f, axs=plt.subplots(2,3,sharex=False,sharey=False, figsize=(20,15),dpi=250)
max_distance=1e6
for i in range(6):
    plt.subplot(2,3,1)
    conecoords=cone(LL[0],LL[1],LLelev[i],LLphi[i],lgE,telangle)
    colors = cm.rainbow(np.linspace(0, 1, 6))
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    #plt.plot(xin,yin,'g.',alpha=0.8,markersize=0.5)
    plt.plot(conecoords[:,0],conecoords[:,1],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.plot(LL[0],LL[1],color='red',marker='v',markersize=6)
    plt.plot(LM[0],LM[1],color='purple',marker='>',markersize=6)
    plt.plot(LA[0],LA[1],color='yellow',marker='^',markersize=6)
    plt.plot(CO[0],CO[1],color='blue',marker='<',markersize=6)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title(f'x-y cone plot, lgE={lgE}')
    plt.subplot(2,3,2)
    plt.title(f'x-z cone plot, lgE={lgE}')

    plt.plot(conecoords[:,0],conecoords[:,2],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.xlabel('x (m)')
    plt.grid()
    plt.ylabel('z (m)')
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    plt.gca().set_aspect('equal')
    plt.subplot(2,3,3)
    plt.title(f'y-z cone plot, lgE={lgE}')
    plt.plot(conecoords[:,1],conecoords[:,2],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.xlabel('y (m)')
    plt.grid()
    plt.ylabel('z (m)')
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    plt.gca().set_aspect('equal')

lgE=17
for i in range(6):
    plt.subplot(2,3,4)
    conecoords=cone(LL[0],LL[1],LLelev[i],LLphi[i],lgE)
    colors = cm.rainbow(np.linspace(0, 1, 6))
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    #plt.plot(xin,yin,'g.',alpha=0.8,markersize=0.5)
    plt.plot(conecoords[:,0],conecoords[:,1],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.plot(LL[0],LL[1],color='red',marker='v',markersize=6)
    plt.plot(LM[0],LM[1],color='purple',marker='>',markersize=6)
    plt.plot(LA[0],LA[1],color='yellow',marker='^',markersize=6)
    plt.plot(CO[0],CO[1],color='blue',marker='<',markersize=6)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.title(f'x-y cone plot, lgE={lgE}')

    plt.subplot(2,3,5)
    plt.title(f'x-z cone plot, lgE={lgE}')
    plt.plot(conecoords[:,0],conecoords[:,2],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    plt.grid()
    plt.gca().set_aspect('equal')
    plt.subplot(2,3,6)
    plt.title(f'y-z cone plot, lgE={lgE}')
    plt.plot(conecoords[:,1],conecoords[:,2],'.',markersize=1,color=colors[i],label=f'Telescope {i+1}')
    plt.xlabel('y (m)')
    plt.ylabel('z (m)')
    plt.xlim([-max_distance,max_distance])
    plt.ylim([-max_distance,max_distance])
    plt.grid()
    plt.gca().set_aspect('equal')

plt.savefig('conetest1.png')"""
#def inside_cone(coord,a,b,c,lgE): #Terminar esto