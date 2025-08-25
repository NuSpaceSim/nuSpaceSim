from .augermc import *
a=6378137  #earth major axis WGS84
b=6356752.314245 #minor axis
h=1416
def find_trajectory_point_ecef_analytical(latcore,loncore,heightcore, az, beta, target_height):
    ecef_core = latlongtoECEF(latcore, loncore, heightcore)

    #print(np.shape(ecef_core))
    x0=ecef_core[:,0]
    y0=ecef_core[:,1]
    z0=ecef_core[:,2]

    enu_dir=[np.cos(az)*np.cos(beta),np.sin(az)*np.cos(beta),np.sin(beta)]
    ecef_dir=enutoecef_vector(ecef_core,enu_dir, lat=latcore,lon=loncore)

    dx, dy, dz = ecef_dir[0]
    # Target height above initial height
    
    # Coefficients for quadratic equation
    a_h = a + target_height
    b_h = b + target_height
    A = (dx**2 + dy**2) / a_h**2 + dz**2 / b_h**2
    B = 2 * ((x0 * dx + y0 * dy) / a_h**2 + (z0 * dz) / b_h**2)
    C = (x0**2 + y0**2) / a_h**2 + z0**2 / b_h**2 - 1
    
    # Solve quadratic equation: A s^2 + B s + C = 0
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        raise ValueError("No real solution exists for the given trajectory and height.")
    
    # Choose the positive root (forward direction)

    #Now selecting the negative root to go downwards
    s = (-B + np.sqrt(discriminant)) / (2 * A)
#    if s < 0:
#        s = (-B -  np.sqrt(discriminant)) / (2 * A)  # Try the other root
#        if s < 0:
#            raise ValueError("No positive solution for s found.")
    
    # Calculate final point
    x = x0 + s * dx
    y = y0 + s * dy
    z = z0 + s * dz
    targetecef=np.column_stack((x, y, z))
    return targetecef, s

beta=np.radians(10)
azim=np.radians(50.773521-180)
target_height=1
atmstart,dist=find_trajectory_point_ecef_analytical(LLlat,LLlong,h,azim,beta,target_height)
ecef_core = latlongtoECEF(LLlat, LLlong, h)
#print(np.shape(ecef_core))
x0=ecef_core[:,0]
y0=ecef_core[:,1]
z0=ecef_core[:,2]

enu_dir=[np.cos(azim)*np.cos(beta),np.sin(azim)*np.cos(beta),np.sin(beta)]
ecef_dir=enutoecef_vector(ecef_core,enu_dir, lat=LLlat,lon=LLlong)
startingecef=starting_point(ecef_core,ecef_dir)
disttravelled=np.linalg.norm(ecef_core-startingecef,axis=1)
print(disttravelled,'starting_point Distance from start to core')
print(dist,'New Distance from atmstart to core')



#print(dist, 'Distance from atmstart to core')
delta=100
Xfirst_offline=integrated_grammage_opt(atmstart,ecef_core,delta)
print(Xfirst_offline,'Grammage from core to target height')

from scipy.integrate import quad
from scipy.optimize import root_scalar


def calculate_endpoint(start_pos, direction, grammage):
    """
    Calculate the end point in ECEF after travelling a given grammage along the direction.
    start_pos: np.ndarray, shape (3,) or (N,3), ECEF position in meters
    direction: np.ndarray, shape (3,) or (N,3), direction vector (will be normalized)
    grammage: float or np.ndarray shape (N,), grammage in g/cm²
    Returns end_pos: np.ndarray, shape (3,) or (N,3)
    """
    start_pos = np.atleast_2d(start_pos)
    direction = np.atleast_2d(direction)
    grammage = np.atleast_1d(grammage)
    if grammage.shape[0] == 1:
        grammage = np.repeat(grammage, start_pos.shape[0])
    
    N = start_pos.shape[0]
    assert direction.shape[0] == N and grammage.shape[0] == N
    
    dir_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    unit_dir = direction / dir_norm
    
    density_scale = 1e6  # g/cm³ to g/m³
    grammage_scale = 1e4  # g/cm² to g/m²
    
    end_points = np.zeros((N, 3))
    
    for i in range(N):
        P = start_pos[i]
        D = unit_dir[i]
        target = grammage[i] * grammage_scale
        
        def density_at_t(t):
            pos = P + t * D
            height_m = altitude_from_ecef(pos)
            height_m = np.maximum(height_m, 0.0)  # avoid negative heights
            height_km = height_m / 1000.0
            rho_g_cm3 = atmdensity_interpolation(height_km)
            return rho_g_cm3 * density_scale
        
        def cumul_integral(s):
            if s <= 0:
                return 0.0
            integ, _ = quad(density_at_t, 0, s, epsabs=1e-2, epsrel=1e-3)
            return integ
        
        # Find upper bound for s
        upper = 1000.0  # start with 1 km
        max_upper = 1e7  # 10,000 km max
        while cumul_integral(upper) < target and upper < max_upper:
            upper *= 2
        if upper >= max_upper:
            raise ValueError(f"Cannot reach the required grammage {grammage[i]} g/cm²; path too long or density too low.")
        
        # Solve for s
        sol = root_scalar(lambda s: cumul_integral(s) - target, bracket=[0, upper], xtol=0.1)  # 0.1m tolerance, but 5m ok
        s_found = sol.root
        
        end_point = P + s_found * D
        end_points[i] = end_point
    
    if N == 1:
        return end_points[0]
    return end_points

xmaxpos=calculate_endpoint(ecef_core,ecef_dir,Xfirst_offline)
hxmax=altitude_from_ecef(xmaxpos)
print(hxmax,'Height at X')
pos2=calculate_endpoint(xmaxpos,ecef_dir,24)
print(altitude_from_ecef(pos2),'Height after 24 g/cm2 more')
def xmax_inside_fov(lgE,groundecef,xmaxecef, id,ntels=1
                             ,distfactor=0.1,extraangle=np.radians(2),radiusfactor=1.01):
    code=[2,3,5,7]
    eyevector=gen_eye_vectors(telphi,teltheta)

    for i in range(ntels):#telpos.shape[0]
        telextraangle=telangle+extraangle
        thetatelsup=teltheta[2*i]+telextraangle
        thetatelinf=teltheta[2*i]-telextraangle
        teli=(id%code[i]==0)
        energy=lgE[teli]
        xmaxecefi=xmaxecef[teli]
        groundecefi=groundecef[teli]

        xmaxenui=eceftoenu(telposecef[i,:],xmaxecefi)


        r=Rcutoff(energy)*radiusfactor
        fovdist=r*distfactor
        dgroundxmax=np.linalg.norm(xmaxecefi-groundecefi,axis=1)
        dtelxmax=np.linalg.norm(xmaxecefi-telposecef[i],axis=1)
        xmaxnorm=xmaxenui/np.linalg.norm(xmaxenui,axis=1,keepdims=True)
        cosdphi=np.dot(xmaxnorm[:,0:2],eyevector[i,0:2])   #azimuth angle difference with center of telescope
        theta=np.arccos(np.sqrt(xmaxnorm[:,0]**2+xmaxnorm[:,1]**2))*np.sign(xmaxnorm[:,2]) #elevation angle of intersection vector 
        infov=(cosdphi>=np.cos(exacttelangle*5+telangle)) &  (theta<=thetatelsup) &  (theta>=thetatelinf) & (dtelxmax<=fovdist)
        
        index = np.arange(len(id))[teli][~infov] #for outside

        id[index]=id[index]/code[i]

    return id, dgroundxmax, dtelxmax
