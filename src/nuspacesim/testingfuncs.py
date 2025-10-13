from .augermc import *


a=6378137  #earth major axis WGS84
b=6356752.314245 #minor axis
h=1416.2
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

beta=np.radians(5)
azim=np.radians(10.025630-180)
target_height=1
#atmstart,dist=find_trajectory_point_ecef_analytical(LLlat,LLlong,h,azim,beta,target_height)
ecef_core = latlongtoECEF(LLlat,LLlong,h)
startatm=1414

#print(np.shape(ecef_core))
x0=ecef_core[:,0]
y0=ecef_core[:,1]
z0=ecef_core[:,2]

enu_dir=[np.cos(azim)*np.cos(beta),np.sin(azim)*np.cos(beta),np.sin(beta)]
ecef_dir=enutoecef_vector(ecef_core,enu_dir, lat=LLlat,lon=LLlong)
startingecef,_=starting_point(ecef_core,ecef_dir,startatm)
slants=np.arange(0,26000,10)
xstep=1 #g/cm2
midpoints=np.zeros((len(slants),3))
midpoints[0,:]=startingecef
heights=np.zeros(len(slants))
heights[0]=altitude_from_ecef(startingecef)
xtravelled=0

for i in range(len(slants)-1):
    steppos=midpoints[i,:]
    stepheight=heights[i]
    while xtravelled<slants[i+1]:
        xtravelled=xtravelled+xstep
        stepdensity=atmdensity_interpolation(stepheight/1000) #g/cm3
        stepdist=xstep/stepdensity/1e2 #in m
        steppos=steppos+ecef_dir*stepdist
        #print(steppos,'new pos')
        stepheight=altitude_from_ecef(steppos)
    midpoints[i+1,:]=steppos
    heights[i+1]=stepheight

midpointsfast=calculate_endpoint_grammarray(startingecef,ecef_dir,slants)
heightsfast=altitude_from_ecef(midpointsfast)
dists=np.linalg.norm(midpoints-ecef_core,axis=1)
#disttravelled=np.linalg.norm(ecef_core-startingecef,axis=1)
#print(disttravelled,'starting_point Distance from start to core')
all=np.vstack([slants,heightsfast,dists]).T
i=0
print('DATA',all[i:i+10,:])
data = np.loadtxt("sampling_90.5.txt", comments="#")

heightoffline = data[:, 1]   # 2nd column (height [m])
densityoffline = data[:, 2]  # 3rd column (density [g/cm^3])


# WGS84 constants
WGS84_A = 6378137.0        # semi-major axis (m)
WGS84_B = 6356752.3142     # semi-minor axis (m)
A2 = WGS84_A ** 2
B2 = WGS84_B ** 2

def intersection_with_ellipsoid(height_m, core, direction):
    """
    Intersect the line p(t) = core + t*dir with the WGS-84 ellipsoidal shell
    at altitude `height_m` above the ellipsoid.
    Returns a list with 0, 1, or 2 points (each shape (3,)).
    """
    a = 6378137.0
    b = 6356752.314245

    a_h = a + float(height_m)
    b_h = b + float(height_m)

    d = np.array(direction, dtype=float)
    nd = np.linalg.norm(d)
    if nd == 0:
        return []
    d /= nd
    p0 = np.array(core, dtype=float)

    A = (d[0]**2 + d[1]**2) / (a_h**2) + (d[2]**2) / (b_h**2)
    B = 2.0 * ((p0[0]*d[0] + p0[1]*d[1]) / (a_h**2) + (p0[2]*d[2]) / (b_h**2))
    C = (p0[0]**2 + p0[1]**2) / (a_h**2) + (p0[2]**2) / (b_h**2) - 1.0

    disc = B*B - 4*A*C
    if disc < 0:
        return []

    sqrt_disc = np.sqrt(max(0.0, disc))
    t1 = (-B - sqrt_disc) / (2*A)
    t2 = (-B + sqrt_disc) / (2*A)

    #p1 = p0 + t1*d
    p2 = p0 + t2*d
    return p2


def auger_atm_table(
    startingecef, ecef_dir, core,
    deltaX,                      # [g/cm^2] slant-depth step (pass positive)
    depth_of_height,             # X(h_km) -> depth [g/cm^2]
    height_of_depth,             # h_km(X) -> height [km]
    altitude_from_ecef,          # alt_m(ECEF) -> meters
    startatm,         # [m] start of atmosphere (for sanity checks)
    upward=True,
    minVerticalDepth=0.00101949,        # [g/cm^2]
    maxVerticalDepth=1032.88,     # [g/cm^2]
):
    """
    Upward-focused replication of the C++ inclined atmosphere loop.
    - Local DOWN angle: cosTheta = -d̂·n̂_up.
    - deltaX is positive; verticalDeltaX = deltaX * cosTheta (so depth decreases when cosTheta<0).
    - lastPoint is the forward intersection with the 99.999 km shell: p = core + t*dir, choose min t>0.
    - At each step, select the shell intersection with the smallest t strictly greater than current t.
    - verticalHeight = atmHeightVsDepth.Y(tmpDepth) (meters), slantDepth = log(tmpSlantDepth).
    """

    a2 = (6378137.0)**2
    b2 = (6356752.314245)**2

    startingecef = np.asarray(startingecef, dtype=float)
    dir_unit     = np.asarray(ecef_dir, dtype=float)
    core         = np.asarray(core, dtype=float)

    # --- Choose lastPoint at 99.999 km: smallest positive t from core ---
    H_LAST_M = 99999.0
    lastPoint = intersection_with_ellipsoid(H_LAST_M, core, dir_unit)


    # Initial state
    iPoint = startingecef.copy()
    cosTheta = local_cosTheta(iPoint,dir_unit)

    verticalDeltaX=0
    tmpHeight_m = float(altitude_from_ecef(iPoint))                  # meters
    tmpDepth    = float(depth_of_height(tmpHeight_m/1000.0))         # g/cm^2

    X0 = float(depth_of_height(startatm/1000.0))
    # For upward: cosTheta<0 and (tmpDepth-X0)<0 -> positive
    tmpSlantDepth = cosTheta*(tmpDepth - X0) if upward else tmpDepth
    verticalHeight = []
    slantDepth     = []
    distanceToImpact = []
    # First entry (C++)
    verticalHeight.append(1000.0 * float(height_of_depth(tmpDepth)))  # meters
    slantDepth.append(tmpSlantDepth)
    distanceToImpact.append(0.0 - np.linalg.norm(startingecef - core))



    while True:
        cosTheta = local_cosTheta(iPoint,dir_unit)
        verticalDeltaX = deltaX * cosTheta
        tmpDepth += verticalDeltaX

        # Stop if outside vertical-depth bounds
        if (tmpDepth >= maxVerticalDepth) or (tmpDepth <= minVerticalDepth):
            tmpHeight_m = altitude_from_ecef(lastPoint)
            verticalDeltaX=depth_of_height(tmpHeight_m/1000.0)-(tmpDepth-verticalDeltaX)
            tmpSlantDepth += verticalDeltaX/cosTheta

            verticalHeight.append(tmpHeight_m)
            slantDepth.append(tmpSlantDepth)
            distanceToImpact.append(np.linalg.norm(lastPoint - core))
            break

        # Advance one slant step
        tmpSlantDepth += deltaX

        # Invert X -> h (C++: tmpHeight = atmHeightVsDepth.Y(tmpDepth))
        tmpHeight_m = 1000.0 * float(height_of_depth(tmpDepth))

        # Intersections with this shell
        nextpoint = intersection_with_ellipsoid(tmpHeight_m, core, dir_unit)

        # Store results
        verticalHeight.append(tmpHeight_m)
        slantDepth.append(tmpSlantDepth)
        distanceToImpact.append(
            np.linalg.norm(nextpoint - startingecef) - np.linalg.norm(core - startingecef)
        )

        # Advance point and parameter
        iPoint = nextpoint
    return (
        np.asarray(verticalHeight, dtype=float),
        np.asarray(slantDepth, dtype=float),
        np.asarray(distanceToImpact, dtype=float),
    )

testdist=20000
grammage=integrated_grammage_opt(ecef_core,ecef_core+testdist*ecef_dir,10)
print(grammage)
endpos=calculate_endpoint_grammarray(ecef_core,ecef_dir,grammage)
print(np.linalg.norm(endpos-(ecef_core+testdist*ecef_dir),axis=1))
#starttest=ecef_core+testdist*ecef_dir

vertheights, slants, dists = auger_atm_table(
    startingecef[0], ecef_dir[0], ecef_core[0], 10,
    depth_spline,height_spline, altitude_from_ecef,startatm
)
all=np.vstack([slants,vertheights,dists]).T
print('DATA',all[500:510])
print(ecef_core,ecef_dir)

exit()
# Plot
plt.figure()
plt.plot(heightoffline,densityoffline, label='Offline Data', color='blue')
plt.plot(heightoffline,atmdensity_interpolation(heightoffline/1000), label='Atm interp', color='red', linestyle='--')
plt.ylabel("Density [g/cm³]")
plt.xlabel("Height [m]")
plt.yscale('log')
plt.title("Density vs Height")
plt.grid(True)
plt.legend()
plt.savefig('atmdensity.png')

diff=densityoffline-atmdensity_interpolation(heightoffline/1000)
reldiff=diff/densityoffline
plt.figure()
plt.plot(heightoffline,reldiff*100, label='Difference', color='green')
plt.ylabel("Relative Density Difference %")
plt.xlabel("Height [m]")
plt.savefig('atmdiff.png')



print(ecef_core[0],ecef_dir[0])


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

#xmaxpos=calculate_endpoint(ecef_core,ecef_dir,Xfirst_offline)
#hxmax=altitude_from_ecef(xmaxpos)
#print(hxmax,'Height at X')
#pos2=calculate_endpoint(xmaxpos,ecef_dir,24)
#print(altitude_from_ecef(pos2),'Height after 24 g/cm2 more')
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
""""""