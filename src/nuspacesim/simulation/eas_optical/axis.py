from typing import Any, Protocol
from functools import cached_property
import numpy as np
from abc import ABC, abstractmethod
from importlib.resources import as_file, files
from scipy.constants import value,nano
from scipy.spatial.transform import Rotation as R
from scipy.integrate import cumulative_trapezoid, quad
from functools import cached_property
# from scipy.stats import norm

from ..atmosphere.atmosphere import Atmosphere
from .generate_Cherenkov import MakeYield
from .shower import Shower

class Counters(ABC):
    '''This is the class containing the neccessary methods for finding the
    vectors from a shower axis to a user defined array of Cherenkov detectors
    with user defined size'''

    def __init__(self, input_vectors: np.ndarray, input_radius: np.ndarray):
        self.vectors = input_vectors
        self.input_radius = input_radius

    @property
    def vectors(self) -> np.ndarray:
        '''Vectors to user defined Cherenkov counters getter'''
        return self._vectors

    @vectors.setter
    def vectors(self, input_vectors: np.ndarray):
        '''Vectors to user defined Cherenkov counters setter'''
        if type(input_vectors) != np.ndarray:
            input_vectors = np.array(input_vectors)
        if input_vectors.shape[1] != 3 or len(input_vectors.shape) != 2:
            raise ValueError("Input is not an array of vectors.")
        self._vectors = input_vectors

    @property
    def input_radius(self) -> np.ndarray | float:
        '''This is the input counter radius getter.'''
        return self._input_radius

    @input_radius.setter
    def input_radius(self, input_value: np.ndarray | float):
        '''This is the input counter radius setter.'''
        if type(input_value) != np.ndarray:
            input_value = np.array(input_value)
        if np.size(input_value) == 1:
            self._input_radius = np.full(self.vectors.shape[0], input_value)
        else:
            self._input_radius = input_value
            

    @property
    def N_counters(self) -> int:
        '''Number of counters.'''
        return self.vectors.shape[0]

    @property
    def r(self) -> np.ndarray:
        '''distance to each counter property definition'''
        return vector_magnitude(self.vectors)

    @abstractmethod
    def area(self, *args, **kwargs) -> np.ndarray:
        '''This is the abstract method for the detection surface area normal to
        the axis as seen from each point on the axis. Its shape must be
        broadcastable to the size of the travel_length array, i.e. either a
        single value or (# of counters, # of axis points)'''

    @abstractmethod
    def omega(self, *args, **kwargs) -> np.ndarray:
        '''This abstract method should compute the solid angle of each counter
        as seen by each point on the axis'''

    def area_normal(self) -> np.ndarray:
        '''This method returns the full area of the counting aperture.'''
        return np.pi * self.input_radius**2

    @staticmethod
    def law_of_cosines(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        '''This method returns the angle C across from side c in triangle abc'''
        cos_C = (c**2 - a**2 - b**2)/(-2*a*b)
        cos_C[cos_C > 1.] = 1.
        cos_C[cos_C < -1.] = -1.
        return np.arccos(cos_C)

    def travel_vectors(self, axis_vectors) -> np.ndarray:
        '''This method returns the vectors from each entry in vectors to
        the user defined array of counters'''
        return self.vectors.reshape(-1,1,3) - axis_vectors

    def travel_length(self, axis_vectors) -> np.ndarray:
        '''This method computes the distance from each point on the axis to
        each counter'''
        return vector_magnitude(self.travel_vectors(axis_vectors))

    def cos_Q(self, axis_vectors) -> np.ndarray:
        '''This method returns the cosine of the angle between the z-axis and
        the vector from the axis to the counter'''
        travel_n = self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]
        return np.abs(travel_n[:,:,-1])

    def travel_n(self, axis_vectors) -> np.ndarray:
        '''This method returns the unit vectors pointing along each travel
        vector.
        '''
        return self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]

    def calculate_theta(self, axis_vectors) -> np.ndarray:
        '''This method calculates the angle between the axis and counters'''
        travel_length = self.travel_length(axis_vectors)
        axis_length = np.broadcast_to(vector_magnitude(axis_vectors), travel_length.shape)
        counter_length = np.broadcast_to(self.r, travel_length.T.shape).T
        return self.law_of_cosines(axis_length, travel_length, counter_length)

class Timing(ABC):
    '''This is the abstract base class which contains the methods needed for
    timing photons from their source points to counting locations. Each timing
    bin corresponds to the spatial/depth bins along the shower axis.
    '''
    c = value('speed of light in vacuum')

    def counter_time(self) -> np.ndarray:
        '''This method returns the time it takes after the shower starts along
        the axis for each photon bin to hit each counter

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.axis_time + self.travel_time + self.delay()

    @property
    def travel_time(self) -> np.ndarray:
        '''This method calculates the the time it takes for something moving at
        c to go from points on the axis to the counters.

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.counters.travel_length(self.axis.vectors) / self.c / nano

    @property
    @abstractmethod
    def axis_time(self) -> np.ndarray:
        '''This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of shape: (# of axis points,)
        '''

    @abstractmethod
    def delay(self) -> np.ndarray:
        '''This method calculates the delay a traveling photon would experience
        (compared to travelling at c) starting from given axis points to the
        detector locations

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''

class Yield(Protocol):
    '''This is the protocol needed from a yield object.
    '''

    @property
    def l_mid(self) -> float:
        ...

class Attenuation(ABC):
    '''This is the abstract base class whose specific implementations will
    calculate the fraction of light removed from the signal at each atmospheric
    step.
    '''
    # atm = USStandardAtmosphere()

    with as_file(files('nuspacesim.data.CHASM_tables')/'abstable.npz') as file:
        abstable = np.load(file)

    ecoeff = abstable['ecoeff']
    l_list = abstable['wavelength']
    altitude_list = abstable['height']

    @property
    def yield_array(self) -> list[Yield]:
        return self._yield_array
    
    @yield_array.setter
    def yield_array(self, arr: list[Yield]):
        self._yield_array = arr

    def vertical_log_fraction(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically.
    
        The returned array is of size:
        # of yield bins, with each entry being of size:
        # of axis points
        '''
        log_fraction_array = np.empty_like(self.yield_array, dtype='O')
        for i, y in enumerate(self.yield_array):
            ecoeffs = self.ecoeff[np.abs(y.l_mid-self.l_list).argmin()]
            e_of_h = np.interp(self.axis.h, self.altitude_list, ecoeffs)
            frac_surviving = np.exp(-e_of_h)
            frac_step_surviving = 1. - np.diff(frac_surviving[::-1], append = 1.)[::-1]
            log_fraction_array[i] = np.log(frac_step_surviving)
        return log_fraction_array

    @property
    def lambda_mids(self):
        '''This property is a numpy array of the middle of each wavelength bin'''
        l_mid_array = np.empty_like(self.yield_array)
        for i, y in enumerate(self.yield_array):
            l_mid_array[i] = y.l_mid
        return l_mid_array
    @staticmethod
    def nm_to_cm(l):
        return l*nano*1.e2

    def rayleigh_cs(self, h, l = 400.):
        '''This method returns the Rayleigh scattering cross section as a
        function of both the height in the atmosphere and the wavelength of
        the scattered light. This does not include the King correction factor.

        Parameters:
        h - height (m) single value or np.ndarray
        l - wavelength (nm) single value or np.ndarray

        Returns:
        sigma (cm^2 / particle)
        '''
        l_cm = self.nm_to_cm(l) #convert to cm
        N = self.atm.number_density(h) / 1.e6 #convert to particles/cm^3
        f1 = (24. * np.pi**3) / (N**2 * l_cm**4)
        n = self.atm.delta(h) + 1
        f2 = (n**2 - 1) / (n**2 + 2)
        return f1 * f2**2

    def fraction_passed(self):
        '''This method returns the fraction of light originating at each
        step on the axis which survives to reach each counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        log_fraction_passed_array = self.log_fraction_passed()
        fraction_passed_array = np.empty_like(log_fraction_passed_array, dtype= 'O')
        for i, lfp in enumerate(log_fraction_passed_array):
            lfp[lfp>0.] = -100.
            fraction_passed_array[i] = np.exp(lfp)
        return fraction_passed_array

    @abstractmethod
    def log_fraction_passed(self) -> np.ndarray:
        '''This method should return the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''

class CurvedAtmCorrection(ABC):
    '''This is the abstract base class for performing a curved atmosphere
    correction.
    '''

    @abstractmethod
    def curved_correction(self, vert: np.ndarray) -> np.ndarray:
        '''This method ahould perform the integration of the quantity vert
        for either an upward or downward axis.
        '''

    def __call__(self, vert: np.ndarray) -> np.ndarray:
        return self.curved_correction(vert)

class AxisParams(Protocol):
    '''This is the protocol for an axis parameter container.
    '''
    @property
    def zenith(self) -> float:
        ...

    @property
    def azimuth(self) -> float:
        ...
    
    @property
    def ground_level(self) -> float:
        ...

    @property
    def maximum_altitude(self) -> float:
        ...

    @property
    def ATM(self) -> Atmosphere:
        ...


class Axis(ABC):
    '''This is the abstract base class which contains the methods for computing
    the cartesian vectors and corresponding slant depths of an air shower'''

    earth_radius = 6.371e6 #meters
    lX = -100. #This is the default value for the distance to the axis in log moliere units (in this case log(-inf) = 0, or on the axis)
    

    def __init__(self, params: AxisParams):
        self.params = params
        self.atm = self.params.ATM
        self.ground_level = params.ground_level
        self.zenith = params.zenith
        self.azimuth = params.azimuth
        self.maximum_altitude = params.maximum_altitude
        self.altitude = self.set_initial_altitude()

    @property
    def zenith(self) -> float:
        '''polar angle  property getter'''
        return self._zenith

    @zenith.setter
    def zenith(self, zenith):
        '''zenith angle property setter'''
        if zenith > np.pi/2:
            raise ValueError('Zenith angle cannot be greater than pi / 2')
        if zenith < 0.:
            raise ValueError('Zenith angle cannot be less than 0')
        self._zenith = zenith

    @property
    def azimuth(self) -> float:
        '''azimuthal angle property getter'''
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        '''azimuthal angle property setter'''
        if azimuth >= 2 * np.pi:
            raise ValueError('Azimuthal angle must be less than 2 * pi')
        if azimuth < 0.:
            raise ValueError('Azimuthal angle cannot be less than 0')
        self._azimuth = azimuth

    @property
    def ground_level(self) -> float:
        '''ground level property getter'''
        return self._ground_level

    @ground_level.setter
    def ground_level(self, value):
        '''ground level property setter'''
        if value > self.atm.maximum_height:
            raise ValueError('Ground level too high')
        self._ground_level = value

    def set_initial_altitude(self) -> np.ndarray:
        '''altitude property definition'''
        return np.linspace(self.ground_level, self.maximum_altitude, self.params.N_POINTS)

    @property
    def dh(self) -> np.ndarray:
        '''This method sets the dh attribute'''
        dh = self.h[1:] - self.h[:-1]
        return np.concatenate((np.array([0]),dh))

    @property
    def h(self) -> np.ndarray:
        '''This is the height above the ground attribute'''
        hs = self.altitude - self.ground_level
        hs[0] = 1.e-5
        return hs

    @property
    def delta(self) -> np.ndarray:
        '''delta property definition (index of refraction - 1)'''
        return self.atm.delta(self.altitude)

    @property
    def density(self) -> np.ndarray:
        '''Axis density property definition (kg/m^3)'''
        return self.atm.density(self.altitude)

    @property
    def moliere_radius(self) -> np.ndarray:
        '''Moliere radius property definition (m)'''
        return 96. / self.density

    def h_to_axis_R_LOC(self,h,theta) -> np.ndarray:
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights (m above ground level)
        theta: polar angle of shower axis (radians)

        returns: r (m) (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = self.earth_radius + self.ground_level
        r_CoE= h + R # distance from the center of the earth to the specified height
        rs = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        # rs -= rs[0]
        # rs[0] = 1.
        return rs # Need to find a better way to define axis zero point, currently they are all shifted by a meter to prevent divide by zero errors

    @classmethod
    def theta_normal(cls,h,r) -> np.ndarray:
        '''This method calculates the angle the axis makes with respect to
        vertical in the atmosphere (at that height).

        Parameters:
        h: array of heights (m above sea level)
        theta: array of polar angle of shower axis (radians)

        Returns:
        The corrected angles(s)
        '''
        cq = (r**2 + h**2 + 2*cls.earth_radius*h)/(2*r*(cls.earth_radius+h))
        cq[cq>1.] = 1.
        cq[cq<-1.] = -1.
        # cq = ((cls.earth_radius+h)**2+r**2-cls.earth_radius**2)/(2*r*(cls.earth_radius+h))
        return np.arccos(cq)
    
    def vector_altitude(self, input_vectors: np.ndarray) -> np.ndarray:
        '''This method calculates the altitude of the endpoint of a vector in the 
        CHASM coordinate system.

        Vectors should be of shape (n,3) and the returned array is of shape (n,).
        '''
        if input_vectors.shape[1] != 3 or len(input_vectors.shape) != 2:
            raise ValueError("Input is not an array of vectors.")
        
        vector_r = vector_magnitude(input_vectors)
        h0tocore = self.earth_radius + self.ground_level
        theta = np.arcsin(input_vectors[:,2]/vector_r) + np.pi / 2

        h_plus_R_e2 = vector_r**2 + h0tocore**2 - 2 * vector_r * h0tocore * np.cos(theta)
        return np.sqrt(h_plus_R_e2) - self.earth_radius

    @property
    def theta_difference(self) -> np.ndarray:
        '''This property is the difference between the angle a vector makes with
        the z axis and the angle it makes with vertical in the atmosphere at
        all the axis heights.
        '''
        return self.zenith - self.theta_normal(self.h, self.r)

    @property
    def dr(self) -> np.ndarray:
        '''This method sets the dr attribute'''
        dr = self.r[1:] - self.r[:-1]
        return np.concatenate((np.array([0]),dr))

    @property
    def vectors(self) -> np.ndarray:
        '''axis vector property definition

        returns a vector from the origin to a distances r
        along the axis'''
        ct = np.cos(self.zenith)
        st = np.sin(self.zenith)
        cp = np.cos(self.azimuth)
        sp = np.sin(self.azimuth)
        axis_vectors = np.empty([np.shape(self.r)[0],3])
        axis_vectors[:,0] = self.r * st * cp
        axis_vectors[:,1] = self.r * st * sp
        axis_vectors[:,2] = self.r * ct
        return axis_vectors

    def depth(self, r: np.ndarray) -> np.ndarray:
        '''This method is the depth as a function of distance along the shower
        axis'''
        return np.interp(r, self.r, self.X)

    def get_timing(self, curved_correction: CurvedAtmCorrection) -> Timing:
        '''This function returns an instantiated timing object appropriate for
        the axis implementation.'''
        return self.get_timing_class()(curved_correction)

    def get_attenuation(self, curved_correction: CurvedAtmCorrection, y: list[MakeYield]) -> Attenuation:
        '''This function returns an instantiated attenuation object appropriate for
        the axis implementation.'''
        return self.get_attenuation_class()(curved_correction, y)
    
    def get_curved_atm_correction(self, counters: Counters) -> CurvedAtmCorrection:
        return self.get_curved_atm_correction_class()(self, counters)

    @property
    @abstractmethod
    def r(self) -> np.ndarray:
        '''r property definition'''
        # return self.h_to_axis_R_LOC(self.h, self.zenith)

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        '''This method sets the depth along the shower axis attribute'''

    # @abstractmethod
    # def slant_depth_integrand(self, h: float | np.ndarray) -> float | np.ndarray:
    #     '''This method is the integrand as a function of altitude for calculating slant
    #     depth.
    #     '''

    @abstractmethod
    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''

    @abstractmethod
    def theta(self) -> np.ndarray:
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''

    @abstractmethod
    def get_timing_class(self) -> Timing:
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_attenuation_class(self) -> Attenuation:
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_gg_file(self) -> str:
        '''This method should return the gg array for the particular axis type.
        For linear axes, it should return the regular gg array. For a mesh axis,
        it should return the gg array for the axis' particular log(moliere)
        interval.
        '''

    @abstractmethod
    def reset_for_profile(self, shower: Shower) -> None:
        '''This method re-defines the height attribute depending on where the shower 
        falls along the axis. The purpose of this is to avoid running expensive 
        calculations where no significant amount of particles exist.
        '''

    @abstractmethod
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        '''This method returns the curved atm correction class for the speccific axis 
        implementation.
        '''

class MakeSphericalCounters(Counters):
    '''This is the implementation of the Counters abstract base class for
    CORSIKA IACT style spherical detection volumes.

    Parameters:
    input_vectors: rank 2 numpy array of shape (# of counters, 3) which is a
    list of vectors (meters).
    input_radius: Either a single value for an array of values corresponding to
    the spherical radii of the detection volumes (meters).
    '''

    def __repr__(self):
        return "SphericalCounters({:.2f} Counters with average area~ {:.2f})".format(
        self.vectors.shape[0], self.area_normal().mean())

    def area(self) -> np.ndarray:
        '''This is the implementation of the area method, which calculates the
        area of spherical counters as seen from the axis.
        '''
        return self.area_normal()

    def omega(self, axis: Axis) -> np.ndarray:
        '''This method computes the solid angle of each counter as seen by
        each point on the axis'''
        return (self.area() / (self.travel_length(axis).T)**2).T

def vector_magnitude(vectors: np.ndarray) -> np.ndarray:
    '''This method computes the length of an array of vectors'''
    return np.sqrt((vectors**2).sum(axis = -1))

class LateralSpread:
    '''This class interacts with the table of NKG universal lateral distributions
    '''
    with as_file(files('nuspacesim.data.CHASM_tables')/'n_t_lX_of_t_lX.npz') as file:
        lx_table = np.load(file)

    t = lx_table['ts']
    lX = lx_table['lXs']
    n_t_lX_of_t_lX = lx_table['n_t_lX_of_t_lX']

    @classmethod
    def get_t_indices(cls, input_t: np.ndarray) -> np.ndarray:
        '''This method returns the indices of the stages in the table closest to
        the input stages.
        '''
        return np.abs(input_t[:, np.newaxis] - cls.t).argmin(axis=1)

    @classmethod
    def get_lX_index(cls, input_lX: float) -> np.ndarray:
        '''This method returns the index closest to the input lX within the 5
        tabulated lX values.
        '''
        return np.abs(input_lX - cls.lX).argmin()

    @classmethod
    def nch_fractions(cls, input_ts: np.ndarray, input_lX: float) -> np.ndarray:
        '''This method returns the fraction of charged particles at distance
        exp(lX) moliere units from the shower axis at an array of stages
        (input_ts).
        '''
        t_indices = cls.get_t_indices(input_ts)
        lX_index = cls.get_lX_index(input_lX)
        return cls.n_t_lX_of_t_lX[t_indices,lX_index]

def axis_to_mesh(lX: float, axis: Axis, shower: Shower, N_ring: int = 20) -> tuple:
    '''This function takes an shower axis and creates a 3d mesh of points around
    the axis (in coordinates where the axis is the z-axis)
    Parameters:
    axis: axis type object
    shower: shower type object
    Returns:
    an array of vectors to points in the mesh
    size (, 3)
    The corresponding # of charged particles at each point
    The corresponding array of stages (for use in universality calcs)
    The corresponding array of deltas (for use in universality calcs)
    The corresponding array of shower steps (dr) in m (for Cherenkov yield calcs)
    The corresponding array of altitudes (for timing calcs)

    '''
    X = np.exp(lX) #number of moliere units for the radius of the ring
    X_to_m = X * axis.moliere_radius
    X_to_m[X_to_m>axis.config.MAX_RING_SIZE] = axis.config.MAX_RING_SIZE
    axis_t = shower.stage(axis.X)
    total_nch = shower.profile(axis.X) * LateralSpread.nch_fractions(axis_t,lX)
    axis_d = axis.delta
    axis_dr = axis.dr
    axis_altitude = axis.altitude
    r = axis.r
    ring_theta = np.arange(0,N_ring) * 2 * np.pi / N_ring
    ring_x = X_to_m[:, np.newaxis] * np.cos(ring_theta)
    ring_y = X_to_m[:, np.newaxis] * np.sin(ring_theta)
    x = ring_x.flatten()
    y = ring_y.flatten()
    z = np.repeat(r, N_ring)
    t = np.repeat(axis_t, N_ring)
    d = np.repeat(axis_d, N_ring)
    nch = np.repeat(total_nch / N_ring, N_ring)
    dr = np.repeat(axis_dr, N_ring)
    a = np.repeat(axis_altitude, N_ring)
    return np.array((x,y,z)).T, nch, t, d, dr, a

def rotate_mesh(mesh: np.ndarray, theta: float, phi: float) -> np.ndarray:
    '''This function rotates an array of vectors by polar angle theta and
    azimuthal angle phi.

    Parameters:
    mesh: numpy array of axis vectors shape = (# of vectors, 3)
    theta: axis polar angle (radians)
    phi: axis azimuthal angle (radians)
    Returns a numpy array the same shape as the original list of vectors
    '''
    theta_rot_axis = np.array([0,1,0]) #y axis
    phi_rot_axis = np.array([0,0,1]) #z axis
    theta_rot_vector = theta * theta_rot_axis
    phi_rot_vector = phi * phi_rot_axis
    theta_rotation = R.from_rotvec(theta_rot_vector)
    phi_rotation = R.from_rotvec(phi_rot_vector)
    mesh_rot_by_theta = theta_rotation.apply(mesh)
    mesh_rot_by_theta_then_phi = phi_rotation.apply(mesh_rot_by_theta)
    return mesh_rot_by_theta_then_phi

class MeshAxis(Axis):
    '''This class is the implementation of an axis where the sampled points are
    spread into a mesh.
    '''
    lXs = np.arange(-6,0) #Log moliere radii corresponding to the bin edges of the tabulated gg files.

    def __init__(self, lX_interval: tuple, linear_axis: Axis, shower: Shower):
        self.lX_interval = lX_interval
        self.lX = np.mean(lX_interval)
        self.linear_axis = linear_axis
        self.params = linear_axis.params
        self.atm = linear_axis.atm
        self.zenith = linear_axis.zenith
        self.azimuth = linear_axis.azimuth
        self.ground_level = linear_axis.ground_level
        self.shower = shower
        mesh, self.nch, self._t, self._d, self._dr, self._a  = axis_to_mesh(self.lX, 
                                                                            self.linear_axis, 
                                                                            self.shower,
                                                                            N_ring=self.params.N_IN_RING)
        self.meshX = np.repeat(self.X,self.params.N_IN_RING)
        self.rotated_mesh = rotate_mesh(mesh, linear_axis.zenith, linear_axis.azimuth)

    @property
    def lX_inteval(self) -> tuple:
        '''lX_inteval property getter'''
        return self._lX_inteval

    @lX_inteval.setter
    def lX_inteval(self, interval):
        '''lX_inteval angle property setter'''
        # if type(interval) != tuple:
        #     raise ValueError('lX interval needs to be a tuple')
        # if interval not in zip(self.lXs[:-1], self.lXs[1:]):
        #     raise ValueError('lX interval not in tabulated ranges.')
        self._lX_inteval = interval

    @property
    def delta(self) -> np.ndarray:
        '''Override of the delta property so each one corresponds to its
        respective mesh point.'''
        return self._d

    @property
    def vectors(self) -> np.ndarray:
        '''axis vector property definition

        returns vectors from the origin to mesh axis points.
        '''
        return self.rotated_mesh

    @property
    def r(self) -> np.ndarray:
        '''overrided r property definition'''
        return vector_magnitude(self.rotated_mesh)

    @property
    def dr(self) -> np.ndarray:
        '''overrided dr property definition'''
        return self._dr

    @property
    def altitude(self) -> np.ndarray:
        '''overrided h property definition'''
        return self._a

    @property
    def X(self) -> np.ndarray:
        '''This method sets the depth along the shower axis attribute'''
        return self.linear_axis.X
    
    def slant_depth_integrand(self, h: float | np.ndarray) -> float | np.ndarray:
        return self.linear_axis.slant_depth_integrand(h)

    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''
        return self.linear_axis.distance(X)

    def theta(self, axis_vectors, counters: Counters) -> np.ndarray:
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''
        return self.linear_axis.theta(axis_vectors, counters)

    def get_timing_class(self) -> Timing:
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_timing_class()

    def get_attenuation_class(self) -> Attenuation:
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_attenuation_class()

    def get_gg_file(self) -> str:
        '''This method returns the gg array file for the axis' particular
        log(moliere) interval.
        '''
        start, end = self.find_nearest_interval()
        return f'gg_t_delta_theta_lX_{start}_to_{end}.npz'

    def find_nearest_interval(self) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1],self.lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]

    # def get_gg_file(self):
    #     '''This method returns the original gg array file.
    #     '''
    #     # return 'gg_t_delta_theta_lX_-6_to_-5.npz'
    #     # return 'gg_t_delta_theta_mc.npz'
    #     return 'gg_t_delta_theta_2020_normalized.npz'

    def reset_for_profile(self, shower: Shower) -> None:
        return self.linear_axis.reset_for_profile(shower)
    
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        return self.linear_axis.get_curved_atm_correction_class()

class MeshShower(Shower):
    '''This class is the implementation of a shower where the shower particles are
    distributed to a mesh axis rather than just the longitudinal axis.
    '''

    def __init__(self, mesh_axis: MeshAxis):
        self.mesh_axis = mesh_axis

    def stage(self, X: np.ndarray) -> np.ndarray:
        '''This method returns the corresponding stage of each mesh point.
        '''
        return self.mesh_axis._t

    def profile(self, X: np.ndarray) -> np.ndarray:
        '''This method returns the number of charged particles at each mesh
        point
        '''
        return self.mesh_axis.nch

class MakeUpwardAxis(Axis):
    '''This is the implementation of an axis for an upward going shower, depths
    are added along the axis in the upward direction'''

    @cached_property
    def X(self) -> np.ndarray:
        '''This method sets the depth attribute'''
        # rho = self.atm.density(self.altitude)
        # return cumulative_trapezoid(rho,self.r, initial=0.) / 10.
        # axis_deltaX = np.sqrt(rho[1:] * rho[:-1]) * self.dr[1:] / 10# converting to g/cm^2
        # return np.concatenate((np.array([0]),np.cumsum(axis_deltaX)))
        depths = np.zeros_like(self.r)
        A = self.altitude[:-1]
        B = self.altitude[1:]
        depths[1:] = np.array([quad(self.slant_depth_integrand,a,b)[0] for a,b in zip(A,B)])
        return np.cumsum(depths / 10.)

    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X, self.r)

    def theta(self, axis_vectors, counters: Counters) -> np.ndarray:
        '''In this case we need pi minus the interal angle across from the
        distance to the counter'''
        return np.pi - counters.calculate_theta(axis_vectors)
    
    def reset_for_profile(self, shower: Shower) -> None:
        '''This method resets the attributes of the class based on where the shower
        occurs on the axis. We dont need to run universality calculations where
        there's no shower.
        '''
        # ids = shower.profile(self.X) >= self.params.MIN_CHARGED_PARTICLES
        ids = shower.profile(self.X) >= self.params.MIN_CHARGED_PARTICLES * shower.N_max
        self.altitude = self.altitude[np.argmax(ids):]
        self.X = self.X[np.argmax(ids):]

class MakeUpwardAxisCurvedAtm(MakeUpwardAxis):
    '''This is the implementation of an upward going shower axis with a flat
    planar atmosphere'''

    def __repr__(self):
        return "UpwardAxisCurvedAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
                                        self.zenith, self.azimuth, self.ground_level)

    @property
    def r(self) -> np.ndarray:
        '''This is the axis distance property definition'''
        return self.h_to_axis_R_LOC(self.h, self.zenith)

    def slant_depth_integrand(self, alt: float | np.ndarray) -> float | np.ndarray:
        '''This is the integrand needed to calculate slant depth.
        '''
        num = (alt + self.earth_radius)
        denom = np.sqrt((np.cos(self.zenith)**2)*self.earth_radius**2 + alt**2 + 2*alt*self.earth_radius)
        dL_dz = num / denom
        return self.atm.density(alt) * dL_dz

    def get_timing_class(self) -> Timing:
        '''This method returns the upward flat atm timing class'''
        return UpwardTimingCurved

    def get_attenuation_class(self) -> Attenuation:
        '''This method returns the flat atmosphere attenuation object for upward
        axes'''
        return UpwardAttenuationCurved

    def get_gg_file(self) -> str:
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'
    
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        return UpwardCurvedCorrection

class UpwardCurvedCorrection(CurvedAtmCorrection):
    '''This is the implementation of the curved atmosphere integration for
    upward going showers.
    '''
    def __init__(self, axis: MakeUpwardAxisCurvedAtm, counters: Counters) -> None:
        self.axis = axis
        self.counters = counters
        self.cQ = counters.cos_Q(axis.vectors)
        self.cQd = np.cos(axis.theta_difference)
        self.sQd = np.sin(axis.theta_difference)
        self.Q = np.arccos(self.cQ)

    def curved_correction(self, vert: np.ndarray) -> np.ndarray:
        '''This is the integration.
        '''
        integrals = np.empty_like(self.Q)
        for i in range(integrals.shape[1]):
            test_Q = np.linspace(self.Q[:,i].min(), self.Q[:,i].max(), 5)
            test_cQ = np.cos(test_Q)
            test_sQ = np.sin(test_Q)
            t1 = test_cQ[:,np.newaxis] * self.cQd[i:] #these three lines are what's different for up vs down
            t2 = test_sQ[:,np.newaxis] * self.sQd[i:]
            test_integrals = np.sum(vert[i:] / (t1 + t2), axis = 1)
            integrals[:,i] = np.interp(self.Q[:,i], test_Q, test_integrals)
        return integrals

class UpwardTimingCurved(Timing):
    '''This is the implementation of timing for a upward going shower with
    correction for atmospheric curveature.
    '''

    def __init__(self, curved_correction: UpwardCurvedCorrection):
        self.curved_correction = curved_correction
        self.axis = curved_correction.axis
        self.counters = curved_correction.counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"UpwardTimingCurved(axis=({self.axis.__repr__}), \
                                   counters=({self.counters.__repr__}))"

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return self.axis.r / self.c / nano

    def delay(self) -> np.ndarray:
        '''This method returns the delay photons would experience when they
        travel from a point on the axis to a counter. The returned array is of
        shape: (# of counters, # of axis points)
        '''
        vsd = self.axis.delta * self.axis.dh / self.c / nano #vertical stage delay
        return self.curved_correction(vsd)
    
class UpwardAttenuationCurved(Attenuation):
    '''This is the implementation of signal attenuation for an upward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, curved_correction: UpwardCurvedCorrection, yield_array: np.ndarray):
        self.curved_correction = curved_correction
        self.axis = curved_correction.axis
        self.counters = curved_correction.counters
        self.yield_array = yield_array
        self.atm = self.axis.atm

    def log_fraction_passed(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            # log_frac_passed_list[i] = upward_curved_correction(self.axis, self.counters, v_log_frac)
            log_frac_passed_list[i] = self.curved_correction(v_log_frac)
        return log_frac_passed_list

