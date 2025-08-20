from typing import Protocol, Any
import numpy as np
from astropy.table import QTable, Column
from astropy import units as u
from dataclasses import dataclass, field
from importlib.resources import as_file, files

from .shower import Shower
from .axis import Axis, Counters, MeshAxis, MeshShower, MakeSphericalCounters, CurvedAtmCorrection
from .generate_Cherenkov import MakeYield
from .cherenkov_photon_array import CherenkovPhotonArray

class Signal:
    '''This class calculates the Cherenkov signal from a given shower axis at
    given counters
    '''

    def __init__(self, shower: Shower, axis: Axis, counters: Counters, yield_array: list[MakeYield], gg: CherenkovPhotonArray, y: dict[str,np.ndarray]):
        self.shower = shower
        self.axis = axis
        self.gga = gg
        self.yield_file = y
        self.counters = counters
        self.yield_array = yield_array
        self.t = self.shower.stage(self.axis.X)
        self.t[self.t>14.] = 14.
        self.Nch = self.shower.profile(self.axis.X)
        self.theta = self.axis.theta(axis.vectors, counters)
        self.omega = self.counters.omega(self.axis.vectors)

    def __repr__(self):
        return f"Signal({self.shower.__repr__()}, {self.axis.__repr__()}, {self.counters.__repr__()})"

    def calculate_gg(self):
        '''This funtion returns the interpolated values of gg at a given deltas
        and thetas

        returns:
        the angular distribution values at the desired thetas
        The returned array is of size:
        (# of counters, # of axis points)
        '''
        gg = np.empty_like(self.theta)
        for i in range(gg.shape[1]):
            gg_td = self.gga.angular_distribution(self.t[i], self.axis.delta[i])
            gg[:,i] = np.interp(self.theta[:,i], self.gga.theta, gg_td)
        gg[self.theta>np.pi/2] = 0.
        return gg
    
    @property
    def photon_array_shape(self) -> tuple:
        '''This property is the shape of the outputted photons array.
        '''
        return (self.counters.N_counters,len(self.yield_array),self.axis.r.size)

    def calculate_yield(self, y: MakeYield):
        ''' This function returns the total number of Cherenkov photons emitted
        at a given stage of a shower per all solid angle.

        returns: the total number of photons per all solid angle
        size: (# of axis points)
        '''
        Y = y.y_list(self.t, self.axis.delta)
        return 2. * self.Nch * self.axis.dr * Y
    
    def calculate_ng(self) -> np.ndarray:
        '''This method returns the number of Cherenkov photons going toward
        each counter from every axis bin

        The returned array is of size:
        (# of counters, # of yield bins, # of axis points)
        '''
        gg = self.calculate_gg()
        ng_array = np.empty(self.photon_array_shape)
        for i, y in enumerate(self.yield_array):
            y.set_yield_attributes(self.yield_file)
            ng_array[:,i,:] = gg * self.calculate_yield(y) * self.omega
        return ng_array

class Element(Protocol):
    '''This is the protocol for a simulation element. It needs a type, either
    axis, shower, counters, or yield'''
    @property
    def element_type(self) -> str:
        ...

    def create(self) -> Any:
        ...

def x_y_cx_cy(source_points: np.ndarray, counters: Counters) -> tuple[np.ndarray]:
    '''This function calculates the x and y directional cosines for paths from source points to 
    Cherenkov counters.
    '''
    if not isinstance(counters, MakeSphericalCounters):
        raise ValueError('Only spherical counters work when generating eventio format.')
    travel_vectors = counters.travel_vectors(source_points)
    travel_r = counters.travel_length(source_points)
    cx = travel_vectors[:,:,0] / travel_r
    cy = travel_vectors[:,:,1] / travel_r
    r = (counters.input_radius * np.sqrt(np.random.uniform(size=cx.shape)).reshape(cx.shape).T)
    phi = np.random.uniform(size=cx.shape).reshape(cx.shape) * 2 * np.pi
    x = r.T * np.cos(phi)
    y = r.T * np.sin(phi)
    return x*100, y*100, cx, cy #convert to cm

@dataclass
class ShowerSignal:
    '''This is a data container for a shower simulation's Cherenkov 
    Photons, arrival times and counting locations.
    '''
    counters: Counters #counters object
    axis: Axis #axis object
    shower: Shower #shower object
    source_points: np.ndarray = field(repr=False) #vectors to axis points
    wavelengths: np.ndarray = field(repr=False) #wavelength of each bin, shape = (N_wavelengths)
    photons: np.ndarray = field(repr=False) #number of photons from each step to each counter, shape = (N_counters, N_wavelengths, N_axis_points)
    times: np.ndarray = field(repr=False) #arrival times of photons from each step to each counter, shape = (N_counters, N_axis_points)
    charged_particles: np.ndarray = field(repr=False)
    depths: np.ndarray = field(repr=False)
    total_photons: np.ndarray = field(repr=False)
    cos_theta: np.ndarray = field(repr=False)

    def rand_xy(self) -> tuple[np.ndarray]:
        '''This method generates a random x and y in the ellipse made by the 
        shadow of the spherical detector.
        '''
        travel_vectors = self.counters.travel_vectors(self.source_points)
        cosQ = self.counters.cos_Q(self.source_points)
        incoming_angle = np.arctan2(travel_vectors[:,:,1],travel_vectors[:,:,0])

        '''generate random points in a circle with radius of detector sphere.'''
        shape = (self.photons.shape[0],self.photons.shape[-1])
        r = (self.counters.input_radius*100 * np.sqrt(np.random.uniform(size=shape)).reshape(shape).T).T
        phi = np.random.uniform(size=shape).reshape(shape) * 2 * np.pi

        a = r / cosQ #major axis length

        '''Transform points to detector shadow.'''
        x = a * np.cos(incoming_angle) * np.cos(phi) - r * np.sin(incoming_angle) * np.sin(phi)
        y = a * np.sin(incoming_angle) * np.cos(phi) + r * np.cos(incoming_angle) * np.sin(phi)
        return x, y

    def cx_cy(self) -> tuple[np.ndarray]:
        '''This method calculates the directional cosines x and y for the rays
        from the source points to the counters.
        '''
        travel_vectors = self.counters.travel_vectors(self.source_points)
        travel_r = np.sqrt((travel_vectors**2).sum(axis = -1))
        cx = travel_vectors[:,:,0] / travel_r
        cy = travel_vectors[:,:,1] / travel_r
        return cx, cy

    def get_bunches(self, tel_id: int) -> np.ndarray:
        '''This method returns a list of photon bunches for the shower.
        Each has an x and y relative to the telescope, directional cosines
        cx and cy for the incoming ray, arrival time, source height (zem),
        number of photons in the bunch, and wavelength.
        The returned array is of shape (N_axis_points*N_wavelengths,8)
        '''
        self.x, self.y = self.rand_xy()
        self.cx, self.cy = self.cx_cy()
        photons = self.photons[tel_id]
        filter_mask = photons.sum(axis=0) > 1.e-5
        photons = photons[:,filter_mask]
        bunches = np.empty((photons.shape[0], photons.shape[1], 8), dtype=float)
        for i, l in enumerate(self.wavelengths):
            bunches[i,:,0] = self.x[tel_id,filter_mask]
            bunches[i,:,1] = self.y[tel_id,filter_mask]
            bunches[i,:,2] = self.cx[tel_id,filter_mask]
            bunches[i,:,3] = self.cy[tel_id,filter_mask]
            bunches[i,:,4] = self.times[tel_id,filter_mask]
            bunches[i,:,5] = self.source_points[filter_mask,2]*100 #convert to cm
            bunches[i,:,6] = photons[i,:]
            bunches[i,:,7] = -l
        return bunches.reshape((-1,8)).astype(np.float32)

def diffunc(r, rc=32.47947175, s=-22.11021685):
    return np.exp(-(r-s)/rc) 

class ShowerSimulation:
    '''This class is the framework for creating a simulation'''
    lXs = np.linspace(-6,1,15)
    lX_intervals = list(zip(lXs[:-1], lXs[1:]))

    def __init__(self):
        self.ingredients = {
        'axis': None,
        'shower': None,
        'counters': None,
        'yield': None
        }
        self.table_lXs = np.arange(-6,0)
        self.table_lX_intervals = list(zip(self.table_lXs[:-1], self.table_lXs[1:]))
        self.lX_mids = np.array([np.mean(interval) for interval in self.table_lX_intervals])
        self.yield_files = {interval:self.load_table_file('y_t_delta_lX_', interval) for interval in self.table_lX_intervals}
        self.ggs = {interval:CherenkovPhotonArray(self.load_table_file('gg_t_delta_theta_lX_', interval)) for interval in self.table_lX_intervals}
        with as_file(files('nuspacesim.data.CHASM_tables')/'gg_t_delta_theta_mc.npz') as file:
            self.linear_gg = CherenkovPhotonArray(np.load(file))
        with as_file(files('nuspacesim.data.CHASM_tables')/'y_t_delta.npz') as file:
            self.linear_y = np.load(file)

    @staticmethod
    def load_table_file(prefix: str, interval: tuple[int]) -> dict[str:np.ndarray]:
        '''This method loads a table file from the data directory.
        '''
        filename = prefix + f'{interval[0]}_to_{interval[1]}.npz'
        with as_file(files('nuspacesim.data.CHASM_tables')/filename) as file:
            arraydict = np.load(file)
        return arraydict

    def find_nearest_interval(self, lX: float) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.table_lXs[:-1], lX)
        if index == 0:
            return self.table_lXs[0], self.table_lXs[1]
        else:
            return self.table_lXs[index-1], self.table_lXs[index]

    def add(self, element: Element):
        '''Add a element to the list of elements for the sim to perform'''
        self.ingredients[element.element_type] = element.create()
        if self.has_all_elements():
            self.set_sim()

    def remove(self, type):
        '''Remove all ingredients of a certain type from simulation'''
        self.ingredients[type] = None

    def has_all_elements(self) -> bool:
        '''This method checks to see if the simulation has the neccesary
        elements to generate a Cherenkov signal.
        '''
        for element_type in self.ingredients:
            if self.ingredients[element_type] == None:
                return False
        return True

    @property
    def shower(self) -> Shower:
        '''Simulation shower property'''
        return self._shower
    
    @shower.setter
    def shower(self, shower: Shower) -> None:
        self._shower = shower

    @property
    def axis(self) -> Axis:
        '''Simulation axis property'''
        return self._axis
    
    @axis.setter
    def axis(self, axis: Axis) -> None:
        self._axis = axis
    
    @property
    def y(self) -> list[MakeYield]:
        '''Simulation yield property'''
        return self._y
    
    @y.setter
    def y(self, y: list[MakeYield]) -> None:
        self._y = y

    @property
    def counters(self) -> Counters:
        '''Simulation counters property'''
        return self._counters
    
    @counters.setter
    def counters(self, counters: Counters) -> None:
        self._counters = counters

    def set_sim(self) -> None:
        '''This method sets the attributes of the sim.
        '''
        self.shower = self.ingredients['shower']
        self.counters = self.ingredients['counters']
        self.y = self.ingredients['yield']
        self.axis = self.ingredients['axis']
        self.axis.reset_for_profile(self.shower)
        self.N_c = self.counters.N_counters

    @property
    def N_bunches_mesh(self) -> int:
        '''This property is the total number of Cherenkov photon sample points
        when the mesh option is used.
        '''
        return self.axis.config.N_IN_RING * self.axis.r.size * len(self.lX_intervals)

    @staticmethod
    def get_attenuated_photons_array(signal: Signal, curved_correction: CurvedAtmCorrection):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        attenuation = signal.axis.get_attenuation(curved_correction,signal.yield_array)
        fraction_array = attenuation.fraction_passed()
        photons_array = signal.calculate_ng()
        attenuated_photons = np.zeros_like(photons_array)
        for i_a, fractions in enumerate(fraction_array):
            attenuated_photons[:,i_a,:] = photons_array[:,i_a,:] * fractions
        return attenuated_photons

    def get_mesh_signal(self, att: bool) -> ShowerSignal:
        '''This method returns a ShowerSignal object with the photons calculated
        using mesh sampling.
        '''
        N_axis_points = self.axis.config.N_IN_RING * self.axis.r.size
        axis_vectors = np.empty((len(self.lX_intervals), N_axis_points, 3))
        photons_array = np.empty((self.N_c, len(self.y), len(self.lX_intervals), N_axis_points))
        cQ_array = np.empty((self.N_c, len(self.y), len(self.lX_intervals), N_axis_points))
        times_array = np.empty((self.N_c, len(self.lX_intervals), N_axis_points))
        charged_particle_array = np.empty((len(self.lX_intervals), N_axis_points))
        depth_array = np.empty_like(charged_particle_array)
        diff = diffunc(self.counters.r)

        #calculate signal at each mesh ring
        for i, lX in enumerate(self.lX_intervals):
            meshaxis = MeshAxis(lX, self.axis, self.shower)
            meshshower = MeshShower(meshaxis)
            table_interval = self.find_nearest_interval(meshaxis.lX)
            gg = self.ggs[table_interval]
            y = self.yield_files[table_interval]
            signal = Signal(meshshower,meshaxis,self.counters,self.y,gg,y)
            curved_correction = meshaxis.get_curved_atm_correction(self.counters)

            axis_vectors[i,:] = meshaxis.vectors

            if att:
                # p = self.get_attenuated_photons_array(signal, curved_correction)
                # photons_array[:,:,i] = p - (p.T * diff).T
                photons_array[:,:,i] = self.get_attenuated_photons_array(signal, curved_correction)
            else:
                photons_array[:,:,i] = signal.calculate_ng()

            times_array[:,i] = meshaxis.get_timing(curved_correction).counter_time()
            cQ_array[:,:,i] = curved_correction.cQ[:,np.newaxis,:]

            #also save profile info for charged particles distributed into the rings
            charged_particle_array[i] = meshaxis.nch
            depth_array[i] = meshaxis.meshX
        
        #sum photons at each depth step
        tot_at_X = photons_array.sum(axis=2).sum(axis=1).sum(axis=0).reshape(self.axis.r.size,-1).sum(axis=1)

        #flatten over mesh rings
        photons_array = photons_array.reshape((photons_array.shape[0],photons_array.shape[1],-1))
        cQ_array = cQ_array.reshape((photons_array.shape[0],photons_array.shape[1],-1))
        times_array = times_array.reshape((times_array.shape[0],-1))
        axis_vectors = axis_vectors.reshape((-1,3))    
        charged_particle_array = charged_particle_array.flatten()
        depth_array = depth_array.flatten()
        
        return ShowerSignal(self.counters, 
                            self.axis,
                            self.shower,
                            axis_vectors, 
                            np.array([y.l_mid for y in self.y]),
                            photons_array,
                            times_array,
                            charged_particle_array,
                            depth_array,
                            tot_at_X,
                            cQ_array[:,0,:]) #just take the first wavelength entry, the angles are the same for all

    def get_signal(self, att: bool) -> ShowerSignal:
        '''This method returns a ShowerSignal object with the photons calculated
        along the axis.
        '''
        signal = Signal(self.shower,self.axis,self.counters,self.y,self.linear_gg,self.linear_y)
        curved_correction = self.axis.get_curved_atm_correction(self.counters)

        if att:
            photons_array = self.get_attenuated_photons_array(signal, curved_correction)
        else:
            photons_array = signal.calculate_ng()
        times_array = self.axis.get_timing(curved_correction).counter_time()
        cq = curved_correction.cQ
        return ShowerSignal(self.counters, 
                            self.axis,
                            self.shower,
                            self.axis.vectors, 
                            np.array([y.l_mid for y in self.y]),
                            photons_array,
                            times_array,
                            self.shower.profile(self.axis.X),
                            self.axis.X,
                            photons_array.sum(axis=1).sum(axis=0),
                            cq.reshape((cq.shape[0],-1,cq.shape[1])))

    def run(self, mesh: bool = False, att: bool = False) -> ShowerSignal:
        '''This method calculates the Cherenkov signal of a shower, and 
        stores it in a ShowerSignal object.
        '''
        if not self.has_all_elements():
            print('Sim needs a shower, counters, axis, and yield.')
            return None

        if mesh:
            return(self.get_mesh_signal(att))
        else:
            return(self.get_signal(att))

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
