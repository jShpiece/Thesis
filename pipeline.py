import numpy as np
from utils import chi2, chi2wrapper, generate_combinations, lens
import scipy.optimize as opt


class Source:
    def __init__(self, x, y, e1, e2, f1, f2):
        self.x = x
        self.y = y
        self.e1 = e1
        self.e2 = e2
        self.f1 = f1
        self.f2 = f2
    
    def generate_initial_guess(self):
        phi = np.arctan2(self.f2, self.f1)
        gamma = np.sqrt(self.e1**2 + self.e2**2)
        flexion = np.sqrt(self.f1**2 + self.f2**2)
        r = gamma / flexion
        return Lens(self.x + r * np.cos(phi), self.y + r * np.sin(phi), 2 * gamma * np.abs(r))


class Lens:
    def __init__(self, x, y, te):
        self.x = x
        self.y = y
        self.te = te

    def optimize_lens_positions(self, sources, sigs, sigf):
        # Given a set of initial guesses for lens positions, find the optimal lens positions
        # via local minimization

        for i in range(len(self.x)):
            one_source = Source(sources.x[i], sources.y[i], sources.e1[i], sources.e2[i], sources.f1[i], sources.f2[i])
            params = [one_source, sigf, sigs]
            guess = [self.x[i], self.y[i], self.te[i]] # Class is already initialized with initial guesses\
            # Adjust the tolerance to make the minimization more accurate
            result = opt.minimize(chi2wrapper, guess, args=(params), method='Nelder-Mead', tol=1e-8)
            self.x[i], self.y[i], self.te[i] = result.x
        

    def filter_lens_positions(self, sources, xmax, threshold_distance=1):
        # Filter out lenses that are too close to sources or too far from the center
        distances_to_sources = np.sqrt((self.x[:, None] - sources.x)**2 + (self.y[:, None] - sources.y)**2)
        too_close_to_sources = (distances_to_sources < threshold_distance).any(axis=1)
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 2 * xmax

        valid_indices = ~(too_close_to_sources | too_far_from_center)
        self.x, self.y, self.te = self.x[valid_indices], self.y[valid_indices], self.te[valid_indices]
    

    def merge_close_lenses(self, merger_threshold=1):
        #Merge lenses that are too close to each other
        i = 0
        while i < len(self.x):
            for j in range(i+1, len(self.x)):
                distance = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
                if distance < merger_threshold:
                    weight_i, weight_j = self.te[i], self.te[j]
                    self.x[i] = (self.x[i]*weight_i + self.x[j]*weight_j) / (weight_i + weight_j)
                    self.y[i] = (self.y[i]*weight_i + self.y[j]*weight_j) / (weight_i + weight_j)
                    self.te[i] = (weight_i + weight_j) / 2
                    self.x, self.y, self.te = np.delete(self.x, j), np.delete(self.y, j), np.delete(self.te, j)
                    break
            else:
                i += 1
    
    def iterative_elimination(self, chi2val, sources, sigf, sigs, lens_floor=1):
        best_chi2val = chi2val
        best_indices = None

        for combination in generate_combinations(len(self.x), lens_floor):
            test_lens = Lens(self.x[list(combination)], self.y[list(combination)], self.te[list(combination)])
            new_chi2val = get_chi2_value(sources, test_lens, sigf, sigs) 
            if new_chi2val < best_chi2val:
                best_chi2val, best_indices = new_chi2val, combination
            
        if best_indices is not None:
            self.x, self.y, self.te = self.x[list(best_indices)], self.y[list(best_indices)], self.te[list(best_indices)]
        else:
            self.x, self.y, self.te = np.array([]), np.array([]), np.array([])


def createSources(xlarr,ylarr,tearr,ns=1,randompos=True,sigf=0.1,sigs=0.1,xmax=5):
    #Create sources for a lensing system and apply the lensing signal

    #Create the sources - require that they be distributed sphericaly
    if randompos == True:
        r = xmax*np.sqrt(np.random.random(ns))
        phi = 2*np.pi*np.random.random(ns)
    else: #Uniformly spaced sources - single choice of r, uniform phi
        r = xmax / 2
        phi = 2*np.pi*(np.arange(ns)+0.5)/(ns)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    #Apply the lens 
    e1data = np.zeros(ns)
    e2data = np.zeros(ns)
    f1data = np.zeros(ns)
    f2data = np.zeros(ns)

    for i in range(ns):
        e1data[i],e2data[i],f1data[i],f2data[i] = lens(x[i],y[i],xlarr,ylarr,tearr)
    
    #Add noise
    e1data += np.random.normal(0,sigs,ns)
    e2data += np.random.normal(0,sigs,ns)
    f1data += np.random.normal(0,sigf,ns)
    f2data += np.random.normal(0,sigf,ns)

    sources = Source(x, y, e1data, e2data, f1data, f2data)

    return sources


def print_step_info(flags,message,sources,lenses,chi2val):
    if flags:
        print(message)
        print('Number of lenses: ', len(lenses.x))
        print('Number of sources: ', len(sources.x))
        if chi2val is not None:
            print('Chi^2: ', chi2val)


def get_chi2_value(sources, lenses, sigf, sigs):
    dof = 4 * len(sources.x) - 3 * len(lenses.x)
    if dof <= 0:
        return np.inf # If dof is negative, there are more lenses than we can fit
    return chi2(sources, lenses.x, lenses.y, lenses.te, sigf, sigs) / dof


def fit_lensing_field(sources,sigs,sigf,xmax,lens_floor=1,flags = False):
    '''
    Given a set of sources and their lensing parameters, find the optimal lens positions
    within a 3d parameter space (x, y, te) via local minimization. Then, filter out lenses
    that are too close to sources or too far from the center. Finally, iteratively eliminate
    lenses that do not improve the chi^2 value.

    Parameters
    ----------
    sources : Source
        A Source object containing the sources and their lensing parameters
    sigs : float
        The standard deviation of the source ellipticity
    sigf : float
        The standard deviation of the source flexion
    xmax : float
        The maximum distance from the center of the field to the edge
    lens_floor : int
        The minimum number of lenses to keep
    flags : bool
        Whether to print out step information
    '''

    # Initialize candidate lenses from source guesses
    lenses = sources.generate_initial_guess()
    chi2val = get_chi2_value(sources, lenses, sigf, sigs)

    # Optimize lens positions via local minimization
    lenses.optimize_lens_positions(sources, sigs, sigf)
    chi2val = get_chi2_value(sources, lenses, sigf, sigs)
    print_step_info(flags, "Initial chi^2:", sources, lenses, chi2val)
    
    # Filter out lenses that are too close to sources or too far from the center
    lenses.filter_lens_positions(sources, xmax)
    chi2val = get_chi2_value(sources, lenses, sigf, sigs)
    print_step_info(flags, "After winnowing:", sources, lenses, chi2val)

    # Merge lenses that are too close to each other
    lenses.merge_close_lenses()
    chi2val = get_chi2_value(sources, lenses, sigf, sigs)
    print_step_info(flags, "After merging:", sources, lenses, chi2val)
    
    # Iteratively eliminate lenses that do not improve the chi^2 value
    lenses.iterative_elimination(chi2val, sources, sigf, sigs, lens_floor=lens_floor)
    chi2val = get_chi2_value(sources, lenses, sigf, sigs)
    print_step_info(flags, "After iterative minimization:", sources, lenses, chi2val)

    return lenses, chi2val