import numpy as np
from utils import chi2, generate_combinations, lens
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor

# ------------------------------
# Classes
# ------------------------------

class Source:
    # Class to store source information. Each source has a position (x, y) 
    # and ellipticity (e1, e2) and flexion (f1, f2)
    def __init__(self, x, y, e1, e2, f1, f2):
        self.x = x
        self.y = y
        self.e1 = e1
        self.e2 = e2
        self.f1 = f1
        self.f2 = f2

    def filter_sources(self, emax=0.5, fmax = 0.5):
        # This function removes sources that we suspect are strongly lensed
        # based on their ellipticity and flexion
        # For now, we use a simple cutoff

        # Filter out sources with too large of an ellipticity
        e = np.sqrt(self.e1**2 + self.e2**2)
        valid_indices = e < emax

        # Filter out sources with too large of a flexion
        f = np.sqrt(self.f1**2 + self.f2**2)
        valid_indices = valid_indices & (f < fmax)

        self.x, self.y, self.e1, self.e2, self.f1, self.f2 = self.x[valid_indices], self.y[valid_indices], self.e1[valid_indices], self.e2[valid_indices], self.f1[valid_indices], self.f2[valid_indices]

    
    def generate_initial_guess(self):
        # Generate initial guesses for lens positions from source positions
        phi = np.arctan2(self.f2, self.f1)
        gamma = np.sqrt(self.e1**2 + self.e2**2)
        flexion = np.sqrt(self.f1**2 + self.f2**2)
        r = gamma / flexion
        return Lens(self.x + r * np.cos(phi), self.y + r * np.sin(phi), 2 * gamma * np.abs(r), 0)


class Lens:
    # Class to store lens information. Each lens has a position (x, y) and an Einstein radius (te)
    def __init__(self, x, y, te, chi2):
        self.x = x
        self.y = y
        self.te = te
        self.chi2 = chi2

    # The following functions are used to optimize lens positions, fitting the lens positions
    # and strengths to a source population

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
        # Okay, what if we just chose the 'lens_floor' lenses that each had the lowest chi2 value?

        # First, sort the lenses by chi2 value
        sorted_indices = np.argsort(self.chi2)
        self.x, self.y, self.te = self.x[sorted_indices], self.y[sorted_indices], self.te[sorted_indices]

        # Then, choose the 'lens_floor' lenses with the lowest chi2 value
        self.x, self.y, self.te = self.x[:lens_floor], self.y[:lens_floor], self.te[:lens_floor]
        # And... that's it. We don't need to do anything else here.
        '''
        # Go through all possible combinations of 'lens_floor' lenses and return the combination
        # that minimizes the chi^2 value. Repeat until the chi^2 value stops improving.
        best_chi2val = chi2val
        best_indices = None

        combinations = list(generate_combinations(len(self.x), lens_floor))
        for combination in combinations:
            test_lenses = Lens(self.x[list(combination)], self.y[list(combination)], self.te[list(combination)], self.chi2[list(combination)])
            test_chi2val = np.sum(test_lenses.chi2) / (4 * len(sources.x) - 3 * len(test_lenses.x))
            if test_chi2val < best_chi2val:
                best_chi2val = test_chi2val
                best_indices = combination

        if best_indices is not None:
            self.x, self.y, self.te = self.x[list(best_indices)], self.y[list(best_indices)], self.te[list(best_indices)]
        else:
            self.x, self.y, self.te = np.array([]), np.array([]), np.array([])

        '''

    def update_chi2_values(self, sources, sigf, sigs):
        # Calculate the raw chi^2 value for each lens, given a set of sources
        self.chi2 = np.zeros(len(self.x))
        for i in range(len(self.x)):
            one_lens = Lens(self.x[i], self.y[i], self.te[i], 0)
            self.chi2[i] = chi2(sources, one_lens, sigf, sigs)


# ------------------------------
# Initialization functions
# ------------------------------

def createSources(lenses,ns=1,randompos=True,sigf=0.1,sigs=0.1,xmax=5):
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
        e1data[i],e2data[i],f1data[i],f2data[i] = lens(x[i],y[i],lenses)
    
    #Add noise
    e1data += np.random.normal(0,sigs,ns)
    e2data += np.random.normal(0,sigs,ns)
    f1data += np.random.normal(0,sigf,ns)
    f2data += np.random.normal(0,sigf,ns)

    sources = Source(x, y, e1data, e2data, f1data, f2data)
    # Remove sources that we suspect are strongly lensed
    sources.filter_sources()

    return sources


def createLenses(nlens=1,randompos=True,xmax=10):
    #For now, fix theta_E at 1
    tearr = np.ones(nlens) 
    if randompos == True:
        xlarr = -xmax + 2*xmax*np.random.random(nlens)
        ylarr = -xmax + 2*xmax*np.random.random(nlens)
    else: #Uniformly spaced lenses
        xlarr = -xmax + 2*xmax*(np.arange(nlens)+0.5)/(nlens)
        ylarr = np.zeros(nlens)
    
    lenses = Lens(xlarr, ylarr, tearr, 0)
    return lenses


# ------------------------------
# Helper functions
# ------------------------------

def print_step_info(flags,message,sources,lenses,chi2val):
    # Helper function to print out step information
    if flags:
        print(message)
        print('Number of lenses: ', len(lenses.x))
        print('Number of sources: ', len(sources.x))
        if chi2val is not None:
            print('Chi^2: ', chi2val)


def chi2wrapper(guess,params):
    # Wrapper function for chi2 to allow for minimization
    lenses = Lens(guess[0], guess[1], guess[2], 0)
    return chi2(params[0],lenses,params[1],params[2])


def get_chi2_value(sources, lenses):
    # Calculate the chi^2 value for a given set of sources and lenses
    # Compute as the reduced chi^2 value
    dof = 4 * len(sources.x) - 3 * len(lenses.x)
    if dof <= 0:
        return np.inf # If dof is negative, there are more lenses than we can fit
    return np.sum(lenses.chi2) / dof


# ------------------------------
# Main function
# ------------------------------

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
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)

    # Optimize lens positions via local minimization
    lenses.optimize_lens_positions(sources, sigs, sigf)
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)
    print_step_info(flags, "Initial chi^2:", sources, lenses, chi2val)
    
    # Filter out lenses that are too close to sources or too far from the center
    lenses.filter_lens_positions(sources, xmax)
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)
    print_step_info(flags, "After winnowing:", sources, lenses, chi2val)

    # Merge lenses that are too close to each other
    lenses.merge_close_lenses()
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)
    print_step_info(flags, "After merging:", sources, lenses, chi2val)
    
    # Iteratively eliminate lenses that do not improve the chi^2 value
    lenses.iterative_elimination(chi2val, sources, sigf, sigs, lens_floor=lens_floor)
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)
    print_step_info(flags, "After iterative minimization:", sources, lenses, chi2val)

    return lenses, chi2val

    # Perform a final local minimization on the remaining lenses
    guess = np.concatenate((lenses.x, lenses.y, lenses.te))
    params = [sources, sigf, sigs]
    result = opt.minimize(chi2wrapper, guess, args=(params), method='Nelder-Mead', tol=1e-8)
    lenses = Lens(result.x[0:len(lenses.x)], result.x[len(lenses.x):2*len(lenses.x)], result.x[2*len(lenses.x):], 0)
    lenses.update_chi2_values(sources, sigf, sigs)
    chi2val = get_chi2_value(sources, lenses)
    print_step_info(flags, "After final minimization:", sources, lenses, chi2val)

    return lenses, chi2val