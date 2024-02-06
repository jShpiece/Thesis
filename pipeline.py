import numpy as np
import scipy.optimize as opt

# ------------------------------
# Classes
# ------------------------------

class Source:
    # Class to store source information. Each source has a position (x, y) 
    # and ellipticity (e1, e2) and flexion (f1, f2)
    def __init__(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg):
        self.x = x
        self.y = y
        self.e1 = e1
        self.e2 = e2
        self.f1 = f1
        self.f2 = f2
        self.g1 = g1
        self.g2 = g2
        self.sigs = sigs
        self.sigf = sigf
        self.sigg = sigg


    def filter_sources(self, a, max_flexion=0.5):
        # Make cuts in the source data based on size and flexion
        valid_indices = (np.abs(self.f1) <= max_flexion) & (np.abs(self.f2) <= max_flexion)
        self.x, self.y = self.x[valid_indices], self.y[valid_indices]
        self.e1, self.e2 = self.e1[valid_indices], self.e2[valid_indices]
        self.f1, self.f2 = self.f1[valid_indices], self.f2[valid_indices]
        self.g1, self.g2 = self.g1[valid_indices], self.g2[valid_indices]
        return valid_indices


    def generate_initial_guess(self):
        # Generate initial guesses for possible lens positions based on the source ellipticity and flexion
        phi = np.arctan2(self.f2, self.f1)
        gamma = np.sqrt(self.e1**2 + self.e2**2)
        flexion = np.sqrt(self.f1**2 + self.f2**2)
        gflexion = np.sqrt(self.g1**2 + self.g2**2)
        r = gamma / flexion # A characteristic distance from the source
        te = 2 * gamma * np.abs(r) # The Einstein radius of the lens


        return Lens(np.array(self.x + r * np.cos(phi)), np.array(self.y + r * np.sin(phi)), np.array(te), np.empty_like(self.x))


    def apply_lensing_effects(self, lenses):
        # Apply the lensing effects of a set of lenses to the sources
        if type(lenses.x) is not np.ndarray:
            lenses.x = np.array([lenses.x])
            lenses.y = np.array([lenses.y])
            lenses.te = np.array([lenses.te])

        for i in range(len(lenses.x)):
            dx = self.x - lenses.x[i]
            dy = self.y - lenses.y[i]
            r = np.sqrt(dx**2 + dy**2)

            cosphi = dx/r
            sinphi = dy/r
            cos2phi = cosphi*cosphi-sinphi*sinphi
            sin2phi = 2*cosphi*sinphi
            cos3phi = cosphi*cosphi*cosphi - 3*cosphi*sinphi*sinphi
            sin3phi = 3*cosphi*cosphi*sinphi - sinphi*sinphi*sinphi

            self.e1 += -lenses.te[i]/(2*r)*cos2phi
            self.e2 += -lenses.te[i]/(2*r)*sin2phi
            self.f1 += -dx*lenses.te[i]/(2*r*r*r)
            self.f2 += -dy*lenses.te[i]/(2*r*r*r)
            self.g1 += (3*lenses.te[i]) / (2*r**2) * cos3phi
            self.g2 += (3*lenses.te[i]) / (2*r**2) * sin3phi


class Lens:
    # Class to store lens information. Each lens has a position (x, y) and an Einstein radius (te)
    def __init__(self, x, y, te, chi2):
        self.x = x
        self.y = y
        self.te = te
        self.chi2 = chi2


    def optimize_lens_positions(self, sources, use_flags):
        # Given a set of initial guesses for lens positions, find the optimal lens positions
        # via local minimization
        max_attempts = 1
        for i in range(len(self.x)):
            one_source = Source(sources.x[i], sources.y[i], 
                                sources.e1[i], sources.e2[i], sources.f1[i], sources.f2[i], sources.g1[i], sources.g2[i],
                                sources.sigs[i], sources.sigf[i], sources.sigg[i])
            guess = [self.x[i], self.y[i], self.te[i]] # Class is already initialized with initial guesses
            best_result = None
            best_params = guess
            for _ in range(max_attempts):
                result = opt.minimize(
                    chi2wrapper, guess, args=([one_source, use_flags]), 
                    method='Nelder-Mead', 
                    tol=1e-8, 
                    options={'maxiter': 500}
                )

                if best_result is None or result.fun < best_result.fun:
                    best_result = result
                    best_params = result.x

            self.x[i], self.y[i], self.te[i] = best_params[0], best_params[1], best_params[2]
        

    def filter_lens_positions(self, sources, xmax, threshold_distance=0.1):
        # Filter out lenses that are too close to sources or too far from the center
        distances_to_sources = np.sqrt((self.x - sources.x)**2 + (self.y - sources.y)**2)
        too_close_to_sources = distances_to_sources < threshold_distance
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 2 * xmax
        zero_te_indices = np.abs(self.te) < 10**-3

        valid_indices = ~(too_close_to_sources | too_far_from_center | zero_te_indices)        
        self.x, self.y, self.te, self.chi2 = self.x[valid_indices], self.y[valid_indices], self.te[valid_indices], self.chi2[valid_indices]
    

    def merge_close_lenses(self, merger_threshold=5):
        #Merge lenses that are too close to each other
        i = 0
        while i < len(self.x):
            for j in range(i+1, len(self.x)):
                # Check every pair of lenses
                # If they are too close, merge them
                distance = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
                if distance < merger_threshold:
                    weight_i, weight_j = self.te[i], self.te[j]
                    self.x[i] = (self.x[i]*weight_i + self.x[j]*weight_j) / (weight_i + weight_j)
                    self.y[i] = (self.y[i]*weight_i + self.y[j]*weight_j) / (weight_i + weight_j)
                    self.te[i] = (weight_i + weight_j) 
                    self.x, self.y, self.te, self.chi2 = np.delete(self.x, j), np.delete(self.y, j), np.delete(self.te, j), np.delete(self.chi2, j)
                    break
            else:
                i += 1
    

    def select_lowest_chi2(self, lens_floor=1):
        # Simply choose the 'lens_floor' lenses with the lowest chi^2 values
        # and eliminate the rest
        
        # Sort the lenses by chi^2 value
        sorted_indices = np.argsort(self.chi2)
        self.x, self.y, self.te, self.chi2 = self.x[sorted_indices], self.y[sorted_indices], self.te[sorted_indices], self.chi2[sorted_indices]

        # Select the 'lens_floor' lenses with the lowest chi^2 values
        if len(self.x) > lens_floor:
            self.x, self.y, self.te, self.chi2 = self.x[:lens_floor], self.y[:lens_floor], self.te[:lens_floor], self.chi2[:lens_floor]


    def iterative_elimination(self, sources, reducedchi2, use_flags):
        # Iteratively eliminate lenses that do not improve the chi^2 value
        lens_floors = np.arange(1, len(self.x) + 1)
        best_dist = np.abs(reducedchi2 - 1)
        best_lenses = self
        for lens_floor in lens_floors:
            # Clone the lenses object
            test_lenses = Lens(self.x, self.y, self.te, self.chi2)
            test_lenses.select_lowest_chi2(lens_floor=lens_floor)
            reducedchi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = np.abs(reducedchi2 - 1)
            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses
        # Update the lenses object with the best set of lenses
        self.x, self.y, self.te, self.chi2 = best_lenses.x, best_lenses.y, best_lenses.te, best_lenses.chi2
    

    def full_minimization(self, sources, use_flags):
        guess = self.te
        params = [self.x, self.y, sources, use_flags]
        max_attempts = 5  # Number of optimization attempts with different initial guesses
        method = 'Powell'

        best_result = None
        best_params = guess

        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper2, guess, args=params,
                method=method,  
                tol=1e-8,  # Adjust tolerance for each attempt
                options={'maxiter': 1000}
            )

            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        
        # Notify the user if the optimization failed
        if not best_result.success:
            print("Optimization failed!")

        self.te = best_params


    def update_chi2_values(self, sources, use_flags):
        # Change the chi^2 values of the lenses to reflect the current set of sources
        # And return the reduced chi^2 value of the set of lenses
        for i in range(len(self.x)):
            one_lens = Lens(self.x[i], self.y[i], self.te[i], 0)
            self.chi2[i] = calc_raw_chi2(sources, one_lens, use_flags)
        dof = calc_degrees_of_freedom(sources, self, use_flags)
        total_chi2 = calc_raw_chi2(sources, self, use_flags) 
        return total_chi2 / dof


# ------------------------------
# Initialization functions
# ------------------------------

def createSources(lenses,ns=1,randompos=True,sigs=0.1,sigf=0.01,sigg=0.02,xmax=5):
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

    # Initialize lensing parameters with gaussian noise
    e1data = np.random.normal(0,sigs,ns)
    e2data = np.random.normal(0,sigs,ns)
    f1data = np.random.normal(0,sigf,ns)
    f2data = np.random.normal(0,sigf,ns)
    g1data = np.random.normal(0,sigg,ns)
    g2data = np.random.normal(0,sigg,ns)

    sources = Source(x, y, e1data, e2data, f1data, f2data, g1data, g2data, sigs*np.ones(ns), sigg*np.ones(ns), sigg*np.ones(ns))
    # Apply the lensing effects of the lenses
    sources.apply_lensing_effects(lenses)

    return sources


def createLenses(nlens=1,randompos=True,xmax=10,strength_choice='identical'):
    if randompos == True:
        xlarr = -xmax + 2*xmax*np.random.random(nlens)
        ylarr = -xmax + 2*xmax*np.random.random(nlens)
    else: #Uniformly spaced lenses
        xlarr = -xmax + 2*xmax*(np.arange(nlens)+0.5)/(nlens)
        ylarr = np.zeros(nlens)
    
    # Now we assign einstein radii based on the strength_choice
    if strength_choice == 'identical':
        tearr = np.ones(nlens)
    elif strength_choice == 'random':
        tearr = np.random.random(nlens) * 20
    elif strength_choice == 'uniform':
        tearr = np.linspace(0.1, 20, nlens)
    elif strength_choice == 'cluster':
        tearr = np.ones(nlens)
        tearr[0] = 10
    else:
        raise ValueError("Invalid strength_choice")

    
    lenses = Lens(xlarr, ylarr, tearr, np.empty(nlens))
    return lenses


# ------------------------------
# Chi^2 functions
# ------------------------------

def eR_penalty_function(eR, lower_limit=-20.0, upper_limit=20.0, lambda_penalty_upper=10.0):
    # Hard lower limit
    if eR < lower_limit:
        return 1e8 #Use an arbitrarily large number - NOT infinity (will yield NaNs)

    # Soft upper limit
    if eR > upper_limit:
        return lambda_penalty_upper * (eR - upper_limit) ** 2

    return 0.0


def calc_degrees_of_freedom(sources, lenses, use_flags):
    dof = (2 * np.sum(use_flags)) * len(sources.x) - 3 * len(lenses.x)
    if dof <= 0:
        return np.inf
    return dof


def calc_raw_chi2(sources, lenses, use_flags):
    # Compute the raw chi^2 value for a given set of sources and lenses
    # Determine which lensing signals to use, based on the use_flags
    use_shear = use_flags[0]
    use_flexion = use_flags[1]
    use_g_flexion = use_flags[2]
    # The source clone object represents the sources after the lensing effects of our 'test' lenses
    source_clone = Source(
        sources.x, sources.y, np.zeros_like(sources.e1), np.zeros_like(sources.e2), 
        np.zeros_like(sources.f1), np.zeros_like(sources.f2), np.zeros_like(sources.g1), np.zeros_like(sources.g2), 
        sources.sigs, sources.sigf, sources.sigg
        )
    source_clone.apply_lensing_effects(lenses) 
    
    # Now we compare the lensing signals on our source_clone object to the original sources
    chi1e = ((source_clone.e1 - sources.e1) / sources.sigs) ** 2
    chi2e = ((source_clone.e2 - sources.e2) / sources.sigs) ** 2
    chi1f = ((source_clone.f1 - sources.f1) / sources.sigf) ** 2
    chi2f = ((source_clone.f2 - sources.f2) / sources.sigf) ** 2
    chi1g = ((source_clone.g1 - sources.g1) / sources.sigg) ** 2
    chi2g = ((source_clone.g2 - sources.g2) / sources.sigg) ** 2


    shear_chi2 = (chi1e + chi2e) * use_shear #Check whether this run is using shear
    flexion_chi2 = (chi1f + chi2f) * use_flexion #Check whether this run is using flexion
    g_flexion_chi2 = (chi1g + chi2g) * use_g_flexion #Check whether this run is using g_flexion
    chi2 = np.sum(shear_chi2 + flexion_chi2 + g_flexion_chi2) 

    penalty = 0
    
    for eR in lenses.te:
        penalty += eR_penalty_function(eR)
    
    return chi2 + penalty


# ------------------------------
# Helper functions
# ------------------------------

def print_step_info(flags,message,lenses,reducedchi2):
    # Helper function to print out step information
    if flags:
        print(message)
        print('Number of lenses: ', len(lenses.x))
        if reducedchi2 is not None:
            print('Chi^2: ', reducedchi2)


def chi2wrapper(guess, params):
    # Wrapper function for chi2 to allow for minimization for a single lens object
    lenses = Lens(guess[0], guess[1], guess[2], [0])
    return calc_raw_chi2(params[0],lenses, params[1])


def chi2wrapper2(guess, params):
    # Wrapper function for chi2 to allow constrained minimization - only the einstein radii are allowed to vary
    # This time, the lens object contains the full set of lenses
    lenses = Lens(params[0], params[1], guess, np.empty_like(params[0]))
    dof = calc_degrees_of_freedom(params[2], lenses, params[3])
    return calc_raw_chi2(params[2],lenses, params[3]) / dof - 1


# ------------------------------
# Main function
# ------------------------------

def fit_lensing_field(sources, xmax, flags = False, use_flags = [True, True, True]):
    '''
    This function takes in a set of sources - with positions, ellipticity, and flexion 
    signals, and attempts to reconstruct the lensing field that produced them. 
    The lensing field is represented by a set of lenses - with positions and Einstein radii. 
    The lenses are modeled as Singular Isothermal Spheres (SIS). 

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
        The minimum number of lenses to keep - a target number of lenses based on priors
    use_flags : list of bool
        A list of booleans indicating which lensing signals to use
            0 - shear
            1 - flexion
            2 - g_flexion
    flags : bool
        Whether to print out step information
    '''


    # Initialize candidate lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "Initial Guesses:", lenses, reducedchi2)


    # Optimize lens positions via local minimization
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "Local Minimization:", lenses, reducedchi2)


    # Filter out lenses that are too close to sources or too far from the center
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Filtering:", lenses, reducedchi2)


    # Merge lenses that are too close to each other
    # Try a merger threshold based on the source density
    ns = len(sources.x) / (2 * xmax)**2
    merger_threshold = (1 / np.sqrt(ns))
    # If the merger threshold is too small, set it to 1
    if merger_threshold < 1:
        merger_threshold = 1
    lenses.merge_close_lenses(merger_threshold=10) #This is a placeholder value
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Merging:", lenses, reducedchi2)


    # Choose the 'lens_floor' lenses which gives the lowest reduced chi^2 value
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Iterative Elimination:", lenses, reducedchi2)


    # Perform a final local minimization on the remaining lenses
    # NOTE - if the number of lenses is too large, this step can take a long time
    # Right now, skip this step if there are more than 100 lenses
    if len(lenses.x) < 100:
        lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        print_step_info(flags, "After Final Minimization:", lenses, reducedchi2)


    return lenses, reducedchi2