import numpy as np
import scipy.optimize as opt

# ------------------------------
# Classes
# ------------------------------

class Source:
    # Class to store source information. Each source has a position (x, y) 
    # and ellipticity (e1, e2), flexion (f1, f2), and g_flexion (g1, g2) signals
    # as well as the standard deviations of these signals (sigs, sigf, sigg)
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
        r = gamma / flexion # A characteristic distance from the source
        te = 2 * gamma * r # The Einstein radius of the lens

        return Lens(np.array(self.x + r * np.cos(phi)), np.array(self.y + r * np.sin(phi)), np.array(te), np.empty_like(self.x))


    def apply_SIS_lensing(self, lenses):
        """
        Apply the lensing effects to the source using the Singular Isothermal Sphere (SIS) model. 
        This model primarily utilizes the Einstein radii of each lens to determine its effect.

        Parameters:
        - lenses: An object containing the lens properties (x, y, Einstein radii 'te').
                Expected to be arrays but can handle single values.
        """
        
        # Convert lens properties to numpy arrays of at least one dimension, ensuring floating point type
        lenses.x = np.array(lenses.x, ndmin=1, dtype=float)
        lenses.y = np.array(lenses.y, ndmin=1, dtype=float)
        lenses.te = np.array(lenses.te, ndmin=1, dtype=float)

        # Ensure the source properties are also of float type to handle the results of division and multiplication
        self.e1 = self.e1.astype(float)
        self.e2 = self.e2.astype(float)
        self.f1 = self.f1.astype(float)
        self.f2 = self.f2.astype(float)
        self.g1 = self.g1.astype(float)
        self.g2 = self.g2.astype(float)
        
        # Iterate over each lens to apply its effect
        for lens_x, lens_y, einstein_radius in zip(lenses.x, lenses.y, lenses.te):
            # Compute displacement from lens to source
            dx = self.x - lens_x
            dy = self.y - lens_y
            r = np.sqrt(dx**2 + dy**2)
            
            # Calculate trigonometric terms needed for lensing equations
            cos_phi = dx / r
            sin_phi = dy / r
            cos2phi = cos_phi * cos_phi - sin_phi * sin_phi
            sin2phi = 2 * cos_phi * sin_phi
            cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
            sin3phi = sin2phi * cos_phi + cos2phi * sin_phi
            
            # Apply SIS lensing effects for ellipticity (e1, e2), flexion (f1, f2), and g-flexion (g1, g2)
            self.e1 += -einstein_radius / (2 * r) * cos2phi
            self.e2 += -einstein_radius / (2 * r) * sin2phi
            self.f1 += -dx * einstein_radius / (2 * r**3)
            self.f2 += -dy * einstein_radius / (2 * r**3)
            self.g1 += (3 * einstein_radius) / (2 * r**2) * cos3phi
            self.g2 += (3 * einstein_radius) / (2 * r**2) * sin3phi

    
    def apply_NFW_lensing(self, halos):
        # Apply the lensing effects of a set of halos to the sources
        # Model the halos as Navarro-Frenk-White (NFW) profiles
        # Then the primary parameters are the masses and concentrations of the halos

        gamma1, gamma2 = halos.calc_shear_signal(self.x, self.y)
        flex1, flex2 = halos.calc_F_signal(self.x, self.y)
        gflex1, gflex2 = halos.calc_G_signal(self.x, self.y)

        # Ensure the source properties are also of float type to handle the results of division and multiplication
        self.e1 = self.e1.astype(float)
        self.e2 = self.e2.astype(float)
        self.f1 = self.f1.astype(float)
        self.f2 = self.f2.astype(float)
        self.g1 = self.g1.astype(float)
        self.g2 = self.g2.astype(float)

        self.e1 += gamma1
        self.e2 += gamma2
        self.f1 += flex1
        self.f2 += flex2
        self.g1 += gflex1
        self.g2 += gflex2


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
                                sources.e1[i], sources.e2[i], 
                                sources.f1[i], sources.f2[i], 
                                sources.g1[i], sources.g2[i],
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
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 1 * xmax
        zero_te_indices = np.abs(self.te) < 10**-3

        valid_indices = ~(too_close_to_sources | too_far_from_center | zero_te_indices)        
        self.x, self.y, self.te, self.chi2 = self.x[valid_indices], self.y[valid_indices], self.te[valid_indices], self.chi2[valid_indices]


    def merge_close_lenses(self, merger_threshold=5):

        def perform_merger(i, j):
            # Given two lenses, merge them and place the new lens at the weighted average position
            # and with the average Einstein radius of the pair
            weight_i, weight_j = np.abs(self.te[i]), np.abs(self.te[j]) # Weights must be positive
            self.x[i] = (self.x[i]*weight_i + self.x[j]*weight_j) / (weight_i + weight_j)
            self.y[i] = (self.y[i]*weight_i + self.y[j]*weight_j) / (weight_i + weight_j)
            self.te[i] = (weight_i + weight_j) / 2
            self.x, self.y, self.te, self.chi2 = np.delete(self.x, j), np.delete(self.y, j), np.delete(self.te, j), np.delete(self.chi2, j)

        #Merge lenses that are too close to each other
        i = 0
        while i < len(self.x):
            for j in range(i+1, len(self.x)):
                # Check every pair of lenses
                # If they are too close, merge them
                distance = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
                if distance < merger_threshold:
                    perform_merger(i, j)
                    break
            else:
                i += 1
    

    def select_lowest_chi2(self, lens_floor=1):
        # Function that enables the iterative elimination of lenses
        # Select the 'lens_floor' lenses with the lowest chi^2 values
        
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
        best_result = None
        best_params = guess

        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper2, guess, args=params,
                method='Powell',  
                tol=1e-8,  
                options={'maxiter': 1000}
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        self.te = best_params


    def update_chi2_values(self, sources, use_flags):
        # Change the chi^2 values of the lenses to reflect the current set of sources
        # And return the reduced chi^2 value of the set of lenses
        for i in range(len(self.x)):
            one_lens = Lens(self.x[i], self.y[i], self.te[i], 0)
            self.chi2[i] = calculate_chi_squared(sources, one_lens, use_flags)
        dof = calc_degrees_of_freedom(sources, self, use_flags)
        total_chi2 = calculate_chi_squared(sources, self, use_flags) 
        return total_chi2 / dof


# ------------------------------
# Chi^2 functions
# ------------------------------

def eR_penalty_function(eR, limit=40.0, lambda_penalty_upper=1000.0):
    # Soft limits - allow the Einstein radius to be negative
    if np.abs(eR) > limit:
        return lambda_penalty_upper * (np.abs(eR) - limit) ** 2

    return 0.0


def calc_degrees_of_freedom(sources, lenses, use_flags):
    # Compute the number of degrees of freedom for a given set of sources and lenses
    # A source has 2 parameters per signal being used
    # A lens has 3 parameters
    dof = ((2 * np.sum(use_flags)) * len(sources.x)) - (3 * len(lenses.x))
    if dof <= 0:
        return np.inf
    return dof


def calculate_chi_squared(sources, lenses, flags) -> float:
    """
    Calculate the chi-squared statistic for the deviation of lensed source properties from their original values,
    considering specified lensing effects and adding penalties for certain lens properties.

    Parameters:
    - sources (Source): An object containing source properties and their uncertainties.
    - lenses (Lenses): An object representing the test lenses which affect the source properties.
    - flags (list of bool): Flags indicating which lensing effects to include [use_shear, use_flexion, use_g_flexion].

    Returns:
    - float: The total chi-squared value including penalties.
    """

    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = flags

    # Initialize a clone of sources with zeroed lensing signals
    source_clone = Source(
        x=sources.x, y=sources.y,
        e1=np.zeros_like(sources.e1), e2=np.zeros_like(sources.e2),
        f1=np.zeros_like(sources.f1), f2=np.zeros_like(sources.f2),
        g1=np.zeros_like(sources.g1), g2=np.zeros_like(sources.g2),
        sigs=sources.sigs, sigf=sources.sigf, sigg=sources.sigg
    )

    # Apply lensing effects to the cloned source
    source_clone.apply_SIS_lensing(lenses)

    # Calculate chi-squared for each lensing signal component
    chi_squared_components = {
        'shear': ((source_clone.e1 - sources.e1) ** 2 + (source_clone.e2 - sources.e2) ** 2) / sources.sigs**2,
        'flexion': ((source_clone.f1 - sources.f1) ** 2 + (source_clone.f2 - sources.f2) ** 2) / sources.sigf**2,
        'g_flexion': ((source_clone.g1 - sources.g1) ** 2 + (source_clone.g2 - sources.g2) ** 2) / sources.sigg**2
    }

    # Sum the chi-squared values, considering only the enabled lensing effects
    total_chi_squared = 0
    if use_shear:
        total_chi_squared += np.sum(chi_squared_components['shear'])
    if use_flexion:
        total_chi_squared += np.sum(chi_squared_components['flexion'])
    if use_g_flexion:
        total_chi_squared += np.sum(chi_squared_components['g_flexion'])

    # Calculate and add penalties for the lenses
    penalty = sum(eR_penalty_function(eR) for eR in lenses.te)

    # Return the total chi-squared including penalties
    return total_chi_squared + penalty


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
    return calculate_chi_squared(params[0],lenses, params[1])


def chi2wrapper2(guess, params):
    # Wrapper function for chi2 to allow constrained minimization - 
    #    only the einstein radii are allowed to vary
    # This time, the lens object contains the full set of lenses
    lenses = Lens(params[0], params[1], guess, np.empty_like(params[0]))
    dof = calc_degrees_of_freedom(params[2], lenses, params[3])
    # Return the reduced chi^2 value - 1, to be minimized
    return np.abs(calculate_chi_squared(params[2],lenses, params[3]) / dof - 1)


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

    # Choose the 'lens_floor' lenses which gives the best reduced chi^2 value
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Iterative Elimination:", lenses, reducedchi2)

    # Merge lenses that are too close to each other
    ns = len(sources.x) / (2 * xmax)**2
    merger_threshold = (1 / np.sqrt(ns)) 
    lenses.merge_close_lenses(merger_threshold=merger_threshold) #This is a placeholder value
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Merging:", lenses, reducedchi2)

    # Perform a final local minimization on the remaining lenses
    # NOTE - if the number of lenses is too large, this step can take a long time
    # Right now, skip this step if there are more than 100 lenses
    if len(lenses.x) < 100:
        lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        print_step_info(flags, "After Final Minimization:", lenses, reducedchi2)

    # No matter what signal combination was used, return a chi2 using all signals
    # This is because the final fit should be the best possible fit

    reducedchi2 = lenses.update_chi2_values(sources, [True, True, True])
    
    return lenses, reducedchi2