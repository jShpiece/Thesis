import numpy as np
import scipy.optimize as opt
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
import utils # Homemade utility functions
import minimizer # Homemade minimizer module

# ------------------------------
# Constants
# ------------------------------
M_solar = 1.989 * 10**30 # kg
G = 6.67430 * 10**-11 # m^3 kg^-1 s^-2
c = 299792458 # m/s

# ------------------------------
# Classes
# ------------------------------

class Source:
    # Class to store source information. Each source has a position (x, y) 
    # and ellipticity (e1, e2), flexion (f1, f2), and g_flexion (g1, g2) signals
    # as well as the standard deviations of these signals (sigs, sigf, sigg)
    def __init__(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg):
        # Make sure all inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.e1 = np.atleast_1d(e1)
        self.e2 = np.atleast_1d(e2)
        self.f1 = np.atleast_1d(f1)
        self.f2 = np.atleast_1d(f2)
        self.g1 = np.atleast_1d(g1)
        self.g2 = np.atleast_1d(g2)
        self.sigs = np.atleast_1d(sigs)
        self.sigf = np.atleast_1d(sigf)
        self.sigg = np.atleast_1d(sigg)


    def filter_sources(self, max_flexion=0.1):
        # Make cuts in the source data based on size and flexion
        valid_indices = (np.abs(self.f1) <= max_flexion) & (np.abs(self.f2) <= max_flexion)
        self.x, self.y = self.x[valid_indices], self.y[valid_indices]
        self.e1, self.e2 = self.e1[valid_indices], self.e2[valid_indices]
        self.f1, self.f2 = self.f1[valid_indices], self.f2[valid_indices]
        self.g1, self.g2 = self.g1[valid_indices], self.g2[valid_indices]
        self.sigs = self.sigs[valid_indices]
        self.sigf = self.sigf[valid_indices]
        self.sigg = self.sigg[valid_indices]
        return valid_indices


    def generate_initial_guess(self, lens_type='SIS', z_l = 0.5, z_s = 0.8):
        # Generate initial guesses for possible lens positions based on the source ellipticity and flexion
        phi = np.arctan2(self.f2, self.f1)
        gamma = np.sqrt(self.e1**2 + self.e2**2)
        flexion = np.sqrt(self.f1**2 + self.f2**2)

        if lens_type == 'SIS':
            r = gamma / flexion # A characteristic distance from the source
            te = 2 * gamma * r # The Einstein radius of the lens
            return Lens(np.array(self.x + r * np.cos(phi)), np.array(self.y + r * np.sin(phi)), np.array(te), np.empty_like(self.x))
        elif lens_type == 'NFW':
            r = 1.45 * gamma / flexion # A characteristic distance from the source

            def estimate_nfw_mass_from_flexion(mass, params):
                # unpack params
                xl, yl, xs, ys, f1, f2 = params
                # Create a halo
                halo = Halo(xl, yl, np.zeros_like(xl), np.array([5]), np.array(mass), np.array([z_l]), np.empty_like(xl))
                halo.calculate_concentration()
                # create a source
                source = Source(xs, ys, np.zeros_like(xs), np.zeros_like(ys), f1, f2, np.zeros_like(f1), np.zeros_like(f2), np.zeros_like(f1), np.zeros_like(f1), np.zeros_like(f1))
                # get the flexion signals
                shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(halo, source, z_s)
                # Use this as a function to minimize - find the mass that minimizes the difference between the flexion signals
                return np.sqrt((flex_1 - f1)**2 + (flex_2 - f2)**2)
            
            xl = np.array(self.x + r * np.cos(phi))
            yl = np.array(self.y + r * np.sin(phi))

            # Estimate the mass of the NFW halo
            masses = []
            for i in range(len(self.x)):
                mass_guess = 10**13 # Initial guess for the mass
                limits = [(10**10, 10**16)] # Mass limits
                result = opt.minimize(estimate_nfw_mass_from_flexion, mass_guess, args=([xl[i], yl[i], self.x[i], self.y[i], self.f1[i], self.f2[i]]), method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000}, bounds=limits)
                masses.append(result.x[0] * 2)
            
            halo = Halo(xl, yl, np.zeros_like(xl), np.array([5]), np.array(masses), np.array(z_l), np.empty_like(xl))
            halo.calculate_concentration()
            return halo


    def apply_noise(self):
        # Apply noise to the source - lensing properties
        self.e1 += np.random.normal(0, self.sigs)
        self.e2 += np.random.normal(0, self.sigs)
        self.f1 += np.random.normal(0, self.sigf)
        self.f2 += np.random.normal(0, self.sigf)
        self.g1 += np.random.normal(0, self.sigg)
        self.g2 += np.random.normal(0, self.sigg)


    def apply_SIS_lensing(self, lenses):
        """
        Apply the lensing effects to the source using the Singular Isothermal Sphere (SIS) model. 
        This model primarily utilizes the Einstein radii of each lens to determine its effect.

        Parameters:
        - lenses: An object containing the lens properties (x, y, Einstein radii 'te').
                Expected to be arrays but can handle single values.
        """
        
        e1, e2, f1, f2, g1, g2 = utils.calculate_lensing_signals_sis(lenses, self)

        self.e1 += e1
        self.e2 += e2
        self.f1 += f1
        self.f2 += f2
        self.g1 += g1
        self.g2 += g2

    
    def apply_NFW_lensing(self, halos, z_source=0.8):

        # ...let's simplify
        # do this one lens and one source at a time
        '''
        for i in range(len(halos.x)):
            this_halo = Halo(halos.x[i], halos.y[i], halos.z[i], halos.concentration[i], halos.mass[i], halos.redshift, [0])
            for j in range(len(self.x)):
                this_source = Source(self.x[j], self.y[j], self.e1[j], self.e2[j], self.f1[j], self.f2[j], self.g1[j], self.g2[j], self.sigs[j], self.sigf[j], self.sigg[j])
                shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(this_halo, this_source, z_source)
                self.e1[j] += shear_1
                self.e2[j] += shear_2
                self.f1[j] += flex_1
                self.f2[j] += flex_2
                self.g1[j] += gflex_1
                self.g2[j] += gflex_2
        '''
        shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(halos, self, z_source)
        self.e1 += shear_1
        self.e2 += shear_2
        self.f1 += flex_1
        self.f2 += flex_2
        self.g1 += gflex_1
        self.g2 += gflex_2
        


class Lens:
    # Class to store lens information. Each lens has a position (x, y) and an Einstein radius (te)
    def __init__(self, x, y, te, chi2):
        # When initializing the Lens object, make sure all inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.te = np.atleast_1d(te)
        self.chi2 = np.atleast_1d(chi2)


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
                    chi2wrapper, guess, args=(['SIS','unconstrained',one_source, use_flags]), 
                    method='Nelder-Mead', 
                    tol=1e-8, 
                    options={'maxiter': 500}
                )

                if best_result is None or result.fun < best_result.fun:
                    best_result = result
                    best_params = result.x

            self.x[i], self.y[i], self.te[i] = best_params[0], best_params[1], best_params[2]


    def filter_lens_positions(self, sources, xmax, threshold_distance=0.5):
        # Filter out lenses that are too close to sources or too far from the center

        # Calculate the distances between each lens and each source
        distances_to_sources = np.sqrt((self.x[:, None] - sources.x)**2 + (self.y[:, None] - sources.y)**2)
        too_close_to_sources = distances_to_sources < threshold_distance
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 1 * xmax
        zero_te_indices = np.abs(self.te) < 10**-3

        valid_indices = np.logical_not(np.any(too_close_to_sources, axis=1)) & np.logical_not(too_far_from_center) & np.logical_not(zero_te_indices)
        # Does the shape of valid_indices match the shape of the lens positions?
        assert len(valid_indices) == len(self.x), "Valid indices must have the same length as the lens positions: {} vs {}".format(len(valid_indices), len(self.x))
        # Getting an error here - IndexError, too many indices for array
        # It seems like valid indices is a 2D array, while it should be a 1D array
        # I will try to flatten it
        valid_indices = valid_indices.flatten()
        self.x = self.x[valid_indices]
        self.y = self.y[valid_indices]
        self.te = self.te[valid_indices]
        self.chi2 = self.chi2[valid_indices]


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
        params = ['SIS','constrained',self.x, self.y, sources, use_flags]
        max_attempts = 5  # Number of optimization attempts with different initial guesses
        best_result = None
        best_params = guess

        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper, guess, args=params,
                method='Powell',  
                tol=1e-8,  
                options={'maxiter': 1000}
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        self.te = best_params


    def update_chi2_values(self, sources, use_flags):
        # Given a set of sources, update the chi^2 values for each lens
        global_chi2 = calculate_chi_squared(sources, self, use_flags, lensing='SIS')
        chi2_values = np.zeros(len(self.x))
        if len(self.x) == 1:
            # Only one lens - calculate the chi2 value for this lens
            chi2_values[0] = calculate_chi_squared(sources, self, use_flags, lensing='SIS', use_weights=True)
        else:
            for i in range(len(self.x)):
                # Only pass in the i-th lens
                one_lens = Lens(self.x[i], self.y[i], self.te[i], [0])
                chi2_values[i] = calculate_chi_squared(sources, one_lens, use_flags, lensing='SIS', use_weights=True)
        self.chi2 = chi2_values
        dof = calc_degrees_of_freedom(sources, self, use_flags)
        if dof == 0:
            print('Degrees of freedom is zero')
            return global_chi2
        return global_chi2 / dof


class Halo:
    # Class to store halo information. Each halo has a position (x, y, z), concentration, mass, redshift, and chi^2 value
    # Define a constant for the mass of the Sun

    def __init__(self, x, y, z, concentration, mass, redshift, chi2):
        # Initialize the halo object with the given parameters
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.z = np.atleast_1d(z)
        self.concentration = np.atleast_1d(concentration)
        self.mass = np.atleast_1d(mass)
        # Ensure the mass array is not empty
        if mass.size == 0:
            print(mass)
            raise ValueError('Mass cannot be empty')
        self.mass = np.atleast_1d(np.abs(mass)) # Masses must be positive
        self.redshift = redshift # Redshift of the cluster, assumed to be the same for all halos
        self.chi2 = np.atleast_1d(chi2)

    # --------------------------------------------
    # Halo Calculation Functions
    # --------------------------------------------

    def project_to_2D(self):
        """
        Projects a set of 3D points onto the plane formed by the first two principal eigenvectors.
        This will shift from our halos being in object 3D space to being in a projected 2D space.
        """

        # Sanity checks
        assert len(self.x) == len(self.y) == len(self.z), "The x, y, and z arrays must have the same length."
        assert self.x.ndim == self.y.ndim == self.z.ndim == 1, "The x, y, and z arrays must be 1D."
        assert len(self.x) > 1, "At least two points are required."

        # Combine the x, y, z coordinates into a single matrix
        points = np.vstack((self.x, self.y, self.z)).T

        # Calculate the covariance matrix
        cov_matrix = np.cov(points, rowvar=False)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project the points onto the plane formed by the first two principal eigenvectors
        projected_points = np.dot(points, eigenvectors[:, :2])

        x = projected_points[:, 0]
        y = projected_points[:, 1]
        # Make sure these are arrays
        if np.isscalar(x):
            x = np.array([x])
        if np.isscalar(y):
            y = np.array([y])

        self.x, self.y = x, y
        self.z = np.zeros(len(self.x)) # Set the z values to zero now that we are in 2D


    def calc_R200(self):
        # Compute the R200 radius for each halo
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value
        mass = np.abs(self.mass) * M_solar # Convert to kg - mass is allowed to be negative, but it should be positive for this calculation
        R200 = ((3 / (800 * np.pi)) * (mass / rho_c))**(1/3) # In meters
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        return (200/3) * (self.concentration**3) / (np.log(1 + self.concentration) - (self.concentration / (1 + self.concentration)))


    def calc_corresponding_einstein_radius(self, source_redshift):
        # Compute the Einstein radius for a given source redshift
        Dl, Ds, Dls = utils.angular_diameter_distances(self.redshift, source_redshift)
        eR = np.sqrt((4 * G * self.mass * M_solar) / (c**2) * (Dls / (Ds * Dl))) * 206265 # Convert to arcseconds
        # Return the Einstein radius - make sure its an array
        if np.isscalar(eR):
            eR = np.array([eR])
        return eR


    def calculate_concentration(self):
        # Compute the concentration parameter for each halo
        # This is done with the Duffy et al. (2008) relation
        # This relation is valid for 0 < z < 2 - this covers the range of redshifts we are interested in
        self.mass += 1e-10 # Add a small value to the mass to avoid division by zero
        self.concentration = 5.71 * (np.abs(self.mass) / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47) 

    # --------------------------------------------
    # Pipeline Functions
    # --------------------------------------------

    def update_chi2_values(self, sources, use_flags):
        # Given a set of sources, update the chi^2 values for each lens

        global_chi2 = calculate_chi_squared(sources, self, use_flags, lensing='NFW')
        dof = calc_degrees_of_freedom(sources, self, use_flags)
        reducedchi2 = global_chi2 / dof
        
        chi2_values = np.zeros(len(self.x))
        if len(self.x) == 1:
            # Only one lens - calculate the chi2 value for this lens
            chi2_values[0] = calculate_chi_squared(sources, self, use_flags, lensing='NFW', use_weights=False)
        else:
            for i in range(len(self.x)):
                # Only pass in the i-th lens
                one_halo = Halo(self.x[i], self.y[i], self.z[i], self.concentration[i], self.mass[i], self.redshift, [0])
                chi2_values[i] = calculate_chi_squared(sources, one_halo, use_flags, lensing='NFW', use_weights=False)
        self.chi2 = chi2_values
        if dof == 0:
            print('Degrees of freedom is zero')
            return global_chi2
        return reducedchi2


    def optimize_lens_positions(self, sources, use_flags):
        # Given a set of initial guesses for lens positions, find the optimal lens positions
        # via local minimization

        learning_rates = [1e-2, 1e-2, 1e-4] 
        num_iterations = 10**3
        beta1 = 0.9
        beta2 = 0.999

        for i in range(len(self.x)):
            guess = [self.x[i], self.y[i], np.log10(self.mass[i])] # Class is already initialized with initial guesses
            # guess = [self.x[i], self.y[i], self.mass[i]] # Class is already initialized with initial guesses
            params = ['NFW','unconstrained', sources, use_flags, self.concentration[i], self.redshift]
            result, _ = minimizer.adam_optimizer(chi2wrapper, guess, learning_rates, num_iterations, beta1, beta2, params=params)

            self.x[i], self.y[i], self.mass[i] = result[0], result[1], 10**result[2] # Optimize the mass in log space, then convert back to linear space
            # self.x[i], self.y[i], self.mass[i] = result[0], result[1], result[2] 
            self.calculate_concentration() # Remember to update the concentration parameter


    def filter_lens_positions(self, sources, xmax, threshold_distance=0.5):
        # Filter out halos that are too close to sources or too far from the center

        # Compute the distance of each lens from each source
        distances_to_sources = np.sqrt((self.x[:, None] - sources.x)**2 + (self.y[:, None] - sources.y)**2)
        # Identify lenses that are too close to sources
        too_close_to_sources = np.any(distances_to_sources < threshold_distance, axis=1)
        # Identify lenses that are too far from the center
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > xmax
        # Identify lenses with zero mass (or too small to be considered lenses)
        zero_mass_lenses = np.abs(self.mass) < 10**10
        # Identify lenses with a mass greater than one we could reasonably expect
        too_large_mass = np.abs(self.mass) > 10**16

        # Remove lenses that are too close to sources, too far from the center, or have zero mass
        valid_indices = ~too_close_to_sources & ~too_far_from_center & ~zero_mass_lenses & ~too_large_mass
        # Update the lens positions
        self.x, self.y = self.x[valid_indices], self.y[valid_indices]
        self.mass = self.mass[valid_indices]
        self.chi2 = self.chi2[valid_indices]
        self.concentration = self.concentration[valid_indices]


    def merge_close_lenses(self, merger_threshold=5):
        def perform_merger(i, j):
            # Given two lenses, merge them and place the new lens at the weighted average position
            # and with the average Einstein radius of the pair
            weight_i, weight_j = np.abs(self.mass[i]), np.abs(self.mass[j]) # Weights must be positive
            self.x[i] = (self.x[i]*weight_i + self.x[j]*weight_j) / (weight_i + weight_j)
            self.y[i] = (self.y[i]*weight_i + self.y[j]*weight_j) / (weight_i + weight_j)
            self.mass[i] = (weight_i + weight_j) / 2
            self.x, self.y, self.mass, self.chi2 = np.delete(self.x, j), np.delete(self.y, j), np.delete(self.mass, j), np.delete(self.chi2, j)
            self.concentration = np.delete(self.concentration, j)

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


    def iterative_elimination(self, sources, reducedchi2, use_flags):
        # Iteratively eliminate lenses that do not improve the chi^2 value

        def select_lowest_chi2(x, y, mass, concentration, chi2, lens_floor=1):
            # Function that enables the iterative elimination of lenses
            sorted_indices = np.argsort(chi2)
            x, y, mass, concentration, chi2 = x[sorted_indices], y[sorted_indices], mass[sorted_indices], concentration[sorted_indices], chi2[sorted_indices]
            if len(x) > lens_floor:
                x, y, mass, concentration, chi2 = x[:lens_floor], y[:lens_floor], mass[:lens_floor], concentration[:lens_floor], chi2[:lens_floor]

            return x, y, mass, concentration, chi2
        
        def filter_combinations(combinations):
            # We only need to keep 'unique' combinations - there's no difference between [1, 2] and [2, 1]
            # We can sort the combinations and then use np.unique to get the unique combinations
            sorted_combinations = np.sort(combinations, axis=1)
            unique_combinations = np.unique(sorted_combinations, axis=0)
            return unique_combinations
        
        def check_combinations(x, y, mass, concentration, chi2, ceil=5):
            # Check all possible combinations of lenses up to a certain number
            # Return the combination with the lowest chi2 value
            best_chi2 = np.inf
            best_x, best_y, best_mass, best_concentration = x, y, mass, concentration

            for i in range(2, ceil):
                # Check every combination of i lenses
                allowed_combinations = utils.find_combinations(range(len(x)), i) # This will return a list of tuples, which are the indices of the lenses to consider
                if allowed_combinations is None or len(allowed_combinations) == 0:
                    # If there are no allowed combinations, continue
                    continue
                allowed_combinations = filter_combinations(allowed_combinations)
                for combination in allowed_combinations:
                    # Clone the lenses object
                    test_x, test_y, test_mass, test_concentration, test_chi2 = x[list(combination)], y[list(combination)], mass[list(combination)], concentration[list(combination)], chi2[list(combination)]
                    test_lenses = Halo(test_x, test_y, np.zeros_like(test_x), test_concentration, test_mass, self.redshift, test_chi2)
                    reducedchi2 = test_lenses.update_chi2_values(sources, use_flags)
                    if reducedchi2 < best_chi2:
                        best_x, best_y, best_mass, best_concentration, best_chi2 = test_x, test_y, test_mass, test_concentration, reducedchi2
            return best_x, best_y, best_mass, best_concentration, best_chi2

        max_lenses = np.min([len(self.x) + 1, 1000]) # Maximum number of lenses to consider
        lens_floors = np.arange(1, max_lenses)
        best_dist = np.abs(reducedchi2 - 1)
        best_lenses = self
        for lens_floor in lens_floors:
            # Clone the lenses object
            x, y, mass, concentration, chi2 = select_lowest_chi2(self.x, self.y, self.mass, self.concentration, self.chi2, lens_floor=lens_floor)
            if len(x) < 20: # If there are more than 20 lenses, checking all combinations will be too slow
                x, y, mass, concentration, chi2 = check_combinations(x, y, mass, concentration, chi2)

            test_lenses = Halo(x, y, np.zeros_like(x), concentration, mass, self.redshift, chi2)


            reducedchi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = np.abs(reducedchi2 - 1)
            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses
        # Update the lenses object with the best set of lenses
        self.x, self.y, self.mass, self.chi2 = best_lenses.x, best_lenses.y, best_lenses.mass, best_lenses.chi2
        self.calculate_concentration()


    def full_minimization(self, sources, use_flags):
        '''
        learning_rates = [1e-4, 1e-5]  # Adjust learning rate for mass and concentration parameters
        num_iterations = 10**4
        
        for i in range(len(self.x)):
            # Do the minimization one lens at a time - hopefully this will drive the mass of false lenses to zero
            guess = [np.log10(self.mass[i]), self.concentration[i]]
            params = ['NFW','dual_constraint',self.x[i], self.y[i], self.redshift, sources, use_flags]
            result, path = minimizer.gradient_descent(chi2wrapper, guess, learning_rates=learning_rates, num_iterations=num_iterations, params=params)
            self.mass[i], self.concentration[i] = 10**result[0], result[1]
        '''

        # Okay - hail mary time
        # Perform a global optimization of the mass parameters
        from scipy.optimize import differential_evolution
        bounds = [(10**9, 10**16)] * len(self.mass) # This will be the bounds for the mass parameters
        params = ('NFW', 'constrained', self.x, self.y, self.redshift, self.concentration, sources, use_flags)
        result = differential_evolution(chi2wrapper, bounds, args=(params,))
        self.mass = result.x
        '''
        # Do the minimization one lens at a time - hopefully this will drive the mass of false lenses to zero
        # Try just minimizing the mass
        for i in range(len(self.x)):
            guess = [np.log10(self.mass[i])]
            learning_rates = [0.1]
            # Shape the learning rates to match the shape of the guess
            # Guess will be a 2D array, 2 x number of lenses
            # Learning rates will be a 2D array, 2 x 1
            # We need to reshape the learning rates to match the shape of the guess
            # learning_rates = np.array(learning_rates).reshape(-1, 1)
            params = ['NFW','constrained',self.x[i], self.y[i], self.redshift, self.concentration[i], sources, use_flags]
            result, path = minimizer.gradient_descent(chi2wrapper, guess, learning_rates=learning_rates, num_iterations=num_iterations, params=params)
            self.mass[i] = 10**result[0]
        '''
        # return path

# ------------------------------
# Chi^2 functions
# ------------------------------

def calc_degrees_of_freedom(sources, lenses, use_flags):
    # Compute the number of degrees of freedom for a given set of sources and lenses
    # A source has 2 parameters per signal (use_flags tells us how many signals are used)
    # A lens has 3 parameters (two positional, one 'strength')
    dof = ((2 * np.sum(use_flags)) * len(sources.x)) - (3 * len(lenses.x))
    if dof <= 0:
        return np.inf
    return dof


def calculate_chi_squared(sources, lenses, flags, lensing='SIS', use_weights = False) -> float:
    """
    Calculate the chi-squared statistic for the deviation of lensed source properties from their original values,
    considering specified lensing effects and adding penalties for certain lens properties.

    Parameters:
    - sources (Source): An object containing source properties and their uncertainties.
    - lenses (Lenses): An object representing the test lenses which affect the source properties.
    - flags (list of bool): Flags indicating which lensing effects to include [use_shear, use_flexion, use_g_flexion].
    - lensing (str): The lensing model to use ('SIS' or 'NFW').
    - use_weights (bool): Whether to weight the sources by lens-source distance in the chi-squared calculation.

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

    assert np.all(source_clone.e1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.e2 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.f1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.f2 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.g1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.g2 == 0), "The cloned source should have zeroed lensing signals."

    # Apply lensing effects to the cloned source
    if lensing == 'SIS':
        source_clone.apply_SIS_lensing(lenses)
    elif lensing == 'NFW':
        source_clone.apply_NFW_lensing(lenses)
    else:
        raise ValueError("Invalid lensing model: {}".format(lensing))

    # Calculate chi-squared for each lensing signal component
    chi_squared_components = {
        'shear': ((source_clone.e1 - sources.e1) ** 2 + (source_clone.e2 - sources.e2) ** 2) / sources.sigs**2,
        'flexion': ((source_clone.f1 - sources.f1) ** 2 + (source_clone.f2 - sources.f2) ** 2) / sources.sigf**2,
        'g_flexion': ((source_clone.g1 - sources.g1) ** 2 + (source_clone.g2 - sources.g2) ** 2) / sources.sigg**2
    }

    # Sum the chi-squared values, considering only the enabled lensing effects
    total_chi_squared = use_shear * chi_squared_components['shear'] + use_flexion * chi_squared_components['flexion'] + use_g_flexion * chi_squared_components['g_flexion']    

    if use_weights:
        weights = utils.compute_source_weights(lenses, sources)
        total_chi_squared = np.sum(weights * total_chi_squared)
    else:
        total_chi_squared = np.sum(total_chi_squared)

    def eR_penalty_function(eR, limit=40.0, lambda_penalty_upper=1000.0):
        # Soft limits - allow the Einstein radius to be negative
        if np.abs(eR) > limit:
            return lambda_penalty_upper * (np.abs(eR) - limit) ** 2

        return 0.0

    # Calculate and add penalties for the lenses
    if lensing == 'SIS':
        penalty = sum(eR_penalty_function(eR) for eR in lenses.te)
        total_chi_squared += penalty
    elif lensing == 'NFW':
        pass

    # Return the total chi-squared including penalties
    return total_chi_squared 


def chi2wrapper(guess, params):
    """
    Consolidated chi-squared wrapper function for various lensing models and constraints.
    This wrapper is passed to the optimizer to minimize the chi-squared value.
    
    Args:
        guess (list): List of guessed parameters.
        params (list): List of parameters required for lensing models.
        Different model types require different parameters, but all params must share the first two entries
        which are the model type and constraint type
        model_type (str): Type of lensing model ('SIS' or 'NFW').
        constraint_type (str): Type of constraint ('unconstrained' or 'constrained').
    
    Returns:
        float: Chi-squared value to be minimized.
    """

    # Check if params were passed as a tuple - if so, unpack them into a list
    if isinstance(params, tuple):
        params = list(params)

    model_type, constraint_type = params[0], params[1]
    params = params[2:]
    
    if model_type == 'SIS':
        if constraint_type == 'unconstrained':
            lenses = Lens(guess[0], guess[1], guess[2], [0])
            return calculate_chi_squared(params[0], lenses, params[1], use_weights=True)
        elif constraint_type == 'constrained':
            lenses = Lens(params[0], params[1], guess, np.empty_like(params[0]))
            dof = calc_degrees_of_freedom(params[2], lenses, params[3])
            return np.abs(calculate_chi_squared(params[2], lenses, params[3]) / dof - 1)
    
    elif model_type == 'NFW':
        if constraint_type == 'unconstrained':
            lenses = Halo(guess[0], guess[1], np.zeros_like(guess[0]), params[2], 10**guess[2], params[3], [0])
            # lenses = Halo(guess[0], guess[1], np.zeros_like(guess[0]), params[2], guess[2], params[3], [0])
            lenses.calculate_concentration()
            return calculate_chi_squared(params[0], lenses, params[1], lensing='NFW', use_weights=False)
        elif constraint_type == 'constrained':
            lenses = Halo(params[0], params[1], np.zeros_like(params[0]), params[3], guess, params[2], np.empty_like(params[0]))
            lenses.calculate_concentration() # Update the concentration based on the new mass
            return calculate_chi_squared(params[4], lenses, params[5], lensing='NFW', use_weights=False)
        elif constraint_type == 'dual_constraint':
            # In this case, we are optimizing both mass and concentration
            lenses = Halo(params[0], params[1], np.zeros_like(params[0]), guess[1], 10**guess[0], params[2], np.empty_like(params[0]))
            return calculate_chi_squared(params[3], lenses, params[4], lensing='NFW', use_weights=False)
    else:
        raise ValueError("Invalid lensing model: {}".format(model_type))

# ------------------------------
# Main function
# ------------------------------

def fit_lensing_field(sources, xmax, flags = False, use_flags = [True, True, True], lens_type='SIS'):
    '''
    This function takes in a set of sources - with positions, ellipticity, and flexion 
    signals, and attempts to reconstruct the lensing field that produced them. 
    The lensing field is represented by a set of lenses - with positions and Einstein radii. 
    The lenses are modeled as Singular Isothermal Spheres (SIS) by default, but can be
    modeled as NFW halos as well.
    Parameters:
    - sources (Source): An object containing source properties and their uncertainties.
    - xmax (float): The maximum distance from the center of the field to consider for lenses.
    - flags (bool): Whether to print out step information.
    - use_flags (list of bool): Flags indicating which lensing effects to include [use_shear, use_flexion, use_g_flexion].
    - lens_type (str): The type of lens to use - 'SIS' or 'NFW'.
    Returns:
    - lenses (Lens): An object representing the lenses that best fit the source properties.
    - reducedchi2 (float): The reduced chi-squared value for the best fit.
    '''

    def print_step_info(flags,message,lenses,reducedchi2):
        # Helper function to print out step information
        if flags:
            print(message)
            print('Number of lenses: ', len(lenses.x))
            if reducedchi2 is not None:
                print('Chi^2: ', reducedchi2)

    # Initialize candidate lenses from source guesses
    lenses = sources.generate_initial_guess(lens_type=lens_type, z_l=0.194)
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
        _ = lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        print_step_info(flags, "After Final Minimization:", lenses, reducedchi2)

    # No matter what signal combination was used, return a chi2 using all signals
    # This is because the final fit should be the best possible fit

    reducedchi2 = lenses.update_chi2_values(sources, [True, True, True])
    
    return lenses, reducedchi2