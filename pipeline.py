import numpy as np
import scipy.optimize as opt
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
import utils
import minimizer
M_solar = 1.989 * 10**30 # kg

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
            te = 2 * gamma * r / 206265 # The Einstein radius of the lens in radians
            from astropy.constants import c, G
            _, Ds, Dls = utils.angular_diameter_distances(z_l, z_s)
            rho_c = cosmo.critical_density(z_l).to(u.kg / u.m**3)
            mass = (te * (c**2 / (2 * np.pi * G)) * (Ds / Dls) * (3 / (800 * np.pi * rho_c))**(1/3))**(3/2)
            mass = mass.to(u.Msun).value

            xl = np.array(self.x + r * np.cos(phi))
            yl = np.array(self.y + r * np.sin(phi))
            halo = Halo(xl, yl, np.zeros_like(xl), np.array([5]), np.array(mass), np.array(z_l), np.empty_like(xl))
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

    
    def apply_NFW_lensing(self, halos, z_source=0.8):
        e1, e2, f1, f2, g1, g2 = utils.calculate_lensing_signals_nfw(halos, self, z_source)

        self.e1 += e1
        self.e2 += e2
        self.f1 += f1
        self.f2 += f2
        self.g1 += g1
        self.g2 += g2


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
                    chi2wrapper, guess, args=([one_source, use_flags]), 
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
        # Given a set of sources, update the chi^2 values for each lens
        global_chi2 = calculate_chi_squared(sources, self, use_flags, lensing='SIS')
        chi2_values = np.zeros(len(self.x))
        if len(self.x) == 1:
            # Only one lens - calculate the chi2 value for this lens
            chi2_values[0] = assign_lens_chi2_values(self, sources, use_flags)
        else:
            for i in range(len(self.x)):
                # Only pass in the i-th lens
                one_lens = Lens(self.x[i], self.y[i], self.te[i], np.empty_like(self.x))
                chi2_values[i] = assign_lens_chi2_values(one_lens, sources, use_flags)
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
        R200 = (3 / (800 * np.pi) * (np.abs(self.mass) * M_solar / rho_c))**(1/3) # In meters
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        return (200/3) * (self.concentration**3) / (np.log(1 + self.concentration) - self.concentration / (1 + self.concentration))


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
        # Note - numpy doesn't like negative powers on lists, even if the answer isn't complex
        # Get around this by taking taking the absolute value of arrays (then multiplying by -1 if necessary) (actually don't do this - mass and concentration should be positive)
        # It also breaks down if the mass is 0 (we'll be dividing by the mass)
        self.mass += 1e-10 # Add a small value to the mass to avoid division by zero
        self.concentration = 5.71 * (np.abs(self.mass) / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47) 

    # --------------------------------------------
    # Pipeline Functions
    # --------------------------------------------

    def update_chi2_values(self, sources, use_flags):
        # Given a set of sources, update the chi^2 values for each lens

        global_chi2 = calculate_chi_squared(sources, self, use_flags, lensing='NFW')
        chi2_values = np.zeros(len(self.x))
        if len(self.x) == 1:
            # Only one lens - calculate the chi2 value for this lens
            chi2_values[0] = assign_halo_chi2_values(self, sources, use_flags)
        else:
            for i in range(len(self.x)):
                # Only pass in the i-th lens
                one_halo = Halo(self.x[i], self.y[i], self.z[i], self.concentration[i], self.mass[i], self.redshift, [0])
                chi2_values[i] = assign_halo_chi2_values(one_halo, sources, use_flags)
        self.chi2 = chi2_values
        dof = calc_degrees_of_freedom(sources, self, use_flags)
        if dof == 0:
            print('Degrees of freedom is zero')
            return global_chi2
        return global_chi2 / dof


    def optimize_lens_positions(self, sources, use_flags):
        # Given a set of initial guesses for lens positions, find the optimal lens positions
        # via local minimization

        learning_rates = [0.1, 0.1, 0.1]  # Adjust learning rate for mass parameter
        num_iterations = 1000
        beta1 = 0.9
        beta2 = 0.999

        for i in range(len(self.x)):
            one_source = Source(sources.x[i], sources.y[i], 
                                sources.e1[i], sources.e2[i], 
                                sources.f1[i], sources.f2[i], 
                                sources.g1[i], sources.g2[i],
                                sources.sigs[i], sources.sigf[i], sources.sigg[i])
            guess = [self.x[i], self.y[i], np.log10(self.mass[i])] # Class is already initialized with initial guesses
            params = [sources, use_flags, self.concentration[i], self.redshift]
            # result, trail = minimizer.gradient_descent(chi2wrapper3, guess, learning_rate, num_iterations, momentum, params)
            result, trail = minimizer.adam_optimizer(chi2wrapper3, guess, learning_rates, num_iterations, beta1, beta2, params=params)

            self.x[i], self.y[i], self.mass[i] = result[0], result[1], 10**result[2] # Optimize the mass in log space, then convert back to linear space
            # Now update the concentrations
            self.calculate_concentration()
        return trail


    def filter_lens_positions(self, sources, xmax, threshold_distance=0.5):
        # Filter out halos that are too close to sources or too far from the center

        # Compute the distance of each lens from each source
        distances_to_sources = np.sqrt((self.x[:, None] - sources.x)**2 + (self.y[:, None] - sources.y)**2)
        # Identify lenses that are too close to sources
        too_close_to_sources = np.any(distances_to_sources < threshold_distance, axis=1)
        # Identify lenses that are too far from the center
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 1.5 * xmax
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


    def select_lowest_chi2(self, lens_floor=1):
        # Function that enables the iterative elimination of lenses
        # Select the 'lens_floor' lenses with the lowest chi^2 values
        
        # Sort the lenses by chi^2 value
        sorted_indices = np.argsort(self.chi2)
        self.x, self.y, self.mass, self.concentration, self.chi2 = self.x[sorted_indices], self.y[sorted_indices], self.mass[sorted_indices], self.concentration[sorted_indices], self.chi2[sorted_indices]

        # Select the 'lens_floor' lenses with the lowest chi^2 values
        if len(self.x) > lens_floor:
            self.x, self.y, self.z, self.mass, self.concentration, self.chi2 = self.x[:lens_floor], self.y[:lens_floor], self.z[:lens_floor], self.mass[:lens_floor], self.concentration[:lens_floor], self.chi2[:lens_floor]


    def iterative_elimination(self, sources, reducedchi2, use_flags):
        # Iteratively eliminate lenses that do not improve the chi^2 value
        # lens_floors = np.arange(1, len(self.x) + 1)
        max_lenses = np.min([len(self.x) + 1, 100]) # Maximum number of lenses to consider
        lens_floors = np.arange(1, max_lenses)
        best_dist = np.abs(reducedchi2 - 1)
        best_lenses = self
        for lens_floor in lens_floors:
            # Clone the lenses object
            test_lenses = Halo(self.x, self.y, self.z, self.concentration, self.mass, self.redshift, self.chi2)
            test_lenses.select_lowest_chi2(lens_floor=lens_floor)

            reducedchi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = np.abs(reducedchi2 - 1)
            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses
        # Update the lenses object with the best set of lenses
        self.x, self.y, self.mass, self.chi2 = best_lenses.x, best_lenses.y, best_lenses.mass, best_lenses.chi2
        self.calculate_concentration()


    def full_minimization(self, sources, use_flags):
        guess = self.mass
        params = [self.x, self.y, self.redshift, self.concentration, sources, use_flags]
        max_attempts = 5  # Number of optimization attempts with different initial guesses
        best_result = None
        best_params = guess

        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper4, guess, args=params,
                method='Powell',  
                tol=1e-8,  
                options={'maxiter': 1000}
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        self.mass = best_params


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


def calculate_chi_squared(sources, lenses, flags, lensing='SIS') -> float:
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
    if lensing == 'SIS':
        source_clone.apply_SIS_lensing(lenses)
    elif lensing == 'NFW':
        source_clone.apply_NFW_lensing(lenses)

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
    if lensing == 'SIS':
        penalty = sum(eR_penalty_function(eR) for eR in lenses.te)
    elif lensing == 'NFW':
        penalty = 0

    # Return the total chi-squared including penalties
    return total_chi_squared + penalty


def assign_lens_chi2_values(lenses, sources, use_flags):
    # Given a single lens object, assign a weighted chi2 value based on source distances
    xl = lenses.x
    yl = lenses.y
    assert len(xl) == len(yl), "The x and y arrays must have the same length."
    assert xl.ndim == yl.ndim == 1, "The x and y arrays must be 1D."
    assert len(xl) == 1, "Only one lens is allowed."

    xs = sources.x
    ys = sources.y

    r = np.sqrt((xl[:, None] - xs)**2 + (yl[:, None] - ys)**2 + 0.01**2) # Add a small value to avoid division by zero
    # Choose a characteristic distance such that 1/5 of the sources are within this distance
    # Calculate distances from the current lens to all sources
    distances = np.sqrt((xl - xs)**2 + (yl - ys)**2)
    
    # Sort distances
    sorted_distances = np.sort(distances)
    
    # Determine the index for the desired fraction
    index = int(len(sorted_distances) * (0.1))
    
    # Set r0 as the distance at the calculated index
    r0 = sorted_distances[index]

    weights = np.exp(-r**2 / r0**2)
    # Normalize the weights
    weights /= np.sum(weights, axis=1)[:, None]
    assert np.allclose(np.sum(weights, axis=1), 1), "Weights must sum to 1 - they sum to {}".format(np.sum(weights, axis=1))
    assert weights.shape == (len(xl), len(xs)), "Weights must have shape (len(xl), len(xs))."

    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = use_flags

    # Initialize a clone of sources with zeroed lensing signals
    source_clone = Source(
        x=sources.x, y=sources.y,
        e1=np.zeros_like(sources.e1), e2=np.zeros_like(sources.e2),
        f1=np.zeros_like(sources.f1), f2=np.zeros_like(sources.f2),
        g1=np.zeros_like(sources.g1), g2=np.zeros_like(sources.g2),
        sigs=sources.sigs, sigf=sources.sigf, sigg=sources.sigg
    )

    source_clone.apply_SIS_lensing(lenses)

    # Weigh the chi^2 values by the distance of each source from each lens
    total_chi_squared = 0
    for i in range(len(source_clone.x)):
        # Get the chi2 values for each source, weighted by the source-lens distance
        shear_chi2 = ((sources.e1[i] - source_clone.e1[i]) / sources.sigs[i])**2 + ((sources.e2[i] - source_clone.e2[i]) / sources.sigs[i])**2
        flexion_chi2 = ((sources.f1[i] - source_clone.f1[i]) / sources.sigf[i])**2 + ((sources.f2[i] - source_clone.f2[i]) / sources.sigf[i])**2
        g_flexion_chi2 = ((sources.g1[i] - source_clone.g1[i]) / sources.sigg[i])**2 + ((sources.g2[i] - source_clone.g2[i]) / sources.sigg[i])**2
        total_chi_squared += weights[:, i].dot(shear_chi2*use_shear + flexion_chi2*use_flexion + g_flexion_chi2*use_g_flexion)

    return total_chi_squared[0]


def assign_halo_chi2_values(lenses, sources, use_flags):
    # Given a single lens object, assign a weighted chi2 value based on source distances

    weights = utils.compute_source_weights(lenses, sources)    
    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = use_flags

    # Initialize a clone of sources with zeroed lensing signals
    source_clone = Source(
        x=sources.x, y=sources.y,
        e1=np.zeros_like(sources.e1), e2=np.zeros_like(sources.e2),
        f1=np.zeros_like(sources.f1), f2=np.zeros_like(sources.f2),
        g1=np.zeros_like(sources.g1), g2=np.zeros_like(sources.g2),
        sigs=sources.sigs, sigf=sources.sigf, sigg=sources.sigg
    )

    source_clone.apply_NFW_lensing(lenses)

    # Weigh the chi^2 values by the distance of each source from each lens
    total_chi_squared = 0
    for i in range(len(source_clone.x)):
        # Calculate chi2 for each component
        shear_chi2 = ((sources.e1[i] - source_clone.e1[i]) / sources.sigs[i])**2 + ((sources.e2[i] - source_clone.e2[i]) / sources.sigs[i])**2
        flexion_chi2 = ((sources.f1[i] - source_clone.f1[i]) / sources.sigf[i])**2 + ((sources.f2[i] - source_clone.f2[i]) / sources.sigf[i])**2
        g_flexion_chi2 = ((sources.g1[i] - source_clone.g1[i]) / sources.sigg[i])**2 + ((sources.g2[i] - source_clone.g2[i]) / sources.sigg[i])**2
        
        # Ensure correct shape for weights
        chi2_sum = shear_chi2 * use_shear + flexion_chi2 * use_flexion + g_flexion_chi2 * use_g_flexion
        total_chi_squared += weights[:, i].dot(chi2_sum)

    return total_chi_squared[0]

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


def chi2wrapper3(guess, params):
    # Wrapper function for chi2 to allow for minimization for a single lens object
    # Guess = [x, y, mass]
    # Params = [source, use_flags, concentration, redshift]
    lenses = Halo(guess[0], guess[1], np.zeros_like(guess[0]), params[2], 10**guess[2], params[3], [0])
    #lenses = Halo(guess[0], guess[1], np.zeros_like(guess[0]), params[2], params[4], params[3], [0]) # Try a run with mass as a parameter instead of a guess
    lenses.calculate_concentration() # The concentration needs to be updated, because the mass will change during optimization
    return assign_halo_chi2_values(lenses, params[0], params[1])


def chi2wrapper4(guess, params):
    # Wrapper function for chi2 to allow constrained minimization - 
    #    only the masses are allowed to vary
    # This time, the lens object contains the full set of lenses
    # Guess = [mass1, mass2, ...]
    # Params = [x, y, redshift, concentration, sources, use_flags]
    lenses = Halo(params[0], params[1], np.zeros_like(params[0]), params[3], guess, params[2], [0])
    dof = calc_degrees_of_freedom(params[4], lenses, params[5])
    # Return the reduced chi^2 value - 1, to be minimized
    return np.abs(calculate_chi_squared(params[4], lenses, params[5], lensing='NFW') / dof - 1)

# ------------------------------
# Main function
# ------------------------------

def fit_lensing_field(sources, xmax, flags = False, use_flags = [True, True, True], lens_type='SIS'):
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
    lenses = sources.generate_initial_guess(lens_type)
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