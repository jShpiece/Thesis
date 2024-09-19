import numpy as np
import scipy.optimize as opt
import utils # Homemade utility functions
import minimizer # Homemade minimizer module
import source_obj
import halo_obj
import metric

# ------------------------------
# Pipeline Functions
# ------------------------------

def generate_initial_guess(sources, lens_type='SIS', z_l = 0.5, z_s = 0.8):
    # Generate initial guesses for possible lens positions based on the source ellipticity and flexion
    phi = np.arctan2(sources.f2, sources.f1)
    gamma = np.sqrt(sources.e1**2 + sources.e2**2)
    flexion = np.sqrt(sources.f1**2 + sources.f2**2)

    if lens_type == 'SIS':
        r = gamma / flexion # A characteristic distance from the source
        te = 2 * gamma * r # The Einstein radius of the lens
        return halo_obj.SIS_Lens(np.array(sources.x + r * np.cos(phi)), np.array(sources.y + r * np.sin(phi)), np.array(te), np.empty_like(sources.x))
    
    elif lens_type == 'NFW':
        r = 1.45 * gamma / flexion # A characteristic distance from the source

        def estimate_nfw_mass_from_flexion(mass, params):
            # unpack params
            xl, yl, xs, ys, f1, f2 = params
            # Create a halo
            lenses = halo_obj.NFW_Lens(xl, yl, np.zeros_like(xl), np.array([5]), np.array(mass), np.array([z_l]), np.empty_like(xl))
            lenses.calculate_concentration()
            # create a source
            source = sources.Source(xs, ys, np.zeros_like(xs), np.zeros_like(ys), f1, f2, np.zeros_like(f1), np.zeros_like(f2), np.zeros_like(f1), np.zeros_like(f1), np.zeros_like(f1))
            # get the flexion signals
            shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(halo_obj, source, z_s)
            # Use this as a function to minimize - find the mass that minimizes the difference between the flexion signals
            return np.sqrt((flex_1 - f1)**2 + (flex_2 - f2)**2)
        
        xl = np.array(sources.x + r * np.cos(phi))
        yl = np.array(sources.y + r * np.sin(phi))

        # Estimate the mass of the NFW halo
        masses = []
        for i in range(len(sources.x)):
            mass_guess = 10**13 # Initial guess for the mass
            limits = [(10**10, 10**16)] # Mass limits
            result = opt.minimize(estimate_nfw_mass_from_flexion, mass_guess, args=([xl[i], yl[i], sources.x[i], sources.y[i], sources.f1[i], sources.f2[i]]), method='Nelder-Mead', tol=1e-6, options={'maxiter': 1000}, bounds=limits)
            masses.append(result.x[0] * 2)
        
        lenses = halo_obj.NFW_Lens(xl, yl, np.zeros_like(xl), np.array([5]), np.array(masses), np.array(z_l), np.empty_like(xl))
        halo_obj.calculate_concentration()
        return lenses


def optimize_lens_positions(sources, lenses, use_flags, lens_type='SIS'):
    # Given a set of initial guesses for lens positions, find the optimal lens positions via local minimization
    learning_rates = [1e-2, 1e-2, 1e-4] # Learning rates for the optimizer
    num_iterations = 10**3 # Number of iterations for the optimizer
    beta1 = 0.9 # Beta1 parameter for the optimizer
    beta2 = 0.999 # Beta2 parameter for the optimizer

    if lens_type == 'SIS':       
        for i in range(len(lenses.x)):
            one_source = source_obj.Source(sources.x[i], sources.y[i], 
                                sources.e1[i], sources.e2[i], 
                                sources.f1[i], sources.f2[i], 
                                sources.g1[i], sources.g2[i],
                                sources.sigs[i], sources.sigf[i], sources.sigg[i])
            guess = [lenses.x[i], lenses.y[i], lenses.te[i]] # Class is already initialized with initial guesses
            params = ['SIS','unconstrained', one_source, use_flags]
            result, _ = minimizer.adam_optimizer(chi2wrapper, guess, learning_rates, num_iterations, beta1, beta2, params=params)

            lenses.x[i], lenses.y[i], lenses.te[i] = result[0], result[1], result[2]

    elif lens_type == 'NFW':
        for i in range(len(lenses.x)):
            guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])] # Class is already initialized with initial guesses
            params = ['NFW','unconstrained', sources, use_flags, lenses.concentration[i], lenses.redshift]
            result, _ = minimizer.adam_optimizer(chi2wrapper, guess, learning_rates, num_iterations, beta1, beta2, params=params)

            lenses.x[i], lenses.y[i], lenses.mass[i] = result[0], result[1], 10**result[2] # Optimize the mass in log space, then convert back to linear space
            lenses.calculate_concentration() # Remember to update the concentration parameter
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def filter_lens_positions(sources, lenses, xmax, threshold_distance=0.5, lens_type='SIS'):
    # Calculate distances between each lens and each source
    distances = np.sqrt((lenses.x[:, None] - sources.x) ** 2 + (lenses.y[:, None] - sources.y) ** 2)
    
    # Conditions to identify invalid lenses
    too_close = np.any(distances < threshold_distance, axis=1)
    too_far = np.sqrt(lenses.x ** 2 + lenses.y ** 2) > xmax

    if lens_type == 'SIS':
        # SIS-specific condition: Einstein radius too small
        invalid_te = np.abs(lenses.te) < 1e-3
        valid_indices = ~(too_close | too_far | invalid_te)
        
        # Filter lenses
        lenses.x, lenses.y, lenses.te, lenses.chi2 = [l[valid_indices] for l in (lenses.x, lenses.y, lenses.te, lenses.chi2)]
        
    elif lens_type == 'NFW':
        # NFW-specific conditions: invalid mass values
        invalid_mass = (np.abs(lenses.mass) < 1e10) | (np.abs(lenses.mass) > 1e16)
        valid_indices = ~(too_close | too_far | invalid_mass)
        
        # Filter lenses
        lenses.x, lenses.y, lenses.mass, lenses.concentration, lenses.chi2 = [l[valid_indices] for l in (lenses.x, lenses.y, lenses.mass, lenses.concentration, lenses.chi2)]
    
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def merge_close_lenses(lenses, merger_threshold=5, lens_type='SIS'):
    # Determine lens strength based on type
    strength = np.abs(lenses.te if lens_type == 'SIS' else lenses.mass)
    
    def merge_lenses(i, j):
        # Merge two lenses and update the properties of lens i
        weight_i, weight_j = strength[i], strength[j]
        total_weight = weight_i + weight_j
        
        lenses.x[i] = (lenses.x[i] * weight_i + lenses.x[j] * weight_j) / total_weight
        lenses.y[i] = (lenses.y[i] * weight_i + lenses.y[j] * weight_j) / total_weight
        strength[i] = total_weight / 2
        
        # Remove lens j from all properties
        delete_indices(j)
        
    def delete_indices(index):
        nonlocal strength
        lenses.x, lenses.y, lenses.chi2 = [np.delete(arr, index) for arr in (lenses.x, lenses.y, lenses.chi2)]
        strength = np.delete(strength, index)
        if lens_type == 'NFW':
            lenses.concentration = np.delete(lenses.concentration, index)
    
    i = 0
    while i < len(lenses.x):
        j = i + 1
        while j < len(lenses.x):
            if np.sqrt((lenses.x[i] - lenses.x[j]) ** 2 + (lenses.y[i] - lenses.y[j]) ** 2) < merger_threshold:
                merge_lenses(i, j)
            else:
                j += 1
        i += 1

    # Update lens properties based on type
    if lens_type == 'SIS':
        lenses.te = strength
    elif lens_type == 'NFW':
        lenses.mass = strength
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def iterative_elimination(sources, lenses, reducedchi2, use_flags, lens_type='SIS'):
    
    def select_lowest_chi2(x, y, te, chi2, lens_floor=1):
        sorted_indices = np.argsort(chi2)
        selected_indices = sorted_indices[:lens_floor] if len(x) > lens_floor else sorted_indices
        return x[selected_indices], y[selected_indices], te[selected_indices], chi2[selected_indices]

    def find_best_combination(x, y, mass, concentration, chi2, max_combinations=5):
        best_chi2 = np.inf
        best_params = (x, y, mass, concentration)
        
        for num in range(2, max_combinations):
            combinations = utils.filter_combinations(utils.find_combinations(range(len(x)), num))
            for indices in combinations:
                selected_params = tuple(arr[list(indices)] for arr in (x, y, mass, concentration, chi2))
                test_lenses = halo_obj.NFW_Lens(*selected_params, lenses.redshift)
                current_chi2 = test_lenses.update_chi2_values(sources, use_flags)
                
                if current_chi2 < best_chi2:
                    best_chi2 = current_chi2
                    best_params = selected_params[:-1]
                    
        return *best_params, best_chi2

    def eliminate_lenses(lenses, lens_floor_range, lens_update_fn, calculate_fn):
        best_dist = abs(reducedchi2 - 1)
        best_lenses = lenses

        for floor in lens_floor_range:
            # Select lowest chi2 lenses
            x, y, te_or_mass, chi2 = select_lowest_chi2(lenses.x, lenses.y, getattr(lenses, 'te', lenses.mass), lenses.chi2, lens_floor=floor)
            
            # For NFW, check combinations if lenses are less than 20 - otherwise there are too many combinations
            if lens_type == 'NFW' and len(x) < 20:
                x, y, te_or_mass, chi2 = find_best_combination(x, y, te_or_mass, lenses.concentration, chi2)

            # Create and update test lenses object
            test_lenses = lens_update_fn(x, y, te_or_mass, chi2)
            current_chi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = abs(current_chi2 - 1)

            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses

        # Update the original lenses object with the best set of lenses found
        lenses.x, lenses.y, getattr(lenses, 'te', 'mass'), lenses.chi2 = best_lenses.x, best_lenses.y, best_lenses.te_or_mass, best_lenses.chi2
        calculate_fn()

    if lens_type == 'SIS':
        eliminate_lenses(
            lenses,
            np.arange(1, len(lenses.x) + 1),
            lambda x, y, te, chi2: halo_obj.SIS_Lens(x, y, te, chi2),
            lenses.calculate_concentration
        )

    elif lens_type == 'NFW':
        eliminate_lenses(
            lenses,
            range(1, min(len(lenses.x) + 1, 1000)),
            lambda x, y, mass, chi2: halo_obj.NFW_Lens(x, y, np.zeros_like(x), lenses.concentration, mass, lenses.redshift, chi2),
            lenses.calculate_concentration
        )

    return lenses


def optimize_lens_strength(sources, lenses, use_flags, lens_type='SIS'):
    '''Do a minimization on the strength parameter(s) of the lens(es)'''
    if lens_type == 'SIS':
        guess = lenses.te
        params = ['SIS','constrained',lenses.x, lenses.y, sources, use_flags]
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
        lenses.te = best_params
    
    elif lens_type == 'NFW':
        # Okay - hail mary time
        # Perform a global optimization of the mass parameters
        '''
        from scipy.optimize import differential_evolution
        bounds = [(10**9, 10**16)] * len(self.mass) # This will be the bounds for the mass parameters
        params = ('NFW', 'constrained', self.x, self.y, self.redshift, self.concentration, sources, use_flags)
        result = differential_evolution(chi2wrapper, bounds, args=(params,))
        self.mass = result.x
        '''
        # Do the minimization one lens at a time - hopefully this will drive the mass of false lenses to zero
        # Try just minimizing the mass
        num_iterations = 10**3
        for i in range(len(lenses.x)):
            guess = [np.log10(lenses.mass[i])]
            learning_rates = [1e-2]
            params = ['NFW','constrained',lenses.x[i], lenses.y[i], lenses.redshift, lenses.concentration[i], sources, use_flags]
            result, _ = minimizer.gradient_descent(chi2wrapper, guess, learning_rates=learning_rates, num_iterations=num_iterations, params=params)
            lenses.mass[i] = 10**result[0]
            lenses.calculate_concentration()

    return lenses


# ------------------------------
# Helper Functions
# ------------------------------

def update_chi2_values(sources, lenses, use_flags):
    # Given a set of sources, update the chi^2 values for each lens

    global_chi2 = metric.calculate_chi_squared(sources, lenses, use_flags, lensing='NFW')
    dof = metric.calc_degrees_of_freedom(sources, lenses, use_flags)
    reducedchi2 = global_chi2 / dof
    
    chi2_values = np.zeros(len(lenses.x))
    if len(lenses.x) == 1:
        # Only one lens - calculate the chi2 value for this lens
        chi2_values[0] = metric.calculate_chi_squared(sources, lenses, use_flags, lensing='NFW', use_weights=False)
    else:
        for i in range(len(lenses.x)):
            # Only pass in the i-th lens
            one_halo = halo_obj.NFW_Lens(lenses.x[i], lenses.y[i], lenses.z[i], lenses.concentration[i], lenses.mass[i], lenses.redshift, [0])
            chi2_values[i] = metric.calculate_chi_squared(sources, one_halo, use_flags, lensing='NFW', use_weights=False)
    lenses.chi2 = chi2_values
    if dof == 0:
        print('Degrees of freedom is zero')
        return global_chi2
    return reducedchi2


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
            lenses = halo_obj.SIS_Lens(guess[0], guess[1], guess[2], [0])
            return metric.calculate_chi_squared(params[0], lenses, params[1], use_weights=True)
        elif constraint_type == 'constrained':
            lenses = halo_obj.SIS_Lens(params[0], params[1], guess, np.empty_like(params[0]))
            dof = metric.calc_degrees_of_freedom(params[2], lenses, params[3])
            return np.abs(metric.calculate_chi_squared(params[2], lenses, params[3]) / dof - 1)
    
    elif model_type == 'NFW':
        if constraint_type == 'unconstrained':
            lenses = halo_obj.NFW_Lens(guess[0], guess[1], np.zeros_like(guess[0]), params[2], 10**guess[2], params[3], [0])
            lenses.calculate_concentration()
            return metric.calculate_chi_squared(params[0], lenses, params[1], lensing='NFW', use_weights=False)
        elif constraint_type == 'constrained':
            lenses = halo_obj.NFW_Lens(params[0], params[1], np.zeros_like(params[0]), params[3], 10**guess, params[2], np.empty_like(params[0]))
            lenses.calculate_concentration() # Update the concentration based on the new mass
            return metric.calculate_chi_squared(params[4], lenses, params[5], lensing='NFW', use_weights=False)
        elif constraint_type == 'dual_constraint':
            # In this case, we are optimizing both mass and concentration
            lenses = halo_obj.NFW_Lens(params[0], params[1], np.zeros_like(params[0]), guess[1], 10**guess[0], params[2], np.empty_like(params[0]))
            return metric.calculate_chi_squared(params[3], lenses, params[4], lensing='NFW', use_weights=False)
    else:
        raise ValueError("Invalid lensing model: {}".format(model_type))