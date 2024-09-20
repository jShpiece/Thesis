"""
Module for gravitational lensing pipeline functions.

This module provides functions to generate initial guesses for lens positions,
optimize lens positions and strengths, filter and merge lens positions,
and other utilities used in gravitational lensing analysis.

Functions:
    - generate_initial_guess
    - optimize_lens_positions
    - filter_lens_positions
    - merge_close_lenses
    - iterative_elimination
    - optimize_lens_strength
    - update_chi2_values
    - chi2wrapper
"""

import numpy as np
import scipy.optimize as opt
import utils  # Custom utility functions
import minimizer  # Custom minimizer module
import source_obj
import halo_obj
import metric


def generate_initial_guess(sources, lens_type='SIS', z_l=0.5, z_s=0.8):
    """
    Generates initial guesses for lens positions based on source ellipticity and flexion.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lens_type (str): Type of lens model ('SIS' or 'NFW'). Default is 'SIS'.
        z_l (float): Redshift of the lens. Default is 0.5.
        z_s (float): Redshift of the source. Default is 0.8.

    Returns:
        SIS_Lens or NFW_Lens: Initial lens object with estimated positions and parameters.
    """
    # Calculate angle phi and magnitude of shear and flexion
    phi = np.arctan2(sources.f2, sources.f1)
    gamma = np.hypot(sources.e1, sources.e2)
    flexion = np.hypot(sources.f1, sources.f2)

    if lens_type == 'SIS':
        # Characteristic distance from the source
        r = gamma / flexion
        # Einstein radius of the lens
        te = 2 * gamma * r
        # Estimate lens positions
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        return halo_obj.SIS_Lens(xl, yl, te, np.empty_like(sources.x))

    elif lens_type == 'NFW':
        r = 1.45 * gamma / flexion

        def estimate_nfw_mass_from_flexion(mass, params):
            """
            Estimates NFW mass from flexion by minimizing the difference between observed and model flexion.

            Parameters:
                mass (float): Mass of the NFW halo.
                params (tuple): Parameters needed for calculation.

            Returns:
                float: Difference between calculated and observed flexion.
            """
            xl, yl, xs, ys, f1_obs, f2_obs = params
            # Create a halo with the given mass
            lenses = halo_obj.NFW_Lens(
                xl, yl, np.zeros_like(xl), np.array([5]), np.array([mass]), np.array([z_l]), np.empty_like(xl)
            )
            lenses.calculate_concentration()
            # Create a source object with zero shear and observed flexion
            source = source_obj.Source(
                xs, ys, np.zeros_like(xs), np.zeros_like(ys),
                f1_obs, f2_obs, np.zeros_like(f1_obs), np.zeros_like(f2_obs),
                np.zeros_like(f1_obs), np.zeros_like(f1_obs), np.zeros_like(f1_obs)
            )
            # Calculate flexion signals from the model
            _, _, flex_1, flex_2, _, _ = utils.calculate_lensing_signals_nfw(lenses, source, z_s)
            # Return the difference between model and observed flexion
            return np.sqrt((flex_1 - f1_obs)**2 + (flex_2 - f2_obs)**2)

        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)

        # Estimate the mass of the NFW halo for each source
        masses = []
        for i in range(len(sources.x)):
            mass_guess = 1e13  # Initial mass guess
            params = (xl[i], yl[i], sources.x[i], sources.y[i], sources.f1[i], sources.f2[i])
            result = opt.minimize(
                estimate_nfw_mass_from_flexion,
                mass_guess,
                args=(params,),
                method='Nelder-Mead',
                tol=1e-6,
                options={'maxiter': 1000}
            )
            masses.append(result.x[0] * 2)  # Adjust mass as per model requirements

        lenses = halo_obj.NFW_Lens(
            xl, yl, np.zeros_like(xl), np.array([5]), np.array(masses), np.array([z_l]), np.empty_like(xl)
        )
        lenses.calculate_concentration()
        return lenses

    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')


def optimize_lens_positions(sources, lenses, use_flags, lens_type='SIS'):
    """
    Optimizes lens positions via local minimization.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Initial lens object with estimated positions and parameters.
        use_flags (list): Flags indicating which data to use in optimization.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Lenses with optimized positions.
    """
    # Optimizer parameters
    learning_rates = [1e-2, 1e-2, 1e-4]  # Learning rates for position and strength
    num_iterations = 1000
    beta1 = 0.9
    beta2 = 0.999

    if lens_type == 'SIS':
        for i in range(len(lenses.x)):
            # Create a single-source object
            one_source = source_obj.Source(
                sources.x[i], sources.y[i],
                sources.e1[i], sources.e2[i],
                sources.f1[i], sources.f2[i],
                sources.g1[i], sources.g2[i],
                sources.sigs[i], sources.sigf[i], sources.sigg[i]
            )
            # Initial guess for the optimizer
            guess = [lenses.x[i], lenses.y[i], lenses.te[i]]
            params = ['SIS', 'unconstrained', one_source, use_flags]
            # Optimize using custom Adam optimizer
            result, _ = minimizer.adam_optimizer(
                chi2wrapper, guess, learning_rates, num_iterations, beta1, beta2, params=params
            )
            # Update lens parameters
            lenses.x[i], lenses.y[i], lenses.te[i] = result[0], result[1], result[2]

    elif lens_type == 'NFW':
        for i in range(len(lenses.x)):
            # Initial guess: x, y, log10(mass)
            guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])]
            params = ['NFW', 'unconstrained', sources, use_flags, lenses.concentration[i], lenses.redshift]
            # Optimize using custom Adam optimizer
            result, _ = minimizer.adam_optimizer(
                chi2wrapper, guess, learning_rates, num_iterations, beta1, beta2, params=params
            )
            # Update lens parameters
            lenses.x[i], lenses.y[i], lenses.mass[i] = result[0], result[1], 10 ** result[2]
            lenses.calculate_concentration()  # Update concentration
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def filter_lens_positions(sources, lenses, xmax, threshold_distance=0.5, lens_type='SIS'):
    """
    Filters out invalid lenses based on distance criteria.

    Parameters:
        sources (Source): Source object containing source positions.
        lenses (SIS_Lens or NFW_Lens): Lens object containing lens positions and parameters.
        xmax (float): Maximum allowed distance from the origin.
        threshold_distance (float): Minimum allowed distance between any lens and source. Default is 0.5.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Filtered lenses.
    """
    # Calculate distances between each lens and each source
    distances = np.sqrt((lenses.x[:, None] - sources.x) ** 2 + (lenses.y[:, None] - sources.y) ** 2)
    
    # Identify lenses that are too close to any source
    too_close = np.any(distances < threshold_distance, axis=1)
    # Identify lenses that are too far from the center
    too_far = np.sqrt(lenses.x ** 2 + lenses.y ** 2) > xmax

    if lens_type == 'SIS':
        # SIS-specific condition: Einstein radius too small
        invalid_te = np.abs(lenses.te) < 1e-3
        valid_indices = ~(too_close | too_far | invalid_te)
        # Filter lenses
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.te = lenses.te[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

    elif lens_type == 'NFW':
        # NFW-specific conditions: invalid mass values
        invalid_mass = (lenses.mass < 1e10) | (lenses.mass > 1e16)
        valid_indices = ~(too_close | too_far | invalid_mass)
        # Filter lenses
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.mass = lenses.mass[valid_indices]
        lenses.concentration = lenses.concentration[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def merge_close_lenses(lenses, merger_threshold=5, lens_type='SIS'):
    """
    Merges lenses that are closer than a specified threshold.

    Parameters:
        lenses (SIS_Lens or NFW_Lens): Lens object containing lens positions and parameters.
        merger_threshold (float): Distance threshold for merging lenses. Default is 5.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Lenses after merging close lenses.
    """
    # Determine lens strength based on type
    strength = np.abs(lenses.te if lens_type == 'SIS' else lenses.mass)

    def merge_lenses(i, j):
        """
        Merges lens at index j into lens at index i, updating positions and strengths.

        Parameters:
            i (int): Index of the lens to keep.
            j (int): Index of the lens to merge and remove.
        """
        weight_i, weight_j = strength[i], strength[j]
        total_weight = weight_i + weight_j
        # Update position of lens i
        lenses.x[i] = (lenses.x[i] * weight_i + lenses.x[j] * weight_j) / total_weight
        lenses.y[i] = (lenses.y[i] * weight_i + lenses.y[j] * weight_j) / total_weight
        # Update strength of lens i
        strength[i] = total_weight / 2  # Adjust strength as needed
        # Remove lens j
        delete_indices(j)

    def delete_indices(index):
        """
        Deletes lens at the specified index from all lens properties.

        Parameters:
            index (int): Index of the lens to remove.
        """
        nonlocal strength
        lenses.x = np.delete(lenses.x, index)
        lenses.y = np.delete(lenses.y, index)
        lenses.chi2 = np.delete(lenses.chi2, index)
        strength = np.delete(strength, index)
        if lens_type == 'NFW':
            lenses.concentration = np.delete(lenses.concentration, index)

    i = 0
    while i < len(lenses.x):
        j = i + 1
        while j < len(lenses.x):
            distance = np.hypot(lenses.x[i] - lenses.x[j], lenses.y[i] - lenses.y[j])
            if distance < merger_threshold:
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
    """
    Iteratively eliminates lenses to improve the fit by minimizing the reduced chi-squared value.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Lens object containing lens positions and parameters.
        reducedchi2 (float): Current reduced chi-squared value.
        use_flags (list): Flags indicating which data to use in calculations.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Optimized lenses after iterative elimination.
    """

    def select_lowest_chi2(x, y, te_or_mass, chi2, lens_floor=1):
        """
        Selects the lenses with the lowest chi-squared values.

        Parameters:
            x (np.ndarray): x-positions of lenses.
            y (np.ndarray): y-positions of lenses.
            te_or_mass (np.ndarray): Strength parameters (Einstein radius or mass).
            chi2 (np.ndarray): Chi-squared values for each lens.
            lens_floor (int): Minimum number of lenses to keep.

        Returns:
            Tuple[np.ndarray]: Selected x, y, te_or_mass, and chi2 arrays.
        """
        sorted_indices = np.argsort(chi2)
        selected_indices = sorted_indices[:max(lens_floor, len(x))]
        return x[selected_indices], y[selected_indices], te_or_mass[selected_indices], chi2[selected_indices]

    def find_best_combination(x, y, mass, concentration, chi2, max_combinations=5):
        """
        Finds the best combination of lenses that minimizes the chi-squared value.

        Parameters:
            x (np.ndarray): x-positions of lenses.
            y (np.ndarray): y-positions of lenses.
            mass (np.ndarray): Masses of lenses.
            concentration (np.ndarray): Concentration parameters of lenses.
            chi2 (np.ndarray): Chi-squared values for each lens.
            max_combinations (int): Maximum number of combinations to consider.

        Returns:
            Tuple[np.ndarray]: Best x, y, mass, concentration, and chi2 arrays.
        """
        best_chi2 = np.inf
        best_params = (x, y, mass, concentration)

        for num in range(2, max_combinations):
            indices_list = utils.find_combinations(range(len(x)), num)
            filtered_combinations = utils.filter_combinations(indices_list)
            for indices in filtered_combinations:
                indices = list(indices)
                selected_params = (
                    x[indices], y[indices], mass[indices], concentration[indices], chi2[indices]
                )
                test_lenses = halo_obj.NFW_Lens(
                    selected_params[0], selected_params[1], np.zeros_like(selected_params[0]),
                    selected_params[3], selected_params[2], lenses.redshift, selected_params[4]
                )
                current_chi2 = test_lenses.update_chi2_values(sources, use_flags)
                if current_chi2 < best_chi2:
                    best_chi2 = current_chi2
                    best_params = selected_params[:-1]  # Exclude chi2
        return *best_params, best_chi2

    def eliminate_lenses(lenses, lens_floor_range, lens_update_fn, calculate_fn):
        """
        Eliminates lenses to minimize the distance between reduced chi-squared and 1.

        Parameters:
            lenses (SIS_Lens or NFW_Lens): Current lens object.
            lens_floor_range (range): Range of minimum number of lenses to consider.
            lens_update_fn (Callable): Function to create new lens objects.
            calculate_fn (Callable): Function to calculate additional lens parameters if needed.
        """
        best_dist = abs(reducedchi2 - 1)
        best_lenses = lenses

        for floor in lens_floor_range:
            # Select lenses with lowest chi-squared values
            x, y, te_or_mass, chi2 = select_lowest_chi2(
                lenses.x, lenses.y,
                lenses.te if lens_type == 'SIS' else lenses.mass,
                lenses.chi2,
                lens_floor=floor
            )

            # For NFW lenses with fewer than 20 lenses, find best combinations
            if lens_type == 'NFW' and len(x) < 20:
                x, y, te_or_mass, concentration, chi2 = find_best_combination(
                    x, y, te_or_mass, lenses.concentration, chi2
                )

            # Create new lenses
            test_lenses = lens_update_fn(x, y, te_or_mass, chi2)
            current_chi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = abs(current_chi2 - 1)

            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses

        # Update lenses with the best found lenses
        lenses.x = best_lenses.x
        lenses.y = best_lenses.y
        lenses.chi2 = best_lenses.chi2
        if lens_type == 'SIS':
            lenses.te = best_lenses.te
        elif lens_type == 'NFW':
            lenses.mass = best_lenses.mass
            lenses.concentration = best_lenses.concentration

        calculate_fn()  # Calculate additional lens parameters if needed

    if lens_type == 'SIS':
        eliminate_lenses(
            lenses,
            range(1, len(lenses.x) + 1),
            lambda x, y, te, chi2: halo_obj.SIS_Lens(x, y, te, chi2),
            lambda: None  # No additional calculation needed
        )

    elif lens_type == 'NFW':
        eliminate_lenses(
            lenses,
            range(1, min(len(lenses.x) + 1, 1000)),
            lambda x, y, mass, chi2: halo_obj.NFW_Lens(
                x, y, np.zeros_like(x), lenses.concentration, mass, lenses.redshift, chi2
            ),
            lenses.calculate_concentration
        )

    return lenses


def optimize_lens_strength(sources, lenses, use_flags, lens_type='SIS'):
    """
    Optimizes the strength parameters (Einstein radius or mass) of the lenses.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Lens object containing lens positions and parameters.
        use_flags (list): Flags indicating which data to use in optimization.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Lenses with optimized strengths.
    """
    if lens_type == 'SIS':
        guess = lenses.te
        params = ['SIS', 'constrained', lenses.x, lenses.y, sources, use_flags]
        max_attempts = 5
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
        # Optimize mass for each lens individually
        num_iterations = 1000
        for i in range(len(lenses.x)):
            guess = [np.log10(lenses.mass[i])]
            learning_rates = [1e-2]
            params = [
                'NFW', 'constrained',
                lenses.x[i], lenses.y[i], lenses.redshift,
                lenses.concentration[i], sources, use_flags
            ]
            result, _ = minimizer.gradient_descent(
                chi2wrapper, guess, learning_rates=learning_rates,
                num_iterations=num_iterations, params=params
            )
            lenses.mass[i] = 10 ** result[0]
            lenses.calculate_concentration()
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def update_chi2_values(sources, lenses, use_flags):
    """
    Updates the chi-squared values for each lens based on the current source data.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (NFW_Lens): Lens object containing lens positions and parameters.
        use_flags (list): Flags indicating which data to use in calculations.

    Returns:
        float: Reduced chi-squared value.
    """
    # Calculate global chi-squared and degrees of freedom
    global_chi2 = metric.calculate_chi_squared(sources, lenses, use_flags, lensing='NFW')
    dof = metric.calc_degrees_of_freedom(sources, lenses, use_flags)
    reduced_chi2 = global_chi2 / dof if dof != 0 else np.inf

    # Initialize chi-squared values for individual lenses
    chi2_values = np.zeros(len(lenses.x))

    if len(lenses.x) == 1:
        # Only one lens
        chi2_values[0] = global_chi2
    else:
        for i in range(len(lenses.x)):
            # Calculate chi-squared for each lens individually
            one_halo = halo_obj.NFW_Lens(
                lenses.x[i], lenses.y[i], lenses.z[i],
                lenses.concentration[i], lenses.mass[i],
                lenses.redshift, [0]
            )
            chi2_values[i] = metric.calculate_chi_squared(
                sources, one_halo, use_flags, lensing='NFW', use_weights=False
            )

    lenses.chi2 = chi2_values

    if dof == 0:
        print('Degrees of freedom is zero')
        return global_chi2

    return reduced_chi2


def chi2wrapper(guess, params):
    """
    Chi-squared wrapper function for optimization.

    Parameters:
        guess (list): List of guessed parameters.
        params (list): List of parameters required for lensing models.

    Returns:
        float: Chi-squared value to be minimized.
    """
    # Ensure params is a list
    if isinstance(params, tuple):
        params = list(params)

    model_type, constraint_type = params[0], params[1]
    params = params[2:]

    if model_type == 'SIS':
        if constraint_type == 'unconstrained':
            # Create lens with guessed parameters
            lenses = halo_obj.SIS_Lens(guess[0], guess[1], guess[2], [0])
            return metric.calculate_chi_squared(params[0], lenses, params[1], use_weights=True)
        elif constraint_type == 'constrained':
            # Optimize strength parameter(s)
            lenses = halo_obj.SIS_Lens(params[0], params[1], guess, np.empty_like(params[0]))
            dof = metric.calc_degrees_of_freedom(params[2], lenses, params[3])
            chi2 = metric.calculate_chi_squared(params[2], lenses, params[3])
            return np.abs(chi2 / dof - 1)

    elif model_type == 'NFW':
        if constraint_type == 'unconstrained':
            lenses = halo_obj.NFW_Lens(
                guess[0], guess[1], np.zeros_like(guess[0]),
                params[2], 10 ** guess[2], params[3], [0]
            )
            lenses.calculate_concentration()
            return metric.calculate_chi_squared(params[0], lenses, params[1], lens_type='NFW', use_weights=False)
        elif constraint_type == 'constrained':
            lenses = halo_obj.NFW_Lens(
                params[0], params[1], np.zeros_like(params[0]),
                params[3], 10 ** guess, params[2], np.empty_like(params[0])
            )
            lenses.calculate_concentration()
            return metric.calculate_chi_squared(params[4], lenses, params[5], lens_type='NFW', use_weights=False)
        elif constraint_type == 'dual_constraint':
            # Optimize both mass and concentration
            lenses = halo_obj.NFW_Lens(
                params[0], params[1], np.zeros_like(params[0]),
                guess[1], 10 ** guess[0], params[2], np.empty_like(params[0])
            )
            return metric.calculate_chi_squared(params[3], lenses, params[4], lens_type='NFW', use_weights=False)
    else:
        raise ValueError(f"Invalid lensing model: {model_type}")