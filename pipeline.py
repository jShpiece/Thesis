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
from scipy.optimize import minimize
import utils  # Custom utility functions
import minimizer  # Custom minimizer module
import source_obj # Source object
import halo_obj # Halo object
import metric # Metric calculation functions


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
    
    if lens_type == 'NFW':

        # Calculate angle phi and magnitudes of shear and flexion
        phi = np.arctan2(sources.f2, sources.f1)
        gamma = np.hypot(sources.e1, sources.e2)
        flexion = np.hypot(sources.f1, sources.f2)

        # Avoid division by zero in flexion
        flexion = np.where(flexion == 0, 1e-10, flexion)

        # Estimate radial distance to lens
        # The factor 1.45 is empirical; adjust based on model calibration if necessary
        r = 1.45 * gamma / flexion

        # Estimate lens positions
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)

        # Prepare arrays to store estimated masses
        masses = np.zeros_like(sources.x)

        # Loop over each source to estimate the mass
        for i in range(len(sources.x)):
            # Define the objective function for mass estimation
            def mass_objective(mass):
                mass = np.abs(mass)  # Ensure mass is positive
                # Create a single-lens NFW_Lens object
                lens = halo_obj.NFW_Lens(
                    x=xl[i],
                    y=yl[i],
                    z=0.0,
                    concentration=5.0,  # Or use mass-concentration relation
                    mass=mass,
                    redshift=z_l,
                    chi2=0.0
                )
                # Calculate concentration if necessary
                lens.calculate_concentration()

                # Create a single-source object
                source = source_obj.Source(
                    x=sources.x[i],
                    y=sources.y[i],
                    e1=0.0,
                    e2=0.0,
                    f1=0.0,
                    f2=0.0,
                    g1=0.0,
                    g2=0.0,
                    sigs=1.0,
                    sigf=1.0,
                    sigg=1.0
                )
                # Compute the lensing signals from the lens
                _, _, f1_model, f2_model, _, _ = utils.calculate_lensing_signals_nfw(lens, source, z_s)

                # Compute the difference between observed and modeled flexion
                flexion_diff = np.sqrt((f1_model - sources.f1[i])**2 + (f2_model - sources.f2[i])**2)
                return flexion_diff

            # Perform scalar minimization to estimate the mass
            result = opt.minimize_scalar(
                mass_objective,
                bounds=(1e10, 1e16),  # Adjust bounds as appropriate
                method='bounded',
                options={'xatol': 1e-6}
            )
            masses[i] = result.x

        # Create the NFW_Lens object with estimated masses and positions
        lenses = halo_obj.NFW_Lens(
            x=xl,
            y=yl,
            z=np.zeros_like(xl),
            concentration=np.zeros_like(xl),  # We'll compute based on mass
            mass=masses,
            redshift=z_l,
            chi2=np.zeros_like(xl)
        )
        # Calculate concentrations
        lenses.calculate_concentration()

        return lenses

    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')


def optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type='SIS', z_source=0.8):
    """
    Optimizes lens positions via local minimization.
    Currently only minimizes relative to sources within a certain distance of the lens.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Initial lens object with estimated positions and parameters.
        use_flags (list): Flags indicating which data to use in optimization.
        lens_type (str): Type of lens model ('SIS' or 'NFW').

    Returns:
        SIS_Lens or NFW_Lens: Lenses with optimized positions.
    """
    # Optimizer parameters
    num_iterations = 1e6

    if lens_type == 'SIS':
        learning_rates = [1e-2, 1e-2, 1e-2]  # Learning rates for position and strength
        beta1 = 0.9
        beta2 = 0.999
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
            # Initial guess: [x, y, log10(mass)]
            initial_guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])]

            # Define bounds for x, y, and log10(mass)
            bounds = [
                (lenses.x[i] - xmax*2, lenses.x[i] + xmax*2),  
                (lenses.y[i] - xmax*2, lenses.y[i] + xmax*2),  # Allow the minimizer to move the lens out of the field - corresponds to a "delete" operation
                (10, 16)  # Mass bounds in log10(M_sun)
            ]

            # Minimize relative to sources only within a certain distance of the lens
            filtered_sources = sources.copy()
            distance = np.hypot(lenses.x[i] - sources.x, lenses.y[i] - sources.y)
            filtered_sources.remove(np.where(distance > 20)[0])

            # Objective function to minimize
            def objective_function(params):
                xi, yi, log_mass = params
                mass = 10 ** log_mass

                # Update lens parameters
                lens = halo_obj.NFW_Lens(
                    x=xi,
                    y=yi,
                    z=lenses.z[i],
                    concentration=lenses.concentration[i],
                    mass=mass,
                    redshift=lenses.redshift,
                    chi2=0.0
                )
                # Update concentration based on mass
                lens.calculate_concentration()

                # Compute chi-squared for this lens and all sources
                chi2 = metric.calculate_chi_squared(filtered_sources, lens, use_flags, lens_type='NFW', z_source=z_source)
                return chi2

            # Run the optimizer
            result = minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': num_iterations, 'ftol': 1e-6}
            )

            # Update lens parameters with optimized values
            optimized_params = result.x
            lenses.x[i] = optimized_params[0]
            lenses.y[i] = optimized_params[1]
            lenses.mass[i] = 10 ** optimized_params[2]
            # Update concentration after mass change
            lenses.calculate_concentration()

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
    # Check if this identifies every lens as too close
    if np.all(too_close):
        print('All lenses are too close to at least one source.')
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


def merge_close_lenses(lenses, merger_threshold=5, lens_type='SIS', z_source=0.8):
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
        lenses.remove([j])

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
    if lens_type == 'NFW':
        lenses.calculate_concentration()

    # Do a couple of safety checks
    if len(lenses.x) == 0:
        raise ValueError('All lenses have been removed during merging.')
    if lens_type == 'NFW':
        assert np.all(lenses.mass > 0), 'Negative mass detected after merging.'
        assert np.all(lenses.concentration > 0), 'Negative concentration detected after merging.'
        assert len(lenses.mass) == len(lenses.concentration), 'Mass and concentration arrays have different lengths.'
        assert len(lenses.x) == len(lenses.y) == len(lenses.mass), 'Inconsistent array lengths after merging: {len(lenses.x)}, {len(lenses.y)}, {len(lenses.mass)}'
    return lenses


def forward_lens_selection(
    sources, candidate_lenses, use_flags, lens_type='NFW', z_source=0.8,
    base_tolerance=0.001, mass_scale=1e13, exponent=-0.5
    ):
    """
    Selects the best combination of lenses by iteratively adding lenses
    to minimize the reduced chi-squared value, using an adaptive tolerance
    that depends on the mass of the candidate lens.
    
    Parameters:
        sources (Source): Object containing source positions and measured lensing signals.
        candidate_lenses (Lens): Candidate lenses (NFW_Lens or SIS_Lens object).
        use_flags (list of bool): Flags indicating which lensing signals to use.
        lens_type (str): Type of lensing model ('NFW' or 'SIS'). Default is 'NFW'.
        base_tolerance (float): Base tolerance for improvement. Default is 0.003.
        mass_scale (float): Mass scale for adaptive tolerance (e.g., 1e13 solar masses).
        exponent (float): Exponent for mass dependence. Negative values increase tolerance for lower masses.
    
    Returns:
        selected_lenses (Lens): Lens object with selected lenses that minimize the reduced chi-squared.
        best_reduced_chi2 (float): The minimized reduced chi-squared value.
    """
    
    # Initialize an empty lens object based on lens_type
    if lens_type == 'NFW':
        selected_lenses = halo_obj.NFW_Lens(
            x=np.array([]),
            y=np.array([]),
            z=np.array([]),
            concentration=np.array([]),
            mass=np.array([]),
            redshift=candidate_lenses.redshift,
            chi2=np.array([])
        )
    elif lens_type == 'SIS':
        selected_lenses = halo_obj.SIS_Lens(
            x=np.array([]),
            y=np.array([]),
            te=np.array([]),
            chi2=np.array([])
        )
    else:
        raise ValueError("Unsupported lens type. Choose 'NFW' or 'SIS'.")

    remaining_indices = np.arange(len(candidate_lenses.x))
    best_reduced_chi2 = np.inf
    improved = True

    while improved and len(remaining_indices) > 0:
        improved = False
        chi2_list = []
        lens_indices = []

        # Evaluate the effect of adding each remaining lens
        for idx in remaining_indices:
            # Create a test lens set with the candidate lens added
            if lens_type == 'NFW':
                test_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    z=np.append(selected_lenses.z, candidate_lenses.z[idx]),
                    concentration=np.append(selected_lenses.concentration, candidate_lenses.concentration[idx]),
                    mass=np.append(selected_lenses.mass, candidate_lenses.mass[idx]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx])
                )
            elif lens_type == 'SIS':
                test_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    te=np.append(selected_lenses.te, candidate_lenses.te[idx]),
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx])
                )

            # Compute chi-squared and reduced chi-squared
            chi2 = metric.calculate_chi_squared(sources, test_lenses, use_flags, lens_type=lens_type, z_source=z_source)
            dof = metric.calc_degrees_of_freedom(sources, test_lenses, use_flags)
            reduced_chi2 = chi2 / dof if dof > 0 else np.inf
            chi2_list.append(reduced_chi2)
            lens_indices.append(idx)

        # Find the lens whose addition leads to the best reduced chi-squared
        min_chi2 = min(chi2_list)
        min_index = chi2_list.index(min_chi2)
        idx_to_add = lens_indices[min_index]

        # Adaptive tolerance calculation based on the mass of the lens
        lens_mass = candidate_lenses.mass[idx_to_add]
        adaptive_tolerance = base_tolerance * (lens_mass / mass_scale) ** exponent

        # Check if adding the lens improves the reduced chi-squared beyond adaptive tolerance
        if min_chi2 < best_reduced_chi2 - adaptive_tolerance:
            # Update the best lenses and reduced chi-squared
            best_reduced_chi2 = min_chi2

            # Add the lens to selected_lenses
            if lens_type == 'NFW':
                selected_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx_to_add]),
                    z=np.append(selected_lenses.z, candidate_lenses.z[idx_to_add]),
                    concentration=np.append(selected_lenses.concentration, candidate_lenses.concentration[idx_to_add]),
                    mass=np.append(selected_lenses.mass, candidate_lenses.mass[idx_to_add]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx_to_add])
                )
            elif lens_type == 'SIS':
                selected_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx_to_add]),
                    te=np.append(selected_lenses.te, candidate_lenses.te[idx_to_add]),
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx_to_add])
                )

            # Remove the lens from remaining_indices
            remaining_indices = np.delete(remaining_indices, min_index)
            improved = True
        else:
            # No improvement beyond adaptive tolerance, stop the iteration
            break
    
    # Check to see if the number of lenses is zero
    if len(selected_lenses.x) == 0:
        print('No lenses selected.')
        return None, np.inf
    else:
        return selected_lenses, best_reduced_chi2


def optimize_lens_strength(sources, lenses, use_flags, lens_type='SIS', z_source=0.8):
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
        # Optimize mass for each lens individually - this is verified to be the better approach as of 9/26/2024
        
        for i in range(len(lenses.x)):
            guess = [np.log10(lenses.mass[i])]
            params = [
                'NFW', 'constrained',
                lenses.x[i], lenses.y[i], lenses.redshift,
                lenses.concentration[i], sources, use_flags, 
                z_source
            ]

            result = opt.minimize(
                chi2wrapper, guess, args=params,
                method='L-BFGS-B',
                bounds=[(10, 16)],
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            lenses.mass[i] = 10 ** result.x
            lenses.calculate_concentration()
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def update_chi2_values(sources, lenses, use_flags, lens_type='NFW', z_source=0.8):
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
    
    global_chi2 = metric.calculate_chi_squared(sources, lenses, use_flags, lens_type=lens_type, z_source = z_source)
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
                sources, one_halo, use_flags, lens_type=lens_type, z_source = z_source
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
            return metric.calculate_chi_squared(params[0], lenses, params[1], use_weights=False)
        elif constraint_type == 'constrained':
            # Optimize strength parameter(s)
            lenses = halo_obj.SIS_Lens(params[0], params[1], guess, np.empty_like(params[0]))
            dof = metric.calc_degrees_of_freedom(params[2], lenses, params[3])
            chi2 = metric.calculate_chi_squared(params[2], lenses, params[3])
            return np.abs(chi2 / dof - 1)

    elif model_type == 'NFW':
        if constraint_type == 'unconstrained':
            print('This function was called')
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
            return metric.calculate_chi_squared(params[4], lenses, params[5], lens_type='NFW', z_source=params[6])
        
        elif constraint_type == 'dual':
            # Fit both mass and concentration
            lenses = halo_obj.NFW_Lens(
                params[0], params[1], np.zeros_like(params[0]),
                guess[1], 10 ** guess[0], params[2], np.empty_like(params[0])
            )
            return metric.calculate_chi_squared(params[3], lenses, params[4], lens_type='NFW', use_weights=False)


    else:
        raise ValueError(f"Invalid lensing model: {model_type}")