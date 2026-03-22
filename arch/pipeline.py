"""
Module for gravitational lensing pipeline functions.

This module provides functions to generate initial guesses for lens positions,
optimize lens positions and strengths, filter and merge lens positions,
and other utilities used in gravitational lensing analysis.

Statistical functions (calculate_total_chi2, compute_lambda_sl, calc_strong_dof)
live in arch.metric.

Functions:
    - generate_initial_guess
    - optimize_lens_positions
    - filter_lens_positions
    - merge_close_lenses
    - forward_lens_selection
    - optimize_lens_strength
    - update_chi2_values
    - chi2wrapper
"""

import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize, minimize_scalar

import arch.utils as utils  # Custom utility functions
import arch.source_obj as source_obj # Source object
import arch.halo_obj as halo_obj # Halo object
import arch.metric as metric # Metric calculation functions

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
        # Avoid division by zero in flexion
        flexion = np.where(flexion == 0, 1e-10, flexion)

        # Estimate radial distance to lens
        # The factor 1.45 is determined numerically; adjust based on model calibration if necessary
        r = 2.0 * gamma / flexion

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
                    concentration=0.0, # Placeholder
                    mass=mass,
                    redshift=z_l,
                    chi2=0.0
                )
                lens.calculate_concentration()

                # Create a single-source object
                source = source_obj.Source(
                    x=sources.x[i], y=sources.y[i],
                    e1=0.0, e2=0.0,
                    f1=0.0, f2=0.0,
                    g1=0.0, g2=0.0,
                    sigs=1.0, sigf=1.0, sigg=1.0,
                    redshift=sources.redshift[i]
                )
                # Compute the lensing signals from the lens
                _, _, _, f1_model, f2_model, _, _ = utils.calculate_lensing_signals_nfw(lens, source)

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

def optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type='SIS',
                           use_strong_lensing: bool = False, lambda_sl: float = None,
                           all_sources: bool = False):
    """
    Optimizes lens positions via local minimization.
    By default minimizes relative to sources within 20 arcsec of the lens (NFW).

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Initial lens object with estimated positions and parameters.
        use_flags (list): Flags indicating which data to use in optimization.
        lens_type (str): Type of lens model ('SIS' or 'NFW').
        use_strong_lensing (bool): Whether to include strong lensing in the objective.
        lambda_sl (float or None): Pre-computed SL weight. If None, fallback is used.
        all_sources (bool): If True, use all sources (no 20-arcsec filter). Use for
            SL-aware refinement after forward selection. Default False.

    Returns:
        SIS_Lens or NFW_Lens: Lenses with optimized positions.
    """
    # Optimizer parameters
    num_iterations = 1e6

    if lens_type == 'SIS':
        for i in range(len(lenses.x)):
            one_source = source_obj.Source(
                sources.x[i], sources.y[i],
                sources.e1[i], sources.e2[i],
                sources.f1[i], sources.f2[i],
                sources.g1[i], sources.g2[i],
                sources.sigs[i], sources.sigf[i], sources.sigg[i],
                sources.redshift[i],
                strong_systems=getattr(sources, "strong_systems", None),
            )
            guess = [lenses.x[i], lenses.y[i], lenses.te[i]]
            opts = {"use_strong_lensing": use_strong_lensing, "lambda_sl": lambda_sl}
            params = ['SIS', 'unconstrained', one_source, use_flags, opts]
            result = minimize(
                chi2wrapper, guess, args=params, method='L-BFGS-B',
                options={'maxiter': int(1e6), 'ftol': 1e-6}
            )
            lenses.x[i], lenses.y[i], lenses.te[i] = result.x[0], result.x[1], result.x[2]

    elif lens_type == 'NFW':
        for i in range(len(lenses.x)):
            # Initial guess: [x, y, log10(mass)] - optimize in log-space for mass
            initial_guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])]

            # Define bounds for x, y, and log10(mass)
            bounds = [
                (lenses.x[i] - xmax*2, lenses.x[i] + xmax*2),  
                (lenses.y[i] - xmax*2, lenses.y[i] + xmax*2),  # Allow the minimizer to move the lens out of the field - corresponds to a "delete" operation
                (10, 17)  # Mass bounds in log10(M_sun)
            ]

            # Select source subset: all sources (SL refinement pass) or within 20" (initial pass)
            if all_sources:
                opt_sources = sources
            else:
                opt_sources = sources.copy()
                distance = np.hypot(lenses.x[i] - sources.x, lenses.y[i] - sources.y)
                opt_sources.remove(np.where(distance > 20)[0])

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

                # Compute chi-squared:
            # - all_sources=False (initial pass): WL-only with filtered local sources
            # - all_sources=True (refinement pass): WL + no-mag SL with all sources
                if use_strong_lensing and lambda_sl is not None and all_sources:
                    chi2, _, _ = metric.calculate_total_chi2(
                        opt_sources, lens, use_flags, lens_type='NFW',
                        use_strong_lensing=True, lambda_sl=lambda_sl,
                        use_magnification_correction_sl=False,
                    )
                else:
                    chi2 = metric.calculate_chi_squared(opt_sources, lens, use_flags, lens_type='NFW')
                return chi2

            # Run the optimizer
            result = minimize(
                objective_function,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': num_iterations, 'ftol': 1e-6}
            )
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
    # Identify lenses that are too far from the center
    too_far = np.sqrt(lenses.x ** 2 + lenses.y ** 2) > xmax * 1.5

    if lens_type == 'SIS':
        # SIS-specific condition: Einstein radius too small or negative
        invalid_te = lenses.te  < 1e-3 
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
    
    # Check if any lenses remain after filtering - if not, raise an error
    if len(lenses.x) == 0:
        raise ValueError('No valid lenses remain after filtering.')

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
        strength[i] = total_weight / 2  # Update strength of lens i
        lenses.remove([j]) # Remove lens at index j

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
    return lenses

def forward_lens_selection(
    sources, candidate_lenses, use_flags, lens_type='NFW',
    base_tolerance=0.003, mass_scale=1e13, exponent=-1.0,
    use_strong_lensing: bool = False, lambda_sl: float = None
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
        use_strong_lensing (bool): Whether to include strong lensing in the objective.
        lambda_sl (float or None): Pre-computed SL weight. If None, fallback is used.
    
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

            # Compute chi-squared and reduced chi-squared (Include SL if applicable)
            # No magnification correction: magnification-corrected chi2_SL is
            # non-monotonic in distance from truth, which would bias selection.
            chi2, dof, _ = metric.calculate_total_chi2(
                sources, test_lenses, use_flags, lens_type=lens_type,
                use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            reduced_chi2 = chi2 / dof if dof > 0 else np.inf
            chi2_list.append(reduced_chi2)
            lens_indices.append(idx)

        # Find the lens whose addition leads to the best reduced chi-squared
        min_chi2 = min(chi2_list)
        min_index = chi2_list.index(min_chi2)
        idx_to_add = lens_indices[min_index]

        # Adaptive tolerance calculation based on the strength of the lens
        if lens_type == 'NFW':
            lens_strength = candidate_lenses.mass[idx_to_add]
            adaptive_tolerance = base_tolerance * (lens_strength / mass_scale) ** exponent
        elif lens_type == 'SIS':
            lens_strength = np.abs(candidate_lenses.te[idx_to_add])
            # For SIS, use Einstein radius for adaptive tolerance
            # Adjust mass_scale to a suitable te_scale (e.g., 1.0 arcsec)
            te_scale = 1.0  # Characteristic Einstein radius scale
            adaptive_tolerance = base_tolerance * (lens_strength / te_scale) ** exponent
        else:
            raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

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

def optimize_lens_strength(sources, lenses, use_flags, lens_type='SIS',
                          use_strong_lensing: bool = False, lambda_sl: float = None):
    """
    Optimizes the strength parameters (Einstein radius or mass) of the lenses.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (SIS_Lens or NFW_Lens): Lens object containing lens positions and parameters.
        use_flags (list): Flags indicating which data to use in optimization.
        lens_type (str): Type of lens model ('SIS' or 'NFW').
        use_strong_lensing (bool): Whether to include strong lensing in the objective.
        lambda_sl (float or None): Pre-computed SL weight. If None, fallback is used.

    Returns:
        SIS_Lens or NFW_Lens: Lenses with optimized strengths.
    """
    opts = {"use_strong_lensing": use_strong_lensing, "lambda_sl": lambda_sl}

    if lens_type == 'SIS':
        guess = lenses.te
        params = ['SIS', 'constrained', lenses.x, lenses.y, sources, use_flags, opts]
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
        for i in range(len(lenses.x)):
            guess = [np.log10(lenses.mass[i])]
            params = [
                'NFW', 'constrained',
                lenses.x[i], lenses.y[i], lenses.redshift,
                lenses.concentration[i], sources, use_flags, opts
            ]

            chi2_fn = lambda x: chi2wrapper(x, params)
            # Robust 1-D bounded search over log10(M)
            res = minimize_scalar(
                chi2_fn,
                bounds=(10.0, 17.0),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 2000}
            )
            lenses.mass[i] = 10 ** res.x
            lenses.calculate_concentration()
    else:
        raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

    return lenses


def update_chi2_values(sources, lenses, use_flags, lens_type='NFW',
                      use_strong_lensing: bool = False, lambda_sl: float = None):
    """
    Updates per-lens chi2 (WL-only per-lens is fine) and returns reduced chi2 for the
    combined WL+SL objective at the global level.

    Parameters:
        sources (Source): Source object.
        lenses (SIS_Lens or NFW_Lens): Lens object.
        use_flags (list of bool): Which lensing signals to use.
        lens_type (str): 'SIS' or 'NFW'.
        use_strong_lensing (bool): Whether to include strong lensing.
        lambda_sl (float or None): Pre-computed SL weight.

    Returns:
        float: Reduced chi-squared for the combined WL+SL objective.

    NOTE: per-lens chi2 is left as WL-only here (prototype). The returned reduced chi2
    is computed from the combined objective.
    """
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        sources, lenses, use_flags, lens_type=lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    reduced_chi2 = chi2_total / dof_total if dof_total != 0 else np.inf

    # --- Keep existing per-lens bookkeeping (WL-only) ---
    global_chi2_wl = comps["chi2_wl"]

    if len(lenses.x) == 1:
        lenses.chi2[0] = global_chi2_wl
    else:
        for i in range(len(lenses.x)):
            if lens_type == "NFW":
                one_halo = halo_obj.NFW_Lens(
                    lenses.x[i], lenses.y[i], lenses.z[i],
                    lenses.concentration[i], lenses.mass[i],
                    lenses.redshift, [0]
                )
                one_halo.calculate_concentration()
            elif lens_type == "SIS":
                one_halo = halo_obj.SIS_Lens(
                    lenses.x[i], lenses.y[i], lenses.te[i],
                    [0]
                )
            else:
                raise ValueError('Invalid lens type - must be either "SIS" or "NFW"')

            lenses.chi2[i] = metric.calculate_chi_squared(
                sources, one_halo, use_flags, lens_type=lens_type
            )

    return reduced_chi2

def chi2wrapper(guess, params):
    """
    Wrapper used by scipy optimizers.

    Backwards compatible: if lambda_sl is not passed in the opts dict,
    it defaults to None (which triggers the fallback inside
    calculate_total_chi2).

    The recommended usage is for callers to pre-compute lambda_sl via
    ``compute_lambda_sl()`` and inject it into the opts dict:

        opts = {"use_strong_lensing": True, "lambda_sl": 2.3}
        params = ['SIS', 'unconstrained', sources, use_flags, opts]
    """
    if isinstance(params, tuple):
        params = list(params)

    model_type, constraint_type = params[0], params[1]
    tail = params[2:]

    # Optional options dict as final element (recommended)
    use_strong_lensing = False
    lambda_sl = None
    if len(tail) > 0 and isinstance(tail[-1], dict):
        opts = tail[-1]
        use_strong_lensing = bool(opts.get("use_strong_lensing", False))
        lambda_sl = opts.get("lambda_sl", None)
        tail = tail[:-1]

    if model_type == 'SIS':
        if constraint_type == 'unconstrained':
            # params expected: [sources, use_flags]
            sources = tail[0]
            use_flags = tail[1]
            lenses = halo_obj.SIS_Lens(guess[0], guess[1], guess[2], [0])
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="SIS",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == 'constrained':
            # params expected: [x_array, y_array, sources, use_flags]
            xl, yl, sources, use_flags = tail[0], tail[1], tail[2], tail[3]
            lenses = halo_obj.SIS_Lens(xl, yl, guess, np.empty_like(xl))
            chi2_total, dof_total, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="SIS",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return np.abs(chi2_total / dof_total - 1) if dof_total > 0 else np.inf

    elif model_type == 'NFW':
        if constraint_type == 'unconstrained':
            # tail expected: [sources, use_flags, concentration?, redshift?] (existing code style)
            sources = tail[0]
            use_flags = tail[1]
            concentration = tail[2]
            redshift = tail[3]
            lenses = halo_obj.NFW_Lens(
                guess[0], guess[1], np.zeros_like(guess[0]),
                concentration, 10 ** guess[2], redshift, [0]
            )
            lenses.calculate_concentration()
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == 'constrained':
            # params: [x, y, z_lens, concentration, sources, use_flags]
            xl, yl, z_lens, concentration, sources, use_flags = tail[0], tail[1], tail[2], tail[3], tail[4], tail[5]
            lenses = halo_obj.NFW_Lens(
                xl, yl, np.zeros_like(xl),
                concentration, 10 ** guess, z_lens, np.empty_like(np.atleast_1d(xl))
            )
            lenses.calculate_concentration()
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == 'dual':
            xl, yl, z_lens, sources, use_flags = tail[0], tail[1], tail[2], tail[3], tail[4]
            lenses = halo_obj.NFW_Lens(
                xl, yl, np.zeros_like(np.atleast_1d(xl)),
                guess[1], 10 ** guess[0], z_lens, np.empty_like(np.atleast_1d(xl))
            )
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

    raise ValueError(f"Invalid lensing model/constraint: {model_type} / {constraint_type}")