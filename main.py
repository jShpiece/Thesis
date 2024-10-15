"""
Module for fitting gravitational lensing fields.

This module provides the `fit_lensing_field` function, which reconstructs the lensing field
from a set of source observations (positions, ellipticity, flexion signals).
The lensing field is represented by a set of lenses (positions and parameters),
and can be modeled as Singular Isothermal Spheres (SIS) or Navarro-Frenk-White (NFW) halos.

Functions:
    - fit_lensing_field
"""

import pipeline

def fit_lensing_field(sources, xmax, flags=False, use_flags=[True, True, True], lens_type='SIS'):
    """
    Reconstructs the lensing field that produced the observed source properties.

    This function takes in a set of sources with positions, ellipticity, and flexion signals,
    and attempts to reconstruct the lensing field that produced them.
    The lensing field is represented by a set of lenses with positions and parameters.
    Lenses can be modeled as Singular Isothermal Spheres (SIS) by default, or as
    Navarro-Frenk-White (NFW) halos.

    Parameters:
        sources (Source): An object containing source properties and their uncertainties.
        xmax (float): The maximum distance from the center of the field to consider for lenses.
        flags (bool, optional): Whether to print out step information. Default is False.
        use_flags (list of bool, optional): Flags indicating which lensing effects to include
            [use_shear, use_flexion, use_g_flexion]. Default is [True, True, True].
        lens_type (str, optional): The type of lens to use ('SIS' or 'NFW'). Default is 'SIS'.

    Returns:
        Tuple[Lens, float]: A tuple containing:
            - lenses (Lens): An object representing the lenses that best fit the source properties.
            - reduced_chi2 (float): The reduced chi-squared value for the best fit.
    """
    def print_step_info(flags, message, lenses, reduced_chi2):
        """
        Helper function to print out step information during the fitting process.

        Parameters:
            flags (bool): Whether to print out the information.
            message (str): Message describing the current step.
            lenses (Lens): The current set of lenses.
            reduced_chi2 (float): The current reduced chi-squared value.
        """
        if flags:
            print(message)
            print(f'Number of lenses: {len(lenses.x)}')
            if reduced_chi2 is not None:
                print(f'Reduced Chi^2: {reduced_chi2:.4f}')

    # Step 1: Generate initial candidate lenses from source guesses
    z_lens = 0.194  # Example redshift of the lens
    lenses = pipeline.generate_initial_guess(sources, lens_type=lens_type, z_l=z_lens)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type=lens_type)
    print_step_info(flags, "Initial Guesses:", lenses, reduced_chi2)

    # Step 2: Optimize lens positions via local minimization
    lenses = pipeline.optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type=lens_type)
    print_step_info(flags, "After Local Minimization:", lenses, reduced_chi2)

    # Step 3: Filter out lenses that are too close to sources or too far from the center
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type=lens_type)
    print_step_info(flags, "After Filtering:", lenses, reduced_chi2)

    # Step 4: Iteratively eliminate lenses to find the best reduced chi-squared value
    lenses, _ = pipeline.forward_lens_selection(sources, lenses, use_flags, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type=lens_type)
    print_step_info(flags, "After Forward Selection:", lenses, reduced_chi2)

    # Step 5: Merge lenses that are too close to each other
    # Calculate source density (ns) for merger threshold calculation
    area = (2 * xmax) ** 2  # Total area of the field
    ns = len(sources.x) / area  # Source density per unit area
    merger_threshold = ns**(-1/2) if ns > 0 else 1.0  # Avoid division by zero
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold=merger_threshold, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type=lens_type)
    print_step_info(flags, "After Merging Lenses:", lenses, reduced_chi2)

    # Step 6: Perform a final optimization on the lens strengths
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type=lens_type)
    # Always use all signals for final fit
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, [True, True, True], lens_type=lens_type)
    print_step_info(flags, "After Final Optimization:", lenses, reduced_chi2)

    return lenses, reduced_chi2
