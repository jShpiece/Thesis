"""
Module for statistical calculations in gravitational lensing analysis.

This module provides functions to calculate the degrees of freedom and the chi-squared
statistic for gravitational lensing models, aiding in the assessment of lens model fits
to observed data.

Functions:
    - calc_degrees_of_freedom
    - calculate_chi_squared
"""

import numpy as np
import copy

def calc_degrees_of_freedom(sources, lenses, use_flags):
    """
    Calculate the degrees of freedom for a given set of sources and lenses.

    Parameters:
        sources (Source): Source object containing source positions and lensing signals.
        lenses (Lens): Lens object containing lens positions and parameters.
        use_flags (list of bool): Flags indicating which lensing signals are used 
                                [use_shear, use_flexion, use_g_flexion].

    Returns:
        int or float: The degrees of freedom for the system. Returns np.inf if degrees
                    of freedom are zero or negative.
    """
    # Number of lensing signals used (True in use_flags)
    num_signals = np.sum(use_flags)
    # Each signal component has 2 parameters (e.g., e1 and e2 for shear)
    num_source_params = 2 * num_signals * len(sources.x)
    # Each lens has 3 parameters: x, y, and strength (e.g., Einstein radius)
    num_lens_params = 3 * len(lenses.x)
    # Degrees of freedom: observations minus fitted parameters
    dof = num_source_params - num_lens_params
    if dof <= 0:
        return np.inf
    return dof

def calculate_chi_squared(sources, lenses, flags, lens_type='SIS', z_source = 0.8) -> float:
    """
    Calculate the chi-squared statistic for the difference between observed and modeled source properties.

    This function computes the chi-squared value by comparing the observed lensing signals
    (e.g., shear, flexion) with those predicted by the lens model, taking into account the
    uncertainties in the observations.

    Parameters:
        sources (Source): Source object containing observed source properties and uncertainties.
        lenses (Lens): Lens object containing lens properties.
        flags (list of bool): Flags indicating which lensing effects to include 
                            [use_shear, use_flexion, use_g_flexion].
        lens_type (str): The lensing model to use ('SIS' or 'NFW'). Default is 'SIS'.
        use_weights (bool): If True, weight the chi-squared contributions by source weights.
                            Default is False.
        sigma (float): Standard deviation for Gaussian weighting. Default is 1.0.

    Returns:
        float: The total chi-squared value, including any penalties for lens properties.
    """
    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = flags

    # Create a copy of sources and reset lensing signals to zero
    source_clone = copy.deepcopy(sources)
    source_clone.zero_lensing_signals()

    # Apply lensing effects to the cloned source based on the lensing model
    source_clone.apply_lensing(lenses, lens_type=lens_type, z_source=z_source)

    # Calculate the squared differences for each lensing signal component
    chi_squared_components = {}
    if use_shear:
        chi_squared_shear = (
            (source_clone.e1 - sources.e1) ** 2 +
            (source_clone.e2 - sources.e2) ** 2
        ) / sources.sigs ** 2
        chi_squared_components['shear'] = chi_squared_shear

    if use_flexion:
        chi_squared_flexion = (
            (source_clone.f1 - sources.f1) ** 2 +
            (source_clone.f2 - sources.f2) ** 2
        ) / sources.sigf ** 2
        chi_squared_components['flexion'] = chi_squared_flexion

    if use_g_flexion:
        chi_squared_g_flexion = (
            (source_clone.g1 - sources.g1) ** 2 +
            (source_clone.g2 - sources.g2) ** 2
        ) / sources.sigg ** 2
        chi_squared_components['g_flexion'] = chi_squared_g_flexion

    # Sum the chi-squared components
    total_chi_squared_array = np.zeros_like(sources.x)
    for component in chi_squared_components.values():
        total_chi_squared_array += component

    total_chi_squared = np.sum(total_chi_squared_array)


    # Define penalty functions for lens parameters
    def einstein_radius_penalty(eR, limit=40.0, penalty_factor=1000.0):
        """
        Penalty function for the Einstein radius of SIS lenses.

        Parameters:
            eR (float): Einstein radius.
            limit (float): Upper limit for the Einstein radius before penalties apply.
            penalty_factor (float): Penalty scaling factor.

        Returns:
            float: Penalty value for the given Einstein radius.
        """
        if np.abs(eR) > limit:
            return penalty_factor * (np.abs(eR) - limit) ** 2
        return 0.0

    # Calculate and add penalties for the lenses
    if lens_type == 'SIS':
        # Apply penalties for SIS lenses if Einstein radius exceeds limit
        penalties = sum(einstein_radius_penalty(eR) for eR in lenses.te)
        total_chi_squared += penalties
    elif lens_type == 'NFW':
        # No penalties defined for NFW lenses in this function
        pass



    # Return the total chi-squared including penalties
    return total_chi_squared

def gaussian_weighting(sources, lenses, sigma=50.0, min_distance_threshold=1e-5):
    """
    Computes Gaussian weights for each source based on its distance to the closest lens.
    
    Parameters:
        sources (Source): Source object containing source positions (sources.x, sources.y).
        lenses (NFW_Lens): NFW_Lens object containing lens positions (lenses.x, lenses.y).
        sigma (float): Standard deviation of the Gaussian function. Controls the weight fall-off. Default is 1.0.
        min_distance_threshold (float): Minimum distance to avoid zero distance issues. Default is 1e-5.
        
    Returns:
        np.ndarray: Array of weights for each source.
    """
    # Check if sigma is valid
    if sigma <= 0:
        raise ValueError(f"Invalid sigma value: {sigma}. Sigma must be positive and non-zero.")

    # Number of sources
    n_sources = len(sources.x)
    
    # Initialize weight array
    weights = np.zeros(n_sources)

    # Calculate weights for each source based on the distance to the nearest lens
    for i in range(n_sources):
        # Calculate the distance from the current source to all lenses
        distances = np.sqrt((sources.x[i] - lenses.x) ** 2 + (sources.y[i] - lenses.y) ** 2)
        
        # Apply minimum distance threshold to avoid zero distances
        distances = np.clip(distances, min_distance_threshold, None)
        
        # Find the minimum distance to any lens
        min_distance = np.min(distances)
        
        # Calculate Gaussian weight based on the minimum distance
        weights[i] = np.exp(-min_distance ** 2 / (2 * sigma ** 2))
    
    # Check for sum of weights before normalization to avoid division by zero
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        # raise ValueError("Sum of Gaussian weights is zero. Check source and lens positions or modify sigma.")
        weights += 1e-5

    # Normalize weights to sum to 1
    weights /= weight_sum
    
    return weights