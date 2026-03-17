"""
Module for statistical calculations in gravitational lensing analysis.

This module provides functions to calculate the degrees of freedom and the chi-squared
statistic for gravitational lensing models, aiding in the assessment of lens model fits
to observed data.

Functions:
    - calc_degrees_of_freedom
    - calculate_chi_squared
    - calc_strong_dof
    - compute_lambda_sl
    - calculate_total_chi2
"""

import numpy as np
import copy

import arch.utils as utils

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


def calculate_chi_squared(sources, lenses, flags, lens_type='SIS') -> float:
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
    source_clone.apply_lensing(lenses, lens_type=lens_type)

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


def calc_strong_dof(sources) -> int:
    """
    Degrees of freedom contribution from strong-lensing source-plane scatter.

    For each multiply-imaged system i with N_i images:
        data constraints = 2 * N_i   (x, y per image)
        nuisance params  = 2         (beta_x, beta_y)
        dof_i = 2 * (N_i - 1)

    Total dof_sl = 2 * sum_i (N_i - 1)

    Returns 0 if no strong systems exist.

    Parameters
    ----------
    sources : Source
        Must have a ``strong_systems`` attribute (list of StrongLensingSystem).

    Returns
    -------
    int
        Strong-lensing degrees of freedom.
    """
    if not hasattr(sources, "strong_systems") or sources.strong_systems is None:
        return 0
    if len(sources.strong_systems) == 0:
        return 0
    return int(2 * sum((sys.n_images - 1) for sys in sources.strong_systems))

 
 
def compute_lambda_sl(sources, lenses, use_flags, lens_type='SIS'):
    """
    Pre-compute the strong-lensing weight lambda_sl at the current
    (typically initial) parameter values.
 
    This function is intended to be called **once** before an optimiser
    loop begins.  The returned scalar is then passed as a fixed constant
    into every ``calculate_total_chi2`` / ``chi2wrapper`` call so that
    the objective function seen by the optimiser is smooth and
    stationary.
 
    The weight equalises the reduced chi-squared of the WL and SL
    contributions:
 
        lambda_sl = (chi2_WL / dof_WL) / (chi2_SL / dof_SL)
 
    If either dataset has zero degrees of freedom, zero chi-squared,
    or if strong lensing is absent, we return 1.0 (the proper-likelihood
    default, i.e. assume both likelihoods are correctly normalised).
 
    Parameters
    ----------
    sources : Source
        Must carry ``strong_systems`` if SL is to contribute.
    lenses : SIS_Lens or NFW_Lens
        Current (initial) lens model parameters.
    use_flags : list of bool
        [use_shear, use_flexion, use_g_flexion].
    lens_type : str
        'SIS' or 'NFW'.
 
    Returns
    -------
    lambda_sl : float
        Pre-computed weight for the SL chi2 term.
    """
    # ── WL evaluation ──
    chi2_wl = calculate_chi_squared(sources, lenses, use_flags, lens_type=lens_type)
    dof_wl = calc_degrees_of_freedom(sources, lenses, use_flags)
 
    # ── SL evaluation ──
    has_sl = (hasattr(sources, "strong_systems")
              and sources.strong_systems is not None
              and len(sources.strong_systems) > 0)
 
    if not has_sl:
        return 1.0  # no SL data — default to proper-likelihood weight
 
    if lens_type == "SIS":
        chi2_sl = utils.chi2_strong_source_plane_sis(lenses, sources.strong_systems)
    elif lens_type == "NFW":
        chi2_sl = utils.chi2_strong_source_plane_nfw(lenses, sources.strong_systems)
    else:
        return 1.0  # unknown lens type — fall back to proper-likelihood default
    dof_sl = calc_strong_dof(sources)
 
    # Guard against degenerate cases
    if dof_wl <= 0 or dof_sl <= 0 or chi2_sl <= 0 or chi2_wl <= 0:
        return 1.0
 
    rchi2_wl = chi2_wl / dof_wl
    rchi2_sl = chi2_sl / dof_sl
 
    if rchi2_sl <= 0:
        return 1.0
 
    print(f"Pre-computed lambda_sl: {rchi2_wl:.3f} / {rchi2_sl:.3f} = {rchi2_wl / rchi2_sl:.3f}")
    return float(rchi2_wl / rchi2_sl)
 
 

def calculate_total_chi2(
    sources,
    lenses,
    use_flags,
    lens_type: str = "NFW",
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
):
    """
    Total chi2 = chi2_WL + lambda_sl * chi2_SL  (SL implemented for SIS and NFW).
 
    The relative weight lambda_sl between weak and strong lensing can be
    supplied in three ways (in order of precedence):
 
        1. Explicitly via the ``lambda_sl`` keyword  — used as-is.
           This is the recommended path: the caller pre-computes
           lambda_sl once at the initial parameter values via
           ``compute_lambda_sl()`` and holds it fixed throughout
           the entire optimisation call so the objective is smooth.
 
        2. If ``lambda_sl is None`` and strong lensing is active,
           a fallback reduced-chi2 ratio is computed at the *current*
           parameter values.  This is provided as a safety net but
           should NOT be relied upon inside an optimiser loop (it
           makes the objective non-stationary).
 
        3. If strong lensing is inactive, lambda_sl = 0 regardless.
 
    Returns
    -------
    chi2_total : float
    dof_total  : int
    components : dict with keys chi2_wl, chi2_sl, dof_wl, dof_sl, lambda_sl
    """
    # ── WL part (existing behavior) ──
    chi2_wl = calculate_chi_squared(sources, lenses, use_flags, lens_type=lens_type)
    dof_wl = calc_degrees_of_freedom(sources, lenses, use_flags)
 
    chi2_sl = 0.0
    dof_sl = 0
 
    # ── SL part (only if present AND requested) ──
    has_sl = (hasattr(sources, "strong_systems")
              and sources.strong_systems is not None
              and len(sources.strong_systems) > 0)
 
    if has_sl and use_strong_lensing:
        if lens_type == "SIS":
            chi2_sl = utils.chi2_strong_source_plane_sis(lenses, sources.strong_systems)
        elif lens_type == "NFW":
            chi2_sl = utils.chi2_strong_source_plane_nfw(lenses, sources.strong_systems)
        else:
            raise NotImplementedError(
                f"Strong-lensing chi2 not implemented for lens_type='{lens_type}'."
            )
        dof_sl = calc_strong_dof(sources)
 
    # ── Determine lambda ──
    if lambda_sl is not None:
        # Path 1: caller supplied a pre-computed weight (recommended)
        _lambda = float(lambda_sl)
    elif use_strong_lensing and dof_sl > 0 and chi2_sl > 0:
        # Path 2: fallback reduced-chi2 equalisation at current params
        rchi2_wl = chi2_wl / dof_wl if dof_wl > 0 else 1.0
        rchi2_sl = chi2_sl / dof_sl if dof_sl > 0 else 1.0
        _lambda = rchi2_wl / rchi2_sl if rchi2_sl > 0 else 1.0
    else:
        # Path 3: no strong lensing contribution
        _lambda = 0.0
 
    chi2_total = float(chi2_wl) + _lambda * float(chi2_sl)
    dof_total = int(dof_wl) + int(dof_sl)
 
    components = {
        "chi2_wl": float(chi2_wl),
        "chi2_sl": float(chi2_sl),
        "dof_wl": int(dof_wl),
        "dof_sl": int(dof_sl),
        "lambda_sl": float(_lambda),
    }
    return chi2_total, dof_total, components