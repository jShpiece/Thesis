"""Greedy forward lens selection minimizing reduced chi-squared."""

import numpy as np

import arch.halo_obj as halo_obj
import arch.metric as metric


def forward_lens_selection(
    sources,
    candidate_lenses,
    use_flags,
    lens_type="NFW",
    base_tolerance=0.003,
    mass_scale=1e13,
    exponent=-1.0,
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
    kappa_scale: float = 0.1,
    strong_systems=None,
    return_lambda_sl: bool = False,
):
    """
    Selects the best combination of lenses by iteratively adding lenses
    to minimize the reduced chi-squared value, using an adaptive
    tolerance that depends on the strength of the candidate lens.

    Strong-lensing handling differs by lens type:
        - SIS:        SL can be included during selection if the caller
                      passes use_strong_lensing=True and a pre-computed
                      lambda_sl.  Existing behavior preserved.
        - NFW:        Same as SIS.
        - POWER_LAW:  SL is DELIBERATELY EXCLUDED from the selection
                      loop — empirical experience with the NFW pipeline
                      showed that injecting lambda_sl during selection
                      degrades mass recovery substantially.  After
                      selection completes, lambda_sl is computed once
                      via a reduced-chi-squared ratio (matching the
                      NFW-pipeline convention) and returned to the
                      caller, who freezes it through merging and
                      strength optimization.

    Adaptive tolerance:
        Delta_chi2_nu_tol = base_tolerance * (strength / strength_scale)^exponent
        - SIS:        strength = |te|, strength_scale = 1.0 arcsec
        - NFW:        strength = mass, strength_scale = mass_scale (default 1e13 M_sun)
        - POWER_LAW:  strength = kappa_star, strength_scale = kappa_scale (default 0.1)

    For exponent < 0 (default -1), this means weak halos must justify
    inclusion by a proportionally larger improvement; strong halos are
    held to a looser bar.

    Parameters
    ----------
    sources : Source
    candidate_lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
    use_flags : sequence of three bool
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    base_tolerance, mass_scale, exponent : adaptive tolerance parameters
    use_strong_lensing : bool
        SIS/NFW only.  Power-law mode forces this to False during
        selection.
    lambda_sl : float or None
        SIS/NFW pre-computed SL weight.  Power-law mode ignores this
        input and computes lambda_sl after selection completes.
    kappa_scale : float
        POWER_LAW characteristic kappa_star scale.  Default 0.1.
    strong_systems : iterable of StrongLensingSystem or None
        POWER_LAW only.  Required if any strong-lensing data exists,
        used to compute the post-selection lambda_sl.  If None, the
        returned lambda_sl is 0.0 (WL-only fit).
    return_lambda_sl : bool
        POWER_LAW only.  If True, return (selected_lenses, chi2_nu,
        lambda_sl_final) instead of (selected_lenses, chi2_nu).

    Returns
    -------
    selected_lenses, best_reduced_chi2 [, lambda_sl_final if requested]
    """
    # ----------------------------------------------------------------
    # Initialize an empty lens object based on lens_type
    # ----------------------------------------------------------------
    if lens_type == "NFW":
        selected_lenses = halo_obj.NFW_Lens(
            x=np.array([]),
            y=np.array([]),
            z=np.array([]),
            concentration=np.array([]),
            mass=np.array([]),
            redshift=candidate_lenses.redshift,
            chi2=np.array([]),
        )
    elif lens_type == "SIS":
        selected_lenses = halo_obj.SIS_Lens(
            x=np.array([]),
            y=np.array([]),
            te=np.array([]),
            chi2=np.array([]),
        )
    elif lens_type == "POWER_LAW":
        selected_lenses = halo_obj.PowerLawHalo(
            x=np.array([]),
            y=np.array([]),
            kappa_star=np.array([]),
            slope=np.array([]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift,
            chi2=np.array([]),
        )
        # Force WL-only during the selection loop.
        use_strong_lensing = False
        lambda_sl = None
    else:
        raise ValueError("Unsupported lens type. Choose 'NFW', 'SIS', or 'POWER_LAW'.")

    remaining_indices = np.arange(len(candidate_lenses.x))
    best_reduced_chi2 = np.inf
    improved = True

    # ----------------------------------------------------------------
    # Main selection loop
    # ----------------------------------------------------------------
    while improved and len(remaining_indices) > 0:
        improved = False
        chi2_list = []
        lens_indices = []

        for idx in remaining_indices:
            # Build a test lens set with the candidate added
            if lens_type == "NFW":
                test_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    z=np.append(selected_lenses.z, candidate_lenses.z[idx]),
                    concentration=np.append(
                        selected_lenses.concentration, candidate_lenses.concentration[idx]
                    ),
                    mass=np.append(selected_lenses.mass, candidate_lenses.mass[idx]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx]),
                )
            elif lens_type == "SIS":
                test_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    te=np.append(selected_lenses.te, candidate_lenses.te[idx]),
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx]),
                )
            elif lens_type == "POWER_LAW":
                test_lenses = halo_obj.PowerLawHalo(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    kappa_star=np.append(
                        selected_lenses.kappa_star, candidate_lenses.kappa_star[idx]
                    ),
                    slope=np.append(selected_lenses.slope, candidate_lenses.slope[idx]),
                    theta_star=candidate_lenses.theta_star,
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx]),
                )

            # Compute reduced chi-squared (WL only for POWER_LAW)
            chi2, dof, _ = metric.calculate_total_chi2(
                sources,
                test_lenses,
                use_flags,
                lens_type=lens_type,
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
            )
            reduced_chi2 = chi2 / dof if dof > 0 else np.inf
            chi2_list.append(reduced_chi2)
            lens_indices.append(idx)

        # Best candidate this iteration
        min_chi2 = min(chi2_list)
        min_index = chi2_list.index(min_chi2)
        idx_to_add = lens_indices[min_index]

        # Adaptive tolerance based on the candidate's strength
        if lens_type == "NFW":
            lens_strength = candidate_lenses.mass[idx_to_add]
            adaptive_tolerance = base_tolerance * (lens_strength / mass_scale) ** exponent
        elif lens_type == "SIS":
            lens_strength = np.abs(candidate_lenses.te[idx_to_add])
            te_scale = 1.0
            adaptive_tolerance = base_tolerance * (lens_strength / te_scale) ** exponent
        elif lens_type == "POWER_LAW":
            lens_strength = np.abs(candidate_lenses.kappa_star[idx_to_add])
            adaptive_tolerance = base_tolerance * (lens_strength / kappa_scale) ** exponent

        # Accept if improvement exceeds tolerance
        if min_chi2 < best_reduced_chi2 - adaptive_tolerance:
            best_reduced_chi2 = min_chi2

            if lens_type == "NFW":
                selected_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx_to_add]),
                    z=np.append(selected_lenses.z, candidate_lenses.z[idx_to_add]),
                    concentration=np.append(
                        selected_lenses.concentration, candidate_lenses.concentration[idx_to_add]
                    ),
                    mass=np.append(selected_lenses.mass, candidate_lenses.mass[idx_to_add]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx_to_add]),
                )
            elif lens_type == "SIS":
                selected_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx_to_add]),
                    te=np.append(selected_lenses.te, candidate_lenses.te[idx_to_add]),
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx_to_add]),
                )
            elif lens_type == "POWER_LAW":
                selected_lenses = halo_obj.PowerLawHalo(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx_to_add]),
                    kappa_star=np.append(
                        selected_lenses.kappa_star, candidate_lenses.kappa_star[idx_to_add]
                    ),
                    slope=np.append(selected_lenses.slope, candidate_lenses.slope[idx_to_add]),
                    theta_star=candidate_lenses.theta_star,
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2, candidate_lenses.chi2[idx_to_add]),
                )

            remaining_indices = np.delete(remaining_indices, min_index)
            improved = True
        else:
            break

    # ----------------------------------------------------------------
    # Empty selection
    # ----------------------------------------------------------------
    if len(selected_lenses.x) == 0:
        print("No lenses selected.")
        if lens_type == "POWER_LAW" and return_lambda_sl:
            return None, np.inf, 0.0
        return None, np.inf

    # ----------------------------------------------------------------
    # POWER_LAW post-selection: compute lambda_sl once and freeze
    # ----------------------------------------------------------------
    if lens_type == "POWER_LAW":
        lambda_sl_final = metric._compute_lambda_sl_power_law(
            sources,
            selected_lenses,
            use_flags,
            strong_systems=strong_systems,
        )
        if return_lambda_sl:
            return selected_lenses, best_reduced_chi2, lambda_sl_final
        # Default tuple unchanged for backward-compat callers
        return selected_lenses, best_reduced_chi2

    return selected_lenses, best_reduced_chi2
