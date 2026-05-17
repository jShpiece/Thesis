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

import copy

import numpy as np

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


def calculate_chi_squared(sources, lenses, flags, lens_type="SIS") -> float:
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
            (source_clone.e1 - sources.e1) ** 2 + (source_clone.e2 - sources.e2) ** 2
        ) / sources.sigs**2
        chi_squared_components["shear"] = chi_squared_shear

    if use_flexion:
        chi_squared_flexion = (
            (source_clone.f1 - sources.f1) ** 2 + (source_clone.f2 - sources.f2) ** 2
        ) / sources.sigf**2
        chi_squared_components["flexion"] = chi_squared_flexion

    if use_g_flexion:
        chi_squared_g_flexion = (
            (source_clone.g1 - sources.g1) ** 2 + (source_clone.g2 - sources.g2) ** 2
        ) / sources.sigg**2
        chi_squared_components["g_flexion"] = chi_squared_g_flexion

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
    if lens_type == "SIS":
        # Apply penalties for SIS lenses if Einstein radius exceeds limit
        penalties = sum(einstein_radius_penalty(eR) for eR in lenses.te)
        total_chi_squared += penalties
    elif lens_type == "NFW":
        # No penalties defined for NFW lenses in this function
        pass

    # Return the total chi-squared including penalties
    return total_chi_squared


def calc_strong_dof(sources) -> int:
    """
    Degrees of freedom contribution from strong-lensing constraints.

    Two independent observables per multiply-imaged system:

    1. Source-plane scatter (positional):
        For system i with N_i images:
            data constraints = 2 * N_i   (x, y per image)
            nuisance params  = 2         (beta_x, beta_y)
            dof_scatter_i    = 2 * (N_i - 1)

    2. Flux ratios (when flux data is available):
        For system i with N_i images:
            data constraints = N_i - 1   (ratios relative to reference)
            nuisance params  = 0         (F_source cancels)
            dof_flux_i       = N_i - 1

    Total dof_sl = sum_i [ 2*(N_i - 1) + flux_dof_i ]

    For a 2-image system: dof = 2 (position) + 1 (flux) = 3.

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

    dof = 0
    for sls in sources.strong_systems:
        n = sls.n_images
        # Positional scatter: always present
        dof += 2 * (n - 1)
        # Flux ratios: only if system has flux data
        if getattr(sls, "has_flux", False):
            dof += n - 1

    return int(dof)


def compute_lambda_sl(sources, lenses, use_flags, lens_type="SIS"):
    """
    Compute lambda_sl such that SL and WL contribute COMPARABLE absolute
    chi-squared at the current lens model.

    Joint chi^2 = chi^2_WL + lambda_sl * chi^2_SL

    The convention used here is:

        lambda_sl = chi^2_WL / chi^2_SL

    so that lambda_sl * chi^2_SL = chi^2_WL at the lens model where it's
    computed.  This makes the SL contribution to the joint chi^2 a
    significant fraction (specifically half, when evaluated at the
    chosen reference model) — large enough to actually pull the
    optimizer, but not so large that SL overwhelms WL.

    Why NOT lambda = rchi^2_WL / rchi^2_SL (the previous convention):
        That formula divides by dof on both sides, equalising the
        per-DOF chi-squared between the two probes.  It works correctly
        when both probes are roughly satisfied (both rchi^2 ~ 1).
        But it fails when the SL data are NOT satisfied at the WL-only
        minimum (which is typical because WL doesn't know about SL):
        then rchi^2_SL >> 1, and lambda is set very small — to "make
        the rchi^2 equal", the formula tells SL to weight itself less.
        That's the wrong direction.  We want SL to pull HARDER when
        it disagrees with WL, not retreat.

        The absolute-chi^2 convention here scales the contribution by
        problem size (number of constraints * typical residual^2)
        rather than per-DOF average residual.  At the WL-only minimum,
        this gives lambda ~ chi^2_WL / chi^2_SL ~ O(1) — SL contributes
        roughly its absolute chi^2 worth of pull on the joint fit.

    Parameters
    ----------
    sources : Source
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Lens model at which lambda is computed.  Convention: this
        should be the post-WL-forward-selection model (where chi^2_WL
        is well-fit but chi^2_SL is generally not yet).
    use_flags : sequence of bool
    lens_type : str

    Returns
    -------
    lambda_sl : float
    """
    # WL contribution
    chi2_wl = calculate_chi_squared(sources, lenses, use_flags, lens_type=lens_type)
    dof_wl = calc_degrees_of_freedom(sources, lenses, use_flags)

    # SL contribution
    has_sl = (
        hasattr(sources, "strong_systems")
        and sources.strong_systems is not None
        and len(sources.strong_systems) > 0
    )

    if not has_sl:
        return 1.0  # no SL data - default

    # Source-plane scatter + flux ratio.
    # use_magnification_correction=False: at the WL-only model the predicted
    # |mu| at SL image positions is unreliable (10–100x), which would compress
    # sigma_beta to ~0.001 arcsec and inflate chi2_SL by 100–10,000x.  Using
    # sigma_beta = sigma_theta directly gives a physically meaningful scale for
    # the lambda ratio.  The full magnification correction is still applied
    # during the actual joint fitting.
    if lens_type == "SIS":
        chi2_scatter = utils.chi2_strong_source_plane_sis(
            lenses, sources.strong_systems, use_magnification_correction=False
        )
        chi2_flux = utils.chi2_flux_sis(lenses, sources.strong_systems)
    elif lens_type == "NFW":
        chi2_scatter = utils.chi2_strong_source_plane_nfw(
            lenses, sources.strong_systems, use_magnification_correction=False
        )
        chi2_flux = utils.chi2_flux_nfw(lenses, sources.strong_systems)
    elif lens_type == "POWER_LAW":
        chi2_scatter = utils.chi2_strong_source_plane_power_law(
            lenses, sources.strong_systems, use_magnification_correction=False
        )
        chi2_flux = utils.chi2_flux_power_law(lenses, sources.strong_systems)
    else:
        return 1.0

    chi2_sl = chi2_scatter + chi2_flux
    dof_sl = calc_strong_dof(sources)

    # Guard against degenerate cases
    if dof_wl <= 0 or dof_sl <= 0 or chi2_sl <= 0 or chi2_wl <= 0:
        return 1.0

    # Geometric-mean lambda: chi2_WL / sqrt(chi2_SL * dof_SL).
    #
    # The pure absolute ratio chi2_WL/chi2_SL gives O(1) only when chi2_SL ~
    # chi2_WL at the WL-only minimum.  For clusters like Abell 2744 the WL
    # model is ~1–2 arcsec off at SL image positions, and sigma_theta = 0.07
    # arcsec (JWST precision) → chi2_SL/chi2_WL ~ 20–80, making lambda too
    # small for SL to influence forward selection.
    #
    # The geometric mean is the midpoint (in log space) between:
    #   chi2_WL / chi2_SL  — absolute ratio (current, too small)
    #   chi2_WL / dof_SL   — DOF-normalised (target at joint minimum, too large)
    # It gives O(1) for rchi2_SL in the range 20–100 without any free
    # parameters and without letting either probe overwhelm the other.
    lambda_raw = chi2_wl / np.sqrt(chi2_sl * dof_sl)

    lambda_max = 100.0
    result = min(lambda_raw, lambda_max)
    cap_note = f"  (capped at {lambda_max:.0f})" if lambda_raw > lambda_max else ""
    print(
        f"Pre-computed lambda_sl: chi^2_WL/sqrt(chi^2_SL*dof_SL) = "
        f"{chi2_wl:.1f}/sqrt({chi2_sl:.1f}*{dof_sl}) = {lambda_raw:.3f}{cap_note}"
    )
    return float(result)


def _compute_lambda_sl_power_law(sources, halos, use_flags, strong_systems=None):
    """
    Compute lambda_sl as a reduced-chi-squared ratio after WL forward
    selection completes.  This is the power-law analog of the
    convention adopted in the NFW pipeline.

    The motivation: the WL and SL chi^2 contributions need to be
    rescaled so that neither dominates the joint objective at the
    converged WL-only solution.  Using
        lambda_sl = (chi2_WL / dof_WL) / (chi2_SL / dof_SL)
    ensures that the per-degree-of-freedom contributions are equal at
    selection time, after which lambda_sl is FROZEN through merging
    and strength optimization to keep the joint objective stationary.

    If no strong-lensing systems are supplied or sigma_n cannot be
    computed, returns 0.0 (effectively WL-only downstream).

    Parameters
    ----------
    sources : Source
    halos : PowerLawHalo
        Selected halos at the WL forward-selection minimum.
    use_flags : sequence of three bool
    strong_systems : iterable of StrongLensingSystem or None

    Returns
    -------
    lambda_sl : float
    """
    if strong_systems is None:
        return 0.0

    # WL contribution
    chi2_wl = chi2_wl_power_law(
        halos,
        sources,
        use_flags=use_flags,
        apply_penalties=False,
    )
    if not np.isfinite(chi2_wl) or chi2_wl <= 0:
        return 0.0

    # SL contribution.  Compute sigma_n from the WL Hessian first so
    # the profile-uncertainty term is properly accounted for.
    # use_magnification_correction=False for the same reason as in
    # compute_lambda_sl: the WL-only model's |mu| at SL positions is
    # unreliable and would inflate chi2_SL by 100–10,000x.
    try:
        sigma_n = posterior_sigma_n(
            halos,
            sources,
            use_flags=use_flags,
        )
    except Exception:
        sigma_n = None

    chi2_sl = utils.chi2_strong_source_plane_power_law(
        halos,
        strong_systems,
        sigma_n=sigma_n,
        alpha_cal=1.0,
        use_magnification_correction=False,
    )

    if not np.isfinite(chi2_sl) or chi2_sl <= 0:
        return 0.0

    dof_sl = calc_strong_dof(sources)
    if dof_sl <= 0:
        return 0.0

    # Geometric-mean lambda: same convention as compute_lambda_sl.
    lambda_raw = chi2_wl / np.sqrt(chi2_sl * dof_sl)
    lambda_max = 100.0
    print(
        f"Pre-computed lambda_sl (power-law): chi^2_WL/sqrt(chi^2_SL*dof_SL) = "
        f"{chi2_wl:.1f}/sqrt({chi2_sl:.1f}*{dof_sl}) = {lambda_raw:.3f}"
        + (f"  (capped at {lambda_max:.0f})" if lambda_raw > lambda_max else "")
    )
    return float(min(lambda_raw, lambda_max))


def calculate_total_chi2(
    sources,
    lenses,
    use_flags,
    lens_type: str = "NFW",
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
    use_magnification_correction_sl: bool = True,
):
    """
    Total chi2 = chi2_WL + lambda_sl * chi2_SL.

    Now supports POWER_LAW in addition to SIS and NFW.  The POWER_LAW
    SL chi2 uses the same components as SIS/NFW:
        chi2_SL = chi2_scatter + chi2_flux

    where chi2_scatter is computed by chi2_strong_source_plane_power_law
    and chi2_flux by chi2_flux_power_law.  Both functions live in
    utils.py and operate directly on PowerLawHalo.

    The relative weight lambda_sl between weak and strong lensing can be
    supplied in three ways (in order of precedence):

        1. Explicitly via the ``lambda_sl`` keyword  — used as-is.
           This is the recommended path: the caller pre-computes
           lambda_sl once at the initial parameter values via
           ``compute_lambda_sl()`` and holds it fixed throughout
           the entire optimisation call so the objective is smooth.

           For POWER_LAW, lambda_sl is computed AFTER forward selection
           rather than before — see the POWER_LAW branch in
           main.fit_lensing_field.

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
    components : dict
        Keys: chi2_wl, chi2_sl, chi2_scatter, chi2_flux,
              dof_wl, dof_sl, lambda_sl
    """
    # ── WL part ──
    chi2_wl = calculate_chi_squared(sources, lenses, use_flags, lens_type=lens_type)
    dof_wl = calc_degrees_of_freedom(sources, lenses, use_flags)

    chi2_scatter = 0.0
    chi2_flux = 0.0
    chi2_sl = 0.0
    dof_sl = 0

    # ── SL part (only if present AND requested) ──
    has_sl = (
        hasattr(sources, "strong_systems")
        and sources.strong_systems is not None
        and len(sources.strong_systems) > 0
    )

    if has_sl and use_strong_lensing:
        # Source-plane scatter (positional constraint)
        if lens_type == "SIS":
            chi2_scatter = utils.chi2_strong_source_plane_sis(lenses, sources.strong_systems)
        elif lens_type == "NFW":
            chi2_scatter = utils.chi2_strong_source_plane_nfw(lenses, sources.strong_systems)
        elif lens_type == "POWER_LAW":
            chi2_scatter = utils.chi2_strong_source_plane_power_law(lenses, sources.strong_systems)
        else:
            raise NotImplementedError(
                f"Strong-lensing chi2 not implemented for lens_type='{lens_type}'."
            )

        # Flux ratios (mass constraint — only for systems with flux data)
        if lens_type == "SIS":
            chi2_flux = utils.chi2_flux_sis(lenses, sources.strong_systems)
        elif lens_type == "NFW":
            chi2_flux = utils.chi2_flux_nfw(lenses, sources.strong_systems)
        elif lens_type == "POWER_LAW":
            chi2_flux = utils.chi2_flux_power_law(lenses, sources.strong_systems)

        chi2_sl = chi2_scatter + chi2_flux
        dof_sl = calc_strong_dof(sources)

    # ── Determine lambda ──
    if lambda_sl is not None:
        _lambda = float(lambda_sl)
    elif use_strong_lensing and dof_sl > 0 and chi2_sl > 0:
        rchi2_wl = chi2_wl / dof_wl if dof_wl > 0 else 1.0
        rchi2_sl = chi2_sl / dof_sl if dof_sl > 0 else 1.0
        _lambda = rchi2_wl / rchi2_sl if rchi2_sl > 0 else 1.0
    else:
        _lambda = 0.0

    chi2_total = float(chi2_wl) + _lambda * float(chi2_sl)

    _dof_wl = int(dof_wl) if np.isfinite(dof_wl) else 0
    _dof_sl = int(dof_sl) if np.isfinite(dof_sl) else 0
    dof_total = _dof_wl + _dof_sl

    components = {
        "chi2_wl": float(chi2_wl),
        "chi2_sl": float(chi2_sl),
        "chi2_scatter": float(chi2_scatter),
        "chi2_flux": float(chi2_flux),
        "dof_wl": _dof_wl,
        "dof_sl": _dof_sl,
        "lambda_sl": float(_lambda),
    }
    return chi2_total, dof_total, components


def chi2_wl_power_law(
    halos, sources, use_flags=(True, True, True), apply_penalties=True, penalty_factor=1.0e6
):
    """
    Weak-lensing chi-squared for a power-law halo model.

    Sums the squared residuals between observed and modeled lensing
    signal components (shear, first flexion, second flexion), each
    weighted by its per-source measurement uncertainty.  This is the
    objective used during per-halo local optimization, forward selection,
    and strength optimization.

        chi2_WL = sum_s [ (e1_obs - e1_mod)^2 / sigs^2
                        + (e2_obs - e2_mod)^2 / sigs^2
                        + (f1_obs - f1_mod)^2 / sigf^2
                        + (f2_obs - f2_mod)^2 / sigf^2
                        + (g1_obs - g1_mod)^2 / sigg^2
                        + (g2_obs - g2_mod)^2 / sigg^2 ]

    Strong-lensing constraints are deliberately excluded; they are
    handled separately by chi2_strong_source_plane_power_law and
    combined globally via lambda_sl in calculate_total_chi2.

    Parameters
    ----------
    halos : PowerLawHalo
        Power-law halo model.  Must have x, y, kappa_star, slope arrays
        plus scalar theta_star and redshift.
    sources : Source
        Observed sources, carrying e1, e2, f1, f2, g1, g2 (signals),
        sigs, sigf, sigg (per-component uncertainties), and per-source
        redshift.
    use_flags : tuple of three bool
        (use_shear, use_flexion, use_g_flexion) — toggle each signal
        family.  Default: all three on, matching ARCH SIS/NFW conventions.
    apply_penalties : bool
        If True, add soft barrier penalties for halos that have drifted
        outside the physical bounds 0 < n < 2 and kappa_star > 0.  These
        are intended as a safety net for unconstrained Nelder-Mead steps;
        when bounded optimizers (e.g. L-BFGS-B) are used, set to False.
    penalty_factor : float
        Strength of the soft barrier penalty.

    Returns
    -------
    chi2 : float
        Total weak-lensing chi-squared.
    """
    use_shear, use_flexion, use_g_flexion = use_flags

    # --- Predicted signals at every source ---
    e1_p, e2_p, f1_p, f2_p, g1_p, g2_p = utils.calculate_lensing_signals_power_law(halos, sources)

    chi2 = 0.0
    if use_shear:
        chi2 += np.sum(((e1_p - sources.e1) ** 2 + (e2_p - sources.e2) ** 2) / sources.sigs**2)
    if use_flexion:
        chi2 += np.sum(((f1_p - sources.f1) ** 2 + (f2_p - sources.f2) ** 2) / sources.sigf**2)
    if use_g_flexion:
        chi2 += np.sum(((g1_p - sources.g1) ** 2 + (g2_p - sources.g2) ** 2) / sources.sigg**2)

    if apply_penalties:
        chi2 += _power_law_bound_penalty(halos, penalty_factor=penalty_factor)

    return float(chi2)


def posterior_sigma_n(
    halos,
    sources,
    use_flags=(True, True, True),
    relative_step=1.0e-3,
    absolute_step_x=0.05,
    absolute_step_kappa=5.0e-4,
    absolute_step_slope=5.0e-3,
    eigval_threshold=1.0e-8,
    return_info=False,
):
    """
    Posterior uncertainty on the slope parameter, sigma_n, per halo.

    Computes the full 4N_halo x 4N_halo Hessian H of chi2_wl_power_law at
    the current parameter values via central finite differences, inverts
    it via eigendecomposition with small-eigenvalue regularization, and
    extracts the marginal posterior variance on each halo's slope.

    The relation used is:

        C = 2 * H^{-1}                  (covariance from chi2 Hessian)
        sigma_n_j = sqrt(C[slope_j, slope_j])

    The factor of 2 comes from chi2 = -2 ln L (up to a constant), so the
    quadratic expansion (1/2) H delta^T delta of chi2 corresponds to a
    Gaussian posterior with covariance 2 H^{-1}.

    Marginal vs conditional.  This function returns the MARGINAL sigma_n,
    which accounts for cross-halo correlations through the full inverse
    Hessian.  This is the correct quantity for the SL profile-uncertainty
    term sigma_beta_prof (Phase 0) and is generally larger than the
    conditional sigma_n that would result from per-halo Hessians.

    Eigenvalue regularization.  Pure central-difference Hessians on
    multi-halo problems can produce small spurious negative eigenvalues
    from FP roundoff in cross-derivative cancellations (the cross-term
    is a difference of four nearly-equal large numbers).  We regularize
    by eigendecomposing H and clipping eigenvalues from below at
    `eigval_threshold * max(|eigval|)`, equivalent to the standard
    pseudoinverse with rcond=eigval_threshold.  Set
    eigval_threshold=0 to disable.

    Convergence assumed.  This function presumes `halos` is at (or very
    near) the chi2_wl_power_law minimum.  Away from the minimum the
    quadratic-approximation interpretation of H breaks down.

    Parameters
    ----------
    halos : PowerLawHalo
        Power-law halo collection at the converged WL minimum.
    sources : Source
        Observed sources, must carry e1, e2, f1, f2, g1, g2, sigs, sigf,
        sigg, redshift.
    use_flags : (bool, bool, bool)
        Toggle (use_shear, use_flexion, use_g_flexion) — must match the
        flags used during the WL fit.
    relative_step : float
        Multiplicative factor on parameter magnitude for the FD step.
        Effective step is max(relative_step * |p|, absolute_step_*).
    absolute_step_x, absolute_step_kappa, absolute_step_slope : float
        Floors on the FD step size for x/y, kappa_star, and slope.
    eigval_threshold : float
        Minimum allowed eigenvalue, as a fraction of the maximum
        eigenvalue magnitude.  Default 1e-8.
    return_info : bool
        If True, also return a diagnostics dict.

    Returns
    -------
    sigma_n : ndarray, shape (N_halo,)
        Marginal posterior uncertainty on each halo's slope.
    info : dict (optional)
        Keys: hessian, covariance, eigvals, eigvals_clipped,
        n_clipped, condition_number, step_sizes, chi2_at_minimum.
    """
    N_h = halos.x.size
    N_p = 4 * N_h

    # --- Pack parameters into a flat vector: [x..., y..., k_star..., slope...] ---
    p0 = np.concatenate([halos.x, halos.y, halos.kappa_star, halos.slope]).astype(float)

    # --- Adaptive step sizes ---
    steps = np.zeros(N_p)
    for j in range(N_h):
        steps[0 * N_h + j] = max(relative_step * abs(halos.x[j]), absolute_step_x)
        steps[1 * N_h + j] = max(relative_step * abs(halos.y[j]), absolute_step_x)
        steps[2 * N_h + j] = max(relative_step * abs(halos.kappa_star[j]), absolute_step_kappa)
        steps[3 * N_h + j] = max(relative_step * abs(halos.slope[j]), absolute_step_slope)

    # --- chi2 evaluator from flat parameter vector ---
    def chi2_from_p(p):
        h = halos.copy()
        h.x = p[0 * N_h : 1 * N_h].copy()
        h.y = p[1 * N_h : 2 * N_h].copy()
        h.kappa_star = np.abs(p[2 * N_h : 3 * N_h]).copy()
        h.slope = p[3 * N_h : 4 * N_h].copy()
        return chi2_wl_power_law(h, sources, use_flags=use_flags, apply_penalties=False)

    chi2_0 = chi2_from_p(p0)

    # --- Hessian via central finite differences ---
    H = np.zeros((N_p, N_p))

    # Diagonal: H_ii ≈ (f(p+h e_i) - 2 f(p) + f(p-h e_i)) / h^2
    for i in range(N_p):
        ei = np.zeros(N_p)
        ei[i] = 1.0
        f_p = chi2_from_p(p0 + steps[i] * ei)
        f_m = chi2_from_p(p0 - steps[i] * ei)
        H[i, i] = (f_p - 2.0 * chi2_0 + f_m) / steps[i] ** 2

    # Off-diagonal: H_ij ≈ (f++ - f+- - f-+ + f--) / (4 h_i h_j)
    for i in range(N_p):
        for j in range(i + 1, N_p):
            ei = np.zeros(N_p)
            ei[i] = 1.0
            ej = np.zeros(N_p)
            ej[j] = 1.0
            f_pp = chi2_from_p(p0 + steps[i] * ei + steps[j] * ej)
            f_pm = chi2_from_p(p0 + steps[i] * ei - steps[j] * ej)
            f_mp = chi2_from_p(p0 - steps[i] * ei + steps[j] * ej)
            f_mm = chi2_from_p(p0 - steps[i] * ei - steps[j] * ej)
            H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            H[j, i] = H[i, j]

    # --- Eigendecomposition-based pseudoinverse with clipping ---
    eigvals, eigvecs = np.linalg.eigh(H)
    max_abs = np.max(np.abs(eigvals))
    threshold = eigval_threshold * max_abs

    eigvals_clipped = np.where(eigvals < threshold, threshold, eigvals)
    n_clipped = int(np.sum(eigvals < threshold))

    # H^{-1} = V diag(1/lambda_clipped) V^T;  cov = 2 H^{-1}
    cov = 2.0 * (eigvecs @ np.diag(1.0 / eigvals_clipped) @ eigvecs.T)

    cond_num = (
        eigvals_clipped.max() / eigvals_clipped.min() if eigvals_clipped.min() > 0 else np.inf
    )

    # --- Extract per-halo sigma_n from slope diagonal block ---
    sigma_n = np.zeros(N_h)
    for j in range(N_h):
        slope_idx = 3 * N_h + j
        var = cov[slope_idx, slope_idx]
        sigma_n[j] = np.sqrt(var) if (np.isfinite(var) and var > 0) else np.nan

    if not return_info:
        return sigma_n

    info = {
        "hessian": H,
        "covariance": cov,
        "eigvals": eigvals,
        "eigvals_clipped": eigvals_clipped,
        "n_clipped": n_clipped,
        "condition_number": cond_num,
        "step_sizes": steps,
        "chi2_at_minimum": chi2_0,
        "n_halos": N_h,
        "n_params": N_p,
    }
    return sigma_n, info


def _power_law_bound_penalty(halos, penalty_factor=1.0e6):
    """
    Soft barrier penalty for halos that have drifted outside the
    physical parameter bounds.

        slope:       0 < n < 2
        kappa_star:  kappa_star > 0

    Returns
    -------
    penalty : float
        Sum of squared boundary violations, scaled by penalty_factor.
        Zero if all halos are inside the physical region.
    """
    n = np.atleast_1d(halos.slope)
    k = np.atleast_1d(halos.kappa_star)

    # Slope: penalize n <= 0 and n >= 2 quadratically in distance from boundary
    pen_n_low = np.where(n <= 0.0, (0.0 - n) ** 2, 0.0).sum()
    pen_n_hi = np.where(n >= 2.0, (n - 2.0) ** 2, 0.0).sum()
    pen_k = np.where(k <= 0.0, (0.0 - k) ** 2, 0.0).sum()

    return penalty_factor * float(pen_n_low + pen_n_hi + pen_k)


def calc_dof_wl_power_law(sources, halos, use_flags=(True, True, True)):
    """
    Degrees of freedom for the weak-lensing chi-squared with power-law halos.

    Each signal component contributes 2 numbers per source (the two
    polarization components), times the number of enabled signal families.
    Each halo contributes 4 free parameters: (x, y, kappa_star, slope).
    Note that theta_star is fixed by global convention and not counted.

    Parameters
    ----------
    sources : Source
        Source object.  Only sources.x is used for counting.
    halos : PowerLawHalo
        Halo model.  Only halos.x is used for counting.
    use_flags : tuple of three bool
        (use_shear, use_flexion, use_g_flexion).

    Returns
    -------
    dof : int or float
        Degrees of freedom.  np.inf if dof <= 0.
    """
    num_signals = int(sum(use_flags))
    num_source_params = 2 * num_signals * len(sources.x)
    num_halo_params = 4 * len(halos.x)
    dof = num_source_params - num_halo_params
    if dof <= 0:
        return np.inf
    return int(dof)
