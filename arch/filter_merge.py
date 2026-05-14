"""Lens filtering (geometric + physical) and close-lens merging."""

import numpy as np


def filter_lens_positions(
    sources,
    lenses,
    xmax,
    threshold_distance=0.5,
    lens_type="SIS",
    slope_boundary_tol=1.0e-2,
    kappa_boundary_tol=1.0e-2,
    slope_lo=0.4,
    slope_hi=1.7,
    kappa_star_lo=1.0e-6,
    kappa_star_hi=10.0,
    kappa_floor_n_sigma=0.0,
    kappa_test_radius=10.0,
):
    """
    Filters out invalid lenses based on geometric and physical criteria.

    Common filters (all lens types):
      - Lenses within `threshold_distance` of any source (avoids the
        formal divergence at theta = 0).
      - Lenses outside `xmax * 1.5` from the origin (drifted off-field).

    Lens-type-specific filters:
      - SIS:        Einstein radius >= 1e-3 arcsec.
      - NFW:        Mass in [1e10, 1e16] M_sun.
      - POWER_LAW:  Slope `n` AND kappa_star NOT pinned at their
                    optimizer bounds, plus an optional flexion-noise
                    floor on kappa_star.

    The bound-pinning checks for POWER_LAW catch optimizer failures
    where Nelder-Mead landed at the bound — these halos' parameters
    are not physically meaningful.  Both slope AND kappa_star are
    checked because the (high-kappa_star, low-n) corner of parameter
    space corresponds to a near-uniform mass sheet that the optimizer
    sometimes drifts into when chi^2 gradients are weak (observed on
    El Gordo with n=0.051, kappa_*=9.73 — a uniform kappa~3 sheet
    rather than localized halos).

    The kappa_star noise-floor filter is OFF BY DEFAULT
    (kappa_floor_n_sigma=0).  In real data, the source-distribution
    sigma_f estimate is dominated by intrinsic flexion variance rather
    than measurement noise, which makes the analytical noise floor too
    aggressive.  forward_lens_selection performs a more principled
    signal-vs-noise discrimination by accepting only candidates that
    lower the global chi-squared.

    Parameters
    ----------
    sources : Source
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
    xmax : float
        Field half-width (arcsec).  Lenses outside xmax*1.5 are dropped.
    threshold_distance : float
        Minimum allowed distance (arcsec) between any lens and source.
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    slope_boundary_tol : float
        POWER_LAW only.  Halos with slope within `slope_boundary_tol`
        of (slope_lo, slope_hi) are dropped.  Default 1e-2 — wider
        than the previous 1e-3 to catch optimizer outputs that sit
        just inside the bound (e.g., n = 0.051 with bound at 0.05).
    kappa_boundary_tol : float
        POWER_LAW only.  Same logic for kappa_star bounds (defaults
        to 1% on each end).
    slope_lo, slope_hi : float
        POWER_LAW slope bounds.  Defaults (0.4, 1.7) match the
        position-optimizer bounds.  Outside this range the profile is
        either a near-uniform sheet (n < 0.4) or near-singular
        (n > 1.7), neither of which represents a real cluster halo.
    kappa_star_lo, kappa_star_hi : float
        POWER_LAW kappa_star bounds.  Defaults (1e-6, 10) match the
        position-optimizer bounds.
    kappa_floor_n_sigma : float
        POWER_LAW only.  Multiplicative factor on median(sigf).  Default
        0.0 (filter disabled).
    kappa_test_radius : float
        POWER_LAW only.  Radius for the noise-floor test.  Default 10.

    Returns
    -------
    Lens collection.

    Raises
    ------
    ValueError
        If no lenses remain after filtering.
    """
    # Common geometric filters (all lens types)
    distances = np.sqrt((lenses.x[:, None] - sources.x) ** 2 + (lenses.y[:, None] - sources.y) ** 2)
    too_close = np.any(distances < threshold_distance, axis=1)
    too_far = np.sqrt(lenses.x**2 + lenses.y**2) > xmax * 1.5

    if lens_type == "SIS":
        invalid_te = lenses.te < 1e-3
        valid_indices = ~(too_close | too_far | invalid_te)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.te = lenses.te[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

    elif lens_type == "NFW":
        invalid_mass = (lenses.mass < 1e10) | (lenses.mass > 1e16)
        valid_indices = ~(too_close | too_far | invalid_mass)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.mass = lenses.mass[valid_indices]
        lenses.concentration = lenses.concentration[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

    elif lens_type == "POWER_LAW":
        # (1) Slope pinned at boundary?  Tolerance is absolute (n
        # is a unitless dimensionless slope, so 1e-2 is appropriate).
        slope_pinned = (lenses.slope < slope_lo + slope_boundary_tol) | (
            lenses.slope > slope_hi - slope_boundary_tol
        )

        # (2) kappa_star pinned at boundary?  kappa_star spans many
        # decades, so check fractionally.
        kappa_pinned = (lenses.kappa_star < kappa_star_lo * (1.0 + kappa_boundary_tol)) | (
            lenses.kappa_star > kappa_star_hi * (1.0 - kappa_boundary_tol)
        )

        # (3) kappa_star below flexion-noise floor (only if enabled)
        if kappa_floor_n_sigma > 0:
            median_sigf = float(np.median(np.atleast_1d(sources.sigf)))
            n_arr = np.maximum(lenses.slope, slope_lo)
            kappa_star_min = (
                kappa_floor_n_sigma
                * median_sigf
                * kappa_test_radius ** (n_arr + 1.0)
                / (n_arr * lenses.theta_star**n_arr)
            )
            kappa_below_floor = lenses.kappa_star < kappa_star_min
        else:
            kappa_below_floor = np.zeros(len(lenses.x), dtype=bool)
            median_sigf = float(np.median(np.atleast_1d(sources.sigf)))

        # Diagnostic counts so the failure message can identify which
        # filter dominated.
        n_total = len(lenses.x)
        n_too_close = int(too_close.sum())
        n_too_far = int(too_far.sum())
        n_slope_pinned = int(slope_pinned.sum())
        n_kappa_pinned = int(kappa_pinned.sum())
        n_kappa_floor = int(kappa_below_floor.sum())

        valid_indices = ~(too_close | too_far | slope_pinned | kappa_pinned | kappa_below_floor)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.kappa_star = lenses.kappa_star[valid_indices]
        lenses.slope = lenses.slope[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

        if len(lenses.x) == 0:
            raise ValueError(
                f"No valid lenses remain after filtering (POWER_LAW).\n"
                f"  Started with: {n_total} candidates\n"
                f"  Too close to a source (<{threshold_distance} arcsec): "
                f"{n_too_close}\n"
                f"  Drifted outside {xmax * 1.5} arcsec: {n_too_far}\n"
                f"  Slope pinned at ({slope_lo}, {slope_hi}) boundary: "
                f"{n_slope_pinned}\n"
                f"  kappa_star pinned at ({kappa_star_lo}, {kappa_star_hi}) "
                f"boundary: {n_kappa_pinned}\n"
                f"  kappa_star below noise floor: {n_kappa_floor} "
                f"(filter {'on' if kappa_floor_n_sigma > 0 else 'off'})\n"
                f"  median(sigf) = {median_sigf:.3e}\n"
                f"If slope or kappa pinning dominate, the position "
                f"optimizer is hitting bounds — consider widening or "
                f"narrowing the bounds via slope_lo/slope_hi/"
                f"kappa_star_lo/kappa_star_hi."
            )

    else:
        raise ValueError("Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'.")

    if len(lenses.x) == 0:
        raise ValueError("No valid lenses remain after filtering.")

    return lenses


def merge_close_lenses(lenses, merger_threshold=5, lens_type="SIS"):
    """
    Merges lenses that are closer than a specified threshold.

    For all lens types, the merge updates the surviving lens's POSITION
    via a strength-weighted average and removes the merged-in lens.
    Strength parameters (te / mass / kappa_star+slope) are NOT updated
    by this function — they remain at the surviving lens's pre-merge
    values and are refit by strength optimization downstream.  This
    matches the existing SIS/NFW pattern where `strength[i] = total/2`
    is bookkeeping for the inner loop only and `lenses.te[i]` /
    `lenses.mass[i]` are unchanged by the merge.

    For POWER_LAW the design is the same.  The choice is deliberate:
    two power-law profiles with different slopes do NOT sum to a
    power-law profile,
        kappa_star_i (theta/theta_star)^(-n_i)
        + kappa_star_j (theta/theta_star)^(-n_j)
    is a power law only when n_i = n_j, so any analytic combination
    of (kappa_star, n) would be unprincipled.  Leaving the surviving
    halo's (kappa_star, n) as the seed and letting strength
    optimization (step 19) refit them is the correct architectural
    choice.

    Parameters
    ----------
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Lens collection to merge.
    merger_threshold : float
        Distance threshold (arcsec) for merging.  Default 5.
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Merged lens collection.
    """
    # Determine lens strength based on type (used for position weighting)
    if lens_type == "SIS":
        strength = np.abs(lenses.te)
    elif lens_type == "NFW":
        strength = np.abs(lenses.mass)
    elif lens_type == "POWER_LAW":
        strength = np.abs(lenses.kappa_star)
    else:
        raise ValueError("Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'.")

    def merge_lenses(i, j):
        """
        Merges lens at index j into lens at index i, updating position
        only.  Strength parameters are unchanged for the surviving
        lens — strength optimization refits them later.
        """
        weight_i, weight_j = strength[i], strength[j]
        total_weight = weight_i + weight_j

        # Strength-weighted position (all types)
        lenses.x[i] = (lenses.x[i] * weight_i + lenses.x[j] * weight_j) / total_weight
        lenses.y[i] = (lenses.y[i] * weight_i + lenses.y[j] * weight_j) / total_weight

        strength[i] = total_weight / 2  # Update strength of lens i
        lenses.remove([j])  # Remove lens at index j

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
    if lens_type == "NFW":
        lenses.calculate_concentration()
    # POWER_LAW: no analog needed — theta_star is a fixed convention,
    # not a derived property.  (kappa_star, slope) are passed to
    # strength optimization as-is.

    return lenses
