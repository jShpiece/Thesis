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


def generate_initial_guess(sources, lens_type='SIS', z_l=0.5, z_s=0.8,
                           theta_star=30.0,
                           use_peak_finding=False,
                           peak_finding_kwargs=None):
    """
    Generates initial guesses for lens positions based on source
    ellipticity and flexion signals.

    For ``lens_type='SIS'`` and ``lens_type='NFW'``, this is the original
    one-candidate-per-source seeding logic.  For ``lens_type='POWER_LAW'``,
    the routine inverts the two ratio invariants
        |G|/|F|     = (2 + n) / (2 - n)         (slope invariant)
        |gamma|/|F| = theta / (2 - n)           (distance invariant)
    via cast_votes_power_law to produce per-source candidates with
    estimated (x, y, kappa_star, n).  An optional `use_peak_finding`
    switch routes through seed_from_votes to aggregate the per-source
    votes into a smaller, higher-confidence candidate pool — useful for
    forward selection on dense fields where iterating over thousands of
    per-source seeds is wasteful.

    Parameters
    ----------
    sources : Source
        Source object with positions and lensing signals.
    lens_type : str
        One of 'SIS', 'NFW', or 'POWER_LAW'.
    z_l : float
        Redshift of the lens (used by NFW and POWER_LAW for distances).
    z_s : float
        Redshift of the source (used by NFW for the mass minimization).
        For POWER_LAW the per-source redshifts in `sources.redshift` are
        used directly via the lensing-efficiency correction.
    theta_star : float
        Pivot radius (arcsec) for the POWER_LAW profile.  Ignored for
        SIS and NFW.  Default 30 arcsec — a reasonable cluster-scale
        choice; see Phase 0 derivations document.
    use_peak_finding : bool
        Power-law mode only.  If True, run seed_from_votes on the
        per-source votes to extract a smaller pool of vote-map peaks.
        If False (default), return one candidate per valid source
        (matches SIS/NFW seeding granularity).
    peak_finding_kwargs : dict or None
        Power-law mode only, only used when use_peak_finding=True.
        Forwarded to seed_from_votes; useful keys include
        ``n_peaks``, ``smoothing_sigma``, ``peak_threshold_rel``,
        ``aggregate_radius``, ``peak_min_distance``, ``n_pix``.

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Candidate lens collection with seeded parameters.

    Raises
    ------
    ValueError
        If `lens_type` is not one of 'SIS', 'NFW', 'POWER_LAW'.
    """
    # --------------------------------------------------------------
    # Common inversion (used by SIS and NFW; POWER_LAW has its own)
    # --------------------------------------------------------------
    if lens_type in ('SIS', 'NFW'):
        phi = np.arctan2(sources.f2, sources.f1)
        gamma = np.hypot(sources.e1, sources.e2)
        flexion = np.hypot(sources.f1, sources.f2)

    # --------------------------------------------------------------
    # SIS path  (unchanged from original implementation)
    # --------------------------------------------------------------
    if lens_type == 'SIS':
        r = gamma / flexion
        te = 2 * gamma * r
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        return halo_obj.SIS_Lens(xl, yl, te, np.empty_like(sources.x))

    # --------------------------------------------------------------
    # NFW path  (unchanged from original implementation)
    # --------------------------------------------------------------
    if lens_type == 'NFW':
        flexion = np.where(flexion == 0, 1e-10, flexion)
        r = 2.0 * gamma / flexion
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        masses = np.zeros_like(sources.x)

        for i in range(len(sources.x)):
            def mass_objective(mass):
                mass = np.abs(mass)
                lens = halo_obj.NFW_Lens(
                    x=xl[i], y=yl[i], z=0.0,
                    concentration=0.0, mass=mass,
                    redshift=z_l, chi2=0.0,
                )
                lens.calculate_concentration()
                source = source_obj.Source(
                    x=sources.x[i], y=sources.y[i],
                    e1=0.0, e2=0.0,
                    f1=0.0, f2=0.0,
                    g1=0.0, g2=0.0,
                    sigs=1.0, sigf=1.0, sigg=1.0,
                    redshift=sources.redshift[i],
                )
                _, _, _, f1_model, f2_model, _, _ = utils.calculate_lensing_signals_nfw(
                    lens, source)
                return np.sqrt((f1_model - sources.f1[i]) ** 2
                               + (f2_model - sources.f2[i]) ** 2)

            result = opt.minimize_scalar(
                mass_objective,
                bounds=(1e10, 1e16), method='bounded',
                options={'xatol': 1e-6},
            )
            masses[i] = result.x

        lenses = halo_obj.NFW_Lens(
            x=xl, y=yl,
            z=np.zeros_like(xl),
            concentration=np.zeros_like(xl),
            mass=masses, redshift=z_l,
            chi2=np.zeros_like(xl),
        )
        lenses.calculate_concentration()
        return lenses

    # --------------------------------------------------------------
    # POWER_LAW path  (new — dispatches to cast_votes_power_law)
    # --------------------------------------------------------------
    if lens_type == 'POWER_LAW':
        votes = cast_votes_power_law(
            sources,
            theta_star=theta_star,
            redshift=z_l,
        )

        if use_peak_finding:
            # Aggregate per-source votes into a small set of peaks.
            kwargs = peak_finding_kwargs or {}
            seed_halos = seed_from_votes(
                votes, sources,
                theta_star=theta_star,
                redshift=z_l,
                **kwargs,
            )
            return seed_halos

        # Default: one candidate per valid source.  Sources that didn't
        # produce a usable vote (low SNR, foreground, etc.) are dropped.
        valid = votes["valid"]
        if not np.any(valid):
            # No usable votes — return an empty halo collection rather
            # than raising, so the pipeline can decide how to handle it.
            return halo_obj.PowerLawHalo(
                x=np.array([]), y=np.array([]),
                kappa_star=np.array([]), slope=np.array([]),
                theta_star=theta_star, redshift=z_l,
                chi2=np.array([]),
            )

        return halo_obj.PowerLawHalo(
            x=votes["x_vote"][valid],
            y=votes["y_vote"][valid],
            kappa_star=votes["kappa_star_est"][valid],
            slope=votes["n_est"][valid],
            theta_star=theta_star,
            redshift=z_l,
            chi2=np.zeros(int(valid.sum())),
        )

    raise ValueError("Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'.")

def optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type='SIS',
                            use_strong_lensing: bool = False,
                            lambda_sl: float = None,
                            local_radius: float = 20.0):
    """
    Optimizes lens positions via local minimization.
    Currently only minimizes relative to sources within a certain
    distance of the lens.

    Optimizer choice:
      - SIS:        L-BFGS-B (3 parameters via chi2wrapper).
      - NFW:        L-BFGS-B (3 parameters: x, y, log10 mass).
      - POWER_LAW:  Nelder-Mead (4 parameters: x, y, log10 kappa_star, n).
    L-BFGS-B is unreliable for the power-law objective because the
    weak-lensing chi^2 spans many orders of magnitude over small
    parameter perturbations (|F|^2 ~ theta^(-2-2n) has very steep
    derivatives near small theta), and finite-difference gradients
    systematically drive parameters to their bounds.  Nelder-Mead is
    gradient-free and converges reliably; bounds are honored via
    SciPy's bounded simplex (>=1.7).

    Strong-lensing constraints are deliberately excluded from per-lens
    optimization for all lens types: SL is a global observable of the
    full mass distribution and conflating it with per-halo objectives
    is a category error.  This matches the architectural decision
    already in force for the NFW pipeline.

    Parameters
    ----------
    sources : Source
        Source object containing source positions and lensing signals.
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Initial lens object with estimated positions and parameters.
    xmax : float
        Sets the half-width of position bounds for the optimizer
        (used by NFW and POWER_LAW; ignored by SIS).
    use_flags : list
        Flags indicating which signal families (shear / flexion /
        g-flexion) to use in optimization.
    lens_type : str
        One of 'SIS', 'NFW', 'POWER_LAW'.
    use_strong_lensing : bool
        Whether to include strong lensing in the objective (SIS only;
        the SL handling for NFW and POWER_LAW is delegated to forward
        selection and strength optimization, where SL constraints
        belong).
    lambda_sl : float or None
        Pre-computed SL weight (SIS only).
    local_radius : float
        Radius (arcsec) within which sources are used for per-lens
        optimization.  NFW and POWER_LAW only.  Default 20 arcsec
        (matches existing NFW behavior).

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Lenses with optimized parameters.
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
            # Initial guess: [x, y, log10(mass)] — optimize log-space for mass
            initial_guess = [lenses.x[i], lenses.y[i], np.log10(lenses.mass[i])]

            # Bounds for x, y, and log10(mass)
            bounds = [
                (lenses.x[i] - xmax * 2, lenses.x[i] + xmax * 2),
                (lenses.y[i] - xmax * 2, lenses.y[i] + xmax * 2),
                (10, 17),  # Mass bounds in log10(M_sun)
            ]

            # Minimize relative to sources within local_radius
            filtered_sources = sources.copy()
            distance = np.hypot(lenses.x[i] - sources.x, lenses.y[i] - sources.y)
            filtered_sources.remove(np.where(distance > local_radius)[0])

            def objective_function(params):
                xi, yi, log_mass = params
                mass = 10 ** log_mass
                lens = halo_obj.NFW_Lens(
                    x=xi, y=yi,
                    z=lenses.z[i],
                    concentration=lenses.concentration[i],
                    mass=mass, redshift=lenses.redshift,
                    chi2=0.0,
                )
                lens.calculate_concentration()
                return metric.calculate_chi_squared(
                    filtered_sources, lens, use_flags, lens_type='NFW',
                )

            result = minimize(
                objective_function, initial_guess,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': num_iterations, 'ftol': 1e-6},
            )
            lenses.x[i] = result.x[0]
            lenses.y[i] = result.x[1]
            lenses.mass[i] = 10 ** result.x[2]
            lenses.calculate_concentration()

    elif lens_type == 'POWER_LAW':
        # Optimizer parameter bounds.  Defined ONCE here (instead of
        # rebuilt inside the loop) so the seed-clipping below references
        # the same numbers — preventing "initial guess out of bounds"
        # warnings from cast_votes producing extreme kappa_star
        # estimates from low-SNR sources.
        LOG10_K_LO, LOG10_K_HI = -6.0, 1.0     # log10(kappa_star)
        N_LO, N_HI = 0.05, 1.95                # slope

        for i in range(len(lenses.x)):
            # Initial guess: [x, y, log10(kappa_star), n] — log-space for k*
            x0 = float(lenses.x[i])
            y0 = float(lenses.y[i])

            # Clip kappa_star and slope into their optimizer bounds
            # BEFORE taking log10 / using as initial guess.
            # cast_votes_power_law returns per-source estimates that
            # invert noisy |F| / |G| / |gamma| ratios; on real data,
            # this routinely produces values outside the optimizer's
            # parameter bounds.  The optimizer then warns "Initial
            # guess is not within the specified bounds" and projects
            # to the boundary, leaving the simplex unable to move
            # upward.  Clipping here avoids that pathology.
            log_k0 = float(np.clip(np.log10(max(lenses.kappa_star[i], 1e-30)),
                                   LOG10_K_LO, LOG10_K_HI))
            n0 = float(np.clip(lenses.slope[i], N_LO, N_HI))
            initial_guess = [x0, y0, log_k0, n0]

            bounds = [
                (x0 - xmax * 2, x0 + xmax * 2),
                (y0 - xmax * 2, y0 + xmax * 2),
                (LOG10_K_LO, LOG10_K_HI),
                (N_LO, N_HI),
            ]

            # Restrict to sources within local_radius of the candidate
            filtered_sources = sources.copy()
            distance = np.hypot(x0 - sources.x, y0 - sources.y)
            filtered_sources.remove(np.where(distance > local_radius)[0])

            # If too few sources fall inside the local radius, optimization
            # will be ill-posed.  Skip this candidate and leave it unchanged
            # so filter_lens_positions can drop it later.
            if filtered_sources.x.size < 4:  # 4 free params, need >=4 obs
                continue

            theta_star_lens = lenses.theta_star
            redshift_lens = lenses.redshift

            def objective_function(params):
                return _chi2_wrapper_power_law(
                    params, filtered_sources, use_flags,
                    theta_star=theta_star_lens,
                    redshift=redshift_lens,
                )

            # Use Nelder-Mead rather than L-BFGS-B.  The chi^2 surface
            # for the power-law model spans many orders of magnitude
            # over small parameter changes (kappa^2 ~ |F|^2 ~ theta^(-2-2n)
            # has very steep derivatives near small theta), and L-BFGS-B's
            # finite-difference gradients fail to find the minimum,
            # systematically driving parameters to their bounds.
            # Nelder-Mead is gradient-free and converges reliably; bounds
            # are honored via SciPy's bounded simplex (>=1.7).
            #
            # We provide an explicit initial simplex with per-parameter
            # step sizes calibrated to the WL chi^2 sensitivity (positions
            # in arcsec, log10(kappa_star) in dex, slope in n).  Default
            # NM simplex sizes are too small and let the optimizer get
            # stuck near the seed when the initial chi^2 is far from
            # the minimum.
            x0_arr = np.array(initial_guess, dtype=float)
            simplex_steps = np.array([2.0, 2.0, 0.3, 0.2])  # arcsec, arcsec, dex, slope
            initial_simplex = np.zeros((5, 4))
            initial_simplex[0] = x0_arr
            for k in range(4):
                vertex = x0_arr.copy()
                vertex[k] += simplex_steps[k]
                # Honor bounds in the initial simplex
                lo, hi = bounds[k]
                if vertex[k] > hi:
                    vertex[k] = x0_arr[k] - simplex_steps[k]
                    if vertex[k] < lo:
                        vertex[k] = 0.5 * (lo + hi)
                initial_simplex[k + 1] = vertex

            result = minimize(
                objective_function, initial_guess,
                method='Nelder-Mead',
                bounds=bounds,
                options={
                    'maxiter': int(num_iterations),
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True,
                    'initial_simplex': initial_simplex,
                },
            )

            # Update halo parameters from result
            lenses.x[i] = result.x[0]
            lenses.y[i] = result.x[1]
            lenses.kappa_star[i] = 10 ** result.x[2]
            lenses.slope[i] = result.x[3]

    else:
        raise ValueError(
            "Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'."
        )

    return lenses

def _chi2_wrapper_power_law(params, sources, use_flags,
                            theta_star, redshift):
    """
    Parameter-vector wrapper around chi2_wl_power_law for the
    Nelder-Mead optimizer used in optimize_lens_positions.

    Unpacks a flat 4-vector (x, y, log10(kappa_star), slope) into a
    single-halo PowerLawHalo and evaluates the WL chi-squared.

    Parametrizing kappa_star in log space lets the simplex make
    sensible step proportions across the multi-decade range of
    physical kappa_star values; the bounds applied externally
    ((-6, 1) on log10(kappa_star)) keep the optimizer in a sensible
    region without barrier penalties.

    Parameters
    ----------
    params : sequence of 4 floats
        (x, y, log10(kappa_star), slope).
    sources : Source
        Sources within the local optimization window.
    use_flags : tuple of three bool
        (use_shear, use_flexion, use_g_flexion).
    theta_star : float
        Pivot radius (arcsec).
    redshift : float
        Cluster redshift.

    Returns
    -------
    chi2 : float
        Weak-lensing chi-squared.  Strong-lensing constraints are
        deliberately excluded at this stage.
    """
    x, y, log_k, n = params
    halo = halo_obj.PowerLawHalo(
        x=np.array([x]),
        y=np.array([y]),
        kappa_star=np.array([10 ** log_k]),
        slope=np.array([n]),
        theta_star=theta_star,
        redshift=redshift,
        chi2=np.array([0.0]),
    )
    # apply_penalties=False: the L-BFGS-B bounds enforce 0.05 <= n <= 1.95
    # and kappa_star > 1e-6, so soft penalties are unnecessary here.
    return metric.chi2_wl_power_law(
        halo, sources, use_flags=use_flags, apply_penalties=False,
    )

def filter_lens_positions(sources, lenses, xmax,
                          threshold_distance=0.5,
                          lens_type='SIS',
                          slope_boundary_tol=1.0e-3,
                          kappa_floor_n_sigma=0.0,
                          kappa_test_radius=10.0):
    """
    Filters out invalid lenses based on geometric and physical criteria.

    Common filters (all lens types):
      - Lenses within `threshold_distance` of any source (avoids the
        formal divergence at theta = 0).
      - Lenses outside `xmax * 1.5` from the origin (drifted off-field).

    Lens-type-specific filters:
      - SIS:        Einstein radius >= 1e-3 arcsec.
      - NFW:        Mass in [1e10, 1e16] M_sun.
      - POWER_LAW:  Slope `n` not pinned at the (0.05, 1.95) boundary,
                    and (optionally) `kappa_star` above a flexion-noise
                    floor.

    The boundary-pinning check for POWER_LAW catches optimizer failures
    where Nelder-Mead landed at the bound but couldn't find an interior
    minimum — these halos' fitted parameters are not physically
    meaningful and should be dropped before forward selection.

    The kappa_star noise-floor filter is OFF BY DEFAULT
    (kappa_floor_n_sigma=0).  In real data, the source-distribution
    sigma_f estimate is dominated by intrinsic flexion variance rather
    than measurement noise, which makes the analytical noise floor too
    aggressive (rejects everything).  forward_lens_selection performs
    a more principled signal-vs-noise discrimination by accepting only
    candidates that lower the global chi-squared, so the analytical
    floor here is redundant for production data.

    To enable the noise floor (e.g., for synthetic-data tests where
    sigma_f is well-known), set kappa_floor_n_sigma > 0.  The floor is
    computed as:

        kappa_star_min = (kappa_floor_n_sigma * median(sigf))
                         * theta_test^(n+1) / (n * theta_star^n)

    Parameters
    ----------
    sources : Source
        Source object containing source positions and per-source
        signal uncertainties (used by POWER_LAW for the noise floor).
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Lens object containing positions and parameters.
    xmax : float
        Field half-width (arcsec).  Lenses outside xmax*1.5 are dropped.
    threshold_distance : float
        Minimum allowed distance (arcsec) between any lens and source.
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    slope_boundary_tol : float
        POWER_LAW only.  A halo with slope within `slope_boundary_tol`
        of (0.05, 1.95) is treated as boundary-pinned and dropped.
        Default 1e-3.
    kappa_floor_n_sigma : float
        POWER_LAW only.  Multiplicative factor on the median per-source
        flexion noise that defines the kappa_star noise floor.  Default
        0.0 (filter disabled — forward_lens_selection handles noise
        discrimination).  Set to ~0.3-1.0 to enable on synthetic data
        with a well-known sigma_f.
    kappa_test_radius : float
        POWER_LAW only.  Radius at which the kappa_star noise floor is
        evaluated (arcsec).  Default 10 arcsec.  Only used when
        kappa_floor_n_sigma > 0.

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Filtered lens collection.

    Raises
    ------
    ValueError
        If no lenses remain after filtering, or `lens_type` is invalid.
    """
    # Common geometric filters (all lens types)
    distances = np.sqrt(
        (lenses.x[:, None] - sources.x) ** 2
        + (lenses.y[:, None] - sources.y) ** 2
    )
    too_close = np.any(distances < threshold_distance, axis=1)
    too_far = np.sqrt(lenses.x ** 2 + lenses.y ** 2) > xmax * 1.5

    if lens_type == 'SIS':
        invalid_te = lenses.te < 1e-3
        valid_indices = ~(too_close | too_far | invalid_te)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.te = lenses.te[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

    elif lens_type == 'NFW':
        invalid_mass = (lenses.mass < 1e10) | (lenses.mass > 1e16)
        valid_indices = ~(too_close | too_far | invalid_mass)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.mass = lenses.mass[valid_indices]
        lenses.concentration = lenses.concentration[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

    elif lens_type == 'POWER_LAW':
        # (1) Slope pinned at boundary?
        # This step catches optimizer failures where the simplex landed at the slope
        # boundary but couldn't find an interior minimum.  These halos' parameters are not
        # physically meaningful and should be dropped before forward selection.
        slope_pinned = (
            (lenses.slope < 0.05 + slope_boundary_tol)
            | (lenses.slope > 1.95 - slope_boundary_tol)
        )

        # (2) kappa_star below flexion-noise floor (only if enabled)
        # The floor is based on the analytical inversion of the flexion SNR at a test radius,
        if kappa_floor_n_sigma > 0:
            median_sigf = float(np.median(np.atleast_1d(sources.sigf)))
            n_arr = np.maximum(lenses.slope, 0.05)
            kappa_star_min = (
                kappa_floor_n_sigma * median_sigf
                * kappa_test_radius ** (n_arr + 1.0)
                / (n_arr * lenses.theta_star ** n_arr)
            )
            kappa_below_floor = lenses.kappa_star < kappa_star_min
        else:
            kappa_below_floor = np.zeros(len(lenses.x), dtype=bool)
            median_sigf = float(np.median(np.atleast_1d(sources.sigf)))

        # Diagnostic counts
        n_total = len(lenses.x)
        n_too_close = int(too_close.sum())
        n_too_far = int(too_far.sum())
        n_slope_pinned = int(slope_pinned.sum())
        n_kappa_floor = int(kappa_below_floor.sum())

        valid_indices = ~(too_close | too_far | slope_pinned | kappa_below_floor)
        lenses.x = lenses.x[valid_indices]
        lenses.y = lenses.y[valid_indices]
        lenses.kappa_star = lenses.kappa_star[valid_indices]
        lenses.slope = lenses.slope[valid_indices]
        lenses.chi2 = lenses.chi2[valid_indices]

        if len(lenses.x) == 0:
            raise ValueError(
                f"No valid lenses remain after filtering (POWER_LAW).\n"
                f"  Started with: {n_total} candidates\n"
                f"  Too close to a source (<{threshold_distance} arcsec): {n_too_close}\n"
                f"  Drifted outside {xmax * 1.5} arcsec: {n_too_far}\n"
                f"  Slope pinned at (0.05, 1.95) boundary: {n_slope_pinned}\n"
                f"  kappa_star below noise floor: {n_kappa_floor} "
                f"(filter {'on' if kappa_floor_n_sigma > 0 else 'off'})\n"
                f"  median(sigf) = {median_sigf:.3e}\n"
                f"If slope-pinning dominates, the position optimizer is "
                f"hitting bounds — check that cast_votes outputs are "
                f"clipped into optimizer ranges."
            )

    else:
        raise ValueError(
            "Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'."
        )

    if len(lenses.x) == 0:
        raise ValueError('No valid lenses remain after filtering.')

    return lenses

def merge_close_lenses(lenses, merger_threshold=5, lens_type='SIS'):
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
    if lens_type == 'SIS':
        strength = np.abs(lenses.te)
    elif lens_type == 'NFW':
        strength = np.abs(lenses.mass)
    elif lens_type == 'POWER_LAW':
        strength = np.abs(lenses.kappa_star)
    else:
        raise ValueError(
            "Invalid lens type — must be 'SIS', 'NFW', or 'POWER_LAW'."
        )

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
        lenses.remove([j])              # Remove lens at index j

    i = 0
    while i < len(lenses.x):
        j = i + 1
        while j < len(lenses.x):
            distance = np.hypot(lenses.x[i] - lenses.x[j],
                                lenses.y[i] - lenses.y[j])
            if distance < merger_threshold:
                merge_lenses(i, j)
            else:
                j += 1
        i += 1

    # Update lens properties based on type
    if lens_type == 'NFW':
        lenses.calculate_concentration()
    # POWER_LAW: no analog needed — theta_star is a fixed convention,
    # not a derived property.  (kappa_star, slope) are passed to
    # strength optimization as-is.

    return lenses

def forward_lens_selection(
    sources, candidate_lenses, use_flags, lens_type='NFW',
    base_tolerance=0.003, mass_scale=1e13, exponent=-1.0,
    use_strong_lensing: bool = False, lambda_sl: float = None,
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
    if lens_type == 'NFW':
        selected_lenses = halo_obj.NFW_Lens(
            x=np.array([]), y=np.array([]), z=np.array([]),
            concentration=np.array([]), mass=np.array([]),
            redshift=candidate_lenses.redshift, chi2=np.array([]),
        )
    elif lens_type == 'SIS':
        selected_lenses = halo_obj.SIS_Lens(
            x=np.array([]), y=np.array([]),
            te=np.array([]), chi2=np.array([]),
        )
    elif lens_type == 'POWER_LAW':
        selected_lenses = halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=candidate_lenses.theta_star,
            redshift=candidate_lenses.redshift,
            chi2=np.array([]),
        )
        # Force WL-only during the selection loop.
        use_strong_lensing = False
        lambda_sl = None
    else:
        raise ValueError(
            "Unsupported lens type. Choose 'NFW', 'SIS', or 'POWER_LAW'."
        )

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
            if lens_type == 'NFW':
                test_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    z=np.append(selected_lenses.z, candidate_lenses.z[idx]),
                    concentration=np.append(
                        selected_lenses.concentration,
                        candidate_lenses.concentration[idx]),
                    mass=np.append(selected_lenses.mass,
                                   candidate_lenses.mass[idx]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx]),
                )
            elif lens_type == 'SIS':
                test_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    te=np.append(selected_lenses.te,
                                 candidate_lenses.te[idx]),
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx]),
                )
            elif lens_type == 'POWER_LAW':
                test_lenses = halo_obj.PowerLawHalo(
                    x=np.append(selected_lenses.x, candidate_lenses.x[idx]),
                    y=np.append(selected_lenses.y, candidate_lenses.y[idx]),
                    kappa_star=np.append(selected_lenses.kappa_star,
                                         candidate_lenses.kappa_star[idx]),
                    slope=np.append(selected_lenses.slope,
                                    candidate_lenses.slope[idx]),
                    theta_star=candidate_lenses.theta_star,
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx]),
                )

            # Compute reduced chi-squared (WL only for POWER_LAW)
            chi2, dof, _ = metric.calculate_total_chi2(
                sources, test_lenses, use_flags, lens_type=lens_type,
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
        if lens_type == 'NFW':
            lens_strength = candidate_lenses.mass[idx_to_add]
            adaptive_tolerance = (base_tolerance
                                  * (lens_strength / mass_scale) ** exponent)
        elif lens_type == 'SIS':
            lens_strength = np.abs(candidate_lenses.te[idx_to_add])
            te_scale = 1.0
            adaptive_tolerance = (base_tolerance
                                  * (lens_strength / te_scale) ** exponent)
        elif lens_type == 'POWER_LAW':
            lens_strength = np.abs(candidate_lenses.kappa_star[idx_to_add])
            adaptive_tolerance = (base_tolerance
                                  * (lens_strength / kappa_scale) ** exponent)

        # Accept if improvement exceeds tolerance
        if min_chi2 < best_reduced_chi2 - adaptive_tolerance:
            best_reduced_chi2 = min_chi2

            if lens_type == 'NFW':
                selected_lenses = halo_obj.NFW_Lens(
                    x=np.append(selected_lenses.x,
                                candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y,
                                candidate_lenses.y[idx_to_add]),
                    z=np.append(selected_lenses.z,
                                candidate_lenses.z[idx_to_add]),
                    concentration=np.append(
                        selected_lenses.concentration,
                        candidate_lenses.concentration[idx_to_add]),
                    mass=np.append(selected_lenses.mass,
                                   candidate_lenses.mass[idx_to_add]),
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx_to_add]),
                )
            elif lens_type == 'SIS':
                selected_lenses = halo_obj.SIS_Lens(
                    x=np.append(selected_lenses.x,
                                candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y,
                                candidate_lenses.y[idx_to_add]),
                    te=np.append(selected_lenses.te,
                                 candidate_lenses.te[idx_to_add]),
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx_to_add]),
                )
            elif lens_type == 'POWER_LAW':
                selected_lenses = halo_obj.PowerLawHalo(
                    x=np.append(selected_lenses.x,
                                candidate_lenses.x[idx_to_add]),
                    y=np.append(selected_lenses.y,
                                candidate_lenses.y[idx_to_add]),
                    kappa_star=np.append(
                        selected_lenses.kappa_star,
                        candidate_lenses.kappa_star[idx_to_add]),
                    slope=np.append(selected_lenses.slope,
                                    candidate_lenses.slope[idx_to_add]),
                    theta_star=candidate_lenses.theta_star,
                    redshift=candidate_lenses.redshift,
                    chi2=np.append(selected_lenses.chi2,
                                   candidate_lenses.chi2[idx_to_add]),
                )

            remaining_indices = np.delete(remaining_indices, min_index)
            improved = True
        else:
            break

    # ----------------------------------------------------------------
    # Empty selection
    # ----------------------------------------------------------------
    if len(selected_lenses.x) == 0:
        print('No lenses selected.')
        if lens_type == 'POWER_LAW' and return_lambda_sl:
            return None, np.inf, 0.0
        return None, np.inf

    # ----------------------------------------------------------------
    # POWER_LAW post-selection: compute lambda_sl once and freeze
    # ----------------------------------------------------------------
    if lens_type == 'POWER_LAW':
        lambda_sl_final = _compute_lambda_sl_power_law(
            sources, selected_lenses, use_flags,
            strong_systems=strong_systems,
        )
        if return_lambda_sl:
            return selected_lenses, best_reduced_chi2, lambda_sl_final
        # Default tuple unchanged for backward-compat callers
        return selected_lenses, best_reduced_chi2

    return selected_lenses, best_reduced_chi2

def _compute_lambda_sl_power_law(sources, halos, use_flags,
                                 strong_systems=None):
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
    chi2_wl = metric.chi2_wl_power_law(
        halos, sources, use_flags=use_flags, apply_penalties=False,
    )
    dof_wl = metric.calc_dof_wl_power_law(sources, halos, use_flags)
    if not np.isfinite(dof_wl) or dof_wl <= 0:
        return 0.0
    rchi2_wl = chi2_wl / dof_wl

    # SL contribution.  Compute sigma_n from the WL Hessian first so
    # the profile-uncertainty term is properly accounted for.
    try:
        sigma_n = metric.posterior_sigma_n(
            halos, sources, use_flags=use_flags,
        )
    except Exception:
        sigma_n = None

    chi2_sl = utils.chi2_strong_source_plane_power_law(
        halos, strong_systems,
        sigma_n=sigma_n, alpha_cal=1.0,
    )

    # SL DOF: 2 numbers (x, y in source plane) per image after
    # marginalizing one source-plane mean per system, summed over
    # systems.  Match the SIS/NFW convention.
    n_images_total = sum(int(np.atleast_1d(sys.theta_x).size)
                         for sys in strong_systems)
    n_systems = sum(1 for _ in strong_systems)
    dof_sl = 2 * n_images_total - 2 * n_systems
    if dof_sl <= 0:
        return 0.0
    rchi2_sl = chi2_sl / dof_sl
    if rchi2_sl <= 0:
        return 0.0

    return float(rchi2_wl / rchi2_sl)

def optimize_lens_strength(sources, lenses, use_flags, lens_type='SIS',
                           use_strong_lensing: bool = False,
                           lambda_sl: float = None,
                           strong_systems=None):
    """
    Optimizes the strength parameters of the lenses at fixed positions.

    Per-lens-type behavior:
      - SIS:        Multi-parameter Powell minimization over te_i,
                    targeting |chi^2/dof - 1|.  Existing behavior.
      - NFW:        Per-halo scalar minimization over log10(mass_i)
                    with concentration recomputed via the Duffy
                    relation.  Existing behavior.
      - POWER_LAW:  2*N_halo-dim joint Nelder-Mead minimization over
                    {(log10(kappa_star_i), n_i)} with positions held
                    fixed.  Joint optimization is required because
                    kappa_star and n are coupled along a degeneracy
                    valley; fitting them separately misses the
                    correlation.

    For POWER_LAW the targeted objective is the raw combined chi-squared
    chi2_WL + lambda_sl * chi2_SL (matching the NFW convention).  The
    SIS convention of targeting |chi^2/dof - 1| is robust only for
    noisy data near the canonical reduced-chi-squared = 1; for
    noiseless or strongly-fit data it produces spurious global minima
    that are not at truth.  Strong-lensing constraints, when present,
    are added with the supplied lambda_sl held FROZEN throughout the
    optimization — recomputing it inside the inner loop would
    non-stationarize the objective and prevent convergence (matches
    the architectural decision from the NFW pipeline).

    Parameters
    ----------
    sources : Source
        Sources for the WL chi^2.
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Lens collection at fixed positions.
    use_flags : sequence of three bool
        (use_shear, use_flexion, use_g_flexion).
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    use_strong_lensing : bool
        Whether to include SL in the objective.
    lambda_sl : float or None
        Pre-computed and frozen SL weight (POWER_LAW expects the value
        returned by forward_lens_selection with return_lambda_sl=True;
        SIS/NFW pass through to chi2wrapper).
    strong_systems : iterable of StrongLensingSystem or None
        POWER_LAW only.  Required if use_strong_lensing=True.

    Returns
    -------
    SIS_Lens, NFW_Lens, or PowerLawHalo
        Lenses with optimized strengths (positions unchanged).
    """
    opts = {"use_strong_lensing": use_strong_lensing, "lambda_sl": lambda_sl}

    if lens_type == 'SIS':
        guess = lenses.te
        params = ['SIS', 'constrained', lenses.x, lenses.y,
                  sources, use_flags, opts]
        max_attempts = 5
        best_result = None
        best_params = guess
        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper, guess, args=params,
                method='Powell', tol=1e-8,
                options={'maxiter': 1000},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        lenses.te = best_params

    elif lens_type == 'NFW':
        for i in range(len(lenses.x)):
            params = [
                'NFW', 'constrained',
                lenses.x[i], lenses.y[i], lenses.redshift,
                lenses.concentration[i], sources, use_flags, opts,
            ]
            chi2_fn = lambda x: chi2wrapper(x, params)
            res = minimize_scalar(
                chi2_fn, bounds=(10.0, 17.0), method="bounded",
                options={"xatol": 1e-6, "maxiter": 2000},
            )
            lenses.mass[i] = 10 ** res.x
            lenses.calculate_concentration()

    elif lens_type == 'POWER_LAW':
        N_h = len(lenses.x)
        if N_h == 0:
            return lenses

        # Pack initial guess: [log10(k*_0), n_0, log10(k*_1), n_1, ...]
        guess = np.empty(2 * N_h)
        bounds = []
        for i in range(N_h):
            k0 = float(max(lenses.kappa_star[i], 1e-6))
            n0 = float(np.clip(lenses.slope[i], 0.05, 1.95))
            guess[2 * i] = np.log10(k0)
            guess[2 * i + 1] = n0
            bounds.append((-6.0, 1.0))   # log10(kappa_star) range
            bounds.append((0.05, 1.95))  # slope range

        # Calibrated initial simplex steps (matches step 15 calibration):
        # 0.3 dex for log10(kappa_star), 0.2 for slope.  Empirically
        # required for Nelder-Mead to escape local traps when the
        # post-merge seed is offset from the WL+SL minimum.
        simplex_steps = np.empty(2 * N_h)
        simplex_steps[0::2] = 0.3   # log10(kappa_star) steps in dex
        simplex_steps[1::2] = 0.2   # slope steps
        initial_simplex = np.zeros((2 * N_h + 1, 2 * N_h))
        initial_simplex[0] = guess
        for k in range(2 * N_h):
            vertex = guess.copy()
            vertex[k] += simplex_steps[k]
            lo, hi = bounds[k]
            if vertex[k] > hi:
                vertex[k] = guess[k] - simplex_steps[k]
                if vertex[k] < lo:
                    vertex[k] = 0.5 * (lo + hi)
            initial_simplex[k + 1] = vertex

        # Capture lens metadata and constants used inside the objective.
        x_fixed = lenses.x.copy()
        y_fixed = lenses.y.copy()
        theta_star_fixed = lenses.theta_star
        redshift_fixed = lenses.redshift

        def objective_function(packed_params):
            return _strength_chi2_target_power_law(
                packed_params, x_fixed, y_fixed,
                theta_star_fixed, redshift_fixed,
                sources, use_flags,
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,            # FROZEN throughout
                strong_systems=strong_systems,
            )

        result = opt.minimize(
            objective_function, guess,
            method='Nelder-Mead',
            bounds=bounds,
            options={
                'maxiter': int(1e6),
                'xatol': 1e-5,
                'fatol': 1e-5,
                'adaptive': True,
                'initial_simplex': initial_simplex,
            },
        )

        # Unpack and write back
        best = result.x
        for i in range(N_h):
            lenses.kappa_star[i] = 10 ** best[2 * i]
            lenses.slope[i] = best[2 * i + 1]

    else:
        raise ValueError(
            'Invalid lens type — must be "SIS", "NFW", or "POWER_LAW".'
        )

    return lenses

def _strength_chi2_target_power_law(packed_params, x_fixed, y_fixed,
                                    theta_star, redshift,
                                    sources, use_flags,
                                    use_strong_lensing=False,
                                    lambda_sl=None,
                                    strong_systems=None):
    """
    2*N_halo-dim objective for POWER_LAW strength optimization.

    Unpacks [log10(k*_0), n_0, log10(k*_1), n_1, ...] into a multi-halo
    PowerLawHalo with positions fixed at x_fixed, y_fixed, then
    evaluates the combined chi-squared

        chi2_total = chi2_WL + lambda_sl * chi2_SL

    as the target.  This is the same target used by the NFW strength
    optimizer (raw chi^2, not |chi^2/dof - 1|).  The reduced-chi^2-
    targeting form used by SIS is robust only for noisy data near
    chi^2/dof = 1; for noiseless or strongly-fit data it produces
    spurious global minima at chi^2/dof = 1 that are NOT at truth.
    Raw chi^2 is well-behaved at all noise levels.

    Strong-lensing chi^2 is added with the supplied lambda_sl held
    FROZEN — this function never recomputes it.  The user's
    convention from forward_lens_selection is that lambda_sl is
    fixed once after WL selection and propagated unchanged through
    merging and strength optimization.

    Parameters
    ----------
    packed_params : ndarray, shape (2 * N_halo,)
        Flat parameter vector.
    x_fixed, y_fixed : ndarray
        Fixed halo positions (arcsec).
    theta_star : float
    redshift : float
    sources : Source
    use_flags : sequence of three bool
    use_strong_lensing : bool
    lambda_sl : float or None
    strong_systems : iterable of StrongLensingSystem or None

    Returns
    -------
    chi2_total : float
        chi2_WL + lambda_sl * chi2_SL.
    """
    N_h = x_fixed.size

    # Unpack
    k_arr = np.empty(N_h)
    n_arr = np.empty(N_h)
    for i in range(N_h):
        k_arr[i] = 10 ** packed_params[2 * i]
        n_arr[i] = packed_params[2 * i + 1]

    halos = halo_obj.PowerLawHalo(
        x=x_fixed, y=y_fixed,
        kappa_star=k_arr, slope=n_arr,
        theta_star=theta_star, redshift=redshift,
        chi2=np.zeros(N_h),
    )

    chi2_wl = metric.chi2_wl_power_law(
        halos, sources, use_flags=use_flags, apply_penalties=False,
    )

    chi2_total = chi2_wl
    if use_strong_lensing and (lambda_sl is not None) and (strong_systems is not None):
        try:
            sigma_n = metric.posterior_sigma_n(
                halos, sources, use_flags=use_flags,
            )
        except Exception:
            sigma_n = None
        chi2_sl = utils.chi2_strong_source_plane_power_law(
            halos, strong_systems,
            sigma_n=sigma_n, alpha_cal=1.0,
        )
        chi2_total = chi2_wl + lambda_sl * chi2_sl

    return float(chi2_total)

def update_chi2_values(sources, lenses, use_flags, lens_type='NFW',
                       use_strong_lensing: bool = False, lambda_sl: float = None):
    """
    Updates per-lens chi2 (WL-only per-lens is fine) and returns reduced
    chi2 for the combined WL+SL objective at the global level.

    Parameters
    ----------
    sources : Source
        Source object.
    lenses : SIS_Lens, NFW_Lens, or PowerLawHalo
        Lens collection to score.
    use_flags : list of bool
        Which lensing signals to use (shear / flexion / g-flexion).
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.
    use_strong_lensing : bool
        Whether to include strong lensing.
    lambda_sl : float or None
        Pre-computed SL weight (frozen across optimizer calls).

    Returns
    -------
    float
        Reduced chi-squared for the combined WL+SL objective.

    Notes
    -----
    Per-lens chi2 is computed WL-only (consistent with the existing
    SIS/NFW behavior).  For POWER_LAW, the per-lens reconstruction
    builds a single-entry PowerLawHalo carrying the cluster-level
    metadata (theta_star, redshift) shared across all halos, and the
    per-halo (x, y, kappa_star, slope) of that one entry.

    For POWER_LAW, the WL-only path is the only currently-supported
    branch through this function; SL handling for POWER_LAW is
    delegated to forward_lens_selection (which computes lambda_sl
    once after WL selection completes and freezes it through merging
    and strength optimization).  If use_strong_lensing=True is passed
    with lens_type='POWER_LAW', the underlying calculate_total_chi2
    will raise NotImplementedError.
    """
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        sources, lenses, use_flags, lens_type=lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    reduced_chi2 = chi2_total / dof_total if dof_total != 0 else np.inf

    # --- Per-lens bookkeeping (WL-only, by design) ---
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
            elif lens_type == "POWER_LAW":
                # Cluster-level metadata (theta_star, redshift) is
                # shared across all halos in the collection; each
                # single-halo reconstruction carries the same values.
                one_halo = halo_obj.PowerLawHalo(
                    x=[lenses.x[i]], y=[lenses.y[i]],
                    kappa_star=[lenses.kappa_star[i]],
                    slope=[lenses.slope[i]],
                    theta_star=lenses.theta_star,
                    redshift=lenses.redshift,
                    chi2=[0.0],
                )
            else:
                raise ValueError(
                    'Invalid lens type — must be "SIS", "NFW", or "POWER_LAW"'
                )
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
            # Joint (M, c) optimization: guess = [log10(M), c]
            xl, yl, z_lens, sources, use_flags = tail[0], tail[1], tail[2], tail[3], tail[4]
            xl_arr = np.atleast_1d(xl)
            yl_arr = np.atleast_1d(yl)
            log_mass = float(guess[0])
            conc = float(guess[1])
            lenses = halo_obj.NFW_Lens(
                xl_arr, yl_arr, np.zeros_like(xl_arr),
                np.array([conc]), np.array([10.0 ** log_mass]),
                z_lens, np.zeros_like(xl_arr),
            )
            # Do NOT call calculate_concentration() — c is the fit parameter
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources, lenses, use_flags, lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

    raise ValueError(f"Invalid lensing model/constraint: {model_type} / {constraint_type}")

# ====================================
# === Power-law halo voting scheme ===
# ====================================

def cast_votes_power_law(sources,
                         theta_star=30.0,
                         redshift=0.5,
                         amp_floor_F=0.0,
                         amp_floor_G=0.0,
                         weight_power=2.0):
    """
    Per-source candidate generation for power-law halos via the two
    Phase 0 ratio invariants:

        |G|/|F|     = (2 + n) / (2 - n)         (slope invariant)
        |gamma|/|F| = theta / (2 - n)           (distance invariant)
        F_hat       = unit vector toward halo   (radial pointing)

    For each source s, this routine inverts the invariants to estimate
    (n_s, theta_s, x_s, y_s, kappa_star_s) — a one-candidate-per-source
    seed.  These can then be passed to seed_from_votes() for peak
    extraction or directly into forward selection.  This matches the
    ARCH design philosophy used by the SIS and NFW branches of
    generate_initial_guess: every source contributes a candidate,
    and forward selection is what discriminates true halos from
    noise-driven false positives.

    The slope estimator
        n_hat = 2 (R - 1) / (R + 1),    R = |G|/|F|
    is clipped to (0.05, 1.95) for numerical safety.  Sources whose
    |F| or |G| are below their respective amplitude floors get NaN
    entries — the caller can mask these out.  By default the floors
    are 0 (every source with a non-zero signal contributes).

    Parameters
    ----------
    sources : Source
        Source object carrying e1, e2, f1, f2, g1, g2, x, y arrays.
    theta_star : float
        Pivot radius (arcsec).  Must match the value to be used for
        any subsequent power-law fitting.
    redshift : float
        Cluster redshift (used to construct the returned PowerLawHalo).
    amp_floor_F, amp_floor_G : float
        Minimum |F| and |G| amplitudes for a source to contribute a
        valid vote.  Default 0.0 (every source with non-zero signal
        votes — matches the SIS/NFW one-candidate-per-source philosophy).

        Earlier versions defaulted to 3 * median(sigma), suppressing
        sub-3-sigma sources at the seeding stage.  That was a numerical-
        safety workaround that violated the ARCH "trust every source,
        let forward selection discriminate" principle.  Set explicitly
        if a downstream step is sensitive to noisy seeds.
    weight_power : float
        Vote weight is |F|^weight_power.  Default 2 (favours strong-F
        sources, which provide the most accurate radial estimate).

    Returns
    -------
    votes : dict with keys:
        "x_vote", "y_vote"    : (N,) per-source halo position estimates (arcsec)
        "n_est"               : (N,) per-source slope estimates
        "kappa_star_est"      : (N,) per-source kappa_star estimates
        "weight"              : (N,) per-source vote weights
        "valid"               : (N,) bool mask: True where the vote is usable
    """
    e1 = np.atleast_1d(sources.e1).astype(float)
    e2 = np.atleast_1d(sources.e2).astype(float)
    f1 = np.atleast_1d(sources.f1).astype(float)
    f2 = np.atleast_1d(sources.f2).astype(float)
    g1 = np.atleast_1d(sources.g1).astype(float)
    g2 = np.atleast_1d(sources.g2).astype(float)
    xs = np.atleast_1d(sources.x).astype(float)
    ys = np.atleast_1d(sources.y).astype(float)

    gamma_amp = np.hypot(e1, e2)
    F_amp = np.hypot(f1, f2)
    G_amp = np.hypot(g1, g2)

    # Validity criterion — only excludes degenerate signals.  By default
    # (amp_floor_F = amp_floor_G = 0) this is just a non-zero check; the
    # tiny epsilon floor avoids divide-by-zero on literally-zero signals
    # but otherwise lets every source vote.
    eps = 1.0e-30
    valid = ((F_amp > max(amp_floor_F, eps))
             & (G_amp > max(amp_floor_G, eps)))

    # |G|/|F| -> n_hat
    R = np.where(valid, G_amp / np.where(F_amp > 0, F_amp, 1.0), np.nan)
    n_est = np.where(valid, 2.0 * (R - 1.0) / (R + 1.0), np.nan)
    n_est = np.clip(n_est, 0.05, 1.95)

    # |gamma|/|F| * (2 - n) -> theta_hat (scalar distance to halo)
    r_est = np.where(valid,
                     (2.0 - n_est) * gamma_amp
                       / np.where(F_amp > 0, F_amp, 1.0),
                     np.nan)

    # F_hat unit vector (F points radially TOWARD the halo center,
    # following ARCH's flexion sign convention)
    Fhat_x = np.where(valid, f1 / np.where(F_amp > 0, F_amp, 1.0), np.nan)
    Fhat_y = np.where(valid, f2 / np.where(F_amp > 0, F_amp, 1.0), np.nan)

    # Vote position
    x_vote = xs + r_est * Fhat_x
    y_vote = ys + r_est * Fhat_y

    # kappa_star from the |F| identity:
    #   |F| = n * kappa(theta) / theta = n * kappa_star theta_star^n / theta^(n+1)
    # so kappa_star = |F| theta^(n+1) / (n * theta_star^n)
    n_safe = np.where(np.abs(n_est) > 0.05, n_est, 0.05)
    kappa_star_est = np.where(
        valid,
        F_amp * r_est ** (n_safe + 1.0)
            / (n_safe * theta_star ** n_safe),
        np.nan,
    )
    # Lensing-efficiency inversion: the |F|-based estimate above carries
    # an embedded beta(z_s) factor since the observed signals scale by
    # beta.  For at-infinity convention kappa_star (matching the
    # production calculate_lensing_signals_power_law convention),
    # divide by beta(z_s) using the per-source redshift.
    if hasattr(sources, "redshift"):
        zs_arr = np.atleast_1d(sources.redshift).astype(float)
        if zs_arr.size == xs.size:
            try:
                Dl = cosmo.angular_diameter_distance(redshift).to(u.m).value
                sigma_crit_inf = c.value ** 2 / (4.0 * np.pi * G.value * Dl)
                sigma_crit_zs = np.array([
                    critical_surface_density(redshift, zs_arr[k])
                    if zs_arr[k] > redshift else np.inf
                    for k in range(zs_arr.size)
                ])
                beta_zs = sigma_crit_inf / sigma_crit_zs
                # If beta = 0 (foreground source), the source can't vote
                with np.errstate(divide="ignore", invalid="ignore"):
                    kappa_star_est = np.where(
                        beta_zs > 0,
                        kappa_star_est / beta_zs,
                        np.nan,
                    )
                valid = valid & (beta_zs > 0)
            except Exception:
                pass  # Fall back to the no-correction estimate

    weight = np.where(valid, F_amp ** weight_power, 0.0)

    return {
        "x_vote": x_vote,
        "y_vote": y_vote,
        "n_est": n_est,
        "kappa_star_est": kappa_star_est,
        "weight": weight,
        "valid": valid,
    }


def seed_from_votes(votes, sources,
                    theta_star=30.0,
                    redshift=0.5,
                    field_extent=None,
                    n_pix=120,
                    smoothing_sigma=8.0,
                    n_peaks=None,
                    peak_min_distance=10.0,
                    peak_threshold_rel=0.1,
                    aggregate_radius=15.0):
    """
    Aggregate per-source votes into a small number of candidate halos.

    Procedure:
      1. Rasterize the (x_vote, y_vote, weight) votes onto a 2D grid.
      2. Smooth with a Gaussian of width smoothing_sigma (in pixels).
      3. Find local maxima above peak_threshold_rel * global maximum.
      4. For each peak, compute a weighted-mean of (n, kappa_star) over
         the votes within `aggregate_radius` of that peak.
      5. Return a PowerLawHalo collection of candidates.

    Parameters
    ----------
    votes : dict
        Output of cast_votes_power_law.
    sources : Source
        Original source object (used only for sigma estimates).
    theta_star : float
        Pivot radius (arcsec).
    redshift : float
        Cluster redshift.
    field_extent : (xmin, xmax, ymin, ymax) or None
        Field bounding box.  If None, uses the convex hull of the
        source positions plus a 10% buffer.
    n_pix : int
        Number of grid cells per side for the vote raster.
    smoothing_sigma : float
        Gaussian smoothing width in pixels.
    n_peaks : int or None
        If int, return the top-n_peaks brightest peaks.  If None,
        return all peaks above peak_threshold_rel.
    peak_min_distance : float
        Minimum separation between distinct peaks (arcsec).
    peak_threshold_rel : float
        Minimum peak height as a fraction of the global maximum.
    aggregate_radius : float
        Radius around each peak (arcsec) to aggregate votes for the
        per-peak (n, kappa_star) estimate.

    Returns
    -------
    PowerLawHalo
        A collection of candidate halos seeded from the vote-map peaks.
        Position from peak location, slope and normalization from
        weighted aggregation of nearby votes.
    """
    from scipy.ndimage import gaussian_filter

    valid = votes["valid"]
    if not np.any(valid):
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=theta_star, redshift=redshift,
            chi2=np.array([]),
        )

    x_vote = votes["x_vote"][valid]
    y_vote = votes["y_vote"][valid]
    n_est = votes["n_est"][valid]
    k_est = votes["kappa_star_est"][valid]
    w = votes["weight"][valid]

    # Field extent
    if field_extent is None:
        xs_all = np.atleast_1d(sources.x)
        ys_all = np.atleast_1d(sources.y)
        x_pad = 0.1 * (xs_all.max() - xs_all.min())
        y_pad = 0.1 * (ys_all.max() - ys_all.min())
        xmin, xmax = xs_all.min() - x_pad, xs_all.max() + x_pad
        ymin, ymax = ys_all.min() - y_pad, ys_all.max() + y_pad
    else:
        xmin, xmax, ymin, ymax = field_extent

    # Keep only votes inside the field
    inside = ((x_vote >= xmin) & (x_vote <= xmax)
              & (y_vote >= ymin) & (y_vote <= ymax))
    x_vote = x_vote[inside]; y_vote = y_vote[inside]
    n_est = n_est[inside]; k_est = k_est[inside]; w = w[inside]

    if x_vote.size == 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=theta_star, redshift=redshift,
            chi2=np.array([]),
        )

    # Rasterize
    H, xedges, yedges = np.histogram2d(
        x_vote, y_vote, bins=n_pix,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=w,
    )
    H_smooth = gaussian_filter(H, sigma=smoothing_sigma)

    # Peak finding via local-max scan
    threshold = peak_threshold_rel * H_smooth.max()
    if H_smooth.max() <= 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=theta_star, redshift=redshift,
            chi2=np.array([]),
        )

    # Convert peak_min_distance from arcsec to pixels
    dx_pix = (xmax - xmin) / n_pix
    min_dist_pix = max(int(np.ceil(peak_min_distance / dx_pix)), 1)

    peaks_ix, peaks_iy, peaks_val = _find_peaks_2d(
        H_smooth, threshold=threshold, min_distance=min_dist_pix,
    )
    if peaks_ix.size == 0:
        return halo_obj.PowerLawHalo(
            x=np.array([]), y=np.array([]),
            kappa_star=np.array([]), slope=np.array([]),
            theta_star=theta_star, redshift=redshift,
            chi2=np.array([]),
        )

    # Convert pixel indices to arcsec
    # H[ix, iy] corresponds to xedges[ix] <= x < xedges[ix+1]
    x_peak = 0.5 * (xedges[peaks_ix] + xedges[peaks_ix + 1])
    y_peak = 0.5 * (yedges[peaks_iy] + yedges[peaks_iy + 1])

    # Sort by peak value, take top n_peaks if requested
    order = np.argsort(peaks_val)[::-1]
    if n_peaks is not None:
        order = order[:n_peaks]
    x_peak = x_peak[order]; y_peak = y_peak[order]

    # Per-peak aggregation of n and kappa_star
    n_peak_est = np.zeros_like(x_peak)
    k_peak_est = np.zeros_like(x_peak)
    for k in range(x_peak.size):
        d = np.hypot(x_vote - x_peak[k], y_vote - y_peak[k])
        nearby = d < aggregate_radius
        if np.sum(nearby) >= 3:
            ww = w[nearby]
            n_peak_est[k] = np.sum(n_est[nearby] * ww) / np.sum(ww)
            k_peak_est[k] = np.sum(k_est[nearby] * ww) / np.sum(ww)
        else:
            # Not enough nearby votes — fall back to global weighted median
            n_peak_est[k] = float(np.median(n_est))
            k_peak_est[k] = float(np.median(k_est))

    # Clip slopes back to (0, 2)
    n_peak_est = np.clip(n_peak_est, 0.05, 1.95)
    k_peak_est = np.maximum(k_peak_est, 1.0e-6)

    return halo_obj.PowerLawHalo(
        x=x_peak,
        y=y_peak,
        kappa_star=k_peak_est,
        slope=n_peak_est,
        theta_star=theta_star,
        redshift=redshift,
        chi2=np.zeros_like(x_peak),
    )


def _find_peaks_2d(H, threshold=0.0, min_distance=1):
    """
    Simple local-maximum peak finder for a 2D array.

    Returns ix, iy, value arrays of pixel indices of local maxima
    above `threshold`, separated by at least `min_distance` pixels.

    Uses a non-maximum-suppression sweep over a (2 min_distance + 1)
    square neighborhood.  Adequate for the vote-map sizes ARCH uses
    (~120 x 120 pixels), at which numpy slicing is essentially free.
    """
    ny, nx = H.shape
    peaks = []
    for i in range(ny):
        for j in range(nx):
            v = H[i, j]
            if v < threshold:
                continue
            i0 = max(0, i - min_distance)
            i1 = min(ny, i + min_distance + 1)
            j0 = max(0, j - min_distance)
            j1 = min(nx, j + min_distance + 1)
            window = H[i0:i1, j0:j1]
            if v >= window.max() - 1.0e-15:
                peaks.append((i, j, v))
    if not peaks:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([])
    arr = np.array(peaks, dtype=float)
    ix = arr[:, 0].astype(int)
    iy = arr[:, 1].astype(int)
    val = arr[:, 2]
    return ix, iy, val