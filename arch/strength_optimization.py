"""Per-lens and joint strength (te / mass / kappa_star, slope) optimization."""

import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize_scalar

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.utils as utils
from arch.chi2_wrappers import chi2wrapper


def optimize_lens_strength(
    sources,
    lenses,
    use_flags,
    lens_type="SIS",
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
    strong_systems=None,
):
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

    if lens_type == "SIS":
        guess = lenses.te
        params = ["SIS", "constrained", lenses.x, lenses.y, sources, use_flags, opts]
        max_attempts = 5
        best_result = None
        best_params = guess
        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper,
                guess,
                args=params,
                method="Powell",
                tol=1e-8,
                options={"maxiter": 1000},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        lenses.te = best_params

    elif lens_type == "NFW":
        for i in range(len(lenses.x)):
            params = [
                "NFW",
                "constrained",
                lenses.x[i],
                lenses.y[i],
                lenses.redshift,
                lenses.concentration[i],
                sources,
                use_flags,
                opts,
            ]
            chi2_fn = lambda x: chi2wrapper(x, params)
            res = minimize_scalar(
                chi2_fn,
                bounds=(10.0, 17.0),
                method="bounded",
                options={"xatol": 1e-6, "maxiter": 2000},
            )
            lenses.mass[i] = 10**res.x
            lenses.calculate_concentration()

    elif lens_type == "POWER_LAW":
        N_h = len(lenses.x)
        if N_h == 0:
            return lenses

        # Pack initial guess: [log10(k*_0), n_0, log10(k*_1), n_1, ...]
        # Parameter bounds.  Match the tighter bounds used in
        # optimize_lens_positions: kappa_star in (1e-6, 10), slope in
        # (0.4, 1.7).  The slope range is intentionally narrower than
        # the formal (0.05, 1.95) power-law range to exclude the
        # near-uniform-sheet (n -> 0) and near-singular (n -> 2) limits.
        LOG10_K_LO, LOG10_K_HI = -6.0, 1.0
        N_LO, N_HI = 0.1, 1.7

        guess = np.empty(2 * N_h)
        bounds = []
        for i in range(N_h):
            log_k0 = float(
                np.clip(np.log10(max(lenses.kappa_star[i], 1e-30)), LOG10_K_LO, LOG10_K_HI)
            )
            n0 = float(np.clip(lenses.slope[i], N_LO, N_HI))
            guess[2 * i] = log_k0
            guess[2 * i + 1] = n0
            bounds.append((LOG10_K_LO, LOG10_K_HI))
            bounds.append((N_LO, N_HI))

        # Calibrated initial simplex steps (matches step 15 calibration):
        # 0.3 dex for log10(kappa_star), 0.2 for slope.  Empirically
        # required for Nelder-Mead to escape local traps when the
        # post-merge seed is offset from the WL+SL minimum.
        simplex_steps = np.empty(2 * N_h)
        simplex_steps[0::2] = 0.3  # log10(kappa_star) steps in dex
        simplex_steps[1::2] = 0.2  # slope steps
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
                packed_params,
                x_fixed,
                y_fixed,
                theta_star_fixed,
                redshift_fixed,
                sources,
                use_flags,
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,  # FROZEN throughout
                strong_systems=strong_systems,
            )

        result = opt.minimize(
            objective_function,
            guess,
            method="Nelder-Mead",
            bounds=bounds,
            options={
                "maxiter": min(2000 * N_h, 20000),  # Scale maxiter with N_h, but cap at 20k
                "xatol": 1e-5,
                "fatol": 1e-5,
                "adaptive": True,
                "initial_simplex": initial_simplex,
            },
        )

        # Unpack and write back
        best = result.x
        for i in range(N_h):
            lenses.kappa_star[i] = 10 ** best[2 * i]
            lenses.slope[i] = best[2 * i + 1]

    else:
        raise ValueError('Invalid lens type — must be "SIS", "NFW", or "POWER_LAW".')

    return lenses


def _strength_chi2_target_power_law(
    packed_params,
    x_fixed,
    y_fixed,
    theta_star,
    redshift,
    sources,
    use_flags,
    use_strong_lensing=False,
    lambda_sl=None,
    strong_systems=None,
):
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
        x=x_fixed,
        y=y_fixed,
        kappa_star=k_arr,
        slope=n_arr,
        theta_star=theta_star,
        redshift=redshift,
        chi2=np.zeros(N_h),
    )

    chi2_wl = metric.chi2_wl_power_law(
        halos,
        sources,
        use_flags=use_flags,
        apply_penalties=False,
    )

    chi2_total = chi2_wl
    if use_strong_lensing and (lambda_sl is not None) and (strong_systems is not None):
        try:
            sigma_n = metric.posterior_sigma_n(
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
        )
        chi2_total = chi2_wl + lambda_sl * chi2_sl

    return float(chi2_total)
