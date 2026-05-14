"""Chi-squared wrapper functions for scipy optimizers and per-lens bookkeeping."""

import numpy as np

import arch.halo_obj as halo_obj
import arch.metric as metric


def update_chi2_values(
    sources,
    lenses,
    use_flags,
    lens_type="NFW",
    use_strong_lensing: bool = False,
    lambda_sl: float = None,
):
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
        sources,
        lenses,
        use_flags,
        lens_type=lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
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
                    lenses.x[i],
                    lenses.y[i],
                    lenses.z[i],
                    lenses.concentration[i],
                    lenses.mass[i],
                    lenses.redshift,
                    [0],
                )
                one_halo.calculate_concentration()
            elif lens_type == "SIS":
                one_halo = halo_obj.SIS_Lens(lenses.x[i], lenses.y[i], lenses.te[i], [0])
            elif lens_type == "POWER_LAW":
                # Cluster-level metadata (theta_star, redshift) is
                # shared across all halos in the collection; each
                # single-halo reconstruction carries the same values.
                one_halo = halo_obj.PowerLawHalo(
                    x=[lenses.x[i]],
                    y=[lenses.y[i]],
                    kappa_star=[lenses.kappa_star[i]],
                    slope=[lenses.slope[i]],
                    theta_star=lenses.theta_star,
                    redshift=lenses.redshift,
                    chi2=[0.0],
                )
            else:
                raise ValueError('Invalid lens type — must be "SIS", "NFW", or "POWER_LAW"')
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

    if model_type == "SIS":
        if constraint_type == "unconstrained":
            # params expected: [sources, use_flags]
            sources = tail[0]
            use_flags = tail[1]
            lenses = halo_obj.SIS_Lens(guess[0], guess[1], guess[2], [0])
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources,
                lenses,
                use_flags,
                lens_type="SIS",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == "constrained":
            # params expected: [x_array, y_array, sources, use_flags]
            xl, yl, sources, use_flags = tail[0], tail[1], tail[2], tail[3]
            lenses = halo_obj.SIS_Lens(xl, yl, guess, np.empty_like(xl))
            chi2_total, dof_total, _ = metric.calculate_total_chi2(
                sources,
                lenses,
                use_flags,
                lens_type="SIS",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return np.abs(chi2_total / dof_total - 1) if dof_total > 0 else np.inf

    elif model_type == "NFW":
        if constraint_type == "unconstrained":
            # tail expected: [sources, use_flags, concentration?, redshift?] (existing code style)
            sources = tail[0]
            use_flags = tail[1]
            concentration = tail[2]
            redshift = tail[3]
            lenses = halo_obj.NFW_Lens(
                guess[0],
                guess[1],
                np.zeros_like(guess[0]),
                concentration,
                10 ** guess[2],
                redshift,
                [0],
            )
            lenses.calculate_concentration()
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources,
                lenses,
                use_flags,
                lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == "constrained":
            # params: [x, y, z_lens, concentration, sources, use_flags]
            xl, yl, z_lens, concentration, sources, use_flags = (
                tail[0],
                tail[1],
                tail[2],
                tail[3],
                tail[4],
                tail[5],
            )
            lenses = halo_obj.NFW_Lens(
                xl,
                yl,
                np.zeros_like(xl),
                concentration,
                10**guess,
                z_lens,
                np.empty_like(np.atleast_1d(xl)),
            )
            lenses.calculate_concentration()
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources,
                lenses,
                use_flags,
                lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

        elif constraint_type == "dual":
            # Joint (M, c) optimization: guess = [log10(M), c]
            xl, yl, z_lens, sources, use_flags = tail[0], tail[1], tail[2], tail[3], tail[4]
            xl_arr = np.atleast_1d(xl)
            yl_arr = np.atleast_1d(yl)
            log_mass = float(guess[0])
            conc = float(guess[1])
            lenses = halo_obj.NFW_Lens(
                xl_arr,
                yl_arr,
                np.zeros_like(xl_arr),
                np.array([conc]),
                np.array([10.0**log_mass]),
                z_lens,
                np.zeros_like(xl_arr),
            )
            # Do NOT call calculate_concentration() — c is the fit parameter
            chi2_total, _, _ = metric.calculate_total_chi2(
                sources,
                lenses,
                use_flags,
                lens_type="NFW",
                use_strong_lensing=use_strong_lensing,
                lambda_sl=lambda_sl,
                use_magnification_correction_sl=False,
            )
            return chi2_total

    raise ValueError(f"Invalid lensing model/constraint: {model_type} / {constraint_type}")
