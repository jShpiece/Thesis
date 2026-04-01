import arch.pipeline as pipeline
import arch.metric as metric

def fit_lensing_field(sources, xmax, flags=False, use_flags=None, lens_type='SIS',
                      z_lens=0.5, use_strong_lensing: bool = False):
    """
    Reconstructs the gravitational lensing field based on observed source properties.
    
    Parameters:
        sources (Source): Object containing source positions, ellipticity, and flexion signals.
            If ``use_strong_lensing`` is True, ``sources.strong_systems`` must be populated.
        xmax (float): Maximum field radius for lens consideration.
        flags (bool, optional): If True, prints step-by-step progress. Default is False.
        use_flags (list of bool, optional): Specifies which lensing effects to include 
            [shear, flexion, g_flexion]. Default is [True, True, True].
        lens_type (str, optional): Type of lens model ('SIS' or 'NFW'). Default is 'SIS'.
        z_lens (float, optional): Redshift of the lens. Default is 0.5.
        use_strong_lensing (bool, optional): If True, include multiply-imaged systems
            from ``sources.strong_systems`` in the objective. Default is False.

    Returns:
        tuple: (Lenses object, reduced chi-squared value).
    """
    if use_flags is None:
        use_flags = [True, True, True]

    def log_step(message, lenses, reduced_chi2):
        """Logs step information if flags are enabled."""
        if flags:
            print(f"{message}\nLenses: {len(lenses.x)}, Reduced Chi^2: {reduced_chi2:.4f}")

    # Step 1: Generate initial lens candidates
    lenses = pipeline.generate_initial_guess(sources, lens_type, z_lens)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("Initial Guesses:", lenses, reduced_chi2)

    # ── Pre-compute lambda_sl ──
    #
    # The weight λ_SL equalises the reduced-χ² of the WL and SL terms:
    #     λ = (χ²_WL / dof_WL) / (χ²_SL / dof_SL)
    #
    # This ratio is meaningful only when the model is physically plausible,
    # so that rχ² reflects measurement noise rather than gross model error.
    #
    # SIS: the initial guess (θ_E from γ/F) is close enough that rχ²_WL
    #   is representative.  Compute now, before position optimisation.
    #
    # NFW: the initial mass estimate is crude (100 random candidates give
    #   rχ²_WL ~ O(1000)), so λ computed here would be far too large.
    #   Defer to after filtering (Step 3), where implausible candidates
    #   are removed and rχ²_WL reflects the actual WL information content.
    #   NFW position optimisation is WL-only, so λ_SL is not needed
    #   until forward selection (Step 4).
    if use_strong_lensing:
        if lens_type == 'SIS':
            lambda_sl = metric.compute_lambda_sl(sources, lenses, use_flags, lens_type)
            if flags:
                print(f"Pre-computed lambda_sl = {lambda_sl:.6f}")
        else:
            lambda_sl = None   # deferred — computed after filtering below
    else:
        lambda_sl = None

    # Step 2: Optimize lens positions
    lenses = pipeline.optimize_lens_positions(
        sources, lenses, xmax, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    log_step("After Local Minimization:", lenses, reduced_chi2)

    # Step 3: Filter out unsuitable lenses
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    log_step("After Filtering:", lenses, reduced_chi2)

    # Step 4: Select optimal lens set
    #
    # For NFW, lambda_sl is still None here.  forward_selection calls
    # calculate_total_chi2, which falls back to a dynamic lambda at the
    # current parameters.  With ~40 filtered lenses the fallback lambda
    # is tiny (~0.002), so selection is effectively WL-driven — which
    # is physically correct: WL locates substructure, SL calibrates mass.
    lenses, _ = pipeline.forward_lens_selection(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    log_step("After Forward Selection:", lenses, reduced_chi2)

    # ── Deferred lambda_sl for NFW (and any future lens types) ──
    #
    # Now that forward selection has identified a physically plausible
    # model (1–4 halos that explain the WL data), the reduced-χ²
    # ratio is meaningful: rχ²_WL reflects measurement noise, not
    # gross model error.  This lambda is frozen for Steps 5–6.
    if use_strong_lensing and lambda_sl is None:
        lambda_sl = metric.compute_lambda_sl(sources, lenses, use_flags, lens_type)
        if flags:
            print(f"Pre-computed lambda_sl = {lambda_sl:.6f}  (post-selection)")

    # Step 5: Merge nearby lenses
    merger_threshold = (len(sources.x) / (2 * xmax) ** 2) ** (-0.5) if len(sources.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    log_step("After Merging Lenses:", lenses, reduced_chi2)

    # Step 6: Final optimization
    lenses = pipeline.optimize_lens_strength(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, [True, True, True], lens_type,
        use_strong_lensing=use_strong_lensing, lambda_sl=lambda_sl
    )
    log_step("After Final Optimization:", lenses, reduced_chi2)

    return lenses, reduced_chi2