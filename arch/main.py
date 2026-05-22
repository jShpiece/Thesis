"""Top-level ARCH driver: fit_lensing_field."""

import arch.metric as metric
import arch.pipeline as pipeline


def fit_lensing_field(
    sources,
    xmax,
    flags=False,
    use_flags=None,
    lens_type="SIS",
    z_lens=0.5,
    use_strong_lensing: bool = False,
    use_sl_in_selection: bool = False,
):
    """
    Reconstructs the gravitational lensing field based on observed
    source properties.

    Parameters
    ----------
    sources : Source
        Source catalog.  If ``use_strong_lensing`` is True,
        ``sources.strong_systems`` must be populated.
    xmax : float
        Maximum field radius for lens consideration (arcsec).
    flags : bool
        If True, prints step-by-step progress.  Default False.
    use_flags : list of bool, optional
        [use_shear, use_flexion, use_g_flexion].  Default
        [True, True, True].
    lens_type : str
        'SIS', 'NFW', or 'POWER_LAW'.  Default 'SIS'.
    z_lens : float
        Lens redshift.  Default 0.5.
    use_strong_lensing : bool
        Include multiply-imaged systems from ``sources.strong_systems``
        in the post-selection objective (merging and strength
        optimization).  Default False.
    use_sl_in_selection : bool
        Use strong-lensing information during forward selection via a
        two-pass scheme (Pass 1 WL-only, Pass 2 WL+λ_SL on rejected
        candidates).  Default False to preserve legacy single-pass
        behaviour; recommended True for NFW and POWER_LAW when
        ``use_strong_lensing=True`` AND ``sources.strong_systems`` is
        populated.  Has no effect when use_strong_lensing is False.

    Returns
    -------
    lenses, reduced_chi2 : tuple

    Notes
    -----
    Two-pass selection rationale:
        The original SL-in-selection attempt (with λ_SL computed from
        the initial guess and active during single-pass selection)
        degraded NFW mass recovery from 58% to 23% on the Abell 2744
        scenario.  The cause was a poorly-calibrated λ_SL combined
        with the high curvature of χ²_SL near the correct solution,
        which broke the greedy-monotonicity assumption.

        The two-pass scheme calibrates λ_SL at the Pass 1 minimum
        (where deflections are physical and the slope uncertainty is
        small) and resumes selection only on candidates that Pass 1
        rejected.  These are typically the small/peripheral candidates
        that WL alone couldn't justify, but which SL geometry may
        require — substructure halos near critical curves, mass-sheet
        calibration shifts, etc.  Pass 2 cannot remove Pass 1 halos;
        backward elimination is a future extension.

    SL convention (NFW and POWER_LAW), single-pass mode:
        lambda_sl is computed AFTER forward selection on the post-
        selection lens model where the SL χ² is well-defined.  Frozen
        through merging and strength optimization.

    SL convention (NFW and POWER_LAW), two-pass mode:
        Same as single-pass but λ_SL is computed AFTER Pass 1 (not
        AFTER all selection), and is also used during Pass 2.  Frozen
        for merging and strength optimization downstream.

    SIS convention:
        Pre-computation via metric.compute_lambda_sl is retained for
        SIS to preserve legacy behavior on the (rarely-used) SIS path.
        SIS does NOT yet have two-pass support — pass
        use_sl_in_selection=False for SIS.
    """
    if use_flags is None:
        use_flags = [True, True, True]

    def log_step(message, lenses, reduced_chi2):
        if flags:
            print(f"{message}\nLenses: {len(lenses.x)}, "
                  f"Reduced Chi^2: {reduced_chi2:.4f}")

    # ── Step 1: Generate initial lens candidates ──
    lenses = pipeline.generate_initial_guess(sources, lens_type, z_lens)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("Initial Guesses:", lenses, reduced_chi2)

    # ── Pre-compute lambda_sl (SIS only) ──
    if use_strong_lensing and lens_type == "SIS":
        lambda_sl = metric.compute_lambda_sl(sources, lenses, use_flags, lens_type)
        if flags:
            print(f"Pre-computed lambda_sl = {lambda_sl:.6f}")
    else:
        lambda_sl = None

    # ── Step 2: Optimize lens positions ──
    lenses = pipeline.optimize_lens_positions(
        sources, lenses, xmax, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Local Minimization:", lenses, reduced_chi2)

    # ── Step 3: Filter out unsuitable lenses ──
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Filtering:", lenses, reduced_chi2)

    # ── Step 4: Forward selection ──
    if use_strong_lensing and lens_type in ("NFW", "POWER_LAW") and use_sl_in_selection:
        # ── Two-pass path: WL → λ_SL → WL+SL ──
        lenses, _, lambda_sl, diag = pipeline.forward_lens_selection_two_pass(
            sources, lenses, use_flags, lens_type,
            return_diagnostics=True,
        )
        if flags:
            print(
                f"Two-pass selection:  pass1={diag['n_pass1']} halos, "
                f"pass2 added {diag['n_pass2_added']}, "
                f"lambda_sl={diag['lambda_sl']:.6f}, "
                f"chi2: {diag['chi2_pass1']:.3f} -> {diag['chi2_final']:.3f}"
            )

    elif use_strong_lensing and lens_type in ("NFW", "POWER_LAW"):
        # ── Single-pass path (legacy): WL-only selection, then compute λ_SL ──
        if lens_type == "POWER_LAW":
            selected, _, lambda_sl_post = pipeline.forward_lens_selection(
                sources, lenses, use_flags, lens_type,
                use_strong_lensing=False, lambda_sl=None,
                strong_systems=(
                    sources.strong_systems
                    if hasattr(sources, "strong_systems") else None
                ),
                return_lambda_sl=True,
            )
            lenses = selected
            lambda_sl = lambda_sl_post
        else:
            lenses, _ = pipeline.forward_lens_selection(
                sources, lenses, use_flags, lens_type,
                use_strong_lensing=False, lambda_sl=None,
            )
            lambda_sl = metric.compute_lambda_sl(
                sources, lenses, use_flags, lens_type
            )
        if flags:
            print(
                f"Post-selection lambda_sl = "
                f"{(lambda_sl if lambda_sl is not None else 0.0):.6f}"
            )
    else:
        # SIS path with pre-computed lambda_sl, or WL-only run
        lenses, _ = pipeline.forward_lens_selection(
            sources, lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        )

    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Forward Selection:", lenses, reduced_chi2)

    # ── Step 5: Merge nearby lenses ──
    merger_threshold = (
        (len(sources.x) / (2 * xmax) ** 2) ** (-0.5)
        if len(sources.x) > 0 else 1.0
    )
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Merging Lenses:", lenses, reduced_chi2)

    # ── Step 6: Final optimization (strength) ──
    if lens_type == "POWER_LAW":
        lenses = pipeline.optimize_lens_strength(
            sources, lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
            strong_systems=(
                sources.strong_systems
                if (use_strong_lensing and hasattr(sources, "strong_systems"))
                else None
            ),
        )
    else:
        lenses = pipeline.optimize_lens_strength(
            sources, lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        )

    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, [True, True, True], lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Final Optimization:", lenses, reduced_chi2)

    return lenses, reduced_chi2