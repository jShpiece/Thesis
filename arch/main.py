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
        in the objective.  Default False.

    Returns
    -------
    lenses, reduced_chi2 : tuple

    Notes
    -----
    SL convention (NFW and POWER_LAW):
        lambda_sl is computed AFTER forward selection, using the
        post-selection lens model where the SL chi^2 is well-defined.
        It is then FROZEN through the merge and strength stages.

        Computing lambda_sl on the initial-guess lens model (which has
        ~1000 random per-source halos) is unreliable because the SL
        chi^2 evaluated on that model can be artificially small due to
        the profile-model-uncertainty term in chi2_strong_source_plane_*
        scaling with the (huge, random) deflection magnitudes.  Empirically
        observed on Abell 2744 with NFW: pre-selection chi2_SL = 0.077
        gave lambda_sl ~ 178000 (capped at 50), which then drove NFW
        masses up by a factor of ~7 to satisfy the over-weighted SL.

        The post-selection convention sidesteps this: chi2_SL is
        evaluated on the converged WL-only lens model, where the
        deflections are physical and the SL contribution is well-
        scaled.

    SIS convention:
        Pre-computation via metric.compute_lambda_sl is retained for
        SIS to preserve legacy behavior on the (rarely-used) SIS path.
    """
    if use_flags is None:
        use_flags = [True, True, True]

    def log_step(message, lenses, reduced_chi2):
        if flags:
            print(f"{message}\nLenses: {len(lenses.x)}, " f"Reduced Chi^2: {reduced_chi2:.4f}")

    # ── Step 1: Generate initial lens candidates ──
    lenses = pipeline.generate_initial_guess(sources, lens_type, z_lens)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("Initial Guesses:", lenses, reduced_chi2)

    # ── Pre-compute lambda_sl (SIS only) ──
    # NFW and POWER_LAW defer to after forward selection — see docstring.
    if use_strong_lensing and lens_type == "SIS":
        lambda_sl = metric.compute_lambda_sl(sources, lenses, use_flags, lens_type)
        if flags:
            print(f"Pre-computed lambda_sl = {lambda_sl:.6f}")
    else:
        lambda_sl = None

    # ── Step 2: Optimize lens positions ──
    # NFW and POWER_LAW: lambda_sl is None, so position optimization
    # runs WL-only (matching the post-selection convention).
    # SIS: pre-computed lambda_sl is used.
    lenses = pipeline.optimize_lens_positions(
        sources,
        lenses,
        xmax,
        use_flags,
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    reduced_chi2 = pipeline.update_chi2_values(
        sources,
        lenses,
        use_flags,
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Local Minimization:", lenses, reduced_chi2)

    # ── Step 3: Filter out unsuitable lenses ──
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources,
        lenses,
        use_flags,
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Filtering:", lenses, reduced_chi2)

    # ── Step 4: Forward selection ──
    # Both NFW and POWER_LAW: forward selection runs WL-only, then
    # lambda_sl is computed at the post-selection minimum and frozen.
    if use_strong_lensing and lens_type in ("NFW", "POWER_LAW"):
        if lens_type == "POWER_LAW":
            # POWER_LAW: forward_lens_selection has return_lambda_sl=True
            # path that computes lambda_sl internally.
            selected, _, lambda_sl_post = pipeline.forward_lens_selection(
                sources,
                lenses,
                use_flags,
                lens_type,
                use_strong_lensing=False,  # forced WL-only internally
                lambda_sl=None,
                strong_systems=(
                    sources.strong_systems if hasattr(sources, "strong_systems") else None
                ),
                return_lambda_sl=True,
            )
            lenses = selected
            lambda_sl = lambda_sl_post
        else:
            # NFW: forward selection WL-only, then compute lambda_sl
            # via metric.compute_lambda_sl on the post-selection model.
            lenses, _ = pipeline.forward_lens_selection(
                sources,
                lenses,
                use_flags,
                lens_type,
                use_strong_lensing=False,
                lambda_sl=None,
            )
            lambda_sl = metric.compute_lambda_sl(sources, lenses, use_flags, lens_type)
        if flags:
            print(
                f"Post-selection lambda_sl = "
                f"{(lambda_sl if lambda_sl is not None else 0.0):.6f}"
            )
    else:
        # SIS path with pre-computed lambda_sl, or WL-only run
        lenses, _ = pipeline.forward_lens_selection(
            sources,
            lenses,
            use_flags,
            lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        )

    reduced_chi2 = pipeline.update_chi2_values(
        sources,
        lenses,
        use_flags,
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Forward Selection:", lenses, reduced_chi2)

    # ── Step 5: Merge nearby lenses ──
    merger_threshold = (len(sources.x) / (2 * xmax) ** 2) ** (-0.5) if len(sources.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(
        sources,
        lenses,
        use_flags,
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Merging Lenses:", lenses, reduced_chi2)

    # ── Step 6: Final optimization (strength) ──
    if lens_type == "POWER_LAW":
        lenses = pipeline.optimize_lens_strength(
            sources,
            lenses,
            use_flags,
            lens_type,
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
            sources,
            lenses,
            use_flags,
            lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        )

    reduced_chi2 = pipeline.update_chi2_values(
        sources,
        lenses,
        [True, True, True],
        lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
    )
    log_step("After Final Optimization:", lenses, reduced_chi2)

    return lenses, reduced_chi2
