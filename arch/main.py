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
    joint_refine_pass1: bool = True,
    pass2_tolerance_multiplier: float = 10.0,
):
    """
    Reconstructs the gravitational lensing field based on observed
    source properties.

    Parameters
    ----------
    sources : Source
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
        in the post-selection objective.  Default False.
    use_sl_in_selection : bool
        Use strong-lensing information during forward selection via the
        two-pass scheme.  Default False.
    joint_refine_pass1 : bool
        Default True.  When two-pass is active, run joint refinement of
        Pass 1 halos between λ_SL computation and Pass 2.
    pass2_tolerance_multiplier : float
        Default 10.0.  Multiplies the adaptive tolerance during Pass 2
        of two-pass selection only, suppressing marginal additions.
        Higher values are stricter.  Set to 1.0 to recover the pre-
        multiplier behavior.  See forward_selection module docstring
        for the physical rationale.

    Returns
    -------
    lenses, reduced_chi2 : tuple

    Notes
    -----
    Pass 2 tolerance multiplier rationale:
        The base adaptive tolerance scales inversely with candidate
        strength: tau ~ base / strength.  A 10^16 M_sun candidate has
        tau ~ 3e-6, so any nonzero chi^2 improvement passes.  That's
        correct for WL where a massive halo touches many sources, but
        problematic for Pass 2 with SL active: a distant massive halo
        still contributes ~1/r to alpha at SL image positions, so
        small SL chi^2 improvements get rewarded as "easy" because
        the halo is large.

        On A2744 NFW, pass2_tolerance_multiplier=1.0 (legacy) admitted
        a 1.14e16 M_sun halo at (139, 102) — 73 arcsec northeast of
        the refined Pass 1 halo and far outside the SL field — that
        contributed 78% of the recovered total mass but added negligible
        physical information.  pass2_tolerance_multiplier=10.0 makes
        Pass 2 require 10x larger chi^2 improvement per addition,
        suppressing this failure mode.

    See arch.forward_selection module docstring for the full two-pass
    rationale (magnification-correction handling, joint refinement,
    etc.).
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
    post_select_use_mag = True

    if use_strong_lensing and lens_type in ("NFW", "POWER_LAW") and use_sl_in_selection:
        # ── Two-pass path: WL → λ_SL → joint refine → WL+SL ──
        lenses, _, lambda_sl, diag = pipeline.forward_lens_selection_two_pass(
            sources, lenses, use_flags, lens_type,
            return_diagnostics=True,
            joint_refine_pass1=joint_refine_pass1,
            xmax=xmax,
            refine_verbose=flags,
            pass2_tolerance_multiplier=pass2_tolerance_multiplier,
        )
        post_select_use_mag = False
        if flags:
            refine_note = ""
            if diag["joint_refine_pass1"]:
                if diag["refine_applied"]:
                    delta = diag["chi2_refine_before"] - diag["chi2_refine_after"]
                    refine_note = (
                        f", refine: chi^2 {diag['chi2_refine_before']:.3f} -> "
                        f"{diag['chi2_refine_after']:.3f} "
                        f"(-{100.0 * delta / max(diag['chi2_refine_before'], 1e-12):.1f}%)"
                    )
                else:
                    refine_note = ", refine: skipped (no improvement)"
            print(
                f"Two-pass selection:  pass1={diag['n_pass1']} halos"
                f"{refine_note}, "
                f"pass2 added {diag['n_pass2_added']} "
                f"(tol_mult={diag['pass2_tolerance_multiplier']:.1f}), "
                f"lambda_sl={diag['lambda_sl']:.6f}, "
                f"final chi^2={diag['chi2_final']:.3f}"
            )

    elif use_strong_lensing and lens_type in ("NFW", "POWER_LAW"):
        # ── Single-pass path (legacy) ──
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
        lenses, _ = pipeline.forward_lens_selection(
            sources, lenses, use_flags, lens_type,
            use_strong_lensing=use_strong_lensing,
            lambda_sl=lambda_sl,
        )

    reduced_chi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type,
        use_strong_lensing=use_strong_lensing,
        lambda_sl=lambda_sl,
        use_magnification_correction_sl=post_select_use_mag,
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