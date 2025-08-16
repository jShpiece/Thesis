import pipeline

def fit_lensing_field(sources, xmax, flags=False, use_flags=None, lens_type='SIS', z_lens=0.5):
    """
    Reconstructs the gravitational lensing field based on observed source properties.
    
    Parameters:
        sources (Source): Object containing source positions, ellipticity, and flexion signals.
        xmax (float): Maximum field radius for lens consideration.
        flags (bool, optional): If True, prints step-by-step progress. Default is False.
        use_flags (list of bool, optional): Specifies which lensing effects to include 
            [shear, flexion, g_flexion]. Default is [True, True, True].
        lens_type (str, optional): Type of lens model ('SIS' or 'NFW'). Default is 'SIS'.
        z_lens (float, optional): Redshift of the lens. Default is 0.5.

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

    # Step 2: Optimize lens positions
    lenses = pipeline.optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("After Local Minimization:", lenses, reduced_chi2)

    # Step 3: Filter out unsuitable lenses
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("After Filtering:", lenses, reduced_chi2)

    # Step 4: Select optimal lens set
    lenses, _ = pipeline.forward_lens_selection(sources, lenses, use_flags, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("After Forward Selection:", lenses, reduced_chi2)

    # Step 5: Merge nearby lenses
    merger_threshold = (len(sources.x) / (2 * xmax) ** 2) ** (-0.5) if len(sources.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags, lens_type)
    log_step("After Merging Lenses:", lenses, reduced_chi2)

    # Step 6: Final optimization
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, [True, True, True], lens_type)
    log_step("After Final Optimization:", lenses, reduced_chi2)

    return lenses, reduced_chi2