import pipeline

def fit_lensing_field(sources, xmax, flags = False, use_flags = [True, True, True], lens_type='SIS'):
    '''
    This function takes in a set of sources - with positions, ellipticity, and flexion 
    signals, and attempts to reconstruct the lensing field that produced them. 
    The lensing field is represented by a set of lenses - with positions and Einstein radii. 
    The lenses are modeled as Singular Isothermal Spheres (SIS) by default, but can be
    modeled as NFW halos as well.
    Parameters:
    - sources (Source): An object containing source properties and their uncertainties.
    - xmax (float): The maximum distance from the center of the field to consider for lenses.
    - flags (bool): Whether to print out step information.
    - use_flags (list of bool): Flags indicating which lensing effects to include [use_shear, use_flexion, use_g_flexion].
    - lens_type (str): The type of lens to use - 'SIS' or 'NFW'.
    Returns:
    - lenses (Lens): An object representing the lenses that best fit the source properties.
    - reducedchi2 (float): The reduced chi-squared value for the best fit.
    '''

    def print_step_info(flags,message,lenses,reducedchi2):
        # Helper function to print out step information
        if flags:
            print(message)
            print('Number of lenses: ', len(lenses.x))
            if reducedchi2 is not None:
                print('Chi^2: ', reducedchi2)

    # Initialize candidate lenses from source guesses
    lenses = pipeline.generate_initial_guess(lens_type=lens_type, z_l=0.194)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "Initial Guesses:", lenses, reducedchi2)

    # Optimize lens positions via local minimization
    lenses = pipeline.optimize_lens_positions(sources, lenses, use_flags, lens_type)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "Local Minimization:", lenses, reducedchi2)

    # Filter out lenses that are too close to sources or too far from the center
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type=lens_type)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Filtering:", lenses, reducedchi2)

    # Choose the 'lens_floor' lenses which gives the best reduced chi^2 value
    lenses = pipeline.iterative_elimination(sources, lenses, reducedchi2, use_flags, lens_type)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Iterative Elimination:", lenses, reducedchi2)

    # Merge lenses that are too close to each other
    ns = len(sources.x) / (2 * xmax)**2
    merger_threshold = (1 / ns)**0.5
    lenses = pipeline.merge_close_lenses(merger_threshold=merger_threshold) #This is a placeholder value
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print_step_info(flags, "After Merging:", lenses, reducedchi2)

    # Perform a final minimization on the remaining lenses
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type)
    reducedchi2 = lenses.update_chi2_values(sources, [True, True, True]) # Always use all signals for final fit
    print_step_info(flags, "After Final Minimization:", lenses, reducedchi2)

    return lenses, reducedchi2