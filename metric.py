import numpy as np
import copy
import utils

def calc_degrees_of_freedom(sources, lenses, use_flags):
    # Compute the number of degrees of freedom for a given set of sources and lenses
    # A source has 2 parameters per signal (use_flags tells us how many signals are used)
    # A lens has 3 parameters (two positional, one 'strength')
    dof = ((2 * np.sum(use_flags)) * len(sources.x)) - (3 * len(lenses.x))
    if dof <= 0:
        return np.inf
    return dof


def calculate_chi_squared(sources, lenses, flags, lensing='SIS', use_weights = False) -> float:
    """
    Calculate the chi-squared statistic for the deviation of lensed source properties from their original values,
    considering specified lensing effects and adding penalties for certain lens properties.

    Parameters:
    - sources (Source): An object containing source properties and their uncertainties.
    - lenses (Lenses): An object representing the test lenses which affect the source properties.
    - flags (list of bool): Flags indicating which lensing effects to include [use_shear, use_flexion, use_g_flexion].
    - lensing (str): The lensing model to use ('SIS' or 'NFW').
    - use_weights (bool): Whether to weight the sources by lens-source distance in the chi-squared calculation.

    Returns:
    - float: The total chi-squared value including penalties.
    """

    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = flags

    # Initialize a clone of sources with zeroed lensing signals
    source_clone = copy.deepcopy(sources)
    source_clone.zero_lensing_signals()

    assert np.all(source_clone.e1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.e2 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.f1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.f2 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.g1 == 0), "The cloned source should have zeroed lensing signals."
    assert np.all(source_clone.g2 == 0), "The cloned source should have zeroed lensing signals."

    # Apply lensing effects to the cloned source
    if lensing == 'SIS':
        source_clone.apply_SIS_lensing(lenses)
    elif lensing == 'NFW':
        source_clone.apply_NFW_lensing(lenses)
    else:
        raise ValueError("Invalid lensing model: {}".format(lensing))

    # Calculate chi-squared for each lensing signal component
    chi_squared_components = {
        'shear': ((source_clone.e1 - sources.e1) ** 2 + (source_clone.e2 - sources.e2) ** 2) / sources.sigs**2,
        'flexion': ((source_clone.f1 - sources.f1) ** 2 + (source_clone.f2 - sources.f2) ** 2) / sources.sigf**2,
        'g_flexion': ((source_clone.g1 - sources.g1) ** 2 + (source_clone.g2 - sources.g2) ** 2) / sources.sigg**2
    }

    # Sum the chi-squared values, considering only the enabled lensing effects
    total_chi_squared = use_shear * chi_squared_components['shear'] + use_flexion * chi_squared_components['flexion'] + use_g_flexion * chi_squared_components['g_flexion']    

    if use_weights:
        weights = utils.compute_source_weights(lenses, sources)
        total_chi_squared = np.sum(weights * total_chi_squared)
    else:
        total_chi_squared = np.sum(total_chi_squared)

    def eR_penalty_function(eR, limit=40.0, lambda_penalty_upper=1000.0):
        # Soft limits - allow the Einstein radius to be negative
        if np.abs(eR) > limit:
            return lambda_penalty_upper * (np.abs(eR) - limit) ** 2

        return 0.0

    # Calculate and add penalties for the lenses
    if lensing == 'SIS':
        penalty = sum(eR_penalty_function(eR) for eR in lenses.te)
        total_chi_squared += penalty
    elif lensing == 'NFW':
        pass

    # Return the total chi-squared including penalties
    return total_chi_squared 

