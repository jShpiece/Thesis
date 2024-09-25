"""
Module for testing the NFW lensing pipeline.

This module contains functions to test the gravitational lensing pipeline using NFW (Navarro-Frenk-White) lens models.
It includes functions for plotting results, breaking down the pipeline step by step, and running simple tests.

Functions:
    - plot_results
    - build_standardized_field
    - pipeline_breakdown
    - simple_nfw_test
    - run_simple_tests
    - test_mass_recovery_with_noise
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import hist as fancy_hist
import pipeline
import halo_obj
import source_obj
import utils
import time
import copy
import metric

plt.style.use('scientific_presentation.mplstyle')  # Use the scientific presentation style sheet for all plots


# --------------------------------------------
# Plotting Functions
# --------------------------------------------

def plot_results(lens, true_lens, title, reduced_chi2, xmax, ax=None, legend=True, show_mass=False, show_chi2=False):
    """
    Plots the comparison between reconstructed lenses and true lenses.

    Parameters:
        lens (NFW_Lens): Reconstructed lens object.
        true_lens (NFW_Lens): True lens object used in the simulation.
        title (str): Title for the plot.
        reduced_chi2 (float): Reduced chi-squared value.
        xmax (float): Maximum x and y limits for the plot.
        ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, creates a new figure.
        legend (bool, optional): Whether to display the legend. Default is True.
        show_mass (bool, optional): Whether to display mass labels on the plot. Default is False.
        show_chi2 (bool, optional): Whether to display chi-squared values on the plot. Default is False.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # If no lenses are recovered, plot only the true lenses
    if len(lens.x) == 0:
        x_true, y_true, mass_true = true_lens.x, true_lens.y, true_lens.mass
        mass_true_log = np.log10(mass_true)
        true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 50
        ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=0.8, label='True Halos', edgecolors='w', marker='*')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f"{title}\nNo Halos Recovered")
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect('equal')
        if legend:
            ax.legend()
        return

    # Extract positions and masses for true and reconstructed lenses
    x_true, y_true, mass_true = true_lens.x, true_lens.y, true_lens.mass
    x_recon, y_recon, mass_recon = lens.x, lens.y, lens.mass

    # Convert masses to logarithmic scale for better visualization
    mass_true_log = np.log10(mass_true)
    mass_recon_log = np.log10(np.abs(mass_recon) + 1e-10)  # Add small value to avoid log(0)

    # Plot true lenses with distinct markers and sizes
    true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 200  # Scale sizes
    ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=1, label='True Halos', edgecolors='w', marker='*')

    # Plot reconstructed lenses with distinct markers and sizes
    recon_sizes = (mass_recon_log - np.min(mass_recon_log) + 1) * 50  # Scale sizes
    ax.scatter(x_recon, y_recon, s=recon_sizes, c='red', alpha=0.3, label='Recovered Halos', edgecolors='k', marker='o')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f"{title}\nReduced Chi-Squared: {reduced_chi2:.5f}")
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    if legend:
        ax.legend()

    # Optionally display mass or chi-squared values
    if show_mass:
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], f'{mass_recon[i]:.2e}', fontsize=12, color='black')

    if show_chi2:
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], f'{lens.chi2[i]:.2f}', fontsize=12, color='black')


# --------------------------------------------
# Helper Functions
# --------------------------------------------

def build_standardized_field(Nlens, Nsource, lens_mass, xmax, use_noise=False):
    """
    Builds a standardized field with specified parameters.

    Parameters:
        Nlens (int): Number of lenses.
        Nsource (int): Number of sources.
        lens_mass (float): Mass of the lenses.
        xmax (float): Maximum x and y limits for the field.
        use_noise (bool, optional): Whether to add noise to the sources. Default is False.

    Returns:
        tuple: (lenses, sources, noisy) where:
            - lenses (NFW_Lens): The true lens object.
            - sources (Source): The source object with lensing applied.
            - noisy (str): Indicator of whether noise was applied ('noisy' or 'noiseless').
    """
    # Create lens positions
    if Nlens == 1:
        x = np.array([0])
        y = np.array([0])
    elif Nlens == 2:
        x = np.linspace(-xmax / 2, xmax / 2, Nlens)
        y = np.array([0, 0])
    else:
        x = np.linspace(-xmax / 2, xmax / 2, Nlens)
        y = np.linspace(-xmax / 2, xmax / 2, Nlens)
    mass = np.full(Nlens, lens_mass)

    lenses = halo_obj.NFW_Lens(x, y, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
    lenses.calculate_concentration()

    # Create source positions
    if Nsource == 1:
        xs = np.random.uniform(-xmax, xmax, Nsource)
        ys = np.random.uniform(-xmax, xmax, Nsource)
    else:
        n = int(np.sqrt(Nsource))
        xs = np.linspace(-xmax, xmax, n)
        ys = np.linspace(-xmax, xmax, n)
        xs, ys = np.meshgrid(xs, ys)
        xs = xs.flatten()
        ys = ys.flatten()
        Nsource = len(xs)

    sig_s = np.full(Nsource, 0.1)
    sig_f = np.full(Nsource, 0.01)
    sig_g = np.full(Nsource, 0.02)
    sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                                np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)

    if use_noise:
        sources.apply_noise()
        noisy = 'noisy'
    else:
        noisy = 'noiseless'

    sources.apply_lensing(lenses, lens_type='NFW', z_source=0.8)
    sources.filter_sources()
    return lenses, sources, noisy


# --------------------------------------------
# Testing Functions
# --------------------------------------------

def pipeline_breakdown(sources, true_lenses, xmax, use_flags, noisy, name=None, print_steps=False):
    """
    Runs the pipeline step by step and visualizes the results at each step.

    Parameters:
        sources (Source): Source object containing source properties.
        true_lenses (NFW_Lens): True lens object used in the simulation.
        xmax (float): Maximum x and y limits for the plots.
        use_flags (list of bool): Flags indicating which lensing signals to use.
        noisy (str): Indicator of whether noise was applied ('noisy' or 'noiseless').
        name (str, optional): Custom name for saving the plot. If None, a default name is generated.
        print_steps (bool, optional): Whether to print progress messages. Default is False.
    """
    # Set up plot with 6 subplots in a 2x3 layout
    fig, axarr = plt.subplots(2, 3, figsize=(20, 15), sharex=True, sharey=True)

    # Step 1: Generate initial lens guesses
    lenses = pipeline.generate_initial_guess(sources, lens_type='NFW', z_l=0.194)
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Candidate Lens Generation', reduced_chi2, xmax, ax=axarr[0, 0], legend=True)
    if print_steps:
        print('Step 1: Finished initial guesses')

    # Step 2: Optimize lens positions
    lenses = pipeline.optimize_lens_positions(sources, lenses, use_flags, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Individual Lens Optimization', reduced_chi2, xmax, ax=axarr[0, 1])
    if print_steps:
        print('Step 2: Finished optimization')

    # Step 3: Filter lenses by proximity to sources
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Physical Criteria Filtering', reduced_chi2, xmax, ax=axarr[0, 2])
    if print_steps:
        print('Step 3: Finished filtering')

    # Step 4: Iterative lens elimination
    lenses, _ = pipeline.select_best_lenses_forward_selection(sources, lenses, use_flags, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Forward Lens Selection', reduced_chi2, xmax, ax=axarr[1, 0], show_chi2=True)
    if print_steps:
        print('Step 4: Finished iterative elimination')

    # Step 5: Merge closely positioned lenses
    area = np.pi * xmax ** 2
    ns = len(sources.x) / area
    merger_threshold = (1 / np.sqrt(ns)) if ns > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold=merger_threshold, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Lens Merging', reduced_chi2, xmax, ax=axarr[1, 1])
    if print_steps:
        print('Step 5: Finished merging')

    # Step 6: Final lens strength optimization
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Mass Refinement Optimization', reduced_chi2, xmax, ax=axarr[1, 2], show_mass=True)
    if print_steps:
        print('Step 6: Finished final minimization')

    # Overall figure title
    total_true_mass = np.sum(true_lenses.mass)
    total_recovered_mass = np.sum(lenses.mass)
    fig.suptitle(f'True Mass: {total_true_mass:.2e} $M_\\odot$ \n Recovered Mass: {total_recovered_mass:.2e} $M_\\odot$')

    # Determine plot naming based on parameters
    size_map = {1e14: 'large', 1e13: 'medium', 1e12: 'small'}
    size = size_map.get(true_lenses.mass[0], 'other')

    flag_dirs = {
        (True, True, True): 'all',
        (True, True, False): 'shear_f',
        (False, True, True): 'f_g',
        (True, False, True): 'shear_g'
    }
    directory = flag_dirs.get(tuple(use_flags), 'other')

    Nlens = len(true_lenses.x)

    # Generate plot name
    plot_name = (f'Images/NFW_tests/standard_tests/{directory}/{size}_Nlens_{Nlens}_{noisy}.png'
                if name is None else f'Images/NFW_tests/standard_tests/{name}.png')

    fig.savefig(plot_name)
    plt.close()
    
    return lenses


def simple_nfw_test(Nlens, Nsource, xmax, lens_mass, use_noise=True, use_flags=[True, True, True]):
    """
    Creates a simple lensing field and tests the pipeline on it.

    Parameters:
        Nlens (int): Number of lenses to simulate.
        Nsource (int): Number of sources to simulate.
        xmax (float): Maximum x and y limits for the field.
        lens_mass (float): Mass of the lens.
        use_noise (bool, optional): Whether to add noise to the sources. Default is True.
        use_flags (list of bool, optional): Flags indicating which lensing signals to use. Default is [True, True, True].
    """
    start = time.time()
    true_lenses, sources, noisy = build_standardized_field(Nlens, Nsource, lens_mass, xmax, use_noise)
    pipeline.update_chi2_values(sources, true_lenses, use_flags)
    optimized_lenses = pipeline_breakdown(sources, true_lenses, xmax, use_flags, noisy, print_steps=True)
    end = time.time()
    print(f'Test complete - Time taken: {end - start:.2f} seconds')


def run_simple_tests():
    """
    Runs a series of simple tests with varying parameters.
    """
    ns = 0.01  # Source density per unit area
    xmax = 50
    area = np.pi * xmax ** 2
    # Nsource = int(ns * area)  # Number of sources
    Nsource = 100
    masses = [1e14, 1e13, 1e12]
    lens_numbers = [1, 2]
    noise_use = [True]
    # use_flags = [[True, True, False], [True, False, True], [False, True, True], [True, True, True]]
    use_flags = [[True, True, True], [True, True, False]]

    for mass in masses:
        for Nlens in lens_numbers:
            for noise in noise_use:
                for flags in use_flags:
                    simple_nfw_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=flags)
                # simple_nfw_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=use_flags)


def test_mass_recovery_with_noise(lenses, sources, Ntrials=1000):
    """
    Tests mass recovery by optimizing lens strength with noisy sources.

    Parameters:
        lenses (NFW_Lens): True lens object used in the simulation.
        sources (Source): Source object containing source properties.
        Ntrials (int): Number of trials to run. Default is 1000.
    """
    masses = []

    # Initialize progress bar
    utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    for n in range(Ntrials):
        # Clone the source object, apply noise
        source_clone = copy.deepcopy(sources)
        source_clone.apply_noise()

        # Create a lens object with correct position and concentration
        lens = copy.deepcopy(lenses)
        # Randomize the starting mass
        starting_mass = np.random.normal(1e14, 1e12)
        lens.mass[0] = starting_mass

        # Run the final minimization
        lens = pipeline.optimize_lens_strength(source_clone, lens, [True, True, False], lens_type='NFW')
        masses.append(lens.mass[0])

        # Update the progress bar
        utils.print_progress_bar(n+1, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    # Convert masses to log scale
    masses_log = np.log10(np.array(masses))

    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 10))
    fancy_hist(masses_log, ax=ax, bins='freedman', color='black', histtype='step', density=True, label='No Perturbation', linestyle='-')
    ax.set_title('Optimized Mass - One Source')
    ax.axvline(np.log10(1e14), color='black', linestyle='--', label='True Mass')
    ax.axvline(np.mean(masses_log), color='red', linestyle='--', label='Mean Mass')
    ax.legend()
    plt.tight_layout()
    plt.show()


def str_opt_tester(Nlens, Nsource, lens_mass, xmax, use_noise=False, perturb_loc=False, perturb_sig=False):
    """
    Test the strength optimization pipeline.

    Parameters:
        Nlens (int): Number of lenses.
        Nsource (int): Number of sources.
        lens_mass (float): Mass of the lenses.
        xmax (float): Maximum x and y limits for the field.
        use_noise (bool, optional): Whether to add noise to the sources. Default is False.
    """
    true_lens, sources, xmax = build_standardized_field(Nlens, Nsource, lens_mass, xmax, False)
    
    # Randomly place a single source
    xs = np.array([10.0])
    ys = np.array([0.0])
    sigs = np.array([0.1])
    sigf = np.array([0.01])
    sigg = np.array([0.02])
    sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sigs, sigf, sigg)
    sources.apply_lensing(true_lens, lens_type='NFW', z_source=0.8)
    
    masses = []
    perturbation_1 = []
    perturbation_2 = []
    perturbation_3 = []
    perturbation_4 = []

    def perturb_source_signals(target_source, shear_dir, flex_dir):
        if shear_dir == 'plus':
            target_source.e1 *= 1.05
            # target_source.e2 *= 1.05
        elif shear_dir == 'minus':
            target_source.e1 *= 0.95
            # target_source.e2 *= 0.95
        else:
            raise ValueError('Invalid shear direction specified')
        
        if flex_dir == 'plus':
            target_source.f1 *= 1.05
            # target_source.f2 *= 1.05
        elif flex_dir == 'minus':
            target_source.f1 *= 0.95
            # target_source.f2 *= 0.95
        else:
            raise ValueError('Invalid flexion direction specified')
        return target_source

    Ntrials = 1000
    # Initialize progress bar
    utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    for n in range(Ntrials):
        # Clone the source object, otherwise we add noise to the same object each time
        source_clone = copy.deepcopy(sources)
        source_clone.apply_noise()

        # Create a lens object with the exactly correct position and concentration - this assumes that we perfectly located the halo, and the only thing we need to do is to calculate the mass
        lens = copy.deepcopy(true_lens)
        # Randomize the mass
        starting_mass = np.random.normal(1e14, 1e13, 1)
        lens.mass[0] = starting_mass[0]

        if perturb_loc:
            # Randomly perturb the lens location by a *small* amount
            lens.x[0] += np.random.normal(0, 0.1, 1)
            lens.y[0] += np.random.normal(0, 0.1, 1)

        # Run the final minimization
        lens = pipeline.optimize_lens_strength(source_clone, lens, [True, True, False], lens_type='NFW')
        masses.append((lens.mass))

        # Now, clone the source object and perturb it in different ways
        perturbations = [['plus', 'plus'], ['plus', 'minus'], ['minus', 'plus'], ['minus', 'minus']]
        results = [perturbation_1, perturbation_2, perturbation_3, perturbation_4]
        
        if perturb_sig:
            for perturbation in perturbations:
                source_clone_perturbation = copy.deepcopy(source_clone)
                source_clone_perturbation = perturb_source_signals(source_clone_perturbation, perturbation[0], perturbation[1])
                lens_clone = copy.deepcopy(lens)
                lens_clone.mass[0] = starting_mass[0]
                lens_clone = pipeline.optimize_lens_strength(source_clone_perturbation, lens_clone, [True, True, False], lens_type='NFW')
                results[perturbations.index(perturbation)].append((lens_clone.mass))
        
        # Update the progress bar
        utils.print_progress_bar(n+1, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    # Turn lists into numpy arrays, convert to log space
    masses = np.log10(np.array(masses))
    perturbations = [np.log10(np.array(perturbation)) for perturbation in [perturbation_1, perturbation_2, perturbation_3, perturbation_4]]
    # Check that each perturbation has the same length
    assert len(perturbation_1) == len(perturbation_2) == len(perturbation_3) == len(perturbation_4)

    if perturb_sig:
        # Plot the results
        fig, ax = plt.subplots(5, 1, figsize=(10, 20))
        ax = ax.flatten()
        fancy_hist(masses, ax=ax[0], bins='freedman', color='black', histtype='step', density=True)
        ax[0].axvline(np.log10(1e14), color='black', linestyle='--', label='True Mass = {:.2e}'.format(1e14))
        ax[0].axvline(np.mean(masses), color='red', linestyle='--', label='Mean Mass = {:.2e}'.format(10**np.mean(masses)))
        ax[0].set_title('No Perturbation')
        titles = ['Shear +, Flexion +', 'Shear +, Flexion -', 'Shear -, Flexion +', 'Shear -, Flexion -']
        for i in range(4):
            fancy_hist(masses, ax=ax[i+1], bins='freedman', color='black', histtype='step', density=True, label = 'No Perturbation', linestyle='-')
            fancy_hist(perturbations[i], ax=ax[i+1], bins='freedman', color='red', histtype='step', density=True, label = 'Perturbed', linestyle='--')
            ax[i+1].axvline(np.log10(1e14), color='black', linestyle='--', label='True Mass = {:.2e}'.format(1e14))
            ax[i+1].axvline(np.mean(masses), color='red', linestyle='--', label='Mean Mass = {:.2e}'.format(10**np.mean(masses)))
            ax[i+1].axvline(np.mean(perturbations[i]), color='blue', linestyle='--', label='Mean Mass = {:.2e}'.format(10**np.mean(perturbations[i])))
            ax[i+1].set_title(titles[i])
            ax[i+1].legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fancy_hist(masses, ax=ax, bins='freedman', color='black', histtype='step', density=True, label = 'No Perturbation', linestyle='-')
        ax.set_title('Optimized Mass - One Source - Imperfect Lens Location')
        ax.axvline(np.log10(1e14), color='black', linestyle='--', label='True Mass = {:.2e}'.format(1e14))
        ax.axvline(np.mean(masses), color='red', linestyle='--', label='Mean Mass = {:.2e}'.format(10**np.mean(masses)))
        ax.legend()
        plt.tight_layout()
        plt.savefig('Images/NFW_tests/strength_opt/one_source_mass_imperfect_location.png')


def one_source_optimization(Nlens, Nsource, Ntrials, lens_mass, xmax, perturb_loc=False):
    true_lens, sources, _ = build_standardized_field(Nlens, Nsource, lens_mass, xmax, False)
    
    Nsources = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    
    for N in Nsources:
        # Arrange the sources uniformly around the lens
        # r = 10
        # theta = np.linspace(0, 2*np.pi, Nsource)
        # Arrange sources randomly (spherically symmetric)
        r = np.sqrt(np.random.random(N)) * xmax
        theta = np.random.uniform(0, 2*np.pi, N)
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)

        sigs = np.ones_like(xs) * 0.1
        sigf = np.ones_like(xs) * 0.01
        sigg = np.ones_like(xs) * 0.02
        sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sigs, sigf, sigg)
        sources.apply_lensing(true_lens, lens_type='NFW', z_source=0.8)
        
        starting_masses = np.logspace(13, 15, Ntrials)

        
        # Initialize progress bar
        utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
        final_masses = []

        # Initialize progress bar
        for i, starting_mass in enumerate(starting_masses):
            # Clone the source object, otherwise we add noise to the same object each time
            source_clone = copy.deepcopy(sources)
            source_clone.apply_noise()

            # Create a lens object with the exactly correct position and concentration - this assumes that we perfectly located the halo, and the only thing we need to do is to calculate the mass
            lens = copy.deepcopy(true_lens)
            # Randomize the mass
            lens.mass[0] = starting_mass
            if perturb_loc:
                # Randomly perturb the lens location by a *small* amount
                lens.x[0] += np.random.normal(0, 0.5, 1)
                lens.y[0] += np.random.normal(0, 0.5, 1)

            # Run the final minimization
            # lens = pipeline.optimize_lens_strength(source_clone, lens, [True, True, False], lens_type='NFW')
            lens = pipeline.optimize_nfw_lens_strength(source_clone, lens, [True, True, False])
            final_masses.append(lens.mass[0])

            # Update the progress bar
            utils.print_progress_bar(i+1, len(starting_masses), prefix='Progress:', suffix='Complete', length=50)
        

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(10, 20))
        ax.plot(starting_masses, final_masses, color='black', linestyle='-', label='Reduced Chi-Squared')
        ax.set_title('Optimized Mass (Perturbed Loc) - Random Placement - Nsource = {}'.format(N))
        ax.set_xlabel('Starting Mass')
        ax.set_ylabel('Final Mass')
        ax.axhline(1e14, color='red', linestyle='--', label='True Mass')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.tight_layout()
        title = 'Images/NFW_tests/strength_opt/Nsource_{}.png'.format(N)
        if perturb_loc:
            title = 'Images/NFW_tests/strength_opt/Nsource_{}_imperfect_location_random.png'.format(N)
        plt.savefig(title)
        print('Finished Nsource = {}'.format(N))


def test_iter_elim():
    # Create a standardized field
    Nlens = 1
    Nsource = 100
    lens_mass = 1e14
    xmax = 50
    use_flags = [True, True, True]
    true_lens, sources, _ = build_standardized_field(Nlens, Nsource, lens_mass, xmax, True)
    
    # Create a bunch of random lenses
    Nlenses = 100
    xl = np.random.uniform(-xmax, xmax, Nlenses)
    yl = np.random.uniform(-xmax, xmax, Nlenses)
    ml = 10**np.random.uniform(12, 15, Nlenses)
    lenses = halo_obj.NFW_Lens(xl, yl, np.zeros_like(xl), np.zeros(Nlenses), ml, 0.194, np.zeros_like(xl))
    lenses.calculate_concentration()
    # Add the true lens
    lenses.merge(true_lens)
    
    lenses, reduced_chi2 = pipeline.select_best_lenses_forward_selection(sources, lenses, use_flags=use_flags, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)

    # Plot the results
    plot_results(lenses, true_lens, 'Number Selection', reduced_chi2, xmax, show_mass=True)
    plt.show()



if __name__ == '__main__':
    start = time.time()
    # str_opt_tester(1, 1, 1e14, 50, perturb_loc=True, perturb_sig=True)
    # one_source_optimization(1, 1, 100, 1e14, 50, perturb_loc=True)
    # test_iter_elim()
    run_simple_tests()
    end = time.time()
    print(f'Test complete - Time taken: {end - start:.2f} seconds')