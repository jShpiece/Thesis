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
import time
import main
import utils
from scipy.optimize import curve_fit

plt.style.use('scientific_presentation.mplstyle')  # Use the scientific presentation style sheet for all plots

# --------------------------------------------
# Plotting Functions
# --------------------------------------------

def plot_results(lens, true_lens, title, reduced_chi2, xmax, ax=None, legend=False, show_mass=False, show_chi2=False):
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
        x = np.array([-20])
        y = np.array([15])
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


def write_plot_name(Nlens, lens_mass, noisy, use_flags, append=None, directory=None):
    # Determine plot naming based on parameters
    size_map = {1e14: 'large', 1e13: 'medium', 1e12: 'small'}
    size = size_map.get(lens_mass, 'other')

    flag_dirs = {
        (True, True, True): 'all',
        (True, True, False): 'shear_f',
        (False, True, True): 'f_g',
        (True, False, True): 'shear_g'
    }

    if directory is None:
        directory = 'standard_tests/'
        directory += flag_dirs.get(tuple(use_flags), 'other')


    # Generate plot name
    plot_name = (f'Output/NFW_tests/{directory}/{size}_Nlens_{Nlens}_{noisy}.png'
                if append is None else f'Output/NFW_tests/standard_tests/{append}.png')
    
    return plot_name

# --------------------------------------------
# Pipeline Alternatives
# --------------------------------------------


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
    axarr = axarr.flatten()
    reduced_chi2 = 1.0

    # Initialize mass_history
    mass_history = []

    # Step 1: Generate initial lens guesses
    lenses = pipeline.generate_initial_guess(sources, lens_type='NFW', z_l=0.194)
    mass_history.append(lenses.mass.copy())
    # lenses = pipeline.alt_generate_initial_guess(sources, xmax, lens_type='NFW')
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Candidate Lens Generation', reduced_chi2, xmax, ax=axarr[0], legend=True)
    if print_steps:
        print('Step 1: Finished initial guesses')

    # Step 2: Optimize lens positions
    lenses = pipeline.optimize_lens_positions(sources, lenses, xmax, use_flags, lens_type='NFW')
    mass_history.append(lenses.mass.copy())
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Individual Lens Optimization', reduced_chi2, xmax, ax=axarr[1])
    if print_steps:
        print('Step 2: Finished optimization')

    # Step 3: Filter lenses by proximity to sources
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type='NFW')
    mass_history.append(lenses.mass.copy())
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Physical Criteria Filtering', reduced_chi2, xmax, ax=axarr[2])
    if print_steps:
        print('Step 3: Finished filtering')

    # Step 4: Iterative lens elimination
    lenses, _ = pipeline.forward_lens_selection(sources, lenses, use_flags, lens_type='NFW')

    if lenses is None:
        fig.suptitle('No Halos Recovered')
        plot_name = write_plot_name(len(true_lenses.x), true_lenses.mass[0], noisy, use_flags, append=name)
        fig.savefig(plot_name)
        plt.close()
        print(f'Plot saved as {plot_name}')
        return lenses
    
    mass_history.append(lenses.mass.copy())
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Forward Lens Selection', reduced_chi2, xmax, ax=axarr[3])
    if print_steps:
        print('Step 4: Finished forward selection')
    

    # Step 5: Merge closely positioned lenses
    area = np.pi * xmax ** 2
    ns = len(sources.x) / area
    merger_threshold = (1 / np.sqrt(ns)) if ns > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold=merger_threshold, lens_type='NFW')
    mass_history.append(lenses.mass.copy())
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Lens Merging', reduced_chi2, xmax, ax=axarr[4], show_mass=True)
    if print_steps:
        print('Step 5: Finished merging')

    # Step 6: Final lens strength optimization
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type='NFW')
    mass_history.append(lenses.mass.copy())
    reduced_chi2 = pipeline.update_chi2_values(sources, lenses, use_flags)
    plot_results(lenses, true_lenses, 'Mass Refinement Optimization', reduced_chi2, xmax, ax=axarr[5], show_mass=True)
    if print_steps:
        print('Step 6: Finished final minimization')\
        
    # Overall figure title
    total_true_mass = np.sum(true_lenses.mass)
    total_recovered_mass = np.sum(lenses.mass)
    fig.suptitle(f'True Mass: {total_true_mass:.2e} $M_\\odot$ \n Recovered Mass: {total_recovered_mass:.2e} $M_\\odot$')

    plot_name = write_plot_name(len(true_lenses.x), true_lenses.mass[0], noisy, use_flags, append=name)

    fig.savefig(plot_name)
    plt.close()
    print(f'Plot saved as {plot_name}')
    return lenses


def run_simple_tests():
    """
    Runs a series of simple tests with varying parameters.
    """

    def simple_test(Nlens, Nsource, xmax, lens_mass, use_noise=True, use_flags=[True, True, True]):
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
        true_lenses, sources, noisy = build_standardized_field(Nlens, Nsource, lens_mass, xmax, use_noise)
        # Randomize the source positions
        xs = np.random.uniform(-xmax, xmax, Nsource)
        ys = np.random.uniform(-xmax, xmax, Nsource)
        sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.full(Nsource, 0.1), np.full(Nsource, 0.01), np.full(Nsource, 0.02))
        sources.apply_lensing(true_lenses, lens_type='NFW', z_source=0.8)
        sources.apply_noise()
        sources.filter_sources()
        pipeline.update_chi2_values(sources, true_lenses, use_flags)
        _ = pipeline_breakdown(sources, true_lenses, xmax, use_flags, noisy, print_steps=True)

    ns = 0.01  # Source density per unit area
    xmax = 50
    area = np.pi * xmax ** 2
    Nsource = int(ns * area)  # Number of sources
    masses = [1e14, 1e13, 1e12]
    lens_numbers = [1, 2 ]
    noise_use = [True]
    # use_flags = [[True, True, False], [True, False, True], [False, True, True], [True, True, True]]
    use_flags = [[True, True, False]]

    for mass in masses:
        for Nlens in lens_numbers:
            for noise in noise_use:
                for flags in use_flags:
                    simple_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=flags)
                # simple_nfw_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=use_flags)


def plot_random_realizations(recovered_params, true_params, title, xmax):
    """
    Plots the recovered lens parameters from random realizations.

    Parameters:
        recovered_params (dict): Dictionary with arrays of the recovered lens parameters (x, y, mass).
        true_params (dict): Dictionary with arrays of the true lens parameters (x, y, mass).
        title (str): Title for the plot.
        xmax (float): Maximum x and y limits for the plot.
    """
    fig, axarr = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(title)

    hist = axarr[0].hist2d(recovered_params['x'], recovered_params['y'], bins=20, cmap='viridis', norm=plt.cm.colors.LogNorm())

    # Add a colorbar
    cbar = plt.colorbar(hist[3], ax=axarr[0])  # hist[3] is the collection of the artists representing the histogram
    cbar.set_label('Counts (Log Scale)')  # Set the label for the colorbar    # Display the colorbar
    
    axarr[0].scatter(true_params['x'], true_params['y'], s=100, c='red', marker='*', label='True Lens', alpha=0.5)
    axarr[0].set_xlabel('X Position')
    axarr[0].set_ylabel('Y Position')
    axarr[0].set_title(f'Position Distribution (Log Scale)')
    axarr[0].set_xlim(-xmax, xmax)
    axarr[0].set_ylim(-xmax, xmax)
    axarr[0].set_aspect('equal')

    # Plot mass distribution
    # Assuming recovered_params['mass'] is defined and imported
    recovered_mass = np.log10(recovered_params['mass'])
    # Remove any value lower than 1e9 (do this for all results)
    small_mass = np.where(recovered_mass < 9)
    xvals = np.delete(recovered_params['x'], small_mass)
    yvals = np.delete(recovered_params['y'], small_mass)
    recovered_mass = recovered_mass[recovered_mass > 9]
    # Now, remove any value that isn't a detection (where it is more than 5 units from the origin)
    r = np.sqrt(xvals ** 2 + yvals ** 2)
    detected = np.where(r < 5)
    detected_mass = recovered_mass[detected]
    # Plot mass distribution
    bin_num = 40
    fancy_hist(recovered_mass, bins=bin_num, histtype='stepfilled', color='blue', alpha=0.5, label='Recovered Mass', ax=axarr[1], density=True)

    # Define Gaussian function
    def gaussian(x, amp, mean, stdev):
        return amp * np.exp(-(x - mean) ** 2 / (2 * stdev ** 2))

    # Define bimodal Gaussian function
    def bimodal_gaussian(x, amp1, mean1, stdev1, amp2, mean2, stdev2):
        return (gaussian(x, amp1, mean1, stdev1) + gaussian(x, amp2, mean2, stdev2))

    # Calculate histogram
    counts, bin_edges = np.histogram(recovered_mass, bins=bin_num, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit bimodal distribution
    # initial_guess = [1, 11.5, 1, 1, 14, 0.5]  # Adjust as necessary
    initial_guess = [1, 13, 1]
    # Try fitting both a Gaussian and a bimodal Gaussian - use the one with the best fit
    try:
        popt1, pcov1 = curve_fit(gaussian, bin_centers, counts, p0=initial_guess)
    except RuntimeError:
        popt1 = [0, 13, 1]
    try:
        popt2, pcov2 = curve_fit(bimodal_gaussian, bin_centers, counts, p0=initial_guess * 2)
    except RuntimeError:
        popt2 = [0, 13, 1, 0, 13, 1]
    # Determine which fit is better
    residuals1 = counts - gaussian(bin_centers, *popt1)
    residuals2 = counts - bimodal_gaussian(bin_centers, *popt2)
    ss_res1 = np.sum(residuals1 ** 2)
    ss_res2 = np.sum(residuals2 ** 2)
    if ss_res1 < ss_res2:
        func = gaussian
        popt = popt1
    else:
        func = bimodal_gaussian
        popt = popt2

    # Plot the fitted Gaussian curves
    # Plot the bimodal Gaussian fit with better formatting for results
    if func == gaussian:
        fit_label = (
            f'Gaussian Fit:\n'
            f'Mean: {popt[1]:.2f} ± {popt[2]:.2f}\n'
        )
    else:
        fit_label = (
            f'Bimodal Gaussian Fit:\n'
            f'Mean 1: {popt[1]:.2f} ± {popt[2]:.2f}\n'
            f'Mean 2: {popt[4]:.2f} ± {popt[5]:.2f}\n'
        )

    axarr[1].plot(bin_centers, func(bin_centers, *popt), 'k--', label=fit_label)
    axarr[1].axvline(np.log10(true_params['mass']), color='red', linestyle='--', label='True Mass')
    axarr[1].axvline(np.mean(detected_mass), color='green', linestyle='--', label='Mean Detected Mass = {:.2f}'.format(np.mean(detected_mass)))
    axarr[1].axvline(np.median(detected_mass), color='purple', linestyle='--', label='Median Detected Mass = {:.2f}'.format(np.median(detected_mass)))

    # Set axis labels and title
    axarr[1].set_xlabel('Recovered Mass (log $M_\\odot$)')
    axarr[1].set_ylabel('Frequency')
    axarr[1].set_title(f'Mass Distribution')

    # Display the legend with improved formatting
    axarr[1].legend(loc='best', fontsize='small', frameon=True)

    fancy_hist(detected_mass, bins=bin_num, histtype='stepfilled', color='blue', alpha=0.5, label='Detected Mass', ax=axarr[2], density=True)
    axarr[2].axvline(np.log10(true_params['mass']), color='red', linestyle='--', label='True Mass')
    axarr[2].set_xlabel('Detected Mass (log $M_\\odot$) - False Positives Removed')
    axarr[2].set_ylabel('Frequency')
    axarr[2].set_title(f'Detected Mass Distribution')
    axarr[2].legend(loc='best', fontsize='small', frameon=True)


    plt.tight_layout()
    plt.savefig(f'Output/NFW_tests/random_realization/{title}.png')
    # plt.show()
    plt.close() # Close the plot to avoid memory issues


def run_random_realizations(Ntrials, Nlenses=1, Nsources=100, xmax=50, lens_mass=1e14, z_l=0.194, use_flags=[True, True, False], 
                            random_seed=None, substructure=False):
    """
    Runs Ntrials random realizations of the lensing pipeline and collects the recovered lens parameters.

    Parameters:
        Ntrials (int): Number of random realizations to run.
        Nlenses (int): Number of lenses to create in each realization. Default is 1.
        Nsources (int): Number of sources to create in each realization. Default is 100.
        xmax (float): Maximum x and y position for the random lenses. Default is 50.
        lens_mass_range (tuple): Minimum and maximum mass for the random lenses. Default is (1e12, 1e15).
        z_l (float): Redshift of the lens. Default is 0.194.
        use_flags (list): Flags indicating which data to use in calculations. Default is [True, True, False].
        random_seed (int): Seed for random number generator. Default is None (no seed).

    Returns:
        dict: A dictionary with arrays of the recovered lens parameters (x, y, mass).
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Initialize arrays to store recovered lens parameters
    recovered_x = []
    recovered_y = []
    recovered_mass = []

    # Use the same true lens for all trials
    # Lenses are distributed uniformly - if there's one lens it's at the origin, if there are two they're in a line, otherwise they're in a grid
    if Nlenses == 1:
        x = np.array([0])
        y = np.array([0])
    elif Nlenses == 2:
        x = np.linspace(-xmax / 2, xmax / 2, Nlenses)
        y = np.array([0, 0])
    else:
        x = np.linspace(-xmax / 2, xmax / 2, Nlenses)
        y = np.linspace(-xmax / 2, xmax / 2, Nlenses)
    mass = np.full(Nlenses, lens_mass)

    if substructure: # Make the first lens the primary lens, all others are substructure
        mass[:1] /= 10
    
    true_lens = halo_obj.NFW_Lens(x, y, np.zeros_like(x), np.zeros(Nlenses), mass, z_l, np.zeros_like(x))
    true_lens.calculate_concentration()
    true_params = {'x': x, 'y': y, 'mass': mass}

    # Initialize progress bar
    utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    # Loop over Ntrials random realizations
    for trial in range(Ntrials):

        # Now create a random distribution of sources (spherical distribution)
        r_s = np.sqrt(np.random.random(Nsources)) * xmax
        theta_s = np.random.random(Nsources) * 2 * np.pi
        xs = r_s * np.cos(theta_s)
        ys = r_s * np.sin(theta_s)
        sig_s = np.full(Nsources, 0.1)
        sig_f = np.full(Nsources, 0.01)
        sig_g = np.full(Nsources, 0.02)
        sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                                    np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
        sources.apply_lensing(true_lens, lens_type='NFW', z_source=0.8)
        sources.apply_noise()
        sources.filter_sources()

        # Run the pipeline on the sources
        lenses, reduced_chi2 = main.fit_lensing_field(sources, xmax, flags=False, use_flags=use_flags, lens_type='NFW')

        # Extract the recovered lens parameters
        recovered_x.extend(lenses.x)
        recovered_y.extend(lenses.y)
        recovered_mass.extend(lenses.mass)

        # Update progress bar
        utils.print_progress_bar(trial + 1, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    # Convert lists to numpy arrays
    recovered_x = np.array(recovered_x)
    recovered_y = np.array(recovered_y)
    recovered_mass = np.array(recovered_mass)
    recovered_params = {'x': recovered_x, 'y': recovered_y, 'mass': recovered_mass}

    # Return the recovered lens parameters
    return recovered_params, true_params


def test_source_sep():
    Ntrials = 100
    Nlenses = 1
    xmax = 50
    lens_mass = 1e14
    ns = 0.01  # Source density per unit area
    area = np.pi * xmax ** 2
    Nsource = int(ns * area)  # Number of sources
    use_flags = [True, True, False]
    true_lenses = halo_obj.NFW_Lens(np.array([0]), np.array([0]), np.zeros(1), np.zeros(1), np.array([lens_mass]), 0.194, np.zeros(1))
    true_lenses.calculate_concentration()
    
    min_lens_source_dist = []
    sources_within_10 = []
    radius_10_sources = []
    success = []
    for trial in range(Ntrials):
        rs = np.sqrt(np.random.random(Nsource)) * xmax
        thetas = np.random.random(Nsource) * 2 * np.pi
        xs = rs * np.cos(thetas)
        ys = rs * np.sin(thetas)
        sig_s = np.full(Nsource, 0.1)
        sig_f = np.full(Nsource, 0.01)
        sig_g = np.full(Nsource, 0.02)
        sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                                    np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
        sources.apply_lensing(true_lenses, lens_type='NFW', z_source=0.8)
        sources.apply_noise()
        sources.filter_sources()
        lenses, reduced_chi2 = main.fit_lensing_field(sources, xmax, flags=False, use_flags=use_flags, lens_type='NFW')
        if len(lenses.x) == 0:
            min_lens_source_dist.append(np.nan)
            sources_within_10.append(np.nan)
            radius_10_sources.append(np.nan)
            success.append(False)
            continue
        # Calculate the minimum distance between the true lens and any source
        source_distances = np.sqrt((sources.x - true_lenses.x[0]) ** 2 + (sources.y - true_lenses.y[0]) ** 2)
        min_lens_source_dist.append(np.min(source_distances))
        # Calculate the number of sources within 10 units of the true lens
        sources_within_10.append(np.sum(source_distances < 10))
        # Calculate the radius from the true lens which contains 10 sources
        radius_10_sources.append(np.sort(source_distances)[9])
        # Check if the test was successful
        total_mass = np.sum(true_lenses.mass)
        recovered_mass = np.sum(lenses.mass)
        largest_mass = np.max(lenses.mass)
        # Determine success
        lens_distances = np.sqrt((lenses.x - true_lenses.x[0]) ** 2 + (lenses.y - true_lenses.y[0]) ** 2)
        closest_lens_dist = np.min(lens_distances)
        closest_mass = lenses.mass[np.argmin(lens_distances)]
        contained_mass = np.sum(lenses.mass[lens_distances < 5])
        success.append((np.abs(true_lenses.mass[0] - closest_mass) / true_lenses.mass[0] < 0.2) and closest_lens_dist < 5 and contained_mass / total_mass > 0.8)
    
    # Convert lists to numpy arrays
    min_lens_source_dist = np.array(min_lens_source_dist)
    sources_within_10 = np.array(sources_within_10)
    radius_10_sources = np.array(radius_10_sources)
    success = np.array(success)

    # Save the results
    np.save('Output/NFW_tests/random_realization/test_results.npy', {'min_dist': min_lens_source_dist, 'within_10': sources_within_10, 'radius_10': radius_10_sources, 'success': success})
    
    results = np.load('Output/NFW_tests/random_realization/test_results.npy', allow_pickle=True).item()
    # Unpack the results
    min_lens_source_dist = results['min_dist']
    sources_within_10 = results['within_10']
    radius_10_sources = results['radius_10']
    success = results['success']

    # Plot the results - split by success and failure
    fig, axarr = plt.subplots(1, 3, figsize=(20, 6))

    # Plot the minimum distance between the true lens and any source
    fancy_hist(min_lens_source_dist[success], bins='freedman', histtype='stepfilled', color='blue', alpha=0.5, label='Successful Tests', ax=axarr[0], density=True)
    fancy_hist(min_lens_source_dist[~success], bins='freedman', histtype='stepfilled', color='red', alpha=0.5, label='Failed Tests', ax=axarr[0], density=True)
    axarr[0].set_xlabel('Minimum Lens-Source Distance')
    axarr[0].set_ylabel('Frequency')
    axarr[0].set_title('Minimum Lens-Source Distance')
    axarr[0].legend()

    # Plot the number of sources within 10 units of the true lens
    fancy_hist(sources_within_10[success], bins='freedman', histtype='stepfilled', color='blue', alpha=0.5, label='Successful Tests', ax=axarr[1], density=True)
    fancy_hist(sources_within_10[~success], bins='freedman', histtype='stepfilled', color='red', alpha=0.5, label='Failed Tests', ax=axarr[1], density=True)
    axarr[1].set_xlabel('Sources Within 10 Units')
    axarr[1].set_ylabel('Frequency')
    axarr[1].set_title('Number of Sources Within 10 arcsec of lens')
    axarr[1].legend()

    # Plot the radius from the true lens which contains 10 sources
    fancy_hist(radius_10_sources[success], bins='freedman', histtype='stepfilled', color='blue', alpha=0.5, label='Successful Tests', ax=axarr[2], density=True)
    fancy_hist(radius_10_sources[~success], bins='freedman', histtype='stepfilled', color='red', alpha=0.5, label='Failed Tests', ax=axarr[2], density=True)
    axarr[2].set_xlabel('Minimum Radius Containing 10 Sources')
    axarr[2].set_ylabel('Frequency')
    axarr[2].set_title('Radius Containing 10 Sources')
    axarr[2].legend()

    plt.tight_layout()
    plt.savefig('Output/NFW_tests/random_realization/source_dist_test.png')


if __name__ == '__main__':
    # run_simple_tests()

    # Create a very simple system
    Nlens = 1
    Nsource = 100
    xmax = 50
    lens_mass = 1e14
    true_lenses, sources, noisy = build_standardized_field(Nlens, Nsource, lens_mass, xmax, use_noise=False)

    # Run a kaiser squires reconstruction
    X, Y, kappa = utils.perform_kaiser_squire_reconstruction(sources, [-xmax, xmax, -xmax, xmax], 'flexion')

    X, Y, kappa_true = utils.calculate_kappa(true_lenses, [-xmax, xmax, -xmax, xmax], lens_type = 'NFW')

    # plot the results
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Kaiser Squires Reconstruction - Flexion')
    cbar = ax[0].imshow(kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='viridis')
    plt.colorbar(cbar, ax=ax[0])
    ax[0].set_title('Kaiser Squires Reconstruction')
    cbar = ax[1].imshow(kappa_true, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='viridis')
    plt.colorbar(cbar, ax=ax[1])
    ax[1].set_title('True Kappa')
    cbar = ax[2].imshow(kappa - kappa_true, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='viridis')
    plt.colorbar(cbar, ax=ax[2])
    ax[2].set_title('Difference')
    plt.tight_layout()
    plt.savefig('Output/NFW_tests/random_realization/kaiser_squires.png')
    plt.show()

    raise SystemExit
    
    Ntrial = 1000
    Nlenses = [1]
    Nsources = 100
    xmax = 50
    lens_mass = [1e14, 1e13]
    z_l = 0.194
    use_flags = [True, True, False]

    for Nlens in Nlenses:
        for mass in lens_mass:
            # results, true_results = run_random_realizations(Ntrial, Nlens, Nsources, xmax, mass, z_l, use_flags=use_flags, random_seed=42)
            name = f'Ntrial_{Ntrial}_Nlens_{Nlens}_mass_{np.log10(mass)}'
            # Save the results
            # np.save(f'Output/NFW_tests/random_realization/{name}.npy', results)
            results = np.load(f'Output/NFW_tests/random_realization/{name}.npy', allow_pickle=True).item()
            x_true = [0]
            y_true = [0]
            mass_true = [mass]
            true_results = {'x': x_true, 'y': y_true, 'mass': mass_true}
            plot_random_realizations(results, true_results, name, xmax)