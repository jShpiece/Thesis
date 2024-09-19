import numpy as np 
import matplotlib.pyplot as plt
from astropy.visualization import hist as fancy_hist
import pipeline
import halo_obj
import source_obj
import utils
import time
import copy

plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

# --------------------------------------------
# Plotting Functions
# --------------------------------------------

def _plot_results(halo, true_halo, title, reducedchi2, xmax, ax=None, legend=True, show_mass = False, show_chi2 = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # If the halo is empty, plot the true halo only
    if len(halo.x) == 0:
        x_true, y_true, mass_true = [true_halo.x, true_halo.y, true_halo.mass]
        mass_true_log = np.log10(mass_true)
        true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 50
        ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=0.8, label='True Halos', edgecolors='w', marker='*')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title + '\n' + 'No Halos Recovered')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect('equal')
        if legend:
            ax.legend()
        return

    # Extract positions and masses for both sets
    x_true, y_true, mass_true, chi2_true = [true_halo.x, true_halo.y, true_halo.mass, true_halo.chi2]
    x_recon, y_recon, mass_recon, chi2_recon = [halo.x, halo.y, halo.mass, halo.chi2]

    # Normalize the masses for better visualization in a logarithmic range
    mass_true_log = np.log10(mass_true)
    mass_recon_log = np.log10(np.abs(mass_recon) + 1e-10)  # Add a small value to avoid log(0)

    # Plot true properties with distinct markers and sizes
    true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 200  # Scale sizes
    ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=1, label='True Halos', edgecolors='w', marker='*')

    # Plot reconstructed properties with distinct markers and sizes
    recon_sizes = (mass_recon_log - np.min(mass_recon_log) + 1) * 50  # Scale sizes
    ax.scatter(x_recon, y_recon, s=recon_sizes, c='red', alpha=0.3, label='Recovered Halos', edgecolors='k', marker='o')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.5f}'.format(reducedchi2))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    if legend:
        ax.legend()
    
    if show_mass:
    # Label the mass of each of the lenses (in log scale)
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], '{:.2e}'.format(mass_recon[i]), fontsize=12, color='black')
    
    if show_chi2:
        # Label the chi2 value of each of the lenses
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], '{:.2f}'.format(chi2_recon[i]), fontsize=12, color='black')
        # Also label the halos
        '''
        for i in range(len(x_true)):
            ax.text(x_true[i], y_true[i], '{:.2f}'.format(chi2_true[i]), fontsize=12, color='black')
        '''


# --------------------------------------------
# Helper Functions
# --------------------------------------------

def pipeline_breakdown(halos, sources, xmax, use_flags, noisy, name=None, print_steps=False):
    '''This function runs the pipeline step by step, visualizing the results at each step'''

    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(20, 15), sharex=True, sharey=True)

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess(z_l = 0.194, lens_type='NFW')
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Initial Guesses', reducedchi2, xmax, ax=axarr[0,0], legend=True)
    if print_steps:
        print('Finished initial guesses')
    
    # Step 2: Optimize guesses with local minimization
    lenses = pipeline.optimize_lens_positions(sources, lenses, use_flags, lens_type='NFW')
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Initial Optimization', reducedchi2, xmax, ax=axarr[0,1], legend=False)
    if print_steps:
        print('Finished optimization')

    # Step 3: Filter out lenses that are too far from the source population
    lenses = pipeline.filter_lens_positions(sources, lenses, xmax, lens_type='NFW')
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Filtering', reducedchi2, xmax, ax=axarr[0,2], legend=False)
    if print_steps:
        print('Finished filtering')

    # Step 4: Iterative elimination
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Lens Number Selection', reducedchi2, xmax, ax=axarr[1,0], legend=False)
    if print_steps:
        print('Finished iterative elimination')

    # Step 5: Merge lenses that are too close to each other
    ns = len(sources.x) / (np.pi * xmax**2)
    merger_threshold = (1/np.sqrt(ns))
    lenses.merge_close_lenses(merger_threshold=merger_threshold)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Merging', reducedchi2, xmax, ax=axarr[1,1], legend=False, show_chi2=True)
    if print_steps:
        print('Finished merging')

    # Step 6: Final minimization
    lenses = pipeline.optimize_lens_strength(sources, lenses, use_flags, lens_type='NFW')
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Final Minimization', reducedchi2, xmax, ax=axarr[1,2], legend=False, show_mass=True)
    if print_steps:
        print('Finished final minimization')

    fig.suptitle('True Mass: {:.2e} $M_\odot$ \n Recovered Mass: {:.2e} $M_\odot$'.format(np.sum(halos.mass), np.sum(lenses.mass)))
    halo_mass = halos.mass[0]

    if halo_mass == 1e14:
        size = 'large'
    elif halo_mass == 1e13:
        size = 'medium'
    elif halo_mass == 1e12:
        size = 'small'
    else:
        size = 'other'
    
    if use_flags == [True, True, True]:
        directory = 'all'
    elif use_flags == [True, True, False]:
        directory = 'shear_f'
    elif use_flags == [False, True, True]:
        directory = 'f_g'
    elif use_flags == [True, False, True]:
        directory = 'shear_g'
    Nlens = len(halos.x)

    if name == None:
        plot_name = 'Images/NFW_tests/standard_tests/{}/{}_Nlens_{}_{}.png'.format(directory, size, Nlens,noisy)
    else:
        plot_name = 'Images/NFW_tests/standard_tests/{}.png'.format(name)
    fig.savefig(plot_name)
    plt.close()

# --------------------------------------------
# Testing Functions
# --------------------------------------------

def simple_nfw_test(Nlens, Nsource, xmax, halo_mass, use_noise=True, use_flags=[True, True, True]):
    # Create a simple lensing field and test the pipeline on it
    start = time.time()
    halos, sources, noisy = utils.build_standardized_field(Nlens, Nsource, halo_mass, xmax, use_noise)
    halos.update_chi2_values(sources, use_flags)
    pipeline_breakdown(halos, sources, xmax, use_flags, noisy)
    stop = time.time()
    print('Test complete - Time taken: {}'.format(stop - start))
    return


def run_simple_tests():
    ns = 0.01
    xmax = 50
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    masses = [1e14, 1e13]
    lens_numbers = [1, 2]
    noise_use = [True, False]
    use_flags_choices = [[True, True, True], [True, True, False], [False, True, True], [True, False, True]]

    for mass in masses:
        for Nlens in lens_numbers:
            for noise in noise_use:
                # for use_flags in use_flags_choices:
                simple_nfw_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=[True, True, True])


def build_standardized_field(Nlens, Nsource, halo_mass, xmax, use_noise=False):
    # Create a set of lenses
    if Nlens == 1:
        x = np.array([0])
        y = np.array([0])
    elif Nlens == 2:
        x = np.linspace(-xmax/2, xmax/2, Nlens)
        y = np.array([0, 0])
    else:
        x = np.linspace(-xmax/2, xmax/2, Nlens)
        y = np.linspace(-xmax/2, xmax/2, Nlens)
    mass = np.ones(Nlens) * halo_mass

    halos = halo_obj.NFW_Lens(x, y, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
    halos.calculate_concentration()

    
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
    
    sig_s = np.ones(Nsource) * 0.1
    sig_f = np.ones(Nsource) * 0.01
    sig_g = np.ones(Nsource) * 0.02
    sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    if use_noise:
        sources.apply_noise()
        noisy = 'noisy'
    else:
        noisy = 'noiseless'
    sources.apply_NFW_lensing(halos)
    sources.filter_sources()
    return halos, sources, noisy


if __name__ == '__main__':
    halo_obj, sources, xmax = build_standardized_field(1, 100, 1e14, 50, False)
    
    # Randomly place a single source
    xmax = 20
    xs = np.array([10.0])
    ys = np.array([0.0])
    sigs = np.array([0.1])
    sigf = np.array([0.01])
    sigg = np.array([0.02])
    sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sigs, sigf, sigg)
    sources.apply_NFW_lensing(halo_obj)
    
    masses = []
    perturbation_1 = []
    perturbation_2 = []
    perturbation_3 = []
    perturbation_4 = []

    def perturb_source(target_source, shear_dir, flex_dir):
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

    Ntrials = 10
    # Initialize progress bar
    utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    for n in range(Ntrials):
        # Clone the source object, otherwise we add noise to the same object each time
        source_clone = copy.deepcopy(sources)
        source_clone.apply_noise()

        # Create a lens object with the exactly correct position and concentration - this assumes that we perfectly located the halo, and the only thing we need to do is to calculate the mass
        lens = copy.deepcopy(halo_obj)
        # Randomize the mass
        starting_mass = np.random.normal(1e14, 1e13, 1)
        lens.mass[0] = starting_mass[0]

        # Run the final minimization
        lens = pipeline.optimize_lens_strength(source_clone, lens, [True, True, False], lens_type='NFW')
        masses.append((lens.mass))

        # Now, clone the source object and perturb it in different ways
        perturbations = [['plus', 'plus'], ['plus', 'minus'], ['minus', 'plus'], ['minus', 'minus']]
        results = [perturbation_1, perturbation_2, perturbation_3, perturbation_4]
        for perturbation in perturbations:
            source_clone_perturbation = copy.deepcopy(source_clone)
            source_clone_perturbation = perturb_source(source_clone_perturbation, perturbation[0], perturbation[1])
            lens_clone = copy.deepcopy(halo_obj)
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

    # Plot the results
    fig, ax = plt.subplots(5, 1, figsize=(10, 20))
    ax = ax.flatten()
    fancy_hist(masses, ax=ax[0], bins='freedman', color='black', histtype='step', density=True)
    ax[0].set_title('No Perturbation')
    titles = ['Shear +, Flexion +', 'Shear +, Flexion -', 'Shear -, Flexion +', 'Shear -, Flexion -']
    for i in range(4):
        fancy_hist(masses, ax=ax[i+1], bins='freedman', color='black', histtype='step', density=True, label = 'No Perturbation', linestyle='-')
        fancy_hist(perturbations[i], ax=ax[i+1], bins='freedman', color='red', histtype='step', density=True, label = 'Perturbed', linestyle='--')
        ax[i+1].set_title(titles[i])
        ax[i+1].legend()
    plt.tight_layout()
    plt.show()
