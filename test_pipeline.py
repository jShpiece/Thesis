import numpy as np
import matplotlib.pyplot as plt
import pipeline
import utils
import time
from astropy.visualization import hist as fancyhist
from multiprocessing import Pool
from functools import partial
import sklearn.metrics
import mdark


plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

# Define default noise parameters
sigs = 0.1 
sigf = 0.01 
sigg = 0.02 


# ------------------------
# Testing Plotting Functions
# ------------------------

def _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, title, ax=None, legend=True):
    """Private helper function to plot the results of lensing reconstruction."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
    for i, eR in enumerate(lenses.te):
        ax.annotate(np.round(eR, 2), (lenses.x[i], lenses.y[i]))
    ax.scatter(sources.x, sources.y, marker='.', color='blue', alpha=0.5, label='Sources')
    if true_lenses is not None:
        ax.scatter(true_lenses.x, true_lenses.y, marker='x', color='green', label='True Lenses')
    if legend:
        ax.legend(loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if xmax is not None:
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))


def plot_random_realizations(xsol, ysol, er, true_lenses, Nlens, Nsource, Ntrials, xmax, use_flags, title):
    '''
    Plot histograms of the random realizations.
    '''
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    param_labels = [r'$x$', r'$y$', r'$\theta_E$']

    # Plot histograms of the recovered lens positions
    for lens_num in range(Nlens):
        color = colors[lens_num % len(colors)]
        for a, data, param_label in zip(ax, [xsol, ysol, er], param_labels):
            range_val = None
            if param_label in [r'$x$', r'$y$']:
                range_val = (-xmax, xmax)
            elif param_label == r'$\theta_E$':
                range_val = (0, 5)
            fancyhist(data.flatten(), ax=a, bins='scott', histtype='step', density=True, color=colors[0], label=f'Lens {lens_num + 1}', range=range_val)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    eR_median = np.median(er.flatten())
    ax[2].vlines(eR_median, 0, ax[2].get_ylim()[1], color='blue', label='Median = {:.2f}'.format(eR_median))

    # Plot true lens positions
    for a, true_value in zip(ax, [true_lenses.x, true_lenses.y, true_lenses.te]):
        a.vlines(true_value, 0, a.get_ylim()[1], color='red', label='True Lenses')
        a.legend(loc='upper right')

    fig.suptitle(f'Random Realization Test \n {Nlens} lenses, {Nsource} sources \n {Ntrials} trials' + '\n' + title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Images//tests//rr//{Nlens}_lens_{Nsource}_source_{Ntrials}_trials_{use_flags}.png')
    plt.close()

# ------------------------
# Test Implementation Functions
# ------------------------

def run_simple_test(Nlens, Nsource, xmax, flags=False, lens_random=False, source_random=True):
    """Runs a basic test of the algorithm.
    params:
        Nlens: number of lenses
        Nsource: number of sources
        xmax: range of lensing field
        flags: if True, print out intermediate results
    """
    start_time = time.time()

    # Initialize true lens and source configurations
    lenses = utils.createLenses(nlens=Nlens, randompos=lens_random, xmax=xmax)
    sources = utils.createSources(lenses, ns=Nsource, randompos=source_random, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, xmax=xmax, flags=flags, use_flags=[True, True, True])
    
    print('Time elapsed:', time.time() - start_time)
    
    _plot_results(xmax, recovered_lenses, sources, lenses, reducedchi2, 'Lensing Reconstruction')
    plt.savefig('Images//tests//simple_test_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def visualize_pipeline_steps(Nlens, Nsource, xmax, use_flags):
    """Visualizes each step of the lensing reconstruction pipeline.
    params:
        nlens: number of lenses
        nsource: number of sources
        xmax: range of lensing field
    """
    # Setup lensing and source configurations
    true_lenses = utils.createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    sources = utils.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lensing Reconstruction Pipeline', fontsize=16)

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Initial Guesses', ax=axarr[0,0])

    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Initial Optimization', ax=axarr[0,1], legend=False)


    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Filter', ax=axarr[0,2], legend=False)


    # Step 4: Iterative elimination
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Iterative Elimination', ax=axarr[1,0], legend=False)


    # Step 5: Merge lenses that are too close to each other
    ns = Nsource / (2 * xmax)**2
    merger_threshold = 1/np.sqrt(ns)
    lenses.merge_close_lenses(merger_threshold=10)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Merging', ax=axarr[1,1], legend=False)

    # Step 6: Final minimization
    lenses.full_minimization(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Final Minimization', ax=axarr[1,2], legend=False)

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//tests//breakdown_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def process_trial(trial, true_lenses, Nsource, xmax, use_flags):
    sources = utils.createSources(true_lenses, ns=Nsource, sigs=sigs, sigf=sigf, sigg=sigg, randompos=True, xmax=xmax)
    lenses, _ = pipeline.fit_lensing_field(sources, xmax=xmax, flags=False, use_flags=use_flags)
    return lenses


def generate_random_realizations(Ntrials, Nlens, Nsource, xmax, use_flags):
    '''
    Generate random realizations of the lensing field.
    '''
    # True lens configuration
    true_lenses = utils.createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    solutions = []
    num_processes = 40

    with Pool(num_processes) as pool:
        # Map the function over the range of trials
        func = partial(process_trial, true_lenses=true_lenses, Nsource=Nsource, xmax=xmax, use_flags=use_flags)
        solutions = pool.map(func, range(Ntrials))

    # Extract the lens positions from the solutions
    xsol = np.array(np.concatenate([lenses.x for lenses in solutions]))
    ysol = np.array(np.concatenate([lenses.y for lenses in solutions]))
    er = np.array(np.concatenate([lenses.te for lenses in solutions]))

    # Count the number of lenses found in each trial
    num_lenses = np.array([len(lenses.x) for lenses in solutions])

    # Close the pool
    pool.close()
    pool.join()

    return xsol, ysol, er, num_lenses, true_lenses



# ------------------------
# Helper Functions
# ------------------------


def visualize_examples(use_shear, use_flexion, use_g_flexion):
    # Generate a set of simple examples
    use_flags = [use_shear, use_flexion, use_g_flexion]

    # Example 1: 1 lens, 1 source
    visualize_pipeline_steps(1, 1, 10, use_flags)

    # Example 2: 1 lens, 2 sources
    visualize_pipeline_steps(1, 2, 10, use_flags)

    # Example 3: 2 lenses, 1 source
    visualize_pipeline_steps(2, 1, 10, use_flags)

    # Example 4: 1 lens, 10 sources
    visualize_pipeline_steps(1, 10, 10, use_flags)

    # Example 5: 2 lenses, 10 sources
    visualize_pipeline_steps(2, 10, 10, use_flags)

    # Example 6: 1 lens, 100 sources
    visualize_pipeline_steps(1, 100, 50, use_flags)

    # Example 7: 2 lenses, 100 sources
    visualize_pipeline_steps(2, 100, 50, use_flags)


def accuracy_tests(use_shear, use_flexion, use_g_flexion):
    use_flags = [use_shear, use_flexion, use_g_flexion]
    signals = ['Shear', 'Flexion', 'G-Flexion']
    flag_title = 'Signals used: ' + ', '.join([signal for signal, flag in zip(signals, use_flags) if flag])
    # Generate random realizations
    Ntrials = 1000
    Nlens = [1,2]
    Nsource = 100 # Choose source number & xmax such that source density is 0.01 per unit area
    xmax = 50

    for nlens in Nlens:
        xsol, ysol, er, num_recovered, true_lenses = generate_random_realizations(Ntrials, Nlens=nlens, Nsource=Nsource, xmax=xmax, use_flags=use_flags)
        plot_random_realizations(xsol, ysol, er, true_lenses, Nlens=nlens, Nsource=Nsource, Ntrials=Ntrials, xmax=xmax, use_flags=use_flags, title=flag_title)

        plt.figure()
        fancyhist(num_recovered, bins='scott', histtype='step', density=True, label=f'{nlens} Lenses')
        plt.xlabel('Number of Lenses Recovered')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.vlines(nlens, 0, plt.gca().get_ylim()[1], color='red', label='True Number of Lenses')
        plt.vlines(np.median(num_recovered), 0, plt.gca().get_ylim()[1], color='blue', label='Median: {:.2f}'.format(np.median(num_recovered)))
        plt.title(f'Number of Lenses Recovered \n {Nsource} Sources, {Ntrials} Trials' + '\n' + flag_title)
        plt.tight_layout()
        plt.savefig(f'Images//tests//n_recovered//{nlens}_lens_{Nsource}_source_{Ntrials}_{use_flags}.png')
        plt.close()

        print('Finished test with', nlens, 'lenses')


def accuracy_assessment():
    lenses = utils.createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    sources = utils.createSources(lenses, ns=Nsource, sigs=sigs, sigf=sigf, sigg=sigg, randompos=True, xmax=xmax)
    candidate_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, xmax=xmax, flags=True, use_flags=use_flags)

    x_true = lenses.x
    y_true = lenses.y
    x_pred = candidate_lenses.x
    y_pred = candidate_lenses.y

    # Create a distance matrix
    distance_matrix = sklearn.metrics.pairwise_distances(np.array([x_true, y_true]).T, np.array([x_pred, y_pred]).T)

    # Find the nearest neighbour for each lens
    nearest_neighbour = np.argmin(distance_matrix, axis=1)

    # Compare the strengths of the lenses
    true_strength = lenses.te
    pred_strength = candidate_lenses.te

    # Lets get started. First, match the lenses to the candidate lenses
    true_positives = 0
    for i in range(len(nearest_neighbour)):
        # If the distance is small, and the strengths are similar
        # then we have a match
        if distance_matrix[i, nearest_neighbour[i]] < 1 and abs(true_strength[i] - pred_strength[nearest_neighbour[i]]) < 1:
            true_positives += 1
    # Now, count the false positives
    false_positives = len(candidate_lenses.x) - true_positives

    # Now, count the false negatives
    false_negatives = len(lenses.x) - true_positives

    # Plot the results, with the true positives and false positives colored
    plt.figure()
    plt.scatter(x_true, y_true, color='red', label='True Lenses')
    plt.scatter(x_pred, y_pred, color='blue', label='Candidate Lenses')
    # Label predicted lenses that are true positives
    for i in range(len(nearest_neighbour)):
        if distance_matrix[i, nearest_neighbour[i]] < 1 and abs(true_strength[i] - pred_strength[nearest_neighbour[i]]) < 1:
            plt.scatter(x_pred[nearest_neighbour[i]], y_pred[nearest_neighbour[i]], color='green', label='True Positive')
    # Label all lenses with einstein radii
    for i, eR in enumerate(candidate_lenses.te):
        plt.annotate(np.round(eR, 2), (x_pred[i], y_pred[i]))
    plt.xlabel('x')
    plt.ylabel('y')
    title = 'Lensing Reconstruction \n'
    if (true_positives == len(lenses.x)):
        title += 'Perfect Match'
    else:
        title += f'Accuracy: {true_positives / len(lenses.x)}'
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.xlim(-xmax, xmax)
    plt.ylim(-xmax, xmax)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    Nlens = 20
    xmax = 150
    ns = 0.01
    Nsource = int(ns * (2 * xmax)**2)
    use_flags = [True, True, True]

    # Setup lensing and source configurations
    # true_lenses = utils.createLenses(nlens=Nlens, randompos=True, xmax=xmax)
    ID = [11400399088]
    z = 0.194
    halos = mdark.find_halos(ID, z)
    # The halos object is a dictionary, keyed by the ID of the halo
    # We only have one halo, so we can extract it by taking the first key
    halo = halos[ID[0]]
    halo, _, xmax = mdark.build_lensing_field(halo, z, Nsource)

    eR = halo.calc_corresponding_einstein_radius(0.8)
    print('Einstein Radius:', eR)
    # eR /= eR
    true_lenses = pipeline.Lens(halo.x, halo.y, eR, np.ones(len(halo.x)))
    # Recreate the sources object with the new lensing field
    sources = utils.createSources(true_lenses, ns=Nsource, sigs=sigs, sigf=sigf, sigg=sigg, randompos=True, xmax=xmax)

    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lensing Reconstruction Pipeline', fontsize=16)

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Initial Guesses', ax=axarr[0,0])

    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Initial Optimization', ax=axarr[0,1], legend=False)

    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Filter', ax=axarr[0,2], legend=False)

    # Step 4: Iterative elimination
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Iterative Elimination', ax=axarr[1,0], legend=False)

    # Step 5: Merge lenses that are too close to each other
    ns = Nsource / (2 * xmax)**2
    merger_threshold = 1/np.sqrt(ns) 
    lenses.merge_close_lenses(merger_threshold=merger_threshold)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Merging', ax=axarr[1,1], legend=False)

    # Step 6: Final minimization
    lenses.full_minimization(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, 'Final Minimization', ax=axarr[1,2], legend=False)

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//tests//breakdown_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()