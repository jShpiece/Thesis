import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import print_progress_bar
import time
from astropy.visualization import hist as fancyhist
import scipy.optimize as opt

# Define default noise parameters
sigf = 0.01
sigs = 0.1 

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
        ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))


def plot_random_realizations(xsol, ysol, er, true_lenses, Nlens, Nsource, Ntrials, xmax, distinguish_lenses=False):
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
            if distinguish_lenses:
                fancyhist(data[:, lens_num], ax=a, bins='scott', histtype='step', density=True, color=color, label=f'Lens {lens_num + 1}', range=range_val)
            else:
                fancyhist(data.flatten(), ax=a, bins='scott', histtype='step', density=True, color=colors[0], label=f'Lens {lens_num + 1}', range=range_val)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    eR_median = np.median(er.flatten())
    ax[2].vlines(eR_median, 0, ax[2].get_ylim()[1], color='blue', label='Median = {:.2f}'.format(eR_median))

    # Plot true lens positions
    for a, true_value in zip(ax, [true_lenses.x, true_lenses.y, true_lenses.te]):
        a.vlines(true_value, 0, a.get_ylim()[1], color='red', label='True Lenses')
        a.legend(loc='upper right')

    fig.suptitle(f'Random Realization Test \n {Nlens} lenses, {Nsource} sources \n {Ntrials} trials')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Images//rand_real//rr_testIE_{Nlens}_lens_{Nsource}_source_{Ntrials}.png')
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
    lenses = pipeline.createLenses(nlens=Nlens, randompos=lens_random, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=source_random, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = Nlens, flags=flags)
    
    print('Time elapsed:', time.time() - start_time)
    
    _plot_results(xmax, recovered_lenses, sources, lenses.x, lenses.y, reducedchi2, 'Lensing Reconstruction')
    plt.savefig('Images//tests//simple_test_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def visualize_pipeline_steps(Nlens, Nsource, xmax):
    """Visualizes each step of the lensing reconstruction pipeline.
    params:
        nlens: number of lenses
        nsource: number of sources
        xmax: range of lensing field
    """
    # Setup lensing and source configurations
    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
    xl, yl, _ = true_lenses.x, true_lenses.y, true_lenses.te

    # Arrange a plot with 5 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lensing Reconstruction Pipeline', fontsize=16)

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Initial Lens Positions', ax=axarr[0,0])


    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, sigf, sigs)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Optimized Lens Positions', ax=axarr[0,1])


    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Filtered Lens Positions', ax=axarr[0,2])

    # Step 4: Merge lenses that are too close to each other
    lenses.merge_close_lenses()
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Merged Lens Positions', ax=axarr[1,0])

    # Step 5: Iterative elimination
    lens_floors = np.arange(1, len(lenses.x) + 1)
    best_dist = np.abs(reducedchi2 - 1)
    for lens_floor in lens_floors:
        # Clone the lenses object
        test_lenses = pipeline.Lens(lenses.x, lenses.y, lenses.te, lenses.chi2)
        test_lenses.iterative_elimination(lens_floor=lens_floor)
        reducedchi2 = test_lenses.update_chi2_values(sources, sigf, sigs)
        new_dist = np.abs(reducedchi2 - 1)
        if new_dist < best_dist:
            best_dist = new_dist
            best_lenses = test_lenses
        lenses = best_lenses
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Iterative Elimination', ax=axarr[1,1])

    # Step 6: Final minimization
    lenses.full_minimization(sources, sigf, sigs)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Final Minimization', ax=axarr[1,2])

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//tests//breakdown_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def generate_random_realizations(Ntrials, Nlens=1, Nsource=1, xmax=10, sigf=0.01, sigs=0.1):
    '''
    Generate random realizations of the lensing field.
    '''
    # True lens configuration
    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    # Store solution arrays
    xsol, ysol, er = np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens))

    # Initialize a progress bar
    print_progress_bar(0, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)
    
    for trial in range(Ntrials):
        sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
        lenses, _ = pipeline.fit_lensing_field(sources, sigs=sigs, sigf=sigf, xmax=xmax, lens_floor=Nlens, flags=False)
        
        num_lenses_recovered = len(lenses.x)
        # Take the minimum of the recovered lenses and the specified number of lenses
        for i in range(min(num_lenses_recovered, Nlens)):
            xsol[trial, i], ysol[trial, i], er[trial, i] = lenses.x[i], lenses.y[i], lenses.te[i]
    
        # Update progress bar
        print_progress_bar(trial, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)
    
    # Finalize the progress bar so that I don't lose my mind
    print_progress_bar(Ntrials, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)

    np.save('Data//xsol_{}_lens_{}_source'.format(Nlens, Nsource), xsol)
    np.save('Data//ysol_{}_lens_{}_source'.format(Nlens, Nsource), ysol)
    np.save('Data//er_{}_lens_{}_source'.format(Nlens, Nsource), er)

    return xsol, ysol, er, true_lenses


def assess_number_recovered(Nlens, Nsource, xmax, sigf=0.01, sigs=0.1, lens_random=False, source_random=True):
    # Initialize true lens and source configurations
    lenses = pipeline.createLenses(nlens=Nlens, randompos=lens_random, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=source_random, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = Nlens, flags=False)
    # Don't count any lenses with an einstein radius of 0
    num_recovered = 0
    for eR in recovered_lenses.te:
        if eR > 0:
            num_recovered += 1
    return num_recovered


def reconstruct_system(file, flags=False):
    # Take in a file of sources and reconstruct the lensing system

    # Read in the data (csv)
    data = np.genfromtxt(file, delimiter=',')
    # Read the header to get the column names
    with open(file, 'r') as f:
        header = f.readline().strip().split(',')
    # Get the column indices
    xcol, ycol, qcol, phicol, f1col, f2col = header.index('xs'), header.index('ys'), header.index('q'), header.index('phi'), header.index('f1'), header.index('f2')
    # Extract the data - omit the header
    x, y, q, phi, f1, f2 = data[1:, xcol], data[1:, ycol], data[1:, qcol], data[1:, phicol], data[1:, f1col], data[1:, f2col]
    # Set these to be arrays 
    x, y, q, phi, f1, f2 = np.array(x), np.array(y), np.array(q), np.array(phi), np.array(f1), np.array(f2)

    # Calculate the shear from q and phi
    shear_mag = (q-1)/(q+1)
    e1 = shear_mag * np.cos(2*phi)
    e2 = shear_mag * np.sin(2*phi)
  
    # Set xmax to be the largest distance from the center
    centroid = np.mean(x), np.mean(y)
    xmax = np.max(np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2))

    # Set the centroid to be the origin
    x -= centroid[0]
    y -= centroid[1]

    flexion = np.sqrt(f1**2 + f2**2)
    shear = np.sqrt(e1**2 + e2**2)
    sigf = np.std(flexion)
    sigs = np.std(shear)

    # Create a source object
    sources = pipeline.Source(x, y, e1, e2, f1, f2)
    sources.filter_sources(xmax)
    
    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = 1, flags=flags)
    return recovered_lenses, sources, xmax, reducedchi2


if __name__ == '__main__':
    # run_simple_test(1, 100, 50, flags=True)
    # visualize_pipeline_steps(2, 100, 50)

    lenses, sources, xmax, chi2 = reconstruct_system('a2744_clu_lenser.csv', flags=True)
    _plot_results(xmax, lenses, sources, None, chi2, 'Lensing Reconstruction: A2744', legend=False)
    plt.savefig('Images//tests//a2744.png')
    plt.show()

    raise SystemExit
    # ------------------------


    Nsource = 100
    xmax = 50
    Nlens = 2

    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    while True:
        sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
        lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = Nlens, flags=False)

        if len(lenses.x) > 10:
            break
    
    _plot_results(xmax, lenses, sources, true_lenses.x, true_lenses.y, reducedchi2, 'Lensing Reconstruction: More than 10 Lenses Recovered')
    plt.savefig('Images//tests//more_than_10_lenses.png')
    plt.show()

    raise SystemExit
    # ------------------------

    nlens = [1, 2]
    Ntrials = 100
    Nsource = 100
    xmax = 50

    for nl in nlens:
        xsol, ysol, er, true_lenses = generate_random_realizations(Ntrials, Nlens=nl, Nsource=Nsource, xmax=xmax, sigf=0.01, sigs=0.1)
        plot_random_realizations(xsol, ysol, er, true_lenses, nl, Nsource, Ntrials, xmax, distinguish_lenses=False)

    raise SystemExit

    # ------------------------