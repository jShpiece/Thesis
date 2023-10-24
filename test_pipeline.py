import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import print_progress_bar, generate_combinations
import time
from astropy.visualization import hist as fancyhist
import scipy.optimize as opt

# Define default noise parameters
sigf = 0.01
sigs = 0.1

# ------------------------
# Testing Utility Functions
# ------------------------

def _plot_results(xmax, lenses, x, y, xl, yl, chi2val, title, ax=None):
    """Private helper function to plot the results of lensing reconstruction."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
    for i, eR in enumerate(lenses.te):
        ax.annotate(round(eR, 2), (lenses.x[i], lenses.y[i]))
    ax.scatter(x, y, marker='.', color='blue', label='Sources')
    ax.scatter(xl, yl, marker='x', color='green', label='True Lenses')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    ax.set_title(title + '\n' + r' $\chi^2$ = {:.2f}'.format(chi2val))


# ------------------------
# Test Implementation Functions
# ------------------------

def run_simple_test(Nlens,Nsource,xmax,flags=False):
    """Runs a basic test of the algorithm.
    params:
        Nlens: number of lenses
        Nsource: number of sources
        xmax: range of lensing field
        flags: if True, print out intermediate results
    """
    start_time = time.time()

    # Initialize true lens and source configurations
    lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, chi2val = pipeline.fit_lensing_field(sources, sigs=sigs, sigf=sigf, xmax=xmax, lens_floor = Nlens, flags=flags)
    
    print('Time elapsed:', time.time() - start_time)
    
    _plot_results(xmax, recovered_lenses, sources.x, sources.y, lenses.x, lenses.y, chi2val, 'Lensing Reconstruction')
    plt.savefig('Images//simple_test_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def visualize_pipeline_steps(nlens, nsource, xmax):
    """Visualizes each step of the lensing reconstruction pipeline.
    params:
        nlens: number of lenses
        nsource: number of sources
        xmax: range of lensing field
    """
    # Setup lensing and source configurations
    lenses = pipeline.createLenses(nlens=nlens, randompos=False, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
    x,y = sources.x, sources.y
    xl, yl, te = lenses.x, lenses.y, lenses.te

    # Arrange a plot with 5 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lensing Reconstruction Pipeline', fontsize=16)

    # Define a list of pipeline steps and corresponding titles

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Initial Lens Positions', ax=axarr[0,0])

    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, sigs, sigf)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Optimized Lens Positions', ax=axarr[0,1])

    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Filtered Lens Positions', ax=axarr[0,2])

    # Step 4: Merge lenses that are too close to each other
    lenses.merge_close_lenses()
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Merged Lens Positions', ax=axarr[1,0])

    # Step 5: Iterative elimination
    lenses.iterative_elimination(chi2val, sources, sigf, sigs, lens_floor=nlens)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Final Lens Positions', ax=axarr[1,1])

    # Step 6: Final minimization
    params = [sources, sigs, sigf]
    for i in range(len(lenses.x)):
        guess = [lenses.x[i], lenses.y[i], lenses.te[i]]
        result = opt.minimize(pipeline.chi2wrapper, guess, args=(params), method='Nelder-Mead', tol=1e-8)
        lenses.x[i], lenses.y[i], lenses.te[i] = result.x
    _plot_results(xmax, lenses, x, y, xl, yl, chi2val, 'Final Lens Positions', ax=axarr[1,2])

    # Save and show the plot
    # Can we shift the bottom two subplots to better center them?
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//algorithm_visualization.png')
    plt.show()


def find_empty_sols(ntests): 
    # Test function to find out how often the algorithm returns empty solutions
    Nsources = [1,2,3,4,5]

    print('| Nsources | % of solutions empty | % of solutions worse than true solution |')
    for N in Nsources:
        nempty = 0
        n_badfit = 0
        for i in range(ntests):
            # Set up the true lens configuration
            nlens = 1
            xmax = 5 #Range of our lensing field - distance from the origin

            true_lenses = pipeline.createLenses(nlens=nlens,randompos=True,xmax=xmax)

            # Set up the true source configuration
            ns = N
            sources = pipeline.createSources(true_lenses,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)
            true_chi2 = pipeline.get_chi2_value(sources,true_lenses,sigf,sigs)

            # Run the minimization
            lenses,chi2val = pipeline.optimize_lens_positions(sources,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)
            if len(lenses.x) == 0:
                nempty += 1

            if chi2val > true_chi2:
                n_badfit += 1
        
        #Print results 
        print('| {} | {} | {} |'.format(N, nempty/ntests, n_badfit/ntests))


def random_realization(Ntrials, Nlens=1, Nsource=1, xmax=10, sigf=0.01, sigs=0.1):
    '''
    Testing function to generate random realizations of the lensing field and test the algorithm's performance
    params:
        Ntrials: number of trials to run
        Nlens: number of lenses
        Nsource: number of sources
        xmax: range of lensing field
        sigf: noise parameter for flexion
        sigs: noise parameter for shear
    '''
    # Store solution arrays
    xsol, ysol, er = [], [], []
    # True lens configuration
    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    # Initialize a progress bar
    print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    for trial in range(Ntrials):
        # Generate source data based on true lens configuration
        sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

        # Recover lens positions
        lenses, _ = pipeline.fit_lensing_field(sources, sigs=sigs, sigf=sigf, xmax=xmax, lens_floor = Nlens, flags=False)

        # If no lens is recovered, skip the current trial
        if len(lenses.x) == 0:
            continue

        # Store recovered lens positions - no cheating and only picking the best solution, take everything we get
        xsol.append(lenses.x)
        ysol.append(lenses.y)
        er.append(lenses.te)
        
        # Update progress bar
        print_progress_bar(trial + 1, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    #Turn solution arrays into numpy arrays
    xsol = np.array(xsol)
    ysol = np.array(ysol)
    er = np.array(er)

    # Store results
    np.save('data//x.npy', xsol)
    np.save('data//y.npy', ysol)
    np.save('data//theta.npy', er)

    # Plot histograms
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    param_labels = [r'$x$', r'$y$', r'$\theta_E$']

    for lens_num in range(Nlens):
        color = colors[lens_num % len(colors)]
        for a, data, param_label in zip(ax, [xsol,ysol,er], param_labels):
            # Using a fancy histogram function (imported from astropy.visualization)
            fancyhist(data[~np.isnan(data)], ax=a, bins='scott', histtype='step', density=True, color=color)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    # Plot true lens positions
    # Set maximum height of vertical lines to match the maximum height of the histograms
    ymax = ax[0].get_ylim()[1]
    ax[0].vlines(true_lenses.x, 0, ymax, color='red', label='True Lenses')
    ymax = ax[1].get_ylim()[1]
    ax[1].vlines(true_lenses.y, 0, ymax, color='red', label='True Lenses')
    ymax = ax[2].get_ylim()[1]
    ax[2].vlines(true_lenses.te, 0, ymax, color='red', label='True Lenses')

    for a in ax:
        a.legend()
            
    fig.suptitle(f'Random Realization Test - {Nlens} lenses, {Nsource} sources \n {Ntrials} trials')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig(f'Images//random_realization_{Nlens}_lens_{Nsource}_source.png')
    plt.close()


def locate_nan_entries():
    # Test function to figure out where in the algorithm NaNs are being introduced

    # Set up the true lens configuration
    nlens = np.random.randint(1,10)
    xmax = np.random.uniform(2,100) #Range of our lensing field - distance from the origin

    true_lenses = pipeline.createLenses(nlens=nlens,randompos=True,xmax=xmax)
    
    # Set up the true source configuration
    ns = nlens * np.random.randint(1,10)
    sources = pipeline.createSources(true_lenses,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Run the minimization in steps
    lenses = sources.generate_initial_guess()
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in initial lens guesses')
        print(lenses.x,lenses.y,lenses.te,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    lenses.optimize_lens_positions(sources, sigs, sigf)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in optimized lens positions')
        print(lenses.x,lenses.y,lenses.te,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    lenses.filter_lens_positions(sources, xmax)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in filtered lens positions')
        print(lenses.x,lenses.y,lenses.te,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    lenses.merge_close_lenses()
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in merged lens positions')
        print(lenses.x,lenses.y,lenses.te,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    lenses.iterative_elimination(chi2val, sources, sigf, sigs, lens_floor=nlens)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in final lens positions')
        print(lenses.x,lenses.y,lenses.te,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False

    return True


def test_initial_guesser():
    # What's the quality of our guesses?

    ns = 10 #Number of sources
    nl = 2 #Number of lenses
    xmax = 5

    # Set up the true lens configuration
    true_lenses = pipeline.createLenses(nlens=nl,randompos=False,xmax=xmax)

    # Set up the true source configuration
    sources = pipeline.createSources(true_lenses,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Plot these two configurations
    #xmax *= 1.5

    fig, ax = plt.subplots(1,2,figsize=(10,5))
    fig.suptitle('Checking Initial Guesses', fontsize=16)
    # Generate candidate lenses
    lenses = sources.generate_initial_guess()
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)

    _plot_results(xmax, lenses, sources.x, sources.y, xl, yl, chi2val, 'Initial Guesses', ax=ax[0])
    # Draw an arrow from each source to the corresponding lens
    for i in range(len(sources.x)):
        ax[0].arrow(sources.x[i], sources.y[i], lenses.x[i] - sources.x[i], lenses.y[i] - sources.y[i], color='black', alpha=0.5)

    # Perform local minimization
    lenses.optimize_lens_positions(sources, sigs, sigf)
    chi2val = pipeline.get_chi2_value(sources, lenses, sigf, sigs)

    _plot_results(xmax, lenses, sources.x, sources.y, xl, yl, chi2val, 'Optimized Guesses', ax=ax[1])
    # Draw an arrow from each source to the corresponding lens
    for i in range(len(sources.x)):
        ax[1].arrow(sources.x[i], sources.y[i], lenses.x[i] - sources.x[i], lenses.y[i] - sources.y[i], color='black', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('Images//quality_of_guesses.png')
    plt.show()


def test_iterative_elimination():

    # I just want to test the iterative elimination step - lets run this for a couple of distinct trials
    # Trial 1: Small field, small source population
    # Trial 2: Large field, small source population
    # Trial 3: Large field, large source population
    # Trial 4: Small field, large source population

    trials = 100
    nlens = 2
    ns = [10, 10, 100, 100]
    xmax = [10, 100, 100, 10]
    title = ['Small Field, Small Source Population', 'Large Field, Small Source Population', 'Large Field, Large Source Population', 'Small Field, Large Source Population']
    badsol_chi2 = []

    for i in range(4):
        xsol, ysol, er = [], [], []
        lenses = pipeline.createLenses(nlens=nlens, randompos=False, xmax=xmax[i]) # True lens configuration, fixed for all trials
        # Change the einstein radii to random values between 0.5 and 5
        telens = np.random.uniform(0.5, 5, size=nlens)
        # Initialize a progress bar
        print_progress_bar(0, trials, prefix='Trial {} Progress:'.format(i+1), suffix='Complete', length=50)
        for j in range(trials):
            # Set up the true source configuration - this varies with each trial
            sources = pipeline.createSources(lenses, ns=ns[i], sigf=sigf, sigs=sigs, randompos=True, xmax=xmax[i])

            # Now generate a bunch of random lens positions and see if we can recover the true solution
            fake_lenses = ns[i]
            fake_x, fake_y, fake_te = pipeline.createLenses(nlens=fake_lenses, randompos=True, xmax=xmax[i])

            # Combine the true and fake lens positions
            test_lenses = pipeline.Lens(np.concatenate((lenses.x, fake_x)), np.concatenate((lenses.y, fake_y)), np.concatenate((telens, fake_te)))
            chi2val = pipeline.get_chi2_value(sources, test_lenses, sigf, sigs)

            # Run iterative elimination
            xs, ys, ers, chi2val = pipeline.iterative_elimination(test_lenses, chi2val, sources, sigf, sigs, lens_floor=nlens)

            # If the solution does not match the true solution, record the chi2 value as a ratio of the true chi2 value
            badsol_chi2.append(chi2val / pipeline.get_chi2_value(sources, lenses, sigf, sigs))

            # Store the recovered lens positions
            xsol.append(xs)
            ysol.append(ys)
            er.append(ers)

            # Update progress bar
            print_progress_bar(j + 1, trials, prefix='Trial {} Progress:'.format(i+1), suffix='Complete', length=50)

        # Assess accuracy - how many of the recovered lenses are the true lenses?
        xsol = np.array(xsol)
        ysol = np.array(ysol)
        er = np.array(er)
        success = 0

        for k in range(trials):
            # Check if the true lenses are in the recovered lens positions
            if np.all(np.isin(lenses.x, xsol[k])) and np.all(np.isin(lenses.y, ysol[k])) and np.all(np.isin(telens, er[k])):
                success += 1
        
        # Plot the results
        fig, ax = plt.subplots()
        ax.scatter(lenses.x, lenses.y, color='red', marker='X', label='True Lenses')
        ax.scatter(xsol, ysol, color='blue', marker='.', alpha=0.25, label='Recovered Lenses')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-xmax[i], xmax[i])
        ax.set_ylim(-xmax[i], xmax[i])
        ax.set_aspect('equal')
        ax.set_title(title[i] + '\n {} lenses, {} sources, {} trials \n Success rate: {:.2f}%'.format(nlens, ns[i], trials, success/trials * 100))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images//iterative_test_{}.png'.format(i+1))


    # Plot the chi2 values for the bad solutions
    fig, ax = plt.subplots()
    ax.hist(badsol_chi2, bins=100)
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('Frequency')
    ax.set_title('Chi2 Values for Bad Solutions')
    plt.savefig('Images//bad_solution_chi2.png')
    plt.show()


if __name__ == '__main__':
    # run_simple_test(2, 100, 100, flags=False)
    # visualize_pipeline_steps(2, 20, 10)
    random_realization(10**3,1,100,100)

    #run_simple_test(1, 20, 40, flags=False)