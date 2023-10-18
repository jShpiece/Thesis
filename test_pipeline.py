import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import createLenses, createSources, print_progress_bar
import time
import warnings
from astropy.visualization import hist as fancyhist

sigf = 0.01
sigs = 0.1

# ------------------------
# Testing Utility Functions
# ------------------------

def _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title, ax=None):
    """Private helper function to plot the results of lensing reconstruction."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(xlens, ylens, color='red', label='Recovered Lenses')
    for i, eR in enumerate(eRlens):
        ax.annotate(round(eR, 2), (xlens[i], ylens[i]))
    ax.scatter(x, y, marker='.', color='blue', label='Sources')
    ax.scatter(xlarr, ylarr, marker='x', color='green', label='True Lenses')
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

def run_simple_test(Nlens,Nsource,xmax):
    """Runs a basic test of the algorithm."""
    start_time = time.time()

    # Initialize true lens and source configurations
    xlarr, ylarr, tearr = createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    x, y, e1data, e2data, f1data, f2data = createSources(xlarr, ylarr, tearr, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

    # Perform lens position optimization
    xlens, ylens, eRlens, chi2val = pipeline.optimize_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf, xmax=xmax, flags=True)
    
    print('Time elapsed:', time.time() - start_time)
    
    _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, 'Lensing Reconstruction')
    plt.show()


def visualize_pipeline_steps(nlens, nsource, xmax):
    """Visualizes each step of the lensing reconstruction pipeline."""
    # Setup lensing and source configurations
    xlarr, ylarr, tearr = createLenses(nlens=nlens, randompos=False, xmax=xmax)
    x, y, e1data, e2data, f1data, f2data = createSources(xlarr, ylarr, tearr, ns=nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

    # Initialize the plot
    fig, axarr = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle('Lensing Reconstruction Demo - {} lenses, {} sources'.format(nlens, nsource))

    # Define a list of pipeline steps and corresponding titles

    # Step 1: Find initial lens positions
    xlens, ylens, eRlens = pipeline.find_initial_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title = 'Initial Minimization', ax=axarr[0][0])

    # Step 2: Filter lens positions
    xlens, ylens, eRlens = pipeline.filter_lens_positions(xlens, ylens, eRlens, x, y, xmax)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title = 'Lens Filtering', ax=axarr[0][1])

    # Step 3: Merge close lenses
    xlens, ylens, eRlens = pipeline.merge_close_lenses(xlens, ylens, eRlens)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title = 'Lens Merging', ax=axarr[1][0])

    # Step 4: Iterative elimination
    xlens, ylens, eRlens, chi2val = pipeline.iterative_elimination(xlens, ylens, eRlens, chi2val, x, y, e1data, e2data, f1data, f2data, sigf, sigs, lens_floor=nlens)
    _plot_results(xmax, xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title = 'Iterative Elimination', ax=axarr[1][1])

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//algorithm_visualization.png')
    plt.show()


def bulk_test(ntests): 
    warnings.simplefilter('ignore')
    Nsources = [1,2,3,4,5]

    print('| Nsources | % of solutions empty | % of solutions worse than true solution |')
    for N in Nsources:
        nempty = 0
        n_badfit = 0
        for i in range(ntests):
            # Set up the true lens configuration
            nlens = 1
            xmax = 5 #Range of our lensing field - distance from the origin

            xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=True,xmax=xmax)

            # Set up the true source configuration
            ns = N
            x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)
            true_chi2 = pipeline.get_chi2_value(x,y,e1data,e2data,f1data,f2data,xlarr,ylarr,tearr,sigf,sigs)

            # Run the minimization
            xlens,_,_,chi2val = pipeline.optimize_lens_positions(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)
            if len(xlens) == 0:
                nempty += 1

            if chi2val > true_chi2:
                n_badfit += 1
        
        #Print results 
        print('| {} | {} | {} |'.format(N, nempty/ntests, n_badfit/ntests))


def random_realization(Ntrials, Nlens=1, Nsource=1, xmax=10, sigf=0.01, sigs=0.1):
    warnings.simplefilter('ignore')

    # Store solution arrays
    xsol, ysol, er = [], [], []
    # True lens configuration
    true_xlens, true_ylens, true_erlens = createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    # Initialize a progress bar
    print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)

    for trial in range(Ntrials):
        # Generate source data based on true lens configuration
        x, y, e1data, e2data, f1data, f2data = createSources(true_xlens, true_ylens, true_erlens, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

        # Recover lens positions
        recovered_xlens, recovered_ylens, recovered_erlens, _ = pipeline.optimize_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf, xmax=xmax)

        # If no lens is recovered, skip the current trial
        if not len(recovered_xlens):
            continue

        # Store recovered lens positions - no cheating and only picking the best solution, take everything we get
        for x, y, z in zip(recovered_xlens, recovered_ylens, recovered_erlens):
            xsol.append(x)
            ysol.append(y)
            er.append(z)
        
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
            label = 'Lens ' + str(lens_num+1)
            # Using a fancy histogram function (imported from astropy.visualization)
            fancyhist(data[~np.isnan(data)], ax=a, bins='scott', histtype='step', density=True, color=color)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    ax[0].vlines(true_xlens, 0, 1, color='red', label='True Lenses')
    ax[1].vlines(true_ylens, 0, 1, color='red', label='True Lenses')
    ax[2].vlines(true_erlens, 0, 1, color='red', label='True Lenses')

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

    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=True,xmax=xmax)
    
    # Set up the true source configuration
    ns = nlens * np.random.randint(1,10)
    x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Run the minimization in steps
    xlens,ylens,eRlens = pipeline.find_initial_lens_positions(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in initial lens positions')
        print(xlens,ylens,eRlens,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    xlens,ylens,eRlens = pipeline.filter_lens_positions(xlens,ylens,eRlens,x,y,xmax)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in filtered lens positions')
        print(xlens,ylens,eRlens,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    xlens,ylens,eRlens = pipeline.merge_close_lenses(xlens,ylens,eRlens)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in merged lens positions')
        print(xlens,ylens,eRlens,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    xlens,ylens,eRlens,chi2val = pipeline.iterative_elimination(xlens,ylens,eRlens,chi2val, x, y, e1data, e2data, f1data, f2data, sigf, sigs)
    chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
    # Check for NaNs
    if np.isnan(chi2val):
        print('NaNs in final lens positions')
        print(xlens,ylens,eRlens,chi2val)
        print('Parameters:')
        print(nlens,ns,xmax)
        return False
    
    return True


if __name__ == '__main__':
    # run_simple_test(2,4,20)
    # visualize_pipeline_steps(2, 4, 10)
    # bulk_test(100)
    # random_realization(10**4,1,4)
    # random_realization(10**4,2,4)
    # random_realization(10**4,2,20,100)

    # I just want to test the iterative elimination step - lets run this for a couple of distinct trials
    # Trial 1: Small field, small source population
    # Trial 2: Large field, small source population
    # Trial 3: Large field, large source population
    # Trial 4: Small field, large source population

    trials = 1000
    nlens = 2
    ns = [10, 10, 100, 100]
    xmax = [10, 100, 100, 10]
    title = ['Small Field, Small Source Population', 'Large Field, Small Source Population', 'Large Field, Large Source Population', 'Small Field, Large Source Population']
    badsol_chi2 = []

    for i in range(4):
        xsol, ysol, er = [], [], []
        xlens, ylens, telens = createLenses(nlens=nlens, randompos=False, xmax=xmax[i]) # True lens configuration, fixed for all trials

        # Initialize a progress bar
        print_progress_bar(0, trials, prefix='Trial {} Progress:'.format(i+1), suffix='Complete', length=50)
        for j in range(trials):
            # Set up the true source configuration - this varies with each trial
            x, y, e1data, e2data, f1data, f2data = createSources(xlens, ylens, telens, ns=ns[i], sigf=sigf, sigs=sigs, randompos=True, xmax=xmax[i])

            # Now generate a bunch of random lens positions and see if we can recover the true solution
            fake_lenses = ns[i]
            fake_x, fake_y, fake_te = createLenses(nlens=fake_lenses, randompos=True, xmax=xmax[i])

            # Combine the true and fake lens positions
            test_xlens = np.concatenate((xlens, fake_x))
            test_ylens = np.concatenate((ylens, fake_y))
            test_telens = np.concatenate((telens, fake_te))
            chi2val = pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, test_xlens, test_ylens, test_telens, sigf, sigs)

            # Run iterative elimination
            xs, ys, ers, chi2val = pipeline.iterative_elimination(test_xlens, test_ylens, test_telens, chi2val, x, y, e1data, e2data, f1data, f2data, sigf, sigs, lens_floor=nlens)

            # If the solution does not match the true solution, record the chi2 value as a ratio of the true chi2 value
            badsol_chi2.append(chi2val / pipeline.get_chi2_value(x, y, e1data, e2data, f1data, f2data, xlens, ylens, telens, sigf, sigs))

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
            if np.all(np.isin(xlens, xsol[k])) and np.all(np.isin(ylens, ysol[k])) and np.all(np.isin(telens, er[k])):
                success += 1
        
        # Plot the results
        fig, ax = plt.subplots()
        ax.scatter(xlens, ylens, color='red', marker='X', label='True Lenses')
        ax.scatter(xsol, ysol, color='blue', marker='.', alpha=0.25, label='Recovered Lenses')
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-xmax[i], xmax[i])
        ax.set_ylim(-xmax[i], xmax[i])
        ax.set_aspect('equal')
        ax.set_title(title[i] + '\n {} lenses, {} sources, {} trials \n Success rate: {:.2f}%'.format(nlens, ns[i], trials, success/trials * 100))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images//iterative_elimination_test_{}.png'.format(i+1))


    # Plot the chi2 values for the bad solutions
    fig, ax = plt.subplots()
    ax.hist(badsol_chi2, bins=100)
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('Frequency')
    ax.set_title('Chi2 Values for Bad Solutions')
    plt.savefig('Images//bad_solution_chi2.png')
    plt.show()