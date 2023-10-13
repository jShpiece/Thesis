import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import createLenses, createSources
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
    ax.set_title(title + '\n χ^2 = {:.2f}'.format(chi2val))


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
    steps = [
        ("Initial Minimization", pipeline.find_initial_lens_positions),
        ("Winnowing", pipeline.filter_lens_positions),
        ("Merging", pipeline.merge_close_lenses),
        ("Iterative Minimization", pipeline.iterative_elimination)
    ]
    
    for ax, (title, func) in zip(axarr.ravel(), steps):
        xlens, ylens, eRlens = func(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf, xmax=xmax)
        chi2val = pipeline.chi2(x, y, e1data, e2data, f1data, f2data, xlens, ylens, eRlens, sigf, sigs)
        _plot_results(xlens, ylens, eRlens, x, y, xlarr, ylarr, chi2val, title, ax=ax)

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
            xmax = 2 #Range of our lensing field - distance from the origin

            xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=True,xmax=xmax)
            tearr *= 10 #Make the lensing strength stronger

            # Set up the true source configuration
            ns = N
            x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)
            true_chi2 = pipeline.chi2(x,y,e1data,e2data,f1data,f2data,xlarr,ylarr,tearr,sigf,sigs)

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
    xsol, ysol, er = np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens))
    # True lens configuration
    true_xlens, true_ylens, true_erlens = createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    for trial in range(Ntrials):
        # Generate source data based on true lens configuration
        x, y, e1data, e2data, f1data, f2data = createSources(true_xlens, true_ylens, true_erlens, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

        # Recover lens positions
        recovered_xlens, recovered_ylens, recovered_erlens, _ = pipeline.optimize_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf, xmax=xmax)

        # If no lens is recovered, skip the current trial
        if not len(recovered_xlens):
            # We need to fill the solution arrays with Nan values
            xsol[trial] = np.nan
            ysol[trial] = np.nan
            er[trial] = np.nan
            continue

        # Match recovered lenses with true lenses by minimizing distance
        for lens_idx in range(Nlens):
            distance_squared = (true_xlens[lens_idx] - recovered_xlens) ** 2 + (true_ylens[lens_idx] - recovered_ylens) ** 2
            closest_lens_idx = np.argmin(distance_squared)

            xsol[trial][lens_idx] = recovered_xlens[closest_lens_idx]# - true_xlens[lens_idx]
            ysol[trial][lens_idx] = recovered_ylens[closest_lens_idx]# - true_ylens[lens_idx]
            er[trial][lens_idx] = recovered_erlens[closest_lens_idx]# - true_erlens[lens_idx]

    # Store results
    np.save('data//dx.npy', xsol)
    np.save('data//dy.npy', ysol)
    np.save('data//dtheta.npy', er)

    # Plot histograms
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    param_labels = [r'$\Delta x$', r'$\Delta y$', r'$\Delta \theta_E$']

    for lens_num in range(Nlens):
        color = colors[lens_num % len(colors)]
        for a, data, param_label in zip(ax, [xsol[:, lens_num], ysol[:, lens_num], er[:, lens_num]], param_labels):
            label = 'Lens ' + str(lens_num+1)
            # Using a fancy histogram function (imported from astropy.visualization)
            fancyhist(data[~np.isnan(data)], ax=a, bins='scott', histtype='step', density=True, color=color, label=label)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    ax[0].vlines(true_xlens, 0, 1, color='red', label='True Lenses')
    ax[1].vlines(true_ylens, 0, 1, color='red', label='True Lenses')
    ax[2].vlines(true_erlens, 0, 1, color='red', label='True Lenses')

    for a in ax:
        a.legend()
            
    fig.suptitle(f'Random Realization Test - {Nlens} lenses, {Nsource} sources \n {Ntrials} trials')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig(f'Images//random_realization_{Nlens}_lens.png')
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
    # run_simple_test(1,3,10)
    # visualize_pipeline_steps(1,3,10)
    # bulk_test(100)
    #random_realization(10**4,1,4)
    #random_realization(10**4,2,4)
    random_realization(10**4,4,4)


    '''
    # Run the function until it finds a NaN
    # Terminate after 1000 iterations
    for i in range(1000):
        if not locate_nan_entries():
            break
    '''