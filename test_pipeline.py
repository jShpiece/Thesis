import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import createLenses, createSources
import time
import warnings
from astropy.visualization import hist as fancyhist

sigf = 0.01
sigs = 0.1


def simple_implementation():
    start = time.time()
    #Runs a very simple test of the algorithm, no inputs required
    '''Note to self - chi2 should be calculated per degree of freedom'''
    # Set up the true lens configuration
    nlens = 1 # Number of lenses
    xmax = 5 # The range to consider for the lenses.
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=False,xmax=xmax)

    # Set up the true source configuration
    ns = 10 # Number of sources
    x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Run the minimization
    xlens,ylens,eRlens,chi2val = pipeline.optimize_lens_positions(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=True)

    stop = time.time()
    print('Time elapsed: ', stop-start)
    # Plot the results
    plt.figure()
    plt.scatter(xlens,ylens,color='red',label=r'Recovered Lenses')
    for i in range(len(xlens)):
        plt.annotate(np.round(eRlens[i],2),(xlens[i],ylens[i]))
    plt.scatter(x,y,marker='.',color='blue',label='Sources')
    plt.scatter(xlarr,ylarr,marker='x',color='green',label='True Lenses')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-xmax,xmax)
    plt.ylim(-xmax,xmax)
    plt.gca().set_aspect('equal')
    plt.title('Lensing Reconstruction \n $\chi^2$ = %.2f' % chi2val)
    plt.show()


def visualize_algorithm(nlens,nsource,xmax):
    #Use this to display each step of the algorithm
    # Set up the true lens configuration
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=False,xmax=xmax)
    tearr *= 10 #Make the lensing strength stronger
    
    # Set up the true source configuration
    x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=nsource,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Initialize the plot

    fig,ax = plt.subplots(2,2,figsize=(10,10),sharex=True,sharey=True)
    fig.suptitle('Lensing Reconstruction Demo - {} lenses, {} sources'.format(nlens,nsource))

    def plotter(ax,recovered_x,recovered_y,amplitude,chi2val,title):
        ax.scatter(recovered_x,recovered_y,s=2*amplitude,color='red',label=r'Recovered Lenses')
        ax.scatter(x,y,marker='.',color='blue',label='Sources')
        ax.scatter(xlarr,ylarr,marker='x',color='green',label='True Lenses')
        ax.legend(bbox_to_anchor=(1.1, 1.05))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(-xmax,xmax)
        ax.set_ylim(-xmax,xmax)
        ax.set_aspect('equal')
        ax.set_title(title + '\n $\chi^2$ = %.2f' % chi2val)
        #Label each recovered lens with its amplitude
        for i in range(len(recovered_x)):
            ax.annotate(np.round(amplitude[i],2),(recovered_x[i],recovered_y[i]))

    # Run the minimization
    xlens,ylens,eRlens = pipeline.find_initial_lens_positions(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf)
    chi2val = pipeline.chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,eRlens,sigf,sigs)
    dof = 4*len(x) - 3*len(xlens) #Each source has 4 data points, each lens has 3 parameters
    chi2val /= dof #Normalize by the number of degrees of freedom


    # Plot the results
    plotter(ax[0,0],xlens,ylens,eRlens,chi2val,'Initial Minimization')

    # Perform the winnowing
    xlens,ylens,eRlens = pipeline.filter_lens_positions(xlens,ylens,eRlens,x,y,xmax)

    # Plot the results
    plotter(ax[0,1],xlens,ylens,eRlens,chi2val,'Winnowing')

    # Perform the merging
    xlens,ylens,eRlens = pipeline.merge_close_lenses(xlens,ylens,eRlens)
    
    # Plot the results
    plotter(ax[1,0],xlens,ylens,eRlens,chi2val,'Merging')

    # Perform the iterative minimization
    xlens,ylens,eRlens,chi2val = pipeline.iterative_elimination(xlens,ylens,eRlens,chi2val,x,y,e1data,e2data,f1data,f2data,sigf,sigs)

    # Plot the results
    plotter(ax[1,1],xlens,ylens,eRlens,chi2val,'Iterative Minimization')

    plt.savefig('algorithm_visualization.png')
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


def random_realization(Ntrials,Nlens=1,Nsource=1):
    warnings.simplefilter('ignore')
    #Test the accuracy of the algorithm on a random realization of lenses and sources
    Nsource = 3
    xmax = 10

    xsol = np.empty((Ntrials,Nlens))
    ysol = np.empty((Ntrials,Nlens))
    er = np.empty((Ntrials,Nlens))

    true_xlens, true_ylens, true_erlens = createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    for trial in range(Ntrials):
        # Create lenses and sources
        x, y, e1data, e2data, f1data, f2data = createSources(true_xlens, true_ylens, true_erlens, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)

        # Run the minimization
        recovered_xlens, recovered_ylens, recovered_erlens, _ = pipeline.optimize_lens_positions(x, y, e1data, e2data, f1data, f2data, sigs=sigs, sigf=sigf, xmax=xmax)

        # If no lens is recovered, skip the current trial
        if not len(recovered_xlens):
            continue

        # Calculate the difference between the true and recovered lens parameters
        for lens_idx in range(Nlens):
            # Find the recovered lens closest to the true lens
            distance_squared = (true_xlens[lens_idx] - recovered_xlens)**2 + (true_ylens[lens_idx] - recovered_ylens)**2
            closest_lens_idx = np.argmin(distance_squared)
            
            xsol[trial][lens_idx] = recovered_xlens[closest_lens_idx]
            ysol[trial][lens_idx] = recovered_ylens[closest_lens_idx]
            er[trial][lens_idx] = recovered_erlens[closest_lens_idx]

    #Store the results
    np.save('dx.npy',xsol)
    np.save('dy.npy',ysol)
    np.save('dtheta.npy',er)

    fig,ax = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle('Random Realization Test - {} lenses, {} sources'.format(Nlens,Nsource))

    colors = ['black','red','blue','green','orange','purple','pink','brown','gray','cyan']

    for lens_num in range(Nlens):
        color = colors[lens_num % len(colors)]  # Cycle through colors if Nlens > len(colors)

        xdata = xsol[:,lens_num]
        ydata = ysol[:,lens_num]
        erdata = er[:,lens_num]

        fancyhist(xdata[~np.isnan(xdata)],ax=ax[0],bins='scott',histtype='step',density=True,color = color,label=r'$\Delta x$')
        fancyhist(ydata[~np.isnan(ydata)],ax=ax[1],bins='scott',histtype='step',density=True,color = color,label=r'$\Delta y$')
        fancyhist(erdata[~np.isnan(erdata)],ax=ax[2],bins='scott',histtype='step',density=True,color = color,label=r'$\Delta \theta_E$')

    ax[0].set_xlabel(r'$\Delta x$')
    ax[0].set_ylabel('Probability Density')
    ax[1].set_xlabel(r'$\Delta y$')
    ax[1].set_ylabel('Probability Density')
    ax[2].set_xlabel(r'$\Delta \theta_E$')
    ax[2].set_ylabel('Probability Density')

    #Add a vertical line at true positions
    for i in range(Nlens):
        ax[0].vlines(true_xlens[i],ymin=0,ymax=1,color=colors[i % len(colors)])
        ax[1].vlines(true_ylens[i],ymin=0,ymax=1,color=colors[i % len(colors)])
        ax[2].vlines(true_erlens[i],ymin=0,ymax=1,color=colors[i % len(colors)])

    plt.savefig('random_realization_{}_lens'.format(Nlens))


if __name__ == '__main__':
    #visualize_algorithm(1,100,100)
    #simple_implementation()
    #bulk_test(10000)
    random_realization(10**5,1,4)
    random_realization(10**5,2,4)