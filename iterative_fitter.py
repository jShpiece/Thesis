import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import createLenses, createSources, lens
import time
import warnings
from astropy.visualization import hist as fancyhist

sigf = 0.01
sigs = 0.1


def createLenses(nlens=1,randompos=True,xmax=10):
    #For now, fix theta_E at 1
    tearr = np.ones(nlens) 
    if randompos == True:
        xlarr = -xmax + 2*xmax*np.random.random(nlens)
        ylarr = -xmax + 2*xmax*np.random.random(nlens)
    else: #Uniformly spaced lenses
        xlarr = -xmax + 2*xmax*(np.arange(nlens)+0.5)/(nlens)
        ylarr = np.zeros(nlens)
    return xlarr, ylarr, tearr


def createSources(xlarr,ylarr,tearr,ns=1,randompos=True,sigf=0.1,sigs=0.1,xmax=5):
    if randompos == True:
        x = -xmax + 2*xmax*np.random.random(ns)
        y = -xmax + 2*xmax*np.random.random(ns)
    else: #Uniformly spaced sources
        x = -xmax + 2*xmax*np.random.random(ns) #Let the sources be randomly distributed in x only
        y = np.zeros(ns)

    #Apply the lens 
    e1data = np.zeros(ns)
    e2data = np.zeros(ns)
    f1data = np.zeros(ns)
    f2data = np.zeros(ns)

    for i in range(ns):
        e1data[i],e2data[i],f1data[i],f2data[i] = lens(x[i],y[i],xlarr,ylarr,tearr)
    
    #Add noise
    e1data += np.random.normal(0,sigs,ns)
    e2data += np.random.normal(0,sigs,ns)
    f1data += np.random.normal(0,sigf,ns)
    f2data += np.random.normal(0,sigf,ns)

    #Clean up the data - this means removing sources that are too close to the lenses
    #But lets not cheat, because in real life we won't know where the lenses are
    #Instead, we check for flexion signals that are too strong

    for i in range(ns):
        if np.abs(f1data[i]) > 1 or np.abs(f2data[i]) > 1:
            e1data[i] = 0
            e2data[i] = 0
            f1data[i] = 0
            f2data[i] = 0
            #No need to remove the source, just set all its signals to zero
   
    return x,y,e1data,e2data,f1data,f2data


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
    xlens,ylens,eRlens,chi2val = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=True)

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
    xlens,ylens,eRlens = pipeline.initial_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf)
    chi2val = pipeline.chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,eRlens,sigf,sigs)
    dof = 4*len(x) - 3*len(xlens) #Each source has 4 data points, each lens has 3 parameters
    chi2val /= dof #Normalize by the number of degrees of freedom


    # Plot the results
    plotter(ax[0,0],xlens,ylens,eRlens,chi2val,'Initial Minimization')

    # Perform the winnowing
    xlens,ylens,eRlens = pipeline.list_winnower(xlens,ylens,eRlens,x,y,xmax)

    # Plot the results
    plotter(ax[0,1],xlens,ylens,eRlens,chi2val,'Winnowing')

    # Perform the merging
    xlens,ylens,eRlens = pipeline.merge_lenses(xlens,ylens,eRlens)
    
    # Plot the results
    plotter(ax[1,0],xlens,ylens,eRlens,chi2val,'Merging')

    # Perform the iterative minimization
    xlens,ylens,eRlens,chi2val = pipeline.iterative_minimizer(xlens,ylens,eRlens,chi2val,x,y,e1data,e2data,f1data,f2data,sigf,sigs)

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
            xlens,_,_,chi2val = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)
            if len(xlens) == 0:
                nempty += 1

            if chi2val > true_chi2:
                n_badfit += 1
        
        #Print results 
        print('| {} | {} | {} |'.format(N, nempty/ntests, n_badfit/ntests))


def random_realization(Ntrials):
    warnings.simplefilter('ignore')
    #Test the accuracy of the algorithm on a random realization of lenses and sources
    Nlens = 1
    Nsource = 5
    xmax = 2

    dx = np.zeros(Ntrials)
    dy = np.zeros(Ntrials)
    dtheta = np.zeros(Ntrials)

    for i in range(Ntrials):
        #Create the lenses and sources
        xlarr,ylarr,tearr=createLenses(nlens=Nlens,randompos=True,xmax=xmax)
        x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=Nsource,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

        #Run the minimization
        xlens,ylens,eRlens,_ = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)

        #Calculate the difference between the true and recovered lens parameters - right now, take the recovered lens closest to the true lens
        if len(xlens) == 0:
            #If no lenses were recovered, set the difference to infinity
            #dx[i] = np.inf
            #dy[i] = np.inf
            #dtheta[i] = np.inf
            continue
        else:
            closest_lens = np.argmin((xlarr-xlens)**2 + (ylarr-ylens)**2)
            dx[i] = xlarr[0] - xlens[closest_lens]
            dy[i] = ylarr[0] - ylens[closest_lens]
            dtheta[i] = tearr[0] - eRlens[closest_lens]

    #Remove the infinities and nans
    dx = dx[np.isfinite(dx)]
    dy = dy[np.isfinite(dy)]
    dtheta = dtheta[np.isfinite(dtheta)]

    #Store the results
    np.save('dx.npy',dx)
    np.save('dy.npy',dy)
    np.save('dtheta.npy',dtheta)

    #Plot the results
    fig,ax = plt.subplots(1,3,figsize=(15,5))
    fig.suptitle('Random Realization Test - {} lenses, {} sources'.format(Nlens,Nsource))

    fancyhist(dx,ax=ax[0],bins='scott', histtype='step',density=True)
    ax[0].set_xlabel(r'$\Delta x$')
    ax[0].set_ylabel('Probability Density')
    fancyhist(dy,ax=ax[1],bins='scott', histtype='step',density=True)
    ax[1].set_xlabel(r'$\Delta y$')
    ax[1].set_ylabel('Probability Density')
    fancyhist(dtheta,ax=ax[2],bins='scott', histtype='step',density=True)
    ax[2].set_xlabel(r'$\Delta \theta_E$')
    ax[2].set_ylabel('Probability Density')

    plt.savefig('random_realization_{}_lens'.format(Nlens))
    plt.show()




if __name__ == '__main__':
    #visualize_algorithm(1,100,100)
    #simple_implementation()
    #bulk_test(10000)
    random_realization(10000)