import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import createLenses, createSources, lens
import time

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
   
    return x,y,e1data,e2data,f1data,f2data


def simple_implementation():
    #Runs a very simple test of the algorithm, no inputs required
    '''Note to self - chi2 should be calculated per degree of freedom'''
    start = time.time()
    # Set up the true lens configuration
    nlens = 2 # Number of lenses
    xmax = 2 # The range to consider for the lenses.
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=True,xmax=xmax)

    # Set up the true source configuration
    ns = 2
    x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Run the minimization
    xlens,ylens,eRlens,chi2val = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)
    stop = time.time()
    print('Time elapsed: ', stop-start)
    # Plot the results
    plt.figure()
    plt.scatter(xlens,ylens,color='red',label=r'Recovered Lenses: $\theta_E$ = {}'.format(eRlens))
    plt.scatter(x,y,color='blue',label='Sources')
    plt.scatter(xlarr,ylarr,marker='x',color='green',label='True Lenses')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-xmax,xmax)
    plt.ylim(-xmax,xmax)
    plt.gca().set_aspect('equal')
    plt.title('Lensing Reconstruction \n $\chi^2$ = %.2f' % chi2val)
    plt.show()



def bulk_test(ntests): 
    Nsources = [1,2,3,4,5,6,7,8,9,10]

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
            true_chi2 = pipeline.chi2(x,y,e1data,e2data,f1data,f2data,xlarr,ylarr,tearr,sigf,sigs)

            # Run the minimization
            xlens,ylens,eRlens,chi2val = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=False)
            if len(xlens) == 0:
                nempty += 1

            if chi2val > true_chi2:
                n_badfit += 1
        
        #Print results 
        print('| {} | {} | {} |'.format(N, nempty/ntests, n_badfit/ntests))


if __name__ == '__main__':
    bulk_test(100)
