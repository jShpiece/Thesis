import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pipeline
from utils import createLenses, createSources

sigf = 0.01
sigs = 0.1

if __name__ == '__main__':
    '''Note to self - chi2 should be calculated per degree of freedom'''
    # Set up the true lens configuration
    nlens=1 # Number of lenses
    xmax=4. # The range to consider for the lenses.
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=False,xmax=xmax)

    # Set up the true source configuration
    ns = 4
    x,y,e1data,e2data,f1data,f2data=createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    # Run the minimization
    xlens,ylens,eRlens,chi2 = pipeline.perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs=sigs,sigf=sigf,xmax=xmax,flags=True)

    # Plot the results
    plt.figure()
    plt.scatter(xlens,ylens,color='red',label='Recovered Lenses: eR = '+str(eRlens))
    plt.scatter(x,y,color='blue',label='Sources')
    plt.scatter(xlarr,ylarr,marker='x',color='green',label='True Lenses')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-xmax,xmax)
    plt.ylim(-xmax,xmax)
    plt.gca().set_aspect('equal')
    plt.title(r'Result of Minimization: $\chi_\nu^2$ = '+str(np.round(chi2,2)))
    plt.show()
