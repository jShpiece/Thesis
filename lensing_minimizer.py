import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
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
        e1data+=sigs*np.random.normal(ns) 
        f1data+=sigf*np.random.normal(ns)
        if randompos == True:
            e2data+=sigs*np.random.normal(ns)
            f2data+=sigf*np.random.normal(ns)
    return x,y,e1data,e2data,f1data,f2data


def lens(x,y,xlarr,ylarr,tearr):
    dx = x-xlarr
    dy = y-ylarr
    r = np.sqrt(dx**2+dy**2)
    cosphi = dx/r
    sinphi = dy/r
    cos2phi = cosphi*cosphi-sinphi*sinphi
    sin2phi = 2*cosphi*sinphi

    f1 = np.sum(-dx*tearr/(2*r*r*r))
    f2 = np.sum(-dy*tearr/(2*r*r*r))

    e1 = np.sum(-tearr/(2*r)*cos2phi)
    e2 = np.sum(-tearr/(2*r)*sin2phi)

    return e1,e2,f1,f2


def eR_prior(eR):
    alpha = 0.95
    beta = 0.22
    return beta * eR**(-alpha)


def signal_prior(signal1, signal2, sigma):
    #Gaussian prior on signal
    prior = np.exp(-((signal1 + signal2)**2)/(2*sigma**2))
    norm = 1.0/np.sqrt(2*np.pi*sigma**2)
    return norm*prior 


def chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs,fwgt=1.0,swgt=1.0):
    #Okay, lets introduce some weights to act as a prior
    #We want to penalize unrealistic values of theta_E
    #As well as of the shear and flexion
    #We can do this by adding a prior to the chi^2

    #First, we need to compute the prior

    #Now compute the chi^2
    chi2val = 0.0
    for i in range(len(x)):
        e1,e2,f1,f2 = lens(x[i],y[i],xltest,yltest,tetest)
        chif1 = (f1data[i]-f1)**2 / (sigf**2)
        chif2 = (f2data[i]-f2)**2 / (sigf**2)
        chie1 = (e1data[i]-e1)**2 / (sigs**2)
        chie2 = (e2data[i]-e2)**2 / (sigs**2)

        chi2val += fwgt * (chif1 + chif2) + swgt * (chie1 + chie2) 


    return chi2val


def minChi2(x,y,e1data,e2data,f1data,f2data,xlstart,ylstart,testart,sigf,sigs):
    chi2best = chi2(x,y,e1data,e2data,f1data,f2data,xlstart,ylstart,testart,sigf,sigs)
    xlbest = xlstart
    ylbest = ylstart
    tebest = testart
    nlens = len(xlbest)
    npars = 3*nlens
    dpar = 1e-6

    niter = 20
    alpha0 = 0.8
    alpha = alpha0

    for iter in range(niter):
        gradpar = np.zeros(npars)
        for ipar in range(npars):
            xltest = xlbest.copy()
            yltest = ylbest.copy()
            tetest = tebest.copy()
            if int(ipar/nlens) == 0:
                xltest[ipar] += dpar
            elif int(ipar/nlens) == 1:
                yltest[ipar-nlens] += dpar
            elif int(ipar/nlens) == 2:
                tetest[ipar-2*nlens] += dpar
            gradpar[ipar] = (chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs)-chi2best)/dpar
        deltapar = -alpha*gradpar*chi2best / np.sum(gradpar*gradpar)
        xltest = xlbest.copy() + deltapar[0:nlens]
        yltest = ylbest.copy() + deltapar[nlens:2*nlens]
        tetest = tebest.copy() + deltapar[2*nlens:3*nlens]
        chi2test = chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs)
        if chi2test < chi2best:
            xlbest = xltest.copy()
            ylbest = yltest.copy()
            tebest = tetest.copy()
            chi2best = chi2test
        else:
            #print('Step too large, reducing step size')
            alpha = 0.8*alpha
    return xlbest,ylbest,tebest,chi2best


def chi2plot(x,y,e1data,e2data,f1data,f2data,xmax=2,nx=100):
    #Assume theta_E = 1, minimize chi2 with respect to x and y
    dx = 2*xmax/nx
    chi2_all = np.zeros((nx,nx))
    chi2_flex = np.zeros((nx,nx))
    chi2_shear = np.zeros((nx,nx))
    for i in range(nx):
        x1=-xmax+dx*(i+0.5)
        for j in range(nx):
            y1=-xmax+dx*(j+0.5)
            chi2_all[i,j] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([1.0]),sigf,sigs)
            chi2_flex[i,j] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([0.0]),sigf,sigs,swgt=0.0)
            chi2_shear[i,j] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([1.0]),sigf,sigs,fwgt=0.0)
    
    myval = np.exp(-chi2_all.T)
    plt.imshow(myval,interpolation='none',origin='lower',extent=[-xmax,xmax,-xmax,xmax])
    plt.plot(x,y,'x','o',color='red')
    plt.savefig('Images/chi2_all.png')
    plt.close()

    plt.imshow(np.log(chi2_flex.T),interpolation='none',origin='lower',extent=[-xmax,xmax,-xmax,xmax])
    plt.plot(x,y,'x','o',color='red')
    plt.savefig('Images/chi2_flex.png')
    plt.close()

    plt.imshow(np.log(chi2_shear.T),interpolation='none',origin='lower',extent=[-xmax,xmax,-xmax,xmax])
    plt.plot(x,y,'x','o',color='red')
    plt.savefig('Images/chi2_shear.png')
    plt.close()


def chi2plot1d(x,y,e1data,e2data,f1data,f2data,xmax=2,nx=100):
    #Assume theta_E = 1, minimize chi2 with respect to x and y
    dx = 2*xmax/nx
    chi2_all = np.zeros((nx,nx))
    chi2_flex = np.zeros((nx,nx))
    chi2_shear = np.zeros((nx,nx))
    xarr=[]
    for i in range(nx):
        x1=-xmax+dx*(i+0.5)
        xarr.append(x1)
        y1=0.0
        chi2_all[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([1.0]),sigf,sigs)
        chi2_flex[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([1.0]),sigf,sigs,swgt=0.0)
        chi2_shear[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x1]),np.asarray([y1]),np.asarray([1.0]),sigf,sigs,fwgt=0.0)
    
    x0 = x*0.0
    '''
    plt.plot(x,x0,'o',color='red')
    plt.plot(xarr,np.log(chi2_all),'-',color='black')
    plt.savefig('Images/chi2_all_1d.png')
    plt.close()

    plt.plot(x,x0,'o',color='red')
    plt.plot(xarr,np.log(chi2_flex),'-',color='black')
    plt.savefig('Images/chi2_flex_1d.png')
    plt.close()

    plt.plot(x,x0,'o',color='red')
    plt.plot(xarr,np.log(chi2_shear),'-',color='black')
    plt.savefig('Images/chi2_shear_1d.png')
    plt.close()
    '''
    
    return xarr,chi2_all,chi2_flex,chi2_shear


def minimizer_comparison(Niter):
    #Compare the home-grown minimizer to scipy.optimize

    #Set up configuration
    nlens = 1
    xmax = 2.
    xlarr, ylarr, tearr = createLenses(nlens=nlens,randompos=False,xmax=xmax)

    #Set up sources
    ns = 1

    #Create result arrays
    xhome = np.zeros(Niter)
    yhome = np.zeros(Niter)
    tehome = np.zeros(Niter)
    chi2home = np.zeros(Niter)

    xscipy = np.zeros(Niter)
    yscipy = np.zeros(Niter)
    tescipy = np.zeros(Niter)
    chi2scipy = np.zeros(Niter)

    for iter in range(Niter):
        x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns,randompos=True,sigf=sigf,sigs=sigs,xmax=xmax)

        #Make a plot in a 2x2 range showing chi^2

        #chi2plot(x,y,e1data,e2data,f1data,f2data,xmax=xmax,nx=100)
        #chi2plot1d(x,y,e1data,e2data,f1data,f2data,xmax=6,nx=1000)

        #Now minimize chi^2
        xlstart = -xmax+2*xmax*np.random.random(nlens)
        ylstart = -xmax+2*xmax*np.random.random(nlens)
        testart = np.ones(nlens)
        xlbest,ylbest,tebest,chi2best = minChi2(x,y,e1data,e2data,f1data,f2data,xlstart,ylstart,testart,sigf,sigs)
        #print('Best fit parameters: ',xlbest,ylbest,tebest,chi2best)

        def chi2wrapper(guess, params):
            return chi2(params[0],params[1],params[2],params[3],params[4],params[5],guess[0],guess[1],guess[2],params[6],params[7])

        #Compare to scipy.optimize
        guess = np.concatenate((xlstart,ylstart,testart))
        params = [x,y,e1data,e2data,f1data,f2data,sigf,sigs]
        result = optimize.minimize(chi2wrapper,guess,args=(params))
        #print('Scipy result: ',result.x,result.fun)

        #Store results
        xhome[iter] = xlbest[0]
        yhome[iter] = ylbest[0]
        tehome[iter] = tebest[0]
        chi2home[iter] = chi2best

        xscipy[iter] = result.x[0]
        yscipy[iter] = result.x[1]
        tescipy[iter] = result.x[2]
        chi2scipy[iter] = result.fun

    #Remove outliers
    xhome = xhome[np.abs(xhome)<xmax*100]
    yhome = yhome[np.abs(yhome)<xmax*100]
    tehome = tehome[np.abs(tehome)<10]
    chi2home = chi2home[np.abs(chi2home)<1000]

    xscipy = xscipy[np.abs(xscipy)<xmax*100]
    yscipy = yscipy[np.abs(yscipy)<xmax*100]
    tescipy = tescipy[np.abs(tescipy)<10]
    chi2scipy = chi2scipy[np.abs(chi2scipy)<1000]

    #Plot the results
    fig, ax = plt.subplots(1,4,figsize=(20,5))
    fig.suptitle('Comparison of minimizers: {} iterations'.format(Niter))

    fancyhist(xhome, ax=ax[0], bins='scott', histtype='step', density=True, color='k',label='Home: mean = {:.3f}, std = {:.3f}'.format(np.mean(xhome),np.std(xhome)))
    fancyhist(xscipy, ax=ax[0], bins='scott', histtype='step', density=True, color='r',label='Scipy: mean = {:.3f}, std = {:.3f}'.format(np.mean(xscipy),np.std(xscipy)))

    fancyhist(yhome, ax=ax[1], bins='scott', histtype='step', density=True, color='k',label='Home: mean = {:.3f}, std = {:.3f}'.format(np.mean(yhome),np.std(yhome)))
    fancyhist(yscipy, ax=ax[1], bins='scott', histtype='step', density=True, color='r',label='Scipy: mean = {:.3f}, std = {:.3f}'.format(np.mean(yscipy),np.std(yscipy)))

    fancyhist(tehome, ax=ax[2], bins='scott', histtype='step', density=True, color='k',label='Home: mean = {:.3f}, std = {:.3f}'.format(np.mean(tehome),np.std(tehome)))
    fancyhist(tescipy, ax=ax[2], bins='scott', histtype='step', density=True, color='r',label='Scipy: mean = {:.3f}, std = {:.3f}'.format(np.mean(tescipy),np.std(tescipy)))

    fancyhist(chi2home, ax=ax[3], bins='scott', histtype='step', density=True, color='k',label='Home: mean = {:.3f}, std = {:.3f}'.format(np.mean(chi2home),np.std(chi2home)))
    fancyhist(chi2scipy, ax=ax[3], bins='scott', histtype='step', density=True, color='r',label='Scipy: mean = {:.3f}, std = {:.3f}'.format(np.mean(chi2scipy),np.std(chi2scipy)))

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('y')
    ax[2].set_xlabel('theta_E')
    ax[3].set_xlabel('chi2')

    ax[0].set_xlim(-10,10)
    ax[1].set_xlim(-10,10)
    ax[2].set_xlim(-10,10)
    #ax[3].set_xlim(0,100)

    plt.savefig('Images/chi2_comparison.png')
    plt.show()


def iterative_minimization(xs,ys,e1data,e2data,f1data,f2data,title):
    #Perform a minimization with the n=1 sources, then add a source and re-minimize until all sources are included
    ns = len(xs)
    nlens = 1
    xmax = 2.

    #Take an inital guess.  We assume we know (or can guess) the number of lenses
    ylstart=np.zeros(nlens)
    xlstart=-xmax+2*xmax*np.random.random(nlens)
    testart=np.ones(nlens)

    xlstart1 = xlstart
    ylstart1 = ylstart
    testart1 = testart

    #Initialize the plot
    plt.figure( figsize=(10,10) )
    plt.title('Iterative minimization: {}'.format(title))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'olive', 'black']

    xfit = []
    chi2fit = []


    for i in range(ns):
        #print(title + ' Adding source ',i+1,' of ',ns)
        xlbest, ylbest, erbest, chi2best = minChi2(xs[0:i+1],ys[0:i+1],e1data[0:i+1],e2data[0:i+1],f1data[0:i+1],f2data[0:i+1],xlstart1,ylstart1,testart1,sigf,sigs)
        #Update the initial guess
        xlstart1 = xlbest
        ylstart1 = ylbest
        testart1 = erbest

        #Store the results
        xfit.append(xlbest[0])
        chi2fit.append(chi2best)

        #print('Best fit parameters: ',xlbest,ylbest,erbest,chi2best)
        x,chi2all,chi2f,chi2s = chi2plot1d(xs[0:i+1],ys[0:i+1],e1data[0:i+1],e2data[0:i+1],f1data[0:i+1],f2data[0:i+1],xmax=6,nx=1000)
        plt.plot(x, np.log(chi2all), color=colors[i], alpha = 0.2)
        plt.scatter(xlbest, np.log(chi2best), color=colors[i], label = 'Best fit: x = {:.3f}, chi2 = {:.2f}'.format(xlbest[0],chi2best))
        
    plt.xlabel('x')
    plt.ylabel('log(chi2)')
    plt.legend()
    
    plt.savefig('Images/iterative_minimization_{}.png'.format(title))
    print('Saved Images/iterative_minimization_{}.png'.format(title))
    plt.close()

    return xfit, chi2fit


if __name__ == '__main__':
    '''
    # Set up the true lens configuration
    nlens=1
    xmax=2. # The range to consider for the lenses.
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=False,xmax=xmax)

    # Source parameters
    ns=1
    x,y,e1data,e2data,f1data,f2data=createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=False)

    xgrid = np.linspace(-10,10,100)
    dx1 = x[0] - xgrid

    eR_flex = -2 * dx1 * np.abs(dx1) * f1data[0]
    eR_shear = - 2 * np.abs(dx1) * e1data[0]

    xsolution = e1data[0] / f1data[0]
    eRsolution = -2 * xsolution * np.abs(xsolution) * f1data[0]
    print('Analytic solution: ',xsolution,eRsolution)
    print('True solution: ',x[0] - xlarr[0],1.0)
    print('Error: ',xsolution - (x[0] - xlarr[0]),eRsolution - 1.0)

    plt.figure()
    plt.plot(dx1, eR_flex, label='eR flexion prediction')
    plt.plot(dx1, eR_shear, label='eR shear prediction')
    plt.scatter(x[0] - xlarr, tearr, label='true lens')
    plt.scatter(xsolution, eRsolution, label='analytic solution')
    plt.legend()
    plt.xlabel('dx - distance from the source')
    plt.ylabel('eR')
    plt.savefig('Images/eR.png')
    plt.show()
    plt.close()


    #Evaluate chi2 for the analytic solution and the true solution
    chi2_analytic = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xsolution]),np.asarray([0.0]),np.asarray([eRsolution]),sigf,sigs,fwgt=0.0)
    chi2_true = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xlarr[0]]),np.asarray([0.0]),np.asarray([1.0]),sigf,sigs,fwgt=0.0)
    #chi2_zero = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([x[0]]),np.asarray([0.0]),np.asarray([0.0]),sigf,sigs,swgt=0.0)

    #Perform minimization with different initial guesses
    xrandom = -xmax+2*xmax*np.random.random(nlens)
    yguess = 0.0
    eRguess = 1.0

    xlbest, ylbest, erbest, chi2best = minChi2(x,y,e1data,e2data,f1data,f2data,xrandom,ylarr,tearr,sigf,sigs)
    print('Best fit parameters - random guess: ',xlbest,ylbest,erbest,chi2best)

    xlbest, ylbest, erbest, chi2best = minChi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xsolution]),np.asarray([0.0]),np.asarray([eRsolution]),sigf,sigs)
    print('Best fit parameters - analytic solution: ',xlbest,ylbest,erbest,chi2best)

    xlbest, ylbest, erbest, chi2best = minChi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xlarr[0]]),np.asarray([0.0]),np.asarray([1.0]),sigf,sigs)
    print('Best fit parameters - true solution: ',xlbest,ylbest,erbest,chi2best)

    
    print('Chi2 for analytic solution: ',chi2_analytic)
    print('Chi2 for true solution: ',chi2_true)
    #print('Chi2 for zero solution: ',chi2_zero)

    chi2_flex = np.zeros(100) # chi2 for flexion only, because we're making this prediction from flexion only
    chi2_shear = np.zeros(100) # chi2 for shear only, because we're making this prediction from shear only
    chi2_all1 = np.zeros(100) # chi2 for all parameters, using the guess from flexion
    chi2_all2 = np.zeros(100) # chi2 for all parameters, using the guess from shear

    for i in range(100):
        chi2_flex[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xgrid[i]]),np.asarray([0.0]),np.asarray([eR_flex[i]]),sigf,sigs,swgt=0.0)
        chi2_shear[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xgrid[i]]),np.asarray([0.0]),np.asarray([eR_shear[i]]),sigf,sigs,fwgt=0.0)
        chi2_all1[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xgrid[i]]),np.asarray([0.0]),np.asarray([eR_flex[i]]),sigf,sigs)
        chi2_all2[i] = chi2(x,y,e1data,e2data,f1data,f2data,np.asarray([xgrid[i]]),np.asarray([0.0]),np.asarray([eR_shear[i]]),sigf,sigs)

    plt.figure()
    plt.plot(xgrid, np.log(chi2_flex+10**-20), label='chi2_flex')
    plt.plot(xgrid, np.log(chi2_shear+10**-20), label='chi2_shear')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('ln(chi2)')
    plt.savefig('Images/chi2_flex.png')
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(xgrid, np.log(chi2_all1+10**-20), label='chi2 - flex prediction')
    plt.plot(xgrid, np.log(chi2_all2+10**-20), label='chi2 - shear prediction')
    plt.scatter(xlarr, tearr, label='true')
    plt.scatter(x[0], 1.0, label='source')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('ln(chi2)')
    plt.savefig('Images/chi2.png')
    plt.show()
    plt.close()

    '''
  
    #Set up configuration
    nlens = 1
    xmax = 2.
    xlarr, ylarr, tearr = createLenses(nlens=nlens,randompos=False,xmax=xmax)

    #Set up sources
    ns = 10
    xs, ys, e1data, e2data, f1data, f2data = createSources(xlarr,ylarr,tearr,ns,randompos=False,sigf=sigf,sigs=sigs,xmax=xmax)

    #Create plot for each possible combination of sources
    #For 10 sources, there are 10! = 3628800 possible combinations
    #Clearly, we can't make a plot for each of these
    #Instead, we'll choose certain orders that we think are interesting
    #Lets take 5 possible orders: two random orders, an order arranged by flexion, by shear, and by x (this last one should also essentially be random)

    order1 = [0,1,2,3,4,5,6,7,8,9]
    order2 = [9,8,7,6,5,4,3,2,1,0]
    #Find the order arranged by flexion (small to large)
    order3 = np.argsort(np.abs(f1data))
    order3 = order3[::-1]
    #Find the order arranged by shear
    order4 = np.argsort(np.abs(e1data))
    order4 = order4[::-1]
    #Find the order arranged by x
    order5 = np.argsort(xs)

    #Now make the plots
    xfit1, chi2fit1 = iterative_minimization(xs[order1],ys[order1],e1data[order1],e2data[order1],f1data[order1],f2data[order1],'random_ordering_1')
    xfit2, chi2fit2 = iterative_minimization(xs[order2],ys[order2],e1data[order2],e2data[order2],f1data[order2],f2data[order2],'random_ordering_2')
    xfit3, chi2fit3 = iterative_minimization(xs[order3],ys[order3],e1data[order3],e2data[order3],f1data[order3],f2data[order3],'flexion_ordering')
    xfit4, chi2fit4 = iterative_minimization(xs[order4],ys[order4],e1data[order4],e2data[order4],f1data[order4],f2data[order4],'shear_ordering')
    xfit5, chi2fit5 = iterative_minimization(xs[order5],ys[order5],e1data[order5],e2data[order5],f1data[order5],f2data[order5],'x_ordering')

    plt.figure()
    plt.title('Comparison of iterative minimization')
    plt.plot(order1, xfit1, '-', color='red', label='Random ordering 1')
    plt.plot(order1, xfit2, '-', color='blue', label='Random ordering 2')
    plt.plot(order1, xfit3, '-', color='green', label='Flexion ordering')
    plt.plot(order1, xfit4, '-', color='orange', label='Shear ordering')
    plt.plot(order1, xfit5, '-', color='purple', label='x ordering')
    plt.xlabel('Source number')
    plt.ylabel('Best fit x')
    plt.legend()
    plt.ylim(-6,6)
    plt.savefig('Images/iterative_minimization_comparison_ns_{}.png'.format(ns))
    