import numpy as np
import matplotlib.pyplot as plt

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

    e1 = np.sum(-tearr/r*cos2phi)
    e2 = np.sum(-tearr/r*sin2phi)

    return e1,e2,f1,f2

def chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs,fwgt=1.0,swgt=1.0):
    #Note - this is a simplified model, without penalties from bayesian priors

    chi2val = 0.0
    for i in range(len(x)):
        e1,e2,f1,f2 = lens(x[i],y[i],xltest,yltest,tetest)
        chi2val += (fwgt*(f1data[i]-f1)**2)/(sigf**2) + (fwgt*(f2data[i]-f2)**2)/(sigf**2) + (swgt*(e1data[i]-e1)**2)/(sigs**2) + (swgt*(e2data[i]-e2)**2)/(sigs**2)

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
            print('Step too large, reducing step size')
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



#Set up configuration
nlens = 1
xmax = 2.
xlarr, ylarr, tearr = createLenses(nlens=nlens,randompos=False,xmax=xmax)

#Set up sources
ns = 2
sigf = 0.01
sigs = 0.1
x,y,e1data,e2data,f1data,f2data = createSources(xlarr,ylarr,tearr,ns,randompos=True,sigf=sigf,sigs=sigs,xmax=xmax)

#Make a plot in a 2x2 range showing chi^2

chi2plot(x,y,e1data,e2data,f1data,f2data,xmax=xmax,nx=100)
chi2plot1d(x,y,e1data,e2data,f1data,f2data,xmax=6,nx=1000)

#Now minimize chi^2
xlstart = -xmax+2*xmax*np.random.random(nlens)
ylstart = -xmax+2*xmax*np.random.random(nlens)
testart = np.ones(nlens)
xlbest,ylbest,tebest,chi2best = minChi2(x,y,e1data,e2data,f1data,f2data,xlstart,ylstart,testart,sigf,sigs)
print('Best fit parameters: ',xlbest,ylbest,tebest,chi2best)