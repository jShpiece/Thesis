import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

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


def lens(x,y,xlarr,ylarr,tearr):
    #Compute the lensing signals on a single source 
    #from a set of lenses
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


def chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs,fwgt=1.0,swgt=1.0):
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


def chi2wrapper(guess,params):
    return chi2(params[0],params[1],params[2],params[3],params[4],params[5],guess[0],guess[1],guess[2],params[6],params[7])
    

def list_winnower(xl,yl,eR):
    #Given a set of lens candidates, remove any that aren't 
    #considered viable. Follow the following rules
    #1. If there are multiple lenses within 1 arcsecond of each other,
    #   only keep the one with the highest eR
    #2. If there are any lenses with negative einstein radii, remove them
    #3. If there are any lenses with eR > 60, remove them
    #4. If there are any lenses with eR < 1, remove them
    #5. If there are lenses outside of the image, remove them

    #print('Starting with ',len(eR),' candidate lenses')

    #First, remove any lenses with negative einstein radii - as well as any too small to be realisticly detected
    #Also remove any lenses with eR > 60 (these are likely to be spurious)
    good = np.where((eR > 10**-3) & (eR < 60))[0]
    #print('Removing ',len(eR)-len(good),' lenses with negative eR')
    xl = xl[good]
    yl = yl[good]
    eR = eR[good]

    #Next, remove any lenses outside of the image
    good = np.where((xl > -100) & (xl < 100) & (yl > -100) & (yl < 100))[0]
    #print('Removing ',len(eR)-len(good),' lenses outside of the image')
    xl = xl[good]
    yl = yl[good]
    eR = eR[good] 

    return xl,yl,eR


def iterative_minimizer(xlens,ylens,telens,chi2val):
    while True:
        #Loop through the lenses and remove them one at a time
        #to see if the chi^2 value improves
        for i in range(len(xlens)):
            xltest = np.delete(xlens,i)
            yltest = np.delete(ylens,i)
            tetest = np.delete(telens,i)
            chi2test = chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs)
            if chi2test < chi2val:
                xlens = xltest
                ylens = yltest
                telens = tetest
                chi2val = chi2test
                break
        else:
            print('No more lenses to remove')
            break
    
    if len(xlens) == 0:
        print('Oops, no lenses left!')
    
    return xlens, ylens, telens, chi2val


if __name__ == '__main__':
    # Set up the true lens configuration
    nlens=3 # Number of lenses
    xmax=10. # The range to consider for the lenses.
    xlarr,ylarr,tearr=createLenses(nlens=nlens,randompos=False,xmax=xmax)

    # Source parameters
    ns=10 # Number of sources
    x,y,e1data,e2data,f1data,f2data=createSources(xlarr,ylarr,tearr,ns=ns,sigf=sigf,sigs=sigs,randompos=True,xmax=xmax)

    #perform local minimization
    xlens = np.zeros(ns)
    ylens = np.zeros(ns)
    telens = np.zeros(ns)
    for i in range(ns):
        #Perform a local minimization\
        params = [[x[i]],[y[i]],[e1data[i]],[e2data[i]],[f1data[i]],[f2data[i]],sigf,sigs]
        guess = [np.random.rand()*xmax,np.random.rand()*xmax,1]
        res = opt.minimize(chi2wrapper,guess,args=(params))
        xlens[i] = res.x[0]
        ylens[i] = res.x[1]
        telens[i] = res.x[2]
    chi2val = chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,telens,sigf,sigs)

    #Perform the winnowing
    xlens, ylens, telens = list_winnower(xlens,ylens,telens)
    #Perform iterative minimization
    xlens, ylens, telens, chi2val = iterative_minimizer(xlens,ylens,telens,chi2val)

    #Plot the results
    plt.figure()
    plt.title('Lensing field reconstruction \n chi^2 = '+str(np.round(chi2val,3)))
    plt.scatter(xlarr,ylarr,s=tearr*50,color='red',marker='x',label='True Lenses',alpha=1.0)
    plt.scatter(xlens,ylens,color='blue',s=telens*50,label='Recovered Lenses: eR = '+str(np.round(telens,3)),alpha=1.0)
    plt.xlim(-xmax,xmax)
    plt.ylim(-xmax,xmax)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()