import numpy as np
from utils import chi2, chi2wrapper, lens
import scipy.optimize as opt


def choose_guess(x,y,e1data,e2data,f1data,f2data):
    #Choose a guess for the lens position given the source data
    gamma = np.sqrt(e1data**2+e2data**2)
    flexion = np.sqrt(f1data**2+f2data**2)
    phi_flexion = np.arctan2(f2data,f1data) 

    #look, what if we just use the angle from flexion? We know that this should point right to the lens. 
    #It doesn't seem like we will lose too much information, especially since this is just a starting guess
    phi = phi_flexion 
    r = gamma / flexion

    xguess = x + r * np.cos(phi)
    yguess = y + r * np.sin(phi)
    eRguess = 2 * gamma * np.abs(r)

    return np.array([[xguess], [yguess], [eRguess]])


def initial_minimization(x,y,e1data,e2data,f1data,f2data,sigs,sigf):
    #This function is used to find the initial guess for the lens positions
    ns = len(x)
    xlens = np.zeros(ns)
    ylens = np.zeros(ns)
    telens = np.zeros(ns)
    for i in range(ns):
       #Perform a local minimization
        params = [[x[i]],[y[i]],[e1data[i]],[e2data[i]],[f1data[i]],[f2data[i]],sigf,sigs]
        guess = choose_guess(x[i],y[i],e1data[i],e2data[i],f1data[i],f2data[i])
        res = opt.minimize(chi2wrapper,guess,args=(params))
        xlens[i] = res.x[0]
        ylens[i] = res.x[1]
        telens[i] = res.x[2]
    return xlens, ylens, telens


def list_winnower(xl,yl,eR,x,y,xmax,threshold_distance=1):
    winnowed_x = []
    winnowed_y = []
    winnowed_eR = []

    for i in range(len(xl)):
        #Check if the lens is within threshold_distance of any sources
        too_close = any(np.sqrt((xl[i]-x)**2+(yl[i]-y)**2) < threshold_distance)
        if too_close:
            continue

        #Check if the lens is too far from the center
        if np.sqrt(xl[i]**2+yl[i]**2) > 2*xmax:
            continue      

        #If we get here, the lens is an acceptable candidate
        winnowed_x.append(xl[i])
        winnowed_y.append(yl[i])
        winnowed_eR.append(eR[i])

    return np.array(winnowed_x), np.array(winnowed_y), np.array(winnowed_eR)


def merge_lenses(xlens,ylens,telens, merger_threshold=1):
    #Merge lenses that are within merger_threshold of each other
    for i in range(len(xlens)):
        for j in range(i+1,len(xlens)):
            if np.sqrt((xlens[i]-xlens[j])**2+(ylens[i]-ylens[j])**2) < merger_threshold:
                #Merge the lenses - take a weighted average of positions
                xlens[i] = (xlens[i]*telens[i]+xlens[j]*telens[j])/(telens[i]+telens[j])
                ylens[i] = (ylens[i]*telens[i]+ylens[j]*telens[j])/(telens[i]+telens[j])
                telens[i] = (telens[i]+telens[j])/2
                #Delete the second lens
                xlens = np.delete(xlens,j)
                ylens = np.delete(ylens,j)
                telens = np.delete(telens,j)
                break
        else:
            #If we get here, we have not merged any lenses, so we can move on
            continue
        #If we get here, we have deleted a lens, so we need to start over   
        break

    return xlens, ylens, telens


def iterative_minimizer(xlens,ylens,telens,chi2val,x,y,e1data,e2data,f1data,f2data,sigf,sigs):
    while True:
        #Loop through the lenses and remove them one at a time
        #to see if the chi^2 value improves
        for i in range(len(xlens)):
            xltest = np.delete(xlens,i)
            yltest = np.delete(ylens,i)
            tetest = np.delete(telens,i)
            dof = 4*len(x) - 3*len(xltest) #Each source has 4 data points, each lens has 3 parameters
            chi2test = chi2(x,y,e1data,e2data,f1data,f2data,xltest,yltest,tetest,sigf,sigs) / dof
            if chi2test < chi2val:
                #If we get here, the chi^2 value has improved, so we remove the lens
                xlens = xltest
                ylens = yltest
                telens = tetest
                chi2val = chi2test
                break
        else:
            #If we get here, no lenses were removed, so we have found the best fit
            break
    
    if len(xlens) == 0:
        pass
        #print('Oops, no lenses left!')
    
    return xlens, ylens, telens, chi2val


def perform_minimization(x,y,e1data,e2data,f1data,f2data,sigs,sigf,xmax,flags = False):
    #Make the noise choices global so that we don't have to pass them around
    #This is where the magic happens
    #We run through the full algorithm, given the source data, and return the best fit lens parameters

    xlens, ylens, telens = initial_minimization(x,y,e1data,e2data,f1data,f2data,sigs,sigf)
    
    dof = 4*len(x) - 3*len(xlens) #Each source has 4 data points, each lens has 3 parameters`
    chi2val = chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,telens,sigf,sigs) / dof #Normalize by the number of degrees of freedom


    if flags:
        print('Initial chi^2: ', chi2val)
        print('Initial number of lenses: ', len(xlens))
        print('Initial number of sources: ', len(x))
        print('Initial number of degrees of freedom: ', dof)

    xlens,ylens,telens = list_winnower(xlens,ylens,telens,x,y,xmax)

    if flags:
        print('Number of lenses after winnowing: ', len(xlens))
        chi2val = chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,telens,sigf,sigs)
        print('Chi^2 after winnowing: ', chi2val)

    xlens,ylens,telens = merge_lenses(xlens,ylens,telens)
    
    if flags:
        print('Number of lenses after merging: ', len(xlens))
        chi2val = chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,telens,sigf,sigs)
        print('Chi^2 after merging: ', chi2val)


    xlens,ylens,telens,chi2val = iterative_minimizer(xlens,ylens,telens,chi2val,x,y,e1data,e2data,f1data,f2data,sigf,sigs)

    if flags:
        print('Number of lenses after iterative minimization: ', len(xlens))
        chi2val = chi2(x,y,e1data,e2data,f1data,f2data,xlens,ylens,telens,sigf,sigs)
        print('Chi^2 after iterative minimization: ', chi2val)

    return xlens, ylens, telens, chi2val