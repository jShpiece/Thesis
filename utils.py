#This script holds useful utility functions for my research
import numpy as np
import matplotlib.pyplot as plt

def printProgressBar(iteration, total, prefix = '', suffix = '', 
                     decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print() 


def convolver(img, kernel):
    '''This function takes an image array and 
    convolves it with some PSF kernel in fourier space'''
    img_ft = np.fft.fftn(img, norm='ortho')
    kernel_ft = np.fft.fftn(kernel, norm='ortho')
    convolved_img_fourier = img_ft * kernel_ft
    convolved_img =  np.real(np.fft.fftshift(
        np.fft.ifftn(convolved_img_fourier, img.shape, norm='ortho')))
    return convolved_img


def makeGaussian(stamp, sigma):
    '''Creates a 2-d circular gaussian kernel 
    with a given stamp size and sigma'''
    yp, xp = np.mgrid[-stamp / 2:stamp / 2, -stamp / 2:stamp / 2]  # coordinates in postage stamp
    f_g = np.exp(-((xp / sigma) ** 2 + (yp / sigma) ** 2) / 2)
    f_g /= np.sum(f_g)
    return f_g


def stn_flexion(eR, n, sigma, rmin, rmax):
    #This function calculates the signal to noise ratio of the flexion signal
    term1 = eR * np.sqrt(np.pi * n) / (sigma * rmin)
    term2 = np.log(rmax / rmin) / np.sqrt(rmax**2 / rmin**2 - 1)
    return term1 * term2


def stn_shear(eR, n, sigma, rmin, rmax):
    #This function calculates the signal to noise ratio of the shear signal
    term1 = eR * np.sqrt(np.pi * n) / (sigma)
    term2 = (1 - rmin/rmax) / np.sqrt(1 - (rmin/rmax)**2)
    return term1 * term2


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
    