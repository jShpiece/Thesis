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


def process_weights(weights, eR_range):
    '''
    Take a likelihood map in 3D parameter space, integrate over the eR axis,
    and convolve with a gaussian kernel to smooth out the map. Then normalize
    the map and return it.
    '''
    res = len(eR_range)
    maps = []
    for i in range(len(weights)):
        llmap = np.trapz(weights[i], eR_range, axis=0)
        llmap = convolver(llmap, makeGaussian(res, 1))
        maps.append(llmap)
    return maps


def find_eR(maps, coords):
    '''
    Take a likelihood map in 3D parameter space and return the eR value
    at each maxima.
    '''
    output = []
    for i in range(len(maps)):
        eR = []
        x = coords[i][0]
        y = coords[i][1]
        for j in range(len(x)):
            possible_eR = maps[i][:,y[j],x[j]]
            eR.append(np.argmax(possible_eR, axis=0))
        output.append(eR)
    return output


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


def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    '''This function takes in a function and returns the gradient of that function
    It is used to correct the bounds being used in SCIPY minimization functions, 
    because otherwise the minimization function will sometimes try to evaluate the
    function outside of the bounds, which causes an error.'''
    
    """bounds: list of tuples (lower, upper)"""
    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad
    return gradient