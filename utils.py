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


def process_weights(weights, eR_range, size):
    '''
    Take a likelihood map in 3D parameter space, integrate over the eR axis,
    and convolve with a gaussian kernel to smooth out the map. Then normalize
    the map and return it.
    '''
    llmap = np.trapz(weights, eR_range[::-1], axis=0)
    kernel = makeGaussian(size,2)
    llmap = convolver(llmap,kernel)
    llmap = np.abs(llmap)
    llmap /= np.sum(llmap)

    return llmap


def find_eR(map, x, y, eR_range):
    '''
    Take a likelihood map in 3D parameter space and return the eR value
    at each maxima.
    '''
    eR = []
    #plt.figure()

    for i in range(len(x)):
        possible_eR = map[:,y[i],x[i]]
        eR.append(eR_range[np.argmax(possible_eR)])

        #plt.plot(eR_range,possible_eR)
    #plt.yscale('log')
    #plt.xlabel('eR')
    #plt.ylabel('Likelihood')
    #plt.show()

    return eR