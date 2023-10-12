import numpy as np

# ------------------------
# Terminal Utility Functions
# ------------------------

def print_progress_bar(iteration, total, prefix='', suffix='', 
                       decimals=1, length=100, fill='█', print_end="\r"):
    """Prints a progress bar in the terminal."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    if iteration == total: 
        print() 



# ------------------------
# Image Processing Functions
# ------------------------

def convolve_image(img, kernel):
    """Convolves an image with a kernel using Fourier transform."""
    img_ft = np.fft.fftn(img, norm='ortho')
    kernel_ft = np.fft.fftn(kernel, norm='ortho')
    convolved_img_fourier = img_ft * kernel_ft
    return np.real(np.fft.fftshift(
        np.fft.ifftn(convolved_img_fourier, img.shape, norm='ortho')))


def create_gaussian_kernel(stamp_size, sigma):
    """Generates a 2D Gaussian kernel."""
    yp, xp = np.mgrid[-stamp_size / 2:stamp_size / 2, -stamp_size / 2:stamp_size / 2]
    gaussian = np.exp(-((xp / sigma) ** 2 + (yp / sigma) ** 2) / 2)
    return gaussian / np.sum(gaussian)



# ------------------------
# Lensing Utility Functions
# ------------------------

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


def eR_penalty_function(eR, lower_limit=0.0, upper_limit=20.0, lambda_penalty_upper=10.0):
    # Hard lower limit
    if eR < lower_limit:
        return 1e8 #Use an arbitrarily large number - NOT infinity (will yield NaNs)

    # Soft upper limit
    if eR > upper_limit:
        return lambda_penalty_upper * (eR - upper_limit) ** 2

    return 0.0


def chi2(x, y, e1data, e2data, f1data, f2data, xltest, yltest, tetest, sigf, sigs, fwgt=1.0, swgt=1.0):
   
    # Initialize chi^2 value
    chi2val = 0.0
    
    # Loop through the data points to compute the chi^2 terms
    for i in range(len(x)):
        # Assuming your 'lens' function can handle tetest being an array
        # If not, you may need to modify this part
        e1, e2, f1, f2 = lens(x[i], y[i], xltest, yltest, tetest)
        
        chif1 = (f1data[i] - f1) ** 2 / (sigf ** 2)
        chif2 = (f2data[i] - f2) ** 2 / (sigf ** 2)
        chie1 = (e1data[i] - e1) ** 2 / (sigs ** 2)
        chie2 = (e2data[i] - e2) ** 2 / (sigs ** 2)
        
        chi2val += fwgt * (chif1 + chif2) + swgt * (chie1 + chie2)
    
    # Add the penalty term for Einstein radii outside the threshold
    total_penalty = 0.0
    try:
        for eR in tetest:
            total_penalty += eR_penalty_function(eR)
    except TypeError:
        total_penalty += eR_penalty_function(tetest)

    chi2val += total_penalty
    
    return chi2val


def chi2wrapper(guess,params):
    return chi2(params[0],params[1],params[2],params[3],params[4],params[5],guess[0],guess[1],guess[2],params[6],params[7])