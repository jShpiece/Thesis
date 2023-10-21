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


def lens(x,y,lenses):
    #Compute the lensing signals on a single source 
    #from a set of lenses
    dx = x-lenses.x
    dy = y-lenses.y
    r = np.sqrt(dx**2+dy**2)
    cosphi = dx/r
    sinphi = dy/r
    cos2phi = cosphi*cosphi-sinphi*sinphi
    sin2phi = 2*cosphi*sinphi

    f1 = np.sum(-dx*lenses.te/(2*r*r*r))
    f2 = np.sum(-dy*lenses.te/(2*r*r*r))

    e1 = np.sum(-lenses.te/(2*r)*cos2phi)
    e2 = np.sum(-lenses.te/(2*r)*sin2phi)

    return e1,e2,f1,f2


def eR_penalty_function(eR, lower_limit=0.0, upper_limit=20.0, lambda_penalty_upper=10.0):
    # Hard lower limit
    if eR < lower_limit:
        return 1e8 #Use an arbitrarily large number - NOT infinity (will yield NaNs)

    # Soft upper limit
    if eR > upper_limit:
        return lambda_penalty_upper * (eR - upper_limit) ** 2

    return 0.0


def chi2(sources, lenses, sigf, sigs, fwgt=1.0, swgt=1.0):
    x, y, e1data, e2data, f1data, f2data = sources.x, sources.y, sources.e1, sources.e2, sources.f1, sources.f2
    # Initialize chi^2 value
    chi2val = 0.0
    
    # Loop through the data points to compute the chi^2 terms
    # What if we only get one data point?
    # Turn it into an array of length 1
    if not isinstance(x, np.ndarray):
        x = np.array([x])
        y = np.array([y])
        e1data = np.array([e1data])
        e2data = np.array([e2data])
        f1data = np.array([f1data])
        f2data = np.array([f2data])
    
    for i in range(len(x)):
        # Assuming your 'lens' function can handle tetest being an array
        # If not, you may need to modify this part
        e1, e2, f1, f2 = lens(x[i], y[i], lenses)
        
        chif1 = (f1data[i] - f1) ** 2 / (sigf ** 2)
        chif2 = (f2data[i] - f2) ** 2 / (sigf ** 2)
        chie1 = (e1data[i] - e1) ** 2 / (sigs ** 2)
        chie2 = (e2data[i] - e2) ** 2 / (sigs ** 2)
        
        chi2val += fwgt * (chif1 + chif2) + swgt * (chie1 + chie2)
    
    # Add the penalty term for Einstein radii outside the threshold
    total_penalty = 0.0
    try:
        for eR in lenses.te:
            total_penalty += eR_penalty_function(eR)
    except TypeError:
        total_penalty += eR_penalty_function(lenses.te)

    chi2val += total_penalty
    
    return chi2val


def generate_combinations(n, m, start=0, curr=[]):
    # Generate all combinations of m elements from a set of n elements
    if m == 0:
        yield curr
        return
    for i in range(start, n):
        yield from generate_combinations(n, m-1, i+1, curr + [i])

