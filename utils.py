import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c, G
from astropy import units as u
import scipy.ndimage
import pipeline

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
# General Calculation Functions
# ------------------------
        

def project_onto_principal_axis(x, y, z):
    """Projects a set of 3D points onto the plane formed by the first two principal eigenvectors."""

    # Sanity checks
    assert len(x) == len(y) == len(z), "The x, y, and z arrays must have the same length."
    assert x.ndim == y.ndim == z.ndim == 1, "The x, y, and z arrays must be 1D."
    assert len(x) > 1, "At least two points are required."

    # Combine the x, y, z coordinates into a single matrix
    points = np.vstack((x, y, z)).T

    # Calculate the covariance matrix
    cov_matrix = np.cov(points, rowvar=False)

    # Compute the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Project the points onto the plane formed by the first two principal eigenvectors
    projected_points = np.dot(points, eigenvectors[:, :2])

    return projected_points[:, 0], projected_points[:, 1]


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


def calculate_kappa(lenses, extent, smoothing_scale) -> tuple:
    '''
    Calculates the convergence map for a given set of lenses.
    Parameters:
        lenses (Lenses): The lenses.
        extent (tuple): The extent of the convergence map in arcsec.
        smoothing_scale (float): The smoothing scale in arcsec.
    '''
    # Take absolute value of difference in extent to avoid negative step size
    X = np.linspace(extent[0], extent[1], np.abs(int(extent[1] - extent[0])))
    Y = np.linspace(extent[2], extent[3], np.abs(int(extent[3] - extent[2])))
    X, Y = np.meshgrid(X, Y)
    kappa = np.zeros_like(X)

    # Calculate the convergence map
    for k in range(len(lenses.x)):
        r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2 + 0.5**2) # Add 0.5 to avoid division by 0 
        kappa += lenses.te[k] / (2 * r)
    
    # Smooth the convergence map with a gaussian kernel (use an external function)
    kappa = scipy.ndimage.gaussian_filter(kappa, sigma=smoothing_scale)
    return X, Y, kappa


def calculate_mass(kappa_array, z_l, z_s, pixel_scale):
    """
    Calculates the total mass within a convergence map.
    ------------------------------------------
    Parameters:
        kappa_array (2D array): The convergence map.
        z_l (float): The redshift of the lens.
        z_s (float): The redshift of the source.
        pixel_scale (float): The pixel scale of the convergence map in arcsec.
    """
    # Define some constants
    h = cosmo.H0.value / 100
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad).value
    central_pixel = kappa_array.shape[0] // 2
    # Including our angular diameter distances for this redshift
    D_l = cosmo.angular_diameter_distance(z_l).to(u.meter)
    D_s = cosmo.angular_diameter_distance(z_s).to(u.meter)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.meter)
    # Now compute the critical surface mass density in units of kg/m^2
    Sigma_crit = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
    # Compute the area per pixel in square meters
    area_per_pixel = (pixel_scale_rad * D_l)**2  # Area in m^2

    # Build distance map from the center - this is an array identical to kappa, with each pixel containing the distance from the center
    distance_map = np.zeros_like(kappa_array)
    for i in range(kappa_array.shape[0]):
        for j in range(kappa_array.shape[1]):
            distance_map[i, j] = np.sqrt((i - central_pixel)**2 + (j - central_pixel)**2)
    
    # Find the mass which lies within 1 Mpc of the center of the map
    # Convert 1 Mpc to radians, at the relevant distance
    one_mpc_rad = (500 * u.kpc).to(u.meter) / D_l
    # Converge the radius to pixels
    one_mpc_rad /= pixel_scale_rad
    # Find the pixels that lie inside this radius (measured from the center of the map)
    pixels_within_radius = distance_map <= one_mpc_rad
    # Calculate the mass within this radius
    total_mass = np.sum(kappa_array[pixels_within_radius]) * Sigma_crit * area_per_pixel
    total_mass_solar = (total_mass).to(u.M_sun).value * h # Give all masses in units of h^-1 M_sun

    return total_mass_solar


def mass_sheet_transformation(kappa, k):
    return k*kappa + (1 - k)


# ------------------------------
# Initialization functions
# ------------------------------

def createSources(lenses,ns=1,randompos=True,sigs=0.1,sigf=0.01,sigg=0.02,xmax=5,lens_type='SIS'):
    #Create sources for a lensing system and apply the lensing signal
    #Create the sources - require that they be distributed sphericaly
    if randompos == True:
        r = xmax*np.sqrt(np.random.random(ns))
        phi = 2*np.pi*np.random.random(ns)
    else: #Uniformly spaced sources - single choice of r, uniform phi
        r = xmax / 2
        phi = 2*np.pi*(np.arange(ns)+0.5)/(ns)
    
    x = r*np.cos(phi)
    y = r*np.sin(phi)

    # Initialize lensing parameters with gaussian noise
    e1data = np.random.normal(0,sigs,ns)
    e2data = np.random.normal(0,sigs,ns)
    f1data = np.random.normal(0,sigf,ns)
    f2data = np.random.normal(0,sigf,ns)
    g1data = np.random.normal(0,sigg,ns)
    g2data = np.random.normal(0,sigg,ns)

    sources = pipeline.Source(x, y, e1data, e2data, f1data, f2data, g1data, g2data, sigs*np.ones(ns), sigf*np.ones(ns), sigg*np.ones(ns))
    # Apply the lensing effects of the lenses
    if lens_type == 'SIS':
        sources.apply_SIS_lensing(lenses)
    elif lens_type == 'NFW':
        # Note - in this case the 'lenses' object is actually a 'halos' object
        # The effect is the same in practice
        sources.apply_NFW_lensing(lenses)

    return sources


def createLenses(nlens=1,randompos=True,xmax=10,strength_choice='identical'):
    if randompos == True:
        xlarr = -xmax + 2*xmax*np.random.random(nlens)
        ylarr = -xmax + 2*xmax*np.random.random(nlens)
    else: #Uniformly spaced lenses
        xlarr = -xmax + 2*xmax*(np.arange(nlens)+0.5)/(nlens)
        ylarr = np.zeros(nlens)
    
    # Now we assign einstein radii based on the strength_choice
    if strength_choice == 'identical':
        tearr = np.ones(nlens)
    elif strength_choice == 'random':
        tearr = np.random.random(nlens) * 20
    elif strength_choice == 'uniform':
        tearr = np.linspace(0.1, 20, nlens)
    elif strength_choice == 'cluster':
        tearr = np.ones(nlens)
        tearr[0] = 10
    else:
        raise ValueError("Invalid strength_choice")

    
    lenses = pipeline.Lens(xlarr, ylarr, tearr, np.empty(nlens))
    return lenses