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
# Cosmology Utility Functions
# ------------------------

def angular_diameter_distances(z1, z2):
    dl = cosmo.angular_diameter_distance(z1).to(u.m).value
    ds = cosmo.angular_diameter_distance(z2).to(u.m).value
    dls = cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.m).value
    return dl, ds, dls


def critical_surface_density(z1, z2):
    dl, ds, dls = angular_diameter_distances(z1, z2)
    kappa_c = (c.value**2 / (4 * np.pi * G.value)) * (ds / (dl * dls))
    return kappa_c


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
    if smoothing_scale:
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
    
    # Find the mass which lies within 200 kpc of the center of the map
    r_map = (200 * u.kpc).to(u.meter) / D_l
    # Converge the radius to pixels
    r_map /= pixel_scale_rad
    # Find the pixels that lie inside this radius (measured from the center of the map)
    pixels_within_radius = distance_map <= r_map
    # Calculate the mass within this radius
    total_mass = np.sum(kappa_array[pixels_within_radius]) * Sigma_crit * area_per_pixel
    total_mass_solar = (total_mass).to(u.M_sun).value * h # Give all masses in units of h^-1 M_sun

    return total_mass_solar


def mass_sheet_transformation(kappa, k):
    return k*kappa + (1 - k)


def calculate_lensing_signals_nfw(halos, sources, z_source):
    # Apply the lensing effects of a set of halos to the sources
    # Model the halos as Navarro-Frenk-White (NFW) profiles
    # Then the primary parameters are the masses and concentrations of the halos

    # Define angular diameter distances
    Dl = angular_diameter_distances(halos.redshift, z_source)[0]

    # Compute R200
    r200, r200_arcsec = halos.calc_R200()
    rs = r200 / halos.concentration

    # Compute the critical surface density at the halo redshift
    rho_c = cosmo.critical_density(halos.redshift).to(u.kg / u.m**3).value 
    rho_s = rho_c * halos.calc_delta_c() 
    sigma_c = critical_surface_density(halos.redshift, z_source)
    # Each halo has a characteristic surface density kappa_s
    kappa_s = rho_s * rs / sigma_c
    flexion_s = kappa_s * Dl / rs

    # Compute the distances between each source and each halo
    # x is the distance from the halo in units of the NFW scale radius
    x = np.sqrt((sources.x - halos.x[:, np.newaxis])**2 + (sources.y - halos.y[:, np.newaxis])**2) / (r200_arcsec[:, np.newaxis] / halos.concentration[:, np.newaxis])
    # If x is less than 0.01, set it to 0.01 - the nfw profile is not well defined at x = 0
    x = np.where(x < 0.01, 0.01, x)

    # Define the radial terms that go into lensing calculations - these are purely functions of x
    def radial_term_1(x):
        # This is called f(x) in theory
        with np.errstate(invalid='ignore'):
            # Note that the two statements are almost identical - switching arctanh and arctan, and 1-x and x-1
            sol = np.where(x < 1, 
                        1 - (2 / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x))), # x < 1
                        1 - (2 / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (1 + x))) # x > 1
                        ) 
        return sol
    
    def radial_term_2(x):
        # This is called g(x) in theory
        with np.errstate(invalid='ignore'):
            sol = np.where(x < 1,
                        8 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / (x**2 * np.sqrt(1 - x**2))
                        + (4 / x**2) * np.log(x / 2)
                        - 2 / (x**2 - 1)
                        + 4 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / ((x**2 - 1) * np.sqrt(1 - x**2)),
                        np.where(x == 1,
                                    10 / 3 + 4 * np.log(1 / 2),
                                    8 * np.arctan(np.sqrt((x - 1) / (1 + x))) / (x**2 * np.sqrt(x**2 - 1))
                                    + (4 / x**2) * np.log(x / 2)
                                    - 2 / (x**2 - 1)
                                    + 4 * np.arctan(np.sqrt((x - 1) / (1 + x))) / ((x**2 - 1)**(3/2))
                                    )
                        )
        return sol

    def radial_term_3(x):
        # Compute the radial term - this is called h(x) in theory
        with np.errstate(invalid='ignore'):
            sol = np.where(x < 1, 
                    1 / (1 - x**2) * (1 / x - ((2*x) / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x)))), # x < 1
                    1 / (x**2 - 1) * (((2 * x) / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (1 + x))) - 1 / x) # x > 1
                    )
        return sol
    
    def radial_term_4(x):
        # Compute the radial term - this is called I(x) in theory
        with np.errstate(invalid='ignore'):
            leading_term = 8 / x**3 - 20 / x + 15 * x
            sol = np.where(x < 1, leading_term * (2 / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x))), # x < 1
                            leading_term * (2 / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (x + 1))) # x > 1
                            )
        return sol

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)

    # Compute the lensing signals

    # Begin with the distances and angles between the source and the halo
    dx = sources.x - halos.x[:, np.newaxis]
    dy = sources.y - halos.y[:, np.newaxis]
    r = np.sqrt(dx**2 + dy**2)
    r = np.where(r == 0, 0.01, r) # Avoid division by zero
    cos_phi = dx / r # Cosine of the angle between the source and the halo
    sin_phi = dy / r # Sine of the angle between the source and the halo
    cos2phi = cos_phi**2 - sin_phi**2 # Cosine of 2*phi
    sin2phi = 2 * cos_phi * sin_phi # Sine of 2*phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi # Cosine of 3*phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi # Sine of 3*phi

    # Compute lensing magnitudes (in arcseconds)
    shear_mag = - kappa_s[:, np.newaxis] * term_2
    flexion_mag = (-2 * flexion_s[:, np.newaxis]) * (((2 * x * term_1) / (x**2 - 1)**2) - term_3 / (x**2 - 1)) / 206265 # Convert to arcseconds
    g_flexion_mag = (2 * flexion_s[:, np.newaxis] * ((8 / x**3) * np.log(x / 2) + ((3/x)*(1 - 2*x**2) + term_4) / (x**2 - 1)**2)) / 206265

    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flexion_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flexion_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    return shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2


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

    sources = pipeline.Source(x, y, 
                              np.zeros_like(x), np.zeros_like(y),
                                np.zeros_like(x), np.zeros_like(y),
                                np.zeros_like(x), np.zeros_like(y), 
                                np.ones_like(x) * sigs, np.ones_like(x) * sigf, np.ones_like(x) * sigg)
    # Now apply noise
    sources.apply_noise()
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
        r = xmax*np.sqrt(np.random.random(nlens))
        phi = 2*np.pi*np.random.random(nlens)  
        xlarr = r*np.cos(phi)
        ylarr = r*np.sin(phi)

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