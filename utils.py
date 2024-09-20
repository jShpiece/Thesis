"""
Utility functions for gravitational lensing analysis.

This module provides a collection of utility functions used in gravitational lensing studies,
including cosmological calculations, image processing, chi-squared utilities, and lensing signal
computations.

Functions:
    - print_progress_bar
    - angular_diameter_distances
    - critical_surface_density
    - convolve_image
    - create_gaussian_kernel
    - compute_source_weights
    - find_combinations
    - filter_combinations
    - stn_flexion
    - stn_shear
    - calculate_kappa
    - calculate_mass
    - mass_sheet_transformation
    - calculate_lensing_signals_sis
    - calculate_lensing_signals_nfw
"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c, G
from astropy import units as u
import scipy.ndimage
from itertools import combinations


# ------------------------
# Terminal Utility Functions
# ------------------------

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Displays a progress bar in the terminal.

    Parameters:
        iteration (int): Current iteration (must be <= total).
        total (int): Total iterations.
        prefix (str): Prefix string to display before the progress bar.
        suffix (str): Suffix string to display after the progress bar.
        decimals (int): Number of decimals to display in the percentage complete.
        length (int): Character length of the progress bar.
        fill (str): Bar fill character.
        print_end (str): End character (e.g., "\r", "\r\n").

    Example:
        print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', length=50)
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration >= total:
        print()

# ------------------------
# Cosmology Utility Functions
# ------------------------

def angular_diameter_distances(z1, z2):
    """
    Calculates the angular diameter distances required for lensing calculations.

    Parameters:
        z1 (float): Redshift of the lens.
        z2 (float): Redshift of the source.

    Returns:
        tuple: (Dl, Ds, Dls) in meters, where:
            - Dl: Angular diameter distance to the lens.
            - Ds: Angular diameter distance to the source.
            - Dls: Angular diameter distance between the lens and the source.
    """
    dl = cosmo.angular_diameter_distance(z1).to(u.m).value
    ds = cosmo.angular_diameter_distance(z2).to(u.m).value
    dls = cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.m).value
    return dl, ds, dls


def critical_surface_density(z1, z2):
    """
    Computes the critical surface density for gravitational lensing.

    Parameters:
        z1 (float): Redshift of the lens.
        z2 (float): Redshift of the source.

    Returns:
        float: Critical surface density in kg/m^2.
    """
    dl, ds, dls = angular_diameter_distances(z1, z2)
    sigma_crit = (c.value**2 / (4 * np.pi * G.value)) * (ds / (dl * dls))
    return sigma_crit


# ------------------------
# Image Processing Functions
# ------------------------

def convolve_image(img, kernel):
    """
    Convolves an image with a kernel using Fourier transforms.

    Parameters:
        img (np.ndarray): Input image array.
        kernel (np.ndarray): Convolution kernel array.

    Returns:
        np.ndarray: Convolved image.
    """
    img_ft = np.fft.fftn(img, norm='ortho')
    kernel_ft = np.fft.fftn(kernel, s=img.shape, norm='ortho')
    convolved_img_ft = img_ft * kernel_ft
    convolved_img = np.real(np.fft.ifftn(convolved_img_ft, norm='ortho'))
    return np.fft.fftshift(convolved_img)


def create_gaussian_kernel(stamp_size, sigma):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
        stamp_size (int): Size of the kernel (stamp_size x stamp_size).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D Gaussian kernel normalized to sum to 1.
    """
    yp, xp = np.mgrid[-stamp_size / 2:stamp_size / 2, -stamp_size / 2:stamp_size / 2]
    gaussian = np.exp(-((xp / sigma) ** 2 + (yp / sigma) ** 2) / 2)
    return gaussian / np.sum(gaussian)


# ------------------------
# Chi-Squared Utility Functions
# ------------------------

def compute_source_weights(lenses, sources, r_char=10):
    """
    Computes Gaussian weights for lens-source pairs based on their separation.

    Parameters:
        lenses: Lens object containing positions of lenses.
        sources: Source object containing positions of sources.
        r_char (float): Characteristic length scale for weighting.

    Returns:
        np.ndarray: Array of weights for each source.
    """
    weights = np.zeros((len(lenses.x), len(sources.x)))
    for i in range(len(lenses.x)):
        dx = lenses.x[i] - sources.x
        dy = lenses.y[i] - sources.y
        r = np.hypot(dx, dy)
        weights[i] = np.exp(-r / r_char)

    # Normalize the weights for each lens
    weights /= weights.sum(axis=1, keepdims=True)
    return weights[0]  # Assuming single lens for now


def find_combinations(values, k):
    """
    Finds all combinations of the given values of length k.

    Parameters:
        values (iterable): Iterable of values to combine.
        k (int): Length of combinations.

    Returns:
        list: List of combinations.
    """
    return list(combinations(values, k))


def filter_combinations(combination_list):
    """
    Filters out duplicate combinations from a list of combinations.

    Parameters:
        combination_list (list): List of combinations.

    Returns:
        np.ndarray: Array of unique combinations.
    """
    sorted_combinations = np.sort(combination_list, axis=1)
    unique_combinations = np.unique(sorted_combinations, axis=0)
    return unique_combinations


# ------------------------
# Lensing Utility Functions
# ------------------------

def stn_flexion(eR, n, sigma, rmin, rmax):
    """
    Calculates the signal-to-noise ratio of the flexion signal.

    Parameters:
        eR (float): Einstein radius.
        n (float): Number density of sources.
        sigma (float): Noise level.
        rmin (float): Minimum radius.
        rmax (float): Maximum radius.

    Returns:
        float: Signal-to-noise ratio for flexion.
    """
    term1 = eR * np.sqrt(np.pi * n) / (sigma * rmin)
    term2 = np.log(rmax / rmin) / np.sqrt(rmax**2 / rmin**2 - 1)
    return term1 * term2


def stn_shear(eR, n, sigma, rmin, rmax):
    """
    Calculates the signal-to-noise ratio of the shear signal.

    Parameters:
        eR (float): Einstein radius.
        n (float): Number density of sources.
        sigma (float): Noise level.
        rmin (float): Minimum radius.
        rmax (float): Maximum radius.

    Returns:
        float: Signal-to-noise ratio for shear.
    """
    term1 = eR * np.sqrt(np.pi * n) / sigma
    term2 = (1 - rmin / rmax) / np.sqrt(1 - (rmin / rmax) ** 2)
    return term1 * term2


def calculate_kappa(lenses, extent, smoothing_scale):
    """
    Calculates the convergence (kappa) map for a given set of lenses.

    Parameters:
        lenses: Lens object containing positions and strengths.
        extent (tuple): (xmin, xmax, ymin, ymax) defining the map extent in arcseconds.
        smoothing_scale (float): Smoothing scale in arcseconds.

    Returns:
        tuple: (X, Y, kappa) where X and Y are meshgrid arrays, and kappa is the convergence map.
    """
    xmin, xmax, ymin, ymax = extent
    x_range = np.linspace(xmin, xmax, int(xmax - xmin))
    y_range = np.linspace(ymin, ymax, int(ymax - ymin))
    X, Y = np.meshgrid(x_range, y_range)
    kappa = np.zeros_like(X)

    # Calculate the convergence map
    for k in range(len(lenses.x)):
        dx = X - lenses.x[k]
        dy = Y - lenses.y[k]
        r = np.hypot(dx, dy) + 0.5  # Avoid division by zero
        kappa += lenses.te[k] / (2 * r)

    # Apply Gaussian smoothing if smoothing_scale is provided
    if smoothing_scale:
        kappa = scipy.ndimage.gaussian_filter(kappa, sigma=smoothing_scale)

    return X, Y, kappa


def calculate_mass(kappa_array, z_l, z_s, pixel_scale):
    """
    Calculates the total mass within a convergence map.

    Parameters:
        kappa_array (np.ndarray): Convergence map array.
        z_l (float): Redshift of the lens.
        z_s (float): Redshift of the source.
        pixel_scale (float): Pixel scale in arcseconds.

    Returns:
        float: Total mass in solar masses (h^-1 M_sun) within 200 kpc.
    """
    h = cosmo.H0.value / 100
    pixel_scale_rad = (pixel_scale * u.arcsec).to(u.rad).value
    central_pixel = kappa_array.shape[0] // 2

    # Angular diameter distances
    D_l = cosmo.angular_diameter_distance(z_l).to(u.m)
    D_s = cosmo.angular_diameter_distance(z_s).to(u.m)
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m)

    # Critical surface mass density
    Sigma_crit = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
    Sigma_crit = Sigma_crit.to(u.kg / u.m**2).value

    # Area per pixel in m^2
    area_per_pixel = (pixel_scale_rad * D_l.value) ** 2

    # Create distance map from the center
    indices = np.indices(kappa_array.shape)
    distance_map = np.hypot(indices[0] - central_pixel, indices[1] - central_pixel)
    distance_map *= pixel_scale_rad * D_l.value  # Convert to meters

    # Define radius of 200 kpc in meters
    r_200 = (200 * u.kpc).to(u.m).value

    # Find pixels within 200 kpc
    pixels_within_radius = distance_map <= r_200

    # Calculate the total mass
    total_mass = np.sum(kappa_array[pixels_within_radius]) * Sigma_crit * area_per_pixel
    total_mass_solar = (total_mass * u.kg).to(u.Msun).value * h

    return total_mass_solar


def mass_sheet_transformation(kappa, k):
    """
    Applies the mass-sheet transformation to a convergence map.

    Parameters:
        kappa (np.ndarray): Original convergence map.
        k (float): Mass-sheet factor.

    Returns:
        np.ndarray: Transformed convergence map.
    """
    return k * kappa + (1 - k)


def calculate_lensing_signals_sis(lenses, sources):
    """
    Calculates lensing signals (shear, flexion, g-flexion) for SIS lenses.

    Parameters:
        lenses: SIS_Lens object containing lens positions and Einstein radii.
        sources: Source object containing source positions.

    Returns:
        tuple: (shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2)
    """
    dx = sources.x - lenses.x[:, np.newaxis]
    dy = sources.y - lenses.y[:, np.newaxis]
    r = np.hypot(dx, dy)

    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi ** 2 - sin_phi ** 2
    sin2phi = 2 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    shear_mag = -lenses.te[:, np.newaxis] / (2 * r)
    flexion_mag = -lenses.te[:, np.newaxis] / (2 * r ** 2)
    g_flexion_mag = 3 * lenses.te[:, np.newaxis] / (2 * r ** 2)

    # Sum over all lenses
    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flexion_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flexion_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    return shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2


def calculate_lensing_signals_nfw(halos, sources, z_source):
    """
    Calculates lensing signals (shear, flexion, g-flexion) for NFW halos.

    Parameters:
        halos: NFW_Lens object containing halo properties.
        sources: Source object containing source positions.
        z_source (float): Redshift of the source.

    Returns:
        tuple: (shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2)
    """
    # Angular diameter distances
    Dl, _, _ = angular_diameter_distances(halos.redshift, z_source)

    # Calculate R200 and scale radius
    r200, r200_arcsec = halos.calc_R200()
    rs = r200 / halos.concentration  # Scale radius in meters

    # Critical surface density at lens redshift
    sigma_crit = critical_surface_density(halos.redshift, z_source)
    rho_c = cosmo.critical_density(halos.redshift).to(u.kg / u.m**3).value
    delta_c = halos.calc_delta_c()
    rho_s = rho_c * delta_c
    kappa_s = rho_s * rs / sigma_crit  # Dimensionless surface density
    flexion_s = (kappa_s * Dl) / (rs * u.radian.to(u.arcsecond))  # In inverse arcseconds

    # Distances between sources and halos
    dx = sources.x - halos.x[:, np.newaxis]
    dy = sources.y - halos.y[:, np.newaxis]
    r = np.hypot(dx, dy)
    r = np.where(r == 0, 0.01, r)  # Avoid division by zero

    x = np.abs(r / (r200_arcsec[:, np.newaxis] / halos.concentration[:, np.newaxis]))

    # Radial terms for lensing calculations
    def radial_term_1(x):
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1

        sol[mask1] = 1 - (2 / np.sqrt(1 - x[mask1] ** 2)) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1])))
        sol[mask2] = 1 - (2 / np.sqrt(x[mask2] ** 2 - 1)) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2])))

        return sol

    def radial_term_2(x):
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x > 1
        mask3 = x == 1

        k = (1 - x) / (1 + x)

        sol[mask1] = (
            8 * np.arctanh(np.sqrt(k[mask1])) / (x[mask1] ** 2 * np.sqrt(1 - x[mask1] ** 2))
            + 4 * np.log(x[mask1] / 2) / x[mask1] ** 2
            - 2 / (x[mask1] ** 2 - 1)
            + 4 * np.arctanh(np.sqrt(k[mask1])) / ((x[mask1] ** 2 - 1) * np.sqrt(1 - x[mask1] ** 2))
        )

        sol[mask2] = (
            8 * np.arctan(np.sqrt((x[mask2] - 1) / (x[mask2] + 1))) / (x[mask2] ** 2 * np.sqrt(x[mask2] ** 2 - 1))
            + 4 * np.log(x[mask2] / 2) / x[mask2] ** 2
            - 2 / (x[mask2] ** 2 - 1)
            + 4 * np.arctan(np.sqrt((x[mask2] - 1) / (x[mask2] + 1))) / ((x[mask2] ** 2 - 1) ** (3 / 2))
        )

        sol[mask3] = 10 / 3 + 4 * np.log(1 / 2)

        return sol

    def radial_term_3(x):
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1

        sol[mask1] = (
            1 / (1 - x[mask1] ** 2)
            * (
                1 / x[mask1]
                - (2 * x[mask1]) / np.sqrt(1 - x[mask1] ** 2) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1])))
            )
        )
        sol[mask2] = (
            1 / (x[mask2] ** 2 - 1)
            * (
                (2 * x[mask2]) / np.sqrt(x[mask2] ** 2 - 1) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2])))
                - 1 / x[mask2]
            )
        )

        return sol

    def radial_term_4(x):
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1
        leading_term = 8 / x ** 3 - 20 / x + 15 * x

        k = (1 - x) / (1 + x)

        sol[mask1] = (2 / np.sqrt(1 - x[mask1] ** 2)) * np.arctanh(np.sqrt(k[mask1]))
        sol[mask2] = (2 / np.sqrt(x[mask2] ** 2 - 1)) * np.arctan(np.sqrt(-k[mask2]))

        sol *= leading_term

        return sol

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)

    # Angle computations
    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi ** 2 - sin_phi ** 2
    sin2phi = 2 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    # Lensing magnitudes
    shear_mag = -kappa_s[:, np.newaxis] * term_2

    def calc_flexion(flexion_s, x, term_1, term_3):
        I1 = -2 * flexion_s[:, np.newaxis]
        I2 = 2 * x * term_1 / (x ** 2 - 1) ** 2
        I3 = term_3 / (x ** 2 - 1)
        return I1 * (I2 - I3)

    def calc_g_flexion(flexion_s, x, term_4):
        I1 = 2 * flexion_s[:, np.newaxis]
        I2 = (8 / x ** 3) * np.log(x / 2)
        I3 = (3 / x) * (1 - 2 * x ** 2) + term_4
        I4 = (x ** 2 - 1) ** 2
        return I1 * (I2 + (I3 / I4))

    flexion_mag = calc_flexion(flexion_s, x, term_1, term_3)
    g_flexion_mag = calc_g_flexion(flexion_s, x, term_4)

    # Sum over all halos
    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flexion_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flexion_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    return shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2
