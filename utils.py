"""
Utility functions for gravitational lensing analysis.

This module provides a collection of utility functions used in gravitational lensing studies,
including cosmological calculations, image processing, chi-squared utilities, and lensing signal
computations.

"""

import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy.constants import c, G
from astropy import units as u
from itertools import combinations
from scipy.integrate import quad
import matplotlib.pyplot as plt
import halo_obj
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.signal.windows import tukey


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


def nfw_projected_mass(
    halos,
    r_p,
    return_2d=False,
    nx=100,
    ny=100,
    x_range=(-300, 300),
    y_range=(-300, 300),
    x_center=0.0,
    y_center=0.0,
    z_source=0.8
):
    """
    Compute the projected mass for an NFW halo, with an option to return a 2D map.

    Parameters:
    - halos: NFW_Lens object containing halo properties.
    - r_p: float or array-like
        Projected radius (kpc) or array of radii for 1D mass integration.
    - return_2d: bool
        If True, return a 2D grid of projected mass.
    - nx, ny: int
        Number of grid points along x and y axes if return_2d is True.
    - x_range, y_range: tuple
        Range of x and y (in kpc) for the 2D grid if return_2d is True.
    - x_center, y_center: float
        Coordinates (in kpc) of the halo center in the 2D grid.

    Returns:
    - If return_2d=False:
        M_proj: float or ndarray
            The integrated projected mass within radius r_p (in M_sun).
    - If return_2d=True:
        M_2D: 2D ndarray
            A 2D map of the projected mass (in M_sun) over the specified grid.
    """

    R200_m = halos.calc_R200()[0]  # in meters
    R200_kpc = R200_m * 3.24078e-20  # in kpc
    c = halos.concentration
    r_s = R200_kpc / c  # kpc

    rho_c = cosmo.critical_density(halos.redshift).to(u.M_sun / u.kpc**3).value
    delta_c = halos.calc_delta_c()
    sigma_c = critical_surface_density(halos.redshift, z_source)
    sigma_c = sigma_c * u.M_sun / u.kpc**2
    sigma_c = sigma_c.value

    def Sigma(R):
        x = R / r_s
        leading_term = 2 * rho_c * r_s * delta_c
        if x < 1:
            factor = 1.0 / (x**2 - 1)
            term1 = 1.0
            term2 = (2.0 / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            return leading_term * factor * (term1 - term2)
        elif np.isclose(x, 1.0):
            return leading_term / 3.0
        else:
            factor = 1.0 / (x**2 - 1)
            term1 = 1.0
            term2 = (2.0 / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (1 + x)))
            return leading_term * factor * (term1 - term2)

    if not return_2d:
        r_p = np.atleast_1d(r_p)
        M_proj = np.zeros_like(r_p, dtype=float)

        for i, r in enumerate(r_p):
            integrand = lambda rr: 2 * np.pi * Sigma(rr) * rr
            M_proj[i], _ = quad(integrand, 0, r, limit=1000)

        return M_proj if len(M_proj) > 1 else M_proj[0]

    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    area_per_pixel = dx * dy

    kappa = np.zeros((ny, nx))

    for i in range(ny):
        for j in range(nx):
            # Distance from halo center at (x_center, y_center)
            R = np.sqrt((x_vals[j] - x_center)**2 + (y_vals[i] - y_center)**2)
            surface_density = Sigma(R)
            kappa[i, j] = surface_density / sigma_c

    return kappa, area_per_pixel


# ------------------------
# Image Processing Functions
# ------------------------


def CIC_2d(fsize, npixels, xpos, ypos, signal_values):
    """
    Cloud-in-Cell (CIC) 2D binning of scattered data onto a square grid.

    Parameters:
        fsize: total field size (assumed square, float)
        npixels: number of pixels per side
        xpos, ypos: source coordinates (same units as fsize)
        signal_values: values to bin (same length as xpos/ypos)

    Returns:
        grid: 2D numpy array (npixels x npixels) with CIC-interpolated values
    """
    grid = np.zeros((npixels, npixels), dtype=float)
    pix_scl = fsize / npixels

    row = ypos / pix_scl
    col = xpos / pix_scl

    irow = np.floor(row).astype(int)
    icol = np.floor(col).astype(int)
    drow = row - irow
    dcol = col - icol

    index = np.vstack([irow, icol])
    unique_indices, unique_inverse, counts = np.unique(index, axis=1, return_inverse=True, return_counts=True)
    output = counts[np.searchsorted(unique_indices[0], index[0])]
    # signal_values /= output

    # Perform accumulation using advanced indexing and broadcasting
    grid[unique_indices[0] % npixels, unique_indices[1]% npixels] += np.bincount(unique_inverse, weights=(1 - drow) * (1 - dcol) * signal_values, minlength=len(unique_indices.T))
    grid[(unique_indices[0] + 1)% npixels, unique_indices[1]% npixels] += np.bincount(unique_inverse, weights=(drow) * (1 - dcol) * signal_values, minlength=len(unique_indices.T))
    grid[unique_indices[0]% npixels, (unique_indices[1] + 1)% npixels] += np.bincount(unique_inverse, weights=(1 - drow) * (dcol) * signal_values, minlength=len(unique_indices.T))
    grid[(unique_indices[0] + 1)% npixels, (unique_indices[1] + 1)% npixels] += np.bincount(unique_inverse, weights=(drow) * (dcol) * signal_values, minlength=len(unique_indices.T))

    return grid


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


def find_peaks_and_masses(kappa_map, z_lens, z_source, radius_kpc=200, threshold=0.05):
    """
    Identify mass peaks and compute enclosed mass within a fixed radius.
    Assumes the kappa map is gridded in arcseconds (not pixels).

    Parameters:
    - kappa_map: 2D convergence map, gridded in arcseconds
    - z_lens: lens redshift
    - z_source: source redshift
    - radius_kpc: physical radius around each peak to sum kappa over
    - threshold: minimum convergence value to consider as a peak

    Returns:
    - peaks_arcsec: list of (RA_offset, Dec_offset) coordinates in arcsec
    - masses: corresponding mass values in M_sun
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    import numpy as np

    # Smooth the convergence map
    kappa_smooth = gaussian_filter(kappa_map, sigma=2)
    local_max = maximum_filter(kappa_smooth, size=5)
    peaks_mask = (kappa_smooth == local_max) & (kappa_smooth > threshold)
    peak_indices = np.argwhere(peaks_mask)  # (y_arcsec, x_arcsec) integers

    # Conversion: arcsec → kpc
    arcsec_to_kpc = cosmo.angular_diameter_distance(z_lens).to(u.kpc).value * (np.pi / (180. * 3600.))
    radius_arcsec = radius_kpc / arcsec_to_kpc  # circular aperture radius

    # Critical surface density
    sigma_c = critical_surface_density(z_lens, z_source)
    # Convert to M_sun / kpc^2 from kg / m^2 (sigma_c is just a number)
    sigma_c = (sigma_c * u.kg / u.m**2).to(u.M_sun / u.kpc**2).value

    Ny, Nx = kappa_map.shape
    y_coords, x_coords = np.arange(Ny), np.arange(Nx)
    Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing='ij')  # in arcsec

    peaks_arcsec = []
    masses = []

    for y0, x0 in peak_indices:
        ra_offset = x0  # x-axis = RA in arcsec
        dec_offset = y0  # flip y-axis to match Dec increasing upward
        peaks_arcsec.append((ra_offset, dec_offset))

        dist = np.sqrt((X_grid - x0)**2 + (Y_grid - y0)**2)
        mask = dist <= radius_arcsec
        enclosed_kappa = np.sum(kappa_map[mask])
        area_kpc2 = np.sum(mask) * arcsec_to_kpc**2
        mass = sigma_c * enclosed_kappa * arcsec_to_kpc**2  # M_sun
        masses.append(np.abs(mass))

    return peaks_arcsec, masses


def calculate_kappa(lenses, extent, lens_type='SIS', source_redshift=0.8):
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
    def radial_term_1(x):
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1

        sol[mask1] = 1 - (2 / np.sqrt(1 - x[mask1] ** 2)) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1])))
        sol[mask2] = 1 - (2 / np.sqrt(x[mask2] ** 2 - 1)) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2])))

        return sol
    
    if lens_type == 'SIS':
        for k in range(len(lenses.x)):
            dx = X - lenses.x[k]
            dy = Y - lenses.y[k]
            r = np.hypot(dx, dy) + 0.5  # Avoid division by zero
            kappa += lenses.te[k] / (2 * r)

    elif lens_type == 'NFW':
        # Critical surface density at lens redshift
        sigma_crit = critical_surface_density(lenses.redshift, source_redshift)
        rho_c = cosmo.critical_density(lenses.redshift).to(u.kg / u.m**3).value
        delta_c = lenses.calc_delta_c()
        rho_s = rho_c * delta_c
        r200, r200_arcsec = lenses.calc_R200()
        rs_1 = r200 / lenses.concentration  # Scale radius in meters
        rs_2 = r200_arcsec / lenses.concentration  # Scale radius in arcseconds

        for k in range(len(lenses.x)):
            dx = X - lenses.x[k]
            dy = Y - lenses.y[k]
            r = np.hypot(dx, dy) + 0.5
            x = np.abs(r / rs_2[k])
            term_1 = radial_term_1(x)
            kappa_s = rho_s[k] * rs_1[k] / sigma_crit  # Dimensionless surface density
            kappa += 2 * kappa_s * term_1 / (x ** 2 - 1)

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


def estimate_mass_sheet_factor(kappa):
    """
    Estimates the mass-sheet factor k such that the transformed convergence
    goes to zero at the boundary.

    Parameters:
        kappa (np.ndarray): Original convergence map.

    Returns:
        float: Estimated mass-sheet transformation factor k.
    """
    # Extract boundary values
    top = kappa[0, :]
    bottom = kappa[-1, :]
    left = kappa[:, 0]
    right = kappa[:, -1]

    # Combine edge values and compute mean
    edge_values = np.concatenate([top, bottom, left[1:-1], right[1:-1]])
    mean_edge_kappa = np.mean(edge_values)

    # Estimate k using the constraint: mean_edge_kappa_transformed = 0
    k = 1 / (1 - mean_edge_kappa)
    return k


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


def compare_mass_estimates(halos, plot_name, plot_title, cluster_name='Abell_2744'):
    """
    Compares our mass reconstruction to literature estimates for Abell 2744.
    """
    
    # Literature mass estimates: dictionary of form {label: (mass, radius)}
    mass_estimates_abell = {
        'MARS': (1.73e14, 200), 
        'Bird': (1.93e14, 200), 
        'GRALE': (2.25e14, 250),
        'Merten et al.': (2.24e14, 250)
    }
    mass_estimates_elgordo = {
        'Cerny et al.': (1.1e15, 500),
        'Caminha et all.': (1.84e15, 1000),
        'Diego et al': (0.8e15, 500),
    }

    # Choose a cluster
    if cluster_name == 'ABELL_2744':
        mass_estimates = mass_estimates_abell
        z_cluster = 0.308
        z_source = 1.0
    elif cluster_name == 'EL_GORDO':
        mass_estimates = mass_estimates_elgordo
        z_cluster = 0.870
        z_source = 4.25
    else:
        raise ValueError('Invalid cluster name. Choose from "ABELL_2744" or "EL_GORDO".')
    
    # Radii in kpc
    # Take the minimum mass estimate and the maximum
    r_min = min([r for _, r in mass_estimates.values()])
    r_max = max([r for _, r in mass_estimates.values()])
    r = np.linspace(r_min * 0.75, 1.25*r_max, 100)
    
    # Move the halos to be centered on the primary halo
    
    # largest_halo = np.argmax(halos.mass)
    # Calculate the centroid as the center of mass of the halos
    centroid = np.array([np.sum(halos.x * halos.mass) / np.sum(halos.mass), np.sum(halos.y * halos.mass) / np.sum(halos.mass)])
    # centroid = [100,40] # hardcoding the centroid for now - this is where we expect the center of mass for el gordo
    halos.x -= centroid[0] + 0.5
    halos.y -= centroid[1] + 0.5

    # 2D grid setup (in kpc)
    x_range, y_range = (-r[-1], r[-1]), (-r[-1], r[-1])
    nx, ny = int(x_range[1] - x_range[0]), int(y_range[1] - y_range[0])
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    kappa_total = np.zeros((ny, nx), dtype=float)
    
    # Sum the mass grids from all halos
    for i in range(len(halos.x)):
        # Compute 2D mass for this halo, centered at (halo.x, halo.y) if needed
        # Construct the halo (non-iterable)
        halo = halo_obj.NFW_Lens(halos.x[i], halos.y[i], [0], halos.concentration[i], halos.mass[i], z_cluster, halos.chi2[i])
        kappa, area_per_pixel = nfw_projected_mass(
            halo,
            r_p=0,          # Dummy placeholder since we want the 2D grid
            return_2d=True,
            nx=nx,
            ny=ny,
            x_range=x_range,
            y_range=y_range,
            x_center=halo.x, 
            y_center=halo.y,
            z_source=z_source
        )
        if np.any(np.isinf(kappa)):
            print(f'Warning: NaN values found in kappa for halo at ({halo.x}, {halo.y}) with mass {halo.mass}. Skipping this halo.')
            continue
        kappa_total += kappa
    
    # Now find a way to break the mass sheet degeneracy
    # Choose a value of k
    if cluster_name == 'ABELL_2744':
        # k value taken from literature
        kappa_total = mass_sheet_transformation(kappa_total, k=2)
    elif cluster_name == 'EL_GORDO':
        kappa_total = mass_sheet_transformation(kappa_total, k=2)

    # Then convert back to a 2D mass distribution
    sigma_c = critical_surface_density(z_cluster, z_source)
    sigma_c = sigma_c * u.M_sun / u.kpc**2
    sigma_c = sigma_c.value
    M_2D_total = kappa_total * sigma_c * area_per_pixel
    
    # Coordinates of each pixel in the grid
    XX, YY = np.meshgrid(x_vals, y_vals)
    RR = np.sqrt(XX**2 + YY**2)  # Distance of each pixel from (0,0), adjust if needed
    
    # Compute the enclosed mass at each radius in r
    mass_enclosed = np.zeros_like(r)
    for i, radius in enumerate(r):
        mask = (RR <= radius)
        mass_enclosed[i] = np.sum(M_2D_total[mask])
    
    # Tell me how far off we are from the literature estimates
    for label, (mass_lit, r_lit) in mass_estimates.items():
        mass_recon = np.interp(r_lit, r, mass_enclosed) # Interpolate to find the mass at the literature radius
        # Write masses in scientific notation
        mass_lit_val = f'{mass_lit:.2e}'
        mass_recon_val = f'{mass_recon:.2e}'
        print(f'{label}: Literature mass = {mass_lit_val}, Reconstruction = {mass_recon_val}, % Error = {100 * (mass_recon - mass_lit) / mass_lit}')

    # Plot results
    fig, ax = plt.subplots()
    fig.suptitle(plot_title)
    
    # Our reconstruction
    ax.plot(r, mass_enclosed, label='Reconstruction')
    
    # Literature estimates
    markers = ['o', 's', 'D', '^']
    for i, (label, (mass_lit, r_lit)) in enumerate(mass_estimates.items()):
        ax.scatter(r_lit, mass_lit, label=label, marker=markers[i % len(markers)])
    
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'Mass ($M_\odot$)')
    ax.set_yscale('log')
    ax.legend() 
    plt.savefig(plot_name)
    
    halos.x += centroid[0] # Put the halos back where they were
    halos.y += centroid[1]


def perform_kaiser_squire_reconstruction(sources, extent, signal='shear',
                                         smoothing_scale=5.0, resolution_scale=1.0,
                                         weights=None, apodize=False):
    """
    Kaiser-Squires Fourier inversion to reconstruct convergence (kappa) from lensing signal.

    Parameters:
        sources: object with .x, .y, and lensing signal attributes ('e1','e2' or 'f1','f2')
        extent: (xmin, xmax, ymin, ymax) in arcseconds
        signal: 'shear' or 'flexion'
        smoothing_scale: real-space Gaussian smoothing (pixels)
        resolution_scale: pixels per arcsecond
        weights: 1D array of per-object weights (e.g., SNR^2), optional
        apodize: whether to apply a Tukey window to suppress edge effects

    Returns:
        X, Y: coordinate grids (2D)
        kappa: reconstructed convergence field (2D)
    """

    # Extract positions
    x_s, y_s = sources.x, sources.y

    if signal == 'shear':
        S1, S2 = sources.e1, sources.e2
    elif signal == 'flexion':
        S1, S2 = sources.f1, sources.f2
    else:
        raise ValueError("Signal must be 'shear' or 'flexion'.")

    # Default weights = 1
    if weights is None:
        weights = np.ones_like(x_s)

    # Grid settings
    xmin, xmax, ymin, ymax = extent
    fsize = max(xmax - xmin, ymax - ymin)
    npixels = int(resolution_scale * fsize)
    dx = fsize / npixels

    # Shift positions relative to grid
    x_s_rel = x_s - xmin - 0.5 * dx
    y_s_rel = y_s - ymin - 0.5 * dx

    # Weighted signal binning via CIC
    S1w = S1 * weights
    S2w = S2 * weights

    S1_grid = CIC_2d(fsize=fsize, npixels=npixels, xpos=x_s_rel, ypos=y_s_rel, signal_values=S1w)
    S2_grid = CIC_2d(fsize=fsize, npixels=npixels, xpos=x_s_rel, ypos=y_s_rel, signal_values=S2w)
    W_grid  = CIC_2d(fsize=fsize, npixels=npixels, xpos=x_s_rel, ypos=y_s_rel, signal_values=weights)

    # Normalize fields by weights to get weighted average
    S1_grid /= (W_grid + 1e-10)
    S2_grid /= (W_grid + 1e-10)

    # Optional real-space smoothing
    if smoothing_scale > 0:
        S1_grid = gaussian_filter(S1_grid, sigma=smoothing_scale)
        S2_grid = gaussian_filter(S2_grid, sigma=smoothing_scale)

    npixels = S1_grid.shape[0]

    # --- Apodization ---
    from scipy.signal.windows import hann
    if apodize:
        window = hann(npixels)
        apod_window = np.outer(window, window)
        S1_grid = S1_grid * apod_window
        S2_grid = S2_grid * apod_window

    # --- Padding (reflection to reduce discontinuities) ---
    pad_fraction = 0.2
    pad = int(npixels * pad_fraction)
    S1_padded = np.pad(S1_grid, pad_width=pad, mode='reflect')
    S2_padded = np.pad(S2_grid, pad_width=pad, mode='reflect')

    # FFT
    ft_S1 = np.fft.fft2(S1_padded, norm='ortho')
    ft_S2 = np.fft.fft2(S2_padded, norm='ortho')

    # Fourier-space coordinates
    lx = np.fft.fftfreq(S1_padded.shape[1], d=dx) * 2 * np.pi
    ly = np.fft.fftfreq(S1_padded.shape[0], d=dx) * 2 * np.pi
    Lx, Ly = np.meshgrid(lx, ly, indexing='xy')
    l_squared = Lx**2 + Ly**2
    l_squared[0, 0] = 1e-10  # prevent div by zero

    # Kaiser-Squires inversion kernel
    if signal == 'shear':
        ft_kappa = ((Lx**2 - Ly**2) * ft_S1 + 2 * Lx * Ly * ft_S2) / l_squared
    elif signal == 'flexion':
        ft_kappa = -1j * (Lx * ft_S1 + Ly * ft_S2) / l_squared

    # Optional Gaussian low-pass filter in Fourier space
    if smoothing_scale > 0:
        sigma_k = 1.0 / (smoothing_scale * dx)
        gaussian_k = np.exp(-0.5 * l_squared / sigma_k**2)
        ft_kappa *= gaussian_k

    # Inverse FFT to real space
    kappa_padded = np.fft.ifft2(ft_kappa, norm='ortho').real
    kappa = kappa_padded[pad:-pad, pad:-pad]

    # Grid coordinates
    x = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, npixels)
    y = np.linspace(ymin + 0.5 * dx, ymax - 0.5 * dx, npixels)
    X, Y = np.meshgrid(x, y, indexing='xy')

    return X, Y, kappa