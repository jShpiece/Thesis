"""
Utility functions for gravitational lensing analysis.

This module provides a collection of utility functions used in gravitational lensing studies,
including cosmological calculations, image processing, chi-squared utilities, and lensing signal
computations.

"""

import numpy as np
from astropy.constants import c, G
from astropy import units as u
from scipy.integrate import quad
import matplotlib.pyplot as plt
import arch.halo_obj as halo_obj
from scipy.ndimage import gaussian_filter, maximum_filter
from astropy.cosmology import Planck18 as cosmo

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


# ------------------------
# Lensing Utility Functions
# ------------------------

def find_peaks_and_masses(kappa_map, z_lens, z_source, radius_kpc=200):
    """
    Identify mass peaks and compute enclosed mass within a fixed radius.
    Assumes the kappa map is gridded in arcseconds (not pixels).

    Parameters:
    - kappa_map: 2D convergence map, gridded in arcseconds
    - z_lens: lens redshift
    - z_source: source redshift
    - radius_kpc: physical radius around each peak to sum kappa over
    - threshold: minimum relative value to consider a peak

    Returns:
    - peaks_arcsec: list of (RA_offset, Dec_offset) coordinates in arcsec
    - masses: corresponding mass values in M_sun
    """

    # Smooth the convergence map
    threshold = 0.1  # Minimum relative value to consider a peak
    kappa_smooth = gaussian_filter(kappa_map, sigma=2)
    local_max = maximum_filter(kappa_smooth, size=5)
    peaks_mask = (kappa_smooth == local_max) & (kappa_smooth > threshold * np.max(kappa_smooth))
    peak_indices = np.argwhere(peaks_mask)  # (y_arcsec, x_arcsec) integers

    # Conversion: arcsec → kpc
    arcsec_to_kpc = cosmo.angular_diameter_distance(z_lens).to(u.kpc).value * (np.pi / (180. * 3600.)) # 1 kpc in arcsec
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
        mass = sigma_c * enclosed_kappa * arcsec_to_kpc**2  # M_sun
        masses.append(np.abs(mass))

    return peaks_arcsec, masses

def calculate_kappa(lenses, extent, lens_type='SIS', source_redshift=0.8, k_val=0.95):
    """
    Calculates the convergence (kappa) map for a given set of lenses.

    Parameters:
        lenses: Lens object containing positions and strengths.  Type
            depends on `lens_type`:
              - SIS: SIS_Lens with .te
              - NFW: NFW_Lens with .mass, .concentration, .redshift
              - POWER_LAW: PowerLawHalo with .kappa_star, .slope,
                .theta_star, .redshift
        extent (tuple): (xmin, xmax, ymin, ymax) defining the map extent
            in arcseconds.
        lens_type (str): 'SIS', 'NFW', or 'POWER_LAW'.
        source_redshift (float): Source redshift used to compute the
            beta(z_s) factor for POWER_LAW (and the critical surface
            density for NFW).
        k_val (float): Mass-sheet transformation parameter (driving
            kappa to zero at the field edge).

    Returns:
        tuple: (X, Y, kappa) where X and Y are meshgrid arrays, and
            kappa is the convergence map.
    """
    xmin, xmax, ymin, ymax = extent

    # Pad the grid to drive kappa to zero at the edges
    pad_val = 100
    xmin -= pad_val
    xmax += pad_val
    ymin -= pad_val
    ymax += pad_val
    x_range = np.linspace(xmin, xmax, int(xmax - xmin))
    y_range = np.linspace(ymin, ymax, int(ymax - ymin))
    X, Y = np.meshgrid(x_range, y_range)
    kappa = np.zeros_like(X)

    # Radial NFW shape function
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

    elif lens_type == 'POWER_LAW':
        # kappa(theta) = beta(z_s) * kappa_star * (theta / theta_star)^(-n)
        #
        # kappa_star is defined at z_s -> infinity (Wright & Brainerd
        # convention).  At a finite source redshift we apply the
        # lensing-efficiency factor beta(z_s) = sigma_crit(inf) / sigma_crit(z_s)
        # = D_ls / D_s, which scales the at-infinity convergence to the
        # observation-plane value.
        z_l = float(lenses.redshift)
        if source_redshift <= z_l:
            beta = 0.0
        else:
            sigma_crit_zs = critical_surface_density(z_l, source_redshift)
            # sigma_crit at z_s -> infinity is c^2 / (4 pi G D_l)
            D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
            sigma_crit_inf = c.value ** 2 / (4 * np.pi * G.value * D_l)
            beta = sigma_crit_inf / sigma_crit_zs

        theta_star = float(lenses.theta_star)
        for k in range(len(lenses.x)):
            dx = X - lenses.x[k]
            dy = Y - lenses.y[k]
            r = np.hypot(dx, dy) + 0.5  # avoid central singularity (NFW convention)
            kappa += beta * lenses.kappa_star[k] * (r / theta_star) ** (-lenses.slope[k])

    else:
        raise ValueError("Invalid lens_type. Must be 'SIS', 'NFW', or 'POWER_LAW'.")

    # Break the mass sheet degeneracy by requiring kappa to go to zero at the edges
    kappa = mass_sheet_transformation(kappa, k=k_val)

    # Remove the padding
    X = X[pad_val:-pad_val, pad_val:-pad_val]
    Y = Y[pad_val:-pad_val, pad_val:-pad_val]
    kappa = kappa[pad_val:-pad_val, pad_val:-pad_val]

    return X, Y, kappa

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

def calculate_deflection_sis(lenses, theta_x, theta_y, eps=1.0e-6):
    """
    Compute the total SIS deflection field alpha(theta) from a set of SIS lenses.

    SIS deflection for one lens:
        alpha_vec = theta_E * (dtheta_vec / |dtheta|)

    Conventions:
        - theta_x, theta_y are image-plane coordinates in arcsec (same frame as lenses.x/y).
        - lenses.te is the Einstein radius theta_E in arcsec.
        - Returns alpha_x, alpha_y in arcsec.

    Parameters
    ----------
    lenses : SIS_Lens-like
        Must have array-like attributes: x, y, te (Einstein radius, arcsec).
        x,y are lens centers in arcsec.
    theta_x, theta_y : array-like
        Evaluation positions in arcsec.
    eps : float
        Softening to avoid division by zero at r=0 (arcsec).

    Returns
    -------
    alpha_x, alpha_y : ndarray
        Total deflection components at each evaluation point, shape (N_pts,).
    """
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)

    xl = np.atleast_1d(lenses.x).astype(float)
    yl = np.atleast_1d(lenses.y).astype(float)
    te = np.atleast_1d(lenses.te).astype(float)

    # Broadcast: (N_lens, N_pts)
    dx = tx[None, :] - xl[:, None]
    dy = ty[None, :] - yl[:, None]
    r = np.hypot(dx, dy)

    # Avoid r=0 singularity
    r = np.where(r < eps, eps, r)

    # Unit vector from lens to evaluation point
    ux = dx / r
    uy = dy / r

    # SIS deflection magnitude = theta_E
    alpha_x = np.sum(te[:, None] * ux, axis=0)
    alpha_y = np.sum(te[:, None] * uy, axis=0)

    return alpha_x, alpha_y

def backproject_source_positions_sis(lenses, theta_x, theta_y, eps=1.0e-6):
    """
    Back-project observed image positions theta -> source-plane positions beta
    using the SIS lens equation:
        beta = theta - alpha(theta)

    Returns
    -------
    beta_x, beta_y : ndarray, ndarray
        Source-plane coordinates (arcsec), shape (N_pts,).
    """
    ax, ay = calculate_deflection_sis(lenses, theta_x, theta_y, eps=eps)
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)
    return tx - ax, ty - ay

def magnification_sis(lenses, theta_x, theta_y, eps=1.0e-6):
    """
    Scalar magnification |mu| at arbitrary image-plane positions for a
    composite SIS deflector.

    For a single SIS halo j at angular separation r_j from the evaluation
    point, the deflection gradient (2x2 matrix) is:

        (d alpha_j / d theta)_ab = (theta_E,j / r_j) * P_ab(phi_j)

    where P_ab is the rank-1 projector along the lens-image direction:

        P = [[sin^2 phi,  -sin phi cos phi],
             [-sin phi cos phi,  cos^2 phi]]

    with phi_j = arctan2(dy_j, dx_j) measured from the lens centre.

    For N_lens SIS halos the total Jacobian is:

        A = I - sum_j (d alpha_j / d theta)

    and the signed magnification is mu = 1 / det(A).
    We return |mu| = 1 / |det(A)|.

    Analytic check (single SIS):
        det(A) = 1 - theta_E / r  =>  |mu| = 1 / |1 - theta_E / r|

    Parameters
    ----------
    lenses : SIS_Lens-like
        Must have array-like attributes x, y, te.
    theta_x, theta_y : array-like
        Image-plane positions in arcsec, shape (N_pts,).
    eps : float
        Softening length (arcsec) to avoid the r = 0 singularity.

    Returns
    -------
    abs_mu : ndarray, shape (N_pts,)
        Absolute magnification |mu| at each evaluation point.
    det_A : ndarray, shape (N_pts,)
        Signed determinant of the Jacobian (useful for parity checks).
    """
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)

    xl = np.atleast_1d(lenses.x).astype(float)
    yl = np.atleast_1d(lenses.y).astype(float)
    te = np.atleast_1d(lenses.te).astype(float)

    # Broadcast: shape (N_lens, N_pts)
    dx = tx[None, :] - xl[:, None]
    dy = ty[None, :] - yl[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)

    # Trig factors (N_lens, N_pts)
    cos_phi = dx / r
    sin_phi = dy / r

    # Prefactor theta_E / r for each lens-image pair
    te_over_r = te[:, None] / r  # (N_lens, N_pts)

    # Accumulate the four Jacobian components A = I - sum_j dα_j/dθ
    #   dα_x/dθ_x = (θ_E/r) sin²φ      dα_x/dθ_y = -(θ_E/r) sinφ cosφ
    #   dα_y/dθ_x = -(θ_E/r) sinφ cosφ  dα_y/dθ_y = (θ_E/r) cos²φ
    sum_dax_dtx = np.sum(te_over_r * sin_phi ** 2, axis=0)          # (N_pts,)
    sum_dax_dty = np.sum(-te_over_r * sin_phi * cos_phi, axis=0)    # (N_pts,)
    sum_day_dtx = sum_dax_dty                                       # symmetric
    sum_day_dty = np.sum(te_over_r * cos_phi ** 2, axis=0)          # (N_pts,)

    A11 = 1.0 - sum_dax_dtx
    A12 = -sum_dax_dty
    A21 = -sum_day_dtx
    A22 = 1.0 - sum_day_dty

    det_A = A11 * A22 - A12 * A21

    abs_mu = 1.0 / np.maximum(np.abs(det_A), 1.0e-30)

    return abs_mu, det_A

def sigma_beta_from_magnification(sigma_theta, abs_mu, mu_floor=0.01):
    """
    Convert image-plane positional uncertainty to source-plane uncertainty
    using the magnification.

    The lensing Jacobian maps image-plane displacements to source-plane
    displacements:  d(beta) = A * d(theta).  For isotropic image-plane
    errors sigma_theta, the scalar source-plane error is approximately:

        sigma_beta ≈ sigma_theta / |mu|

    where |mu| = 1/|det A|.  This is exact when the Jacobian is close
    to a scalar multiple of the identity (i.e. shear is subdominant
    compared to convergence), and remains a good approximation for the
    SIS profile where the tangential eigenvalue is unity.

    A floor on |mu| is imposed to prevent sigma_beta from diverging at
    the critical curve, where |mu| -> infinity and sigma_beta -> 0.
    Physically, images very close to the critical curve are smeared into
    arcs and their centroid uncertainty does not actually shrink to zero;
    the floor absorbs finite-source-size and PSF effects that regularise
    the divergence.

    Parameters
    ----------
    sigma_theta : float or ndarray
        Image-plane positional uncertainty (arcsec).
    abs_mu : ndarray
        Absolute magnification |mu| at each image position.
    mu_floor : float
        Minimum allowed inverse-magnification |1/mu|, i.e. maximum
        effective |mu| = 1/mu_floor.  Default 0.01 corresponds to
        |mu|_max = 100.

    Returns
    -------
    sigma_beta : ndarray
        Source-plane uncertainty at each image position (arcsec).
    """
    sigma_theta = np.atleast_1d(sigma_theta).astype(float)
    abs_mu = np.atleast_1d(abs_mu).astype(float)

    # inv_mu = 1/|mu|, floored to prevent divergence
    inv_mu = np.maximum(1.0 / np.maximum(abs_mu, 1.0e-30), mu_floor)

    return sigma_theta * inv_mu

def chi2_strong_source_plane_sis(lenses, strong_systems, eps=1.0e-6,
                                 return_breakdown=False,
                                 use_magnification_correction=True):
    """
    Source-plane scatter chi^2 for multiply-imaged systems under SIS lenses.

    For each system i with images m:
        beta_{i,m} = theta_{i,m} - alpha(theta_{i,m})
        beta_bar_i = weighted mean of beta_{i,m}
        chi2_i = sum_m |beta_{i,m} - beta_bar_i|^2 / sigma_beta^2

    When use_magnification_correction is True (default), sigma_beta is
    computed via the magnification tensor:

        sigma_beta_m = sigma_theta_m / |mu_m|

    This accounts for the compression of source-plane errors near the
    critical curve, giving highly magnified images their proper
    statistical weight.  When False, sigma_beta = sigma_theta (the
    original, approximate behaviour).

    Parameters
    ----------
    lenses : SIS_Lens-like
        Must have x, y, te arrays.
    strong_systems : iterable
        Iterable of StrongLensingSystem-like objects with attributes:
            - system_id : str
            - theta_x : array-like
            - theta_y : array-like
            - sigma_theta : float or array-like
        z_source can exist but is not used for SIS deflection in this minimal model.
    eps : float
        Softening for r=0 in deflection (arcsec).
    return_breakdown : bool
        If True, also return a dict keyed by system_id with per-system chi2 and metadata.
    use_magnification_correction : bool
        If True (default), convert sigma_theta to sigma_beta via the
        magnification.  If False, use sigma_theta directly as sigma_beta
        (original behaviour, retained for comparison tests).

    Returns
    -------
    chi2_sl : float
        Total chi^2 across systems.
    breakdown : dict (optional)
        Per-system diagnostics.
    """
    chi2_total = 0.0
    breakdown = {}

    for sys in strong_systems:
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)

        if tx.shape != ty.shape:
            raise ValueError(f"[{getattr(sys, 'system_id', 'unknown')}] theta_x/theta_y shape mismatch.")

        # Back-project to source plane
        bx, by = backproject_source_positions_sis(lenses, tx, ty, eps=eps)

        # ── Sigma handling ──────────────────────────────────────────
        # Start from image-plane positional uncertainty
        sig = getattr(sys, "sigma_theta", 0.1)
        if np.isscalar(sig):
            sig_theta = np.full_like(bx, float(sig), dtype=float)
        else:
            sig_theta = np.atleast_1d(sig).astype(float)
            if sig_theta.shape != bx.shape:
                raise ValueError(f"[{getattr(sys, 'system_id', 'unknown')}] sigma_theta shape mismatch.")

        if use_magnification_correction:
            # Compute magnification at each image position
            abs_mu, det_A = magnification_sis(lenses, tx, ty, eps=eps)
            # Convert to source-plane uncertainty: sigma_beta = sigma_theta / |mu|
            sigx = sigma_beta_from_magnification(sig_theta, abs_mu)
            sigy = sigma_beta_from_magnification(sig_theta, abs_mu)
        else:
            # Original behaviour: sigma_beta = sigma_theta (no correction)
            sigx = sig_theta
            sigy = sig_theta

        # Weighted mean source position
        wx = 1.0 / np.maximum(sigx, 1.0e-12) ** 2
        wy = 1.0 / np.maximum(sigy, 1.0e-12) ** 2

        bx_bar = np.sum(wx * bx) / np.sum(wx) 
        by_bar = np.sum(wy * by) / np.sum(wy)

        # Source-plane scatter chi2
        chi2_i = np.sum(((bx - bx_bar) / sigx) ** 2 + ((by - by_bar) / sigy) ** 2)

        chi2_total += float(chi2_i)

        if return_breakdown:
            sid = getattr(sys, "system_id", "unknown")
            bd = {
                "chi2": float(chi2_i),
                "n_images": int(bx.size),
                "beta_bar": (float(bx_bar), float(by_bar)),
                "beta": np.column_stack([bx, by]),
                "sigma_beta_x": sigx.copy(),
                "sigma_beta_y": sigy.copy(),
            }
            if use_magnification_correction:
                bd["abs_mu"] = abs_mu.copy()
                bd["det_A"] = det_A.copy()
                bd["sigma_theta"] = sig_theta.copy()
            breakdown[sid] = bd

    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total

def chi2_flux_sis(lenses, strong_systems, eps=1.0e-6,
                  return_breakdown=False):
    """
    Flux-ratio chi^2 for multiply-imaged systems under SIS lenses.

    For each system i with observed flux data, the model predicts
    flux ratios from the magnification:

        R_model_m = |mu_m| / |mu_ref|

    where |mu| is computed at each image position via magnification_sis.
    The observed ratios are computed inline from the system's flux and
    sigma_flux arrays (reference image = brightest).

    The chi^2 per system is:

        chi2_i = sum_{m != ref} [(R_obs_m - R_model_m) / sigma_R_m]^2

    Systems without flux data (has_flux=False) are silently skipped,
    so this function is backward-compatible with position-only systems.

    Parameters
    ----------
    lenses : SIS_Lens
        Must have x, y, te arrays.
    strong_systems : iterable of StrongLensingSystem
        Systems with optional flux and sigma_flux attributes.
    eps : float
        Softening for magnification evaluation (arcsec).
    return_breakdown : bool
        If True, also return per-system diagnostics.

    Returns
    -------
    chi2_flux : float
        Total flux-ratio chi^2 across all systems with flux data.
    breakdown : dict (optional)
        Per-system diagnostics keyed by system_id.
    """
    chi2_total = 0.0
    breakdown = {}

    for sls in strong_systems:
        if not getattr(sls, "has_flux", False):
            continue

        tx = np.atleast_1d(sls.theta_x).astype(float)
        ty = np.atleast_1d(sls.theta_y).astype(float)
        F = np.atleast_1d(sls.flux).astype(float)
        sigF = np.atleast_1d(sls.sigma_flux).astype(float)

        # Observed flux ratios relative to brightest image
        ref_idx = int(np.argmax(F))
        F_ref = F[ref_idx]
        sigF_ref = sigF[ref_idx]
        R_obs = F / F_ref
        frac_i = sigF / np.maximum(F, 1e-30)
        frac_ref = sigF_ref / max(F_ref, 1e-30)
        sigma_R = R_obs * np.sqrt(frac_i**2 + frac_ref**2)
        sigma_R[ref_idx] = 0.0

        # Model magnifications at each image position
        abs_mu, det_A = magnification_sis(lenses, tx, ty, eps=eps)

        # Model flux ratios
        mu_ref = abs_mu[ref_idx]
        R_model = abs_mu / np.maximum(mu_ref, 1.0e-30)

        # chi2: skip the reference image (sigma_R = 0 there)
        mask = np.arange(len(tx)) != ref_idx
        residuals = (R_obs[mask] - R_model[mask]) / np.maximum(sigma_R[mask], 1.0e-30)
        chi2_i = float(np.sum(residuals**2))
        chi2_total += chi2_i

        if return_breakdown:
            sid = getattr(sls, "system_id", "unknown")
            breakdown[sid] = {
                "chi2": chi2_i,
                "n_images": int(tx.size),
                "ref_index": ref_idx,
                "R_obs": R_obs.copy(),
                "R_model": R_model.copy(),
                "sigma_R": sigma_R.copy(),
                "abs_mu": abs_mu.copy(),
                "det_A": det_A.copy(),
            }

    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total

def calculate_lensing_signals_power_law(halos, sources):
    """
    Lensing signals (shear, flexion, g-flexion) for power-law convergence
    halos with per-source redshifts.

    Profile (per halo, at z_s -> infinity):
        kappa_inf(theta) = kappa_star * (theta / theta_star) ** (-n)

    For a source at finite z_s, the effective convergence scales with
    the lensing efficiency beta(z_s) = D_ls / D_s = Sigma_crit(z_l, inf)
    / Sigma_crit(z_l, z_s).  Sources in front of the lens (z_s <= z_l)
    contribute zero.

    Component identities (from Phase 0 derivations):
        gamma_t(theta) = [n / (2 - n)] * kappa(theta)
        |F|(theta)     = n * kappa(theta) / theta
        |G|(theta)     = [n * (2 + n) / (2 - n)] * kappa(theta) / theta

    Sign conventions match calculate_lensing_signals_sis:
        shear_mag     = -gamma_t   (ARCH sign)
        flexion_mag   = -|F|       (F points radially inward, toward halo)
        g_flexion_mag = +|G|       (G radially outward, spin-3)

    Parameters:
        halos: PowerLawHalo object containing positions (x, y), kappa_star,
            slope, theta_star, and redshift. All halos assumed at the same
            redshift.
        sources: Source object containing positions (x, y) and per-source
            redshifts (redshift).

    Returns:
        tuple: (shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2)
        Each is shape (N_sources,).
    """
    # --- Normalize halo and source fields ---
    x_l = np.atleast_1d(halos.x)
    y_l = np.atleast_1d(halos.y)
    k_star = np.atleast_1d(halos.kappa_star)
    n_l = np.atleast_1d(halos.slope)
    theta_star = float(halos.theta_star)

    x_s = np.atleast_1d(sources.x)
    y_s = np.atleast_1d(sources.y)
    z_s = np.atleast_1d(sources.redshift).astype(float)

    # Lens redshift: scalar or array with identical values (matches NFW pattern)
    z_l_arr = np.asarray(halos.redshift)
    if z_l_arr.ndim == 0:
        z_l = float(z_l_arr)
    else:
        z_l = float(z_l_arr.flat[0])
        if not np.allclose(z_l_arr, z_l, rtol=0, atol=1e-10):
            raise ValueError("All halos must share the same redshift "
                             "for this function.")

    N_h, N_s = x_l.size, x_s.size

    # --- Lensing efficiency beta(z_s) = D_ls / D_s ---
    # Equivalent to Sigma_crit(z_l, inf) / Sigma_crit(z_l, z_s).
    # Use existing critical_surface_density helper for consistency with NFW.
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit_inf = c.value ** 2 / (4.0 * np.pi * G.value * Dl)

    def _sigma_crit_vec(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(
                lambda zs: critical_surface_density(zl_scalar, zs),
                otypes=[float],
            )
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit_vec(z_l, z_s)            # (N_s,)
    behind = z_s > z_l
    # beta = sigma_crit_inf / sigma_crit_s, zero for non-behind sources
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)
    beta_hs = beta_s[None, :]                           # (1, N_s) -> broadcasts to (N_h, N_s)

    # --- Angular separations (arcsec) ---
    dx = x_s[None, :] - x_l[:, None]                    # (N_h, N_s)
    dy = y_s[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r == 0.0, 1.0e-2, r)                   # softening, matches NFW

    # Trig combinations for spin-2 (shear) and spin-3 (g-flexion)
    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi ** 2 - sin_phi ** 2
    sin2phi = 2.0 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    # --- Profile evaluation per (halo, source) ---
    # Broadcast slope and kappa_star across sources
    n_hs = n_l[:, None]                                 # (N_h, 1)
    k_hs = k_star[:, None]                              # (N_h, 1)

    # Guard against the formal n=2 divergence
    two_minus_n = 2.0 - n_hs
    two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)

    # kappa(theta, z_s) = beta(z_s) * kappa_star * (theta/theta_star)^(-n)
    kappa = beta_hs * k_hs * (r / theta_star) ** (-n_hs)   # (N_h, N_s)

    # Component magnitudes per (halo, source), unsigned
    gamma_t = (n_hs / two_minus_n) * kappa
    F_amp = n_hs * kappa / r
    G_amp = (n_hs * (2.0 + n_hs) / two_minus_n) * kappa / r

    # Apply ARCH sign conventions
    shear_mag = -gamma_t
    flexion_mag = -F_amp
    g_flexion_mag = +G_amp

    # --- Sum over halos ---
    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flexion_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flexion_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    return shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2

def calculate_deflection_power_law(halos, theta_x, theta_y, z_source, eps=1.0e-6):
    """
    Compute the total deflection field alpha(theta) for a set of power-law
    convergence halos.

    Power-law deflection for one halo (axisymmetric, 0 < n < 2):
        alpha_vec(theta) = beta(z_s) * (2 * kappa_star / (2 - n))
                           * theta_star^n * r^(-n) * (theta_vec - theta_lens)
    where r = |theta_vec - theta_lens| and
        beta(z_s) = D_ls(z_l, z_s) / D_s(z_s)
                  = Sigma_crit(z_l, inf) / Sigma_crit(z_l, z_s)
    is the lensing efficiency, with kappa_star defined at z_s -> infinity
    (matches calculate_lensing_signals_power_law convention).

    Sources in front of the lens (z_s <= z_l) contribute zero deflection.

    Conventions:
        - theta_x, theta_y are image-plane coordinates in arcsec, in the
          same frame as halos.x / halos.y.
        - alpha points from the halo center toward the evaluation point,
          so the lens equation beta = theta - alpha pulls source positions
          back toward the halo (consistent with calculate_deflection_sis).
        - Returns alpha_x, alpha_y in arcsec.

    Parameters
    ----------
    halos : PowerLawHalo-like
        Must have array-like attributes x, y, kappa_star, slope, plus
        scalar theta_star and redshift.
    theta_x, theta_y : array-like
        Evaluation positions in arcsec.
    z_source : float or array-like
        Source redshift. Either a scalar (broadcast to all evaluation
        points) or an array matching theta_x.
    eps : float
        Softening to avoid division by zero at r=0 (arcsec).

    Returns
    -------
    alpha_x, alpha_y : ndarray
        Total deflection components at each evaluation point, shape (N_pts,).
    """
    # --- Normalize inputs ---
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)

    xl = np.atleast_1d(halos.x).astype(float)
    yl = np.atleast_1d(halos.y).astype(float)
    k_star = np.atleast_1d(halos.kappa_star).astype(float)
    n_l = np.atleast_1d(halos.slope).astype(float)
    theta_star = float(halos.theta_star)

    # Lens redshift: scalar or array with identical values
    z_l_arr = np.asarray(halos.redshift)
    if z_l_arr.ndim == 0:
        z_l = float(z_l_arr)
    else:
        z_l = float(z_l_arr.flat[0])
        if not np.allclose(z_l_arr, z_l, rtol=0, atol=1e-10):
            raise ValueError("All halos must share the same redshift "
                             "for this function.")

    # Source redshift: scalar or array matching theta_x
    z_s = np.atleast_1d(z_source).astype(float)
    if z_s.size == 1 and tx.size > 1:
        z_s = np.full_like(tx, z_s[0])
    elif z_s.size != tx.size:
        raise ValueError(f"z_source size {z_s.size} does not match "
                         f"theta_x size {tx.size}.")

    # --- Lensing efficiency beta(z_s) = D_ls / D_s ---
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit_inf = c.value ** 2 / (4.0 * np.pi * G.value * Dl)

    def _sigma_crit_vec(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(
                lambda zs: critical_surface_density(zl_scalar, zs),
                otypes=[float],
            )
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit_vec(z_l, z_s)            # (N_pts,)
    behind = z_s > z_l
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)

    # --- Geometry: (N_lens, N_pts) ---
    dx = tx[None, :] - xl[:, None]
    dy = ty[None, :] - yl[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)                       # softening

    # --- Per-halo deflection magnitude prefactor ---
    # alpha_mag(theta) = beta(z_s) * [2 kappa_star / (2-n)] * theta_star^n * theta^(1-n)
    # Cast slope/kappa_star to (N_lens, 1) for broadcasting against (N_lens, N_pts).
    n_h = n_l[:, None]
    k_h = k_star[:, None]
    two_minus_n = 2.0 - n_h
    two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)

    coeff = (2.0 * k_h / two_minus_n) * (theta_star ** n_h)   # (N_lens, 1)
    # alpha as a vector: coeff * r^(-n) * (dx, dy)
    # NB: r^(1-n) * (dx/r) = r^(-n) * dx, which is what we want.
    factor = coeff * r ** (-n_h)                              # (N_lens, N_pts)

    # Apply per-source efficiency (broadcasts over lens axis)
    factor = factor * beta_s[None, :]

    # --- Sum over halos ---
    alpha_x = np.sum(factor * dx, axis=0)
    alpha_y = np.sum(factor * dy, axis=0)

    return alpha_x, alpha_y

def backproject_source_positions_power_law(halos, theta_x, theta_y, z_source,
                                           eps=1.0e-6):
    """
    Back-project observed image positions theta -> source-plane positions beta
    using the power-law lens equation:
        beta = theta - alpha(theta)

    Parameters
    ----------
    halos : PowerLawHalo-like
    theta_x, theta_y : array-like
        Image-plane positions (arcsec).
    z_source : float or array-like
        Source redshift(s) for the images.
    eps : float
        Softening for r=0 in deflection (arcsec).

    Returns
    -------
    beta_x, beta_y : ndarray
        Source-plane coordinates (arcsec), shape (N_pts,).
    """
    ax, ay = calculate_deflection_power_law(
        halos, theta_x, theta_y, z_source, eps=eps
    )
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)
    return tx - ax, ty - ay

def magnification_power_law(halos, theta_x, theta_y, z_source, eps=1.0e-6):
    """
    Scalar magnification |mu| at arbitrary image-plane positions for a
    composite power-law deflector.

    For a single power-law halo j the deflection is
        alpha_vec = M_j(r) * (theta_vec - theta_lens),
    where M_j(r) = beta(z_s) * (2 kappa_star_j / (2 - n_j))
                   * theta_star^{n_j} * r^{-n_j}.

    The deflection gradient is
        d alpha_i / d theta_k = M_j(r) * (delta_ik - n_j * theta_hat_i theta_hat_k),
    where theta_hat is the unit vector from halo center to image.  For the
    axisymmetric single-halo case this gives the clean closed form
        det A = (1 - M)(1 - (1 - n) M),
    used as a sharp consistency check in the test suite.

    For N_halo halos the total Jacobian is
        A_ab = delta_ab - sum_j d alpha_j_a / d theta_b
    and |mu| = 1 / |det A|.

    Parameters
    ----------
    halos : PowerLawHalo
        Must have x, y, kappa_star, slope arrays plus scalar theta_star
        and redshift.
    theta_x, theta_y : array-like
        Image-plane positions (arcsec).
    z_source : float or array-like
        Source redshift(s).  Scalar broadcasts to all positions.
    eps : float
        Softening for r=0 (arcsec).

    Returns
    -------
    abs_mu : ndarray, shape (N_pts,)
        Absolute magnification |mu|.
    det_A : ndarray, shape (N_pts,)
        Signed determinant of the Jacobian (useful for parity checks).
    """
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)

    xl = np.atleast_1d(halos.x).astype(float)
    yl = np.atleast_1d(halos.y).astype(float)
    k_star = np.atleast_1d(halos.kappa_star).astype(float)
    n_l = np.atleast_1d(halos.slope).astype(float)
    theta_star = float(halos.theta_star)

    # Lens redshift
    z_l_arr = np.asarray(halos.redshift)
    if z_l_arr.ndim == 0:
        z_l = float(z_l_arr)
    else:
        z_l = float(z_l_arr.flat[0])
        if not np.allclose(z_l_arr, z_l, rtol=0, atol=1e-10):
            raise ValueError("All halos must share the same redshift.")

    # Source redshift -> per-image
    z_s = np.atleast_1d(z_source).astype(float)
    if z_s.size == 1 and tx.size > 1:
        z_s = np.full_like(tx, z_s[0])
    elif z_s.size != tx.size:
        raise ValueError("z_source size must be 1 or match theta_x.")

    # Lensing efficiency beta(z_s) per image
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit_inf = c.value ** 2 / (4.0 * np.pi * G.value * Dl)

    def _sigma_crit_vec(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(
                lambda zs: critical_surface_density(zl_scalar, zs),
                otypes=[float],
            )
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit_vec(z_l, z_s)
    behind = z_s > z_l
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)   # (N_pts,)

    # Geometry: (N_lens, N_pts)
    dx = tx[None, :] - xl[:, None]
    dy = ty[None, :] - yl[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)

    cos_phi = dx / r
    sin_phi = dy / r

    # M_j(r) per (halo, image)
    n_h = n_l[:, None]
    k_h = k_star[:, None]
    two_minus_n = 2.0 - n_h
    two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)
    M = (2.0 * k_h / two_minus_n) * (theta_star ** n_h) * r ** (-n_h)
    M = M * beta_s[None, :]

    # Per-halo Jacobian terms d alpha_a / d theta_b = M * (delta_ab - n hat_a hat_b)
    # Sum over halos
    sum_dax_dtx = np.sum(M * (1.0 - n_h * cos_phi ** 2), axis=0)   # (N_pts,)
    sum_dax_dty = np.sum(-M * n_h * cos_phi * sin_phi, axis=0)
    sum_day_dtx = sum_dax_dty                                       # symmetric
    sum_day_dty = np.sum(M * (1.0 - n_h * sin_phi ** 2), axis=0)

    A11 = 1.0 - sum_dax_dtx
    A12 = -sum_dax_dty
    A21 = -sum_day_dtx
    A22 = 1.0 - sum_day_dty

    det_A = A11 * A22 - A12 * A21
    abs_mu = 1.0 / np.maximum(np.abs(det_A), 1.0e-30)

    return abs_mu, det_A

def chi2_strong_source_plane_power_law(halos, strong_systems,
                                       sigma_n=None,
                                       alpha_cal=1.0,
                                       eps=1.0e-6,
                                       return_breakdown=False,
                                       use_magnification_correction=True,
                                       use_profile_uncertainty=True):
    """
    Source-plane scatter chi^2 for multiply-imaged systems under power-law
    halos.

    For each system i with images m at redshift z_s,i:
        beta_{i,m} = theta_{i,m} - alpha(theta_{i,m}, z_s,i)
        beta_bar_i = inverse-variance weighted mean of beta_{i,m}
        chi2_i     = sum_m |beta_{i,m} - beta_bar_i|^2 / sigma_beta_total^2

    sigma_beta_total^2 combines (in quadrature):

        Measurement term:
            sigma_beta_meas = sigma_theta / |mu(theta_m, z_s)|
            (when use_magnification_correction = True; else sigma_theta)

        Profile-uncertainty term (Phase 0):
            sigma_beta_prof_j(theta_m) =
                alpha_cal * |alpha_j(theta_m, z_s)| * sigma_n_j * |ln(r_jm/theta_star)|
            sigma_beta_prof(theta_m)^2 = sum_j sigma_beta_prof_j(theta_m)^2

        Total:
            sigma_beta_total^2 = sigma_beta_meas^2 + sigma_beta_prof^2

    The profile-uncertainty term encodes the residual degeneracy between
    the parametric power-law and the true mass distribution.  It uses
    sigma_n_j (per-halo posterior uncertainty on the slope, computed
    once after WL fitting) and the |ln(r/theta_star)| factor that emerges
    from differentiating the deflection w.r.t. n at fixed theta.

    Parameters
    ----------
    halos : PowerLawHalo
    strong_systems : iterable of StrongLensingSystem
        Each system carries: theta_x, theta_y, sigma_theta, z_source,
        and (optionally) system_id.
    sigma_n : array-like or None
        Per-halo posterior uncertainty on the slope (length matches
        halos.x).  If None or all zeros, the profile-uncertainty term is
        skipped (sigma_beta_total = sigma_beta_meas).
    alpha_cal : float
        Empirical calibration coefficient for the profile term, default 1.
    eps : float
        Softening for r=0 in deflection (arcsec).
    return_breakdown : bool
        If True, also return per-system diagnostics.
    use_magnification_correction : bool
        If True, sigma_beta_meas = sigma_theta / |mu|; else sigma_theta.
    use_profile_uncertainty : bool
        If False, skip sigma_beta_prof regardless of sigma_n.

    Returns
    -------
    chi2_sl : float
        Total chi^2 across all systems.
    breakdown : dict (optional)
        Per-system diagnostics keyed by system_id.
    """
    # Halo geometry / parameters once (independent of system)
    xl = np.atleast_1d(halos.x).astype(float)
    yl = np.atleast_1d(halos.y).astype(float)
    k_star = np.atleast_1d(halos.kappa_star).astype(float)
    n_l = np.atleast_1d(halos.slope).astype(float)
    theta_star = float(halos.theta_star)
    N_h = xl.size

    # Slope-uncertainty handling
    if (not use_profile_uncertainty) or sigma_n is None:
        sig_n = np.zeros(N_h)
    else:
        sig_n = np.atleast_1d(sigma_n).astype(float)
        if sig_n.size != N_h:
            raise ValueError(f"sigma_n length {sig_n.size} != number of "
                             f"halos {N_h}.")

    chi2_total = 0.0
    breakdown = {}

    for sys in strong_systems:
        sid = getattr(sys, "system_id", "unknown")
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)
        if tx.shape != ty.shape:
            raise ValueError(f"[{sid}] theta_x/theta_y shape mismatch.")

        z_s = float(getattr(sys, "z_source", 1.5))
        N_im = tx.size

        # --- Back-project to source plane ---
        bx, by = backproject_source_positions_power_law(
            halos, tx, ty, z_s, eps=eps,
        )

        # --- Image-plane sigma_theta (scalar or per-image) ---
        sig = getattr(sys, "sigma_theta", 0.1)
        if np.isscalar(sig):
            sig_theta = np.full_like(bx, float(sig), dtype=float)
        else:
            sig_theta = np.atleast_1d(sig).astype(float)
            if sig_theta.shape != bx.shape:
                raise ValueError(f"[{sid}] sigma_theta shape mismatch.")

        # --- Measurement source-plane sigma ---
        if use_magnification_correction:
            abs_mu, det_A = magnification_power_law(
                halos, tx, ty, z_s, eps=eps,
            )
            sig_beta_meas = sigma_beta_from_magnification(sig_theta, abs_mu)
        else:
            sig_beta_meas = sig_theta
            abs_mu = None
            det_A = None

        # --- Profile-uncertainty source-plane sigma ---
        if np.any(sig_n > 0.0):
            # Per-halo deflection magnitude and distance to each image
            dx = tx[None, :] - xl[:, None]                   # (N_h, N_im)
            dy = ty[None, :] - yl[:, None]
            r = np.hypot(dx, dy)
            r = np.where(r < eps, eps, r)

            # Need alpha magnitude per halo per image — compute via the
            # closed form rather than calling calculate_deflection_power_law
            # (which sums over halos).
            Dl = cosmo.angular_diameter_distance(halos.redshift).to(u.m).value
            sigma_crit_inf = c.value ** 2 / (4.0 * np.pi * G.value * Dl)
            sigma_crit_s = critical_surface_density(halos.redshift, z_s)
            beta_zs = sigma_crit_inf / sigma_crit_s if z_s > halos.redshift else 0.0

            two_minus_n = 2.0 - n_l
            two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)
            coeff = (2.0 * k_star / two_minus_n) * (theta_star ** n_l)  # (N_h,)

            alpha_mag = beta_zs * coeff[:, None] * r ** (1.0 - n_l[:, None])  # (N_h, N_im)

            # Phase 0:  sigma_prof_j(theta) = alpha_cal * |alpha_j(theta)| *
            #                                 sigma_n_j * |ln(r_jm / theta_star)|
            ln_term = np.abs(np.log(r / theta_star))                 # (N_h, N_im)
            sig_per_halo = (alpha_cal * alpha_mag
                            * sig_n[:, None] * ln_term)              # (N_h, N_im)
            # RSS over halos (independent slope uncertainties)
            sig_beta_prof = np.sqrt(np.sum(sig_per_halo ** 2, axis=0))   # (N_im,)
        else:
            sig_beta_prof = np.zeros(N_im)

        # Total
        sig_beta = np.sqrt(sig_beta_meas ** 2 + sig_beta_prof ** 2)
        sig_beta = np.maximum(sig_beta, 1.0e-12)

        # --- Inverse-variance weighted mean source position ---
        w = 1.0 / sig_beta ** 2
        bx_bar = np.sum(w * bx) / np.sum(w)
        by_bar = np.sum(w * by) / np.sum(w)

        # --- Source-plane scatter chi2 for this system ---
        chi2_i = np.sum(((bx - bx_bar) / sig_beta) ** 2
                        + ((by - by_bar) / sig_beta) ** 2)
        chi2_total += float(chi2_i)

        if return_breakdown:
            bd = {
                "chi2": float(chi2_i),
                "n_images": int(N_im),
                "z_source": z_s,
                "beta_bar": (float(bx_bar), float(by_bar)),
                "beta": np.column_stack([bx, by]),
                "sigma_beta_meas": sig_beta_meas.copy(),
                "sigma_beta_prof": sig_beta_prof.copy(),
                "sigma_beta_total": sig_beta.copy(),
            }
            if abs_mu is not None:
                bd["abs_mu"] = abs_mu.copy()
                bd["det_A"] = det_A.copy()
            breakdown[sid] = bd

    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total

def chi2_flux_power_law(halos, strong_systems,
                        eps=1.0e-6,
                        return_breakdown=False,
                        marginalize_normalization=True):
    """
    Flux-ratio chi-squared for multiply-imaged systems under power-law halos.

    For each system i with images m=1..N at redshift z_s,i, the model
    magnification at image m is

        |mu_m|_mod = 1 / |det A(theta_m, z_s,i)|,

    computed via magnification_power_law (closed-form Jacobian for the
    power-law profile).  The function then compares model to observed
    flux ratios.

    Two conventions are supported:

    1. **Flux-ratio chi2 with marginalization (default):**
       Observed fluxes f_m^obs and uncertainties sigma_f,m are read from
       sys.meta["flux"] and sys.meta["sigma_flux"].  An overall flux
       scale s is marginalized analytically (closed-form linear least
       squares):

           s_hat  = sum_m (f_m^obs * |mu_m|_mod / sigma_f,m^2)
                    / sum_m (|mu_m|_mod^2 / sigma_f,m^2)

           chi2_i = sum_m (f_m^obs - s_hat * |mu_m|_mod)^2 / sigma_f,m^2

       Equivalent to chi2 on flux ratios with the optimal reference
       chosen, but numerically more stable (no division by a noisy
       reference flux).

    2. **Flux-ratio chi2 with explicit reference (marginalize=False):**
       Observed flux ratios r_m = f_m^obs / f_1^obs and uncertainties
       sigma_r,m are read from sys.meta["flux_ratios"] and
       sys.meta["sigma_flux_ratios"], both length N (with index 0 the
       reference, conventionally r_0=1, sigma_r,0=0).  Image 0 is
       skipped in the sum:

           chi2_i = sum_{m>0} (r_m - |mu_m|/|mu_0|)^2 / sigma_r,m^2

    Systems with no flux data (no `flux`/`flux_ratios` keys in
    sys.meta) contribute zero chi2 and are silently skipped.

    Architectural note: the flux-ratio computation is INLINED in this
    function, not exposed as a `flux_ratios()` property on the halo
    class.  This mirrors the convention adopted for chi2_flux_nfw /
    chi2_flux_sis after a flux_ratios property caused a TypeError in
    the NFW path.

    Parameters
    ----------
    halos : PowerLawHalo
    strong_systems : iterable of StrongLensingSystem
        Each system MAY carry, in its meta dict:
            "flux"             : per-image observed fluxes (length N).
            "sigma_flux"       : per-image flux uncertainties (length N).
        OR, if marginalize_normalization=False:
            "flux_ratios"      : f_m / f_0 ratios (length N, ratios[0]=1).
            "sigma_flux_ratios": uncertainties on the ratios (length N).
        Systems with neither set are skipped (no chi2 contribution).
    eps : float
        Softening for r=0 in the magnification calculation (arcsec).
    return_breakdown : bool
        If True, also return a per-system diagnostics dict.
    marginalize_normalization : bool
        Selects between the two conventions above.

    Returns
    -------
    chi2_flux : float
        Total flux-ratio chi^2 across systems.
    breakdown : dict (optional)
        Per-system diagnostics keyed by system_id.
    """
    chi2_total = 0.0
    breakdown = {}

    for sys in strong_systems:
        sid = getattr(sys, "system_id", "unknown")
        meta = getattr(sys, "meta", {}) or {}

        if marginalize_normalization:
            # Look for per-image fluxes and uncertainties
            f_obs = meta.get("flux", None)
            sig_f = meta.get("sigma_flux", None)
            if f_obs is None or sig_f is None:
                if return_breakdown:
                    breakdown[sid] = {"chi2": 0.0, "skipped": True}
                continue
            f_obs = np.atleast_1d(f_obs).astype(float)
            sig_f = np.atleast_1d(sig_f).astype(float)
            tx = np.atleast_1d(sys.theta_x).astype(float)
            ty = np.atleast_1d(sys.theta_y).astype(float)
            if f_obs.size != tx.size:
                raise ValueError(f"[{sid}] flux length {f_obs.size} != "
                                 f"theta_x length {tx.size}.")
            if sig_f.shape != f_obs.shape:
                raise ValueError(f"[{sid}] sigma_flux shape mismatch.")

            # Model magnifications at each image
            abs_mu, _ = magnification_power_law(
                halos, tx, ty, sys.z_source, eps=eps,
            )

            # Closed-form marginalization over the overall flux scale s.
            # Minimizes  sum_m ((f_m - s |mu_m|) / sigma_m)^2  over s:
            w = 1.0 / np.maximum(sig_f, 1.0e-30) ** 2
            num = np.sum(f_obs * abs_mu * w)
            den = np.sum(abs_mu ** 2 * w)
            s_hat = num / den if den > 0 else 0.0

            residuals = f_obs - s_hat * abs_mu
            chi2_i = float(np.sum((residuals / np.maximum(sig_f, 1.0e-30)) ** 2))

            if return_breakdown:
                breakdown[sid] = {
                    "chi2": chi2_i,
                    "n_images": int(tx.size),
                    "z_source": float(sys.z_source),
                    "abs_mu_mod": abs_mu.copy(),
                    "s_hat": float(s_hat),
                    "predicted_flux": (s_hat * abs_mu).copy(),
                    "residual_flux": residuals.copy(),
                    "skipped": False,
                }

        else:
            # Explicit-reference flux ratios
            r_obs = meta.get("flux_ratios", None)
            sig_r = meta.get("sigma_flux_ratios", None)
            if r_obs is None or sig_r is None:
                if return_breakdown:
                    breakdown[sid] = {"chi2": 0.0, "skipped": True}
                continue
            r_obs = np.atleast_1d(r_obs).astype(float)
            sig_r = np.atleast_1d(sig_r).astype(float)
            tx = np.atleast_1d(sys.theta_x).astype(float)
            ty = np.atleast_1d(sys.theta_y).astype(float)
            if r_obs.size != tx.size:
                raise ValueError(f"[{sid}] flux_ratios length != n_images.")
            if sig_r.shape != r_obs.shape:
                raise ValueError(f"[{sid}] sigma_flux_ratios shape mismatch.")

            abs_mu, _ = magnification_power_law(
                halos, tx, ty, sys.z_source, eps=eps,
            )
            mu_ref = abs_mu[0]
            if mu_ref <= 0.0:
                # Pathological case — reference image at a critical curve
                if return_breakdown:
                    breakdown[sid] = {"chi2": 0.0, "skipped": True,
                                      "reason": "mu_ref <= 0"}
                continue
            r_mod = abs_mu / mu_ref

            # Skip index 0 (the reference, by definition r=1 +- 0)
            chi2_i = 0.0
            for m in range(1, r_obs.size):
                if sig_r[m] > 0:
                    chi2_i += ((r_obs[m] - r_mod[m]) / sig_r[m]) ** 2
            chi2_i = float(chi2_i)

            if return_breakdown:
                breakdown[sid] = {
                    "chi2": chi2_i,
                    "n_images": int(tx.size),
                    "z_source": float(sys.z_source),
                    "abs_mu_mod": abs_mu.copy(),
                    "flux_ratios_mod": r_mod.copy(),
                    "skipped": False,
                }

        chi2_total += chi2_i

    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total


# ────────────────────────────────────────────────────────
#  NFW strong-lensing functions
# ────────────────────────────────────────────────────────

def _nfw_radial_g(x):
    """
    Numerically stable g(x) = ln(x/2) + h(x) for the NFW deflection integral.

    The direct formula g = ln(x/2) + h(x) suffers catastrophic cancellation
    for x << 1, where both terms are O(ln(1/x)) with opposite signs.

    For x < 1 we instead use the equivalent form:

        g(x) = f(sqrt(1-x^2)) / (2*sqrt(1-x^2))
        f(s)  = (s+1)*ln(1+s) + (s-1)*ln(1-s) - 2*s*ln(2)

    which avoids the large cancellation and is accurate to ~15 significant
    figures for all x in (0, 1).

    For x > 1 the direct formula ln(x/2) + arctan(sqrt(x^2-1))/sqrt(x^2-1)
    has no cancellation and is computed directly.

    Limiting values: g(0) = 0, g(1) = 1 - ln(2) ≈ 0.307.
    """
    x = np.asarray(x, dtype=float)
    g = np.empty_like(x)
    ln2 = np.log(2.0)

    m_lt1 = x < 1
    m_eq1 = np.abs(x - 1) < 1e-8
    m_gt1 = x > 1 + 1e-8

    # x < 1: stable formula
    xs = x[m_lt1]
    xs_safe = np.where(xs < 1e-10, 1e-10, xs)  # avoid underflow in 1-x^2
    s = np.sqrt(np.maximum(1.0 - xs_safe**2, 0.0))
    s_safe = np.where(s < 1e-300, 1e-300, s)
    one_minus_s = np.maximum(1.0 - s, 1e-300)
    f_s = (s + 1) * np.log(1 + s) + (s - 1) * np.log(one_minus_s) - 2 * s * ln2
    g[m_lt1] = f_s / (2 * s_safe)

    # x ≈ 1: limit g(1) = 1 - ln(2)
    g[m_eq1] = 1.0 - ln2

    # x > 1: direct formula (no cancellation)
    xg = x[m_gt1]
    h_g = np.arctan(np.sqrt(xg**2 - 1)) / np.sqrt(xg**2 - 1)
    g[m_gt1] = np.log(xg / 2.0) + h_g

    return g

def _nfw_radial_h(x):
    """
    Radial function h(x) for NFW profile (identical to radial_term_5 in
    calculate_lensing_signals_nfw, extracted here for standalone use).
 
        h(x) = arctanh(sqrt(1-x^2)) / sqrt(1-x^2)   for x < 1
             = 1                                       for x = 1
             = arctan(sqrt(x^2-1)) / sqrt(x^2-1)      for x > 1
 
    Parameters
    ----------
    x : ndarray
        Dimensionless radius x = theta / theta_s.
 
    Returns
    -------
    h : ndarray
        Same shape as x.
    """
    x = np.asarray(x, dtype=float)
    sol = np.ones_like(x)
    m1 = x < 1
    m3 = x > 1
    sol[m1] = np.arctanh(np.sqrt(1 - x[m1]**2)) / np.sqrt(1 - x[m1]**2)
    sol[m3] = np.arctan(np.sqrt(x[m3]**2 - 1)) / np.sqrt(x[m3]**2 - 1)
    return sol

def _nfw_kappa_and_gamma(kappa_s, x, h_x=None, g_x=None):
    """
    Convergence kappa(x) and tangential shear |gamma_t(x)| for an NFW lens.
 
    Parameters
    ----------
    kappa_s : ndarray
        Characteristic convergence rho_s * r_s / Sigma_crit.  Shape broadcastable
        with x (typically (N_halo, N_pts)).
    x : ndarray
        Dimensionless radius theta / theta_s.
    h_x : ndarray or None
        Pre-computed h(x).  Computed if None.
    g_x : ndarray or None
        Pre-computed g(x) = ln(x/2) + h(x).  Computed if None.
 
    Returns
    -------
    kappa : ndarray
    abs_gamma : ndarray
        Both same shape as x.
    """
    if h_x is None:
        h_x = _nfw_radial_h(x)
    if g_x is None:
        g_x = np.log(x / 2.0) + h_x
 
    # Convergence: kappa(x) = 2 kappa_s (1 - h(x)) / (x^2 - 1)
    # Needs careful limit at x=1 where both numerator and denominator -> 0.
    xsq_m1 = x**2 - 1
    safe_denom = np.where(np.abs(xsq_m1) < 1e-8, 1.0, xsq_m1)
    kappa_raw = 2 * kappa_s * (1 - h_x) / safe_denom
    # At x=1: kappa = 2 kappa_s / 3  (L'Hopital)
    kappa = np.where(np.abs(xsq_m1) < 1e-8, 2 * kappa_s / 3.0, kappa_raw)
 
    # Mean convergence inside x: kbar(x) = (4 kappa_s / x^2) * [g(x)]
    #   (Bartelmann 1996 eq. 13, Wright & Brainerd 2000 eq. 13)
    x_safe = np.where(np.abs(x) < 1e-10, 1e-10, x)
    kbar = (4 * kappa_s / x_safe**2) * g_x
 
    # Tangential shear: |gamma_t| = kbar - kappa
    abs_gamma = np.abs(kbar - kappa)
 
    return kappa, abs_gamma

def calculate_deflection_nfw(halos, theta_x, theta_y, z_source, eps=1.0e-6):
    """
    Compute the total NFW reduced deflection field alpha(theta) for a set of
    NFW halos at arbitrary image-plane positions.
 
    The reduced deflection for a single NFW halo at dimensionless radius
    x = |theta - theta_lens| / theta_s is:
 
        |alpha(x)| = 4 kappa_s theta_s g(x) / x
 
    where g(x) = ln(x/2) + h(x), kappa_s = rho_s r_s / Sigma_crit(z_l, z_s),
    and theta_s = r_s / D_l  (angular scale radius in arcsec).
 
    The deflection is directed radially from each halo centre, and the total
    deflection is the vector sum over all halos.
 
    Parameters
    ----------
    halos : NFW_Lens
        Halo object with x, y, mass, concentration, redshift attributes.
    theta_x, theta_y : array-like
        Image-plane positions (arcsec), shape (N_pts,).
    z_source : float
        Source redshift for this set of images (sets Sigma_crit).
    eps : float
        Softening to avoid r=0 singularity (arcsec).
 
    Returns
    -------
    alpha_x, alpha_y : ndarray, shape (N_pts,)
        Deflection components (arcsec).
    """
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)
 
    x_l = np.atleast_1d(halos.x).astype(float)
    y_l = np.atleast_1d(halos.y).astype(float)
    c_l = np.atleast_1d(halos.concentration).astype(float)
 
    # Lens redshift (scalar)
    z_l_arr = np.asarray(halos.redshift)
    z_l = float(z_l_arr.flat[0]) if z_l_arr.ndim > 0 else float(z_l_arr)
 
    # Cosmological distances
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit = critical_surface_density(z_l, z_source)
 
    # Halo structural parameters
    rho_c = cosmo.critical_density(z_l).to(u.kg / u.m**3).value
    delta_c = np.atleast_1d(halos.calc_delta_c())
    rho_s = rho_c * delta_c                              # (N_halo,)
 
    r200_m, r200_arcsec = halos.calc_R200()
    r200_m = np.atleast_1d(r200_m)
    r200_arcsec = np.atleast_1d(r200_arcsec)
    rs_m = r200_m / c_l                                  # (N_halo,) [metres]
    theta_s = r200_arcsec / c_l                          # (N_halo,) [arcsec]
 
    kappa_s = (rho_s * rs_m) / sigma_crit                # (N_halo,)
 
    # Angular separations: (N_halo, N_pts)
    dx = tx[None, :] - x_l[:, None]
    dy = ty[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)
 
    # Dimensionless radius x = theta / theta_s
    x = r / theta_s[:, None]  # (N_halo, N_pts)
 
    # Radial function g(x) = ln(x/2) + h(x)  [stable at all x]
    h_x = _nfw_radial_h(x)
    g_x = _nfw_radial_g(x)

    # Deflection magnitude per halo: 4 kappa_s theta_s g(x) / x
    x_safe = np.where(x < 1e-10, 1e-10, x)
    alpha_mag = 4.0 * kappa_s[:, None] * theta_s[:, None] * g_x / x_safe
    # (N_halo, N_pts)
 
    # Unit radial vector from each halo to evaluation point
    ux = dx / r
    uy = dy / r
 
    # Vector sum over halos
    alpha_x = np.sum(alpha_mag * ux, axis=0)  # (N_pts,)
    alpha_y = np.sum(alpha_mag * uy, axis=0)
 
    return alpha_x, alpha_y

def backproject_source_positions_nfw(halos, theta_x, theta_y, z_source, eps=1.0e-6):
    """
    Back-project image positions to the source plane under an NFW lens model:
 
        beta = theta - alpha(theta)
 
    Parameters
    ----------
    halos : NFW_Lens
        Halo model.
    theta_x, theta_y : array-like
        Image-plane positions (arcsec).
    z_source : float
        Source redshift.
    eps : float
        Softening (arcsec).
 
    Returns
    -------
    beta_x, beta_y : ndarray
        Source-plane positions (arcsec).
    """
    ax, ay = calculate_deflection_nfw(halos, theta_x, theta_y, z_source, eps=eps)
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)
    return tx - ax, ty - ay

def magnification_nfw(halos, theta_x, theta_y, z_source, eps=1.0e-6):
    """
    Absolute magnification |mu| at image-plane positions for a composite
    NFW deflector.
 
    Uses the analytic convergence and shear rather than numerical
    differentiation of the deflection:
 
        det(A) = (1 - kappa)^2 - |gamma|^2
        |mu| = 1 / |det(A)|
 
    Parameters
    ----------
    halos : NFW_Lens
    theta_x, theta_y : array-like
        Image-plane positions (arcsec), shape (N_pts,).
    z_source : float
        Source redshift (sets Sigma_crit for each halo).
    eps : float
        Softening (arcsec).
 
    Returns
    -------
    abs_mu : ndarray, shape (N_pts,)
    det_A  : ndarray, shape (N_pts,)
    """
    tx = np.atleast_1d(theta_x).astype(float)
    ty = np.atleast_1d(theta_y).astype(float)
 
    x_l = np.atleast_1d(halos.x).astype(float)
    y_l = np.atleast_1d(halos.y).astype(float)
    c_l = np.atleast_1d(halos.concentration).astype(float)
 
    z_l_arr = np.asarray(halos.redshift)
    z_l = float(z_l_arr.flat[0]) if z_l_arr.ndim > 0 else float(z_l_arr)
 
    sigma_crit = critical_surface_density(z_l, z_source)
 
    rho_c = cosmo.critical_density(z_l).to(u.kg / u.m**3).value
    delta_c = np.atleast_1d(halos.calc_delta_c())
    rho_s = rho_c * delta_c
 
    r200_m, r200_arcsec = halos.calc_R200()
    r200_m = np.atleast_1d(r200_m)
    r200_arcsec = np.atleast_1d(r200_arcsec)
    rs_m = r200_m / c_l
    theta_s = r200_arcsec / c_l
 
    kappa_s = (rho_s * rs_m) / sigma_crit  # (N_halo,)
 
    # Angular separations
    dx = tx[None, :] - x_l[:, None]
    dy = ty[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)
 
    x = r / theta_s[:, None]  # (N_halo, N_pts)
 
    # h(x) and g(x)  [stable at all x]
    h_x = _nfw_radial_h(x)
    g_x = _nfw_radial_g(x)

    # Per-halo kappa and |gamma| at each point
    kappa_per_halo, gamma_per_halo = _nfw_kappa_and_gamma(
        kappa_s[:, None], x, h_x=h_x, g_x=g_x
    )
    # (N_halo, N_pts) each
 
    # The convergence adds linearly; the shear adds as a tensor.
    # For multiple circularly-symmetric halos centred at different positions,
    # we must sum the shear *as a spin-2 field* then take the magnitude.
    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi
 
    # Tangential shear per halo is negative (γ_t < 0 convention) so
    # γ₁ = -|γ_t| cos(2φ), γ₂ = -|γ_t| sin(2φ)
    # (sign convention: tangential shear from a mass overdensity is negative
    #  in the γ₁ frame aligned with the radial direction)
    gamma1_per_halo = -gamma_per_halo * cos2phi
    gamma2_per_halo = -gamma_per_halo * sin2phi
 
    # Sum over halos
    kappa_total = np.sum(kappa_per_halo, axis=0)   # (N_pts,)
    gamma1_total = np.sum(gamma1_per_halo, axis=0)
    gamma2_total = np.sum(gamma2_per_halo, axis=0)
    gamma_mag_total = np.hypot(gamma1_total, gamma2_total)
 
    # Jacobian determinant
    det_A = (1 - kappa_total)**2 - gamma_mag_total**2
    abs_mu = 1.0 / np.maximum(np.abs(det_A), 1.0e-30)
 
    return abs_mu, det_A

def chi2_strong_source_plane_nfw(halos, strong_systems, eps=1.0e-6,
                                  return_breakdown=False,
                                  use_magnification_correction=True,
                                  delta_n=0.15):
    """
    Source-plane scatter chi^2 for multiply-imaged systems under NFW halos.

    For each system i with images m:
        beta_{i,m} = theta_{i,m} - alpha_NFW(theta_{i,m}; z_{s,i})
        beta_bar_i = weighted mean of beta_{i,m}
        chi2_i     = sum_m |beta_{i,m} - beta_bar_i|^2 / sigma_beta_m^2

    The source-plane uncertainty per image has two components:

        sigma_beta^2 = (sigma_theta / |mu|)^2
                       + alpha^2 * delta_n^2 * ln^2(theta / theta_E)

    The first term is the magnification-corrected astrometric measurement
    error.  The second is the *profile model uncertainty*: for a power-law
    lens with deflection alpha = theta^n * theta_E^(1-n), the deflection
    sensitivity to the profile slope is d(alpha)/dn = alpha * ln(theta/theta_E),
    which vanishes at the Einstein radius but grows logarithmically away
    from it.  For SIS (n=0 exactly), delta_n = 0 and this term disappears.
    For NFW, the effective slope varies with radius and depends on the
    mass-concentration relation, so delta_n > 0 accounts for the
    systematic uncertainty from the imperfect profile model.

    Parameters
    ----------
    halos : NFW_Lens
        Must have x, y, mass, concentration, redshift attributes.
    strong_systems : iterable of StrongLensingSystem
        Each system has theta_x, theta_y, sigma_theta, z_source.
    eps : float
        Softening (arcsec).
    return_breakdown : bool
        If True, also return per-system diagnostics.
    use_magnification_correction : bool
        If True, convert sigma_theta -> sigma_beta via |mu|.
    delta_n : float
        Profile slope uncertainty.  Typical value ~0.15 from the
        intrinsic scatter in the mass-concentration relation.
        Set to 0 to recover the pure measurement-error treatment.

    Returns
    -------
    chi2_sl : float
    breakdown : dict (optional)
    """
    chi2_total = 0.0
    breakdown = {}

    # Mass-weighted lens centroid (for theta_E estimation)
    hx = np.atleast_1d(halos.x).astype(float)
    hy = np.atleast_1d(halos.y).astype(float)
    hm = np.atleast_1d(halos.mass).astype(float)
    mtot = np.sum(hm)
    if mtot > 0:
        xc = np.sum(hx * hm) / mtot
        yc = np.sum(hy * hm) / mtot
    else:
        xc, yc = np.mean(hx), np.mean(hy)

    for sys in strong_systems:
        tx = np.atleast_1d(sys.theta_x).astype(float)
        ty = np.atleast_1d(sys.theta_y).astype(float)
        z_s = float(sys.z_source)

        if tx.shape != ty.shape:
            raise ValueError(
                f"[{getattr(sys, 'system_id', 'unknown')}] theta_x/theta_y shape mismatch."
            )

        # ── Back-project to source plane using NFW deflection ──
        bx, by = backproject_source_positions_nfw(halos, tx, ty, z_s, eps=eps)

        # ── Sigma handling ──
        sig = getattr(sys, "sigma_theta", 0.1)
        if np.isscalar(sig):
            sig_theta = np.full_like(bx, float(sig), dtype=float)
        else:
            sig_theta = np.atleast_1d(sig).astype(float)
            if sig_theta.shape != bx.shape:
                raise ValueError(
                    f"[{getattr(sys, 'system_id', 'unknown')}] sigma_theta shape mismatch."
                )

        # ── Measurement uncertainty (magnification-corrected) ──
        if use_magnification_correction:
            abs_mu, det_A = magnification_nfw(halos, tx, ty, z_s, eps=eps)
            sig_meas = sigma_beta_from_magnification(sig_theta, abs_mu)
        else:
            sig_meas = sig_theta.copy()
            abs_mu = None
            det_A = None

        # ── Profile model uncertainty ──
        # From the power-law decomposition: delta_alpha = alpha * delta_n * ln(theta/theta_E)
        # Propagated to the source plane as an additional sigma_beta term.
        if delta_n > 0:
            # Deflection magnitude at each image: alpha = theta - beta
            alpha_x = tx - bx
            alpha_y = ty - by
            alpha_mag = np.hypot(alpha_x, alpha_y)

            # Image distance from lens centroid
            r_img = np.hypot(tx - xc, ty - yc)
            r_img = np.maximum(r_img, eps)

            # Estimate theta_E as the mean image distance from centroid.
            # For a 2-image system, images bracket theta_E, so
            # theta_E ~ (r_+ + r_-) / 2.  This is exact for SIS and
            # a good approximation for any smooth circularly symmetric lens.
            theta_E_est = np.mean(r_img)
            theta_E_est = max(theta_E_est, eps)

            log_ratio = np.abs(np.log(r_img / theta_E_est))
            sig_profile = alpha_mag * delta_n * log_ratio

            # Total: add in quadrature
            sigx = np.sqrt(sig_meas**2 + sig_profile**2)
            sigy = sigx.copy()
        else:
            sigx = sig_meas
            sigy = sig_meas

        # ── Weighted mean source position ──
        wx = 1.0 / np.maximum(sigx, 1.0e-12) ** 2
        wy = 1.0 / np.maximum(sigy, 1.0e-12) ** 2

        bx_bar = np.sum(wx * bx) / np.sum(wx)
        by_bar = np.sum(wy * by) / np.sum(wy)

        # ── Source-plane scatter chi2 ──
        chi2_i = np.sum(((bx - bx_bar) / sigx) ** 2 + ((by - by_bar) / sigy) ** 2)
        chi2_total += float(chi2_i)

        if return_breakdown:
            sid = getattr(sys, "system_id", "unknown")
            bd = {
                "chi2": float(chi2_i),
                "n_images": int(bx.size),
                "beta_bar": (float(bx_bar), float(by_bar)),
                "beta": np.column_stack([bx, by]),
                "sigma_beta_x": sigx.copy(),
                "sigma_beta_y": sigy.copy(),
            }
            if use_magnification_correction and abs_mu is not None:
                bd["abs_mu"] = abs_mu.copy()
                bd["det_A"] = det_A.copy()
                bd["sigma_theta"] = sig_theta.copy()
            if delta_n > 0:
                bd["sigma_meas"] = sig_meas.copy()
                bd["sigma_profile"] = sig_profile.copy()
                bd["theta_E_est"] = float(theta_E_est)
                bd["alpha_mag"] = alpha_mag.copy()
            breakdown[sid] = bd
 
    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total

def chi2_flux_nfw(halos, strong_systems, eps=1.0e-6,
                  return_breakdown=False):
    """
    Flux-ratio chi^2 for multiply-imaged systems under NFW halos.

    Exactly parallels chi2_flux_sis but uses NFW magnification
    (which requires z_source for each system to set Sigma_crit).

    For each system i with observed flux data:

        R_model_m = |mu_m(theta_m; z_s)| / |mu_ref(theta_ref; z_s)|
        chi2_i = sum_{m != ref} [(R_obs_m - R_model_m) / sigma_R_m]^2

    The magnification |mu| = 1 / |det(A)| depends on kappa and gamma,
    which have different radial profiles than the deflection alpha.
    This makes the flux ratio sensitive to a different combination of
    (position, mass) than the image separation, breaking the
    position-mass degeneracy.

    Systems without flux data (has_flux=False) are silently skipped.

    Parameters
    ----------
    halos : NFW_Lens
        Must have x, y, mass, concentration, redshift attributes.
    strong_systems : iterable of StrongLensingSystem
        Systems with optional flux and sigma_flux attributes.
    eps : float
        Softening for magnification evaluation (arcsec).
    return_breakdown : bool
        If True, also return per-system diagnostics.

    Returns
    -------
    chi2_flux : float
        Total flux-ratio chi^2 across all systems with flux data.
    breakdown : dict (optional)
        Per-system diagnostics keyed by system_id.
    """
    chi2_total = 0.0
    breakdown = {}

    for sls in strong_systems:
        if not getattr(sls, "has_flux", False):
            continue

        tx = np.atleast_1d(sls.theta_x).astype(float)
        ty = np.atleast_1d(sls.theta_y).astype(float)
        z_s = float(sls.z_source)
        F = np.atleast_1d(sls.flux).astype(float)
        sigF = np.atleast_1d(sls.sigma_flux).astype(float)

        # Observed flux ratios relative to brightest image
        ref_idx = int(np.argmax(F))
        F_ref = F[ref_idx]
        sigF_ref = sigF[ref_idx]
        R_obs = F / F_ref
        frac_i = sigF / np.maximum(F, 1e-30)
        frac_ref = sigF_ref / max(F_ref, 1e-30)
        sigma_R = R_obs * np.sqrt(frac_i**2 + frac_ref**2)
        sigma_R[ref_idx] = 0.0

        # Model magnifications at each image position (z_source-dependent)
        abs_mu, det_A = magnification_nfw(halos, tx, ty, z_s, eps=eps)

        # Model flux ratios
        mu_ref = abs_mu[ref_idx]
        R_model = abs_mu / np.maximum(mu_ref, 1.0e-30)

        # chi2: skip the reference image (sigma_R = 0 there)
        mask = np.arange(len(tx)) != ref_idx
        residuals = (R_obs[mask] - R_model[mask]) / np.maximum(sigma_R[mask], 1.0e-30)
        chi2_i = float(np.sum(residuals**2))
        chi2_total += chi2_i

        if return_breakdown:
            sid = getattr(sls, "system_id", "unknown")
            breakdown[sid] = {
                "chi2": chi2_i,
                "n_images": int(tx.size),
                "ref_index": ref_idx,
                "R_obs": R_obs.copy(),
                "R_model": R_model.copy(),
                "sigma_R": sigma_R.copy(),
                "abs_mu": abs_mu.copy(),
                "det_A": det_A.copy(),
            }

    if return_breakdown:
        return chi2_total, breakdown
    return chi2_total

def calculate_lensing_signals_nfw(halos, sources):
    """
    Lensing signals (shear, flexion, g-flexion) for NFW halos with per-source redshifts.
    Assumes all halos are at the same redshift given by halos.redshift (scalar or array of equal values).

    Returns:
        (shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2)
        Each is (N_sources,), or scalars if a single source is passed.
    """
    # --- Normalize halo & source fields (allow scalar or array) ---
    x_l = np.atleast_1d(halos.x)
    y_l = np.atleast_1d(halos.y)
    c_l = np.atleast_1d(halos.concentration)

    x_s = np.atleast_1d(sources.x)
    y_s = np.atleast_1d(sources.y)
    z_s = np.atleast_1d(sources.redshift)

    # Lens redshift: scalar or array with identical values
    z_l_arr = np.asarray(halos.redshift)
    if z_l_arr.ndim == 0:
        z_l = float(z_l_arr)
    else:
        z_l = float(z_l_arr.flat[0])
        if not np.allclose(z_l_arr, z_l, rtol=0, atol=1e-10):
            raise ValueError("All halos must share the same redshift for this function.")

    # Shapes
    N_h, N_s = x_l.size, x_s.size

    # --- Geometry ---
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value  # scalar [m]

    # σ_crit(z_l, z_s) per source, with vectorized fallback
    def _sigma_crit(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(lambda zs: critical_surface_density(zl_scalar, zs), otypes=[float])
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit(z_l, z_s)  # (N_s,)
    behind = (z_s > z_l)
    sigma_crit_s = np.where(behind, sigma_crit_s, np.inf)  # zero-out non-behind via ∞
    sigma_crit_hs = sigma_crit_s[None, :]                  # (1, N_s) -> (N_h, N_s)

    # --- Halo structure ---
    r200_m, r200_arcsec = halos.calc_R200()        # expect (N_h,), but accept scalars
    r200_m = np.atleast_1d(r200_m)
    r200_arcsec = np.atleast_1d(r200_arcsec)

    rs_m  = r200_m / c_l                           # (N_h,)
    rho_c = cosmo.critical_density(z_l).to(u.kg/u.m**3).value  # scalar
    delta_c = np.atleast_1d(halos.calc_delta_c())  # (N_h,)
    rho_s = rho_c * delta_c                        # (N_h,)

    kappa_s = (rho_s * rs_m)[:, None] / sigma_crit_hs  # (N_h, N_s)

    # Flexion scale
    rad_to_arcsec = u.radian.to(u.arcsecond)
    flexion_s = (kappa_s * Dl) / (rs_m[:, None] * rad_to_arcsec)  # (N_h, N_s)

    # --- Angular separations (arcsec) ---
    dx = x_s[None, :] - x_l[:, None]   # (N_h, N_s)
    dy = y_s[None, :] - y_l[:, None]
    r  = np.hypot(dx, dy)
    r  = np.where(r == 0.0, 1.0e-2, r)

    theta_s = (r200_arcsec / c_l)[:, None]  # (N_h, 1)
    x = np.abs(r / theta_s)                 # (N_h, N_s)

    # --- Radial terms ---
    def radial_term_1(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        t1 = np.sqrt(np.clip((1 - x[m1]) / (1 + x[m1]), 0.0, None))
        sol[m1] = 1 - (2 / np.sqrt(1 - x[m1]**2)) * np.arctanh(t1)
        t2 = np.sqrt(np.clip((x[m2] - 1) / (1 + x[m2]), 0.0, None))
        sol[m2] = 1 - (2 / np.sqrt(x[m2]**2 - 1)) * np.arctan(t2)
        return sol

    def radial_term_2(x):
        # Called g(x), used in shear calculation
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = x > 1
        m3 = x == 1
        k = (1 - x) / (1 + x)
        t1 = np.sqrt(np.clip(k[m1], 0.0, None))
        sol[m1] = (
            8 * np.arctanh(t1) / (x[m1]**2 * np.sqrt(1 - x[m1]**2))
            + 4 * np.log(x[m1] / 2) / x[m1]**2
            - 2 / (x[m1]**2 - 1)
            + 4 * np.arctanh(t1) / ((x[m1]**2 - 1) * np.sqrt(1 - x[m1]**2))
        )
        t2 = np.sqrt(np.clip((x[m2] - 1) / (x[m2] + 1), 0.0, None))
        sol[m2] = (
            8 * np.arctan(t2) / (x[m2]**2 * np.sqrt(x[m2]**2 - 1))
            + 4 * np.log(x[m2] / 2) / x[m2]**2
            - 2 / (x[m2]**2 - 1)
            + 4 * np.arctan(t2) / ((x[m2]**2 - 1)**(3/2))
        )
        sol[m3] = 10.0/3.0 + 4.0*np.log(0.5)
        return sol

    def radial_term_3(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        t1 = np.sqrt(np.clip((1 - x[m1]) / (1 + x[m1]), 0.0, None))
        sol[m1] = (1 / (1 - x[m1]**2)) * (1 / x[m1] - (2 * x[m1]) / np.sqrt(1 - x[m1]**2) * np.arctanh(t1))
        t2 = np.sqrt(np.clip((x[m2] - 1) / (1 + x[m2]), 0.0, None))
        sol[m2] = (1 / (x[m2]**2 - 1)) * ((2 * x[m2]) / np.sqrt(x[m2]**2 - 1) * np.arctan(t2) - 1 / x[m2])
        return sol

    def radial_term_4(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        leading = 8 / x**3 - 20 / x + 15 * x
        k = (1 - x) / (1 + x)
        t1 = np.sqrt(np.clip(k[m1], 0.0, None))
        sol[m1] = (2 / np.sqrt(1 - x[m1]**2)) * np.arctanh(t1)
        t2 = np.sqrt(np.clip(-k[m2], 0.0, None))
        sol[m2] = (2 / np.sqrt(x[m2]**2 - 1)) * np.arctan(t2)
        sol *= leading
        return sol

    def radial_term_5(x):
        # Use this to compute necessary term for kappa
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = x == 1
        m3 = x > 1
        sol[m1] = np.arctanh(np.sqrt(1-x[m1]**2)) / np.sqrt(1-x[m1]**2)
        sol[m2] = 1 
        sol[m3] = np.arctan(np.sqrt(x[m3]**2 - 1)) / np.sqrt(x[m3]**2 - 1)
        return sol
        

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)
    term_5 = radial_term_5(x)

    # --- Angular factors ---
    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    # --- Lensing magnitudes (per halo, per source) ---
    kappa_mag = 2 * kappa_s* (1 - term_5) / (x**2 - 1)
    shear_mag = -kappa_s * term_2

    def calc_flexion(flexion_s, x, term_1, term_3):
        I1 = -2 * flexion_s
        I2 = 2 * x * term_1 / (x**2 - 1)**2
        I3 = term_3 / (x**2 - 1)
        return I1 * (I2 - I3)

    def calc_g_flexion(flexion_s, x, term_4):
        I1 = 2 * flexion_s
        I2 = (8 / x**3) * np.log(x / 2)
        I3 = (3 / x) * (1 - 2 * x**2) + term_4
        I4 = (x**2 - 1)**2
        return I1 * (I2 + (I3 / I4))

    flexion_mag   = calc_flexion (flexion_s, x, term_1, term_3)
    g_flexion_mag = calc_g_flexion(flexion_s, x, term_4)

    # Zero contributions where source is not behind lens
    shear_mag     = np.where(behind[None, :], shear_mag, 0.0)
    flexion_mag   = np.where(behind[None, :], flexion_mag, 0.0)
    g_flexion_mag = np.where(behind[None, :], g_flexion_mag, 0.0)

    # --- Sum over halos => per-source outputs ---
    kappa       = np.sum(kappa_mag              , axis=0) 
    shear_1     = np.sum(shear_mag     * cos2phi, axis=0)
    shear_2     = np.sum(shear_mag     * sin2phi, axis=0)
    flexion_1   = np.sum(flexion_mag   * cos_phi, axis=0)
    flexion_2   = np.sum(flexion_mag   * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    # If a single source was passed, return scalars
    if (np.ndim(sources.x) == 0) and (np.ndim(sources.y) == 0) and (np.ndim(sources.redshift) == 0):
        return (shear_1.item(), shear_2.item(),
                flexion_1.item(), flexion_2.item(),
                g_flexion_1.item(), g_flexion_2.item())
    return kappa, shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2

def power_law_projected_mass(
    halos,
    r_p,
    return_2d=False,
    nx=100,
    ny=100,
    x_range=(-300, 300),
    y_range=(-300, 300),
    x_center=0.0,
    y_center=0.0,
    z_source=0.8,
):
    """
    Compute the projected mass (or convergence map) for a single
    POWER_LAW halo.  Mirrors `nfw_projected_mass`'s API so the cluster-
    level mass-comparison workflow in `compare_mass_estimates` can
    dispatch on lens type without restructuring.

    The power-law convergence at angular radius theta is
        kappa(theta) = beta(z_s) * kappa_star * (theta / theta_star)^(-n)
    where beta(z_s) = D_ls / D_s converts from the at-infinity convention
    (kappa_star is defined for a source at z_s -> infinity, matching the
    Wright & Brainerd 2000 NFW convention used elsewhere in arch).

    Internally everything is in kpc to match nfw_projected_mass; we
    convert the halo's angular pivot theta_star (arcsec) to a physical
    pivot radius r_pivot (kpc) using the lens-redshift kpc/arcsec
    conversion, then evaluate

        kappa(R) = beta * kappa_star * (R / r_pivot)^(-n)

    For `return_2d=False`, the closed-form integrated 2D mass (Phase 0)
    is used directly:
        M_2D(<r) = pi * Sigma_cr * 2 * beta * kappa_star * r_pivot^n
                   * r^(2-n) / (2 - n)

    Parameters
    ----------
    halos : PowerLawHalo
        Single-halo PowerLawHalo (the per-halo loop in
        compare_mass_estimates constructs one of these per iteration).
    r_p : float or array_like
        Projected radius in kpc (only used when return_2d=False).
    return_2d : bool
        If True, return (kappa_grid, area_per_pixel).  If False, return
        the integrated projected mass M(<r_p) in solar masses.
    nx, ny : int
    x_range, y_range : tuple of float
        Grid extent in kpc.
    x_center, y_center : float
        Halo center in kpc.
    z_source : float
        Source redshift used for beta(z_s).

    Returns
    -------
    If return_2d=False:
        M_proj : float or ndarray
            Integrated projected mass within r_p, in M_sun.
    If return_2d=True:
        kappa : 2D ndarray
            Convergence map (dimensionless) on the (nx, ny) grid.
        area_per_pixel : float
            Pixel area in kpc^2 (so kappa * sigma_c_in_M_per_kpc2 *
            area_per_pixel gives M_sun per pixel — same convention as
            nfw_projected_mass).
    """
    # --- Cosmological quantities ---
    z_l = float(halos.redshift)
    if z_source <= z_l:
        # Foreground source — no lensing
        beta = 0.0
    else:
        D_l_m = cosmo.angular_diameter_distance(z_l).to(u.m).value
        D_s_m = cosmo.angular_diameter_distance(z_source).to(u.m).value
        D_ls_m = cosmo.angular_diameter_distance_z1z2(
            z_l, z_source).to(u.m).value
        # beta = D_ls / D_s (equivalent to sigma_crit_inf / sigma_crit_zs)
        beta = D_ls_m / D_s_m

    # kpc per arcsec at the lens redshift
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_l).to(
        u.kpc / u.arcsec).value

    # Halo strength and shape parameters
    n = float(halos.slope)
    kappa_star = float(halos.kappa_star)
    theta_star_arcsec = float(halos.theta_star)
    r_pivot_kpc = theta_star_arcsec * kpc_per_arcsec   # physical pivot radius

    # --- 1D integrated mass branch ---
    if not return_2d:
        r_p_arr = np.atleast_1d(r_p).astype(float)
        # Critical surface density in M_sun / kpc^2
        sigma_c = critical_surface_density(z_l, z_source)
        sigma_c = (sigma_c * u.M_sun / u.kpc ** 2).value

        denom = 2.0 - n
        if abs(denom) < 1e-6:
            denom = 1e-6
        # M(<r) = integral of Sigma over disk of radius r
        # Sigma(R) = sigma_c * kappa(R) = sigma_c * beta * kappa_star * (R/r_pivot)^(-n)
        # M(<r) = 2*pi * integral_0^r Sigma(R) R dR
        #        = 2*pi * sigma_c * beta * kappa_star * r_pivot^n * r^(2-n) / (2-n)
        M_2D = (2.0 * np.pi * sigma_c * beta * kappa_star
                * r_pivot_kpc ** n * r_p_arr ** denom / denom)
        return M_2D if M_2D.size > 1 else float(M_2D[0])

    # --- 2D convergence-grid branch ---
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    area_per_pixel = dx * dy

    XX, YY = np.meshgrid(x_vals, y_vals)
    R = np.sqrt((XX - x_center) ** 2 + (YY - y_center) ** 2)
    # Avoid the central singularity
    R = np.where(R < 0.5, 0.5, R)

    kappa = beta * kappa_star * (R / r_pivot_kpc) ** (-n)
    return kappa, area_per_pixel

def power_law_projected_mass(
    halos,
    r_p,
    return_2d=False,
    nx=100,
    ny=100,
    x_range=(-300, 300),
    y_range=(-300, 300),
    x_center=0.0,
    y_center=0.0,
    z_source=0.8,
):
    """
    Compute the projected mass (or convergence map) for a single
    POWER_LAW halo.  Mirrors `nfw_projected_mass`'s API so the cluster-
    level mass-comparison workflow in `compare_mass_estimates` can
    dispatch on lens type without restructuring.

    The power-law convergence at angular radius theta is
        kappa(theta) = beta(z_s) * kappa_star * (theta / theta_star)^(-n)
    where beta(z_s) = D_ls / D_s converts from the at-infinity convention
    (kappa_star is defined for a source at z_s -> infinity, matching the
    Wright & Brainerd 2000 NFW convention used elsewhere in arch).

    Internally everything is in kpc to match nfw_projected_mass; we
    convert the halo's angular pivot theta_star (arcsec) to a physical
    pivot radius r_pivot (kpc) using the lens-redshift kpc/arcsec
    conversion, then evaluate

        kappa(R) = beta * kappa_star * (R / r_pivot)^(-n)

    For `return_2d=False`, the closed-form integrated 2D mass (Phase 0)
    is used directly:
        M_2D(<r) = pi * Sigma_cr * 2 * beta * kappa_star * r_pivot^n
                   * r^(2-n) / (2 - n)

    Parameters
    ----------
    halos : PowerLawHalo
        Single-halo PowerLawHalo (the per-halo loop in
        compare_mass_estimates constructs one of these per iteration).
    r_p : float or array_like
        Projected radius in kpc (only used when return_2d=False).
    return_2d : bool
        If True, return (kappa_grid, area_per_pixel).  If False, return
        the integrated projected mass M(<r_p) in solar masses.
    nx, ny : int
    x_range, y_range : tuple of float
        Grid extent in kpc.
    x_center, y_center : float
        Halo center in kpc.
    z_source : float
        Source redshift used for beta(z_s).

    Returns
    -------
    If return_2d=False:
        M_proj : float or ndarray
            Integrated projected mass within r_p, in M_sun.
    If return_2d=True:
        kappa : 2D ndarray
            Convergence map (dimensionless) on the (nx, ny) grid.
        area_per_pixel : float
            Pixel area in kpc^2 (so kappa * sigma_c_in_M_per_kpc2 *
            area_per_pixel gives M_sun per pixel — same convention as
            nfw_projected_mass).
    """
    # --- Cosmological quantities ---
    z_l = float(halos.redshift)
    if z_source <= z_l:
        # Foreground source — no lensing
        beta = 0.0
    else:
        D_l_m = cosmo.angular_diameter_distance(z_l).to(u.m).value
        D_s_m = cosmo.angular_diameter_distance(z_source).to(u.m).value
        D_ls_m = cosmo.angular_diameter_distance_z1z2(
            z_l, z_source).to(u.m).value
        # beta = D_ls / D_s (equivalent to sigma_crit_inf / sigma_crit_zs)
        beta = D_ls_m / D_s_m

    # kpc per arcsec at the lens redshift
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_l).to(
        u.kpc / u.arcsec).value

    # Halo strength and shape parameters
    n = float(halos.slope)
    kappa_star = float(halos.kappa_star)
    theta_star_arcsec = float(halos.theta_star)
    r_pivot_kpc = theta_star_arcsec * kpc_per_arcsec   # physical pivot radius

    # --- 1D integrated mass branch ---
    if not return_2d:
        r_p_arr = np.atleast_1d(r_p).astype(float)
        # Critical surface density in M_sun / kpc^2
        sigma_c = critical_surface_density(z_l, z_source)
        sigma_c = (sigma_c * u.M_sun / u.kpc ** 2).value

        denom = 2.0 - n
        if abs(denom) < 1e-6:
            denom = 1e-6
        # M(<r) = integral of Sigma over disk of radius r
        # Sigma(R) = sigma_c * kappa(R) = sigma_c * beta * kappa_star * (R/r_pivot)^(-n)
        # M(<r) = 2*pi * integral_0^r Sigma(R) R dR
        #        = 2*pi * sigma_c * beta * kappa_star * r_pivot^n * r^(2-n) / (2-n)
        M_2D = (2.0 * np.pi * sigma_c * beta * kappa_star
                * r_pivot_kpc ** n * r_p_arr ** denom / denom)
        return M_2D if M_2D.size > 1 else float(M_2D[0])

    # --- 2D convergence-grid branch ---
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    area_per_pixel = dx * dy

    XX, YY = np.meshgrid(x_vals, y_vals)
    R = np.sqrt((XX - x_center) ** 2 + (YY - y_center) ** 2)
    # Avoid the central singularity
    R = np.where(R < 0.5, 0.5, R)

    kappa = beta * kappa_star * (R / r_pivot_kpc) ** (-n)
    return kappa, area_per_pixel


def compare_mass_estimates(halos, plot_name, plot_title,
                           cluster_name='Abell_2744', lens_type='NFW'):
    """
    Compares the cluster-level mass reconstruction to literature
    estimates for Abell 2744 or El Gordo.

    Supports both NFW and POWER_LAW lens types via the `lens_type`
    parameter.  The workflow is identical for both:

        1. For each halo, compute its convergence (kappa) contribution
           on a 2D kpc grid centered on the cluster's mass-weighted
           centroid.  This uses `nfw_projected_mass(return_2d=True)`
           or `power_law_projected_mass(return_2d=True)` accordingly.
        2. Sum kappa across halos and apply a cluster-specific mass-
           sheet transformation (k=2 by literature convention).
        3. Convert kappa -> Sigma -> M_2D per pixel, then compute the
           cumulative aperture mass M(<r) by summing pixels within r.
        4. Overlay literature mass estimates (mass, radius) on the
           cumulative-mass plot for direct comparison.

    Parameters
    ----------
    halos : NFW_Lens or PowerLawHalo
        Lens collection.
    plot_name : str
        Path to save the output figure.
    plot_title : str
        Figure title.
    cluster_name : str
        'ABELL_2744' or 'EL_GORDO'.
    lens_type : str
        'NFW' (default, backward-compatible) or 'POWER_LAW'.

    Notes
    -----
    For POWER_LAW the mass-weighted centroid uses kappa_star as the
    weight (analogous to NFW's mass-weighted centroid).  This is a
    proxy for true mass; halos with larger kappa_star contribute the
    majority of the projected mass.
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
        raise ValueError(
            "Invalid cluster name. Choose from 'ABELL_2744' or 'EL_GORDO'.")

    if lens_type not in ('NFW', 'POWER_LAW'):
        raise ValueError(
            "Invalid lens_type. Choose from 'NFW' or 'POWER_LAW'.")

    # Radii in kpc — span the literature radii with margin
    r_min = min([r for _, r in mass_estimates.values()])
    r_max = max([r for _, r in mass_estimates.values()])
    r = np.linspace(r_min * 0.75, 1.25 * r_max, 100)

    # Mass-weighted centroid; for POWER_LAW use kappa_star as the proxy
    if lens_type == 'NFW':
        weight = halos.mass
    else:  # POWER_LAW
        weight = halos.kappa_star

    weight_total = float(np.sum(weight))
    if weight_total <= 0:
        raise ValueError(
            f"Cannot compute mass-weighted centroid: total weight = {weight_total}")
    centroid = np.array([
        np.sum(halos.x * weight) / weight_total,
        np.sum(halos.y * weight) / weight_total,
    ])
    halos.x -= centroid[0] + 0.5
    halos.y -= centroid[1] + 0.5

    if lens_type == 'POWER_LAW':
        # For POWER_LAW we use the closed-form M_2D(<r) directly,
        # bypassing the kappa-grid path entirely.  The Phase 0 result
        #   M_2D(<r) = 2 * pi * Sigma_cr * beta * kappa_star
        #              * r_pivot^n * r^(2-n) / (2 - n)
        # is exact (no integration), positive by construction (no
        # mass-sheet transform needed), and additive across halos at
        # different positions provided we evaluate cumulative mass
        # within r of the cluster centroid (mass-weighted).
        #
        # The mass-sheet transformation used by the kappa-grid path
        # (kappa_total = 2*kappa - 1 for k=2) drives kappa negative
        # wherever the input kappa is below 0.5, producing negative
        # masses for POWER_LAW where the kappa map is mostly subdued
        # outside halo cores.  The closed-form route avoids this
        # issue entirely.
        z_l = z_cluster
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_l).to(
            u.kpc / u.arcsec).value
        # critical_surface_density returns kg/m^2 as a plain float;
        # convert to M_sun/kpc^2 explicitly (the pattern used at
        # line 270 of utils.py).  The version that just attaches
        # units (.value after multiplying by u.M_sun/u.kpc**2)
        # is a unit-attachment bug that masks the kg/m^2 number
        # as M_sun/kpc^2, leaving the result ~5e8 too small.
        sigma_c_kg_per_m2 = critical_surface_density(z_l, z_source)
        sigma_c = (sigma_c_kg_per_m2 * u.kg / u.m ** 2).to(
            u.M_sun / u.kpc ** 2).value

        # Compute M(<r) cumulative as the SUM over all halos of each
        # halo's individual M_2D(<r_to_halo).  For each radius r in our
        # query grid (centered on the mass-weighted cluster centroid),
        # we evaluate per-halo M(<r_eff_i) where r_eff_i is the radius
        # measured from the halo center, not from the cluster centroid.
        #
        # For halos at the cluster center, r_eff = r and the per-halo
        # M_2D matches the global cumulative.  For off-center halos,
        # r_eff < r when the aperture encloses the halo, and r_eff = 0
        # before that.  This is the correct generalization of the
        # NFW-grid workflow to multi-halo POWER_LAW.
        #
        # Approximation: for cluster-scale halos clustered near the
        # centroid (typical case), all halos contribute fully once the
        # aperture is wider than the typical halo offset (~tens of
        # arcsec).  At smaller r the per-halo offset matters.

        halo_offsets = np.hypot(halos.x, halos.y)  # arcsec from centroid
        halo_offsets_kpc = halo_offsets * kpc_per_arcsec

        n_arr = np.atleast_1d(halos.slope).astype(float)
        kappa_star_arr = np.atleast_1d(halos.kappa_star).astype(float)
        theta_star_arcsec = float(halos.theta_star)
        r_pivot_kpc = theta_star_arcsec * kpc_per_arcsec

        # Beta(z_s) at this source redshift
        if z_source <= z_l:
            beta = 0.0
        else:
            D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
            D_s = cosmo.angular_diameter_distance(z_source).to(u.m).value
            D_ls = cosmo.angular_diameter_distance_z1z2(
                z_l, z_source).to(u.m).value
            beta = D_ls / D_s

        mass_enclosed = np.zeros_like(r)
        for i, radius in enumerate(r):
            total = 0.0
            for k in range(len(halos.x)):
                # Effective radius from this halo center: clip to (0, radius)
                r_eff = max(0.0, radius - halo_offsets_kpc[k])
                if r_eff <= 0:
                    continue
                denom = 2.0 - n_arr[k]
                if abs(denom) < 1e-6:
                    denom = 1e-6
                M_halo = (2.0 * np.pi * sigma_c * beta * kappa_star_arr[k]
                          * r_pivot_kpc ** n_arr[k]
                          * r_eff ** denom / denom)
                total += float(M_halo)
            mass_enclosed[i] = total

    else:
        # NFW path — use the existing kappa-grid + mass-sheet workflow
        # 2D grid setup (in kpc)
        x_range, y_range = (-r[-1], r[-1]), (-r[-1], r[-1])
        nx, ny = int(x_range[1] - x_range[0]), int(y_range[1] - y_range[0])
        x_vals = np.linspace(x_range[0], x_range[1], nx)
        y_vals = np.linspace(y_range[0], y_range[1], ny)
        kappa_total = np.zeros((ny, nx), dtype=float)

        # Convert halo positions from arcsec to kpc using lens-redshift kpc/arcsec
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_cluster).to(
            u.kpc / u.arcsec).value
        halo_x_kpc = halos.x * kpc_per_arcsec
        halo_y_kpc = halos.y * kpc_per_arcsec

        # Sum the kappa grids from all halos
        area_per_pixel = None
        for i in range(len(halos.x)):
            halo = halo_obj.NFW_Lens(
                halos.x[i], halos.y[i], [0],
                halos.concentration[i], halos.mass[i],
                z_cluster, halos.chi2[i],
            )
            kappa, area_per_pixel = nfw_projected_mass(
                halo, r_p=0,
                return_2d=True, nx=nx, ny=ny,
                x_range=x_range, y_range=y_range,
                x_center=halo_x_kpc[i],
                y_center=halo_y_kpc[i],
                z_source=z_source,
            )

            if np.any(np.isinf(kappa)) or np.any(np.isnan(kappa)):
                print(f"Warning: NaN/Inf values in kappa for halo {i} at "
                      f"({halos.x[i]:.2f}, {halos.y[i]:.2f}). Skipping.")
                continue
            kappa_total += kappa

        # Mass-sheet degeneracy: k value taken from literature convention
        if cluster_name == 'ABELL_2744':
            kappa_total = mass_sheet_transformation(kappa_total, k=2)
        elif cluster_name == 'EL_GORDO':
            kappa_total = mass_sheet_transformation(kappa_total, k=2)

        # Convert kappa back to a 2D mass distribution
        sigma_c = critical_surface_density(z_cluster, z_source)
        sigma_c = sigma_c * u.M_sun / u.kpc ** 2
        sigma_c = sigma_c.value
        M_2D_total = kappa_total * sigma_c * area_per_pixel

        # Coordinates of each pixel in the grid
        XX, YY = np.meshgrid(x_vals, y_vals)
        RR = np.sqrt(XX ** 2 + YY ** 2)

        # Compute the enclosed mass at each radius in r
        mass_enclosed = np.zeros_like(r)
        for i, radius in enumerate(r):
            mask = (RR <= radius)
            mass_enclosed[i] = np.sum(M_2D_total[mask])

    # Tell me how far off we are from the literature estimates
    print(f"\n  Mass comparison ({lens_type}):")
    for label, (mass_lit, r_lit) in mass_estimates.items():
        mass_recon = np.interp(r_lit, r, mass_enclosed)
        mass_lit_val = f'{mass_lit:.2e}'
        mass_recon_val = f'{mass_recon:.2e}'
        pct = 100 * (mass_recon - mass_lit) / mass_lit
        print(f"    {label}: lit={mass_lit_val}, recon={mass_recon_val}, "
              f"err={pct:+.1f}%")

    # Plot results
    fig, ax = plt.subplots()
    fig.suptitle(plot_title)
    ax.plot(r, mass_enclosed, label=f'Reconstruction ({lens_type})')

    markers = ['o', 's', 'D', '^']
    for i, (label, (mass_lit, r_lit)) in enumerate(mass_estimates.items()):
        ax.scatter(r_lit, mass_lit, label=label,
                   marker=markers[i % len(markers)])

    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'Mass ($M_\odot$)')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(plot_name)
    plt.close(fig)

    # Put the halos back where they were
    halos.x += centroid[0] + 0.5
    halos.y += centroid[1] + 0.5

    return r, mass_enclosed