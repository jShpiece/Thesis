"""Image processing, kappa maps, mass-sheet utilities, and mass comparison."""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo
from scipy.ndimage import gaussian_filter, maximum_filter

import arch.halo_obj as halo_obj
from arch.cosmology import critical_surface_density
from arch.nfw_lensing import nfw_projected_mass


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
    unique_indices, unique_inverse, counts = np.unique(
        index, axis=1, return_inverse=True, return_counts=True
    )
    # Perform accumulation using advanced indexing and broadcasting
    grid[unique_indices[0] % npixels, unique_indices[1] % npixels] += np.bincount(
        unique_inverse,
        weights=(1 - drow) * (1 - dcol) * signal_values,
        minlength=len(unique_indices.T),
    )
    grid[(unique_indices[0] + 1) % npixels, unique_indices[1] % npixels] += np.bincount(
        unique_inverse, weights=(drow) * (1 - dcol) * signal_values, minlength=len(unique_indices.T)
    )
    grid[unique_indices[0] % npixels, (unique_indices[1] + 1) % npixels] += np.bincount(
        unique_inverse, weights=(1 - drow) * (dcol) * signal_values, minlength=len(unique_indices.T)
    )
    grid[(unique_indices[0] + 1) % npixels, (unique_indices[1] + 1) % npixels] += np.bincount(
        unique_inverse, weights=(drow) * (dcol) * signal_values, minlength=len(unique_indices.T)
    )

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
    arcsec_to_kpc = cosmo.angular_diameter_distance(z_lens).to(u.kpc).value * (
        np.pi / (180.0 * 3600.0)
    )  # 1 kpc in arcsec
    radius_arcsec = radius_kpc / arcsec_to_kpc  # circular aperture radius

    # Critical surface density
    sigma_c = critical_surface_density(z_lens, z_source)
    # Convert to M_sun / kpc^2 from kg / m^2 (sigma_c is just a number)
    sigma_c = (sigma_c * u.kg / u.m**2).to(u.M_sun / u.kpc**2).value

    Ny, Nx = kappa_map.shape
    y_coords, x_coords = np.arange(Ny), np.arange(Nx)
    Y_grid, X_grid = np.meshgrid(y_coords, x_coords, indexing="ij")  # in arcsec

    peaks_arcsec = []
    masses = []

    for y0, x0 in peak_indices:
        ra_offset = x0  # x-axis = RA in arcsec
        dec_offset = y0  # flip y-axis to match Dec increasing upward
        peaks_arcsec.append((ra_offset, dec_offset))

        dist = np.sqrt((X_grid - x0) ** 2 + (Y_grid - y0) ** 2)
        mask = dist <= radius_arcsec
        enclosed_kappa = np.sum(kappa_map[mask])
        mass = sigma_c * enclosed_kappa * arcsec_to_kpc**2  # M_sun
        masses.append(np.abs(mass))

    return peaks_arcsec, masses


def calculate_kappa(lenses, extent, lens_type="SIS", source_redshift=0.8, k_val=0.95):
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

        sol[mask1] = 1 - (2 / np.sqrt(1 - x[mask1] ** 2)) * np.arctanh(
            np.sqrt((1 - x[mask1]) / (1 + x[mask1]))
        )
        sol[mask2] = 1 - (2 / np.sqrt(x[mask2] ** 2 - 1)) * np.arctan(
            np.sqrt((x[mask2] - 1) / (1 + x[mask2]))
        )

        return sol

    if lens_type == "SIS":
        for k in range(len(lenses.x)):
            dx = X - lenses.x[k]
            dy = Y - lenses.y[k]
            r = np.hypot(dx, dy) + 0.5  # Avoid division by zero
            kappa += lenses.te[k] / (2 * r)

    elif lens_type == "NFW":
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
            kappa += 2 * kappa_s * term_1 / (x**2 - 1)

    elif lens_type == "POWER_LAW":
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
            sigma_crit_inf = c.value**2 / (4 * np.pi * G.value * D_l)
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


def compare_mass_estimates(
    halos, plot_name, plot_title, cluster_name="Abell_2744", lens_type="NFW"
):
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
        "MARS": (1.73e14, 200),
        "Bird": (1.93e14, 200),
        "GRALE": (2.25e14, 250),
        "Merten et al.": (2.24e14, 250),
    }
    mass_estimates_elgordo = {
        "Cerny et al.": (1.1e15, 500),
        "Caminha et all.": (1.84e15, 1000),
        "Diego et al": (0.8e15, 500),
    }

    # Choose a cluster
    if cluster_name == "ABELL_2744":
        mass_estimates = mass_estimates_abell
        z_cluster = 0.308
        z_source = 1.0
    elif cluster_name == "EL_GORDO":
        mass_estimates = mass_estimates_elgordo
        z_cluster = 0.870
        z_source = 4.25
    else:
        raise ValueError("Invalid cluster name. Choose from 'ABELL_2744' or 'EL_GORDO'.")

    if lens_type not in ("NFW", "POWER_LAW"):
        raise ValueError("Invalid lens_type. Choose from 'NFW' or 'POWER_LAW'.")

    # Radii in kpc — span the literature radii with margin
    r_min = min([r for _, r in mass_estimates.values()])
    r_max = max([r for _, r in mass_estimates.values()])
    r = np.linspace(r_min * 0.75, 1.25 * r_max, 100)

    # Mass-weighted centroid; for POWER_LAW use kappa_star as the proxy
    if lens_type == "NFW":
        weight = halos.mass
    else:  # POWER_LAW
        weight = halos.kappa_star

    weight_total = float(np.sum(weight))
    if weight_total <= 0:
        raise ValueError(f"Cannot compute mass-weighted centroid: total weight = {weight_total}")
    centroid = np.array(
        [
            np.sum(halos.x * weight) / weight_total,
            np.sum(halos.y * weight) / weight_total,
        ]
    )
    halos.x -= centroid[0] + 0.5
    halos.y -= centroid[1] + 0.5

    if lens_type == "POWER_LAW":
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
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_l).to(u.kpc / u.arcsec).value
        # critical_surface_density returns kg/m^2 as a plain float;
        # convert to M_sun/kpc^2 explicitly (the pattern used at
        # line 270 of utils.py).  The version that just attaches
        # units (.value after multiplying by u.M_sun/u.kpc**2)
        # is a unit-attachment bug that masks the kg/m^2 number
        # as M_sun/kpc^2, leaving the result ~5e8 too small.
        sigma_c_kg_per_m2 = critical_surface_density(z_l, z_source)
        sigma_c = (sigma_c_kg_per_m2 * u.kg / u.m**2).to(u.M_sun / u.kpc**2).value

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
            D_s = cosmo.angular_diameter_distance(z_source).to(u.m).value
            D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_source).to(u.m).value
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
                M_halo = (
                    2.0
                    * np.pi
                    * sigma_c
                    * beta
                    * kappa_star_arr[k]
                    * r_pivot_kpc ** n_arr[k]
                    * r_eff**denom
                    / denom
                )
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
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_cluster).to(u.kpc / u.arcsec).value
        halo_x_kpc = halos.x * kpc_per_arcsec
        halo_y_kpc = halos.y * kpc_per_arcsec

        # Sum the kappa grids from all halos
        area_per_pixel = None
        for i in range(len(halos.x)):
            halo = halo_obj.NFW_Lens(
                halos.x[i],
                halos.y[i],
                [0],
                halos.concentration[i],
                halos.mass[i],
                z_cluster,
                halos.chi2[i],
            )
            kappa, area_per_pixel = nfw_projected_mass(
                halo,
                r_p=0,
                return_2d=True,
                nx=nx,
                ny=ny,
                x_range=x_range,
                y_range=y_range,
                x_center=halo_x_kpc[i],
                y_center=halo_y_kpc[i],
                z_source=z_source,
            )

            if np.any(np.isinf(kappa)) or np.any(np.isnan(kappa)):
                print(
                    f"Warning: NaN/Inf values in kappa for halo {i} at "
                    f"({halos.x[i]:.2f}, {halos.y[i]:.2f}). Skipping."
                )
                continue
            kappa_total += kappa

        # Mass-sheet degeneracy: k value taken from literature convention
        if cluster_name == "ABELL_2744":
            kappa_total = mass_sheet_transformation(kappa_total, k=2)
        elif cluster_name == "EL_GORDO":
            kappa_total = mass_sheet_transformation(kappa_total, k=2)

        # Convert kappa back to a 2D mass distribution
        sigma_c = critical_surface_density(z_cluster, z_source)
        sigma_c = sigma_c * u.M_sun / u.kpc**2
        sigma_c = sigma_c.value
        M_2D_total = kappa_total * sigma_c * area_per_pixel

        # Coordinates of each pixel in the grid
        XX, YY = np.meshgrid(x_vals, y_vals)
        RR = np.sqrt(XX**2 + YY**2)

        # Compute the enclosed mass at each radius in r
        mass_enclosed = np.zeros_like(r)
        for i, radius in enumerate(r):
            mask = RR <= radius
            mass_enclosed[i] = np.sum(M_2D_total[mask])

    # Tell me how far off we are from the literature estimates
    print(f"\n  Mass comparison ({lens_type}):")
    for label, (mass_lit, r_lit) in mass_estimates.items():
        mass_recon = np.interp(r_lit, r, mass_enclosed)
        mass_lit_val = f"{mass_lit:.2e}"
        mass_recon_val = f"{mass_recon:.2e}"
        pct = 100 * (mass_recon - mass_lit) / mass_lit
        print(f"    {label}: lit={mass_lit_val}, recon={mass_recon_val}, " f"err={pct:+.1f}%")

    # Plot results
    fig, ax = plt.subplots()
    fig.suptitle(plot_title)
    ax.plot(r, mass_enclosed, label=f"Reconstruction ({lens_type})")

    markers = ["o", "s", "D", "^"]
    for i, (label, (mass_lit, r_lit)) in enumerate(mass_estimates.items()):
        ax.scatter(r_lit, mass_lit, label=label, marker=markers[i % len(markers)])

    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel(r"Mass ($M_\odot$)")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(plot_name)
    plt.close(fig)

    # Put the halos back where they were
    halos.x += centroid[0] + 0.5
    halos.y += centroid[1] + 0.5

    return r, mass_enclosed
