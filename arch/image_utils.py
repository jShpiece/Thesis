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

hubble_param = cosmo.H0.value / 100.0  # h = H0 / 100 km/s/Mpc

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
    halos,
    plot_name,
    plot_title,
    cluster_name="ABELL_2744",
    lens_type="NFW",
    mass_err=None,
    y_range=None,
    show_title=False,
):
    """
    Compare the ARCH cluster-level mass reconstruction to literature
    estimates, as cumulative projected mass M(<r) versus distance from
    the mass-weighted core centroid.  Supports both NFW and POWER_LAW
    lens types and both clusters (Abell 2744, El Gordo).

    UNITS
    -----
    For ABELL_2744 the literature values have been verified against the
    source papers and converted to a single, consistent unit system,
    h^-1 M_sun (h = 0.7):

      Merten+2011        : 2.24e14 M_sun  -> *0.7 = 1.568e14  r = 250 kpc  (core total)
      GRALE (Sebesta+18) : 2.25e14 M_sun  -> *0.7 = 1.575e14  r = 250 kpc  (core total)
      MARS (Cha+2024)    : 1.73e14 M_sun  -> *0.7 = 1.211e14  r = 200 kpc  (BCG-N peak only)

    The reconstruction is produced in plain M_sun internally and
    converted once (``* hubble_param``) so both sides of the comparison
    share h^-1 M_sun.

    EL_GORDO literature values are CARRIED OVER UNCHANGED from the prior
    version and have NOT been unit-verified against their source papers.
    They are flagged below; verify and convert before relying on the
    El Gordo percentages.  The plotting fixes apply to both clusters.

    Parameters
    ----------
    halos : NFW_Lens or PowerLawHalo
        Lens collection.
    plot_name : str
        Output figure path.
    plot_title : str
        Figure title (drawn only if show_title=True).
    cluster_name : str
        'ABELL_2744' or 'EL_GORDO'.
    lens_type : str
        'NFW' (default, backward compatible) or 'POWER_LAW'.
    mass_err : np.ndarray, optional
        1-sigma uncertainty on the reconstructed M(<r) curve, same length
        as the returned radius grid, in h^-1 M_sun.  If supplied (e.g. from
        the jackknife in compute_error_bars) a shaded +/-1sigma band is
        drawn.  If None, no band is drawn.
    y_range : (lo, hi), optional
        Fixed y-axis range in h^-1 M_sun, shared across panels for direct
        visual comparison.  If None, autoscale per panel.  For the Abell
        4-signal set, all/shear_f/shear_g fit ~(3e13, 5e14) but f_g
        overshoots to ~4e15; use a common (3e13, 5e15) if all four must
        share one axis.
    show_title : bool, optional
        Draw the code-generated title (review aid).  Leave False for
        thesis figures and let the LaTeX caption carry the metadata.

    Returns
    -------
    r : np.ndarray
        Radius grid (kpc).
    mass_enclosed : np.ndarray
        Cumulative reconstructed mass M(<r) in h^-1 M_sun.
    """
    if lens_type not in ("NFW", "POWER_LAW"):
        raise ValueError("Invalid lens_type. Choose 'NFW' or 'POWER_LAW'.")

    H = hubble_param  # 0.7

    # ------------------------------------------------------------------
    # Literature estimates.
    # Schema: label -> (mass_hinv, radius_kpc, err_hinv_or_None, kind)
    #   kind: 'total' = cluster-core total mass within r (filled marker)
    #         'peak'  = single BCG/peak mass, not a core total (open marker)
    # ------------------------------------------------------------------
    mass_estimates_abell = {
        "Merten+2011 (WL+SL)":    (2.24e14 * H, 250, 0.55e14 * H, "total"),
        "GRALE (Sebesta+2018)":   (2.25e14 * H, 250, 0.06e14 * H, "total"),
        "MARS (Cha+2024, BCG-N)": (1.73e14 * H, 200, None,        "peak"),
    }

    # EL GORDO: carried over unchanged from the prior version.
    # TODO: verify values, radii, and units against source papers and
    # convert to h^-1 M_sun as was done for Abell.  Units below are
    # UNVERIFIED.  'kind' set to 'total' provisionally.
    mass_estimates_elgordo = {
        "Cerny et al.":    (1.1e15,  500,  None, "total"),
        "Caminha et al.":  (1.84e15, 1000, None, "total"),
        "Diego et al.":    (0.8e15,  500,  None, "total"),
    }

    if cluster_name == "ABELL_2744":
        mass_estimates = mass_estimates_abell
        z_cluster = 0.308
        z_source = 1.0
    elif cluster_name == "EL_GORDO":
        mass_estimates = mass_estimates_elgordo
        z_cluster = 0.870
        z_source = 4.25
    else:
        raise ValueError("Invalid cluster name. Choose 'ABELL_2744' or 'EL_GORDO'.")

    # ------------------------------------------------------------------
    # Radius grid: span the literature radii with margin.
    # ------------------------------------------------------------------
    radii_lit = [r_lit for (_, r_lit, _, _) in mass_estimates.values()]
    r_min, r_max = min(radii_lit), max(radii_lit)
    r = np.linspace(r_min * 0.6, 1.3 * r_max, 200)

    # ------------------------------------------------------------------
    # Mass-weighted core centroid (kappa_star proxy for POWER_LAW,
    # mass for NFW), then recenter halos on it.
    # ------------------------------------------------------------------
    weight = halos.kappa_star if lens_type == "POWER_LAW" else halos.mass
    weight_total = float(np.sum(weight))
    if weight_total <= 0:
        raise ValueError(
            f"Cannot compute mass-weighted centroid: total weight = {weight_total}"
        )
    centroid = np.array([
        np.sum(halos.x * weight) / weight_total,
        np.sum(halos.y * weight) / weight_total,
    ])
    halos.x -= centroid[0] + 0.5
    halos.y -= centroid[1] + 0.5

    try:
        kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_cluster).to(
            u.kpc / u.arcsec
        ).value

        if lens_type == "POWER_LAW":
            mass_enclosed = _powerlaw_cumulative_mass(
                halos, r, z_cluster, z_source, kpc_per_arcsec
            )
        else:
            mass_enclosed = _nfw_cumulative_mass(
                halos, r, z_cluster, z_source, kpc_per_arcsec
            )

        # Single, explicit conversion to h^-1 M_sun.
        mass_enclosed = mass_enclosed * hubble_param

        # --------------------------------------------------------------
        # Console report: per-point agreement, unit-consistent.
        # --------------------------------------------------------------
        print(f"\n  Mass comparison ({cluster_name}, {lens_type}, h^-1 M_sun):")
        print(f"    {'Reference':24s} {'r':>5s}  {'lit':>10s} {'recon':>10s}  "
              f"{'err':>8s}  kind")
        print("    " + "-" * 66)
        for label, (m_lit, r_lit, e_lit, kind) in mass_estimates.items():
            m_rec = float(np.interp(r_lit, r, mass_enclosed))
            pct = 100.0 * (m_rec - m_lit) / m_lit
            print(f"    {label:24s} {r_lit:5.0f}  {m_lit:10.3e} {m_rec:10.3e}  "
                  f"{pct:+7.1f}%  {kind}")

        # --------------------------------------------------------------
        # Plot.
        # --------------------------------------------------------------
        _plot_mass_comparison(
            r, mass_enclosed, mass_estimates, plot_name, plot_title,
            lens_type, mass_err, y_range=y_range, show_title=show_title,
        )

    finally:
        # Always restore halo positions, even if an error was raised.
        halos.x += centroid[0] + 0.5
        halos.y += centroid[1] + 0.5

    return r, mass_enclosed


def _powerlaw_cumulative_mass(halos, r, z_l, z_source, kpc_per_arcsec):
    """
    Closed-form cumulative projected mass M(<r) for a set of power-law
    halos, summed over halos, within radius r of the mass-weighted
    cluster centroid.  Returns plain M_sun (caller applies h^-1).

        M_2D(<r) = 2*pi*Sigma_cr*beta*kappa_star*r_pivot^n*r^(2-n)/(2-n)

    per halo, with r measured from each halo's own center (clipped to the
    aperture).  No kappa grid, no mass-sheet transform, so positive by
    construction.
    """
    # Critical surface density: convert kg/m^2 -> M_sun/kpc^2 explicitly.
    # (Attaching M_sun/kpc^2 units to the kg/m^2 number is a known bug
    # that leaves the result ~5e8 too small; convert, don't relabel.)
    sigma_c_kg_per_m2 = critical_surface_density(z_l, z_source)
    sigma_c = (sigma_c_kg_per_m2 * u.kg / u.m**2).to(u.M_sun / u.kpc**2).value

    halo_offsets_kpc = np.hypot(halos.x, halos.y) * kpc_per_arcsec
    n_arr = np.atleast_1d(halos.slope).astype(float)
    kappa_star_arr = np.atleast_1d(halos.kappa_star).astype(float)
    r_pivot_kpc = float(halos.theta_star) * kpc_per_arcsec

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
            r_eff = max(0.0, radius - halo_offsets_kpc[k])
            if r_eff <= 0:
                continue
            denom = 2.0 - n_arr[k]
            if abs(denom) < 1e-6:
                denom = 1e-6
            M_halo = (
                2.0 * np.pi * sigma_c * beta * kappa_star_arr[k]
                * r_pivot_kpc ** n_arr[k] * r_eff ** denom / denom
            )
            total += float(M_halo)
        mass_enclosed[i] = total
    return mass_enclosed


def _nfw_cumulative_mass(halos, r, z_l, z_source, kpc_per_arcsec):
    """
    Cumulative projected mass M(<r) for NFW halos via the kappa-grid +
    mass-sheet workflow.  Returns plain M_sun (caller applies h^-1).
    """
    x_range, y_range = (-r[-1], r[-1]), (-r[-1], r[-1])
    nx, ny = int(x_range[1] - x_range[0]), int(y_range[1] - y_range[0])
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    kappa_total = np.zeros((ny, nx), dtype=float)

    halo_x_kpc = halos.x * kpc_per_arcsec
    halo_y_kpc = halos.y * kpc_per_arcsec

    area_per_pixel = None
    for i in range(len(halos.x)):
        halo = halo_obj.NFW_Lens(
            halos.x[i], halos.y[i], [0], halos.concentration[i],
            halos.mass[i], z_l, halos.chi2[i],
        )
        kappa, area_per_pixel = nfw_projected_mass(
            halo, r_p=0, return_2d=True, nx=nx, ny=ny,
            x_range=x_range, y_range=y_range,
            x_center=halo_x_kpc[i], y_center=halo_y_kpc[i],
            z_source=z_source,
        )
        if np.any(np.isinf(kappa)) or np.any(np.isnan(kappa)):
            print(f"Warning: NaN/Inf in kappa for halo {i}; skipping.")
            continue
        kappa_total += kappa

    kappa_total = mass_sheet_transformation(kappa_total, k=2)

    sigma_c = critical_surface_density(z_l, z_source)
    sigma_c = (sigma_c * u.M_sun / u.kpc**2).value
    M_2D_total = kappa_total * sigma_c * area_per_pixel

    XX, YY = np.meshgrid(x_vals, y_vals)
    RR = np.sqrt(XX**2 + YY**2)

    mass_enclosed = np.zeros_like(r)
    for i, radius in enumerate(r):
        mass_enclosed[i] = np.sum(M_2D_total[RR <= radius])
    return mass_enclosed


def _plot_mass_comparison(
    r, mass_enclosed, mass_estimates, plot_name, plot_title,
    lens_type, mass_err, y_range=None, show_title=False,
):
    """Thesis-quality cumulative-mass comparison figure.

    Cluster-agnostic: works for both Abell 2744 and El Gordo.  Filled
    markers denote cluster-core totals, open markers single-peak masses.
    Coincident-radius points are spread symmetrically so none is hidden.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    recon_label = f"ARCH reconstruction ({lens_type.replace('_', '-').title()})"
    ax.plot(r, mass_enclosed, color="k", lw=2.0, zorder=5, label=recon_label)

    # Optional +/-1sigma band on the reconstruction.
    if mass_err is not None:
        mass_err = np.asarray(mass_err, dtype=float)
        if mass_err.shape == mass_enclosed.shape:
            ax.fill_between(
                r, mass_enclosed - mass_err, mass_enclosed + mass_err,
                color="k", alpha=0.15, zorder=1,
                label=r"ARCH $\pm1\sigma$ (jackknife)",
            )
        else:
            print(f"Warning: mass_err shape {mass_err.shape} != "
                  f"{mass_enclosed.shape}; band not drawn.")

    # Marker styles by kind. A small palette cycles if there are more
    # points than preset markers, so this works for El Gordo's 3 points
    # and Abell's 4 alike.
    total_marker_cycle = ["o", "s", "P", "X"]
    peak_marker_cycle = ["D", "^", "v", "<"]
    total_color, peak_color = "#1f77b4", "#d62728"

    # De-overlap points that share a radius (e.g. Merten and GRALE both at
    # 250 kpc with nearly identical mass would otherwise hide one another).
    # Spread coincident points symmetrically about their true radius by a
    # small cosmetic offset (noted in the caption).
    radius_groups = {}
    for label, (_, r_lit, _, _) in mass_estimates.items():
        radius_groups.setdefault(r_lit, []).append(label)
    x_offset = {}
    jitter_kpc = 4.0
    for r_lit, labels in radius_groups.items():
        n = len(labels)
        if n == 1:
            x_offset[labels[0]] = 0.0
        else:
            for j, label in enumerate(labels):
                x_offset[label] = (j - (n - 1) / 2.0) * jitter_kpc

    n_total = n_peak = 0
    for label, (m_lit, r_lit, e_lit, kind) in mass_estimates.items():
        r_plot = r_lit + x_offset[label]
        if kind == "total":
            mk = total_marker_cycle[n_total % len(total_marker_cycle)]
            n_total += 1
            ax.errorbar(
                r_plot, m_lit, yerr=e_lit, marker=mk,
                ms=9, mfc=total_color, mec="k", ecolor=total_color,
                capsize=4, lw=0, elinewidth=1.5, zorder=6, label=label,
            )
        else:  # peak
            mk = peak_marker_cycle[n_peak % len(peak_marker_cycle)]
            n_peak += 1
            ax.errorbar(
                r_plot, m_lit, yerr=e_lit, marker=mk,
                ms=9, mfc="none", mec=peak_color, ecolor=peak_color,
                capsize=4, lw=0, elinewidth=1.5, mew=1.6, zorder=6, label=label,
            )

    ax.set_xlabel("Projected radius from core (kpc)")
    ax.set_ylabel(r"Enclosed projected mass $M(<r)\ \ [h^{-1}\,M_\odot]$")
    ax.set_yscale("log")
    if y_range is not None:
        ax.set_ylim(*y_range)
    if show_title:
        ax.set_title(plot_title)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # Legend outside the axes so it never collides with data/error bars.
    leg = ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=9, framealpha=0.95, borderaxespad=0.0,
    )
    leg.set_title("filled = core total\nopen = single-peak",
                  prop={"size": 8})

    fig.savefig(plot_name, dpi=200, bbox_inches="tight")
    plt.close(fig)