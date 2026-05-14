"""NFW projected mass, radial functions, lensing signals, deflection, backprojection, and magnification."""

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.integrate import quad

from arch.cosmology import critical_surface_density


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
    z_source=0.8,
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
            R = np.sqrt((x_vals[j] - x_center) ** 2 + (y_vals[i] - y_center) ** 2)
            surface_density = Sigma(R)
            kappa[i, j] = surface_density / sigma_c

    return kappa, area_per_pixel


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
    sol[m1] = np.arctanh(np.sqrt(1 - x[m1] ** 2)) / np.sqrt(1 - x[m1] ** 2)
    sol[m3] = np.arctan(np.sqrt(x[m3] ** 2 - 1)) / np.sqrt(x[m3] ** 2 - 1)
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
    sigma_crit = critical_surface_density(z_l, z_source)

    # Halo structural parameters
    rho_c = cosmo.critical_density(z_l).to(u.kg / u.m**3).value
    delta_c = np.atleast_1d(halos.calc_delta_c())
    rho_s = rho_c * delta_c  # (N_halo,)

    r200_m, r200_arcsec = halos.calc_R200()
    r200_m = np.atleast_1d(r200_m)
    r200_arcsec = np.atleast_1d(r200_arcsec)
    rs_m = r200_m / c_l  # (N_halo,) [metres]
    theta_s = r200_arcsec / c_l  # (N_halo,) [arcsec]

    kappa_s = (rho_s * rs_m) / sigma_crit  # (N_halo,)

    # Angular separations: (N_halo, N_pts)
    dx = tx[None, :] - x_l[:, None]
    dy = ty[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)

    # Dimensionless radius x = theta / theta_s
    x = r / theta_s[:, None]  # (N_halo, N_pts)

    # Radial function g(x) = ln(x/2) + h(x)  [stable at all x]
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
    kappa_per_halo, gamma_per_halo = _nfw_kappa_and_gamma(kappa_s[:, None], x, h_x=h_x, g_x=g_x)
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
    kappa_total = np.sum(kappa_per_halo, axis=0)  # (N_pts,)
    gamma1_total = np.sum(gamma1_per_halo, axis=0)
    gamma2_total = np.sum(gamma2_per_halo, axis=0)
    gamma_mag_total = np.hypot(gamma1_total, gamma2_total)

    # Jacobian determinant
    det_A = (1 - kappa_total) ** 2 - gamma_mag_total**2
    abs_mu = 1.0 / np.maximum(np.abs(det_A), 1.0e-30)

    return abs_mu, det_A


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
    behind = z_s > z_l
    sigma_crit_s = np.where(behind, sigma_crit_s, np.inf)  # zero-out non-behind via ∞
    sigma_crit_hs = sigma_crit_s[None, :]  # (1, N_s) -> (N_h, N_s)

    # --- Halo structure ---
    r200_m, r200_arcsec = halos.calc_R200()  # expect (N_h,), but accept scalars
    r200_m = np.atleast_1d(r200_m)
    r200_arcsec = np.atleast_1d(r200_arcsec)

    rs_m = r200_m / c_l  # (N_h,)
    rho_c = cosmo.critical_density(z_l).to(u.kg / u.m**3).value  # scalar
    delta_c = np.atleast_1d(halos.calc_delta_c())  # (N_h,)
    rho_s = rho_c * delta_c  # (N_h,)

    kappa_s = (rho_s * rs_m)[:, None] / sigma_crit_hs  # (N_h, N_s)

    # Flexion scale
    rad_to_arcsec = u.radian.to(u.arcsecond)
    flexion_s = (kappa_s * Dl) / (rs_m[:, None] * rad_to_arcsec)  # (N_h, N_s)

    # --- Angular separations (arcsec) ---
    dx = x_s[None, :] - x_l[:, None]  # (N_h, N_s)
    dy = y_s[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r == 0.0, 1.0e-2, r)

    theta_s = (r200_arcsec / c_l)[:, None]  # (N_h, 1)
    x = np.abs(r / theta_s)  # (N_h, N_s)

    # --- Radial terms ---
    def radial_term_1(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        t1 = np.sqrt(np.clip((1 - x[m1]) / (1 + x[m1]), 0.0, None))
        sol[m1] = 1 - (2 / np.sqrt(1 - x[m1] ** 2)) * np.arctanh(t1)
        t2 = np.sqrt(np.clip((x[m2] - 1) / (1 + x[m2]), 0.0, None))
        sol[m2] = 1 - (2 / np.sqrt(x[m2] ** 2 - 1)) * np.arctan(t2)
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
            8 * np.arctanh(t1) / (x[m1] ** 2 * np.sqrt(1 - x[m1] ** 2))
            + 4 * np.log(x[m1] / 2) / x[m1] ** 2
            - 2 / (x[m1] ** 2 - 1)
            + 4 * np.arctanh(t1) / ((x[m1] ** 2 - 1) * np.sqrt(1 - x[m1] ** 2))
        )
        t2 = np.sqrt(np.clip((x[m2] - 1) / (x[m2] + 1), 0.0, None))
        sol[m2] = (
            8 * np.arctan(t2) / (x[m2] ** 2 * np.sqrt(x[m2] ** 2 - 1))
            + 4 * np.log(x[m2] / 2) / x[m2] ** 2
            - 2 / (x[m2] ** 2 - 1)
            + 4 * np.arctan(t2) / ((x[m2] ** 2 - 1) ** (3 / 2))
        )
        sol[m3] = 10.0 / 3.0 + 4.0 * np.log(0.5)
        return sol

    def radial_term_3(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        t1 = np.sqrt(np.clip((1 - x[m1]) / (1 + x[m1]), 0.0, None))
        sol[m1] = (1 / (1 - x[m1] ** 2)) * (
            1 / x[m1] - (2 * x[m1]) / np.sqrt(1 - x[m1] ** 2) * np.arctanh(t1)
        )
        t2 = np.sqrt(np.clip((x[m2] - 1) / (1 + x[m2]), 0.0, None))
        sol[m2] = (1 / (x[m2] ** 2 - 1)) * (
            (2 * x[m2]) / np.sqrt(x[m2] ** 2 - 1) * np.arctan(t2) - 1 / x[m2]
        )
        return sol

    def radial_term_4(x):
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = ~m1
        leading = 8 / x**3 - 20 / x + 15 * x
        k = (1 - x) / (1 + x)
        t1 = np.sqrt(np.clip(k[m1], 0.0, None))
        sol[m1] = (2 / np.sqrt(1 - x[m1] ** 2)) * np.arctanh(t1)
        t2 = np.sqrt(np.clip(-k[m2], 0.0, None))
        sol[m2] = (2 / np.sqrt(x[m2] ** 2 - 1)) * np.arctan(t2)
        sol *= leading
        return sol

    def radial_term_5(x):
        # Use this to compute necessary term for kappa
        sol = np.zeros_like(x)
        m1 = x < 1
        m2 = x == 1
        m3 = x > 1
        sol[m1] = np.arctanh(np.sqrt(1 - x[m1] ** 2)) / np.sqrt(1 - x[m1] ** 2)
        sol[m2] = 1
        sol[m3] = np.arctan(np.sqrt(x[m3] ** 2 - 1)) / np.sqrt(x[m3] ** 2 - 1)
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
    kappa_mag = 2 * kappa_s * (1 - term_5) / (x**2 - 1)
    shear_mag = -kappa_s * term_2

    def calc_flexion(flexion_s, x, term_1, term_3):
        I1 = -2 * flexion_s
        I2 = 2 * x * term_1 / (x**2 - 1) ** 2
        I3 = term_3 / (x**2 - 1)
        return I1 * (I2 - I3)

    def calc_g_flexion(flexion_s, x, term_4):
        I1 = 2 * flexion_s
        I2 = (8 / x**3) * np.log(x / 2)
        I3 = (3 / x) * (1 - 2 * x**2) + term_4
        I4 = (x**2 - 1) ** 2
        return I1 * (I2 + (I3 / I4))

    flexion_mag = calc_flexion(flexion_s, x, term_1, term_3)
    g_flexion_mag = calc_g_flexion(flexion_s, x, term_4)

    # Zero contributions where source is not behind lens
    shear_mag = np.where(behind[None, :], shear_mag, 0.0)
    flexion_mag = np.where(behind[None, :], flexion_mag, 0.0)
    g_flexion_mag = np.where(behind[None, :], g_flexion_mag, 0.0)

    # --- Sum over halos => per-source outputs ---
    kappa = np.sum(kappa_mag, axis=0)
    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flexion_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flexion_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flexion_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flexion_mag * sin3phi, axis=0)

    # If a single source was passed, return scalars
    if (np.ndim(sources.x) == 0) and (np.ndim(sources.y) == 0) and (np.ndim(sources.redshift) == 0):
        return (
            shear_1.item(),
            shear_2.item(),
            flexion_1.item(),
            flexion_2.item(),
            g_flexion_1.item(),
            g_flexion_2.item(),
        )
    return kappa, shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2
