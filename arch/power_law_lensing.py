"""Power-law halo lensing signals, deflection, backprojection, and magnification."""

import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

from arch.cosmology import critical_surface_density


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
            raise ValueError("All halos must share the same redshift " "for this function.")

    # --- Lensing efficiency beta(z_s) = D_ls / D_s ---
    # Equivalent to Sigma_crit(z_l, inf) / Sigma_crit(z_l, z_s).
    # Use existing critical_surface_density helper for consistency with NFW.
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit_inf = c.value**2 / (4.0 * np.pi * G.value * Dl)

    def _sigma_crit_vec(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(
                lambda zs: critical_surface_density(zl_scalar, zs),
                otypes=[float],
            )
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit_vec(z_l, z_s)  # (N_s,)
    behind = z_s > z_l
    # beta = sigma_crit_inf / sigma_crit_s, zero for non-behind sources
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)
    beta_hs = beta_s[None, :]  # (1, N_s) -> broadcasts to (N_h, N_s)

    # --- Angular separations (arcsec) ---
    dx = x_s[None, :] - x_l[:, None]  # (N_h, N_s)
    dy = y_s[None, :] - y_l[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r == 0.0, 1.0e-2, r)  # softening, matches NFW

    # Trig combinations for spin-2 (shear) and spin-3 (g-flexion)
    cos_phi = dx / r
    sin_phi = dy / r
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2.0 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    # --- Profile evaluation per (halo, source) ---
    # Broadcast slope and kappa_star across sources
    n_hs = n_l[:, None]  # (N_h, 1)
    k_hs = k_star[:, None]  # (N_h, 1)

    # Guard against the formal n=2 divergence
    two_minus_n = 2.0 - n_hs
    two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)

    # kappa(theta, z_s) = beta(z_s) * kappa_star * (theta/theta_star)^(-n)
    kappa = beta_hs * k_hs * (r / theta_star) ** (-n_hs)  # (N_h, N_s)

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
            raise ValueError("All halos must share the same redshift " "for this function.")

    # Source redshift: scalar or array matching theta_x
    z_s = np.atleast_1d(z_source).astype(float)
    if z_s.size == 1 and tx.size > 1:
        z_s = np.full_like(tx, z_s[0])
    elif z_s.size != tx.size:
        raise ValueError(f"z_source size {z_s.size} does not match " f"theta_x size {tx.size}.")

    # --- Lensing efficiency beta(z_s) = D_ls / D_s ---
    Dl = cosmo.angular_diameter_distance(z_l).to(u.m).value
    sigma_crit_inf = c.value**2 / (4.0 * np.pi * G.value * Dl)

    def _sigma_crit_vec(zl_scalar, zs_vec):
        try:
            return critical_surface_density(zl_scalar, zs_vec)
        except Exception:
            vfun = np.vectorize(
                lambda zs: critical_surface_density(zl_scalar, zs),
                otypes=[float],
            )
            return vfun(zs_vec)

    sigma_crit_s = _sigma_crit_vec(z_l, z_s)  # (N_pts,)
    behind = z_s > z_l
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)

    # --- Geometry: (N_lens, N_pts) ---
    dx = tx[None, :] - xl[:, None]
    dy = ty[None, :] - yl[:, None]
    r = np.hypot(dx, dy)
    r = np.where(r < eps, eps, r)  # softening

    # --- Per-halo deflection magnitude prefactor ---
    # alpha_mag(theta) = beta(z_s) * [2 kappa_star / (2-n)] * theta_star^n * theta^(1-n)
    # Cast slope/kappa_star to (N_lens, 1) for broadcasting against (N_lens, N_pts).
    n_h = n_l[:, None]
    k_h = k_star[:, None]
    two_minus_n = 2.0 - n_h
    two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)

    coeff = (2.0 * k_h / two_minus_n) * (theta_star**n_h)  # (N_lens, 1)
    # alpha as a vector: coeff * r^(-n) * (dx, dy)
    # NB: r^(1-n) * (dx/r) = r^(-n) * dx, which is what we want.
    factor = coeff * r ** (-n_h)  # (N_lens, N_pts)

    # Apply per-source efficiency (broadcasts over lens axis)
    factor = factor * beta_s[None, :]

    # --- Sum over halos ---
    alpha_x = np.sum(factor * dx, axis=0)
    alpha_y = np.sum(factor * dy, axis=0)

    return alpha_x, alpha_y


def backproject_source_positions_power_law(halos, theta_x, theta_y, z_source, eps=1.0e-6):
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
    ax, ay = calculate_deflection_power_law(halos, theta_x, theta_y, z_source, eps=eps)
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
    sigma_crit_inf = c.value**2 / (4.0 * np.pi * G.value * Dl)

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
    beta_s = np.where(behind, sigma_crit_inf / sigma_crit_s, 0.0)  # (N_pts,)

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
    M = (2.0 * k_h / two_minus_n) * (theta_star**n_h) * r ** (-n_h)
    M = M * beta_s[None, :]

    # Per-halo Jacobian terms d alpha_a / d theta_b = M * (delta_ab - n hat_a hat_b)
    # Sum over halos
    sum_dax_dtx = np.sum(M * (1.0 - n_h * cos_phi**2), axis=0)  # (N_pts,)
    sum_dax_dty = np.sum(-M * n_h * cos_phi * sin_phi, axis=0)
    sum_day_dtx = sum_dax_dty  # symmetric
    sum_day_dty = np.sum(M * (1.0 - n_h * sin_phi**2), axis=0)

    A11 = 1.0 - sum_dax_dtx
    A12 = -sum_dax_dty
    A21 = -sum_day_dtx
    A22 = 1.0 - sum_day_dty

    det_A = A11 * A22 - A12 * A21
    abs_mu = 1.0 / np.maximum(np.abs(det_A), 1.0e-30)

    return abs_mu, det_A
