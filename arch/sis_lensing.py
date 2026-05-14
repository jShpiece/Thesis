"""SIS lensing signals, deflection, backprojection, magnification, and sigma_beta."""

import numpy as np


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
    cos2phi = cos_phi**2 - sin_phi**2
    sin2phi = 2 * cos_phi * sin_phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

    shear_mag = -lenses.te[:, np.newaxis] / (2 * r)
    flexion_mag = -lenses.te[:, np.newaxis] / (2 * r**2)
    g_flexion_mag = 3 * lenses.te[:, np.newaxis] / (2 * r**2)

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
    sum_dax_dtx = np.sum(te_over_r * sin_phi**2, axis=0)  # (N_pts,)
    sum_dax_dty = np.sum(-te_over_r * sin_phi * cos_phi, axis=0)  # (N_pts,)
    sum_day_dtx = sum_dax_dty  # symmetric
    sum_day_dty = np.sum(te_over_r * cos_phi**2, axis=0)  # (N_pts,)

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
