"""NFW strong-lensing chi2 functions."""

import numpy as np

from arch.nfw_lensing import (
    backproject_source_positions_nfw,
    magnification_nfw,
)
from arch.sis_lensing import sigma_beta_from_magnification


def chi2_strong_source_plane_nfw(
    halos,
    strong_systems,
    eps=1.0e-6,
    return_breakdown=False,
    use_magnification_correction=True,
    delta_n=0.15,
):
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


def chi2_flux_nfw(halos, strong_systems, eps=1.0e-6, return_breakdown=False):
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
