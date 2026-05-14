"""SIS strong-lensing chi2 functions."""

import numpy as np

from arch.sis_lensing import (
    backproject_source_positions_sis,
    magnification_sis,
    sigma_beta_from_magnification,
)


def chi2_strong_source_plane_sis(
    lenses, strong_systems, eps=1.0e-6, return_breakdown=False, use_magnification_correction=True
):
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
            raise ValueError(
                f"[{getattr(sys, 'system_id', 'unknown')}] theta_x/theta_y shape mismatch."
            )

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
                raise ValueError(
                    f"[{getattr(sys, 'system_id', 'unknown')}] sigma_theta shape mismatch."
                )

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


def chi2_flux_sis(lenses, strong_systems, eps=1.0e-6, return_breakdown=False):
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
