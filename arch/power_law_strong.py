"""Power-law strong-lensing chi2 functions."""

import numpy as np
from astropy import units as u
from astropy.constants import G, c
from astropy.cosmology import Planck18 as cosmo

from arch.cosmology import critical_surface_density
from arch.power_law_lensing import (
    backproject_source_positions_power_law,
    magnification_power_law,
)
from arch.sis_lensing import sigma_beta_from_magnification


def chi2_strong_source_plane_power_law(
    halos,
    strong_systems,
    sigma_n=None,
    alpha_cal=1.0,
    eps=1.0e-6,
    return_breakdown=False,
    use_magnification_correction=True,
    use_profile_uncertainty=True,
):
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
            raise ValueError(f"sigma_n length {sig_n.size} != number of " f"halos {N_h}.")

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
            halos,
            tx,
            ty,
            z_s,
            eps=eps,
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
                halos,
                tx,
                ty,
                z_s,
                eps=eps,
            )
            sig_beta_meas = sigma_beta_from_magnification(sig_theta, abs_mu)
        else:
            sig_beta_meas = sig_theta
            abs_mu = None
            det_A = None

        # --- Profile-uncertainty source-plane sigma ---
        if np.any(sig_n > 0.0):
            # Per-halo deflection magnitude and distance to each image
            dx = tx[None, :] - xl[:, None]  # (N_h, N_im)
            dy = ty[None, :] - yl[:, None]
            r = np.hypot(dx, dy)
            r = np.where(r < eps, eps, r)

            # Need alpha magnitude per halo per image — compute via the
            # closed form rather than calling calculate_deflection_power_law
            # (which sums over halos).
            Dl = cosmo.angular_diameter_distance(halos.redshift).to(u.m).value
            sigma_crit_inf = c.value**2 / (4.0 * np.pi * G.value * Dl)
            sigma_crit_s = critical_surface_density(halos.redshift, z_s)
            beta_zs = sigma_crit_inf / sigma_crit_s if z_s > halos.redshift else 0.0

            two_minus_n = 2.0 - n_l
            two_minus_n = np.where(np.abs(two_minus_n) < 1e-6, 1e-6, two_minus_n)
            coeff = (2.0 * k_star / two_minus_n) * (theta_star**n_l)  # (N_h,)

            alpha_mag = beta_zs * coeff[:, None] * r ** (1.0 - n_l[:, None])  # (N_h, N_im)

            # Phase 0:  sigma_prof_j(theta) = alpha_cal * |alpha_j(theta)| *
            #                                 sigma_n_j * |ln(r_jm / theta_star)|
            ln_term = np.abs(np.log(r / theta_star))  # (N_h, N_im)
            sig_per_halo = alpha_cal * alpha_mag * sig_n[:, None] * ln_term  # (N_h, N_im)
            # RSS over halos (independent slope uncertainties)
            sig_beta_prof = np.sqrt(np.sum(sig_per_halo**2, axis=0))  # (N_im,)
        else:
            sig_beta_prof = np.zeros(N_im)

        # Total
        sig_beta = np.sqrt(sig_beta_meas**2 + sig_beta_prof**2)
        sig_beta = np.maximum(sig_beta, 1.0e-12)

        # --- Inverse-variance weighted mean source position ---
        w = 1.0 / sig_beta**2
        bx_bar = np.sum(w * bx) / np.sum(w)
        by_bar = np.sum(w * by) / np.sum(w)

        # --- Source-plane scatter chi2 for this system ---
        chi2_i = np.sum(((bx - bx_bar) / sig_beta) ** 2 + ((by - by_bar) / sig_beta) ** 2)
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


def chi2_flux_power_law(
    halos, strong_systems, eps=1.0e-6, return_breakdown=False, marginalize_normalization=True
):
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
                raise ValueError(
                    f"[{sid}] flux length {f_obs.size} != " f"theta_x length {tx.size}."
                )
            if sig_f.shape != f_obs.shape:
                raise ValueError(f"[{sid}] sigma_flux shape mismatch.")

            # Model magnifications at each image
            abs_mu, _ = magnification_power_law(
                halos,
                tx,
                ty,
                sys.z_source,
                eps=eps,
            )

            # Closed-form marginalization over the overall flux scale s.
            # Minimizes  sum_m ((f_m - s |mu_m|) / sigma_m)^2  over s:
            w = 1.0 / np.maximum(sig_f, 1.0e-30) ** 2
            num = np.sum(f_obs * abs_mu * w)
            den = np.sum(abs_mu**2 * w)
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
                halos,
                tx,
                ty,
                sys.z_source,
                eps=eps,
            )
            mu_ref = abs_mu[0]
            if mu_ref <= 0.0:
                # Pathological case — reference image at a critical curve
                if return_breakdown:
                    breakdown[sid] = {"chi2": 0.0, "skipped": True, "reason": "mu_ref <= 0"}
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
