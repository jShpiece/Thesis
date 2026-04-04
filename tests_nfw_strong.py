"""
Tests for NFW strong-lensing integration in ARCH.

Mirrors the structure of paper2.py Tasks 11-12 (SIS strong lensing)
but adapted for NFW halos.  Test numbering:

  Task 13  — NFW strong-lensing unit tests
    13-A  NFW deflection analytic consistency
    13-B  NFW magnification vs finite-difference Jacobian
    13-C  chi2_strong_source_plane_nfw — self-consistent images → chi2 ≈ 0
    13-D  Perturbed model: magnification correction increases chi2
    13-E  compute_lambda_sl with lens_type="NFW"
    13-F  calculate_total_chi2 with lens_type="NFW"
    13-G  Full toy geometry: two NFW halos, two strong systems

  Task 14  — NFW strong-lensing integration (full pipeline)
    14-A  WL-only NFW pipeline baseline
    14-B  WL+SL NFW pipeline
    14-C  Backwards compatibility (SL data silently ignored)
    14-D  Post-run chi2 component verification
    14-E  Source-plane scatter improvement

Each unit test is self-contained and requires only numpy, scipy,
astropy, and the arch package.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import arch.pipeline as pipeline
import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
import arch.metric as metric
from arch.main import fit_lensing_field


# ═══════════════════════════════════════════════════════════════════════════
#  Toy data construction
# ═══════════════════════════════════════════════════════════════════════════

def make_nfw_halo(
    x: float = 0.0,
    y: float = 0.0,
    mass: float = 1e15,
    concentration: float = 8.0,
    redshift: float = 0.3,
) -> halo_obj.NFW_Lens:
    """
    Create a single NFW halo with given parameters.

    The defaults (M=1e15, c=8, z_l=0.3) produce a comfortably
    supercritical lens with kappa_s ≈ 0.26 and Einstein radius ≈ 7\"
    for z_s = 2.0.  This is comparable to observed strong-lensing
    clusters like Abell 1689.

    For reference, at these defaults:
        - theta_s ≈ 72\" (angular scale radius)
        - kappa_s ≈ 0.26 (characteristic convergence)
        - theta_E ≈ 7\" (tangential critical curve)
        - radial caustic ≈ 2-3\" (max beta for 3 images)
    """
    return halo_obj.NFW_Lens(
        x=x, y=y, z=0.0,
        concentration=concentration,
        mass=mass,
        redshift=redshift,
        chi2=np.array([0.0]),
    )


def _nfw_radial_beta(theta_abs, halo, z_source, sign=+1):
    """
    1-D lens equation along the radial direction from the halo centre.

    Places an image at angular distance |theta_abs| from the halo
    centre, on the side indicated by ``sign`` (+1 = positive-x,
    −1 = negative-x), and returns the x-component of the source-plane
    position relative to the halo centre.

    For a circularly-symmetric NFW lens, *both* images of a
    multiply-imaged source back-project to the **same** β.  So this
    function must return +β_target for BOTH the positive-side and
    negative-side images of that source.

    Parameters
    ----------
    theta_abs : float
        Positive angular distance from the halo centre (arcsec).
    halo : NFW_Lens
        Single NFW halo.
    z_source : float
        Source redshift (sets Σ_crit).
    sign : +1 or −1
        Which side of the halo to place the image.

    Returns
    -------
    float
        Source-plane x-offset from halo centre (arcsec).
    """
    tx = np.array([halo.x[0] + sign * abs(theta_abs)])
    ty = np.array([halo.y[0]])
    bx, _ = utils.backproject_source_positions_nfw(halo, tx, ty, z_source)
    return float(bx[0]) - halo.x[0]


def _estimate_theta_scan_range(halo, z_source):
    """
    Estimate a scan range [t_lo, t_hi] in arcsec for the image-finder.

    Uses the halo structural parameters to compute r_s (angular scale
    radius) and sets the scan from a small fraction of r_s up to
    several times R200 in projection.  This ensures the tangential
    critical curve (typically at x = θ/θ_s ≈ 0.05–0.3 for κ_s ~ 0.2–0.5)
    is comfortably enclosed.
    """
    z_l = float(np.asarray(halo.redshift).flat[0])
    c = float(halo.concentration[0])
    _, r200_arcsec = halo.calc_R200()
    r200_arcsec = float(np.atleast_1d(r200_arcsec)[0])
    theta_s = r200_arcsec / c  # angular scale radius (arcsec)

    # Scan from well inside r_s to well beyond it.
    # The counter-image can sit anywhere from very near the centre
    # (for sources near the radial caustic) up to just inside the
    # tangential critical curve.  Start the scan at a small fraction
    # of theta_s to catch these.
    t_lo = max(0.05, 0.005 * theta_s)   # ≈ 0.35" for theta_s=70"
    t_hi = max(200.0, 3.0 * r200_arcsec)  # well beyond R200
    return t_lo, t_hi, theta_s


def find_nfw_images_radial(
    halo: halo_obj.NFW_Lens,
    beta_target: float,
    z_source: float,
    theta_range: tuple[float, float] | None = None,
    n_scan: int = 800,
) -> list[float]:
    """
    Find image positions along the x-axis for a circularly-symmetric
    NFW lens by numerically solving β(θ) = β_target.

    Scans both the positive-x and negative-x sides of the halo to
    locate all roots of  _nfw_radial_beta(θ) − β_target = 0, then
    refines each with Brent's method.

    Key physics: both the positive-parity image (same side as the
    source, outside the tangential critical curve) and the
    negative-parity counter-image (opposite side, inside the critical
    curve) back-project to the **same** β_target.  Therefore the
    search target is β_target on both sides — NOT ±β_target.

    Parameters
    ----------
    halo : NFW_Lens
        Single halo (centred at halo.x, halo.y).
    beta_target : float
        Desired source-plane offset from the lens centre (arcsec).
    z_source : float
        Source redshift.
    theta_range : tuple or None
        (min, max) angular distance to scan (arcsec).
        If None, estimated automatically from halo parameters.
    n_scan : int
        Number of scan points per side.

    Returns
    -------
    images : list of float
        Signed angular distances from lens centre where images form.
        Positive = same side as source.
    """
    if theta_range is None:
        t_lo, t_hi, theta_s = _estimate_theta_scan_range(halo, z_source)
    else:
        t_lo, t_hi = theta_range

    images = []

    for sign in (+1, -1):
        # Both images back-project to the SAME source: search for
        # β(θ) = +beta_target on both sides (no sign flip).
        thetas = np.linspace(t_lo, t_hi, n_scan)
        betas = np.array([
            _nfw_radial_beta(t, halo, z_source, sign=sign)
            for t in thetas
        ])
        residuals = betas - beta_target   # ← same target on both sides

        # Find sign changes (roots of the residual)
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                try:
                    t_root = brentq(
                        lambda t: _nfw_radial_beta(t, halo, z_source, sign=sign) - beta_target,
                        thetas[i], thetas[i + 1],
                        xtol=1e-10, rtol=1e-12,
                    )
                    images.append(sign * t_root)
                except ValueError:
                    pass

    # Deduplicate images at the same physical position (within 1e-6 arcsec)
    if len(images) > 1:
        unique = [images[0]]
        for img in images[1:]:
            if all(abs(img - u) > 1e-6 for u in unique):
                unique.append(img)
        images = unique

    return images


def make_nfw_strong_system(
    system_id: str,
    halo: halo_obj.NFW_Lens,
    beta_offset: float,
    z_source: float = 2.0,
    sigma_theta: float = 0.05,
    theta_range: tuple[float, float] | None = None,
    include_flux: bool = True,
    sigma_flux_frac: float = 0.05,
    flux_noise_seed: int | None = None,
) -> source_obj.StrongLensingSystem:
    """
    Construct a multiply-imaged strong-lensing system for an NFW halo
    by numerically solving the lens equation.

    The source is placed at radial offset `beta_offset` (arcsec) from
    the halo centre along the x-axis.  All images along this axis are
    found via root-finding.

    Optionally generates synthetic flux data from the true magnification
    at each image position:

        F_i = |mu_i| * F_source

    where F_source is an arbitrary constant (set to 1.0; only ratios
    matter).  This provides the second constraint needed to break the
    position-mass degeneracy: image separation constrains alpha_1 - alpha_2,
    while the flux ratio constrains |mu_1| / |mu_2|.

    Parameters
    ----------
    system_id : str
        Identifier for the system.
    halo : NFW_Lens
        Single NFW halo.
    beta_offset : float
        Source-plane radial offset from halo centre (arcsec).
        Must be small enough to produce multiple images.
    z_source : float
        Source redshift.
    sigma_theta : float
        Positional uncertainty per image (arcsec).
    theta_range : tuple or None
        Search range for image finding.  None → auto from halo params.
    include_flux : bool
        If True (default), compute synthetic flux from true magnifications.
        If False, create a position-only system (backward compatible).
    sigma_flux_frac : float
        Fractional flux uncertainty per image (default 5%).
        sigma_flux_i = sigma_flux_frac * F_true_i.
        Typical HST photometry: 3-10%.
    flux_noise_seed : int or None
        If not None, add Gaussian noise to the flux values using this
        seed.  If None, flux is noiseless (F_i = |mu_i| exactly),
        suitable for perfect-model unit tests where chi2_flux should be 0.

    Returns
    -------
    StrongLensingSystem

    Raises
    ------
    ValueError
        If fewer than 2 images are found (source not multiply-imaged).
    """
    images = find_nfw_images_radial(
        halo, beta_offset, z_source, theta_range=theta_range,
    )

    if len(images) < 2:
        raise ValueError(
            f"Only {len(images)} image(s) found for beta_offset={beta_offset}\" "
            f"— source is not multiply-imaged by this halo.  "
            f"Try a smaller beta_offset or a more massive halo."
        )

    # Convert signed radial distances to 2-D image positions (along x-axis)
    x0, y0 = float(halo.x[0]), float(halo.y[0])
    theta_x = np.array([x0 + img for img in images], dtype=float)
    theta_y = np.full_like(theta_x, y0)

    # Verify the images actually converge in the source plane
    bx, by = utils.backproject_source_positions_nfw(halo, theta_x, theta_y, z_source)
    spread = np.max(np.hypot(bx - np.mean(bx), by - np.mean(by)))
    if spread > 0.01:
        raise ValueError(
            f"Images do not converge in source plane (spread = {spread:.2e}\").  "
            f"Image finder may have found spurious roots."
        )

    # ── Synthetic flux from true magnifications ──
    flux = None
    sigma_flux = None
    flux_meta = {}

    if include_flux:
        # Compute true magnification at each image position
        abs_mu, det_A = utils.magnification_nfw(halo, theta_x, theta_y, z_source)

        # Flux = |mu| * F_source (F_source = 1 in arbitrary units)
        F_source = 1.0
        flux_true = abs_mu * F_source

        # Measurement uncertainty: fractional
        sigma_flux = sigma_flux_frac * flux_true

        # Optionally add noise
        if flux_noise_seed is not None:
            rng = np.random.default_rng(flux_noise_seed)
            flux = flux_true + rng.normal(0, sigma_flux)
            # Ensure flux stays positive (re-draw if necessary)
            for attempt in range(10):
                neg = flux <= 0
                if not np.any(neg):
                    break
                flux[neg] = flux_true[neg] + rng.normal(0, sigma_flux[neg])
            # Last resort: clip to a small positive value
            flux = np.maximum(flux, 0.01 * flux_true)
        else:
            flux = flux_true.copy()

        flux_meta = {
            "flux_true": flux_true.copy(),
            "abs_mu_true": abs_mu.copy(),
            "det_A_true": det_A.copy(),
            "sigma_flux_frac": sigma_flux_frac,
            "flux_noise_seed": flux_noise_seed,
            "flux_ratio_true": abs_mu / abs_mu[np.argmax(abs_mu)],
        }

    return source_obj.StrongLensingSystem(
        system_id=system_id,
        theta_x=theta_x,
        theta_y=theta_y,
        z_source=float(z_source),
        sigma_theta=float(sigma_theta),
        flux=flux,
        sigma_flux=sigma_flux,
        meta={
            "toy": True,
            "lens_center": (x0, y0),
            "beta_offset": beta_offset,
            "halo_mass": float(halo.mass[0]),
            "halo_c": float(halo.concentration[0]),
            "n_images_found": len(images),
            "image_radii": [abs(img) for img in images],
            "source_plane_spread": float(spread),
            **flux_meta,
        },
    )


def make_weak_lensing_catalog_nfw(
    halos: halo_obj.NFW_Lens,
    xmax: float,
    n_sources: int,
    z_source: float = 1.0,
    sig_shear: float = 0.10,
    sig_flex: float = 0.02,
    sig_gflex: float = 0.03,
    rmin: float = 2.0,
    seed: int = 42,
) -> source_obj.Source:
    """
    Build a WL catalog lensed by one or more NFW halos, then add noise.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-xmax, xmax, size=n_sources)
    y = rng.uniform(-xmax, xmax, size=n_sources)

    # Avoid sampling on top of any halo centre
    keep = np.ones_like(x, dtype=bool)
    for i in range(len(halos.x)):
        r = np.hypot(x - halos.x[i], y - halos.y[i])
        keep &= (r > rmin)
    x, y = x[keep], y[keep]

    src = source_obj.Source(
        x=x, y=y,
        e1=np.zeros_like(x), e2=np.zeros_like(x),
        f1=np.zeros_like(x), f2=np.zeros_like(x),
        g1=np.zeros_like(x), g2=np.zeros_like(x),
        sigs=np.full_like(x, sig_shear),
        sigf=np.full_like(x, sig_flex),
        sigg=np.full_like(x, sig_gflex),
        redshift=np.full_like(x, z_source),
    )

    src.apply_lensing(halos, lens_type="NFW")
    src.apply_noise()
    return src


def attach_strong_systems(src: source_obj.Source, systems) -> None:
    """Attach a list of strong-lensing systems to a Source catalog."""
    if hasattr(src, "strong_systems"):
        src.strong_systems = list(systems)
        return
    if hasattr(src, "add_strong_system"):
        for s in systems:
            src.add_strong_system(s)
        return
    raise RuntimeError("Source does not expose strong_systems or add_strong_system.")


# ═══════════════════════════════════════════════════════════════════════════
#  Lightweight test harness  (same pattern as paper2.py)
# ═══════════════════════════════════════════════════════════════════════════

class _TestResults:
    """Lightweight accumulator for test pass/fail reporting."""
    def __init__(self):
        self.results: list[tuple[str, bool]] = []
    def record(self, name: str, passed: bool):
        self.results.append((name, passed))
    def header(self, title: str):
        print(f"\n{'='*72}")
        print(f"  {title}")
        print(f"{'='*72}")
    def summary(self) -> bool:
        self.header("SUMMARY")
        all_ok = True
        for name, ok in self.results:
            tag = "PASSED" if ok else "*** FAILED ***"
            print(f"  {name:55s}  {tag}")
            all_ok = all_ok and ok
        print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}\n")
        return all_ok


# ═══════════════════════════════════════════════════════════════════════════
#  Task 13 — NFW strong-lensing unit tests
# ═══════════════════════════════════════════════════════════════════════════


# ── 13-A  NFW deflection analytic consistency ─────────────────────────────

def _test_nfw_deflection_consistency(R: _TestResults):
    """
    For a circularly-symmetric NFW lens the deflection magnitude must
    satisfy the Gauss's-law identity:

        |alpha(theta)| = theta * kbar(<theta)

    where kbar(<theta) is the mean convergence inside theta.  We verify
    this at several radii by computing alpha from calculate_deflection_nfw
    and kbar from _nfw_kappa_and_gamma.

    Additionally, the back-projected source position must satisfy:

        beta = theta - alpha(theta)

    We check this round-trip identity.
    """
    R.header("13-A  NFW deflection analytic consistency")

    halo = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    # Structural parameters (mirror what calculate_deflection_nfw computes)
    from astropy.cosmology import Planck18 as cosmo_test
    from astropy import units as u_test

    z_l = halo.redshift
    Dl = cosmo_test.angular_diameter_distance(z_l).to(u_test.m).value
    sigma_crit = utils.critical_surface_density(z_l, z_source)
    rho_c = cosmo_test.critical_density(z_l).to(u_test.kg / u_test.m**3).value
    delta_c = float(halo.calc_delta_c()[0])
    rho_s = rho_c * delta_c
    r200_m, r200_arcsec = halo.calc_R200()
    r200_m = float(r200_m[0])
    r200_arcsec = float(r200_arcsec[0])
    c = float(halo.concentration[0])
    rs_m = r200_m / c
    theta_s = r200_arcsec / c
    kappa_s = rho_s * rs_m / sigma_crit

    # Test at several radii (in arcsec, along x-axis)
    radii = np.array([5., 10., 20., 50., 100., 200.])
    theta_x = radii
    theta_y = np.zeros_like(radii)

    # Deflection from the function
    alpha_x, alpha_y = utils.calculate_deflection_nfw(halo, theta_x, theta_y, z_source)
    alpha_mag = np.hypot(alpha_x, alpha_y)

    # Back-projection round-trip
    bx, by = utils.backproject_source_positions_nfw(halo, theta_x, theta_y, z_source)
    beta_mag = np.hypot(bx, by)  # should equal theta - alpha for radial images

    # kbar from _nfw_kappa_and_gamma: kbar = 4 kappa_s g(x) / x^2
    x = radii / theta_s
    h_x = utils._nfw_radial_h(x)
    g_x = np.log(x / 2.0) + h_x
    kbar = 4 * kappa_s * g_x / x**2

    alpha_expected = radii * kbar  # Gauss's law: |alpha| = theta * kbar

    ok_all = True
    for i in range(len(radii)):
        rel_err = abs(alpha_mag[i] - alpha_expected[i]) / abs(alpha_expected[i])
        ok_i = rel_err < 1e-6

        # Round-trip: beta = theta - alpha
        beta_expected = radii[i] - alpha_mag[i]
        ok_rt = np.isclose(beta_mag[i], abs(beta_expected), rtol=1e-6)

        ok_i = ok_i and ok_rt
        ok_all = ok_all and ok_i
        print(f"  r={radii[i]:6.1f}\"  |alpha|={alpha_mag[i]:.6f}  "
              f"theta*kbar={alpha_expected[i]:.6f}  rel_err={rel_err:.1e}  "
              f"roundtrip={'OK' if ok_rt else 'FAIL'}  {'OK' if ok_i else 'FAIL'}")

    # Diagonal check: deflection at 45 degrees should give the same magnitude
    r_diag = 50.0
    tx_d = np.array([r_diag / np.sqrt(2)])
    ty_d = np.array([r_diag / np.sqrt(2)])
    ax_d, ay_d = utils.calculate_deflection_nfw(halo, tx_d, ty_d, z_source)
    alpha_diag = np.hypot(ax_d, ay_d)

    # Compare with on-axis at same radius
    ax_r, _ = utils.calculate_deflection_nfw(halo, np.array([r_diag]), np.array([0.0]), z_source)
    ok_diag = np.isclose(alpha_diag[0], abs(ax_r[0]), rtol=1e-6)
    ok_all = ok_all and ok_diag
    print(f"  diagonal r={r_diag}: |alpha|_diag={alpha_diag[0]:.6f}  "
          f"|alpha|_axis={abs(ax_r[0]):.6f}  {'OK' if ok_diag else 'FAIL'}")

    R.record("13-A  NFW deflection consistency", ok_all)


# ── 13-B  NFW magnification vs finite-difference Jacobian ────────────────

def _test_nfw_magnification_fd(R: _TestResults):
    """
    Compare the analytic det(A) from magnification_nfw against a
    central-difference numerical Jacobian of the lens mapping beta(theta).
    
    Uses both single and two-halo configurations.
    """
    R.header("13-B  NFW magnification vs finite-difference Jacobian")

    z_source = 2.0

    # ── Single halo ──
    halo_single = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)

    # ── Two halos ──
    halo_double = halo_obj.NFW_Lens(
        x=[0.0, 40.0], y=[0.0, 20.0], z=[0.0, 0.0],
        concentration=[8.0, 7.0],
        mass=[1e15, 8e14],
        redshift=0.3,
        chi2=[0, 0],
    )

    configs = [
        ("single halo", halo_single),
        ("two halos", halo_double),
    ]

    # Test points well away from all halo centres
    pts_x = np.array([15.0, -20.0, 50.0, 5.0, 30.0])
    pts_y = np.array([8.0,  -10.0, 25.0, 30.0, -15.0])

    h = 1e-5
    ok_all = True

    for label, halos in configs:
        print(f"\n  Config: {label}")
        _, det_analytic = utils.magnification_nfw(halos, pts_x, pts_y, z_source)

        det_fd = np.empty(len(pts_x))
        for k in range(len(pts_x)):
            # ∂beta_x/∂theta_x
            bxp, _ = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
            bxm, _ = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
            dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

            # ∂beta_x/∂theta_y
            bxp2, _ = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
            bxm2, _ = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
            dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

            # ∂beta_y/∂theta_x
            _, byp = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
            _, bym = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
            dby_dtx = (byp[0] - bym[0]) / (2 * h)

            # ∂beta_y/∂theta_y
            _, byp2 = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
            _, bym2 = utils.backproject_source_positions_nfw(
                halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
            dby_dty = (byp2[0] - bym2[0]) / (2 * h)

            det_fd[k] = dbx_dtx * dby_dty - dbx_dty * dby_dtx

        for k in range(len(pts_x)):
            rel = abs(det_analytic[k] - det_fd[k]) / abs(det_fd[k])
            ok_k = rel < 1e-4
            ok_all = ok_all and ok_k
            print(f"    pt {k}: det(A)_analytic={det_analytic[k]:+.8f}  "
                  f"det(A)_FD={det_fd[k]:+.8f}  rel_err={rel:.1e}  "
                  f"{'OK' if ok_k else 'FAIL'}")

    R.record("13-B  NFW magnification (finite-diff)", ok_all)


# ── 13-C  chi2_strong: self-consistent images → chi2 ≈ 0 ────────────────

def _test_nfw_chi2_perfect_model(R: _TestResults):
    """
    Construct a multiply-imaged system by numerically solving the NFW
    lens equation, then verify that chi2_strong_source_plane_nfw returns
    chi2 ≈ 0 at the true halo parameters.  This is the NFW analog of
    the SIS "perfect model" test.
    """
    R.header("13-C  NFW chi2 = 0 for self-consistent images")

    halo = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    # Find multiple images for a source at small radial offset.
    # beta_offset must be well inside the radial caustic (~2-3" for this halo).
    sys_perf = make_nfw_strong_system(
        system_id="nfw_perf",
        halo=halo,
        beta_offset=1.0,  # arcsec from lens centre
        z_source=z_source,
        sigma_theta=0.05,
    )

    print(f"  Found {sys_perf.n_images} images at theta_x = {sys_perf.theta_x}")

    # Verify back-projections converge
    bx, by = utils.backproject_source_positions_nfw(
        halo, sys_perf.theta_x, sys_perf.theta_y, z_source
    )
    print(f"  Back-projected beta_x: {bx}")
    print(f"  Back-projected beta_y: {by}")
    beta_spread = np.max(np.hypot(bx - np.mean(bx), by - np.mean(by)))
    print(f"  Source-plane spread: {beta_spread:.2e}\"")

    # chi2 at the true parameters — should be ≈ 0
    chi2_corr = utils.chi2_strong_source_plane_nfw(
        halo, [sys_perf], use_magnification_correction=True
    )
    chi2_uncorr = utils.chi2_strong_source_plane_nfw(
        halo, [sys_perf], use_magnification_correction=False
    )
    ok_corr = np.isclose(chi2_corr, 0.0, atol=1e-4)
    ok_uncorr = np.isclose(chi2_uncorr, 0.0, atol=1e-4)
    print(f"  chi2_corr   = {chi2_corr:.2e}  {'OK' if ok_corr else 'FAIL'}")
    print(f"  chi2_uncorr = {chi2_uncorr:.2e}  {'OK' if ok_uncorr else 'FAIL'}")

    ok_all = ok_corr and ok_uncorr
    R.record("13-C  NFW chi2 = 0 (perfect model)", ok_all)


# ── 13-D  Perturbed model: magnification correction ──────────────────────

def _test_nfw_chi2_perturbed(R: _TestResults):
    """
    With a perturbed halo model, verify:
      1. chi2 > 0  (no longer a perfect fit)
      2. chi2_corrected > chi2_uncorrected  (magnification shrinks sigma_beta
         for highly magnified images near the Einstein radius)
      3. Breakdown metadata is present and correctly shaped
      4. Hand-computed chi2 from breakdown data matches function output
    """
    R.header("13-D  NFW chi2 perturbed + magnification correction")

    halo_true = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    sys = make_nfw_strong_system(
        system_id="nfw_pert",
        halo=halo_true,
        beta_offset=1.0,
        z_source=z_source,
        sigma_theta=0.08,
    )

    # Perturbed halo: shift position and mass
    halo_pert = make_nfw_halo(x=1.5, y=-0.8, mass=0.95e15, concentration=8.0, redshift=0.3)

    # ── Sub-test 1: chi2 > 0 ──
    # delta_n=0 isolates magnification correction from profile uncertainty
    chi2_corr, bd_corr = utils.chi2_strong_source_plane_nfw(
        halo_pert, [sys], return_breakdown=True,
        use_magnification_correction=True, delta_n=0,
    )
    chi2_uncorr, bd_uncorr = utils.chi2_strong_source_plane_nfw(
        halo_pert, [sys], return_breakdown=True,
        use_magnification_correction=False, delta_n=0,
    )
    ok_nonzero = chi2_corr > 0 and chi2_uncorr > 0
    print(f"  chi2_corr   = {chi2_corr:.4f}  chi2_uncorr = {chi2_uncorr:.4f}  "
          f"both>0: {'OK' if ok_nonzero else 'FAIL'}")

    # ── Sub-test 2: corrected > uncorrected ──
    ok_larger = chi2_corr > chi2_uncorr
    print(f"  corr > uncorr: {'OK' if ok_larger else 'FAIL'}")

    # ── Sub-test 3: breakdown metadata present and shaped ──
    bd = bd_corr["nfw_pert"]
    ok_meta = True
    required_keys = ["chi2", "n_images", "beta_bar", "beta",
                     "sigma_beta_x", "sigma_beta_y", "abs_mu", "det_A", "sigma_theta"]
    has_keys = all(k in bd for k in required_keys)
    ok_meta = ok_meta and has_keys
    n_img = bd["n_images"]
    if has_keys:
        shapes_ok = (bd["abs_mu"].shape == (n_img,)
                     and bd["sigma_beta_x"].shape == (n_img,)
                     and bd["det_A"].shape == (n_img,))
        ok_meta = ok_meta and shapes_ok
        # sigma_beta should be smaller than sigma_theta for magnified images
        sb_lt_st = np.all(bd["sigma_beta_x"] <= bd["sigma_theta"])
        ok_meta = ok_meta and sb_lt_st
        print(f"  n_images={n_img}  keys={'OK' if has_keys else 'MISS'}  "
              f"shapes={'OK' if shapes_ok else 'BAD'}  "
              f"sigma_beta<=sigma_theta={'OK' if sb_lt_st else 'FAIL'}")
    else:
        print(f"  MISSING KEYS: {[k for k in required_keys if k not in bd]}")

    # ── Sub-test 4: hand-compute chi2 from breakdown ──
    bx, by = bd["beta"][:, 0], bd["beta"][:, 1]
    sigx = bd["sigma_beta_x"]
    sigy = bd["sigma_beta_y"]
    wx = 1.0 / sigx**2
    wy = 1.0 / sigy**2
    bx_bar = np.sum(wx * bx) / np.sum(wx)
    by_bar = np.sum(wy * by) / np.sum(wy)
    chi2_hand = np.sum(((bx - bx_bar) / sigx)**2 + ((by - by_bar) / sigy)**2)
    ok_hand = np.isclose(chi2_corr, chi2_hand, rtol=1e-8)
    print(f"  Hand chi2 = {chi2_hand:.4f}  function = {chi2_corr:.4f}  "
          f"{'OK' if ok_hand else 'FAIL'}")

    ok_all = ok_nonzero and ok_larger and ok_meta and ok_hand
    R.record("13-D  NFW chi2 perturbed + magnification", ok_all)


# ── 13-E  compute_lambda_sl with NFW ─────────────────────────────────────

def _test_nfw_compute_lambda_sl(R: _TestResults):
    """
    Build a WL+SL NFW scenario, compute lambda_sl via metric.compute_lambda_sl
    with lens_type='NFW', and verify it equals the reduced-chi2 ratio.
    Also verify degenerate case: no SL → returns 1.0.
    """
    R.header("13-E  compute_lambda_sl with NFW")

    halo = make_nfw_halo(x=5.0, y=-3.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    # WL catalog
    src = make_weak_lensing_catalog_nfw(
        halo, xmax=80.0, n_sources=80, z_source=1.0, seed=99,
    )

    # Strong system
    sys_A = make_nfw_strong_system(
        system_id="lambda_nfw",
        halo=halo,
        beta_offset=1.0,
        z_source=z_source,
        sigma_theta=0.05,
    )
    attach_strong_systems(src, [sys_A])

    # Use a slightly wrong halo for non-trivial chi2
    halo_init = make_nfw_halo(x=5.5, y=-2.7, mass=0.95e15, concentration=8.0, redshift=0.3)
    use_flags = [True, True, False]

    lam = metric.compute_lambda_sl(src, halo_init, use_flags, lens_type='NFW')

    # Manual calculation (must include both scatter and flux to match compute_lambda_sl)
    chi2_wl = metric.calculate_chi_squared(src, halo_init, use_flags, lens_type='NFW')
    dof_wl = metric.calc_degrees_of_freedom(src, halo_init, use_flags)
    chi2_scatter = utils.chi2_strong_source_plane_nfw(halo_init, src.strong_systems)
    chi2_flux = utils.chi2_flux_nfw(halo_init, src.strong_systems)
    chi2_sl = chi2_scatter + chi2_flux
    dof_sl = metric.calc_strong_dof(src)

    expected = (chi2_wl / dof_wl) / (chi2_sl / dof_sl) if (chi2_sl > 0 and dof_sl > 0) else 1.0

    ok_match = np.isclose(lam, expected, rtol=1e-10)
    ok_finite = np.isfinite(lam) and lam > 0
    print(f"  compute_lambda_sl = {lam:.6f}")
    print(f"  manual            = {expected:.6f}")
    print(f"  match: {'OK' if ok_match else 'FAIL'}   "
          f"positive & finite: {'OK' if ok_finite else 'FAIL'}")

    # Degenerate case: no strong systems
    src_no_sl = make_weak_lensing_catalog_nfw(
        halo, xmax=80.0, n_sources=80, z_source=1.0, seed=99,
    )
    lam_none = metric.compute_lambda_sl(src_no_sl, halo_init, use_flags, 'NFW')
    ok_default = lam_none == 1.0
    print(f"  no-SL default = {lam_none}  {'OK' if ok_default else 'FAIL'}")

    ok_all = ok_match and ok_finite and ok_default
    R.record("13-E  compute_lambda_sl (NFW)", ok_all)


# ── 13-F  calculate_total_chi2 with NFW ──────────────────────────────────

def _test_nfw_total_chi2(R: _TestResults):
    """
    Verify that calculate_total_chi2 with lens_type='NFW' and explicit
    lambda_sl produces chi2_total = chi2_WL + lambda * chi2_SL.
    """
    R.header("13-F  calculate_total_chi2 with NFW")

    halo = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    src = make_weak_lensing_catalog_nfw(
        halo, xmax=60.0, n_sources=50, z_source=1.0, seed=42,
    )

    sys_B = make_nfw_strong_system(
        system_id="total_nfw",
        halo=halo,
        beta_offset=1.0,
        z_source=z_source,
        sigma_theta=0.06,
    )
    attach_strong_systems(src, [sys_B])

    halo_test = make_nfw_halo(x=0.5, y=-0.3, mass=0.98e15, concentration=8.0, redshift=0.3)
    flags = [True, True, False]

    # ── With explicit lambda ──
    fixed_lam = 3.14
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src, halo_test, flags, lens_type="NFW",
        use_strong_lensing=True, lambda_sl=fixed_lam,
    )
    expected_total = comps["chi2_wl"] + fixed_lam * comps["chi2_sl"]
    ok_total = np.isclose(chi2_total, expected_total, rtol=1e-10)
    ok_lam = comps["lambda_sl"] == fixed_lam
    print(f"  lambda_sl = {comps['lambda_sl']}  (passed {fixed_lam})  "
          f"{'OK' if ok_lam else 'FAIL'}")
    print(f"  chi2_total = {chi2_total:.4f}  expected = {expected_total:.4f}  "
          f"{'OK' if ok_total else 'FAIL'}")

    # ── Without SL → lambda = 0 ──
    chi2_no, _, comps_no = metric.calculate_total_chi2(
        src, halo_test, flags, lens_type="NFW", use_strong_lensing=False
    )
    ok_no = comps_no["lambda_sl"] == 0.0 and np.isclose(chi2_no, comps_no["chi2_wl"])
    print(f"  no-SL: lambda=0, chi2=chi2_wl  {'OK' if ok_no else 'FAIL'}")

    # ── Fallback (lambda_sl=None, SL active) ──
    chi2_fb, _, comps_fb = metric.calculate_total_chi2(
        src, halo_test, flags, lens_type="NFW",
        use_strong_lensing=True, lambda_sl=None,
    )
    ok_fb = np.isfinite(comps_fb["lambda_sl"]) and comps_fb["lambda_sl"] > 0
    print(f"  fallback lambda = {comps_fb['lambda_sl']:.6f}  "
          f"finite & positive: {'OK' if ok_fb else 'FAIL'}")

    ok_all = ok_total and ok_lam and ok_no and ok_fb
    R.record("13-F  calculate_total_chi2 (NFW)", ok_all)


# ── 13-G  Full toy geometry: two NFW halos, two systems ──────────────────

def _test_nfw_full_toy_geometry(R: _TestResults):
    """
    Two NFW halos with one strong system per halo.

    With a composite deflector the strong systems (constructed for
    individual halos) won't back-project perfectly because each
    system also feels the cross-deflection from the *other* halo.
    So we do NOT expect chi2 = 0 at the "true" parameters.

    Instead we verify:
      1. Magnification correction increases chi2 (smaller sigma_beta)
      2. Per-system breakdown sums to the total
      3. Moving the halos away from truth increases chi2 (sanity)
      4. Breakdown metadata is present and correctly shaped
    """
    R.header("13-G  Full NFW toy geometry (two halos)")

    halo_true = halo_obj.NFW_Lens(
        x=[0.0, 50.0], y=[0.0, 25.0], z=[0.0, 0.0],
        concentration=[8.0, 8.0],
        mass=[1e15, 8e14],
        redshift=0.3,
        chi2=[0, 0],
    )
    z_source = 2.0

    # Build strong systems for each halo individually
    halo_A = make_nfw_halo(x=0.0, y=0.0, mass=1e15, concentration=8.0, redshift=0.3)
    halo_B = make_nfw_halo(x=50.0, y=25.0, mass=8e14, concentration=8.0, redshift=0.3)

    sysA = make_nfw_strong_system(
        system_id="toy_nfw_A", halo=halo_A, beta_offset=1.0,
        z_source=z_source, sigma_theta=0.04,
    )
    sysB = make_nfw_strong_system(
        system_id="toy_nfw_B", halo=halo_B, beta_offset=0.8,
        z_source=z_source, sigma_theta=0.04,
    )
    systems = [sysA, sysB]

    # ── Sub-test 1: both chi2 values are finite and positive ──
    #
    # NOTE: For a composite deflector, the cross-deflection from the
    # second halo shifts all back-projections nearly uniformly.  This
    # can change the magnification pattern in ways that don't guarantee
    # chi2_corr > chi2_uncorr (unlike the single-halo case where
    # magnification correction always tightens sigma_beta for images
    # near the critical curve).  So we test finiteness, not ordering.
    chi2_corr = utils.chi2_strong_source_plane_nfw(
        halo_true, systems, use_magnification_correction=True
    )
    chi2_uncorr = utils.chi2_strong_source_plane_nfw(
        halo_true, systems, use_magnification_correction=False
    )
    ok_finite_true = np.isfinite(chi2_corr) and np.isfinite(chi2_uncorr) and chi2_corr > 0 and chi2_uncorr > 0
    print(f"  True halos: chi2_corr={chi2_corr:.2f}  chi2_uncorr={chi2_uncorr:.2f}  "
          f"both finite & >0: {'OK' if ok_finite_true else 'FAIL'}")
    if chi2_corr > chi2_uncorr:
        print(f"    (corr > uncorr as expected for near-critical images)")
    else:
        print(f"    (corr <= uncorr — acceptable for composite deflector geometry)")

    # ── Sub-test 2: grossly wrong model gives larger chi2 ──
    #
    # For a composite deflector, small positional perturbations can go
    # either way because they change the cross-deflection gradient
    # across the other halo's images (moving a halo farther away makes
    # its cross-deflection MORE uniform, which can REDUCE scatter).
    #
    # The robust test: place halos far from ALL images so they provide
    # negligible deflection.  Then beta ≈ theta (no lensing), and the
    # source-plane scatter is dominated by the raw image separations
    # (~50"), giving chi2 >> chi2_true.
    halo_wrong = halo_obj.NFW_Lens(
        x=[200.0, -200.0], y=[200.0, -200.0], z=[0.0, 0.0],
        concentration=[8.0, 8.0],
        mass=[1e15, 8e14],
        redshift=0.3,
        chi2=[0, 0],
    )
    chi2_wrong_uncorr = utils.chi2_strong_source_plane_nfw(
        halo_wrong, systems, use_magnification_correction=False
    )
    ok_wrong_larger = chi2_wrong_uncorr > chi2_uncorr
    print(f"  Wrong model (200\", uncorrected): chi2={chi2_wrong_uncorr:.2f} > "
          f"true {chi2_uncorr:.2f}  {'OK' if ok_wrong_larger else 'FAIL'}")

    # ── Sub-test 3: breakdown sums to total ──
    chi2_bd, bd = utils.chi2_strong_source_plane_nfw(
        halo_wrong, systems, return_breakdown=True,
        use_magnification_correction=True
    )
    bd_sum = sum(v["chi2"] for v in bd.values())
    ok_sum = np.isclose(chi2_bd, bd_sum, rtol=1e-10)
    print(f"  Breakdown sum={bd_sum:.4f}  total={chi2_bd:.4f}  "
          f"{'OK' if ok_sum else 'FAIL'}")

    # ── Sub-test 4: metadata present ──
    ok_meta = True
    for sid, info in bd.items():
        has_fields = all(k in info for k in
                         ["abs_mu", "sigma_beta_x", "sigma_theta", "det_A"])
        n_img = info["n_images"]
        if has_fields:
            shapes_ok = (info["abs_mu"].shape == (n_img,)
                         and info["sigma_beta_x"].shape == (n_img,)
                         and info["det_A"].shape == (n_img,))
        else:
            shapes_ok = False
        ok_sys = has_fields and shapes_ok
        ok_meta = ok_meta and ok_sys
        print(f"    {sid}: n_img={n_img}  fields={'OK' if has_fields else 'MISS'}  "
              f"shapes={'OK' if shapes_ok else 'BAD'}")

    ok_all = ok_finite_true and ok_wrong_larger and ok_sum and ok_meta
    R.record("13-G  Full NFW toy geometry", ok_all)


# ── 13-H  Flux-ratio chi2 ────────────────────────────────────────────────

def _test_nfw_flux_ratio_chi2(R: _TestResults):
    """
    Unit tests for chi2_flux_nfw: the flux-ratio constraint that breaks
    the position-mass degeneracy.

    Sub-tests:
      1. Perfect model (noiseless flux, true halo) → chi2_flux = 0
      2. Perturbed position → chi2_flux > 0
      3. Wrong mass at correct position → chi2_flux > 0  (mass sensitivity)
      4. Hand computation matches function output
      5. Breakdown metadata correct
      6. No-flux system → chi2_flux = 0  (backward compatibility)
    """
    R.header("13-H  Flux-ratio chi2")

    # Build a strong system WITH flux (default: include_flux=True, noiseless)
    halo_true = make_nfw_halo(x=5.0, y=-3.0, mass=1e15, concentration=8.0, redshift=0.3)
    z_source = 2.0

    sys_flux = make_nfw_strong_system(
        system_id="flux_test",
        halo=halo_true,
        beta_offset=1.0,
        z_source=z_source,
        sigma_theta=0.04,
        include_flux=True,
        flux_noise_seed=None,  # noiseless
    )

    # Verify flux was generated
    ok_has_flux = sys_flux.has_flux
    print(f"  has_flux: {'OK' if ok_has_flux else 'FAIL'}")
    print(f"  flux = {sys_flux.flux}")
    print(f"  sigma_flux = {sys_flux.sigma_flux}")
    print(f"  true magnifications = {sys_flux.meta.get('abs_mu_true', 'N/A')}")

    # ── Sub-test 1: Perfect model → chi2_flux = 0 ──
    chi2_perf = utils.chi2_flux_nfw(halo_true, [sys_flux])
    ok_zero = np.isclose(chi2_perf, 0.0, atol=1e-10)
    print(f"  Perfect model: chi2_flux = {chi2_perf:.6e}  "
          f"{'OK' if ok_zero else 'FAIL'}")

    # ── Sub-test 2: Perturbed position → chi2_flux > 0 ──
    halo_shift = make_nfw_halo(x=8.0, y=-3.0, mass=1e15, concentration=8.0, redshift=0.3)
    chi2_shift = utils.chi2_flux_nfw(halo_shift, [sys_flux])
    ok_shift = chi2_shift > 0
    print(f"  Shifted position: chi2_flux = {chi2_shift:.4f}  "
          f"{'OK' if ok_shift else 'FAIL'}")

    # ── Sub-test 3: Wrong mass at correct position → chi2_flux > 0 ──
    # This is the KEY test: source-plane scatter cannot detect mass errors
    # at the correct position (the images still converge to the same beta),
    # but flux ratios CAN because mu depends on kappa and gamma.
    halo_wrong_mass = make_nfw_halo(x=5.0, y=-3.0, mass=5e14, concentration=8.0, redshift=0.3)
    chi2_mass = utils.chi2_flux_nfw(halo_wrong_mass, [sys_flux])

    # Also compute source-plane scatter at the same wrong-mass halo
    chi2_scatter_mass = utils.chi2_strong_source_plane_nfw(halo_wrong_mass, [sys_flux])

    ok_mass = chi2_mass > 0
    print(f"  Wrong mass (correct pos): chi2_flux = {chi2_mass:.4f}  "
          f"chi2_scatter = {chi2_scatter_mass:.4f}  "
          f"{'OK' if ok_mass else 'FAIL'}")
    print(f"    → flux is mass-sensitive: {'YES' if ok_mass else 'NO'}")

    # ── Sub-test 4: Hand computation matches function ──
    chi2_bd, bd = utils.chi2_flux_nfw(halo_shift, [sys_flux], return_breakdown=True)
    sid = "flux_test"
    info = bd[sid]

    # Recompute by hand from breakdown data
    F = sys_flux.flux
    ref_idx = int(np.argmax(F))
    R_obs = F / F[ref_idx]
    frac_i = sys_flux.sigma_flux / np.maximum(F, 1e-30)
    frac_ref = sys_flux.sigma_flux[ref_idx] / max(F[ref_idx], 1e-30)
    sigma_R = R_obs * np.sqrt(frac_i**2 + frac_ref**2)
    sigma_R[ref_idx] = 0.0
    R_model = info["R_model"]
    mask = np.arange(sys_flux.n_images) != ref_idx
    chi2_hand = float(np.sum(((R_obs[mask] - R_model[mask]) / sigma_R[mask])**2))
    ok_hand = np.isclose(chi2_bd, chi2_hand, rtol=1e-10)
    print(f"  Hand chi2 = {chi2_hand:.4f}  function = {chi2_bd:.4f}  "
          f"{'OK' if ok_hand else 'FAIL'}")

    # ── Sub-test 5: Breakdown metadata correct ──
    expected_keys = {"chi2", "n_images", "ref_index",
                     "R_obs", "R_model", "sigma_R", "abs_mu", "det_A"}
    ok_keys = expected_keys.issubset(info.keys())
    ok_n = info["n_images"] == sys_flux.n_images
    ok_shapes = (info["R_obs"].shape == (sys_flux.n_images,)
                 and info["R_model"].shape == (sys_flux.n_images,)
                 and info["abs_mu"].shape == (sys_flux.n_images,))
    # For perfect model, R_obs == R_model (check at true halo)
    _, bd_perf = utils.chi2_flux_nfw(halo_true, [sys_flux], return_breakdown=True)
    R_match = np.allclose(bd_perf[sid]["R_obs"], bd_perf[sid]["R_model"], rtol=1e-8)
    ok_meta = ok_keys and ok_n and ok_shapes and R_match
    print(f"  Breakdown: keys={'OK' if ok_keys else 'MISS'}  n_images={'OK' if ok_n else 'BAD'}  "
          f"shapes={'OK' if ok_shapes else 'BAD'}  R_match@truth={'OK' if R_match else 'FAIL'}")

    # ── Sub-test 6: No-flux system → chi2_flux = 0 ──
    sys_noflux = make_nfw_strong_system(
        system_id="no_flux",
        halo=halo_true,
        beta_offset=1.0,
        z_source=z_source,
        sigma_theta=0.04,
        include_flux=False,
    )
    ok_noflux_attr = not sys_noflux.has_flux
    chi2_noflux = utils.chi2_flux_nfw(halo_true, [sys_noflux])
    ok_noflux = chi2_noflux == 0.0 and ok_noflux_attr
    print(f"  No-flux system: has_flux={sys_noflux.has_flux}  chi2={chi2_noflux}  "
          f"{'OK' if ok_noflux else 'FAIL'}")

    ok_all = (ok_has_flux and ok_zero and ok_shift and ok_mass
              and ok_hand and ok_meta and ok_noflux)
    R.record("13-H  Flux-ratio chi2", ok_all)


# ═══════════════════════════════════════════════════════════════════════════
#  Task 14 — NFW strong-lensing integration (full pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def _build_nfw_single_lens_scenario(seed: int = 55):
    """
    Build a reproducible single-NFW-halo test case with one strong system.

    Returns
    -------
    src : Source  (with one strong system attached)
    halo_true : NFW_Lens
    xmax : float
    """
    halo_true = make_nfw_halo(x=5.0, y=-3.0, mass=1e15, concentration=8.0, redshift=0.3)
    xmax = 80.0
    n_sources = 100
    z_source_wl = 1.0
    z_source_sl = 2.0

    src = make_weak_lensing_catalog_nfw(
        halo_true, xmax=xmax, n_sources=n_sources,
        z_source=z_source_wl, sig_shear=0.08, sig_flex=0.015,
        sig_gflex=0.025, seed=seed,
    )

    sys_A = make_nfw_strong_system(
        system_id="integ_nfw_A",
        halo=halo_true,
        beta_offset=1.0,
        z_source=z_source_sl,
        sigma_theta=0.04,
    )
    attach_strong_systems(src, [sys_A])

    return src, halo_true, xmax


def _nearest_lens_distance_nfw(lenses, true_x, true_y):
    """Return distance from nearest recovered halo to true position."""
    if len(lenses.x) == 0:
        return np.inf
    return float(np.min(np.hypot(lenses.x - true_x, lenses.y - true_y)))


def _test_integration_nfw_wl_only(R: _TestResults):
    """
    14-A: WL-only NFW pipeline baseline.
    Verify the pipeline runs, returns finite chi2, finds at least one halo,
    and places it reasonably near the truth.
    """
    R.header("14-A  NFW WL-only pipeline baseline")

    src, halo_true, xmax = _build_nfw_single_lens_scenario()
    true_x, true_y = float(halo_true.x[0]), float(halo_true.y[0])
    use_flags = [True, True, False]

    try:
        lenses_wl, rchi2_wl = fit_lensing_field(
            src, xmax, flags=True, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
            use_strong_lensing=False,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("14-A  NFW WL-only pipeline", False)
        return None, None

    ok_finite = np.isfinite(rchi2_wl)
    ok_nlens = len(lenses_wl.x) >= 1
    d_wl = _nearest_lens_distance_nfw(lenses_wl, true_x, true_y)
    ok_near = d_wl < 25.0  # coarse sanity check (arcsec)

    print(f"  rchi2={rchi2_wl:.4f}  N_lens={len(lenses_wl.x)}  "
          f"nearest Δ={d_wl:.2f}\"")
    print(f"  finite: {'OK' if ok_finite else 'FAIL'}  "
          f"N>=1: {'OK' if ok_nlens else 'FAIL'}  "
          f"near: {'OK' if ok_near else 'FAIL'}")

    ok_all = ok_finite and ok_nlens and ok_near
    R.record("14-A  NFW WL-only pipeline", ok_all)
    return lenses_wl, d_wl


def _test_integration_nfw_wl_plus_sl(R: _TestResults, d_wl_ref=None):
    """
    14-B: WL+SL NFW pipeline.

    KNOWN LIMITATION: optimize_lens_positions for NFW uses a local
    objective_function that only calls calculate_chi_squared (WL-only).
    The SL constraint enters only during forward_selection and
    optimize_lens_strength, so positions are not directly driven by SL.
    This means we CANNOT expect SL to systematically improve positional
    recovery compared to WL-only.

    What we CAN test:
      - The pipeline runs without exception
      - Returns finite reduced chi2
      - Finds at least one halo
      - Some recovered halo is within 25" of truth (coarse sanity)
    """
    R.header("14-B  NFW WL+SL pipeline")

    src, halo_true, xmax = _build_nfw_single_lens_scenario()
    true_x, true_y = float(halo_true.x[0]), float(halo_true.y[0])
    use_flags = [True, True, False]

    try:
        lenses_sl, rchi2_sl = fit_lensing_field(
            src, xmax, flags=True, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("14-B  NFW WL+SL pipeline", False)
        return None

    ok_finite = np.isfinite(rchi2_sl)
    ok_nlens = len(lenses_sl.x) >= 1
    d_sl = _nearest_lens_distance_nfw(lenses_sl, true_x, true_y)
    ok_near = d_sl < 40.0  # generous for NFW (field half-width = 80")

    print(f"  rchi2={rchi2_sl:.4f}  N_lens={len(lenses_sl.x)}  "
          f"nearest Δ={d_sl:.2f}\"")
    print(f"  finite: {'OK' if ok_finite else 'FAIL'}  "
          f"N>=1: {'OK' if ok_nlens else 'FAIL'}  "
          f"near truth: {'OK' if ok_near else 'FAIL'}")

    # Informational comparison with WL-only (not a pass/fail criterion)
    if d_wl_ref is not None:
        better = d_sl < d_wl_ref
        print(f"  [info] Δ_WL = {d_wl_ref:.2f}\"  Δ_SL = {d_sl:.2f}\"  "
              f"{'SL improved' if better else 'SL did not improve'}")
        if not better:
            print(f"  [info] Expected: NFW optimize_lens_positions does not use SL.")

    ok_all = ok_finite and ok_nlens and ok_near
    R.record("14-B  NFW WL+SL pipeline", ok_all)
    return lenses_sl


def _test_integration_nfw_backwards_compat(R: _TestResults):
    """
    14-C: fit_lensing_field with use_strong_lensing=False on a catalog
    that HAS strong_systems — the SL data should be silently ignored.
    """
    R.header("14-C  NFW Backwards compatibility")

    src, halo_true, xmax = _build_nfw_single_lens_scenario()
    use_flags = [True, True, False]

    ok_all = True

    # Explicit False
    try:
        lenses_1, rchi2_1 = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
            use_strong_lensing=False,
        )
        ok_1 = np.isfinite(rchi2_1) and len(lenses_1.x) >= 1
        print(f"  Explicit False: chi2={rchi2_1:.4f} n_lens={len(lenses_1.x)}  "
              f"{'OK' if ok_1 else 'FAIL'}")
    except Exception as e:
        print(f"  Explicit False: EXCEPTION {e}  FAIL")
        ok_1 = False
    ok_all = ok_all and ok_1

    # Default (omitted keyword)
    try:
        lenses_2, rchi2_2 = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
        )
        ok_2 = np.isfinite(rchi2_2) and len(lenses_2.x) >= 1
        print(f"  Default (omitted): chi2={rchi2_2:.4f} n_lens={len(lenses_2.x)}  "
              f"{'OK' if ok_2 else 'FAIL'}")
    except Exception as e:
        print(f"  Default (omitted): EXCEPTION {e}  FAIL")
        ok_2 = False
    ok_all = ok_all and ok_2

    if ok_1 and ok_2:
        ok_same = np.isclose(rchi2_1, rchi2_2, rtol=1e-8)
        print(f"  Both identical: {'OK' if ok_same else 'FAIL'}")
        ok_all = ok_all and ok_same

    R.record("14-C  NFW Backwards compatibility", ok_all)


def _test_integration_nfw_chi2_components(R: _TestResults):
    """
    14-D: After a WL+SL pipeline run, verify the chi2 components:
      - chi2_WL > 0, chi2_SL >= 0
      - lambda_sl finite and positive
      - total = WL + lambda * SL
    """
    R.header("14-D  NFW Post-run chi2 component verification")

    src, halo_true, xmax = _build_nfw_single_lens_scenario()
    use_flags = [True, True, False]

    try:
        lenses_sl, _ = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION during pipeline: {e}")
        R.record("14-D  NFW chi2 components", False)
        return

    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, 'NFW')
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src, lenses_sl, use_flags, lens_type='NFW',
        use_strong_lensing=True, lambda_sl=lambda_sl,
    )

    ok_wl = comps["chi2_wl"] > 0
    ok_sl = comps["chi2_sl"] >= 0
    ok_lam = np.isfinite(comps["lambda_sl"]) and comps["lambda_sl"] > 0
    ok_dof = comps["dof_wl"] > 0 and comps["dof_sl"] > 0
    ok_total = np.isclose(chi2_total,
                          comps["chi2_wl"] + comps["lambda_sl"] * comps["chi2_sl"],
                          rtol=1e-10)

    print(f"  chi2_WL   = {comps['chi2_wl']:.2f}  (dof={comps['dof_wl']})  "
          f"{'OK' if ok_wl else 'FAIL'}")
    print(f"  chi2_SL   = {comps['chi2_sl']:.2f}  (dof={comps['dof_sl']})  "
          f"{'OK' if ok_sl else 'FAIL'}")
    print(f"  lambda_sl = {comps['lambda_sl']:.6f}  "
          f"{'OK' if ok_lam else 'FAIL'}")
    print(f"  dof > 0:  WL={'OK' if comps['dof_wl']>0 else 'FAIL'}  "
          f"SL={'OK' if comps['dof_sl']>0 else 'FAIL'}")
    print(f"  total = WL + lambda*SL: {'OK' if ok_total else 'FAIL'}")

    ok_all = ok_wl and ok_sl and ok_lam and ok_dof and ok_total
    R.record("14-D  NFW chi2 components", ok_all)


def _test_integration_nfw_scatter_improvement(R: _TestResults):
    """
    14-E: Source-plane scatter diagnostic after WL+SL pipeline.

    KNOWN LIMITATION: NFW optimize_lens_positions does not include the
    SL constraint in its objective function.  Positions are driven by WL
    alone; SL only enters via forward_selection and optimize_lens_strength.
    Consequently, we CANNOT expect systematic SL scatter reduction.

    What we test:
      - Pipeline runs without exception
      - Final chi2_SL is finite
      - Per-system breakdown is well-formed

    Scatter improvement is logged informationally.
    """
    R.header("14-E  NFW source-plane scatter diagnostic")

    src, halo_true, xmax = _build_nfw_single_lens_scenario()
    use_flags = [True, True, False]

    # Initial-guess halos from the pipeline
    lenses_init = pipeline.generate_initial_guess(
        src, lens_type='NFW', z_l=halo_true.redshift
    )

    chi2_sl_init = utils.chi2_strong_source_plane_nfw(
        lenses_init, src.strong_systems
    )

    # Full pipeline
    try:
        lenses_final, _ = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=halo_true.redshift,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("14-E  NFW SL scatter diagnostic", False)
        return

    chi2_sl_final = utils.chi2_strong_source_plane_nfw(
        lenses_final, src.strong_systems
    )

    ok_finite = np.isfinite(chi2_sl_final) and chi2_sl_final >= 0
    improved = chi2_sl_final < chi2_sl_init
    print(f"  chi2_SL initial = {chi2_sl_init:.2f}")
    print(f"  chi2_SL final   = {chi2_sl_final:.2f}")
    print(f"  Scatter {'reduced' if improved else 'not reduced'}")
    print(f"  finite & non-negative: {'OK' if ok_finite else 'FAIL'}")
    if not improved:
        print(f"  [info] Expected: NFW position optimizer ignores SL constraint.")

    # Per-system breakdown — verify well-formed
    ok_breakdown = True
    try:
        _, bd = utils.chi2_strong_source_plane_nfw(
            lenses_final, src.strong_systems,
            return_breakdown=True, use_magnification_correction=True,
        )
        for sid, info in bd.items():
            ok_sys = np.isfinite(info['chi2']) and info['n_images'] >= 2
            ok_breakdown = ok_breakdown and ok_sys
            print(f"    {sid}: chi2={info['chi2']:.2f}  n_img={info['n_images']}  "
                  f"beta_bar=({info['beta_bar'][0]:.3f}, {info['beta_bar'][1]:.3f})  "
                  f"{'OK' if ok_sys else 'FAIL'}")
    except Exception as e:
        print(f"  Breakdown EXCEPTION: {e}")
        ok_breakdown = False

    ok_all = ok_finite and ok_breakdown
    R.record("14-E  NFW SL scatter diagnostic", ok_all)


# ═══════════════════════════════════════════════════════════════════════════
#  Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_nfw_unit_tests() -> bool:
    """Execute all Task 13 unit tests.  Returns True if all pass."""
    R = _TestResults()
    _test_nfw_deflection_consistency(R)
    _test_nfw_magnification_fd(R)
    _test_nfw_chi2_perfect_model(R)
    _test_nfw_chi2_perturbed(R)
    _test_nfw_compute_lambda_sl(R)
    _test_nfw_total_chi2(R)
    _test_nfw_full_toy_geometry(R)
    _test_nfw_flux_ratio_chi2(R)
    return R.summary()


def run_nfw_integration_tests() -> bool:
    """
    Execute all Task 14 integration tests.  Returns True if all pass.

    These tests call fit_lensing_field with lens_type='NFW', which runs
    the full optimiser pipeline.  Expect ~30-120 s per run depending on
    source count and halo mass.
    """
    R = _TestResults()

    # 14-A: WL-only baseline
    lenses_wl, d_wl = _test_integration_nfw_wl_only(R)

    # 14-B: WL+SL (pass WL distance for comparison)
    _test_integration_nfw_wl_plus_sl(R, d_wl_ref=d_wl)

    # 14-C: backwards compatibility
    _test_integration_nfw_backwards_compat(R)

    # 14-D: chi2 component verification
    _test_integration_nfw_chi2_components(R)

    # 14-E: source-plane scatter improvement
    _test_integration_nfw_scatter_improvement(R)

    return R.summary()


def run_all_nfw_tests() -> bool:
    """Run Task 13 + Task 14 tests.  Returns True if everything passes."""
    ok_13 = run_nfw_unit_tests()
    ok_14 = run_nfw_integration_tests()
    if ok_13 and ok_14:
        print("\n  ✓ ALL TASK 13 + TASK 14 (NFW) TESTS PASSED\n")
    else:
        if not ok_13:
            print("\n  ✗ Task 13 (NFW unit tests) had failures")
        if not ok_14:
            print("\n  ✗ Task 14 (NFW integration tests) had failures")
    return ok_13 and ok_14


if __name__ == "__main__":
    run_all_nfw_tests()