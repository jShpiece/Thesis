"""Task 13 — NFW strong-lensing unit tests."""

from __future__ import annotations

import sys

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.utils as utils
from scripts.paper2_shared import (
    _find_nfw_beta_rad,
    _find_nfw_einstein_radius,
    _make_nfw_halo,
    _make_nfw_wl_catalog,
    _make_two_image_nfw_system,
    _TestResults,
    attach_strong_systems,
)

# ═══════════════════════════════════════════════════════════════════════════
# 6) Task 13 — NFW Strong Lensing Unit Tests
# ═══════════════════════════════════════════════════════════════════════════
#
# Tests for the NFW strong lensing chain:
#
#   calculate_deflection_nfw → backproject_source_positions_nfw →
#   magnification_nfw → chi2_strong_source_plane_nfw →
#   compute_lambda_sl(NFW) → calculate_total_chi2(NFW)
#
# All tests are numerical rather than analytic (no closed-form NFW image
# positions exist).  Exact image positions are found by 1-D root-finding
# using scipy.optimize.brentq on the NFW lens equation.
#
# NFW double-image geometry (key difference from SIS):
#   Image 1 (major): same side as source, OUTSIDE Einstein ring  r > theta_E
#   Image 2 (minor): OPPOSITE side from source, INSIDE Einstein ring 0<r<theta_E
# ═══════════════════════════════════════════════════════════════════════════


# ── Helper: standard NFW halo factory ────────────────────────────────────

# ── Task 13 summary class ─────────────────────────────────────────────────


class _TestResults13(_TestResults):
    def summary(self) -> bool:
        self.header("TASK 13 — SUMMARY")
        all_ok = True
        for name, ok in self.results:
            tag = "PASSED" if ok else "*** FAILED ***"
            print(f"  {name:60s}  {tag}")
            all_ok = all_ok and ok
        print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}\n")
        return all_ok


# ── 13-A  Single NFW magnification: analytic vs FD Jacobian ──────────────


def _test_nfw_magnification_single_fd(R: _TestResults13):
    """
    Single NFW halo at origin.  Compare det(A) from magnification_nfw against
    the central-difference numerical Jacobian of backproject_source_positions_nfw.
    """
    R.header("13-A  Single NFW magnification: analytic vs FD Jacobian")

    halos = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source = 2.0

    # Test points along the x-axis, all outside the Einstein ring
    pts_x = np.array([5.0, 10.0, 20.0, 35.0, 60.0])
    pts_y = np.zeros_like(pts_x)

    _, det_analytic = utils.magnification_nfw(halos, pts_x, pts_y, z_source)

    h = 1e-4  # FD step size (arcsec)
    ok_all = True
    for k in range(len(pts_x)):
        bxp, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source
        )
        bxm, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source
        )
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source
        )
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source
        )
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source
        )
        _, bym = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source
        )
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source
        )
        _, bym2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source
        )
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx

        rel = abs(det_analytic[k] - det_fd) / max(abs(det_fd), 1e-30)
        ok_k = rel < 1e-4
        ok_all = ok_all and ok_k
        print(
            f'  r={pts_x[k]:5.1f}"  det_analytic={det_analytic[k]:+.8f}  '
            f"det_FD={det_fd:+.8f}  rel_err={rel:.1e}  {'OK' if ok_k else 'FAIL'}"
        )

    R.record("13-A  Single NFW: analytic vs FD Jacobian", ok_all)


# ── 13-B  Composite NFW: analytic vs FD Jacobian ─────────────────────────


def _test_nfw_magnification_composite_fd(R: _TestResults13):
    """
    Two NFW halos at different positions.  Compare the analytic composite
    det(A) from magnification_nfw against the central-difference FD Jacobian.
    """
    R.header("13-B  Composite NFW magnification: analytic vs FD Jacobian")

    halos = halo_obj.NFW_Lens(
        x=[0.0, 60.0],
        y=[0.0, 40.0],
        z=[0.0, 0.0],
        concentration=[1.0, 1.0],
        mass=[5e14, 3e14],
        redshift=0.3,
        chi2=[0.0, 0.0],
    )
    halos.calculate_concentration()
    z_source = 2.0

    # Test points well away from both halo centres
    pts_x = np.array([30.0, -20.0, 100.0, 50.0, -10.0])
    pts_y = np.array([-20.0, 60.0, 10.0, 80.0, -40.0])

    _, det_analytic = utils.magnification_nfw(halos, pts_x, pts_y, z_source)

    h = 1e-4
    ok_all = True
    for k in range(len(pts_x)):
        bxp, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source
        )
        bxm, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source
        )
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source
        )
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source
        )
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source
        )
        _, bym = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source
        )
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source
        )
        _, bym2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source
        )
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx

        rel = abs(det_analytic[k] - det_fd) / max(abs(det_fd), 1e-30)
        ok_k = rel < 2e-4
        ok_all = ok_all and ok_k
        print(
            f"  pt {k}: det_analytic={det_analytic[k]:+.8f}  "
            f"det_FD={det_fd:+.8f}  rel_err={rel:.1e}  {'OK' if ok_k else 'FAIL'}"
        )

    R.record("13-B  Composite NFW: analytic vs FD Jacobian", ok_all)


# ── 13-C  chi2 = 0 at exact image positions ───────────────────────────────


def _test_nfw_chi2_zero_at_true_lens(R: _TestResults13):
    """
    Build numerically-exact image positions for an NFW halo.  Verify:
      1. chi2 ≈ 0 at the true lens (images constructed to satisfy lens equation).
      2. chi2 > 0 at a perturbed lens.
    """
    R.header("13-C  NFW chi2 = 0 at exact images, > 0 at perturbed lens")

    halos_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source = 2.0

    print("  Finding NFW Einstein radius and radial caustic...")
    theta_E = _find_nfw_einstein_radius(halos_true, z_source)
    beta_rad = _find_nfw_beta_rad(halos_true, z_source, theta_E)
    print(f"  theta_E = {theta_E:.3f} arcsec,  beta_rad = {beta_rad:.4f} arcsec")

    # Use beta_rel well inside the radial caustic so a counter-image exists
    br = beta_rad * 0.45
    sys_nfw = _make_two_image_nfw_system(
        "nfw_c", halos_true, (br * 0.8, br * 0.6), z_source, theta_E, sigma_theta=0.05
    )
    print(
        f"  beta_rel = ({br*0.8:.4f}, {br*0.6:.4f}) arcsec  (|beta|={br:.4f}, beta_rad={beta_rad:.4f})"
    )
    print(
        f"  Image positions: theta1=({sys_nfw.theta_x[0]:.3f}, {sys_nfw.theta_y[0]:.3f})"
        f"  theta2=({sys_nfw.theta_x[1]:.3f}, {sys_nfw.theta_y[1]:.3f})"
    )

    # ── Sub-test 1: perfect model, no magnification correction ──
    chi2_true_uncorr = utils.chi2_strong_source_plane_nfw(
        halos_true, [sys_nfw], use_magnification_correction=False
    )
    ok_zero_uncorr = chi2_true_uncorr < 1e-6
    print(
        f"  True lens (uncorrected): chi2 = {chi2_true_uncorr:.2e}  "
        f"{'OK (< 1e-6)' if ok_zero_uncorr else 'FAIL'}"
    )

    # ── Sub-test 2: perfect model, with magnification correction ──
    chi2_true_corr = utils.chi2_strong_source_plane_nfw(
        halos_true, [sys_nfw], use_magnification_correction=True
    )
    ok_zero_corr = chi2_true_corr < 1e-6
    print(
        f"  True lens (corrected):   chi2 = {chi2_true_corr:.2e}  "
        f"{'OK (< 1e-6)' if ok_zero_corr else 'FAIL'}"
    )

    # ── Sub-test 3: perturbed lens (shifted by ~half the Einstein radius) ──
    halos_pert = _make_nfw_halo(x=theta_E * 0.5, y=-theta_E * 0.3, mass=5e14, redshift=0.3)
    chi2_pert = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_nfw], use_magnification_correction=False
    )
    ok_pert = chi2_pert > chi2_true_uncorr and chi2_pert > 1.0
    print(
        f"  Perturbed lens:          chi2 = {chi2_pert:.4f}  "
        f"{'OK (> 1.0 and > chi2_true)' if ok_pert else 'FAIL'}"
    )

    ok_all = ok_zero_uncorr and ok_zero_corr and ok_pert
    R.record("13-C  NFW chi2 = 0 at exact images", ok_all)


# ── 13-D  Magnification correction increases chi2 ─────────────────────────


def _test_nfw_magnification_correction_effect(R: _TestResults13):
    """
    At a perturbed lens, verify:
      1. chi2_corrected > chi2_uncorrected
      2. Magnification values in breakdown match FD Jacobian
      3. sigma_beta_x = sigma_theta / abs_mu for each image
    """
    R.header("13-D  NFW magnification correction increases chi2")

    halos_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source = 2.0
    theta_E = _find_nfw_einstein_radius(halos_true, z_source)
    beta_rad = _find_nfw_beta_rad(halos_true, z_source, theta_E)

    br = beta_rad * 0.45
    sys_nfw = _make_two_image_nfw_system(
        "nfw_d", halos_true, (br * 0.8, br * 0.6), z_source, theta_E, sigma_theta=0.05
    )
    halos_pert = _make_nfw_halo(x=theta_E * 0.5, y=-theta_E * 0.3, mass=5e14, redshift=0.3)

    chi2_corr, bd_corr = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_nfw], return_breakdown=True, use_magnification_correction=True
    )
    chi2_uncorr = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_nfw], use_magnification_correction=False
    )

    # ── Sub-test 1: corrected > uncorrected ──
    ok_larger = chi2_corr > chi2_uncorr
    print(
        f"  chi2_corr={chi2_corr:.4f}  chi2_uncorr={chi2_uncorr:.4f}  "
        f"corr>uncorr: {'OK' if ok_larger else 'FAIL'}"
    )

    # ── Sub-test 2: magnification from breakdown matches FD Jacobian ──
    bd = bd_corr["nfw_d"]
    tx = sys_nfw.theta_x
    ty = sys_nfw.theta_y
    h = 1e-4
    ok_mu = True
    for m in range(len(tx)):
        # FD Jacobian at this image position
        bxp, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] + h]), np.array([ty[m]]), z_source
        )
        bxm, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] - h]), np.array([ty[m]]), z_source
        )
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] + h]), z_source
        )
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] - h]), z_source
        )
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] + h]), np.array([ty[m]]), z_source
        )
        _, bym = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] - h]), np.array([ty[m]]), z_source
        )
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] + h]), z_source
        )
        _, bym2 = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] - h]), z_source
        )
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx
        abs_mu_fd = 1.0 / max(abs(det_fd), 1e-30)
        abs_mu_analytic = bd["abs_mu"][m]

        rel = abs(abs_mu_analytic - abs_mu_fd) / max(abs_mu_fd, 1e-30)
        ok_m = rel < 0.01
        ok_mu = ok_mu and ok_m
        print(
            f"    image {m}: |mu|_analytic={abs_mu_analytic:.4f}  "
            f"|mu|_FD={abs_mu_fd:.4f}  rel_err={rel:.1e}  {'OK' if ok_m else 'FAIL'}"
        )

    # ── Sub-test 3: sigma_beta consistent with magnification ──
    ok_sig = True
    for m in range(len(tx)):
        sig_b = bd["sigma_beta_x"][m]
        mu = bd["abs_mu"][m]
        # sigma_beta_from_magnification applies a floor; recover expected value
        inv_mu = max(1.0 / mu, 0.01)  # default mu_floor
        sig_b_expected = sys_nfw.sigma_theta * inv_mu
        close = np.isclose(sig_b, sig_b_expected, rtol=1e-4)
        ok_sig = ok_sig and close
        print(
            f"    image {m}: sig_b={sig_b:.6f}  expected={sig_b_expected:.6f}  "
            f"{'OK' if close else 'FAIL'}"
        )

    ok_all = ok_larger and ok_mu and ok_sig
    R.record("13-D  NFW magnification correction effect", ok_all)


# ── 13-E  Breakdown structure ──────────────────────────────────────────────


def _test_nfw_breakdown_structure(R: _TestResults13):
    """
    Two strong systems around one NFW halo.  Verify:
      1. Breakdown dict contains all required keys with correct shapes.
      2. sigma_beta_x <= sigma_theta for all images.
      3. Per-system chi2 sum equals total.
    """
    R.header("13-E  NFW breakdown structure: keys, shapes, consistency")

    halos_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source_A = 2.0
    z_source_B = 1.8
    theta_E_A = _find_nfw_einstein_radius(halos_true, z_source_A)
    theta_E_B = _find_nfw_einstein_radius(halos_true, z_source_B)
    beta_rad_A = _find_nfw_beta_rad(halos_true, z_source_A, theta_E_A)
    beta_rad_B = _find_nfw_beta_rad(halos_true, z_source_B, theta_E_B)

    br_A = beta_rad_A * 0.45
    br_B = beta_rad_B * 0.45
    theta_E_min = min(theta_E_A, theta_E_B)

    sys_A = _make_two_image_nfw_system(
        "E_A", halos_true, (br_A * 0.8, br_A * 0.6), z_source_A, theta_E_min, sigma_theta=0.06
    )
    sys_B = _make_two_image_nfw_system(
        "E_B", halos_true, (-br_B * 0.6, br_B * 0.8), z_source_B, theta_E_min, sigma_theta=0.08
    )

    # Perturb the lens slightly (by ~30% of the Einstein radius)
    halos_pert = _make_nfw_halo(x=theta_E_A * 0.3, y=-theta_E_A * 0.18, mass=5e14, redshift=0.3)

    chi2_total, bd = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_A, sys_B], return_breakdown=True, use_magnification_correction=True
    )

    required_keys = {
        "abs_mu",
        "sigma_beta_x",
        "sigma_beta_y",
        "sigma_theta",
        "det_A",
        "chi2",
        "n_images",
        "beta_bar",
        "beta",
    }

    ok_all = True
    for sid, info in bd.items():
        n_img = info["n_images"]

        ok_keys = required_keys.issubset(set(info.keys()))
        ok_shapes = (
            info["abs_mu"].shape == (n_img,)
            and info["sigma_beta_x"].shape == (n_img,)
            and info["sigma_beta_y"].shape == (n_img,)
            and info["det_A"].shape == (n_img,)
            and info["sigma_theta"].shape == (n_img,)
            and info["beta"].shape == (n_img, 2)
        )
        ok_sigma = np.all(info["sigma_beta_x"] <= info["sigma_theta"] + 1e-12)
        ok_sys = ok_keys and ok_shapes and ok_sigma
        ok_all = ok_all and ok_sys
        print(
            f"  {sid}: n_img={n_img}  keys={'OK' if ok_keys else 'MISS'}  "
            f"shapes={'OK' if ok_shapes else 'BAD'}  "
            f"sig_b<=sig_th={'OK' if ok_sigma else 'FAIL'}"
        )

    # Sum of per-system chi2 equals total
    bd_sum = sum(v["chi2"] for v in bd.values())
    ok_sum = np.isclose(chi2_total, bd_sum, rtol=1e-10)
    print(
        f"  chi2 sum: breakdown={bd_sum:.6f}  total={chi2_total:.6f}  "
        f"{'OK' if ok_sum else 'FAIL'}"
    )

    ok_all = ok_all and ok_sum
    R.record("13-E  NFW breakdown structure", ok_all)


# ── 13-F  compute_lambda_sl for NFW ───────────────────────────────────────


def _test_nfw_compute_lambda_sl(R: _TestResults13):
    """
    Build an NFW WL+SL catalog, compute lambda_sl via metric.compute_lambda_sl,
    and verify it equals the reduced-chi2 ratio (chi2_WL/dof_WL) / (chi2_SL/dof_SL).
    """
    R.header("13-F  compute_lambda_sl for NFW")

    halos_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source_wl = 0.8
    z_source_sl = 2.0

    src = _make_nfw_wl_catalog(halos_true, xmax=120.0, n_sources=80, z_source=z_source_wl, seed=42)

    theta_E = _find_nfw_einstein_radius(halos_true, z_source_sl)
    beta_rad = _find_nfw_beta_rad(halos_true, z_source_sl, theta_E)
    br = beta_rad * 0.45
    sys_F = _make_two_image_nfw_system(
        "F_sys", halos_true, (br * 0.8, br * 0.6), z_source_sl, theta_E, sigma_theta=0.06
    )
    attach_strong_systems(src, [sys_F])

    # Slightly wrong halo for non-trivial chi2
    halos_init = _make_nfw_halo(x=theta_E * 0.4, y=-theta_E * 0.2, mass=4.5e14, redshift=0.3)
    use_flags = [True, True, False]

    lam = metric.compute_lambda_sl(src, halos_init, use_flags, lens_type="NFW")

    # Manual calculation
    chi2_wl = metric.calculate_chi_squared(src, halos_init, use_flags, lens_type="NFW")
    dof_wl = metric.calc_degrees_of_freedom(src, halos_init, use_flags)
    chi2_sl = utils.chi2_strong_source_plane_nfw(halos_init, src.strong_systems)
    dof_sl = metric.calc_strong_dof(src)
    expected = (chi2_wl / dof_wl) / (chi2_sl / dof_sl)

    ok_match = np.isclose(lam, expected, rtol=1e-10)
    ok_finite = np.isfinite(lam) and lam > 0
    print(f"  compute_lambda_sl = {lam:.6f}")
    print(f"  manual            = {expected:.6f}")
    print(
        f"  match: {'OK' if ok_match else 'FAIL'}   "
        f"positive & finite: {'OK' if ok_finite else 'FAIL'}"
    )

    # Degenerate case: no strong systems → returns 1.0
    src_no_sl = _make_nfw_wl_catalog(
        halos_true, xmax=120.0, n_sources=80, z_source=z_source_wl, seed=42
    )
    lam_none = metric.compute_lambda_sl(src_no_sl, halos_init, use_flags, "NFW")
    ok_default = lam_none == 1.0
    print(f"  no-SL default = {lam_none}  {'OK' if ok_default else 'FAIL'}")

    ok_all = ok_match and ok_finite and ok_default
    R.record("13-F  compute_lambda_sl for NFW", ok_all)


# ── 13-G  calculate_total_chi2 decomposition for NFW ──────────────────────


def _test_nfw_total_chi2_decomposition(R: _TestResults13):
    """
    Verify calculate_total_chi2 with lens_type='NFW':
      1. chi2_total = chi2_WL + lambda * chi2_SL (explicit lambda)
      2. lambda passthrough
      3. No-SL mode (lambda=0)
      4. Auto-lambda fallback is finite and positive
      5. DOF: dof_sl = 2*(N_images - 1) per system
    """
    R.header("13-G  calculate_total_chi2 decomposition for NFW")

    halos_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source_wl = 0.8
    z_source_sl = 2.0

    src = _make_nfw_wl_catalog(halos_true, xmax=120.0, n_sources=80, z_source=z_source_wl, seed=99)

    theta_E = _find_nfw_einstein_radius(halos_true, z_source_sl)
    beta_rad = _find_nfw_beta_rad(halos_true, z_source_sl, theta_E)
    br = beta_rad * 0.45
    sys_G = _make_two_image_nfw_system(
        "G_sys", halos_true, (br * 0.7, -br * 0.5), z_source_sl, theta_E, sigma_theta=0.05
    )
    attach_strong_systems(src, [sys_G])

    halos_init = _make_nfw_halo(x=theta_E * 0.4, y=-theta_E * 0.2, mass=4.5e14, redshift=0.3)
    use_flags = [True, True, False]
    fixed_lam = 3.77

    # ── Sub-test 1: explicit lambda decomposition ──
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src,
        halos_init,
        use_flags,
        lens_type="NFW",
        use_strong_lensing=True,
        lambda_sl=fixed_lam,
    )
    expected_total = comps["chi2_wl"] + fixed_lam * comps["chi2_sl"]
    ok_decomp = np.isclose(chi2_total, expected_total, rtol=1e-10)
    ok_lam = comps["lambda_sl"] == fixed_lam
    print(
        f"  Explicit lam={fixed_lam}: chi2_total={chi2_total:.4f}  "
        f"expected={expected_total:.4f}  {'OK' if ok_decomp else 'FAIL'}"
    )
    print(f"  Lambda passthrough: {comps['lambda_sl']}  {'OK' if ok_lam else 'FAIL'}")

    # ── Sub-test 2: no-SL mode ──
    chi2_no, _, comps_no = metric.calculate_total_chi2(
        src, halos_init, use_flags, lens_type="NFW", use_strong_lensing=False
    )
    ok_no = comps_no["lambda_sl"] == 0.0 and np.isclose(chi2_no, comps_no["chi2_wl"], rtol=1e-10)
    print(f"  No-SL: lambda=0, chi2=chi2_wl  {'OK' if ok_no else 'FAIL'}")

    # ── Sub-test 3: auto-lambda fallback ──
    chi2_fb, _, comps_fb = metric.calculate_total_chi2(
        src,
        halos_init,
        use_flags,
        lens_type="NFW",
        use_strong_lensing=True,
        lambda_sl=None,
    )
    ok_fb = np.isfinite(comps_fb["lambda_sl"]) and comps_fb["lambda_sl"] > 0
    print(
        f"  Auto-lambda = {comps_fb['lambda_sl']:.6f}  "
        f"finite & positive: {'OK' if ok_fb else 'FAIL'}"
    )

    # ── Sub-test 4: DOF check ──
    dof_sl = comps["dof_sl"]
    expected_dof_sl = 2 * (2 - 1)  # one 2-image system: 2*(N-1) = 2
    ok_dof = dof_sl == expected_dof_sl
    print(f"  dof_sl={dof_sl}  expected={expected_dof_sl}  " f"{'OK' if ok_dof else 'FAIL'}")

    ok_all = ok_decomp and ok_lam and ok_no and ok_fb and ok_dof
    R.record("13-G  calculate_total_chi2 decomposition (NFW)", ok_all)


# ── Runner ────────────────────────────────────────────────────────────────


def run_nfw_sl_tests() -> bool:
    """Execute all Task 13 NFW strong lensing unit tests.  Returns True if all pass."""
    R = _TestResults13()
    _test_nfw_magnification_single_fd(R)
    _test_nfw_magnification_composite_fd(R)
    _test_nfw_chi2_zero_at_true_lens(R)
    _test_nfw_magnification_correction_effect(R)
    _test_nfw_breakdown_structure(R)
    _test_nfw_compute_lambda_sl(R)
    _test_nfw_total_chi2_decomposition(R)
    return R.summary()


if __name__ == "__main__":
    run_nfw_sl_tests()
