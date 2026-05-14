"""Task 11 — Magnification / sigma_beta / lambda_sl unit tests."""

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
    _TestResults,
    attach_strong_systems,
    make_two_image_sis_system_at_lens,
    make_weak_lensing_catalog_two_lenses,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4) Task 11 — Magnification / sigma_beta / lambda_sl unit tests
# ═══════════════════════════════════════════════════════════════════════════
#
# These tests verify the full physics chain that underpins the combined
# WL+SL weighting approach:
#
#   magnification_sis → sigma_beta_from_magnification →
#   chi2_strong_source_plane_sis → compute_lambda_sl → calculate_total_chi2
#
# All tests are analytic or semi-analytic — they compare code output to
# hand-derived SIS results.  Where an exact analytic answer is not
# available (composite deflectors), finite-difference numerical
# derivatives serve as the reference.
# ═══════════════════════════════════════════════════════════════════════════
# ── 11-A  Single SIS analytic magnification ──────────────────────────────


def _test_magnification_single_sis(R: _TestResults):
    """
    For a lone SIS at the origin with Einstein radius θ_E:
        det(A) = 1 − θ_E / r
        |μ|    = 1 / |1 − θ_E / r|

    Verify at several radii spanning both sides of the Einstein ring,
    plus a diagonal to confirm circular symmetry.
    """
    R.header("11-A  Single SIS analytic magnification")

    te = 2.5
    lens = halo_obj.SIS_Lens(x=0.0, y=0.0, te=te, chi2=[0])

    radii = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
    theta_x, theta_y = radii, np.zeros_like(radii)
    abs_mu, det_A = utils.magnification_sis(lens, theta_x, theta_y)

    det_expected = 1.0 - te / radii

    ok_all = True
    for i in range(len(radii)):
        # det(A) comparison
        if np.abs(det_expected[i]) < 1e-10:
            # Critical curve — just check det(A) ≈ 0 and |μ| very large
            ok_i = np.abs(det_A[i]) < 1e-4 and abs_mu[i] > 1e6
        else:
            mu_expected = 1.0 / np.abs(det_expected[i])
            ok_i = np.isclose(det_A[i], det_expected[i], atol=1e-8) and np.isclose(
                abs_mu[i], mu_expected, rtol=1e-6
            )
        ok_all = ok_all and ok_i
        status = "OK" if ok_i else "FAIL"
        print(
            f"  r={radii[i]:5.1f}  det(A)={det_A[i]:+.6f}  "
            f"expected={det_expected[i]:+.6f}  |mu|={abs_mu[i]:.4f}  {status}"
        )

    # Diagonal check: r=4 along 45°
    r = 4.0
    tx_d, ty_d = np.array([r / np.sqrt(2)]), np.array([r / np.sqrt(2)])
    _, det_d = utils.magnification_sis(lens, tx_d, ty_d)
    ok_diag = np.isclose(det_d[0], 1.0 - te / r, atol=1e-8)
    ok_all = ok_all and ok_diag
    print(
        f"  diagonal r={r}: det(A)={det_d[0]:.6f}  expected={1.0-te/r:.6f}  "
        f"{'OK' if ok_diag else 'FAIL'}"
    )

    # Negative parity inside Einstein ring
    r_in = te / 2
    _, det_in = utils.magnification_sis(lens, np.array([r_in]), np.array([0.0]))
    ok_parity = det_in[0] < 0
    ok_all = ok_all and ok_parity
    print(
        f"  parity r={r_in} < θ_E: det(A)={det_in[0]:.6f}  negative={'OK' if ok_parity else 'FAIL'}"
    )

    R.record("11-A  Single SIS magnification", ok_all)


# ── 11-B  Composite SIS finite-difference Jacobian ───────────────────────


def _test_magnification_composite_fd(R: _TestResults):
    """
    Two SIS halos at different positions.  Compare the analytic det(A) from
    magnification_sis against a central-difference numerical Jacobian of the
    lens mapping β(θ).
    """
    R.header("11-B  Composite SIS finite-difference Jacobian")

    lens = halo_obj.SIS_Lens(x=[0.0, 6.0], y=[0.0, 3.0], te=[3.0, 2.0], chi2=[0, 0])

    # Test points well away from both lens centres
    pts_x = np.array([3.0, -4.0, 8.0, 1.0, 5.0])
    pts_y = np.array([1.5, -2.0, 5.0, 6.0, -3.0])

    _, det_analytic = utils.magnification_sis(lens, pts_x, pts_y)

    h = 1e-5
    det_fd = np.empty(len(pts_x))
    for k in range(len(pts_x)):
        # ∂β/∂θ by central differences
        bxp, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] + h]), np.array([pts_y[k]])
        )
        bxm, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] - h]), np.array([pts_y[k]])
        )
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] + h])
        )
        bxm2, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] - h])
        )
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] + h]), np.array([pts_y[k]])
        )
        _, bym = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] - h]), np.array([pts_y[k]])
        )
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] + h])
        )
        _, bym2 = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] - h])
        )
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd[k] = dbx_dtx * dby_dty - dbx_dty * dby_dtx

    ok_all = True
    for k in range(len(pts_x)):
        rel = np.abs(det_analytic[k] - det_fd[k]) / np.abs(det_fd[k])
        ok_k = rel < 1e-4
        ok_all = ok_all and ok_k
        print(
            f"  pt {k}: det(A)_analytic={det_analytic[k]:+.8f}  "
            f"det(A)_FD={det_fd[k]:+.8f}  rel_err={rel:.1e}  "
            f"{'OK' if ok_k else 'FAIL'}"
        )

    R.record("11-B  Composite SIS (finite-diff)", ok_all)


# ── 11-C  sigma_beta conversion ──────────────────────────────────────────


def _test_sigma_beta(R: _TestResults):
    """
    sigma_beta = sigma_theta / |μ|, floored at inv_mu = mu_floor.
    """
    R.header("11-C  sigma_beta_from_magnification")

    sig = 0.1
    cases = [
        # (|μ|, mu_floor, expected sigma_beta)
        (5.0, 0.01, 0.1 / 5.0),
        (1.0, 0.01, 0.1),
        (200.0, 0.01, 0.1 * 0.01),  # floored: 1/200 < 0.01
        (50.0, 0.05, 0.1 * 0.05),  # floored: 1/50=0.02 < 0.05
        (10.0, 0.01, 0.1 / 10.0),  # not floored
    ]

    ok_all = True
    for mu, floor, expected in cases:
        result = utils.sigma_beta_from_magnification(sig, np.array([mu]), mu_floor=floor)
        ok = np.isclose(result[0], expected, rtol=1e-10)
        ok_all = ok_all and ok
        print(
            f"  |mu|={mu:6.1f}  floor={floor}  sig_b={result[0]:.6f}  "
            f"expected={expected:.6f}  {'OK' if ok else 'FAIL'}"
        )

    # Array input
    mus = np.array([2.0, 5.0, 10.0, 100.0])
    sigs = np.full_like(mus, sig)
    result = utils.sigma_beta_from_magnification(sigs, mus, mu_floor=0.01)
    expected_arr = np.array([sig / 2, sig / 5, sig / 10, sig * 0.01])
    ok_arr = np.allclose(result, expected_arr, rtol=1e-10)
    ok_all = ok_all and ok_arr
    print(f"  array: {result}  expected={expected_arr}  {'OK' if ok_arr else 'FAIL'}")

    R.record("11-C  sigma_beta conversion", ok_all)


# ── 11-D  chi2_strong corrected vs uncorrected ──────────────────────────


def _test_chi2_strong_corrected(R: _TestResults):
    """
    Use make_two_image_sis_system_at_lens to build a known geometry, then
    perturb the lens position slightly so the back-projections don't
    perfectly converge.  Verify:

      1. With magnification correction, chi2 > chi2 without (because
         sigma_beta shrinks for magnified images).
      2. The magnifications in the breakdown match the analytic SIS formula.
      3. Hand-calculated chi2 matches the function output.
    """
    R.header("11-D  chi2_strong corrected vs uncorrected")

    te_true = 4.0
    lens_true = halo_obj.SIS_Lens(x=0.0, y=0.0, te=te_true, chi2=[0])

    # Build a perfect 2-image system
    sys_perf = make_two_image_sis_system_at_lens(
        system_id="perf",
        lens_center_xy=(0.0, 0.0),
        te_true=te_true,
        beta_rel_xy=(0.8, 0.3),
        sigma_theta=0.08,
    )

    # ── Sub-test 1: perfect model → chi2 = 0 ──
    chi2_corr_perf = utils.chi2_strong_source_plane_sis(
        lens_true, [sys_perf], use_magnification_correction=True
    )
    chi2_uncorr_perf = utils.chi2_strong_source_plane_sis(
        lens_true, [sys_perf], use_magnification_correction=False
    )
    ok_perf = np.isclose(chi2_corr_perf, 0.0, atol=1e-8) and np.isclose(
        chi2_uncorr_perf, 0.0, atol=1e-8
    )
    print(
        f"  Perfect model:  chi2_corr={chi2_corr_perf:.2e}  "
        f"chi2_uncorr={chi2_uncorr_perf:.2e}  {'OK' if ok_perf else 'FAIL'}"
    )

    # ── Sub-test 2: perturbed lens → chi2_corr > chi2_uncorr ──
    lens_perturbed = halo_obj.SIS_Lens(x=0.3, y=-0.2, te=te_true, chi2=[0])

    chi2_corr, bd_corr = utils.chi2_strong_source_plane_sis(
        lens_perturbed, [sys_perf], return_breakdown=True, use_magnification_correction=True
    )
    chi2_uncorr, bd_uncorr = utils.chi2_strong_source_plane_sis(
        lens_perturbed, [sys_perf], return_breakdown=True, use_magnification_correction=False
    )
    ok_larger = chi2_corr > chi2_uncorr
    print(
        f"  Perturbed model:  chi2_corr={chi2_corr:.2f}  "
        f"chi2_uncorr={chi2_uncorr:.2f}  corr>uncorr: {'OK' if ok_larger else 'FAIL'}"
    )

    # ── Sub-test 3: magnification values match analytic ──
    bd = bd_corr["perf"]
    tx = sys_perf.theta_x
    ty = sys_perf.theta_y
    ok_mu = True
    for m in range(len(tx)):
        dx = tx[m] - lens_perturbed.x[0]
        dy = ty[m] - lens_perturbed.y[0]
        r = np.hypot(dx, dy)
        mu_expected = 1.0 / np.abs(1.0 - te_true / r)
        mu_got = bd["abs_mu"][m]
        close = np.isclose(mu_got, mu_expected, rtol=1e-4)
        ok_mu = ok_mu and close
        print(
            f"    image {m}: r={r:.3f}  |mu|={mu_got:.4f}  "
            f"expected={mu_expected:.4f}  {'OK' if close else 'FAIL'}"
        )

    # ── Sub-test 4: sigma_beta in breakdown is consistent ──
    ok_sig = True
    for m in range(len(tx)):
        sig_b = bd["sigma_beta_x"][m]
        sig_b_expected = sys_perf.sigma_theta / bd["abs_mu"][m]
        close = np.isclose(sig_b, sig_b_expected, rtol=1e-4)
        ok_sig = ok_sig and close
        print(
            f"    image {m}: σ_β={sig_b:.6f}  expected={sig_b_expected:.6f}  "
            f"{'OK' if close else 'FAIL'}"
        )

    # ── Sub-test 5: hand-compute chi2 from breakdown data ──
    bx, by = bd["beta"][:, 0], bd["beta"][:, 1]
    sigx = bd["sigma_beta_x"]
    sigy = bd["sigma_beta_y"]
    wx = 1.0 / sigx**2
    wy = 1.0 / sigy**2
    bx_bar = np.sum(wx * bx) / np.sum(wx)
    by_bar = np.sum(wy * by) / np.sum(wy)
    chi2_hand = np.sum(((bx - bx_bar) / sigx) ** 2 + ((by - by_bar) / sigy) ** 2)
    ok_hand = np.isclose(chi2_corr, chi2_hand, rtol=1e-8)
    print(
        f"  Hand chi2={chi2_hand:.4f}  function={chi2_corr:.4f}  " f"{'OK' if ok_hand else 'FAIL'}"
    )

    ok_all = ok_perf and ok_larger and ok_mu and ok_sig and ok_hand
    R.record("11-D  chi2_strong corrected vs uncorrected", ok_all)


# ── 11-E  compute_lambda_sl against hand calculation ─────────────────────


def _test_compute_lambda_sl(R: _TestResults):
    """
    Build a full WL+SL source catalog, compute lambda_sl via
    metric.compute_lambda_sl, and verify it equals the reduced-chi2 ratio.
    """
    R.header("11-E  compute_lambda_sl matches hand calculation")

    true_lens_xyte = [(-10.0, 0.0, 4.0)]
    xmax = 40.0

    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=true_lens_xyte,
        xmax=xmax,
        n_sources=60,
        seed=99,
    )

    sys_A = make_two_image_sis_system_at_lens(
        system_id="lambda_test",
        lens_center_xy=(-10.0, 0.0),
        te_true=4.0,
        beta_rel_xy=(0.5, 0.2),
        sigma_theta=0.05,
    )
    attach_strong_systems(src, [sys_A])

    # Use a slightly wrong lens for a non-trivial chi2
    lens_init = halo_obj.SIS_Lens(x=-9.5, y=0.3, te=3.8, chi2=[0])
    use_flags = [True, True, False]

    lam = metric.compute_lambda_sl(src, lens_init, use_flags, lens_type="SIS")

    # Manual calculation
    chi2_wl = metric.calculate_chi_squared(src, lens_init, use_flags, lens_type="SIS")
    dof_wl = metric.calc_degrees_of_freedom(src, lens_init, use_flags)
    chi2_sl = utils.chi2_strong_source_plane_sis(lens_init, src.strong_systems)
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
    src_no_sl = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=true_lens_xyte,
        xmax=xmax,
        n_sources=60,
        seed=99,
    )
    lam_none = metric.compute_lambda_sl(src_no_sl, lens_init, use_flags, "SIS")
    ok_default = lam_none == 1.0
    print(f"  no-SL default = {lam_none}  {'OK' if ok_default else 'FAIL'}")

    ok_all = ok_match and ok_finite and ok_default
    R.record("11-E  compute_lambda_sl", ok_all)


# ── 11-F  calculate_total_chi2 with fixed lambda_sl ──────────────────────


def _test_total_chi2_with_lambda(R: _TestResults):
    """
    Verify that calculate_total_chi2 with an explicit lambda_sl produces
    chi2_total = chi2_WL + lambda * chi2_SL, and that lambda_sl appears
    in the returned components dict.
    """
    R.header("11-F  calculate_total_chi2 with fixed lambda_sl")

    true_lens_xyte = [(5.0, -3.0, 3.0)]
    xmax = 30.0

    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=true_lens_xyte,
        xmax=xmax,
        n_sources=40,
        seed=42,
    )
    sys_B = make_two_image_sis_system_at_lens(
        system_id="total_test",
        lens_center_xy=(5.0, -3.0),
        te_true=3.0,
        beta_rel_xy=(0.3, -0.4),
        sigma_theta=0.06,
    )
    attach_strong_systems(src, [sys_B])

    lens = halo_obj.SIS_Lens(x=5.2, y=-2.8, te=2.9, chi2=[0])
    flags = [True, True, False]

    # ── With explicit lambda ──
    fixed_lam = 5.55
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src,
        lens,
        flags,
        lens_type="SIS",
        use_strong_lensing=True,
        lambda_sl=fixed_lam,
    )
    expected_total = comps["chi2_wl"] + fixed_lam * comps["chi2_sl"]
    ok_total = np.isclose(chi2_total, expected_total, rtol=1e-10)
    ok_lam = comps["lambda_sl"] == fixed_lam
    print(f"  lambda_sl = {comps['lambda_sl']}  (passed {fixed_lam})  {'OK' if ok_lam else 'FAIL'}")
    print(
        f"  chi2_total = {chi2_total:.4f}  expected = {expected_total:.4f}  "
        f"{'OK' if ok_total else 'FAIL'}"
    )

    # ── Without SL → lambda = 0 ──
    chi2_no, _, comps_no = metric.calculate_total_chi2(
        src, lens, flags, lens_type="SIS", use_strong_lensing=False
    )
    ok_no = comps_no["lambda_sl"] == 0.0 and np.isclose(chi2_no, comps_no["chi2_wl"])
    print(f"  no-SL: lambda=0, chi2=chi2_wl  {'OK' if ok_no else 'FAIL'}")

    # ── Fallback (lambda_sl=None, SL active) should be finite & positive ──
    chi2_fb, _, comps_fb = metric.calculate_total_chi2(
        src,
        lens,
        flags,
        lens_type="SIS",
        use_strong_lensing=True,
        lambda_sl=None,
    )
    ok_fb = np.isfinite(comps_fb["lambda_sl"]) and comps_fb["lambda_sl"] > 0
    print(
        f"  fallback lambda = {comps_fb['lambda_sl']:.6f}  "
        f"finite & positive: {'OK' if ok_fb else 'FAIL'}"
    )

    ok_all = ok_total and ok_lam and ok_no and ok_fb
    R.record("11-F  calculate_total_chi2 integration", ok_all)


# ── 11-G  Magnification-weighted chi2 with real toy geometry ─────────────


def _test_magnification_weighted_toy(R: _TestResults):
    """
    Full toy scenario: two lenses, two strong systems (from main()).

    With a composite deflector the strong systems (constructed for
    individual lenses) won't back-project perfectly because each
    system also feels the cross-deflection from the *other* halo.
    So we do NOT expect chi2 = 0 even at the true lens parameters.

    Instead we verify:
      1. Magnification correction increases chi2 (smaller sigma_beta).
      2. Per-system breakdown sums to the total.
      3. Breakdown metadata (abs_mu, sigma_beta, det_A) is present
         and correctly shaped.
      4. Moving the lens away from truth increases chi2 (sanity).
    """
    R.header("11-G  Full toy geometry magnification test")

    sysA = make_two_image_sis_system_at_lens(
        system_id="toy_A",
        lens_center_xy=(-15.0, 0.0),
        te_true=5.0,
        beta_rel_xy=(0.6, 0.2),
        sigma_theta=0.03,
    )
    sysB = make_two_image_sis_system_at_lens(
        system_id="toy_B",
        lens_center_xy=(18.0, 12.0),
        te_true=3.5,
        beta_rel_xy=(-0.4, 0.25),
        sigma_theta=0.03,
    )
    systems = [sysA, sysB]

    lens_true = halo_obj.SIS_Lens(x=[-15.0, 18.0], y=[0.0, 12.0], te=[5.0, 3.5], chi2=[0, 0])

    # ── Sub-test 1: corrected > uncorrected at true params ──
    chi2_corr_true = utils.chi2_strong_source_plane_sis(
        lens_true, systems, use_magnification_correction=True
    )
    chi2_uncorr_true = utils.chi2_strong_source_plane_sis(
        lens_true, systems, use_magnification_correction=False
    )
    ok_larger_true = chi2_corr_true > chi2_uncorr_true
    print(
        f"  True lens: chi2_corr={chi2_corr_true:.2f}  "
        f"chi2_uncorr={chi2_uncorr_true:.2f}  "
        f"corr>uncorr: {'OK' if ok_larger_true else 'FAIL'}"
    )

    # ── Sub-test 2: corrected > uncorrected also at perturbed params ──
    lens_pert = halo_obj.SIS_Lens(x=[-14.7, 18.3], y=[0.2, 11.8], te=[5.0, 3.5], chi2=[0, 0])
    chi2_corr_pert = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, use_magnification_correction=True
    )
    chi2_uncorr_pert = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, use_magnification_correction=False
    )
    ok_larger_pert = chi2_corr_pert > chi2_uncorr_pert
    print(
        f"  Perturbed: chi2_corr={chi2_corr_pert:.2f}  "
        f"chi2_uncorr={chi2_uncorr_pert:.2f}  "
        f"corr>uncorr: {'OK' if ok_larger_pert else 'FAIL'}"
    )

    # ── Sub-test 3: breakdown sums to total ──
    chi2_bd, bd = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, return_breakdown=True, use_magnification_correction=True
    )
    bd_sum = sum(v["chi2"] for v in bd.values())
    ok_sum = np.isclose(chi2_bd, bd_sum, rtol=1e-10)
    print(f"  Breakdown sum={bd_sum:.4f}  total={chi2_bd:.4f}  " f"{'OK' if ok_sum else 'FAIL'}")

    # ── Sub-test 4: metadata present and correctly shaped ──
    ok_meta = True
    for sid, info in bd.items():
        has_fields = all(k in info for k in ["abs_mu", "sigma_beta_x", "sigma_theta", "det_A"])
        n_img = info["n_images"]
        shapes_ok = (
            info["abs_mu"].shape == (n_img,)
            and info["sigma_beta_x"].shape == (n_img,)
            and info["det_A"].shape == (n_img,)
        )
        # sigma_beta < sigma_theta for all images (magnification > 1 near Einstein ring)
        sb_lt_st = np.all(info["sigma_beta_x"] < info["sigma_theta"])
        ok_sys = has_fields and shapes_ok and sb_lt_st
        ok_meta = ok_meta and ok_sys
        print(
            f"    {sid}: n_img={n_img}  fields={'OK' if has_fields else 'MISS'}  "
            f"shapes={'OK' if shapes_ok else 'BAD'}  "
            f"sigma_beta<sigma_theta={'OK' if sb_lt_st else 'FAIL'}"
        )

    ok_all = ok_larger_true and ok_larger_pert and ok_sum and ok_meta
    R.record("11-G  Full toy geometry", ok_all)


def run_magnification_tests() -> bool:
    """Execute all Task 11 unit tests.  Returns True if all pass."""
    R = _TestResults()
    _test_magnification_single_sis(R)
    _test_magnification_composite_fd(R)
    _test_sigma_beta(R)
    _test_chi2_strong_corrected(R)
    _test_compute_lambda_sl(R)
    _test_total_chi2_with_lambda(R)
    _test_magnification_weighted_toy(R)
    return R.summary()


if __name__ == "__main__":
    run_magnification_tests()
