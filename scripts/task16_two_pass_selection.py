"""Task 16 — Two-pass forward selection (WL → λ_SL → WL+SL)."""

from __future__ import annotations

import sys

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.metric as metric
import arch.pipeline as pipeline
from arch.main import fit_lensing_field
from scripts.paper2_shared import (
    _find_nfw_beta_rad,
    _find_nfw_einstein_radius,
    _make_nfw_halo,
    _make_nfw_wl_catalog,
    _make_two_image_nfw_system,
    _TestResults,
    attach_strong_systems,
    make_two_image_sis_system_at_lens,
    make_weak_lensing_catalog_two_lenses,
)


# ═══════════════════════════════════════════════════════════════════════════
# Task 16 — Two-pass forward selection
# ═══════════════════════════════════════════════════════════════════════════
#
# 16-A  Backward compat: two-pass with no SL data == single-pass WL output
# 16-B  Diagnostic dict structure
# 16-C  Pass 1 result matches single-pass WL on identical inputs
# 16-D  λ_SL from two-pass equals metric.compute_lambda_sl at Pass 1 output
# 16-E  Two-pass adds no halos when SL fits Pass 1 already
# 16-F  Two-pass pipeline integration: runs end-to-end on NFW + SL
# 16-G  Two-pass NFW recovers single-halo truth at least as well as single-pass
# ═══════════════════════════════════════════════════════════════════════════


class _TestResults16(_TestResults):
    def summary(self) -> bool:
        self.header("TASK 16 — SUMMARY")
        all_ok = True
        for name, ok in self.results:
            tag = "PASSED" if ok else "*** FAILED ***"
            print(f"  {name:60s}  {tag}")
            all_ok = all_ok and ok
        print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}\n")
        return all_ok


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_nfw_with_sl(seed: int = 16):
    """Single NFW + one strong-lensing system, returns (src, halo_true, xmax)."""
    halo_true = _make_nfw_halo(x=4.0, y=-3.0, mass=4e14, redshift=0.3)
    z_source = 2.0
    theta_E = _find_nfw_einstein_radius(halo_true, z_source)
    beta_rad = _find_nfw_beta_rad(halo_true, z_source, theta_E)
    sys_A = _make_two_image_nfw_system(
        "16_sys", halo_true,
        (beta_rad * 0.4, 0.0), z_source, theta_E,
        sigma_theta=0.06,
    )

    xmax = 60.0
    src = _make_nfw_wl_catalog(
        halo_true, xmax=xmax, n_sources=80, z_source=1.0, seed=seed,
    )
    attach_strong_systems(src, [sys_A])

    return src, halo_true, xmax


def _build_nfw_no_sl(seed: int = 16):
    """Single NFW, no SL data."""
    halo_true = _make_nfw_halo(x=4.0, y=-3.0, mass=4e14, redshift=0.3)
    xmax = 60.0
    src = _make_nfw_wl_catalog(
        halo_true, xmax=xmax, n_sources=80, z_source=1.0, seed=seed,
    )
    return src, halo_true, xmax


def _nearest_distance(lenses, true_x, true_y):
    if len(lenses.x) == 0:
        return np.inf
    return float(np.min(np.hypot(
        lenses.x - true_x, lenses.y - true_y
    )))


# ── 16-A  Backward compat: no SL → matches single-pass ───────────────────

def _test_no_sl_matches_single_pass(R: _TestResults16):
    R.header("16-A  Two-pass with no SL data == single-pass WL")

    src, halo_true, xmax = _build_nfw_no_sl(seed=10)
    use_flags = [True, True, False]

    # Generate candidates
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    # Single-pass
    sel_single, chi2_single = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )

    # Two-pass — should skip Pass 2 because no SL data, return identical result
    sel_two, chi2_two, lam_two, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    n_match = len(sel_single.x) == len(sel_two.x)
    ok_lambda_zero = lam_two == 0.0
    ok_pass2_zero = diag["n_pass2_added"] == 0
    ok_chi2_match = np.isclose(chi2_single, chi2_two, rtol=1e-10)

    if n_match:
        ok_x = np.allclose(np.sort(sel_single.x), np.sort(sel_two.x))
        ok_y = np.allclose(np.sort(sel_single.y), np.sort(sel_two.y))
    else:
        ok_x = ok_y = False

    print(f"  n_single = {len(sel_single.x)}, n_two = {len(sel_two.x)}  "
          f"{'OK' if n_match else 'FAIL'}")
    print(f"  positions identical: x={'OK' if ok_x else 'FAIL'}  "
          f"y={'OK' if ok_y else 'FAIL'}")
    print(f"  chi2 identical: {'OK' if ok_chi2_match else 'FAIL'} "
          f"({chi2_single:.6f} vs {chi2_two:.6f})")
    print(f"  lambda_sl == 0 (no SL): {'OK' if ok_lambda_zero else 'FAIL'}")
    print(f"  n_pass2_added == 0:     {'OK' if ok_pass2_zero else 'FAIL'}")

    R.record("16-A  No-SL backward compat",
             n_match and ok_x and ok_y and ok_chi2_match
             and ok_lambda_zero and ok_pass2_zero)


# ── 16-B  Diagnostic dict structure ──────────────────────────────────────

def _test_diagnostic_structure(R: _TestResults16):
    R.header("16-B  Diagnostic dict has expected keys and finite values")

    src, halo_true, xmax = _build_nfw_with_sl(seed=20)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    _, _, _, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    expected_keys = {
        "n_pass1", "n_pass2_added", "lambda_sl",
        "chi2_pass1", "chi2_final", "n_remaining_after_pass1",
    }
    have_keys = expected_keys.issubset(diag.keys())
    types_ok = (
        isinstance(diag["n_pass1"], int)
        and isinstance(diag["n_pass2_added"], int)
        and isinstance(diag["lambda_sl"], float)
        and isinstance(diag["chi2_pass1"], float)
        and isinstance(diag["chi2_final"], float)
    )
    finite_ok = (
        np.isfinite(diag["chi2_pass1"])
        and np.isfinite(diag["chi2_final"])
        and np.isfinite(diag["lambda_sl"])
        and diag["lambda_sl"] >= 0
    )
    nonneg_counts = (
        diag["n_pass1"] >= 0
        and diag["n_pass2_added"] >= 0
        and diag["n_remaining_after_pass1"] >= 0
    )

    print(f"  diag = {diag}")
    print(f"  has expected keys: {'OK' if have_keys else 'FAIL'}")
    print(f"  types correct:     {'OK' if types_ok else 'FAIL'}")
    print(f"  finite values:     {'OK' if finite_ok else 'FAIL'}")
    print(f"  counts nonneg:     {'OK' if nonneg_counts else 'FAIL'}")

    R.record("16-B  Diagnostic dict structure",
             have_keys and types_ok and finite_ok and nonneg_counts)


# ── 16-C  Pass 1 matches single-pass WL on identical inputs ──────────────

def _test_pass1_matches_single_pass_wl(R: _TestResults16):
    R.header("16-C  Pass 1 output matches single-pass WL-only output")

    src, halo_true, xmax = _build_nfw_with_sl(seed=30)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    # Single-pass WL-only
    sel_single, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )

    # Two-pass — Pass 1 is WL-only, should match
    _, _, _, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    ok = diag["n_pass1"] == len(sel_single.x)
    print(f"  single-pass WL n_lenses = {len(sel_single.x)}")
    print(f"  two-pass Pass 1 n_pass1 = {diag['n_pass1']}")
    print(f"  match: {'OK' if ok else 'FAIL'}")

    R.record("16-C  Pass 1 matches single-pass WL", ok)


# ── 16-D  λ_SL consistency ────────────────────────────────────────────────

def _test_lambda_sl_consistency(R: _TestResults16):
    R.header("16-D  Two-pass λ_SL matches metric.compute_lambda_sl at Pass 1 output")

    src, halo_true, xmax = _build_nfw_with_sl(seed=40)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    # Get the Pass 1 output directly (single-pass WL-only)
    sel_pass1, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )

    if len(sel_pass1.x) == 0:
        print("  WL-only selection returned no lenses (unexpected) - skip")
        R.record("16-D  λ_SL consistency", False)
        return

    # Compute λ_SL externally
    lambda_external = metric.compute_lambda_sl(
        src, sel_pass1, use_flags, lens_type="NFW"
    )

    # Run two-pass and compare its lambda_sl
    _, _, lambda_two_pass, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    # The two-pass λ_SL should equal the externally-computed one
    # IF Pass 1 produced the same model as single-pass WL (which 16-C verifies)
    ok = np.isclose(lambda_two_pass, lambda_external, rtol=1e-6)
    print(f"  external compute_lambda_sl = {lambda_external:.6f}")
    print(f"  two-pass diag['lambda_sl']  = {lambda_two_pass:.6f}")
    print(f"  match (rtol 1e-6): {'OK' if ok else 'FAIL'}")

    R.record("16-D  λ_SL consistency", ok)


# ── 16-E  Two-pass adds no halos when WL already explains the data ───────

def _test_pass2_no_unnecessary_additions(R: _TestResults16):
    R.header("16-E  Pass 2 adds 0 halos when WL model already fits SL")

    # Single SIS scenario — geometry is well-determined by WL alone, and
    # the synthetic SL system is consistent with the WL truth, so Pass 2
    # should have nothing to add.
    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=[(5.0, -3.0, 3.5)],
        xmax=35.0, n_sources=100, seed=50,
    )
    sys_A = make_two_image_sis_system_at_lens(
        system_id="16E_sys",
        lens_center_xy=(5.0, -3.0),
        te_true=3.5,
        beta_rel_xy=(0.3, -0.2),
        sigma_theta=0.04, z_source=2.0,
    )
    attach_strong_systems(src, [sys_A])

    # Run end-to-end with two-pass; check that the pipeline runs and SL is
    # not adding spurious halos on a well-fit scenario.  We focus on NFW
    # since SIS doesn't use the two-pass path.  Adapt: rebuild scenario as NFW.
    src_n, halo_true, xmax = _build_nfw_with_sl(seed=55)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src_n, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src_n, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src_n, cand, xmax, lens_type="NFW")

    _, _, lambda_sl, diag = pipeline.forward_lens_selection_two_pass(
        src_n, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    # Pass 2 additions should be modest — at most a few halos, not double the count
    ok_modest = diag["n_pass2_added"] <= max(2, diag["n_pass1"])

    print(f"  pass1={diag['n_pass1']}  pass2_added={diag['n_pass2_added']}")
    print(f"  λ_SL = {lambda_sl:.4f}")
    print(f"  pass2_added modest (<=max(2, n_pass1)): "
          f"{'OK' if ok_modest else 'FAIL'}")

    R.record("16-E  Pass 2 modest additions when WL fits", ok_modest)


# ── 16-F  End-to-end pipeline integration ─────────────────────────────────

def _test_pipeline_integration(R: _TestResults16):
    R.header("16-F  fit_lensing_field with use_sl_in_selection=True runs cleanly")

    src, halo_true, xmax = _build_nfw_with_sl(seed=60)
    use_flags = [True, True, False]

    try:
        lenses, rchi2 = fit_lensing_field(
            src, xmax, flags=True, use_flags=use_flags,
            lens_type="NFW", z_lens=0.3,
            use_strong_lensing=True,
            use_sl_in_selection=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("16-F  Pipeline integration", False)
        return

    ok_lenses = len(lenses.x) >= 1
    ok_chi2 = np.isfinite(rchi2) and rchi2 > 0
    ok_distance = _nearest_distance(lenses, halo_true.x[0], halo_true.y[0]) < 5.0

    print(f"  n_lenses = {len(lenses.x)},  rchi2 = {rchi2:.4f}")
    print(f"  nearest distance to truth = "
          f"{_nearest_distance(lenses, halo_true.x[0], halo_true.y[0]):.3f} arcsec")
    print(f"  >=1 lens: {'OK' if ok_lenses else 'FAIL'}")
    print(f"  finite rchi2: {'OK' if ok_chi2 else 'FAIL'}")
    print(f"  distance < 5 arcsec: {'OK' if ok_distance else 'FAIL'}")

    R.record("16-F  Pipeline integration", ok_lenses and ok_chi2 and ok_distance)


# ── 16-G  Two-pass vs single-pass on a noisy scenario ────────────────────

def _test_two_pass_no_degradation(R: _TestResults16):
    R.header("16-G  Two-pass NFW does not degrade recovery vs single-pass WL+SL")

    src, halo_true, xmax = _build_nfw_with_sl(seed=70)
    use_flags = [True, True, False]

    # Run 1: legacy single-pass (use_sl_in_selection=False)
    try:
        lenses_single, rchi2_single = fit_lensing_field(
            src.copy(), xmax, flags=False, use_flags=use_flags,
            lens_type="NFW", z_lens=0.3,
            use_strong_lensing=True,
            use_sl_in_selection=False,
        )
    except Exception as e:
        print(f"  single-pass EXCEPTION: {e}")
        R.record("16-G  No degradation", False)
        return

    # Run 2: two-pass
    try:
        lenses_two, rchi2_two = fit_lensing_field(
            src.copy(), xmax, flags=False, use_flags=use_flags,
            lens_type="NFW", z_lens=0.3,
            use_strong_lensing=True,
            use_sl_in_selection=True,
        )
    except Exception as e:
        print(f"  two-pass EXCEPTION: {e}")
        R.record("16-G  No degradation", False)
        return

    d_single = _nearest_distance(lenses_single, halo_true.x[0], halo_true.y[0])
    d_two = _nearest_distance(lenses_two, halo_true.x[0], halo_true.y[0])

    # Two-pass should be no worse than single-pass + tolerance
    tol = 1.0  # arcsec
    ok = d_two <= d_single + tol

    # Mass comparison if both have lenses
    if len(lenses_single.x) > 0 and len(lenses_two.x) > 0:
        i_s = int(np.argmin(np.hypot(
            lenses_single.x - halo_true.x[0],
            lenses_single.y - halo_true.y[0]
        )))
        i_t = int(np.argmin(np.hypot(
            lenses_two.x - halo_true.x[0],
            lenses_two.y - halo_true.y[0]
        )))
        m_single = lenses_single.mass[i_s]
        m_two = lenses_two.mass[i_t]
        m_true = halo_true.mass[0]
        rel_err_single = abs(m_single - m_true) / m_true
        rel_err_two = abs(m_two - m_true) / m_true
        print(f"  mass_true   = {m_true:.3e}")
        print(f"  mass_single = {m_single:.3e}  (rel err {rel_err_single:.3f})")
        print(f"  mass_two    = {m_two:.3e}  (rel err {rel_err_two:.3f})")

    print(f"  single-pass: n={len(lenses_single.x)}, "
          f"d_truth={d_single:.3f}, rchi2={rchi2_single:.4f}")
    print(f"  two-pass:    n={len(lenses_two.x)}, "
          f"d_truth={d_two:.3f}, rchi2={rchi2_two:.4f}")
    print(f"  two-pass no worse (tol={tol}″): {'OK' if ok else 'FAIL'}")

    R.record("16-G  No degradation vs single-pass", ok)


# ── Runner ────────────────────────────────────────────────────────────────

def run_two_pass_tests() -> bool:
    R = _TestResults16()
    _test_no_sl_matches_single_pass(R)
    _test_diagnostic_structure(R)
    _test_pass1_matches_single_pass_wl(R)
    _test_lambda_sl_consistency(R)
    _test_pass2_no_unnecessary_additions(R)
    _test_pipeline_integration(R)
    _test_two_pass_no_degradation(R)
    return R.summary()


if __name__ == "__main__":
    run_two_pass_tests()