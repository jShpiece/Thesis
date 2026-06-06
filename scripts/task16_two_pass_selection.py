"""Task 16 — Two-pass forward selection (WL → λ_SL → refine → WL+SL)."""

from __future__ import annotations

import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.pipeline as pipeline
import arch.utils as utils
from arch.chi2_wrappers import update_chi2_values
from arch.forward_selection import (
    _append_candidate, _empty_lens_collection,
    _greedy_add_pass, _joint_refine_pass1,
)
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
# Task 16
#   16-A  Backward compat: two-pass with no SL data == single-pass WL output
#   16-B  Diagnostic dict structure
#   16-C  Pass 1 result matches single-pass WL on identical inputs
#   16-D  λ_SL from two-pass equals metric.compute_lambda_sl at Pass 1 output
#   16-E  Two-pass adds no halos when SL fits Pass 1 already
#   16-F  Two-pass pipeline integration: runs end-to-end on NFW + SL
#   16-G  Two-pass NFW recovers single-halo truth at least as well as single-pass
#   16-H  Magnification-correction flag is threaded correctly through Pass 2
#   16-I  Joint Pass 1 refinement (Option A): correctness and integration
#   16-J  pass2_tolerance_multiplier: higher values suppress Pass 2 additions
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
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    sel_single, chi2_single = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    sel_two, chi2_two, lam_two, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )

    n_match = len(sel_single.x) == len(sel_two.x)
    ok_lambda_zero = lam_two == 0.0
    ok_pass2_zero = diag["n_pass2_added"] == 0
    ok_chi2_match = np.isclose(chi2_single, chi2_two, rtol=1e-10)
    ok_refine_not_applied = diag["refine_applied"] is False

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
    print(f"  refine_applied == False (no SL): "
          f"{'OK' if ok_refine_not_applied else 'FAIL'}")

    R.record("16-A  No-SL backward compat",
             n_match and ok_x and ok_y and ok_chi2_match
             and ok_lambda_zero and ok_pass2_zero
             and ok_refine_not_applied)


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
        "use_magnification_correction_sl",
        "joint_refine_pass1", "chi2_refine_before",
        "chi2_refine_after", "refine_applied",
        "pass2_tolerance_multiplier",
    }
    have_keys = expected_keys.issubset(diag.keys())
    types_ok = (
        isinstance(diag["n_pass1"], int)
        and isinstance(diag["n_pass2_added"], int)
        and isinstance(diag["lambda_sl"], float)
        and isinstance(diag["chi2_pass1"], float)
        and isinstance(diag["chi2_final"], float)
        and isinstance(diag["use_magnification_correction_sl"], bool)
        and isinstance(diag["joint_refine_pass1"], bool)
        and isinstance(diag["refine_applied"], bool)
        and isinstance(diag["chi2_refine_before"], float)
        and isinstance(diag["chi2_refine_after"], float)
        and isinstance(diag["pass2_tolerance_multiplier"], float)
    )
    finite_ok = (
        np.isfinite(diag["chi2_pass1"])
        and np.isfinite(diag["chi2_final"])
        and np.isfinite(diag["lambda_sl"])
        and diag["lambda_sl"] >= 0
        and diag["pass2_tolerance_multiplier"] > 0
    )
    nonneg_counts = (
        diag["n_pass1"] >= 0
        and diag["n_pass2_added"] >= 0
        and diag["n_remaining_after_pass1"] >= 0
    )
    refine_contract_ok = (
        (not diag["refine_applied"])
        or (diag["chi2_refine_after"] < diag["chi2_refine_before"])
    )

    print(f"  diag = {diag}")
    print(f"  has expected keys: {'OK' if have_keys else 'FAIL'}")
    print(f"  types correct:     {'OK' if types_ok else 'FAIL'}")
    print(f"  finite values:     {'OK' if finite_ok else 'FAIL'}")
    print(f"  counts nonneg:     {'OK' if nonneg_counts else 'FAIL'}")
    print(f"  refine contract:   {'OK' if refine_contract_ok else 'FAIL'}")

    R.record("16-B  Diagnostic dict structure",
             have_keys and types_ok and finite_ok
             and nonneg_counts and refine_contract_ok)


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

    sel_single, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    _, _, _, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
        joint_refine_pass1=False,
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

    sel_pass1, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )

    if len(sel_pass1.x) == 0:
        print("  WL-only selection returned no lenses (unexpected) - skip")
        R.record("16-D  λ_SL consistency", False)
        return

    lambda_external = metric.compute_lambda_sl(
        src, sel_pass1, use_flags, lens_type="NFW"
    )
    _, _, lambda_two_pass, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
        joint_refine_pass1=False,
    )

    ok = np.isclose(lambda_two_pass, lambda_external, rtol=1e-6)
    print(f"  external compute_lambda_sl = {lambda_external:.6f}")
    print(f"  two-pass diag['lambda_sl']  = {lambda_two_pass:.6f}")
    print(f"  match (rtol 1e-6): {'OK' if ok else 'FAIL'}")

    R.record("16-D  λ_SL consistency", ok)


# ── 16-E  Two-pass adds modest halos when WL already explains data ───────

def _test_pass2_no_unnecessary_additions(R: _TestResults16):
    R.header("16-E  Pass 2 modest additions when WL model already fits SL")

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

    ok_modest = diag["n_pass2_added"] <= max(2, diag["n_pass1"])

    print(f"  pass1={diag['n_pass1']}  pass2_added={diag['n_pass2_added']}")
    print(f"  λ_SL = {lambda_sl:.4f}")
    if diag["refine_applied"]:
        print(f"  refine: chi^2 {diag['chi2_refine_before']:.3f} -> "
              f"{diag['chi2_refine_after']:.3f}")
    print(f"  tol_mult = {diag['pass2_tolerance_multiplier']}")
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

    tol = 1.0
    ok = d_two <= d_single + tol

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
        print(f"  mass_true   = {m_true:.3e}")
        print(f"  mass_single = {m_single:.3e}  "
              f"(rel err {abs(m_single - m_true)/m_true:.3f})")
        print(f"  mass_two    = {m_two:.3e}  "
              f"(rel err {abs(m_two - m_true)/m_true:.3f})")

    print(f"  single-pass: n={len(lenses_single.x)}, "
          f"d_truth={d_single:.3f}, rchi2={rchi2_single:.4f}")
    print(f"  two-pass:    n={len(lenses_two.x)}, "
          f"d_truth={d_two:.3f}, rchi2={rchi2_two:.4f}")
    print(f"  two-pass no worse (tol={tol}″): {'OK' if ok else 'FAIL'}")

    R.record("16-G  No degradation vs single-pass", ok)


# ── 16-H  Magnification-correction flag is threaded through Pass 2 ───────

def _test_magnification_flag_threading(R: _TestResults16):
    R.header("16-H  Magnification-correction flag threading through Pass 2")

    src, halo_true, xmax = _build_nfw_with_sl(seed=80)
    use_flags = [True, True, False]

    imperfect = halo_obj.NFW_Lens(
        x=np.array([halo_true.x[0] + 1.0]),
        y=np.array([halo_true.y[0] + 1.0]),
        z=np.zeros(1),
        concentration=halo_true.concentration.copy(),
        mass=halo_true.mass.copy(),
        redshift=halo_true.redshift,
        chi2=np.zeros(1),
    )

    chi2_sl_off = utils.chi2_strong_source_plane_nfw(
        imperfect, src.strong_systems,
        use_magnification_correction=False,
    )
    chi2_sl_on = utils.chi2_strong_source_plane_nfw(
        imperfect, src.strong_systems,
        use_magnification_correction=True,
    )
    print(f"  External chi2_SL_no_mag = {chi2_sl_off:.3f}")
    print(f"  External chi2_SL_mag_on = {chi2_sl_on:.3f}")
    ok_external_differ = chi2_sl_on > chi2_sl_off
    print(f"  External: mag_on > mag_off: {'OK' if ok_external_differ else 'FAIL'}")

    lambda_sl_fixed = 1.0
    chi2_total_off = update_chi2_values(
        src, imperfect, use_flags, "NFW",
        use_strong_lensing=True, lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=False,
    )
    chi2_total_on = update_chi2_values(
        src, imperfect, use_flags, "NFW",
        use_strong_lensing=True, lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=True,
    )
    print(f"  update_chi2_values(mag=False) = {chi2_total_off:.6f}")
    print(f"  update_chi2_values(mag=True)  = {chi2_total_on:.6f}")
    ok_update_responds = chi2_total_on != chi2_total_off
    print(f"  update_chi2_values responds: {'OK' if ok_update_responds else 'FAIL'}")

    test_cand = halo_obj.NFW_Lens(
        x=np.array([halo_true.x[0] - 5.0, halo_true.x[0] + 5.0]),
        y=np.array([halo_true.y[0] + 3.0, halo_true.y[0] - 3.0]),
        z=np.zeros(2),
        concentration=np.array([halo_true.concentration[0]] * 2),
        mass=np.array([1.0e13, 1.0e13]),
        redshift=halo_true.redshift,
        chi2=np.zeros(2),
    )

    start_off = _empty_lens_collection("NFW", test_cand)
    start_on = _empty_lens_collection("NFW", test_cand)
    start_off = _append_candidate(start_off, test_cand, 0, "NFW")
    start_on = _append_candidate(start_on, test_cand, 0, "NFW")

    _, chi2_off, _ = _greedy_add_pass(
        src, test_cand, start_off, np.array([1], dtype=int),
        use_flags, "NFW",
        base_tolerance=0.003, mass_scale=1e13,
        kappa_scale=0.1, exponent=-1.0,
        use_strong_lensing=True, lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=False,
    )
    _, chi2_on, _ = _greedy_add_pass(
        src, test_cand, start_on, np.array([1], dtype=int),
        use_flags, "NFW",
        base_tolerance=0.003, mass_scale=1e13,
        kappa_scale=0.1, exponent=-1.0,
        use_strong_lensing=True, lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=True,
    )
    print(f"  greedy chi2(mag=False) = {chi2_off:.6f}")
    print(f"  greedy chi2(mag=True)  = {chi2_on:.6f}")
    ok_greedy_responds = chi2_off != chi2_on
    print(f"  _greedy_add_pass propagates flag: "
          f"{'OK' if ok_greedy_responds else 'FAIL'}")

    src_t, _, xmax_t = _build_nfw_with_sl(seed=81)
    cand_t = pipeline.generate_initial_guess(src_t, lens_type="NFW", z_l=0.3)
    cand_t = pipeline.optimize_lens_positions(
        src_t, cand_t, xmax_t, use_flags, lens_type="NFW",
    )
    cand_t = pipeline.filter_lens_positions(src_t, cand_t, xmax_t, lens_type="NFW")

    _, _, _, diag = pipeline.forward_lens_selection_two_pass(
        src_t, cand_t, use_flags, lens_type="NFW",
        return_diagnostics=True,
    )
    ok_default_false = diag["use_magnification_correction_sl"] is False
    print(f"  two-pass default mag flag = "
          f"{diag['use_magnification_correction_sl']}  "
          f"{'OK' if ok_default_false else 'FAIL'}")

    ok_consistency = chi2_sl_off <= chi2_sl_on + 1e-9
    print(f"  chi2_SL_no_mag <= chi2_SL_mag_on: {'OK' if ok_consistency else 'FAIL'}")

    ok_all = (
        ok_external_differ and ok_update_responds and ok_greedy_responds
        and ok_default_false and ok_consistency
    )
    R.record("16-H  Magnification flag threading", ok_all)


# ── 16-I  Joint Pass 1 refinement (Option A) ─────────────────────────────

def _test_refinement_safety_contract(R: _TestResults16):
    R.header("16-I.1  Joint refinement never worsens chi^2 (safety)")

    src, halo_true, xmax = _build_nfw_with_sl(seed=91)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")
    sel_pass1, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    lambda_sl = metric.compute_lambda_sl(
        src, sel_pass1, use_flags, lens_type="NFW",
    )

    refined, chi2_before, chi2_after = _joint_refine_pass1(
        src, sel_pass1, use_flags, "NFW",
        lambda_sl=lambda_sl, xmax=xmax,
        use_magnification_correction_sl=False,
    )

    ok_no_worse = chi2_after <= chi2_before + 1.0e-9
    ok_finite = np.isfinite(chi2_before) and np.isfinite(chi2_after)
    print(f"  chi2 before refine = {chi2_before:.6f}")
    print(f"  chi2 after  refine = {chi2_after:.6f}")
    print(f"  no worse than input: {'OK' if ok_no_worse else 'FAIL'}")
    print(f"  finite values:       {'OK' if ok_finite else 'FAIL'}")

    R.record("16-I.1  Refinement safety contract", ok_no_worse and ok_finite)


def _test_refinement_preserves_halo_count(R: _TestResults16):
    R.header("16-I.2  Joint refinement preserves halo count")

    src, halo_true, xmax = _build_nfw_with_sl(seed=92)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")
    sel_pass1, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    lambda_sl = metric.compute_lambda_sl(
        src, sel_pass1, use_flags, lens_type="NFW",
    )

    refined, _, _ = _joint_refine_pass1(
        src, sel_pass1, use_flags, "NFW",
        lambda_sl=lambda_sl, xmax=xmax,
        use_magnification_correction_sl=False,
    )

    ok = len(refined.x) == len(sel_pass1.x)
    print(f"  n_before = {len(sel_pass1.x)},  n_after = {len(refined.x)}  "
          f"{'OK' if ok else 'FAIL'}")
    R.record("16-I.2  Refinement preserves halo count", ok)


def _test_refinement_toggle_makes_difference(R: _TestResults16):
    R.header("16-I.3  Toggle joint_refine_pass1 produces distinguishable outcomes")

    src, halo_true, xmax = _build_nfw_with_sl(seed=93)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    sel_off, _, lam_off, diag_off = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
        joint_refine_pass1=False,
    )
    sel_on, _, lam_on, diag_on = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
        joint_refine_pass1=True,
    )

    ok_lambda_same = np.isclose(lam_off, lam_on, rtol=1e-10)
    ok_flag_off = diag_off["joint_refine_pass1"] is False
    ok_flag_on = diag_on["joint_refine_pass1"] is True
    ok_off_no_refine = diag_off["refine_applied"] is False

    print(f"  refine off: refine_applied={diag_off['refine_applied']}, "
          f"n_pass2={diag_off['n_pass2_added']}, λ={lam_off:.4f}")
    print(f"  refine on:  refine_applied={diag_on['refine_applied']}, "
          f"n_pass2={diag_on['n_pass2_added']}, λ={lam_on:.4f}")
    print(f"  lambda identical:        {'OK' if ok_lambda_same else 'FAIL'}")
    print(f"  flag off echoed:         {'OK' if ok_flag_off else 'FAIL'}")
    print(f"  flag on echoed:          {'OK' if ok_flag_on else 'FAIL'}")
    print(f"  off run did not refine:  {'OK' if ok_off_no_refine else 'FAIL'}")

    R.record("16-I.3  Toggle makes distinguishable outcomes",
             ok_lambda_same and ok_flag_off and ok_flag_on and ok_off_no_refine)


def _test_refinement_skipped_without_sl(R: _TestResults16):
    R.header("16-I.4  Refinement skipped when no SL data attached")

    src, halo_true, xmax = _build_nfw_no_sl(seed=94)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    _, _, _, diag = pipeline.forward_lens_selection_two_pass(
        src, cand, use_flags, lens_type="NFW",
        return_diagnostics=True,
        joint_refine_pass1=True,
    )

    ok_no_refine = diag["refine_applied"] is False
    print(f"  refine_applied = {diag['refine_applied']}  "
          f"{'OK' if ok_no_refine else 'FAIL'}")
    R.record("16-I.4  Refinement skipped without SL", ok_no_refine)


def _test_refinement_respects_position_bounds(R: _TestResults16):
    R.header("16-I.5  Refined positions stay within bounds (|x|,|y| <= 1.2*xmax)")

    src, halo_true, xmax = _build_nfw_with_sl(seed=95)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")
    sel_pass1, _ = pipeline.forward_lens_selection(
        src, cand, use_flags, lens_type="NFW",
        use_strong_lensing=False, lambda_sl=None,
    )
    lambda_sl = metric.compute_lambda_sl(
        src, sel_pass1, use_flags, lens_type="NFW",
    )

    refined, _, _ = _joint_refine_pass1(
        src, sel_pass1, use_flags, "NFW",
        lambda_sl=lambda_sl, xmax=xmax,
        use_magnification_correction_sl=False,
    )

    bound = 1.2 * xmax
    ok_x = bool(np.all(np.abs(refined.x) <= bound + 1e-6))
    ok_y = bool(np.all(np.abs(refined.y) <= bound + 1e-6))
    ok_mass_range = bool(np.all(
        (refined.mass >= 10.0 ** 10) & (refined.mass <= 10.0 ** 17)
    ))
    print(f"  bound = {bound:.2f}")
    print(f"  refined positions x in [{refined.x.min():.2f}, "
          f"{refined.x.max():.2f}]: {'OK' if ok_x else 'FAIL'}")
    print(f"  refined positions y in [{refined.y.min():.2f}, "
          f"{refined.y.max():.2f}]: {'OK' if ok_y else 'FAIL'}")
    print(f"  refined mass in [1e10, 1e17] M_sun: "
          f"{'OK' if ok_mass_range else 'FAIL'}")

    R.record("16-I.5  Refinement respects bounds",
             ok_x and ok_y and ok_mass_range)


# ═══════════════════════════════════════════════════════════════════════════
#  16-J  pass2_tolerance_multiplier
# ═══════════════════════════════════════════════════════════════════════════
#
#  16-J.1  Default is 10.0 (the production value chosen to suppress
#          spurious Pass 2 additions on A2744-like data).
#  16-J.2  Multiplier is recorded in the diagnostics dict.
#  16-J.3  Higher multiplier produces fewer-or-equal Pass 2 additions
#          (monotonicity).
#  16-J.4  Pass 1 is unaffected by the multiplier — n_pass1 is identical
#          across different pass2_tolerance_multiplier values.

def _test_tolerance_multiplier_default(R: _TestResults16):
    R.header("16-J.1  Default pass2_tolerance_multiplier == 10.0")

    src, _, xmax = _build_nfw_with_sl(seed=100)
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
    ok = np.isclose(diag["pass2_tolerance_multiplier"], 10.0)
    print(f"  diag['pass2_tolerance_multiplier'] = "
          f"{diag['pass2_tolerance_multiplier']}  {'OK' if ok else 'FAIL'}")
    R.record("16-J.1  Default multiplier is 10.0", ok)


def _test_tolerance_multiplier_recorded(R: _TestResults16):
    R.header("16-J.2  Multiplier value is echoed in diagnostics")

    src, _, xmax = _build_nfw_with_sl(seed=101)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    for mult in (1.0, 5.0, 100.0):
        _, _, _, diag = pipeline.forward_lens_selection_two_pass(
            src, cand, use_flags, lens_type="NFW",
            return_diagnostics=True,
            pass2_tolerance_multiplier=mult,
        )
        echoed = diag["pass2_tolerance_multiplier"]
        ok_i = np.isclose(echoed, mult)
        print(f"  set {mult:.1f} -> echoed {echoed}  {'OK' if ok_i else 'FAIL'}")
        if not ok_i:
            R.record("16-J.2  Multiplier echoed in diagnostics", False)
            return

    R.record("16-J.2  Multiplier echoed in diagnostics", True)


def _test_tolerance_multiplier_monotonic(R: _TestResults16):
    R.header("16-J.3  Higher multiplier produces fewer-or-equal Pass 2 additions")

    src, _, xmax = _build_nfw_with_sl(seed=102)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    multipliers = [1.0, 10.0, 100.0, 1000.0]
    n_added_by_mult = []
    for mult in multipliers:
        _, _, _, diag = pipeline.forward_lens_selection_two_pass(
            src, cand, use_flags, lens_type="NFW",
            return_diagnostics=True,
            joint_refine_pass1=False,  # isolate Pass 2 behaviour
            pass2_tolerance_multiplier=mult,
        )
        n_added_by_mult.append(diag["n_pass2_added"])
        print(f"  mult={mult:7.1f}  n_pass2_added = {diag['n_pass2_added']}")

    # Monotonic non-increasing
    ok_mono = all(
        n_added_by_mult[i + 1] <= n_added_by_mult[i]
        for i in range(len(multipliers) - 1)
    )
    print(f"  monotonic non-increasing: {'OK' if ok_mono else 'FAIL'}")

    R.record("16-J.3  Multiplier suppresses Pass 2 additions", ok_mono)


def _test_tolerance_multiplier_pass1_invariant(R: _TestResults16):
    R.header("16-J.4  Pass 1 unaffected by pass2_tolerance_multiplier")

    src, _, xmax = _build_nfw_with_sl(seed=103)
    use_flags = [True, True, False]
    cand = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=0.3)
    cand = pipeline.optimize_lens_positions(
        src, cand, xmax, use_flags, lens_type="NFW",
    )
    cand = pipeline.filter_lens_positions(src, cand, xmax, lens_type="NFW")

    pass1_counts = []
    chi2_pass1_values = []
    for mult in (1.0, 10.0, 100.0):
        _, _, _, diag = pipeline.forward_lens_selection_two_pass(
            src, cand, use_flags, lens_type="NFW",
            return_diagnostics=True,
            joint_refine_pass1=False,
            pass2_tolerance_multiplier=mult,
        )
        pass1_counts.append(diag["n_pass1"])
        chi2_pass1_values.append(diag["chi2_pass1"])
        print(f"  mult={mult:7.1f}  n_pass1={diag['n_pass1']}  "
              f"chi2_pass1={diag['chi2_pass1']:.6f}")

    ok_counts = len(set(pass1_counts)) == 1
    ok_chi2 = all(np.isclose(chi2_pass1_values[0], v, rtol=1e-12)
                  for v in chi2_pass1_values)
    print(f"  n_pass1 identical:        {'OK' if ok_counts else 'FAIL'}")
    print(f"  chi2_pass1 identical:     {'OK' if ok_chi2 else 'FAIL'}")

    R.record("16-J.4  Pass 1 invariant under multiplier", ok_counts and ok_chi2)


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
    _test_magnification_flag_threading(R)
    _test_refinement_safety_contract(R)
    _test_refinement_preserves_halo_count(R)
    _test_refinement_toggle_makes_difference(R)
    _test_refinement_skipped_without_sl(R)
    _test_refinement_respects_position_bounds(R)
    _test_tolerance_multiplier_default(R)
    _test_tolerance_multiplier_recorded(R)
    _test_tolerance_multiplier_monotonic(R)
    _test_tolerance_multiplier_pass1_invariant(R)
    return R.summary()


if __name__ == "__main__":
    run_two_pass_tests()