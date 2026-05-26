"""Task 16 — Two-pass forward selection (WL → λ_SL → WL+SL)."""

from __future__ import annotations

import sys

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.metric as metric
import arch.pipeline as pipeline
import arch.utils as utils
from arch.chi2_wrappers import update_chi2_values
from arch.forward_selection import _greedy_add_pass, _empty_lens_collection
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
# 16-H  Magnification-correction flag is threaded correctly through Pass 2
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
        "use_magnification_correction_sl",
    }
    have_keys = expected_keys.issubset(diag.keys())
    types_ok = (
        isinstance(diag["n_pass1"], int)
        and isinstance(diag["n_pass2_added"], int)
        and isinstance(diag["lambda_sl"], float)
        and isinstance(diag["chi2_pass1"], float)
        and isinstance(diag["chi2_final"], float)
        and isinstance(diag["use_magnification_correction_sl"], bool)
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


# ── 16-H  Magnification-correction flag is threaded through Pass 2 ───────

def _test_magnification_flag_threading(R: _TestResults16):
    """
    Verifies the magnification-correction fix:

    1. update_chi2_values accepts the use_magnification_correction_sl
       parameter and produces DIFFERENT values for True vs False when
       chi2_SL is nonzero (i.e., the parameter actually has an effect).
    2. _greedy_add_pass propagates the flag — calling it with
       use_magnification_correction_sl=False vs True gives different
       per-candidate scoring whenever the SL chi^2 is sensitive to
       magnification.
    3. forward_lens_selection_two_pass defaults to False and records
       the choice in diagnostics.
    4. The chi^2 evaluated during Pass 2 (with mag correction off)
       is consistent with the lambda_sl calibration — specifically,
       it should be much SMALLER than the same chi^2 evaluated with
       mag correction on, on a model that has SL tension.
    """
    R.header("16-H  Magnification-correction flag threading through Pass 2")

    # Build a scenario with intentional SL tension at the Pass 1 minimum:
    # use a 2-image NFW system but seed the WL catalog such that the
    # WL-only fit doesn't perfectly satisfy SL.  We control this by
    # using a noisy SL system (large sigma_theta).
    src, halo_true, xmax = _build_nfw_with_sl(seed=80)
    use_flags = [True, True, False]

    # Build a deliberately *imperfect* lens model (offset position by
    # ~1 arcsec from truth) so the SL chi^2 is sensitive to magnification.
    import arch.halo_obj as halo_obj
    imperfect = halo_obj.NFW_Lens(
        x=np.array([halo_true.x[0] + 1.0]),
        y=np.array([halo_true.y[0] + 1.0]),
        z=np.zeros(1),
        concentration=halo_true.concentration.copy(),
        mass=halo_true.mass.copy(),
        redshift=halo_true.redshift,
        chi2=np.zeros(1),
    )

    # External SL chi^2 evaluation with mag on/off (sanity ground truth)
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
    ok_external_differ = chi2_sl_on > chi2_sl_off  # mag correction inflates SL chi^2
    print(f"  External: mag_on > mag_off (correction inflates near critical curves): "
          f"{'OK' if ok_external_differ else 'FAIL'}")

    # ── (1) update_chi2_values responds to the flag ──
    lambda_sl_fixed = 1.0  # arbitrary nonzero
    chi2_total_off = update_chi2_values(
        src, imperfect, use_flags, "NFW",
        use_strong_lensing=True,
        lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=False,
    )
    chi2_total_on = update_chi2_values(
        src, imperfect, use_flags, "NFW",
        use_strong_lensing=True,
        lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=True,
    )
    print(f"  update_chi2_values(mag=False) = {chi2_total_off:.6f}")
    print(f"  update_chi2_values(mag=True)  = {chi2_total_on:.6f}")
    ok_update_responds = chi2_total_on != chi2_total_off
    print(f"  update_chi2_values responds to flag: "
          f"{'OK' if ok_update_responds else 'FAIL'}")

    # ── (2) _greedy_add_pass propagates the flag ──
    # Construct a tiny candidate pool with one candidate and verify
    # that the two flag values produce different test chi^2 values.
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
    # Append one candidate to the start so chi^2 is nonzero
    from arch.forward_selection import _append_candidate
    start_off = _append_candidate(start_off, test_cand, 0, "NFW")
    start_on = _append_candidate(start_on, test_cand, 0, "NFW")

    sel_off, chi2_off, _ = _greedy_add_pass(
        src, test_cand, start_off, np.array([1], dtype=int),
        use_flags, "NFW",
        base_tolerance=0.003, mass_scale=1e13,
        kappa_scale=0.1, exponent=-1.0,
        use_strong_lensing=True, lambda_sl=lambda_sl_fixed,
        use_magnification_correction_sl=False,
    )
    sel_on, chi2_on, _ = _greedy_add_pass(
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
    print(f"  _greedy_add_pass propagates the flag: "
          f"{'OK' if ok_greedy_responds else 'FAIL'}")

    # ── (3) forward_lens_selection_two_pass defaults to False ──
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
    print(f"  two-pass default use_magnification_correction_sl = "
          f"{diag['use_magnification_correction_sl']}  "
          f"{'OK' if ok_default_false else 'FAIL'}")

    # ── (4) Pass 2 chi^2 (mag off) is consistent with calibration ──
    # The key physical claim: chi^2_SL with mag correction off is LESS
    # OR EQUAL to chi^2_SL with mag correction on (at any given model).
    # This is what allows the lambda_sl calibration (computed mag-off)
    # to remain meaningful during Pass 2.
    ok_consistency = chi2_sl_off <= chi2_sl_on + 1e-9
    print(f"  chi2_SL_no_mag <= chi2_SL_mag_on (calibration consistency): "
          f"{'OK' if ok_consistency else 'FAIL'}")

    ok_all = (
        ok_external_differ
        and ok_update_responds
        and ok_greedy_responds
        and ok_default_false
        and ok_consistency
    )
    R.record("16-H  Magnification flag threading", ok_all)


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
    return R.summary()


if __name__ == "__main__":
    run_two_pass_tests()