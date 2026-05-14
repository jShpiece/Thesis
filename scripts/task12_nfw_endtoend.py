"""Task 12 — SIS end-to-end integration test via fit_lensing_field."""

from __future__ import annotations

import sys

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.metric as metric
import arch.pipeline as pipeline
import arch.utils as utils
from arch.main import fit_lensing_field
from scripts.paper2_shared import (
    _TestResults,
    attach_strong_systems,
    make_two_image_sis_system_at_lens,
    make_weak_lensing_catalog_two_lenses,
)

# ═══════════════════════════════════════════════════════════════════════════
# 5) Task 12 — End-to-end integration test via fit_lensing_field
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs the full pipeline (main.fit_lensing_field) on a single-SIS toy
# scenario with and without strong lensing.  A single lens avoids the
# composite cross-deflection complication that makes the "perfect model"
# test ill-defined for multi-halo systems.
#
# The test verifies:
#   - Both runs complete without errors
#   - Both return finite positive reduced chi2 and at least one lens
#   - Both recover a lens near the true position (coarse sanity check)
#   - The SL run positions the nearest recovered lens at least as close
#     to truth as the WL-only run (the information gain from SL)
#   - Recovered Einstein radii are in the right ballpark
#   - Backwards compatibility: WL-only run on a catalog that *has*
#     strong systems simply ignores them
# ═══════════════════════════════════════════════════════════════════════════


def _build_single_lens_scenario(seed: int = 55):
    """
    Build a reproducible single-SIS test case.

    Returns
    -------
    src : Source  (with one strong system attached)
    true_x, true_y, true_te : float
    xmax : float
    """
    true_x, true_y, true_te = 5.0, -3.0, 3.5
    xmax = 35.0
    n_sources = 80

    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=[(true_x, true_y, true_te)],
        xmax=xmax,
        n_sources=n_sources,
        sig_shear=0.08,
        sig_flex=0.015,
        sig_gflex=0.025,
        seed=seed,
    )

    sys_A = make_two_image_sis_system_at_lens(
        system_id="integ_sys_A",
        lens_center_xy=(true_x, true_y),
        te_true=true_te,
        beta_rel_xy=(0.5, -0.3),
        sigma_theta=0.04,
        z_source=2.0,
    )
    attach_strong_systems(src, [sys_A])

    return src, true_x, true_y, true_te, xmax


def _nearest_lens_distance(lenses, true_x, true_y):
    """Return the distance from the nearest recovered lens to the true position."""
    if len(lenses.x) == 0:
        return np.inf
    distances = np.hypot(lenses.x - true_x, lenses.y - true_y)
    return float(np.min(distances))


def _nearest_lens_te(lenses, true_x, true_y):
    """Return the Einstein radius of the lens nearest to the true position."""
    if len(lenses.x) == 0:
        return np.nan
    idx = np.argmin(np.hypot(lenses.x - true_x, lenses.y - true_y))
    return float(lenses.te[idx])


def _test_integration_wl_only(R: _TestResults):
    """
    12-A: Run fit_lensing_field with use_strong_lensing=False.
    Verify: completes, finite chi2, ≥1 lens, near truth, sane θ_E.
    """
    R.header("12-A  WL-only baseline via fit_lensing_field")

    src, tx, ty, tte, xmax = _build_single_lens_scenario()
    use_flags = [True, True, False]  # shear + flexion

    try:
        lenses_wl, rchi2_wl = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
            z_lens=0.5,
            use_strong_lensing=False,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("12-A  WL-only baseline", False)
        return None, None

    n_lens = len(lenses_wl.x)
    ok_finite = np.isfinite(rchi2_wl) and rchi2_wl > 0
    ok_nlens = n_lens >= 1
    d_wl = _nearest_lens_distance(lenses_wl, tx, ty)
    ok_near = d_wl < 10.0  # coarse: within 10"
    te_wl = _nearest_lens_te(lenses_wl, tx, ty)
    ok_te = 0.5 < te_wl < 15.0  # within a factor of ~4

    print(f"  N_lenses    = {n_lens}  {'OK' if ok_nlens else 'FAIL'}")
    print(f"  reduced χ²  = {rchi2_wl:.4f}  {'OK' if ok_finite else 'FAIL'}")
    print(f"  nearest Δ   = {d_wl:.2f}\"  (< 10\")  {'OK' if ok_near else 'FAIL'}")
    print(f"  nearest θ_E = {te_wl:.2f}\"  (true = {tte})  {'OK' if ok_te else 'FAIL'}")

    ok_all = ok_finite and ok_nlens and ok_near and ok_te
    R.record("12-A  WL-only baseline", ok_all)
    return lenses_wl, d_wl


def _test_integration_wl_plus_sl(R: _TestResults, d_wl_ref: float = None):
    """
    12-B: Run fit_lensing_field with use_strong_lensing=True.
    Verify: completes, finite chi2, ≥1 lens, near truth, sane θ_E.
    If d_wl_ref is provided, also check that SL is at least as good.
    """
    R.header("12-B  WL+SL via fit_lensing_field")

    src, tx, ty, tte, xmax = _build_single_lens_scenario()
    use_flags = [True, True, False]

    try:
        lenses_sl, rchi2_sl = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
            z_lens=0.5,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("12-B  WL+SL pipeline", False)
        return None

    n_lens = len(lenses_sl.x)
    ok_finite = np.isfinite(rchi2_sl) and rchi2_sl > 0
    ok_nlens = n_lens >= 1
    d_sl = _nearest_lens_distance(lenses_sl, tx, ty)
    ok_near = d_sl < 10.0
    te_sl = _nearest_lens_te(lenses_sl, tx, ty)
    ok_te = 0.5 < te_sl < 15.0

    print(f"  N_lenses    = {n_lens}  {'OK' if ok_nlens else 'FAIL'}")
    print(f"  reduced χ²  = {rchi2_sl:.4f}  {'OK' if ok_finite else 'FAIL'}")
    print(f"  nearest Δ   = {d_sl:.2f}\"  (< 10\")  {'OK' if ok_near else 'FAIL'}")
    print(f"  nearest θ_E = {te_sl:.2f}\"  (true = {tte})  {'OK' if ok_te else 'FAIL'}")

    # ── Comparison with WL-only ──
    ok_improvement = True
    if d_wl_ref is not None:
        # Allow a small tolerance — SL shouldn't make things much worse
        # d_sl <= d_wl + 1" is acceptable (noise can cause minor regression)
        ok_improvement = d_sl <= d_wl_ref + 1.0
        better = d_sl < d_wl_ref
        print(
            f'  Δ_WL = {d_wl_ref:.2f}"  Δ_SL = {d_sl:.2f}"  '
            f"{'improved' if better else 'comparable'}  "
            f"{'OK' if ok_improvement else 'FAIL (SL much worse)'}"
        )

    ok_all = ok_finite and ok_nlens and ok_near and ok_te and ok_improvement
    R.record("12-B  WL+SL pipeline", ok_all)
    return lenses_sl


def _test_integration_backwards_compat(R: _TestResults):
    """
    12-C: fit_lensing_field with use_strong_lensing=False on a catalog
    that HAS strong_systems — the SL data should be silently ignored.
    Also tests that calling without the new keyword works (default=False).
    """
    R.header("12-C  Backwards compatibility")

    src, tx, ty, tte, xmax = _build_single_lens_scenario()
    use_flags = [True, True, False]

    ok_all = True

    # ── Call with explicit use_strong_lensing=False ──
    try:
        lenses_1, rchi2_1 = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
            use_strong_lensing=False,
        )
        ok_1 = np.isfinite(rchi2_1) and len(lenses_1.x) >= 1
        print(
            f"  Explicit False: chi2={rchi2_1:.4f} n_lens={len(lenses_1.x)}  "
            f"{'OK' if ok_1 else 'FAIL'}"
        )
    except Exception as e:
        print(f"  Explicit False: EXCEPTION {e}  FAIL")
        ok_1 = False
    ok_all = ok_all and ok_1

    # ── Call without the keyword at all (should default to False) ──
    try:
        lenses_2, rchi2_2 = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
        )
        ok_2 = np.isfinite(rchi2_2) and len(lenses_2.x) >= 1
        print(
            f"  Default (omitted): chi2={rchi2_2:.4f} n_lens={len(lenses_2.x)}  "
            f"{'OK' if ok_2 else 'FAIL'}"
        )
    except Exception as e:
        print(f"  Default (omitted): EXCEPTION {e}  FAIL")
        ok_2 = False
    ok_all = ok_all and ok_2

    # ── Both runs should give the same result (deterministic) ──
    if ok_1 and ok_2:
        ok_same = np.isclose(rchi2_1, rchi2_2, rtol=1e-8)
        print(f"  Both identical: {'OK' if ok_same else 'FAIL (non-deterministic)'}")
        ok_all = ok_all and ok_same

    R.record("12-C  Backwards compatibility", ok_all)


def _test_integration_chi2_components(R: _TestResults):
    """
    12-D: After the SL pipeline run, manually evaluate the combined chi2
    and verify the WL and SL components are both positive and the lambda_sl
    in the components dict is finite.
    """
    R.header("12-D  Post-run chi2 component verification")

    src, tx, ty, tte, xmax = _build_single_lens_scenario()
    use_flags = [True, True, False]

    try:
        lenses_sl, _ = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION during pipeline: {e}")
        R.record("12-D  chi2 components", False)
        return

    # Evaluate components at the final lens configuration
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, "SIS")
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src,
        lenses_sl,
        use_flags,
        lens_type="SIS",
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
    )

    ok_wl = comps["chi2_wl"] > 0
    ok_sl = comps["chi2_sl"] >= 0
    ok_lam = np.isfinite(comps["lambda_sl"]) and comps["lambda_sl"] > 0
    ok_dof = comps["dof_wl"] > 0 and comps["dof_sl"] > 0
    ok_total = np.isclose(
        chi2_total, comps["chi2_wl"] + comps["lambda_sl"] * comps["chi2_sl"], rtol=1e-10
    )

    print(
        f"  chi2_WL   = {comps['chi2_wl']:.2f}  (dof={comps['dof_wl']})  "
        f"{'OK' if ok_wl else 'FAIL'}"
    )
    print(
        f"  chi2_SL   = {comps['chi2_sl']:.2f}  (dof={comps['dof_sl']})  "
        f"{'OK' if ok_sl else 'FAIL'}"
    )
    print(f"  lambda_sl = {comps['lambda_sl']:.6f}  " f"{'OK' if ok_lam else 'FAIL'}")
    print(
        f"  dof > 0:  WL={'OK' if comps['dof_wl']>0 else 'FAIL'}  "
        f"SL={'OK' if comps['dof_sl']>0 else 'FAIL'}"
    )
    print(f"  total = WL + lam*SL: {'OK' if ok_total else 'FAIL'}")

    ok_all = ok_wl and ok_sl and ok_lam and ok_dof and ok_total
    R.record("12-D  chi2 components", ok_all)


def _test_integration_strong_system_scatter(R: _TestResults):
    """
    12-E: Verify that the recovered lenses actually reduce the source-plane
    scatter of the strong system relative to the initial guess.
    """
    R.header("12-E  Source-plane scatter improvement")

    src, tx, ty, tte, xmax = _build_single_lens_scenario()
    use_flags = [True, True, False]

    # Get initial-guess lenses
    lenses_init = pipeline.generate_initial_guess(src, lens_type="SIS", z_l=0.5)

    chi2_sl_init = utils.chi2_strong_source_plane_sis(lenses_init, src.strong_systems)

    # Run the full pipeline
    try:
        lenses_final, _ = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="SIS",
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("12-E  SL scatter improvement", False)
        return

    chi2_sl_final = utils.chi2_strong_source_plane_sis(lenses_final, src.strong_systems)

    ok_reduced = chi2_sl_final < chi2_sl_init
    print(f"  chi2_SL initial = {chi2_sl_init:.2f}")
    print(f"  chi2_SL final   = {chi2_sl_final:.2f}")
    print(f"  Scatter reduced: {'OK' if ok_reduced else 'FAIL'}")

    # Also get the per-system breakdown
    _, bd = utils.chi2_strong_source_plane_sis(
        lenses_final,
        src.strong_systems,
        return_breakdown=True,
        use_magnification_correction=True,
    )
    for sid, info in bd.items():
        print(
            f"    {sid}: chi2={info['chi2']:.2f}  n_img={info['n_images']}  "
            f"beta_bar=({info['beta_bar'][0]:.3f}, {info['beta_bar'][1]:.3f})"
        )

    R.record("12-E  SL scatter improvement", ok_reduced)


# ── Runner ────────────────────────────────────────────────────────────────


def run_integration_tests() -> bool:
    """
    Execute all Task 12 integration tests.  Returns True if all pass.

    These tests call fit_lensing_field, which runs the full optimiser
    pipeline.  Expect ~10-30 s per run depending on source count.
    """
    R = _TestResults()

    # 12-A: WL-only baseline
    lenses_wl, d_wl = _test_integration_wl_only(R)

    # 12-B: WL+SL (pass WL distance for comparison)
    _test_integration_wl_plus_sl(R, d_wl_ref=d_wl)

    # 12-C: backwards compatibility
    _test_integration_backwards_compat(R)

    # 12-D: chi2 component verification
    _test_integration_chi2_components(R)

    # 12-E: source-plane scatter improvement
    _test_integration_strong_system_scatter(R)

    return R.summary()


if __name__ == "__main__":
    run_integration_tests()
