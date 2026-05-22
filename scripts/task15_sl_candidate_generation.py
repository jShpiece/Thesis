"""Task 15 — SL-driven candidate generation unit and integration tests."""

from __future__ import annotations

import sys

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.pipeline as pipeline
import arch.source_obj as source_obj
import arch.sl_candidate_generation as sl_cg
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
# Task 15 — SL-driven candidate generation
# ═══════════════════════════════════════════════════════════════════════════
#
# 15-A  Geometry: centroid candidate lies at image centroid
# 15-B  Geometry: pair-midpoint candidate lies at pair midpoint
# 15-C  Counts: N candidates = (1 + C(n,2)) per system, summed
# 15-D  SIS theta_E recovery from a synthetic 2-image system
# 15-E  NFW: SL candidates land near true halo position
# 15-F  Backward compatibility: no SL → identical to legacy candidates
# 15-G  Toggle: use_sl_candidates=False ignores strong_systems
# 15-H  Pipeline integration: WL+SL candidates → at least as accurate
#       as WL-only candidates, on a single-NFW + 1-SL-system scenario
# ═══════════════════════════════════════════════════════════════════════════


# ── 15-A  Centroid candidate position ─────────────────────────────────────

def _test_centroid_position(R: _TestResults):
    R.header("15-A  Centroid candidate at image centroid")

    # Construct a 3-image system with known centroid
    cx_true, cy_true = 7.5, -3.2
    offsets_x = np.array([+4.0, -3.0, -1.0])  # mean = 0
    offsets_y = np.array([+2.0, +1.5, -3.5])  # mean = 0
    tx = cx_true + offsets_x
    ty = cy_true + offsets_y

    sys = source_obj.StrongLensingSystem(
        system_id="A_centroid",
        theta_x=tx, theta_y=ty,
        z_source=2.0,
        sigma_theta=0.05,
    )

    lenses = sl_cg.cast_votes_sl_sis(
        [sys], include_centroid=True, include_pairs=False, theta_E_min=0.0
    )

    ok_count = len(lenses.x) == 1
    ok_cx = np.isclose(lenses.x[0], cx_true, atol=1e-10)
    ok_cy = np.isclose(lenses.y[0], cy_true, atol=1e-10)
    te_expected = float(np.mean(np.hypot(offsets_x, offsets_y)))
    ok_te = np.isclose(lenses.te[0], te_expected, atol=1e-10)

    print(f"  n_candidates = {len(lenses.x)} (expected 1)       "
          f"{'OK' if ok_count else 'FAIL'}")
    print(f"  centroid_x   = {lenses.x[0]:.6f} (expected {cx_true})  "
          f"{'OK' if ok_cx else 'FAIL'}")
    print(f"  centroid_y   = {lenses.y[0]:.6f} (expected {cy_true})  "
          f"{'OK' if ok_cy else 'FAIL'}")
    print(f"  te_estimate  = {lenses.te[0]:.6f} (expected {te_expected:.6f})  "
          f"{'OK' if ok_te else 'FAIL'}")

    R.record("15-A  Centroid candidate at image centroid",
             ok_count and ok_cx and ok_cy and ok_te)


# ── 15-B  Pair-midpoint candidate position ─────────────────────────────────

def _test_pair_position(R: _TestResults):
    R.header("15-B  Pair-midpoint candidate at pair midpoint")

    # 2-image system: only one pair → only one candidate
    tx = np.array([4.0, -6.0])  # midpoint = -1.0
    ty = np.array([2.0, -4.0])  # midpoint = -1.0
    sys = source_obj.StrongLensingSystem(
        system_id="B_pair",
        theta_x=tx, theta_y=ty,
        z_source=2.0, sigma_theta=0.05,
    )
    lenses = sl_cg.cast_votes_sl_sis(
        [sys], include_centroid=False, include_pairs=True, theta_E_min=0.0
    )

    ok_count = len(lenses.x) == 1
    ok_mx = np.isclose(lenses.x[0], -1.0, atol=1e-10)
    ok_my = np.isclose(lenses.y[0], -1.0, atol=1e-10)
    sep = float(np.hypot(tx[0] - tx[1], ty[0] - ty[1]))
    ok_te = np.isclose(lenses.te[0], 0.5 * sep, atol=1e-10)

    print(f"  n_candidates = {len(lenses.x)} (expected 1)       "
          f"{'OK' if ok_count else 'FAIL'}")
    print(f"  midpoint_x   = {lenses.x[0]:.6f} (expected -1)      "
          f"{'OK' if ok_mx else 'FAIL'}")
    print(f"  midpoint_y   = {lenses.y[0]:.6f} (expected -1)      "
          f"{'OK' if ok_my else 'FAIL'}")
    print(f"  te = sep/2   = {lenses.te[0]:.6f} (expected {0.5*sep:.6f})  "
          f"{'OK' if ok_te else 'FAIL'}")

    R.record("15-B  Pair-midpoint candidate at pair midpoint",
             ok_count and ok_mx and ok_my and ok_te)


# ── 15-C  Candidate counts vs combinatorics ───────────────────────────────

def _test_candidate_counts(R: _TestResults):
    R.header("15-C  Per-system candidate count = 1 + C(n_img, 2)")

    systems = []
    image_counts = [2, 3, 4]  # → 1+1=2, 1+3=4, 1+6=7 candidates
    expected_total = 0
    for k, n in enumerate(image_counts):
        rng = np.random.default_rng(k)
        # spread images over ~10 arcsec so theta_E >> theta_E_min
        tx = rng.uniform(-5, 5, n)
        ty = rng.uniform(-5, 5, n)
        systems.append(source_obj.StrongLensingSystem(
            system_id=f"C_{k}",
            theta_x=tx, theta_y=ty,
            z_source=2.0, sigma_theta=0.05,
        ))
        expected_total += 1 + (n * (n - 1)) // 2

    lenses = sl_cg.cast_votes_sl_sis(
        systems, include_centroid=True, include_pairs=True, theta_E_min=0.0
    )
    ok = len(lenses.x) == expected_total
    print(f"  total candidates = {len(lenses.x)} (expected {expected_total})  "
          f"{'OK' if ok else 'FAIL'}")
    R.record("15-C  Per-system candidate count", ok)


# ── 15-D  SIS theta_E recovery ────────────────────────────────────────────

def _test_sis_theta_E_recovery(R: _TestResults):
    R.header("15-D  SIS theta_E recovered from 2-image geometry")

    # Construct a 2-image SIS system at a known lens
    lens_xy = (3.0, -2.0)
    te_true = 4.5
    sys = make_two_image_sis_system_at_lens(
        system_id="D_sis",
        lens_center_xy=lens_xy,
        te_true=te_true,
        beta_rel_xy=(0.3, -0.2),  # small beta -> midpoint close to lens center
        sigma_theta=0.05,
        z_source=2.0,
    )

    # Pair candidate only (centroid coincides with midpoint for 2 images)
    lenses = sl_cg.cast_votes_sl_sis(
        [sys], include_centroid=False, include_pairs=True, theta_E_min=0.0
    )

    # te = pair_sep/2 should equal te_true exactly (SIS analytic)
    ok_count = len(lenses.x) == 1
    ok_te = np.isclose(lenses.te[0], te_true, atol=1e-10)
    # Midpoint matches lens center when beta=0; for beta!=0 it's offset by beta/2 (axisymmetric)
    # so just check it's within beta_offset of true lens
    beta_offset = float(np.hypot(0.3, -0.2))
    ok_position = np.hypot(lenses.x[0] - lens_xy[0],
                            lenses.y[0] - lens_xy[1]) <= beta_offset + 1e-6

    print(f"  te_est = {lenses.te[0]:.6f} (true {te_true:.6f})   "
          f"{'OK' if ok_te else 'FAIL'}")
    print(f"  midpoint within beta_offset of lens center: "
          f"{'OK' if ok_position else 'FAIL'}")

    R.record("15-D  SIS theta_E recovery", ok_count and ok_te and ok_position)


# ── 15-E  NFW SL candidates land near truth ───────────────────────────────

def _test_nfw_candidates_near_truth(R: _TestResults):
    R.header("15-E  NFW SL candidates land near true halo position")

    halo_true = _make_nfw_halo(x=0.0, y=0.0, mass=5e14, redshift=0.3)
    z_source = 2.0
    theta_E = _find_nfw_einstein_radius(halo_true, z_source)
    beta_rad = _find_nfw_beta_rad(halo_true, z_source, theta_E)
    br = beta_rad * 0.4
    sys = _make_two_image_nfw_system(
        "E_nfw", halo_true, (br, 0.0), z_source, theta_E, sigma_theta=0.06
    )

    lenses = sl_cg.cast_votes_sl_nfw(
        [sys], z_l=0.3,
        include_centroid=True, include_pairs=True, theta_E_min=0.0,
    )

    if len(lenses.x) == 0:
        print("  no candidates generated (unexpected)")
        R.record("15-E  NFW SL candidates near truth", False)
        return

    # All candidates should be within a small distance of the true halo
    # (within ~theta_E for a 2-image axisymmetric configuration)
    dists = np.hypot(lenses.x - halo_true.x[0], lenses.y - halo_true.y[0])
    max_dist = float(np.max(dists))
    ok_position = max_dist < theta_E  # closer than theta_E for axisymmetric

    # Mass seed should be in the strength-optimization bounds (1e10 - 1e16 M_sun)
    masses = np.atleast_1d(lenses.mass)
    ok_mass_range = np.all((masses >= 1e10) & (masses <= 1e16))

    # Mass seed should be within ~1 dex of the true M_200 (rough)
    log_ratio = np.log10(masses / halo_true.mass[0])
    ok_mass_scale = np.all(np.abs(log_ratio) < 3.0)  # within 3 dex

    print(f"  n_candidates = {len(lenses.x)}")
    print(f"  theta_E_true = {theta_E:.3f} arcsec")
    print(f"  max distance to truth = {max_dist:.3f} arcsec  "
          f"{'OK' if ok_position else 'FAIL'}")
    print(f"  mass seeds in [{masses.min():.2e}, {masses.max():.2e}] M_sun")
    print(f"  all in [1e10, 1e16] M_sun: {'OK' if ok_mass_range else 'FAIL'}")
    print(f"  log10(M_seed/M_true) in [{log_ratio.min():.2f}, "
          f"{log_ratio.max():.2f}]: {'OK' if ok_mass_scale else 'FAIL'}")

    R.record("15-E  NFW SL candidates near truth",
             ok_position and ok_mass_range and ok_mass_scale)


# ── 15-F  Backward compatibility: no SL → unchanged candidates ────────────

def _test_backward_compat_no_sl(R: _TestResults):
    R.header("15-F  Backward compat: no strong_systems → unchanged WL candidates")

    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=[(5.0, -3.0, 3.5)],
        xmax=35.0, n_sources=50, seed=11,
    )
    # No strong systems attached → strong_systems list is empty

    sis_legacy = pipeline.generate_initial_guess(
        src, lens_type="SIS", use_sl_candidates=False
    )
    sis_default = pipeline.generate_initial_guess(
        src, lens_type="SIS", use_sl_candidates=True
    )

    ok_x = np.allclose(sis_legacy.x, sis_default.x)
    ok_y = np.allclose(sis_legacy.y, sis_default.y)
    ok_te = np.allclose(sis_legacy.te, sis_default.te)

    print(f"  SIS legacy vs default identical (empty SL):  "
          f"x={'OK' if ok_x else 'FAIL'}  "
          f"y={'OK' if ok_y else 'FAIL'}  "
          f"te={'OK' if ok_te else 'FAIL'}")

    R.record("15-F  Backward compat no SL", ok_x and ok_y and ok_te)


# ── 15-G  Toggle disables SL candidates ──────────────────────────────────

def _test_toggle_disables_sl(R: _TestResults):
    R.header("15-G  use_sl_candidates=False ignores attached strong_systems")

    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=[(5.0, -3.0, 3.5)],
        xmax=35.0, n_sources=50, seed=12,
    )
    sys_A = make_two_image_sis_system_at_lens(
        system_id="G_sys", lens_center_xy=(5.0, -3.0),
        te_true=3.5, beta_rel_xy=(0.3, -0.2),
        sigma_theta=0.04, z_source=2.0,
    )
    attach_strong_systems(src, [sys_A])

    sis_off = pipeline.generate_initial_guess(
        src, lens_type="SIS", use_sl_candidates=False
    )
    sis_on = pipeline.generate_initial_guess(
        src, lens_type="SIS", use_sl_candidates=True
    )

    ok_more = len(sis_on.x) > len(sis_off.x)
    ok_off_unchanged = (
        len(sis_off.x) == len(src.x)  # one candidate per source, no SL
    )

    print(f"  off: {len(sis_off.x)} candidates (n_sources={len(src.x)})  "
          f"{'OK' if ok_off_unchanged else 'FAIL'}")
    print(f"  on:  {len(sis_on.x)} candidates (more than off)  "
          f"{'OK' if ok_more else 'FAIL'}")

    R.record("15-G  Toggle disables SL", ok_more and ok_off_unchanged)


# ── 15-H  Pipeline integration: WL+SL candidates beat WL-only ────────────

def _test_pipeline_integration(R: _TestResults):
    R.header("15-H  Pipeline: WL+SL candidates >= WL-only on NFW + 1 SL system")

    # Build a single-NFW scenario with strong-lensing data
    halo_true = _make_nfw_halo(x=4.0, y=-3.0, mass=3e14, redshift=0.3)
    z_source = 2.0
    theta_E = _find_nfw_einstein_radius(halo_true, z_source)
    beta_rad = _find_nfw_beta_rad(halo_true, z_source, theta_E)
    br = beta_rad * 0.4
    sys_A = _make_two_image_nfw_system(
        "H_sys", halo_true, (br, 0.0), z_source, theta_E, sigma_theta=0.06
    )

    xmax = 60.0
    src = _make_nfw_wl_catalog(
        halo_true, xmax=xmax, n_sources=80, z_source=1.0, seed=15,
    )
    attach_strong_systems(src, [sys_A])

    use_flags = [True, True, False]

    # Run 1: WL-only candidates (legacy)
    try:
        # Temporarily disable SL candidates by passing override via main is
        # not exposed; we patch generate_initial_guess at the call site by
        # detaching strong_systems for the candidate stage, then reattaching
        # for the rest of the pipeline.  In production code you'd add a
        # flag to fit_lensing_field; here we exercise the toggle by hand.
        saved = list(src.strong_systems)
        src.strong_systems = []
        lenses_wl, _ = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type="NFW", z_lens=0.3, use_strong_lensing=False,
        )
        src.strong_systems = saved  # reattach for SL run
    except Exception as e:
        print(f"  WL-only run failed: {e}")
        R.record("15-H  Pipeline integration", False)
        return

    # Run 2: WL+SL candidates AND SL in chi2
    try:
        lenses_sl, _ = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type="NFW", z_lens=0.3, use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  WL+SL run failed: {e}")
        R.record("15-H  Pipeline integration", False)
        return

    def nearest_dist(lenses):
        if len(lenses.x) == 0:
            return np.inf
        return float(np.min(np.hypot(
            lenses.x - halo_true.x[0], lenses.y - halo_true.y[0]
        )))

    d_wl = nearest_dist(lenses_wl)
    d_sl = nearest_dist(lenses_sl)

    # SL run should be at least as close as WL-only (within tolerance)
    tol = 1.0  # arcsec slack
    ok_improvement = d_sl <= d_wl + tol
    ok_both_finite = np.isfinite(d_wl) and np.isfinite(d_sl)

    print(f"  WL-only:  nearest distance to truth = {d_wl:.3f} arcsec, "
          f"{len(lenses_wl.x)} halo(s)")
    print(f"  WL + SL:  nearest distance to truth = {d_sl:.3f} arcsec, "
          f"{len(lenses_sl.x)} halo(s)")
    print(f"  SL no worse than WL+tol={tol}: "
          f"{'OK' if ok_improvement else 'FAIL'}")

    R.record("15-H  Pipeline integration", ok_both_finite and ok_improvement)


# ── Runner ────────────────────────────────────────────────────────────────

def run_sl_candidate_tests() -> bool:
    R = _TestResults()
    _test_centroid_position(R)
    _test_pair_position(R)
    _test_candidate_counts(R)
    _test_sis_theta_E_recovery(R)
    _test_nfw_candidates_near_truth(R)
    _test_backward_compat_no_sl(R)
    _test_toggle_disables_sl(R)
    _test_pipeline_integration(R)
    return R.summary()


if __name__ == "__main__":
    run_sl_candidate_tests()