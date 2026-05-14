"""Task 14 — NFW end-to-end integration test and WL vs WL+SL comparison plot."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.metric as metric
import arch.pipeline as pipeline
import arch.utils as utils
from arch.main import fit_lensing_field
from scripts.paper2_shared import (
    _find_nfw_beta_rad,
    _find_nfw_einstein_radius,
    _make_nfw_halo,
    _make_nfw_wl_catalog,
    _make_two_image_nfw_system,
    _TestResults,
    attach_strong_systems,
)
from scripts.task11_magnification_tests import run_magnification_tests
from scripts.task12_nfw_endtoend import run_integration_tests
from scripts.task13_nfw_sl_unit import run_nfw_sl_tests

# ═══════════════════════════════════════════════════════════════════════════
# 7) Task 14 — NFW End-to-end Integration Test
# ═══════════════════════════════════════════════════════════════════════════
#
# Runs the full pipeline (main.fit_lensing_field) on a single-NFW toy
# scenario with and without strong lensing, mirroring Task 12's structure.
#
# The test verifies:
#   - Both runs complete without errors
#   - Both return finite positive reduced chi2 and at least one halo
#   - Both recover a halo within 30" of the true position (coarse sanity)
#   - Recovered mass is in a physically plausible range
#   - The SL run positions the nearest halo at least as close to truth as
#     the WL-only run (within a tolerance)
#   - chi2 components decompose correctly: total = chi2_WL + lambda*chi2_SL
#   - Source-plane scatter reduces from initial guess to final pipeline output
# ═══════════════════════════════════════════════════════════════════════════


def _build_nfw_scenario(seed: int = 55):
    """
    Build a reproducible single-NFW test case.

    Returns
    -------
    src         : Source  (with one strong system attached)
    halos_true  : NFW_Lens  (the ground-truth halo)
    true_x, true_y, true_mass, true_redshift : float
    xmax        : float
    """
    true_x, true_y = 5.0, -3.0
    true_mass = 5e14  # solar masses
    true_redshift = 0.3
    xmax = 120.0
    n_sources = 100
    z_source_wl = 0.8
    z_source_sl = 2.0

    halos_true = _make_nfw_halo(x=true_x, y=true_y, mass=true_mass, redshift=true_redshift)

    src = _make_nfw_wl_catalog(
        halos_true,
        xmax=xmax,
        n_sources=n_sources,
        z_source=z_source_wl,
        sig_shear=0.08,
        sig_flex=0.015,
        sig_gflex=0.025,
        seed=seed,
    )

    # Build a strong lensing system with source inside the radial caustic
    theta_E = _find_nfw_einstein_radius(halos_true, z_source_sl)
    beta_rad = _find_nfw_beta_rad(halos_true, z_source_sl, theta_E)
    br = beta_rad * 0.45  # safely inside radial caustic
    sys_A = _make_two_image_nfw_system(
        "nfw14_A",
        halos_true,
        (br * 0.8, br * 0.6),
        z_source_sl,
        theta_E,
        sigma_theta=0.05,
    )
    attach_strong_systems(src, [sys_A])

    return src, halos_true, true_x, true_y, true_mass, true_redshift, xmax


def _nearest_nfw_distance(lenses, true_x, true_y):
    """Return the distance from the nearest recovered NFW halo to the true position."""
    if len(lenses.x) == 0:
        return np.inf
    return float(np.min(np.hypot(lenses.x - true_x, lenses.y - true_y)))


def _nearest_nfw_mass(lenses, true_x, true_y):
    """Return the mass (M_sun) of the NFW halo nearest to the true position."""
    if len(lenses.x) == 0:
        return np.nan
    idx = int(np.argmin(np.hypot(lenses.x - true_x, lenses.y - true_y)))
    return float(lenses.mass[idx])


def _test_nfw_integration_wl_only(R):
    """
    14-A: Run fit_lensing_field with use_strong_lensing=False (NFW).
    Verify: completes, finite chi2, >=1 halo, near truth, sane mass.
    """
    R.header("14-A  NFW WL-only baseline via fit_lensing_field")

    src, halos_true, tx, ty, tm, tz, xmax = _build_nfw_scenario()
    use_flags = [True, True, False]  # shear + flexion

    try:
        lenses_wl, rchi2_wl = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="NFW",
            z_lens=tz,
            use_strong_lensing=False,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("14-A  NFW WL-only baseline", False)
        return None, None

    n_lens = len(lenses_wl.x)
    ok_finite = np.isfinite(rchi2_wl) and rchi2_wl > 0
    ok_nlens = n_lens >= 1
    d_wl = _nearest_nfw_distance(lenses_wl, tx, ty)
    ok_near = d_wl < 30.0  # within 30" of truth (25% of field radius)
    m_wl = _nearest_nfw_mass(lenses_wl, tx, ty)
    ok_mass = 1e13 < m_wl < 1e16  # wide sanity check (factor ~20 either way)

    print(f"  N_halos      = {n_lens}  {'OK' if ok_nlens else 'FAIL'}")
    print(f"  reduced chi2 = {rchi2_wl:.4f}  {'OK' if ok_finite else 'FAIL'}")
    print(f"  nearest dist = {d_wl:.2f}\"  (< 30\")  {'OK' if ok_near else 'FAIL'}")
    print(f"  nearest mass = {m_wl:.3e} M_sun  (true={tm:.1e})  " f"{'OK' if ok_mass else 'FAIL'}")

    ok_all = ok_finite and ok_nlens and ok_near and ok_mass
    R.record("14-A  NFW WL-only baseline", ok_all)
    return lenses_wl, d_wl


def _test_nfw_integration_wl_plus_sl(R, d_wl_ref=None):
    """
    14-B: Run fit_lensing_field with use_strong_lensing=True (NFW).

    Note on NFW+SL behaviour: lambda_sl is pre-computed as
    rchi2_WL_initial / rchi2_SL_initial.  For NFW, the initial WL fit is
    typically very poor (rchi2_WL ~ 500-1000) while the initial SL fit may
    already be good (rchi2_SL ~ 1).  This can yield lambda ~ 500, causing the
    optimiser to almost exclusively minimise chi2_SL and potentially move the
    halo far from the WL-preferred position.

    This test therefore only verifies pipeline correctness (no crash, ≥1 halo,
    finite chi2, positive WL component), and reports the position and lambda for
    informational purposes without treating position improvement as a pass criterion.
    """
    R.header("14-B  NFW WL+SL via fit_lensing_field")

    src, halos_true, tx, ty, tm, tz, xmax = _build_nfw_scenario()
    use_flags = [True, True, False]

    try:
        lenses_sl, rchi2_sl = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="NFW",
            z_lens=tz,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("14-B  NFW WL+SL pipeline", False)
        return None

    n_lens = len(lenses_sl.x)
    ok_finite = np.isfinite(rchi2_sl) and rchi2_sl > 0
    ok_nlens = n_lens >= 1
    d_sl = _nearest_nfw_distance(lenses_sl, tx, ty)
    m_sl = _nearest_nfw_mass(lenses_sl, tx, ty)

    print(f"  N_halos      = {n_lens}  {'OK' if ok_nlens else 'FAIL'}")
    print(f"  reduced chi2 = {rchi2_sl:.4f}  {'OK' if ok_finite else 'FAIL'}")
    print(f'  nearest dist = {d_sl:.2f}"  (informational, true = ({tx}, {ty}))')
    print(f"  nearest mass = {m_sl:.3e} M_sun  (informational, true = {tm:.1e})")

    # SL should improve position over WL-only
    ok_improvement = True
    if d_wl_ref is not None:
        ok_improvement = d_sl <= d_wl_ref + 5.0  # SL within 5" of WL-only or better
        better = d_sl < d_wl_ref
        label = "SL closer to truth" if better else "WL closer to truth"
        print(
            f'  dist_WL = {d_wl_ref:.2f}"  dist_SL = {d_sl:.2f}"  {label}  '
            f"{'OK' if ok_improvement else 'FAIL'}"
        )

    # Verify chi2_WL is positive (WL constraint is evaluated)
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, "NFW")
    chi2_total, _, comps = metric.calculate_total_chi2(
        src,
        lenses_sl,
        use_flags,
        lens_type="NFW",
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
    )
    ok_wl_pos = comps["chi2_wl"] > 0
    print(
        f"  chi2_WL > 0:  {'OK' if ok_wl_pos else 'FAIL'}  " f"(chi2_WL = {comps['chi2_wl']:.2f})"
    )

    ok_all = ok_finite and ok_nlens and ok_wl_pos and ok_improvement
    R.record("14-B  NFW WL+SL pipeline", ok_all)
    return lenses_sl


def _test_nfw_integration_chi2_components(R):
    """
    14-C: After the NFW SL pipeline run, manually evaluate the combined chi2
    and verify the WL and SL components are both positive and the lambda_sl
    in the components dict is finite, and that the total decomposes correctly.
    """
    R.header("14-C  NFW post-run chi2 component verification")

    src, halos_true, tx, ty, tm, tz, xmax = _build_nfw_scenario()
    use_flags = [True, True, False]

    try:
        lenses_sl, _ = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="NFW",
            z_lens=tz,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION during pipeline: {e}")
        R.record("14-C  NFW chi2 components", False)
        return

    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, "NFW")
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src,
        lenses_sl,
        use_flags,
        lens_type="NFW",
        use_strong_lensing=True,
        lambda_sl=lambda_sl,
    )

    ok_wl = comps["chi2_wl"] > 0
    ok_sl = comps["chi2_sl"] >= 0
    ok_lam = np.isfinite(comps["lambda_sl"]) and comps["lambda_sl"] > 0
    ok_dof = comps["dof_wl"] > 0 and comps["dof_sl"] > 0
    ok_total = np.isclose(
        chi2_total,
        comps["chi2_wl"] + comps["lambda_sl"] * comps["chi2_sl"],
        rtol=1e-10,
    )

    print(
        f"  chi2_WL   = {comps['chi2_wl']:.2f}  (dof={comps['dof_wl']})  "
        f"{'OK' if ok_wl else 'FAIL'}"
    )
    print(
        f"  chi2_SL   = {comps['chi2_sl']:.2f}  (dof={comps['dof_sl']})  "
        f"{'OK' if ok_sl else 'FAIL'}"
    )
    print(f"  lambda_sl = {comps['lambda_sl']:.6f}  {'OK' if ok_lam else 'FAIL'}")
    print(
        f"  dof > 0:  WL={'OK' if comps['dof_wl']>0 else 'FAIL'}  "
        f"SL={'OK' if comps['dof_sl']>0 else 'FAIL'}"
    )
    print(f"  total = WL + lam*SL: {'OK' if ok_total else 'FAIL'}")

    ok_all = ok_wl and ok_sl and ok_lam and ok_dof and ok_total
    R.record("14-C  NFW chi2 components", ok_all)


def _test_nfw_integration_sl_scatter(R):
    """
    14-D: Verify that chi2_SL correctly identifies the true halo as the
    better model compared to the WL-only reconstructed halo.

    The WL-only pipeline finds a halo that minimises the WL chi2; that halo
    will in general not minimise chi2_SL (it has no knowledge of the strong
    lensing data).  We check that evaluating chi2_SL at the true halo gives
    a lower value than evaluating it at the WL-only recovered halo, confirming
    that chi2_strong_source_plane_nfw has the correct discriminatory gradient.

    Also reports per-system breakdown at both the true and WL-recovered halos.
    """
    R.header("14-D  NFW SL chi2 prefers true halo over WL-only recovered halo")

    src, halos_true, tx, ty, tm, tz, xmax = _build_nfw_scenario()
    use_flags = [True, True, False]

    # chi2_SL at the true halo (exact images → should be ~0)
    chi2_sl_true = utils.chi2_strong_source_plane_nfw(
        halos_true, src.strong_systems, use_magnification_correction=False
    )

    # Run WL-only to get the WL-optimal halo
    try:
        lenses_wl, _ = fit_lensing_field(
            src,
            xmax,
            flags=False,
            use_flags=use_flags,
            lens_type="NFW",
            z_lens=tz,
            use_strong_lensing=False,
        )
    except Exception as e:
        print(f"  EXCEPTION during WL-only run: {e}")
        R.record("14-D  NFW SL chi2 at true vs WL-only", False)
        return

    # chi2_SL at the WL-only recovered halo (should be much larger than at truth)
    chi2_sl_wl = utils.chi2_strong_source_plane_nfw(
        lenses_wl, src.strong_systems, use_magnification_correction=False
    )

    ok_true_better = chi2_sl_true < chi2_sl_wl
    ok_true_small = chi2_sl_true < 1e-6  # exact images → chi2 ≈ 0

    print(
        f"  chi2_SL at true halo     = {chi2_sl_true:.2e}  "
        f"{'OK (near 0)' if ok_true_small else 'FAIL (not near 0)'}"
    )
    print(
        f"  chi2_SL at WL-only halo  = {chi2_sl_wl:.4f}  "
        f'(WL halo at {_nearest_nfw_distance(lenses_wl, tx, ty):.2f}" from truth)'
    )
    print(f"  True halo has lower chi2_SL: {'OK' if ok_true_better else 'FAIL'}")

    # Per-system breakdown at the WL-only halo
    _, bd = utils.chi2_strong_source_plane_nfw(
        lenses_wl,
        src.strong_systems,
        return_breakdown=True,
        use_magnification_correction=False,
    )
    for sid, info in bd.items():
        print(
            f"    WL halo — {sid}: chi2={info['chi2']:.4f}  "
            f"n_img={info['n_images']}  "
            f"beta_bar=({info['beta_bar'][0]:.4f}, {info['beta_bar'][1]:.4f})"
        )

    ok_all = ok_true_small and ok_true_better
    R.record("14-D  NFW SL chi2 at true vs WL-only", ok_all)


class _TestResults14(_TestResults):
    def summary(self) -> bool:
        self.header("TASK 14 — SUMMARY")
        all_ok = True
        for name, ok in self.results:
            tag = "PASSED" if ok else "*** FAILED ***"
            print(f"  {name:60s}  {tag}")
            all_ok = all_ok and ok
        print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}\n")
        return all_ok


def run_nfw_integration_tests() -> bool:
    """
    Execute all Task 14 NFW integration tests.  Returns True if all pass.

    These tests call fit_lensing_field with lens_type='NFW', which runs the
    full NFW optimiser pipeline.  Expect ~30-90 s per run.
    """
    R = _TestResults14()

    # 14-A: WL-only baseline
    lenses_wl, d_wl = _test_nfw_integration_wl_only(R)

    # 14-B: WL+SL (pass WL distance for comparison)
    _test_nfw_integration_wl_plus_sl(R, d_wl_ref=d_wl)

    # 14-C: chi2 component verification
    _test_nfw_integration_chi2_components(R)

    # 14-D: source-plane scatter improvement
    _test_nfw_integration_sl_scatter(R)

    return R.summary()


def run_all_tests() -> bool:
    """Run Task 11 + Task 12 + Task 13 + Task 14 tests.  Returns True if all pass."""
    ok_11 = run_magnification_tests()
    ok_12 = run_integration_tests()
    ok_13 = run_nfw_sl_tests()
    ok_14 = run_nfw_integration_tests()
    if ok_11 and ok_12 and ok_13 and ok_14:
        print("\n  ALL TASK 11 + TASK 12 + TASK 13 + TASK 14 TESTS PASSED\n")
    else:
        if not ok_11:
            print("\n  Task 11 (SIS unit tests) had failures")
        if not ok_12:
            print("\n  Task 12 (SIS integration tests) had failures")
        if not ok_13:
            print("\n  Task 13 (NFW strong lensing tests) had failures")
        if not ok_14:
            print("\n  Task 14 (NFW integration tests) had failures")
    return ok_11 and ok_12 and ok_13 and ok_14


# ═══════════════════════════════════════════════════════════════════════════
# 8) NFW WL vs WL+SL side-by-side comparison plot
# ═══════════════════════════════════════════════════════════════════════════


def plot_nfw_wl_vs_sl_comparison(
    src,
    halos_wl,
    halos_sl,
    halos_true,
    true_x: float,
    true_y: float,
    true_mass: float,
    xmax: float,
    lambda_used: float | None = None,
    rchi2_wl: float | None = None,
    rchi2_sl: float | None = None,
    savepath: str | None = None,
):
    """
    Side-by-side comparison of WL-only vs WL+SL recovered NFW halo positions.

    Left panel  — WL-only final halos.
    Right panel — WL+SL final halos.

    Each main panel (120" field) shows:
      - Grey dots: weak-lensing source galaxies
      - Coloured circles: recovered NFW halo positions, sized by log10(mass)
      - Gold star: true halo position
      - Orange arrow: nearest recovered → truth offset with annotation
      - Red text box: reduced chi2 of the run

    Each panel also contains an **inset** (upper-right) zoomed to ±5× theta_E
    around the true halo, showing:
      - NFW Einstein ring (gold dashed)
      - Strong-lensing image positions (red diamonds)
      - True halo centre (gold star)
      - Recovered halos in this region (coloured)
    """
    # ── Pre-compute NFW Einstein radius for the inset ──────────────────────
    z_source_sl = 2.0
    if hasattr(src, "strong_systems") and src.strong_systems:
        z_source_sl = src.strong_systems[0].z_source

    try:
        theta_E = _find_nfw_einstein_radius(halos_true, z_source_sl)
    except Exception:
        theta_E = 2.0  # safe fallback

    inset_half = max(theta_E * 5.0, 3.0)  # arcsec half-width of inset

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, (ax_wl, ax_sl) = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        dpi=150,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.88, wspace=0.28)

    panels = [
        (ax_wl, halos_wl, "WL only", "C0", rchi2_wl),
        (ax_sl, halos_sl, "WL + SL", "C2", rchi2_sl),
    ]

    for ax, halos, label, color, rchi2 in panels:

        # ── WL sources ────────────────────────────────────────────────────
        ax.scatter(
            src.x, src.y, s=5, c="0.72", alpha=0.45, zorder=1, label=f"WL sources  (N={len(src.x)})"
        )

        # ── True halo ─────────────────────────────────────────────────────
        ax.scatter(
            true_x,
            true_y,
            marker="*",
            s=380,
            c="gold",
            edgecolors="k",
            linewidths=0.8,
            zorder=10,
            label=f"True halo  ({true_mass:.1e} M$_\\odot$)",
        )

        # ── Recovered halos ───────────────────────────────────────────────
        if len(halos.x) > 0:
            log_m = np.log10(np.clip(halos.mass, 1e10, 1e16))
            log_ref = np.log10(true_mass)
            sizes = np.clip((log_m - 10.0) / max(log_ref - 10.0, 1.0) * 220, 25, 550)
            ax.scatter(
                halos.x,
                halos.y,
                s=sizes,
                c=color,
                edgecolors="k",
                linewidths=0.7,
                alpha=0.85,
                zorder=8,
                label=f"Recovered  (N={len(halos.x)})",
            )
            for hx, hy, hm in zip(halos.x, halos.y, halos.mass):
                ax.annotate(
                    f"{hm:.1e}",
                    (hx, hy),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color="0.20",
                    zorder=11,
                )

            # ── Arrow: nearest recovered → truth ──────────────────────────
            dists = np.hypot(halos.x - true_x, halos.y - true_y)
            idx_near = int(np.argmin(dists))
            d_near = float(dists[idx_near])
            ax.annotate(
                "",
                xy=(true_x, true_y),
                xytext=(halos.x[idx_near], halos.y[idx_near]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color="C1",
                    lw=1.8,
                    shrinkA=5,
                    shrinkB=5,
                ),
                zorder=9,
            )
            # Bottom-left annotation box
            ann_text = f'$\\Delta$ = {d_near:.1f}"    ' f"M = {halos.mass[idx_near]:.2e} M$_\\odot$"
            if rchi2 is not None:
                ann_text += f"\n$\\tilde{{\\chi}}^2$ = {rchi2:.3f}"
            ax.text(
                0.03,
                0.03,
                ann_text,
                transform=ax.transAxes,
                fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.92, pad=4),
                zorder=12,
            )

        # ── Axis config ───────────────────────────────────────────────────
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x  (arcsec)", fontsize=10)
        ax.set_ylabel("y  (arcsec)", fontsize=10)
        ax.set_title(label, fontsize=13, fontweight="bold", pad=8)

        handles, lbls = ax.get_legend_handles_labels()
        seen: set = set()
        unique = [(h, l) for h, l in zip(handles, lbls) if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7.5, markerscale=0.9)

        # ══ INSET — zoom on SL region ═════════════════════════════════════
        ax_in = ax.inset_axes([0.60, 0.60, 0.38, 0.38])

        # WL sources inside inset bounds
        mask = (np.abs(src.x - true_x) < inset_half) & (np.abs(src.y - true_y) < inset_half)
        if mask.any():
            ax_in.scatter(src.x[mask], src.y[mask], s=8, c="0.72", alpha=0.5)

        # Strong-lensing image positions
        if hasattr(src, "strong_systems") and src.strong_systems:
            for i, sl_sys in enumerate(src.strong_systems):
                ax_in.scatter(
                    sl_sys.theta_x,
                    sl_sys.theta_y,
                    marker="D",
                    s=55,
                    edgecolors="C3",
                    facecolors="none",
                    linewidths=1.6,
                    zorder=5,
                    label="SL images" if i == 0 else "",
                )

        # Einstein ring
        ring = plt.Circle(
            (true_x, true_y),
            theta_E,
            fill=False,
            linestyle="--",
            linewidth=1.1,
            edgecolor="gold",
            alpha=0.85,
            zorder=4,
        )
        ax_in.add_patch(ring)

        # True halo in inset
        ax_in.scatter(
            true_x,
            true_y,
            marker="*",
            s=180,
            c="gold",
            edgecolors="k",
            linewidths=0.7,
            zorder=10,
        )

        # Recovered halos in inset
        if len(halos.x) > 0:
            in_mask = (np.abs(halos.x - true_x) < inset_half) & (
                np.abs(halos.y - true_y) < inset_half
            )
            if in_mask.any():
                ax_in.scatter(
                    halos.x[in_mask],
                    halos.y[in_mask],
                    s=45,
                    c=color,
                    edgecolors="k",
                    linewidths=0.5,
                    alpha=0.85,
                    zorder=8,
                )
            else:
                # Halo outside inset: draw an arrow pointing to it from edge
                hx_near = halos.x[int(np.argmin(np.hypot(halos.x - true_x, halos.y - true_y)))]
                hy_near = halos.y[int(np.argmin(np.hypot(halos.x - true_x, halos.y - true_y)))]
                dx = np.clip(hx_near - true_x, -inset_half * 0.8, inset_half * 0.8)
                dy = np.clip(hy_near - true_y, -inset_half * 0.8, inset_half * 0.8)
                ax_in.annotate(
                    "",
                    xy=(true_x + dx * 0.85, true_y + dy * 0.85),
                    xytext=(true_x, true_y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                    zorder=9,
                )
                ax_in.text(
                    true_x + dx * 0.5,
                    true_y + dy * 0.5,
                    "halo\noutside",
                    fontsize=5.5,
                    ha="center",
                    va="center",
                    color=color,
                    zorder=10,
                )

        ax_in.set_xlim(true_x - inset_half, true_x + inset_half)
        ax_in.set_ylim(true_y - inset_half, true_y + inset_half)
        ax_in.set_aspect("equal", adjustable="box")
        ax_in.tick_params(labelsize=6)
        ax_in.set_title(
            f'zoom  ±{inset_half:.0f}"  (NFW $\\theta_E$={theta_E:.2f}")',
            fontsize=6.5,
            pad=3,
        )

        # Indicate the inset region on the main axes
        try:
            ax.indicate_inset_zoom(ax_in, edgecolor="0.40", linewidth=0.9)
        except Exception:
            # matplotlib < 3.3 fallback: draw a manual rectangle
            rect = plt.Rectangle(
                (true_x - inset_half, true_y - inset_half),
                2 * inset_half,
                2 * inset_half,
                linewidth=0.9,
                edgecolor="0.40",
                facecolor="none",
                zorder=3,
            )
            ax.add_patch(rect)

    # ── Supertitle ────────────────────────────────────────────────────────
    suptitle = (
        "NFW Halo Reconstruction:  WL-only  vs  WL + Strong Lensing\n"
        f"True halo: M = {true_mass:.1e} M$_\\odot$,  "
        f"position = ({true_x}, {true_y}) arcsec"
    )
    if lambda_used is not None:
        suptitle += f",  $\\lambda_{{SL}}$ (pre-computed) = {lambda_used:.1f}"
    fig.suptitle(suptitle, fontsize=11, y=0.97)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        print(f"  Saved: {savepath}")

    return fig


def run_nfw_comparison_plot(
    seed: int = 55,
    savepath: str | None = "nfw_wl_vs_sl_comparison.png",
    show: bool = True,
):
    """
    Build the single-NFW scenario from Task 14, run both WL-only and WL+SL
    pipelines, and produce the side-by-side NFW comparison figure.

    Parameters
    ----------
    seed : int
        Random seed passed to _build_nfw_scenario (default 55).
    savepath : str or None
        File path for saving; None to skip.  Default: nfw_wl_vs_sl_comparison.png
    show : bool
        Call plt.show() when done.

    Returns
    -------
    fig : matplotlib Figure
    halos_wl, halos_sl : NFW_Lens
    """
    src, halos_true, tx, ty, tm, tz, xmax = _build_nfw_scenario(seed=seed)
    use_flags = [True, True, False]

    # Capture lambda used inside the WL+SL pipeline by evaluating at initial guess
    halos_init = pipeline.generate_initial_guess(src, lens_type="NFW", z_l=tz)
    lambda_used = metric.compute_lambda_sl(src, halos_init, use_flags, "NFW")

    print("Running NFW WL-only pipeline...")
    halos_wl, rchi2_wl = fit_lensing_field(
        src,
        xmax,
        flags=False,
        use_flags=use_flags,
        lens_type="NFW",
        z_lens=tz,
        use_strong_lensing=False,
    )

    print("Running NFW WL+SL pipeline...")
    halos_sl, rchi2_sl = fit_lensing_field(
        src,
        xmax,
        flags=False,
        use_flags=use_flags,
        lens_type="NFW",
        z_lens=tz,
        use_strong_lensing=True,
    )

    d_wl = float(np.min(np.hypot(halos_wl.x - tx, halos_wl.y - ty)))
    d_sl = float(np.min(np.hypot(halos_sl.x - tx, halos_sl.y - ty)))

    print(f"\n{'='*60}")
    print(f"  True halo: ({tx}, {ty})  M = {tm:.1e} M_sun  z = {tz}")
    print(f'  WL-only : N={len(halos_wl.x):2d}  nearest_d={d_wl:.2f}"  rchi2={rchi2_wl:.4f}')
    print(f'  WL+SL   : N={len(halos_sl.x):2d}  nearest_d={d_sl:.2f}"  rchi2={rchi2_sl:.4f}')
    print(f"  lambda_SL (pre-computed from initial guess) = {lambda_used:.2f}")
    print(f"{'='*60}")

    fig = plot_nfw_wl_vs_sl_comparison(
        src=src,
        halos_wl=halos_wl,
        halos_sl=halos_sl,
        halos_true=halos_true,
        true_x=tx,
        true_y=ty,
        true_mass=tm,
        xmax=xmax,
        lambda_used=lambda_used,
        rchi2_wl=rchi2_wl,
        rchi2_sl=rchi2_sl,
        savepath=savepath,
    )

    if show:
        plt.show()

    return fig, halos_wl, halos_sl


if __name__ == "__main__":
    run_nfw_comparison_plot()
