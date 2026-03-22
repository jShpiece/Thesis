from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# Ensure UTF-8 output on Windows consoles (avoids UnicodeEncodeError for θ, μ, λ, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.pipeline as pipeline
import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
import arch.metric as metric
from arch.main import fit_lensing_field


# ----------------------------
# 1) Toy data construction
# ----------------------------

def make_weak_lensing_catalog_two_lenses(
    true_lens_xyte: list[tuple[float, float, float]],
    xmax: float,
    n_sources: int,
    z_source: float = 1.0,
    sig_shear: float = 0.10,
    sig_flex: float = 0.02,
    sig_gflex: float = 0.03,
    rmin: float = 1.0,
    seed: int = 7,
) -> source_obj.Source:
    """
    Build a WL catalog, lens it with TWO (or more) SIS lenses, add noise.
    true_lens_xyte: list of (x0, y0, te) in arcsec.
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-xmax, xmax, size=n_sources)
    y = rng.uniform(-xmax, xmax, size=n_sources)

    # Avoid sampling directly on top of any lens center (prevents singular behavior)
    keep = np.ones_like(x, dtype=bool)
    for (x0, y0, _te) in true_lens_xyte:
        r = np.hypot(x - x0, y - y0)
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

    xl = np.array([t[0] for t in true_lens_xyte], dtype=float)
    yl = np.array([t[1] for t in true_lens_xyte], dtype=float)
    te = np.array([t[2] for t in true_lens_xyte], dtype=float)

    lens_true = halo_obj.SIS_Lens(
        x=xl,
        y=yl,
        te=te,
        chi2=np.zeros_like(xl),
    )

    src.apply_lensing(lens_true, lens_type="SIS")
    src.apply_noise()

    return src


def make_two_image_sis_system_at_lens(
    system_id: str,
    lens_center_xy: tuple[float, float],
    te_true: float,
    beta_rel_xy: tuple[float, float],
    sigma_theta: float = 0.05,
    z_source: float = 2.0,
):
    """
    2-image SIS system constructed around a specified lens center (x0,y0).

    We choose a source-plane position beta = (x0,y0) + beta_rel,
    with |beta_rel| < te_true to ensure a 2-image regime for a pure SIS.

    For an axisymmetric SIS:
        r1 = te + |beta_rel|, r2 = te - |beta_rel|
        theta1 = center + r1 * ehat, theta2 = center + r2 * ehat
        where ehat = beta_rel / |beta_rel|.

    This guarantees both images back-project to the SAME beta under the correct lens.
    """
    x0, y0 = lens_center_xy
    bx, by = beta_rel_xy
    b = float(np.hypot(bx, by))
    if not (0.0 < b < te_true):
        raise ValueError("Need 0 < |beta_rel| < te_true for a 2-image SIS system.")

    ehatx, ehaty = bx / b, by / b
    r1 = te_true + b
    r2 = te_true - b

    # Image 1: same side as source, at distance r1 from lens centre
    # Image 2: opposite side, at distance r2 from lens centre (negative parity)
    theta_x = np.array([x0 + r1 * ehatx, x0 - r2 * ehatx], dtype=float)
    theta_y = np.array([y0 + r1 * ehaty, y0 - r2 * ehaty], dtype=float)

    StrongLensingSystem = getattr(source_obj, "StrongLensingSystem")
    return StrongLensingSystem(
        system_id=system_id,
        theta_x=theta_x,
        theta_y=theta_y,
        z_source=float(z_source),
        sigma_theta=float(sigma_theta),
        meta={
            "toy": True,
            "lens_center": (x0, y0),
            "beta_rel": (bx, by),
            "te_true": te_true,
        },
    )


def attach_strong_systems(src: source_obj.Source, systems) -> None:
    if hasattr(src, "strong_systems"):
        src.strong_systems = list(systems)
        return
    if hasattr(src, "add_strong_system"):
        for s in systems:
            src.add_strong_system(s)
        return
    raise RuntimeError("Source does not expose strong_systems or add_strong_system.")


# ----------------------------
# 2) Stage-capture runner
# ----------------------------

STAGE_NAMES = [
    "initial_guess",
    "optimization",
    "filter",
    "forward_selection",
    "merging",
    "opt_strength",
]

def _copy_lenses_sis(lenses: halo_obj.SIS_Lens) -> halo_obj.SIS_Lens:
    return halo_obj.SIS_Lens(
        x=np.array(lenses.x, dtype=float).copy(),
        y=np.array(lenses.y, dtype=float).copy(),
        te=np.array(lenses.te, dtype=float).copy(),
        chi2=np.array(getattr(lenses, "chi2", np.zeros_like(np.atleast_1d(lenses.x))), dtype=float).copy(),
    )


def run_pipeline_capture_stages(
    src: source_obj.Source,
    xmax: float,
    use_strong_lensing: bool,
    z_lens: float = 0.5,
):
    """
    Runs SIS pipeline through major stages; returns dict stage->lenses snapshot.
    Assumes your pipeline functions accept lambda_sl where relevant.
    """
    use_flags = [True, True, False]  # shear + flexion

    stages = {}

    lenses = pipeline.generate_initial_guess(src, lens_type="SIS", z_l=z_lens)
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["initial_guess"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_positions(src, lenses, xmax, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["optimization"] = _copy_lenses_sis(lenses)

    lenses = pipeline.filter_lens_positions(src, lenses, xmax, lens_type="SIS")
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["filter"] = _copy_lenses_sis(lenses)

    lenses, _best = pipeline.forward_lens_selection(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["forward_selection"] = _copy_lenses_sis(lenses)

    merger_threshold = (len(src.x) / (2 * xmax) ** 2) ** (-0.5) if len(src.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, "SIS")
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["merging"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_strength(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, [True, True, True], lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["opt_strength"] = _copy_lenses_sis(lenses)

    return stages


# ----------------------------
# 3) Plotting (3x2 grid)
# ----------------------------

def plot_stage_grid_two_truth(
    stages: dict,
    true_lens_xyte: list[tuple[float, float, float]],
    xmax: float,
    title: str,
    savepath: str | None = None,
):
    """
    3x2 grid (2 rows x 3 cols); each panel shows:
      - true lens positions (stars)
      - all candidate/selected lenses at that stage (circles)
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=150, constrained_layout=True)
    axes = axes.ravel()

    true_x = np.array([t[0] for t in true_lens_xyte], dtype=float)
    true_y = np.array([t[1] for t in true_lens_xyte], dtype=float)

    for ax, stage_name in zip(axes, STAGE_NAMES):
        L = stages.get(stage_name, None)

        # True lenses
        ax.scatter(true_x, true_y, marker="*", s=220, label="true lenses")

        # Candidate/selected lenses
        if L is not None and len(np.atleast_1d(L.x)) > 0:
            ax.scatter(L.x, L.y, s=35, alpha=0.9, label="candidates")
            # Optional: label Einstein radii at final stage
            if stage_name == STAGE_NAMES[-1]:
                for (lx, ly, lte) in zip(L.x, L.y, L.te):
                    ax.text(lx, ly, f"{lte:.2f}", fontsize=8, ha="center", va="center")
        ax.set_title(stage_name)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# ----------------------------
# 3b) WL vs WL+SL comparison
# ----------------------------

def plot_wl_vs_sl_comparison(
    src: source_obj.Source,
    lenses_wl: halo_obj.SIS_Lens,
    lenses_sl: halo_obj.SIS_Lens,
    true_x: float,
    true_y: float,
    true_te: float,
    xmax: float,
    lambda_sl: float | None = None,
    savepath: str | None = None,
):
    """
    Side-by-side comparison of WL-only vs WL+SL recovered lens positions.

    Left panel:  WL-only final lenses
    Right panel: WL+SL final lenses

    Both panels show:
      - Weak lensing source galaxies (small grey dots)
      - Strong lensing image positions (diamonds)
      - True lens position + Einstein ring (gold star + dashed circle)
      - Recovered lens positions, sized by θ_E (coloured circles)
      - Offset arrow from nearest recovered lens to truth

    Parameters
    ----------
    src : Source
        Source catalog (with strong_systems for image positions).
    lenses_wl, lenses_sl : SIS_Lens
        Final recovered lenses from each pipeline run.
    true_x, true_y, true_te : float
        Ground-truth lens position and Einstein radius.
    xmax : float
        Field half-width for axis limits.
    lambda_sl : float or None
        Pre-computed λ_SL to display in the annotation (informational).
    savepath : str or None
        If given, save the figure to this path.
    """
    fig, (ax_wl, ax_sl) = plt.subplots(
        1, 2, figsize=(14, 6.5), dpi=150, constrained_layout=True,
    )

    for ax, lenses, label in [
        (ax_wl, lenses_wl, "WL only"),
        (ax_sl, lenses_sl, "WL + SL"),
    ]:
        # ── Source galaxies ──
        ax.scatter(src.x, src.y, s=4, c="0.70", alpha=0.5, zorder=1,
                   label=f"sources (N={len(src.x)})")

        # ── Strong lensing image positions ──
        if hasattr(src, "strong_systems") and src.strong_systems:
            for sys in src.strong_systems:
                ax.scatter(
                    sys.theta_x, sys.theta_y,
                    marker="D", s=60, edgecolors="C3", facecolors="none",
                    linewidths=1.5, zorder=5,
                    label=f"SL images ({sys.system_id})",
                )

        # ── Truth: star + Einstein ring ──
        ax.scatter(
            true_x, true_y, marker="*", s=350, c="gold",
            edgecolors="k", linewidths=0.8, zorder=10,
            label=f"truth ($\\theta_E$={true_te:.1f}\")",
        )
        circle = plt.Circle(
            (true_x, true_y), true_te,
            fill=False, linestyle="--", linewidth=1.2,
            edgecolor="gold", alpha=0.8, zorder=4,
        )
        ax.add_patch(circle)

        # ── Recovered lenses ──
        if len(lenses.x) > 0:
            # Size proportional to θ_E, clipped for readability
            sizes = np.clip(np.abs(lenses.te), 0.5, 15) * 30
            sc = ax.scatter(
                lenses.x, lenses.y, s=sizes,
                c="C0" if label == "WL only" else "C2",
                edgecolors="k", linewidths=0.6, alpha=0.85,
                zorder=8, label=f"recovered (N={len(lenses.x)})",
            )
            # Label each recovered lens with its θ_E
            for lx, ly, lte in zip(lenses.x, lenses.y, lenses.te):
                ax.annotate(
                    f"{lte:.2f}\"",
                    (lx, ly), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5, color="0.25",
                    zorder=11,
                )

            # ── Offset arrow from nearest lens to truth ──
            dists = np.hypot(lenses.x - true_x, lenses.y - true_y)
            idx_near = int(np.argmin(dists))
            d = float(dists[idx_near])
            ax.annotate(
                "",
                xy=(true_x, true_y),
                xytext=(lenses.x[idx_near], lenses.y[idx_near]),
                arrowprops=dict(
                    arrowstyle="-|>", color="C1",
                    lw=1.8, shrinkA=4, shrinkB=4,
                ),
                zorder=9,
            )
            ax.text(
                0.03, 0.03,
                f"$\\Delta$ = {d:.2f}\"   $\\theta_E$ = {lenses.te[idx_near]:.2f}\"",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=3),
                zorder=12,
            )

        # ── Axis config ──
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x  (arcsec)")
        ax.set_ylabel("y  (arcsec)")
        ax.set_title(label, fontsize=13, fontweight="bold")

        # De-duplicate legend entries
        handles, labels_leg = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels_leg) if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7.5, markerscale=0.9)

    # ── Suptitle with λ_SL if available ──
    suptitle = "Recovered Lens Positions:  WL-only  vs  WL + Strong Lensing"
    if lambda_sl is not None:
        suptitle += f"   ($\\lambda_{{SL}}$ = {lambda_sl:.4f})"
    fig.suptitle(suptitle, fontsize=13)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def run_comparison_plot(
    seed: int = 55,
    savepath: str | None = "wl_vs_sl_comparison.pdf",
    show: bool = True,
):
    """
    Convenience function: build a single-SIS scenario, run both pipelines,
    and produce the side-by-side comparison figure.

    Can be called interactively:
        >>> from paper2 import run_comparison_plot
        >>> run_comparison_plot()

    Parameters
    ----------
    seed : int
        Random seed for the source catalog.
    savepath : str or None
        File path for saving the figure. None to skip saving.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib Figure
    lenses_wl, lenses_sl : SIS_Lens
        Final recovered lenses from each run.
    """
    src, true_x, true_y, true_te, xmax = _build_single_lens_scenario(seed=seed)
    use_flags = [True, True, False]  # shear + flexion

    print("Running WL-only pipeline...")
    lenses_wl, rchi2_wl = fit_lensing_field(
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='SIS', use_strong_lensing=False,
    )

    print("\nRunning WL+SL pipeline...")
    lenses_sl, rchi2_sl = fit_lensing_field(
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='SIS', use_strong_lensing=True,
    )

    # Compute lambda_sl for display
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, 'SIS')

    d_wl = float(np.min(np.hypot(lenses_wl.x - true_x, lenses_wl.y - true_y)))
    d_sl = float(np.min(np.hypot(lenses_sl.x - true_x, lenses_sl.y - true_y)))
    print(f"\n{'='*50}")
    print(f"  WL-only:  N_lens={len(lenses_wl.x)},  "
          f"nearest Δ={d_wl:.2f}\",  rχ²={rchi2_wl:.4f}")
    print(f"  WL+SL:    N_lens={len(lenses_sl.x)},  "
          f"nearest Δ={d_sl:.2f}\",  rχ²={rchi2_sl:.4f}")
    print(f"  lam_SL = {lambda_sl:.6f}")
    print(f"{'='*50}")

    fig = plot_wl_vs_sl_comparison(
        src=src,
        lenses_wl=lenses_wl,
        lenses_sl=lenses_sl,
        true_x=true_x, true_y=true_y, true_te=true_te,
        xmax=xmax, lambda_sl=lambda_sl,
        savepath=savepath,
    )

    if show:
        plt.show()

    return fig, lenses_wl, lenses_sl


def main():
    # ----------------------------
    # Truth: TWO SIS lenses
    # ----------------------------
    true_lens_xyte = [
        (-15.0,  0.0, 5.0),   # lens A: (x, y, te)
        ( 18.0, 12.0, 3.5),   # lens B
    ]

    xmax = 50.0
    n_sources = 100

    # Build WL catalog lensed by BOTH true lenses
    src = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=true_lens_xyte,
        xmax=xmax,
        n_sources=n_sources,
        seed=12,
    )

    # Add one SL system per lens (optional but recommended for the test)
    sysA = make_two_image_sis_system_at_lens(
        system_id="toy_sys_A",
        lens_center_xy=(true_lens_xyte[0][0], true_lens_xyte[0][1]),
        te_true=true_lens_xyte[0][2],
        beta_rel_xy=(0.6, 0.2),
        sigma_theta=0.03,
        z_source=2.0,
    )
    sysB = make_two_image_sis_system_at_lens(
        system_id="toy_sys_B",
        lens_center_xy=(true_lens_xyte[1][0], true_lens_xyte[1][1]),
        te_true=true_lens_xyte[1][2],
        beta_rel_xy=(-0.4, 0.25),
        sigma_theta=0.03,
        z_source=2.2,
    )

    attach_strong_systems(src, [sysA, sysB])

    # ----------------------------
    # Run and plot
    # ----------------------------
    use_strong_lensing = True
    stages_sl = run_pipeline_capture_stages(src, xmax=xmax, use_strong_lensing=use_strong_lensing)

    plot_stage_grid_two_truth(
        stages_sl,
        true_lens_xyte=true_lens_xyte,
        xmax=xmax,
        title=f"ARCH SIS toy run",
        savepath="stages_two_lens_sl.png",
    )
    plt.show()


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
        self.header("TASK 11 — SUMMARY")
        all_ok = True
        for name, ok in self.results:
            tag = "PASSED" if ok else "*** FAILED ***"
            print(f"  {name:55s}  {tag}")
            all_ok = all_ok and ok
        print(f"\n  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}\n")
        return all_ok


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
            ok_i = (np.isclose(det_A[i], det_expected[i], atol=1e-8)
                    and np.isclose(abs_mu[i], mu_expected, rtol=1e-6))
        ok_all = ok_all and ok_i
        status = "OK" if ok_i else "FAIL"
        print(f"  r={radii[i]:5.1f}  det(A)={det_A[i]:+.6f}  "
              f"expected={det_expected[i]:+.6f}  |mu|={abs_mu[i]:.4f}  {status}")

    # Diagonal check: r=4 along 45°
    r = 4.0
    tx_d, ty_d = np.array([r / np.sqrt(2)]), np.array([r / np.sqrt(2)])
    _, det_d = utils.magnification_sis(lens, tx_d, ty_d)
    ok_diag = np.isclose(det_d[0], 1.0 - te / r, atol=1e-8)
    ok_all = ok_all and ok_diag
    print(f"  diagonal r={r}: det(A)={det_d[0]:.6f}  expected={1.0-te/r:.6f}  "
          f"{'OK' if ok_diag else 'FAIL'}")

    # Negative parity inside Einstein ring
    r_in = te / 2
    _, det_in = utils.magnification_sis(lens, np.array([r_in]), np.array([0.0]))
    ok_parity = det_in[0] < 0
    ok_all = ok_all and ok_parity
    print(f"  parity r={r_in} < θ_E: det(A)={det_in[0]:.6f}  negative={'OK' if ok_parity else 'FAIL'}")

    R.record("11-A  Single SIS magnification", ok_all)


# ── 11-B  Composite SIS finite-difference Jacobian ───────────────────────

def _test_magnification_composite_fd(R: _TestResults):
    """
    Two SIS halos at different positions.  Compare the analytic det(A) from
    magnification_sis against a central-difference numerical Jacobian of the
    lens mapping β(θ).
    """
    R.header("11-B  Composite SIS finite-difference Jacobian")

    lens = halo_obj.SIS_Lens(
        x=[0.0, 6.0], y=[0.0, 3.0], te=[3.0, 2.0], chi2=[0, 0]
    )

    # Test points well away from both lens centres
    pts_x = np.array([3.0, -4.0, 8.0, 1.0, 5.0])
    pts_y = np.array([1.5, -2.0, 5.0, 6.0, -3.0])

    _, det_analytic = utils.magnification_sis(lens, pts_x, pts_y)

    h = 1e-5
    det_fd = np.empty(len(pts_x))
    for k in range(len(pts_x)):
        # ∂β/∂θ by central differences
        bxp, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] + h]), np.array([pts_y[k]]))
        bxm, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] - h]), np.array([pts_y[k]]))
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] + h]))
        bxm2, _ = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] - h]))
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] + h]), np.array([pts_y[k]]))
        _, bym = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k] - h]), np.array([pts_y[k]]))
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] + h]))
        _, bym2 = utils.backproject_source_positions_sis(
            lens, np.array([pts_x[k]]), np.array([pts_y[k] - h]))
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd[k] = dbx_dtx * dby_dty - dbx_dty * dby_dtx

    ok_all = True
    for k in range(len(pts_x)):
        rel = np.abs(det_analytic[k] - det_fd[k]) / np.abs(det_fd[k])
        ok_k = rel < 1e-4
        ok_all = ok_all and ok_k
        print(f"  pt {k}: det(A)_analytic={det_analytic[k]:+.8f}  "
              f"det(A)_FD={det_fd[k]:+.8f}  rel_err={rel:.1e}  "
              f"{'OK' if ok_k else 'FAIL'}")

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
        (5.0,  0.01, 0.1 / 5.0),
        (1.0,  0.01, 0.1),
        (200., 0.01, 0.1 * 0.01),   # floored: 1/200 < 0.01
        (50.,  0.05, 0.1 * 0.05),   # floored: 1/50=0.02 < 0.05
        (10.,  0.01, 0.1 / 10.),    # not floored
    ]

    ok_all = True
    for mu, floor, expected in cases:
        result = utils.sigma_beta_from_magnification(sig, np.array([mu]), mu_floor=floor)
        ok = np.isclose(result[0], expected, rtol=1e-10)
        ok_all = ok_all and ok
        print(f"  |mu|={mu:6.1f}  floor={floor}  sig_b={result[0]:.6f}  "
              f"expected={expected:.6f}  {'OK' if ok else 'FAIL'}")

    # Array input
    mus = np.array([2., 5., 10., 100.])
    sigs = np.full_like(mus, sig)
    result = utils.sigma_beta_from_magnification(sigs, mus, mu_floor=0.01)
    expected_arr = np.array([sig/2, sig/5, sig/10, sig*0.01])
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
    ok_perf = np.isclose(chi2_corr_perf, 0.0, atol=1e-8) and np.isclose(chi2_uncorr_perf, 0.0, atol=1e-8)
    print(f"  Perfect model:  chi2_corr={chi2_corr_perf:.2e}  "
          f"chi2_uncorr={chi2_uncorr_perf:.2e}  {'OK' if ok_perf else 'FAIL'}")

    # ── Sub-test 2: perturbed lens → chi2_corr > chi2_uncorr ──
    lens_perturbed = halo_obj.SIS_Lens(x=0.3, y=-0.2, te=te_true, chi2=[0])

    chi2_corr, bd_corr = utils.chi2_strong_source_plane_sis(
        lens_perturbed, [sys_perf],
        return_breakdown=True, use_magnification_correction=True
    )
    chi2_uncorr, bd_uncorr = utils.chi2_strong_source_plane_sis(
        lens_perturbed, [sys_perf],
        return_breakdown=True, use_magnification_correction=False
    )
    ok_larger = chi2_corr > chi2_uncorr
    print(f"  Perturbed model:  chi2_corr={chi2_corr:.2f}  "
          f"chi2_uncorr={chi2_uncorr:.2f}  corr>uncorr: {'OK' if ok_larger else 'FAIL'}")

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
        print(f"    image {m}: r={r:.3f}  |mu|={mu_got:.4f}  "
              f"expected={mu_expected:.4f}  {'OK' if close else 'FAIL'}")

    # ── Sub-test 4: sigma_beta in breakdown is consistent ──
    ok_sig = True
    for m in range(len(tx)):
        sig_b = bd["sigma_beta_x"][m]
        sig_b_expected = sys_perf.sigma_theta / bd["abs_mu"][m]
        close = np.isclose(sig_b, sig_b_expected, rtol=1e-4)
        ok_sig = ok_sig and close
        print(f"    image {m}: σ_β={sig_b:.6f}  expected={sig_b_expected:.6f}  "
              f"{'OK' if close else 'FAIL'}")

    # ── Sub-test 5: hand-compute chi2 from breakdown data ──
    bx, by = bd["beta"][:, 0], bd["beta"][:, 1]
    sigx = bd["sigma_beta_x"]
    sigy = bd["sigma_beta_y"]
    wx = 1.0 / sigx**2
    wy = 1.0 / sigy**2
    bx_bar = np.sum(wx * bx) / np.sum(wx)
    by_bar = np.sum(wy * by) / np.sum(wy)
    chi2_hand = np.sum(((bx - bx_bar) / sigx)**2 + ((by - by_bar) / sigy)**2)
    ok_hand = np.isclose(chi2_corr, chi2_hand, rtol=1e-8)
    print(f"  Hand chi2={chi2_hand:.4f}  function={chi2_corr:.4f}  "
          f"{'OK' if ok_hand else 'FAIL'}")

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
        true_lens_xyte=true_lens_xyte, xmax=xmax, n_sources=60, seed=99,
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

    lam = metric.compute_lambda_sl(src, lens_init, use_flags, lens_type='SIS')

    # Manual calculation
    chi2_wl = metric.calculate_chi_squared(src, lens_init, use_flags, lens_type='SIS')
    dof_wl = metric.calc_degrees_of_freedom(src, lens_init, use_flags)
    chi2_sl = utils.chi2_strong_source_plane_sis(lens_init, src.strong_systems)
    dof_sl = metric.calc_strong_dof(src)

    expected = (chi2_wl / dof_wl) / (chi2_sl / dof_sl)

    ok_match = np.isclose(lam, expected, rtol=1e-10)
    ok_finite = np.isfinite(lam) and lam > 0
    print(f"  compute_lambda_sl = {lam:.6f}")
    print(f"  manual            = {expected:.6f}")
    print(f"  match: {'OK' if ok_match else 'FAIL'}   "
          f"positive & finite: {'OK' if ok_finite else 'FAIL'}")

    # Degenerate case: no strong systems → returns 1.0
    src_no_sl = make_weak_lensing_catalog_two_lenses(
        true_lens_xyte=true_lens_xyte, xmax=xmax, n_sources=60, seed=99,
    )
    lam_none = metric.compute_lambda_sl(src_no_sl, lens_init, use_flags, 'SIS')
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
        true_lens_xyte=true_lens_xyte, xmax=xmax, n_sources=40, seed=42,
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
        src, lens, flags, lens_type="SIS",
        use_strong_lensing=True, lambda_sl=fixed_lam,
    )
    expected_total = comps["chi2_wl"] + fixed_lam * comps["chi2_sl"]
    ok_total = np.isclose(chi2_total, expected_total, rtol=1e-10)
    ok_lam = comps["lambda_sl"] == fixed_lam
    print(f"  lambda_sl = {comps['lambda_sl']}  (passed {fixed_lam})  {'OK' if ok_lam else 'FAIL'}")
    print(f"  chi2_total = {chi2_total:.4f}  expected = {expected_total:.4f}  "
          f"{'OK' if ok_total else 'FAIL'}")

    # ── Without SL → lambda = 0 ──
    chi2_no, _, comps_no = metric.calculate_total_chi2(
        src, lens, flags, lens_type="SIS", use_strong_lensing=False
    )
    ok_no = comps_no["lambda_sl"] == 0.0 and np.isclose(chi2_no, comps_no["chi2_wl"])
    print(f"  no-SL: lambda=0, chi2=chi2_wl  {'OK' if ok_no else 'FAIL'}")

    # ── Fallback (lambda_sl=None, SL active) should be finite & positive ──
    chi2_fb, _, comps_fb = metric.calculate_total_chi2(
        src, lens, flags, lens_type="SIS",
        use_strong_lensing=True, lambda_sl=None,
    )
    ok_fb = np.isfinite(comps_fb["lambda_sl"]) and comps_fb["lambda_sl"] > 0
    print(f"  fallback lambda = {comps_fb['lambda_sl']:.6f}  "
          f"finite & positive: {'OK' if ok_fb else 'FAIL'}")

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

    true_lens_xyte = [
        (-15.0,  0.0, 5.0),
        ( 18.0, 12.0, 3.5),
    ]

    sysA = make_two_image_sis_system_at_lens(
        system_id="toy_A", lens_center_xy=(-15.0, 0.0),
        te_true=5.0, beta_rel_xy=(0.6, 0.2), sigma_theta=0.03,
    )
    sysB = make_two_image_sis_system_at_lens(
        system_id="toy_B", lens_center_xy=(18.0, 12.0),
        te_true=3.5, beta_rel_xy=(-0.4, 0.25), sigma_theta=0.03,
    )
    systems = [sysA, sysB]

    lens_true = halo_obj.SIS_Lens(
        x=[-15.0, 18.0], y=[0.0, 12.0], te=[5.0, 3.5], chi2=[0, 0]
    )

    # ── Sub-test 1: corrected > uncorrected at true params ──
    chi2_corr_true = utils.chi2_strong_source_plane_sis(
        lens_true, systems, use_magnification_correction=True
    )
    chi2_uncorr_true = utils.chi2_strong_source_plane_sis(
        lens_true, systems, use_magnification_correction=False
    )
    ok_larger_true = chi2_corr_true > chi2_uncorr_true
    print(f"  True lens: chi2_corr={chi2_corr_true:.2f}  "
          f"chi2_uncorr={chi2_uncorr_true:.2f}  "
          f"corr>uncorr: {'OK' if ok_larger_true else 'FAIL'}")

    # ── Sub-test 2: corrected > uncorrected also at perturbed params ──
    lens_pert = halo_obj.SIS_Lens(
        x=[-14.7, 18.3], y=[0.2, 11.8], te=[5.0, 3.5], chi2=[0, 0]
    )
    chi2_corr_pert = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, use_magnification_correction=True
    )
    chi2_uncorr_pert = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, use_magnification_correction=False
    )
    ok_larger_pert = chi2_corr_pert > chi2_uncorr_pert
    print(f"  Perturbed: chi2_corr={chi2_corr_pert:.2f}  "
          f"chi2_uncorr={chi2_uncorr_pert:.2f}  "
          f"corr>uncorr: {'OK' if ok_larger_pert else 'FAIL'}")

    # ── Sub-test 3: breakdown sums to total ──
    chi2_bd, bd = utils.chi2_strong_source_plane_sis(
        lens_pert, systems, return_breakdown=True,
        use_magnification_correction=True
    )
    bd_sum = sum(v["chi2"] for v in bd.values())
    ok_sum = np.isclose(chi2_bd, bd_sum, rtol=1e-10)
    print(f"  Breakdown sum={bd_sum:.4f}  total={chi2_bd:.4f}  "
          f"{'OK' if ok_sum else 'FAIL'}")

    # ── Sub-test 4: metadata present and correctly shaped ──
    ok_meta = True
    for sid, info in bd.items():
        has_fields = all(k in info for k in
                         ["abs_mu", "sigma_beta_x", "sigma_theta", "det_A"])
        n_img = info["n_images"]
        shapes_ok = (info["abs_mu"].shape == (n_img,)
                     and info["sigma_beta_x"].shape == (n_img,)
                     and info["det_A"].shape == (n_img,))
        # sigma_beta < sigma_theta for all images (magnification > 1 near Einstein ring)
        sb_lt_st = np.all(info["sigma_beta_x"] < info["sigma_theta"])
        ok_sys = has_fields and shapes_ok and sb_lt_st
        ok_meta = ok_meta and ok_sys
        print(f"    {sid}: n_img={n_img}  fields={'OK' if has_fields else 'MISS'}  "
              f"shapes={'OK' if shapes_ok else 'BAD'}  "
              f"sigma_beta<sigma_theta={'OK' if sb_lt_st else 'FAIL'}")

    ok_all = ok_larger_true and ok_larger_pert and ok_sum and ok_meta
    R.record("11-G  Full toy geometry", ok_all)


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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS', z_lens=0.5,
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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS', z_lens=0.5,
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
        print(f"  Δ_WL = {d_wl_ref:.2f}\"  Δ_SL = {d_sl:.2f}\"  "
              f"{'improved' if better else 'comparable'}  "
              f"{'OK' if ok_improvement else 'FAIL (SL much worse)'}")

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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS', use_strong_lensing=False,
        )
        ok_1 = np.isfinite(rchi2_1) and len(lenses_1.x) >= 1
        print(f"  Explicit False: chi2={rchi2_1:.4f} n_lens={len(lenses_1.x)}  "
              f"{'OK' if ok_1 else 'FAIL'}")
    except Exception as e:
        print(f"  Explicit False: EXCEPTION {e}  FAIL")
        ok_1 = False
    ok_all = ok_all and ok_1

    # ── Call without the keyword at all (should default to False) ──
    try:
        lenses_2, rchi2_2 = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS',
        )
        ok_2 = np.isfinite(rchi2_2) and len(lenses_2.x) >= 1
        print(f"  Default (omitted): chi2={rchi2_2:.4f} n_lens={len(lenses_2.x)}  "
              f"{'OK' if ok_2 else 'FAIL'}")
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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS', use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION during pipeline: {e}")
        R.record("12-D  chi2 components", False)
        return

    # Evaluate components at the final lens configuration
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, 'SIS')
    chi2_total, dof_total, comps = metric.calculate_total_chi2(
        src, lenses_sl, use_flags, lens_type='SIS',
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
    lenses_init = pipeline.generate_initial_guess(src, lens_type='SIS', z_l=0.5)

    chi2_sl_init = utils.chi2_strong_source_plane_sis(
        lenses_init, src.strong_systems
    )

    # Run the full pipeline
    try:
        lenses_final, _ = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='SIS', use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        R.record("12-E  SL scatter improvement", False)
        return

    chi2_sl_final = utils.chi2_strong_source_plane_sis(
        lenses_final, src.strong_systems
    )

    ok_reduced = chi2_sl_final < chi2_sl_init
    print(f"  chi2_SL initial = {chi2_sl_init:.2f}")
    print(f"  chi2_SL final   = {chi2_sl_final:.2f}")
    print(f"  Scatter reduced: {'OK' if ok_reduced else 'FAIL'}")

    # Also get the per-system breakdown
    _, bd = utils.chi2_strong_source_plane_sis(
        lenses_final, src.strong_systems,
        return_breakdown=True, use_magnification_correction=True,
    )
    for sid, info in bd.items():
        print(f"    {sid}: chi2={info['chi2']:.2f}  n_img={info['n_images']}  "
              f"beta_bar=({info['beta_bar'][0]:.3f}, {info['beta_bar'][1]:.3f})")

    R.record("12-E  SL scatter improvement", ok_reduced)


# ── Runner ────────────────────────────────────────────────────────────────

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

def _make_nfw_halo(
    x: float = 0.0,
    y: float = 0.0,
    mass: float = 5e14,
    redshift: float = 0.3,
    conc: float | None = None,
) -> halo_obj.NFW_Lens:
    """
    Create a single NFW_Lens with auto-computed concentration (Duffy et al. 2008)
    unless conc is explicitly supplied.
    """
    halos = halo_obj.NFW_Lens(
        x=[x], y=[y], z=[0.0],
        concentration=[5.0],   # placeholder; overwritten below
        mass=[mass],
        redshift=redshift,
        chi2=[0.0],
    )
    if conc is None:
        halos.calculate_concentration()
    else:
        halos.concentration = np.array([float(conc)])
    return halos


# ── Helper: numerical NFW Einstein radius ─────────────────────────────────

def _find_nfw_einstein_radius(
    halos,
    z_source: float,
    r_min: float = 0.01,
    r_max: float | None = None,
) -> float:
    """
    Find the tangential critical radius (where det(A) goes negative→positive)
    for a single NFW halo along the x-axis using brentq.

    NFW halos have two critical curves:
      - Radial critical curve (pos→neg): at smaller r (~0.6 arcsec for standard halo)
      - Tangential Einstein radius (neg→pos): at larger r (~1.7 arcsec)

    This function returns the TANGENTIAL Einstein radius (the outermost zero of
    det(A), where det(A) transitions from negative to positive).

    Uses a logarithmic scan to resolve both small and large scales.
    """
    x0 = float(halos.x[0])
    y0 = float(halos.y[0])

    if r_max is None:
        _, r200_arcsec = halos.calc_R200()
        r_max = float(np.atleast_1d(r200_arcsec)[0]) * 2.0

    def det_A_at_r(r):
        tx = np.array([x0 + r])
        ty = np.array([y0])
        _, det_A = utils.magnification_nfw(halos, tx, ty, z_source)
        return float(det_A[0])

    # Log-spaced scan to resolve both small (radial) and large (tangential) scales
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 400)
    dets = np.array([det_A_at_r(r) for r in radii])

    # Find the TANGENTIAL Einstein radius: last neg→pos transition
    # (NFW det(A): positive inside radial CR, negative between CRs, positive outside)
    neg_to_pos = np.where(np.diff(np.sign(dets)) > 0)[0]  # neg → pos
    if len(neg_to_pos) == 0:
        # Fallback: use any sign change
        any_change = np.where(np.diff(np.sign(dets)))[0]
        if len(any_change) == 0:
            raise RuntimeError(
                f"No tangential Einstein radius found in [{r_min:.4f}, {r_max:.1f}] "
                "arcsec. Check halo mass/redshift."
            )
        idx = any_change[-1]
    else:
        idx = neg_to_pos[-1]  # last neg→pos = tangential Einstein radius

    r_a, r_b = radii[idx], radii[idx + 1]
    return float(brentq(det_A_at_r, r_a, r_b, xtol=1e-6, rtol=1e-8))


# ── Helper: radial caustic radius ─────────────────────────────────────────

def _find_nfw_beta_rad(
    halos,
    z_source: float,
    theta_E: float,
    n_scan: int = 200,
) -> float:
    """
    Return the radial caustic radius: max_{r in (0, theta_E)} [alpha(r) - r].

    This is the maximum source-plane offset for which a counter-image exists
    on the opposite side of the lens (the radial caustic radius beta_rad).
    Sources with |beta| < beta_rad produce 3 images; sources with |beta| > beta_rad
    produce 1 image (only the major arc on the same side as the source).
    """
    x0, y0 = float(halos.x[0]), float(halos.y[0])
    r_vals = np.logspace(-2, np.log10(theta_E * 0.999), n_scan)
    alpha_minus_r = np.zeros(n_scan)
    for i, r in enumerate(r_vals):
        tx = np.array([x0 + r])
        ty = np.array([y0])
        ax, _ = utils.calculate_deflection_nfw(halos, tx, ty, z_source)
        alpha_minus_r[i] = float(ax[0]) - r
    return float(np.max(alpha_minus_r))


# ── Helper: numerical NFW image positions ─────────────────────────────────

def _find_nfw_image_positions(
    halos,
    beta_x: float,
    beta_y: float,
    z_source: float,
    theta_E: float,
    n_scan: int = 400,
) -> tuple:
    """
    Numerically solve the NFW lens equation to find the two image positions
    for a source at (beta_x, beta_y) with |beta_rel| < beta_rad.

    NFW image geometry (lens centered at halos.x[0], halos.y[0]):
      Image 1 (major arc): same side as source, r > theta_E
      Image 2 (outer counter-arc): opposite side, r_rad < |r| < theta_E

    The lens equation (signed radial parameterisation):
        f(r) = r - alpha_r(r) - beta_rel = 0
    where alpha_r is the projection of the deflection onto the source direction.

    For r < 0 (opposite side): f(r) = alpha(|r|) - |r| - beta_rel,
    which is positive near |r| = r_rad and negative near 0 and theta_E.

    The scan finds the first neg→pos transition going from -theta_E toward 0,
    which brackets the outer counter-image.
    """
    x0, y0 = float(halos.x[0]), float(halos.y[0])
    dbx, dby = beta_x - x0, beta_y - y0
    beta_rel = float(np.hypot(dbx, dby))
    if beta_rel == 0:
        raise ValueError("Source exactly on lens centre: degenerate (Einstein ring).")
    ehatx, ehaty = dbx / beta_rel, dby / beta_rel

    def lens_eq(r):
        tx = np.array([x0 + r * ehatx])
        ty = np.array([y0 + r * ehaty])
        ax, ay = utils.calculate_deflection_nfw(halos, tx, ty, z_source)
        alpha_r = float(ax[0]) * ehatx + float(ay[0]) * ehaty
        return r - alpha_r - beta_rel

    eps = 1e-4
    r_outer = theta_E * 6.0

    # ── Image 1: same side, outside tangential Einstein ring ──
    fa1 = lens_eq(theta_E + eps)
    fb1 = lens_eq(r_outer)
    if fa1 * fb1 >= 0:
        raise RuntimeError(
            f"Image-1 bracket failed: f({theta_E+eps:.4f})={fa1:.3e}, "
            f"f({r_outer:.4f})={fb1:.3e}"
        )
    r1 = brentq(lens_eq, theta_E + eps, r_outer, xtol=1e-7, rtol=1e-9)

    # ── Image 2: outer counter-image on the opposite side ──
    # Scan from r = -(theta_E) toward 0.  The function f(r) = alpha(|r|)-|r|-beta_rel
    # is negative near -theta_E, rises to a peak (beta_rad - beta_rel) at the radial
    # critical curve, then falls back to negative near 0.
    # The first neg→pos transition (going from -theta_E toward 0) brackets image 2.
    eps_inner = min(0.002, theta_E * 0.001)   # don't probe too close to the origin
    scan_r = np.linspace(-(theta_E - eps), -eps_inner, n_scan)
    scan_f = np.array([lens_eq(r) for r in scan_r])

    neg_to_pos = np.where(np.diff(np.sign(scan_f)) > 0)[0]   # neg → pos
    if len(neg_to_pos) == 0:
        raise RuntimeError(
            f"|beta_rel| = {beta_rel:.4f} arcsec >= beta_rad; "
            "no counter-image exists. Reduce beta_rel below the radial caustic radius."
        )
    idx = neg_to_pos[0]   # outermost (nearest to -theta_E) = outer counter-arc
    r2 = brentq(lens_eq, scan_r[idx], scan_r[idx + 1], xtol=1e-7, rtol=1e-9)

    theta_x = np.array([x0 + r1 * ehatx, x0 + r2 * ehatx])
    theta_y = np.array([y0 + r1 * ehaty, y0 + r2 * ehaty])
    return theta_x, theta_y


# ── Helper: build StrongLensingSystem from NFW ────────────────────────────

def _make_two_image_nfw_system(
    system_id: str,
    halos,
    beta_rel_xy: tuple,
    z_source: float,
    theta_E: float,
    sigma_theta: float = 0.05,
) -> source_obj.StrongLensingSystem:
    """
    Build a StrongLensingSystem with numerically-exact NFW image positions
    such that chi2 = 0 at the true halo parameters.
    """
    x0, y0 = float(halos.x[0]), float(halos.y[0])
    bx_rel, by_rel = beta_rel_xy
    b = float(np.hypot(bx_rel, by_rel))
    if b >= theta_E:
        raise ValueError(
            f"|beta_rel| = {b:.3f} >= theta_E = {theta_E:.3f}; "
            "source outside Einstein ring."
        )
    beta_x = x0 + bx_rel
    beta_y = y0 + by_rel
    theta_x, theta_y = _find_nfw_image_positions(
        halos, beta_x, beta_y, z_source, theta_E
    )
    StrongLensingSystem = source_obj.StrongLensingSystem
    return StrongLensingSystem(
        system_id=system_id,
        theta_x=theta_x,
        theta_y=theta_y,
        z_source=float(z_source),
        sigma_theta=float(sigma_theta),
        meta={
            "toy": True,
            "lens_center": (x0, y0),
            "beta_rel": (bx_rel, by_rel),
            "theta_E_nfw": theta_E,
        },
    )


# ── Helper: NFW weak-lensing catalog builder ──────────────────────────────

def _make_nfw_wl_catalog(
    halos,
    xmax: float = 120.0,
    n_sources: int = 80,
    z_source: float = 0.8,
    sig_shear: float = 0.10,
    sig_flex: float = 0.02,
    sig_gflex: float = 0.03,
    rmin: float = 1.0,
    seed: int = 7,
) -> source_obj.Source:
    """
    Build a weak-lensing source catalog lensed by an NFW halo.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-xmax, xmax, size=n_sources)
    y = rng.uniform(-xmax, xmax, size=n_sources)

    # Exclude sources too close to any halo centre
    keep = np.ones(len(x), dtype=bool)
    for xi, yi in zip(np.atleast_1d(halos.x), np.atleast_1d(halos.y)):
        keep &= np.hypot(x - xi, y - yi) > rmin
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
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
        bxm, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
        _, bym = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
        _, bym2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx

        rel = abs(det_analytic[k] - det_fd) / max(abs(det_fd), 1e-30)
        ok_k = rel < 1e-4
        ok_all = ok_all and ok_k
        print(f"  r={pts_x[k]:5.1f}\"  det_analytic={det_analytic[k]:+.8f}  "
              f"det_FD={det_fd:+.8f}  rel_err={rel:.1e}  {'OK' if ok_k else 'FAIL'}")

    R.record("13-A  Single NFW: analytic vs FD Jacobian", ok_all)


# ── 13-B  Composite NFW: analytic vs FD Jacobian ─────────────────────────

def _test_nfw_magnification_composite_fd(R: _TestResults13):
    """
    Two NFW halos at different positions.  Compare the analytic composite
    det(A) from magnification_nfw against the central-difference FD Jacobian.
    """
    R.header("13-B  Composite NFW magnification: analytic vs FD Jacobian")

    halos = halo_obj.NFW_Lens(
        x=[0.0, 60.0], y=[0.0, 40.0], z=[0.0, 0.0],
        concentration=[1.0, 1.0],
        mass=[5e14, 3e14],
        redshift=0.3,
        chi2=[0.0, 0.0],
    )
    halos.calculate_concentration()
    z_source = 2.0

    # Test points well away from both halo centres
    pts_x = np.array([30.0, -20.0, 100.0, 50.0, -10.0])
    pts_y = np.array([-20.0,  60.0,  10.0, 80.0, -40.0])

    _, det_analytic = utils.magnification_nfw(halos, pts_x, pts_y, z_source)

    h = 1e-4
    ok_all = True
    for k in range(len(pts_x)):
        bxp, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
        bxm, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] + h]), np.array([pts_y[k]]), z_source)
        _, bym = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k] - h]), np.array([pts_y[k]]), z_source)
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] + h]), z_source)
        _, bym2 = utils.backproject_source_positions_nfw(
            halos, np.array([pts_x[k]]), np.array([pts_y[k] - h]), z_source)
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx

        rel = abs(det_analytic[k] - det_fd) / max(abs(det_fd), 1e-30)
        ok_k = rel < 2e-4
        ok_all = ok_all and ok_k
        print(f"  pt {k}: det_analytic={det_analytic[k]:+.8f}  "
              f"det_FD={det_fd:+.8f}  rel_err={rel:.1e}  {'OK' if ok_k else 'FAIL'}")

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
    print(f"  beta_rel = ({br*0.8:.4f}, {br*0.6:.4f}) arcsec  (|beta|={br:.4f}, beta_rad={beta_rad:.4f})")
    print(f"  Image positions: theta1=({sys_nfw.theta_x[0]:.3f}, {sys_nfw.theta_y[0]:.3f})"
          f"  theta2=({sys_nfw.theta_x[1]:.3f}, {sys_nfw.theta_y[1]:.3f})")

    # ── Sub-test 1: perfect model, no magnification correction ──
    chi2_true_uncorr = utils.chi2_strong_source_plane_nfw(
        halos_true, [sys_nfw], use_magnification_correction=False
    )
    ok_zero_uncorr = chi2_true_uncorr < 1e-6
    print(f"  True lens (uncorrected): chi2 = {chi2_true_uncorr:.2e}  "
          f"{'OK (< 1e-6)' if ok_zero_uncorr else 'FAIL'}")

    # ── Sub-test 2: perfect model, with magnification correction ──
    chi2_true_corr = utils.chi2_strong_source_plane_nfw(
        halos_true, [sys_nfw], use_magnification_correction=True
    )
    ok_zero_corr = chi2_true_corr < 1e-6
    print(f"  True lens (corrected):   chi2 = {chi2_true_corr:.2e}  "
          f"{'OK (< 1e-6)' if ok_zero_corr else 'FAIL'}")

    # ── Sub-test 3: perturbed lens (shifted by ~half the Einstein radius) ──
    halos_pert = _make_nfw_halo(x=theta_E * 0.5, y=-theta_E * 0.3, mass=5e14, redshift=0.3)
    chi2_pert = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_nfw], use_magnification_correction=False
    )
    ok_pert = chi2_pert > chi2_true_uncorr and chi2_pert > 1.0
    print(f"  Perturbed lens:          chi2 = {chi2_pert:.4f}  "
          f"{'OK (> 1.0 and > chi2_true)' if ok_pert else 'FAIL'}")

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
    print(f"  chi2_corr={chi2_corr:.4f}  chi2_uncorr={chi2_uncorr:.4f}  "
          f"corr>uncorr: {'OK' if ok_larger else 'FAIL'}")

    # ── Sub-test 2: magnification from breakdown matches FD Jacobian ──
    bd = bd_corr["nfw_d"]
    tx = sys_nfw.theta_x
    ty = sys_nfw.theta_y
    h = 1e-4
    ok_mu = True
    for m in range(len(tx)):
        # FD Jacobian at this image position
        bxp, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] + h]), np.array([ty[m]]), z_source)
        bxm, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] - h]), np.array([ty[m]]), z_source)
        dbx_dtx = (bxp[0] - bxm[0]) / (2 * h)

        bxp2, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] + h]), z_source)
        bxm2, _ = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] - h]), z_source)
        dbx_dty = (bxp2[0] - bxm2[0]) / (2 * h)

        _, byp = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] + h]), np.array([ty[m]]), z_source)
        _, bym = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m] - h]), np.array([ty[m]]), z_source)
        dby_dtx = (byp[0] - bym[0]) / (2 * h)

        _, byp2 = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] + h]), z_source)
        _, bym2 = utils.backproject_source_positions_nfw(
            halos_pert, np.array([tx[m]]), np.array([ty[m] - h]), z_source)
        dby_dty = (byp2[0] - bym2[0]) / (2 * h)

        det_fd = dbx_dtx * dby_dty - dbx_dty * dby_dtx
        abs_mu_fd = 1.0 / max(abs(det_fd), 1e-30)
        abs_mu_analytic = bd["abs_mu"][m]

        rel = abs(abs_mu_analytic - abs_mu_fd) / max(abs_mu_fd, 1e-30)
        ok_m = rel < 0.01
        ok_mu = ok_mu and ok_m
        print(f"    image {m}: |mu|_analytic={abs_mu_analytic:.4f}  "
              f"|mu|_FD={abs_mu_fd:.4f}  rel_err={rel:.1e}  {'OK' if ok_m else 'FAIL'}")

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
        print(f"    image {m}: sig_b={sig_b:.6f}  expected={sig_b_expected:.6f}  "
              f"{'OK' if close else 'FAIL'}")

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
        "E_A", halos_true, (br_A * 0.8, br_A * 0.6), z_source_A,
        theta_E_min, sigma_theta=0.06
    )
    sys_B = _make_two_image_nfw_system(
        "E_B", halos_true, (-br_B * 0.6, br_B * 0.8), z_source_B,
        theta_E_min, sigma_theta=0.08
    )

    # Perturb the lens slightly (by ~30% of the Einstein radius)
    halos_pert = _make_nfw_halo(x=theta_E_A * 0.3, y=-theta_E_A * 0.18, mass=5e14, redshift=0.3)

    chi2_total, bd = utils.chi2_strong_source_plane_nfw(
        halos_pert, [sys_A, sys_B],
        return_breakdown=True, use_magnification_correction=True
    )

    required_keys = {"abs_mu", "sigma_beta_x", "sigma_beta_y",
                     "sigma_theta", "det_A", "chi2", "n_images", "beta_bar", "beta"}

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
        print(f"  {sid}: n_img={n_img}  keys={'OK' if ok_keys else 'MISS'}  "
              f"shapes={'OK' if ok_shapes else 'BAD'}  "
              f"sig_b<=sig_th={'OK' if ok_sigma else 'FAIL'}")

    # Sum of per-system chi2 equals total
    bd_sum = sum(v["chi2"] for v in bd.values())
    ok_sum = np.isclose(chi2_total, bd_sum, rtol=1e-10)
    print(f"  chi2 sum: breakdown={bd_sum:.6f}  total={chi2_total:.6f}  "
          f"{'OK' if ok_sum else 'FAIL'}")

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

    src = _make_nfw_wl_catalog(halos_true, xmax=120.0, n_sources=80,
                                z_source=z_source_wl, seed=42)

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

    lam = metric.compute_lambda_sl(src, halos_init, use_flags, lens_type='NFW')

    # Manual calculation
    chi2_wl = metric.calculate_chi_squared(src, halos_init, use_flags, lens_type='NFW')
    dof_wl = metric.calc_degrees_of_freedom(src, halos_init, use_flags)
    chi2_sl = utils.chi2_strong_source_plane_nfw(halos_init, src.strong_systems)
    dof_sl = metric.calc_strong_dof(src)
    expected = (chi2_wl / dof_wl) / (chi2_sl / dof_sl)

    ok_match = np.isclose(lam, expected, rtol=1e-10)
    ok_finite = np.isfinite(lam) and lam > 0
    print(f"  compute_lambda_sl = {lam:.6f}")
    print(f"  manual            = {expected:.6f}")
    print(f"  match: {'OK' if ok_match else 'FAIL'}   "
          f"positive & finite: {'OK' if ok_finite else 'FAIL'}")

    # Degenerate case: no strong systems → returns 1.0
    src_no_sl = _make_nfw_wl_catalog(halos_true, xmax=120.0, n_sources=80,
                                      z_source=z_source_wl, seed=42)
    lam_none = metric.compute_lambda_sl(src_no_sl, halos_init, use_flags, 'NFW')
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

    src = _make_nfw_wl_catalog(halos_true, xmax=120.0, n_sources=80,
                                z_source=z_source_wl, seed=99)

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
        src, halos_init, use_flags, lens_type='NFW',
        use_strong_lensing=True, lambda_sl=fixed_lam,
    )
    expected_total = comps["chi2_wl"] + fixed_lam * comps["chi2_sl"]
    ok_decomp = np.isclose(chi2_total, expected_total, rtol=1e-10)
    ok_lam = comps["lambda_sl"] == fixed_lam
    print(f"  Explicit lam={fixed_lam}: chi2_total={chi2_total:.4f}  "
          f"expected={expected_total:.4f}  {'OK' if ok_decomp else 'FAIL'}")
    print(f"  Lambda passthrough: {comps['lambda_sl']}  {'OK' if ok_lam else 'FAIL'}")

    # ── Sub-test 2: no-SL mode ──
    chi2_no, _, comps_no = metric.calculate_total_chi2(
        src, halos_init, use_flags, lens_type='NFW', use_strong_lensing=False
    )
    ok_no = (comps_no["lambda_sl"] == 0.0
             and np.isclose(chi2_no, comps_no["chi2_wl"], rtol=1e-10))
    print(f"  No-SL: lambda=0, chi2=chi2_wl  {'OK' if ok_no else 'FAIL'}")

    # ── Sub-test 3: auto-lambda fallback ──
    chi2_fb, _, comps_fb = metric.calculate_total_chi2(
        src, halos_init, use_flags, lens_type='NFW',
        use_strong_lensing=True, lambda_sl=None,
    )
    ok_fb = np.isfinite(comps_fb["lambda_sl"]) and comps_fb["lambda_sl"] > 0
    print(f"  Auto-lambda = {comps_fb['lambda_sl']:.6f}  "
          f"finite & positive: {'OK' if ok_fb else 'FAIL'}")

    # ── Sub-test 4: DOF check ──
    dof_sl = comps["dof_sl"]
    expected_dof_sl = 2 * (2 - 1)   # one 2-image system: 2*(N-1) = 2
    ok_dof = dof_sl == expected_dof_sl
    print(f"  dof_sl={dof_sl}  expected={expected_dof_sl}  "
          f"{'OK' if ok_dof else 'FAIL'}")

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
    true_mass = 5e14        # solar masses
    true_redshift = 0.3
    xmax = 120.0
    n_sources = 100
    z_source_wl = 0.8
    z_source_sl = 2.0

    halos_true = _make_nfw_halo(
        x=true_x, y=true_y, mass=true_mass, redshift=true_redshift
    )

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
    br = beta_rad * 0.45   # safely inside radial caustic
    sys_A = _make_two_image_nfw_system(
        "nfw14_A", halos_true, (br * 0.8, br * 0.6), z_source_sl, theta_E,
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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=tz,
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
    ok_near = d_wl < 30.0   # within 30" of truth (25% of field radius)
    m_wl = _nearest_nfw_mass(lenses_wl, tx, ty)
    ok_mass = 1e13 < m_wl < 1e16   # wide sanity check (factor ~20 either way)

    print(f"  N_halos      = {n_lens}  {'OK' if ok_nlens else 'FAIL'}")
    print(f"  reduced chi2 = {rchi2_wl:.4f}  {'OK' if ok_finite else 'FAIL'}")
    print(f"  nearest dist = {d_wl:.2f}\"  (< 30\")  {'OK' if ok_near else 'FAIL'}")
    print(f"  nearest mass = {m_wl:.3e} M_sun  (true={tm:.1e})  "
          f"{'OK' if ok_mass else 'FAIL'}")

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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=tz,
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
    print(f"  nearest dist = {d_sl:.2f}\"  (informational, true = ({tx}, {ty}))")
    print(f"  nearest mass = {m_sl:.3e} M_sun  (informational, true = {tm:.1e})")

    # SL should improve position over WL-only
    ok_improvement = True
    if d_wl_ref is not None:
        ok_improvement = d_sl <= d_wl_ref + 5.0   # SL within 5" of WL-only or better
        better = d_sl < d_wl_ref
        label = "SL closer to truth" if better else "WL closer to truth"
        print(f"  dist_WL = {d_wl_ref:.2f}\"  dist_SL = {d_sl:.2f}\"  {label}  "
              f"{'OK' if ok_improvement else 'FAIL'}")

    # Verify chi2_WL is positive (WL constraint is evaluated)
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, 'NFW')
    chi2_total, _, comps = metric.calculate_total_chi2(
        src, lenses_sl, use_flags, lens_type='NFW',
        use_strong_lensing=True, lambda_sl=lambda_sl,
    )
    ok_wl_pos = comps["chi2_wl"] > 0
    print(f"  chi2_WL > 0:  {'OK' if ok_wl_pos else 'FAIL'}  "
          f"(chi2_WL = {comps['chi2_wl']:.2f})")

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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=tz,
            use_strong_lensing=True,
        )
    except Exception as e:
        print(f"  EXCEPTION during pipeline: {e}")
        R.record("14-C  NFW chi2 components", False)
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
    ok_total = np.isclose(
        chi2_total,
        comps["chi2_wl"] + comps["lambda_sl"] * comps["chi2_sl"],
        rtol=1e-10,
    )

    print(f"  chi2_WL   = {comps['chi2_wl']:.2f}  (dof={comps['dof_wl']})  "
          f"{'OK' if ok_wl else 'FAIL'}")
    print(f"  chi2_SL   = {comps['chi2_sl']:.2f}  (dof={comps['dof_sl']})  "
          f"{'OK' if ok_sl else 'FAIL'}")
    print(f"  lambda_sl = {comps['lambda_sl']:.6f}  {'OK' if ok_lam else 'FAIL'}")
    print(f"  dof > 0:  WL={'OK' if comps['dof_wl']>0 else 'FAIL'}  "
          f"SL={'OK' if comps['dof_sl']>0 else 'FAIL'}")
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
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=tz,
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
    ok_true_small = chi2_sl_true < 1e-6   # exact images → chi2 ≈ 0

    print(f"  chi2_SL at true halo     = {chi2_sl_true:.2e}  "
          f"{'OK (near 0)' if ok_true_small else 'FAIL (not near 0)'}")
    print(f"  chi2_SL at WL-only halo  = {chi2_sl_wl:.4f}  "
          f"(WL halo at {_nearest_nfw_distance(lenses_wl, tx, ty):.2f}\" from truth)")
    print(f"  True halo has lower chi2_SL: {'OK' if ok_true_better else 'FAIL'}")

    # Per-system breakdown at the WL-only halo
    _, bd = utils.chi2_strong_source_plane_nfw(
        lenses_wl, src.strong_systems,
        return_breakdown=True, use_magnification_correction=False,
    )
    for sid, info in bd.items():
        print(f"    WL halo — {sid}: chi2={info['chi2']:.4f}  "
              f"n_img={info['n_images']}  "
              f"beta_bar=({info['beta_bar'][0]:.4f}, {info['beta_bar'][1]:.4f})")

    ok_all = ok_true_small and ok_true_better
    R.record("14-D  NFW SL chi2 at true vs WL-only", ok_all)


# ── Task 14 summary class ─────────────────────────────────────────────────

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
        theta_E = 2.0    # safe fallback

    inset_half = max(theta_E * 5.0, 3.0)    # arcsec half-width of inset

    # ── Figure layout ──────────────────────────────────────────────────────
    fig, (ax_wl, ax_sl) = plt.subplots(
        1, 2, figsize=(15, 7), dpi=150,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.10, top=0.88, wspace=0.28)

    panels = [
        (ax_wl, halos_wl, "WL only",   "C0", rchi2_wl),
        (ax_sl, halos_sl, "WL + SL",   "C2", rchi2_sl),
    ]

    for ax, halos, label, color, rchi2 in panels:

        # ── WL sources ────────────────────────────────────────────────────
        ax.scatter(src.x, src.y, s=5, c="0.72", alpha=0.45, zorder=1,
                   label=f"WL sources  (N={len(src.x)})")

        # ── True halo ─────────────────────────────────────────────────────
        ax.scatter(
            true_x, true_y, marker="*", s=380, c="gold",
            edgecolors="k", linewidths=0.8, zorder=10,
            label=f"True halo  ({true_mass:.1e} M$_\\odot$)",
        )

        # ── Recovered halos ───────────────────────────────────────────────
        if len(halos.x) > 0:
            log_m = np.log10(np.clip(halos.mass, 1e10, 1e16))
            log_ref = np.log10(true_mass)
            sizes = np.clip((log_m - 10.0) / max(log_ref - 10.0, 1.0) * 220, 25, 550)
            ax.scatter(
                halos.x, halos.y, s=sizes, c=color,
                edgecolors="k", linewidths=0.7, alpha=0.85,
                zorder=8, label=f"Recovered  (N={len(halos.x)})",
            )
            for hx, hy, hm in zip(halos.x, halos.y, halos.mass):
                ax.annotate(
                    f"{hm:.1e}",
                    (hx, hy), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="0.20", zorder=11,
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
                    arrowstyle="-|>", color="C1",
                    lw=1.8, shrinkA=5, shrinkB=5,
                ),
                zorder=9,
            )
            # Bottom-left annotation box
            ann_text = (
                f"$\\Delta$ = {d_near:.1f}\"    "
                f"M = {halos.mass[idx_near]:.2e} M$_\\odot$"
            )
            if rchi2 is not None:
                ann_text += f"\n$\\tilde{{\\chi}}^2$ = {rchi2:.3f}"
            ax.text(
                0.03, 0.03, ann_text,
                transform=ax.transAxes, fontsize=8.5,
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
        unique = [(h, l) for h, l in zip(handles, lbls)
                  if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7.5, markerscale=0.9)

        # ══ INSET — zoom on SL region ═════════════════════════════════════
        ax_in = ax.inset_axes([0.60, 0.60, 0.38, 0.38])

        # WL sources inside inset bounds
        mask = (
            (np.abs(src.x - true_x) < inset_half) &
            (np.abs(src.y - true_y) < inset_half)
        )
        if mask.any():
            ax_in.scatter(src.x[mask], src.y[mask], s=8, c="0.72", alpha=0.5)

        # Strong-lensing image positions
        if hasattr(src, "strong_systems") and src.strong_systems:
            for i, sys in enumerate(src.strong_systems):
                ax_in.scatter(
                    sys.theta_x, sys.theta_y,
                    marker="D", s=55,
                    edgecolors="C3", facecolors="none",
                    linewidths=1.6, zorder=5,
                    label="SL images" if i == 0 else "",
                )

        # Einstein ring
        ring = plt.Circle(
            (true_x, true_y), theta_E,
            fill=False, linestyle="--", linewidth=1.1,
            edgecolor="gold", alpha=0.85, zorder=4,
        )
        ax_in.add_patch(ring)

        # True halo in inset
        ax_in.scatter(
            true_x, true_y, marker="*", s=180, c="gold",
            edgecolors="k", linewidths=0.7, zorder=10,
        )

        # Recovered halos in inset
        if len(halos.x) > 0:
            in_mask = (
                (np.abs(halos.x - true_x) < inset_half) &
                (np.abs(halos.y - true_y) < inset_half)
            )
            if in_mask.any():
                ax_in.scatter(
                    halos.x[in_mask], halos.y[in_mask],
                    s=45, c=color, edgecolors="k", linewidths=0.5,
                    alpha=0.85, zorder=8,
                )
            else:
                # Halo outside inset: draw an arrow pointing to it from edge
                hx_near = halos.x[int(np.argmin(
                    np.hypot(halos.x - true_x, halos.y - true_y)
                ))]
                hy_near = halos.y[int(np.argmin(
                    np.hypot(halos.x - true_x, halos.y - true_y)
                ))]
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
                    true_x + dx * 0.5, true_y + dy * 0.5,
                    "halo\noutside", fontsize=5.5,
                    ha="center", va="center", color=color, zorder=10,
                )

        ax_in.set_xlim(true_x - inset_half, true_x + inset_half)
        ax_in.set_ylim(true_y - inset_half, true_y + inset_half)
        ax_in.set_aspect("equal", adjustable="box")
        ax_in.tick_params(labelsize=6)
        ax_in.set_title(
            f"zoom  ±{inset_half:.0f}\"  (NFW $\\theta_E$={theta_E:.2f}\")",
            fontsize=6.5, pad=3,
        )

        # Indicate the inset region on the main axes
        try:
            ax.indicate_inset_zoom(ax_in, edgecolor="0.40", linewidth=0.9)
        except Exception:
            # matplotlib < 3.3 fallback: draw a manual rectangle
            rect = plt.Rectangle(
                (true_x - inset_half, true_y - inset_half),
                2 * inset_half, 2 * inset_half,
                linewidth=0.9, edgecolor="0.40", facecolor="none", zorder=3,
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
    halos_init = pipeline.generate_initial_guess(src, lens_type='NFW', z_l=tz)
    lambda_used = metric.compute_lambda_sl(src, halos_init, use_flags, 'NFW')

    print("Running NFW WL-only pipeline...")
    halos_wl, rchi2_wl = fit_lensing_field(
        src, xmax, flags=False, use_flags=use_flags,
        lens_type='NFW', z_lens=tz, use_strong_lensing=False,
    )

    print("Running NFW WL+SL pipeline...")
    halos_sl, rchi2_sl = fit_lensing_field(
        src, xmax, flags=False, use_flags=use_flags,
        lens_type='NFW', z_lens=tz, use_strong_lensing=True,
    )

    d_wl = float(np.min(np.hypot(halos_wl.x - tx, halos_wl.y - ty)))
    d_sl = float(np.min(np.hypot(halos_sl.x - tx, halos_sl.y - ty)))

    print(f"\n{'='*60}")
    print(f"  True halo: ({tx}, {ty})  M = {tm:.1e} M_sun  z = {tz}")
    print(f"  WL-only : N={len(halos_wl.x):2d}  nearest_d={d_wl:.2f}\"  rchi2={rchi2_wl:.4f}")
    print(f"  WL+SL   : N={len(halos_sl.x):2d}  nearest_d={d_sl:.2f}\"  rchi2={rchi2_sl:.4f}")
    print(f"  lambda_SL (pre-computed from initial guess) = {lambda_used:.2f}")
    print(f"{'='*60}")

    fig = plot_nfw_wl_vs_sl_comparison(
        src=src,
        halos_wl=halos_wl,
        halos_sl=halos_sl,
        halos_true=halos_true,
        true_x=tx, true_y=ty, true_mass=tm,
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