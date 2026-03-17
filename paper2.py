from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
    print(f"  λ_SL = {lambda_sl:.6f}")
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
              f"expected={det_expected[i]:+.6f}  |μ|={abs_mu[i]:.4f}  {status}")

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
        print(f"  |μ|={mu:6.1f}  floor={floor}  σ_β={result[0]:.6f}  "
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
        print(f"    image {m}: r={r:.3f}  |μ|={mu_got:.4f}  "
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
    print(f"  total = WL + λ·SL: {'OK' if ok_total else 'FAIL'}")

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


def run_all_tests() -> bool:
    """Run Task 11 + Task 12 tests.  Returns True if everything passes."""
    ok_11 = run_magnification_tests()
    ok_12 = run_integration_tests()
    if ok_11 and ok_12:
        print("\n  ✓ ALL TASK 11 + TASK 12 TESTS PASSED\n")
    else:
        if not ok_11:
            print("\n  ✗ Task 11 (unit tests) had failures")
        if not ok_12:
            print("\n  ✗ Task 12 (integration tests) had failures")
    return ok_11 and ok_12


if __name__ == "__main__":
    run_comparison_plot()