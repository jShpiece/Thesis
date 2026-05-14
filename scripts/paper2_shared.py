"""Shared data builders, runners, plotting helpers, and test infrastructure
used by task11 through task14 scripts.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

# Ensure UTF-8 output on Windows consoles (avoids UnicodeEncodeError for θ, μ, λ, etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import arch.halo_obj as halo_obj
import arch.metric as metric
import arch.pipeline as pipeline
import arch.source_obj as source_obj
import arch.utils as utils
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
    for x0, y0, _te in true_lens_xyte:
        r = np.hypot(x - x0, y - y0)
        keep &= r > rmin
    x, y = x[keep], y[keep]

    src = source_obj.Source(
        x=x,
        y=y,
        e1=np.zeros_like(x),
        e2=np.zeros_like(x),
        f1=np.zeros_like(x),
        f2=np.zeros_like(x),
        g1=np.zeros_like(x),
        g2=np.zeros_like(x),
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
        chi2=np.array(
            getattr(lenses, "chi2", np.zeros_like(np.atleast_1d(lenses.x))), dtype=float
        ).copy(),
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
    _ = pipeline.update_chi2_values(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    stages["initial_guess"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_positions(
        src, lenses, xmax, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    _ = pipeline.update_chi2_values(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    stages["optimization"] = _copy_lenses_sis(lenses)

    lenses = pipeline.filter_lens_positions(src, lenses, xmax, lens_type="SIS")
    _ = pipeline.update_chi2_values(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    stages["filter"] = _copy_lenses_sis(lenses)

    lenses, _best = pipeline.forward_lens_selection(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    _ = pipeline.update_chi2_values(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    stages["forward_selection"] = _copy_lenses_sis(lenses)

    merger_threshold = (len(src.x) / (2 * xmax) ** 2) ** (-0.5) if len(src.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, "SIS")
    _ = pipeline.update_chi2_values(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    stages["merging"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_strength(
        src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
    _ = pipeline.update_chi2_values(
        src, lenses, [True, True, True], lens_type="SIS", use_strong_lensing=use_strong_lensing
    )
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
                for lx, ly, lte in zip(L.x, L.y, L.te):
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
        1,
        2,
        figsize=(14, 6.5),
        dpi=150,
        constrained_layout=True,
    )

    for ax, lenses, label in [
        (ax_wl, lenses_wl, "WL only"),
        (ax_sl, lenses_sl, "WL + SL"),
    ]:
        # ── Source galaxies ──
        ax.scatter(
            src.x, src.y, s=4, c="0.70", alpha=0.5, zorder=1, label=f"sources (N={len(src.x)})"
        )

        # ── Strong lensing image positions ──
        if hasattr(src, "strong_systems") and src.strong_systems:
            for sl_sys in src.strong_systems:
                ax.scatter(
                    sl_sys.theta_x,
                    sl_sys.theta_y,
                    marker="D",
                    s=60,
                    edgecolors="C3",
                    facecolors="none",
                    linewidths=1.5,
                    zorder=5,
                    label=f"SL images ({sl_sys.system_id})",
                )

        # ── Truth: star + Einstein ring ──
        ax.scatter(
            true_x,
            true_y,
            marker="*",
            s=350,
            c="gold",
            edgecolors="k",
            linewidths=0.8,
            zorder=10,
            label=f'truth ($\\theta_E$={true_te:.1f}")',
        )
        circle = plt.Circle(
            (true_x, true_y),
            true_te,
            fill=False,
            linestyle="--",
            linewidth=1.2,
            edgecolor="gold",
            alpha=0.8,
            zorder=4,
        )
        ax.add_patch(circle)

        # ── Recovered lenses ──
        if len(lenses.x) > 0:
            # Size proportional to θ_E, clipped for readability
            sizes = np.clip(np.abs(lenses.te), 0.5, 15) * 30
            ax.scatter(
                lenses.x,
                lenses.y,
                s=sizes,
                c="C0" if label == "WL only" else "C2",
                edgecolors="k",
                linewidths=0.6,
                alpha=0.85,
                zorder=8,
                label=f"recovered (N={len(lenses.x)})",
            )
            # Label each recovered lens with its θ_E
            for lx, ly, lte in zip(lenses.x, lenses.y, lenses.te):
                ax.annotate(
                    f'{lte:.2f}"',
                    (lx, ly),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=7.5,
                    color="0.25",
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
                    arrowstyle="-|>",
                    color="C1",
                    lw=1.8,
                    shrinkA=4,
                    shrinkB=4,
                ),
                zorder=9,
            )
            ax.text(
                0.03,
                0.03,
                f'$\\Delta$ = {d:.2f}"   $\\theta_E$ = {lenses.te[idx_near]:.2f}"',
                transform=ax.transAxes,
                fontsize=9,
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
        src,
        xmax,
        flags=True,
        use_flags=use_flags,
        lens_type="SIS",
        use_strong_lensing=False,
    )

    print("\nRunning WL+SL pipeline...")
    lenses_sl, rchi2_sl = fit_lensing_field(
        src,
        xmax,
        flags=True,
        use_flags=use_flags,
        lens_type="SIS",
        use_strong_lensing=True,
    )

    # Compute lambda_sl for display
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, "SIS")

    d_wl = float(np.min(np.hypot(lenses_wl.x - true_x, lenses_wl.y - true_y)))
    d_sl = float(np.min(np.hypot(lenses_sl.x - true_x, lenses_sl.y - true_y)))
    print(f"\n{'='*50}")
    print(f"  WL-only:  N_lens={len(lenses_wl.x)},  " f'nearest Δ={d_wl:.2f}",  rχ²={rchi2_wl:.4f}')
    print(f"  WL+SL:    N_lens={len(lenses_sl.x)},  " f'nearest Δ={d_sl:.2f}",  rχ²={rchi2_sl:.4f}')
    print(f"  lam_SL = {lambda_sl:.6f}")
    print(f"{'='*50}")

    fig = plot_wl_vs_sl_comparison(
        src=src,
        lenses_wl=lenses_wl,
        lenses_sl=lenses_sl,
        true_x=true_x,
        true_y=true_y,
        true_te=true_te,
        xmax=xmax,
        lambda_sl=lambda_sl,
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
        (-15.0, 0.0, 5.0),  # lens A: (x, y, te)
        (18.0, 12.0, 3.5),  # lens B
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
        title="ARCH SIS toy run",
        savepath="stages_two_lens_sl.png",
    )
    plt.show()


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
        x=[x],
        y=[y],
        z=[0.0],
        concentration=[5.0],  # placeholder; overwritten below
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
    eps_inner = min(0.002, theta_E * 0.001)  # don't probe too close to the origin
    scan_r = np.linspace(-(theta_E - eps), -eps_inner, n_scan)
    scan_f = np.array([lens_eq(r) for r in scan_r])

    neg_to_pos = np.where(np.diff(np.sign(scan_f)) > 0)[0]  # neg → pos
    if len(neg_to_pos) == 0:
        raise RuntimeError(
            f"|beta_rel| = {beta_rel:.4f} arcsec >= beta_rad; "
            "no counter-image exists. Reduce beta_rel below the radial caustic radius."
        )
    idx = neg_to_pos[0]  # outermost (nearest to -theta_E) = outer counter-arc
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
            f"|beta_rel| = {b:.3f} >= theta_E = {theta_E:.3f}; " "source outside Einstein ring."
        )
    beta_x = x0 + bx_rel
    beta_y = y0 + by_rel
    theta_x, theta_y = _find_nfw_image_positions(halos, beta_x, beta_y, z_source, theta_E)
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
        x=x,
        y=y,
        e1=np.zeros_like(x),
        e2=np.zeros_like(x),
        f1=np.zeros_like(x),
        f2=np.zeros_like(x),
        g1=np.zeros_like(x),
        g2=np.zeros_like(x),
        sigs=np.full_like(x, sig_shear),
        sigf=np.full_like(x, sig_flex),
        sigg=np.full_like(x, sig_gflex),
        redshift=np.full_like(x, z_source),
    )
    src.apply_lensing(halos, lens_type="NFW")
    src.apply_noise()
    return src


# ── Single-SIS scenario builder (shared with task12) ─────────────────────


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
