"""
Side-by-side comparison of NFW WL-only vs WL+SL reconstruction.

Usage:
    >>> from nfw_comparison_plot import run_nfw_comparison
    >>> run_nfw_comparison()

Produces a two-panel figure:
    Left:  WL-only recovered halos
    Right: WL+SL recovered halos
Both overlaid on the source catalog with strong-lensing image positions
and ground-truth halo location.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
import arch.metric as metric
from arch.main import fit_lensing_field

# Re-use the toy data builders from the test suite
from tests_nfw_strong import (
    make_nfw_halo,
    make_nfw_strong_system,
    make_weak_lensing_catalog_nfw,
    attach_strong_systems,
)


# ─────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_nfw_wl_vs_sl(
    src: source_obj.Source,
    lenses_wl: halo_obj.NFW_Lens,
    lenses_sl: halo_obj.NFW_Lens,
    halo_true: halo_obj.NFW_Lens,
    xmax: float,
    rchi2_wl: float | None = None,
    rchi2_sl: float | None = None,
    lambda_sl: float | None = None,
    savepath: str | None = None,
):
    """
    Side-by-side comparison of WL-only vs WL+SL recovered NFW halos.

    Both panels show:
      - Weak lensing source galaxies (small grey dots)
      - Strong lensing image positions (diamonds)
      - True halo position (gold star)
      - R200 circle of the true halo (dashed gold)
      - Recovered halo positions, sized by log10(M) (coloured circles)
      - Offset arrow from nearest recovered halo to truth
    """
    true_x = float(halo_true.x[0])
    true_y = float(halo_true.y[0])
    true_mass = float(halo_true.mass[0])

    # R200 for the truth circle
    _, r200_arcsec = halo_true.calc_R200()
    r200_true = float(np.atleast_1d(r200_arcsec)[0])

    fig, (ax_wl, ax_sl) = plt.subplots(
        1, 2, figsize=(15, 7), dpi=150, constrained_layout=True,
    )

    for ax, lenses, label, color, rchi2 in [
        (ax_wl, lenses_wl, "WL only", "C0", rchi2_wl),
        (ax_sl, lenses_sl, "WL + SL", "C2", rchi2_sl),
    ]:
        # ── Source galaxies ──
        ax.scatter(src.x, src.y, s=4, c="0.70", alpha=0.5, zorder=1,
                   label=f"sources (N={len(src.x)})")

        # ── Strong lensing image positions ──
        if hasattr(src, "strong_systems") and src.strong_systems:
            for sys in src.strong_systems:
                ax.scatter(
                    sys.theta_x, sys.theta_y,
                    marker="D", s=70, edgecolors="C3", facecolors="none",
                    linewidths=1.5, zorder=5,
                    label=f"SL images ({sys.system_id})",
                )

        # ── Truth: star + R200 circle ──
        ax.scatter(
            true_x, true_y, marker="*", s=400, c="gold",
            edgecolors="k", linewidths=0.8, zorder=10,
            label=f"truth (M={true_mass:.1e} $M_\\odot$)",
        )
        circle = plt.Circle(
            (true_x, true_y), r200_true,
            fill=False, linestyle="--", linewidth=1.2,
            edgecolor="gold", alpha=0.8, zorder=4,
            label=f"$R_{{200}}$ = {r200_true:.0f}\"",
        )
        ax.add_patch(circle)

        # ── Recovered halos ──
        if len(lenses.x) > 0:
            # Size proportional to log10(mass), clipped for readability
            log_masses = np.log10(np.maximum(lenses.mass, 1e10))
            sizes = np.clip((log_masses - 10) * 30, 20, 300)
            ax.scatter(
                lenses.x, lenses.y, s=sizes,
                c=color, edgecolors="k", linewidths=0.6, alpha=0.85,
                zorder=8, label=f"recovered (N={len(lenses.x)})",
            )

            # Label each halo with its mass
            for lx, ly, lm in zip(lenses.x, lenses.y, lenses.mass):
                ax.annotate(
                    f"{lm:.1e}",
                    (lx, ly), textcoords="offset points",
                    xytext=(6, 6), fontsize=7, color="0.25", zorder=11,
                )

            # ── Offset arrow from nearest halo to truth ──
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
            # Info box with offset and mass of nearest halo
            info_lines = [
                f"$\\Delta$ = {d:.2f}\"",
                f"M = {lenses.mass[idx_near]:.2e} $M_\\odot$",
            ]
            if rchi2 is not None:
                info_lines.append(f"r$\\chi^2$ = {rchi2:.3f}")
            ax.text(
                0.03, 0.03,
                "\n".join(info_lines),
                transform=ax.transAxes, fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=3),
                zorder=12, verticalalignment="bottom",
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
        unique = [(h, l) for h, l in zip(handles, labels_leg)
                  if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7.5, markerscale=0.9)

    # ── Suptitle ──
    suptitle = "NFW Reconstruction:  WL-only  vs  WL + Strong Lensing"
    if lambda_sl is not None and lambda_sl > 1e-8:
        suptitle += f"   ($\\lambda_{{SL}}$ = {lambda_sl:.4f})"
    fig.suptitle(suptitle, fontsize=13)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────

def run_nfw_comparison(
    seed: int = 55,
    savepath: str | None = "nfw_wl_vs_sl_comparison.pdf",
    show: bool = True,
):
    """
    Build a single-NFW-halo toy scenario, run both WL-only and WL+SL
    pipelines, and produce a side-by-side comparison figure.

    Parameters
    ----------
    seed : int
        Random seed for the source catalog.
    savepath : str or None
        File path for saving the figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib Figure
    lenses_wl, lenses_sl : NFW_Lens
    """
    # ── Build scenario ──
    halo_true = make_nfw_halo(x=5.0, y=-3.0, mass=1e15, concentration=8.0, redshift=0.3)
    xmax = 80.0

    src = make_weak_lensing_catalog_nfw(
        halo_true, xmax=xmax, n_sources=100,
        z_source=1.0, sig_shear=0.08, sig_flex=0.015,
        sig_gflex=0.025, seed=seed,
    )

    sys_A = make_nfw_strong_system(
        system_id="integ_nfw_A",
        halo=halo_true,
        beta_offset=1.0,
        z_source=2.0,
        sigma_theta=0.04,
    )
    attach_strong_systems(src, [sys_A])

    use_flags = [True, True, False]

    # ── WL-only ──
    print("Running WL-only NFW pipeline...")
    lenses_wl, rchi2_wl = fit_lensing_field(
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='NFW', z_lens=halo_true.redshift,
        use_strong_lensing=False,
    )

    # ── WL+SL ──
    print("\nRunning WL+SL NFW pipeline...")
    lenses_sl, rchi2_sl = fit_lensing_field(
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='NFW', z_lens=halo_true.redshift,
        use_strong_lensing=True,
    )

    # ── Summary ──
    true_x, true_y = float(halo_true.x[0]), float(halo_true.y[0])
    d_wl = float(np.min(np.hypot(lenses_wl.x - true_x, lenses_wl.y - true_y)))
    d_sl = float(np.min(np.hypot(lenses_sl.x - true_x, lenses_sl.y - true_y)))
    lambda_sl = metric.compute_lambda_sl(src, lenses_sl, use_flags, 'NFW')

    print(f"\n{'='*60}")
    print(f"  WL-only:  N_lens={len(lenses_wl.x)},  "
          f"nearest Δ={d_wl:.2f}\",  rχ²={rchi2_wl:.4f}")
    print(f"  WL+SL:    N_lens={len(lenses_sl.x)},  "
          f"nearest Δ={d_sl:.2f}\",  rχ²={rchi2_sl:.4f}")
    print(f"  λ_SL = {lambda_sl:.6e}")
    print(f"  Improvement: {(1 - d_sl/d_wl)*100:.0f}%")
    print(f"{'='*60}")

    # ── Plot ──
    fig = plot_nfw_wl_vs_sl(
        src=src,
        lenses_wl=lenses_wl,
        lenses_sl=lenses_sl,
        halo_true=halo_true,
        xmax=xmax,
        rchi2_wl=rchi2_wl,
        rchi2_sl=rchi2_sl,
        lambda_sl=lambda_sl,
        savepath=savepath,
    )

    if show:
        plt.show()

    return fig, lenses_wl, lenses_sl


if __name__ == "__main__":
    run_nfw_comparison()