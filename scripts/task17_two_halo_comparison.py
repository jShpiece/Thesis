"""Task 17 — Multi-halo NFW + SL two-halo comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import arch.halo_obj as halo_obj
from arch.main import fit_lensing_field
from tests.tests_nfw_strong import (
    attach_strong_systems,
    make_nfw_halo,
    make_nfw_strong_system,
    make_weak_lensing_catalog_nfw,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Task 17 — Two-halo NFW + SL comparison
# ═══════════════════════════════════════════════════════════════════════════


def build_two_halo_scenario(
    seed: int = 42,
    n_sources: int = 200,
    xmax: float = 100.0,
    z_lens: float = 0.3,
    z_source_wl: float = 1.0,
    z_source_sl: float = 2.0,
):
    """
    Build a two-halo NFW cluster with one SL system per halo.

    Halo A: massive primary at offset from centre
    Halo B: smaller secondary, separated by ~60"

    Returns
    -------
    src : Source (with two strong systems attached)
    halo_true : NFW_Lens (composite two-halo object)
    xmax : float
    """
    # Ground truth: two halos
    halo_true = halo_obj.NFW_Lens(
        x=np.array([-10.0, 50.0]),
        y=np.array([5.0, -15.0]),
        z=np.array([0.0, 0.0]),
        concentration=np.array([8.0, 8.0]),
        mass=np.array([1.0e15, 6.0e14]),
        redshift=z_lens,
        chi2=np.array([0.0, 0.0]),
    )

    # WL catalog lensed by the composite
    src = make_weak_lensing_catalog_nfw(
        halo_true,
        xmax=xmax,
        n_sources=n_sources,
        z_source=z_source_wl,
        sig_shear=0.08,
        sig_flex=0.015,
        sig_gflex=0.025,
        rmin=3.0,
        seed=seed,
    )

    # SL system for halo A (build against isolated halo)
    halo_A_iso = make_nfw_halo(x=-10.0, y=5.0, mass=1.0e15, concentration=8.0, redshift=z_lens)
    sys_A = make_nfw_strong_system(
        system_id="sys_A",
        halo=halo_A_iso,
        beta_offset=1.0,
        z_source=z_source_sl,
        sigma_theta=0.04,
    )

    # SL system for halo B
    halo_B_iso = make_nfw_halo(x=50.0, y=-15.0, mass=6.0e14, concentration=8.0, redshift=z_lens)
    sys_B = make_nfw_strong_system(
        system_id="sys_B",
        halo=halo_B_iso,
        beta_offset=0.8,
        z_source=z_source_sl,
        sigma_theta=0.04,
    )

    attach_strong_systems(src, [sys_A, sys_B])

    return src, halo_true, xmax


def _match_halos(recovered, true_x, true_y, max_distance=40.0):
    """
    For each true halo, find the nearest recovered halo within max_distance.

    Parameters
    ----------
    recovered : NFW_Lens
        Recovered halo set.
    true_x, true_y : array
        True halo positions.
    max_distance : float
        Maximum match distance in arcsec.  Recovered halos farther than
        this are considered unmatched (no real association with the true
        halo) and the match is reported as distance=inf, mass=nan.
        Without this filter, distant spurious halos contaminate the
        statistics by being paired with true halos they don't physically
        explain.

    Returns
    -------
    list of dicts with keys: true_idx, rec_idx, distance, mass_rec.
    Unmatched true halos have rec_idx=None, distance=inf, mass_rec=nan.
    """
    matches = []
    for i in range(len(true_x)):
        if len(recovered.x) == 0:
            matches.append(
                {
                    "true_idx": i,
                    "rec_idx": None,
                    "distance": np.inf,
                    "mass_rec": np.nan,
                    "conc_rec": np.nan,
                }
            )
            continue
        dists = np.hypot(recovered.x - true_x[i], recovered.y - true_y[i])
        j = int(np.argmin(dists))
        d = float(dists[j])
        if d > max_distance:
            matches.append(
                {
                    "true_idx": i,
                    "rec_idx": None,
                    "distance": np.inf,
                    "mass_rec": np.nan,
                    "conc_rec": np.nan,
                }
            )
        else:
            matches.append(
                {
                    "true_idx": i,
                    "rec_idx": j,
                    "distance": d,
                    "mass_rec": float(recovered.mass[j]),
                    "conc_rec": float(recovered.concentration[j]),
                }
            )
    return matches


def plot_two_halo_comparison(
    src,
    lenses_wl,
    lenses_sl,
    halo_true,
    xmax,
    rchi2_wl=None,
    rchi2_sl=None,
    savepath=None,
):
    """Side-by-side WL vs WL+SL for a two-halo scenario."""

    true_x = halo_true.x
    true_y = halo_true.y
    true_mass = halo_true.mass
    _, r200_arcsec = halo_true.calc_R200()
    r200_arcsec = np.atleast_1d(r200_arcsec)

    fig, (ax_wl, ax_sl) = plt.subplots(
        1,
        2,
        figsize=(16, 7.5),
        dpi=150,
        constrained_layout=True,
    )

    for ax, lenses, label, color, rchi2 in [
        (ax_wl, lenses_wl, "WL only", "C0", rchi2_wl),
        (ax_sl, lenses_sl, "WL + SL", "C2", rchi2_sl),
    ]:
        # Sources
        ax.scatter(
            src.x, src.y, s=3, c="0.75", alpha=0.4, zorder=1, label=f"sources (N={len(src.x)})"
        )

        # SL images
        if hasattr(src, "strong_systems"):
            for sl_sys in src.strong_systems:
                ax.scatter(
                    sl_sys.theta_x,
                    sl_sys.theta_y,
                    marker="D",
                    s=55,
                    edgecolors="C3",
                    facecolors="none",
                    linewidths=1.5,
                    zorder=5,
                    label=f"SL: {sl_sys.system_id}",
                )

        # True halos: stars + R200 circles
        for i in range(len(true_x)):
            lbl = f"truth {chr(65+i)} (M={true_mass[i]:.1e})" if i < 2 else None
            ax.scatter(
                true_x[i],
                true_y[i],
                marker="*",
                s=350,
                c="gold",
                edgecolors="k",
                linewidths=0.8,
                zorder=10,
                label=lbl,
            )
            circle = plt.Circle(
                (true_x[i], true_y[i]),
                r200_arcsec[i],
                fill=False,
                linestyle="--",
                linewidth=1.0,
                edgecolor="gold",
                alpha=0.6,
                zorder=4,
            )
            ax.add_patch(circle)

        # Recovered halos
        if len(lenses.x) > 0:
            log_m = np.log10(np.maximum(lenses.mass, 1e10))
            sizes = np.clip((log_m - 10) * 30, 20, 300)
            ax.scatter(
                lenses.x,
                lenses.y,
                s=sizes,
                c=color,
                edgecolors="k",
                linewidths=0.6,
                alpha=0.85,
                zorder=8,
                label=f"recovered (N={len(lenses.x)})",
            )

            for lx, ly, lm in zip(lenses.x, lenses.y, lenses.mass):
                ax.annotate(
                    f"{lm:.1e}",
                    (lx, ly),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=6.5,
                    color="0.3",
                    zorder=11,
                )

            # Arrows to nearest truth for each true halo
            matches = _match_halos(lenses, true_x, true_y)
            info_lines = []
            for m in matches:
                if m["rec_idx"] is not None and m["distance"] < 60:
                    ax.annotate(
                        "",
                        xy=(true_x[m["true_idx"]], true_y[m["true_idx"]]),
                        xytext=(lenses.x[m["rec_idx"]], lenses.y[m["rec_idx"]]),
                        arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.5, shrinkA=3, shrinkB=3),
                        zorder=9,
                    )
                    info_lines.append(
                        f"Halo {chr(65+m['true_idx'])}: "
                        f"Δ={m['distance']:.1f}\", "
                        f"M={m['mass_rec']:.1e}"
                    )

            if rchi2 is not None:
                info_lines.append(f"rχ² = {rchi2:.3f}")
            if info_lines:
                ax.text(
                    0.03,
                    0.03,
                    "\n".join(info_lines),
                    transform=ax.transAxes,
                    fontsize=7.5,
                    bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9, pad=3),
                    zorder=12,
                    verticalalignment="bottom",
                )

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x  (arcsec)")
        ax.set_ylabel("y  (arcsec)")
        ax.set_title(label, fontsize=13, fontweight="bold")

        handles, labels_leg = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels_leg) if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7, markerscale=0.8)

    fig.suptitle("NFW Two-Halo Reconstruction:  WL-only  vs  WL + Strong Lensing", fontsize=13)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def run_two_halo_comparison(
    seed: int = 42,
    savepath: str | None = "nfw_two_halo_comparison.pdf",
    show: bool = True,
):
    """
    Task 17: Build two-halo scenario, run both pipelines, compare.
    """
    src, halo_true, xmax = build_two_halo_scenario(seed=seed)
    use_flags = [True, True, False]

    print("Running WL-only NFW pipeline (two halos)...")
    lenses_wl, rchi2_wl = fit_lensing_field(
        src,
        xmax,
        flags=True,
        use_flags=use_flags,
        lens_type="NFW",
        z_lens=halo_true.redshift,
        use_strong_lensing=False,
    )

    print("\nRunning WL+SL NFW pipeline (two halos)...")
    lenses_sl, rchi2_sl = fit_lensing_field(
        src,
        xmax,
        flags=True,
        use_flags=use_flags,
        lens_type="NFW",
        z_lens=halo_true.redshift,
        use_strong_lensing=True,
    )

    # Summary
    matches_wl = _match_halos(lenses_wl, halo_true.x, halo_true.y)
    matches_sl = _match_halos(lenses_sl, halo_true.x, halo_true.y)

    print(f"\n{'='*60}")
    print("  TWO-HALO NFW COMPARISON")
    print(f"{'='*60}")
    for i, (mw, ms) in enumerate(zip(matches_wl, matches_sl)):
        print(f"  Halo {chr(65+i)} (true M={halo_true.mass[i]:.1e}):")
        print(f"    WL:    Δ={mw['distance']:.2f}\"  M_rec={mw['mass_rec']:.2e}")
        print(f"    WL+SL: Δ={ms['distance']:.2f}\"  M_rec={ms['mass_rec']:.2e}")
    print(f"  WL rχ²={rchi2_wl:.4f}   WL+SL rχ²={rchi2_sl:.4f}")
    print(f"{'='*60}")

    fig = plot_two_halo_comparison(
        src,
        lenses_wl,
        lenses_sl,
        halo_true,
        xmax,
        rchi2_wl=rchi2_wl,
        rchi2_sl=rchi2_sl,
        savepath=savepath,
    )
    if show:
        plt.show()

    return fig, lenses_wl, lenses_sl


if __name__ == "__main__":
    run_two_halo_comparison()
