"""
Task 17: Multi-halo NFW + SL integration test
Task 18: Statistical Monte Carlo over random cluster realizations

Task 17 builds a two-halo NFW cluster with one SL system per halo,
runs WL-only and WL+SL pipelines, and produces a side-by-side figure.

Task 18 draws N random cluster realizations (varying positions, masses,
source noise seeds) and collects positional and mass recovery statistics
for both pipelines, producing summary histograms and scatter plots.

Usage:
    >>> from nfw_multihalo_mc import run_two_halo_comparison, run_monte_carlo
    >>> run_two_halo_comparison()          # Task 17
    >>> run_monte_carlo(n_realizations=50) # Task 18
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional
import time
import os
import multiprocessing
from functools import partial

import arch.halo_obj as halo_obj
import arch.source_obj as source_obj
import arch.utils as utils
import arch.metric as metric
from arch.main import fit_lensing_field

from tests_nfw_strong import (
    make_nfw_halo,
    make_nfw_strong_system,
    make_weak_lensing_catalog_nfw,
    attach_strong_systems,
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
        halo_true, xmax=xmax, n_sources=n_sources,
        z_source=z_source_wl, sig_shear=0.08, sig_flex=0.015,
        sig_gflex=0.025, rmin=3.0, seed=seed,
    )

    # SL system for halo A (build against isolated halo)
    halo_A_iso = make_nfw_halo(
        x=-10.0, y=5.0, mass=1.0e15, concentration=8.0, redshift=z_lens
    )
    sys_A = make_nfw_strong_system(
        system_id="sys_A",
        halo=halo_A_iso,
        beta_offset=1.0,
        z_source=z_source_sl,
        sigma_theta=0.04,
    )

    # SL system for halo B
    halo_B_iso = make_nfw_halo(
        x=50.0, y=-15.0, mass=6.0e14, concentration=8.0, redshift=z_lens
    )
    sys_B = make_nfw_strong_system(
        system_id="sys_B",
        halo=halo_B_iso,
        beta_offset=0.8,
        z_source=z_source_sl,
        sigma_theta=0.04,
    )

    attach_strong_systems(src, [sys_A, sys_B])

    return src, halo_true, xmax


def _match_halos(recovered, true_x, true_y, max_distance=40.0, max_distance=40.0):
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
            matches.append({"true_idx": i, "rec_idx": None, "distance": np.inf,
                            "mass_rec": np.nan, "conc_rec": np.nan})
            continue
        dists = np.hypot(recovered.x - true_x[i], recovered.y - true_y[i])
        j = int(np.argmin(dists))
        d = float(dists[j])
        if d > max_distance:
            matches.append({"true_idx": i, "rec_idx": None, "distance": np.inf,
                            "mass_rec": np.nan, "conc_rec": np.nan})
        else:
            matches.append({
                "true_idx": i,
                "rec_idx": j,
                "distance": d,
                "mass_rec": float(recovered.mass[j]),
                "conc_rec": float(recovered.concentration[j]),
            })
    return matches


def plot_two_halo_comparison(
    src, lenses_wl, lenses_sl, halo_true, xmax,
    rchi2_wl=None, rchi2_sl=None,
    savepath=None,
):
    """Side-by-side WL vs WL+SL for a two-halo scenario."""

    true_x = halo_true.x
    true_y = halo_true.y
    true_mass = halo_true.mass
    _, r200_arcsec = halo_true.calc_R200()
    r200_arcsec = np.atleast_1d(r200_arcsec)

    fig, (ax_wl, ax_sl) = plt.subplots(
        1, 2, figsize=(16, 7.5), dpi=150, constrained_layout=True,
    )

    for ax, lenses, label, color, rchi2 in [
        (ax_wl, lenses_wl, "WL only", "C0", rchi2_wl),
        (ax_sl, lenses_sl, "WL + SL", "C2", rchi2_sl),
    ]:
        # Sources
        ax.scatter(src.x, src.y, s=3, c="0.75", alpha=0.4, zorder=1,
                   label=f"sources (N={len(src.x)})")

        # SL images
        if hasattr(src, "strong_systems"):
            for sys in src.strong_systems:
                ax.scatter(
                    sys.theta_x, sys.theta_y,
                    marker="D", s=55, edgecolors="C3", facecolors="none",
                    linewidths=1.5, zorder=5,
                    label=f"SL: {sys.system_id}",
                )

        # True halos: stars + R200 circles
        for i in range(len(true_x)):
            lbl = f"truth {chr(65+i)} (M={true_mass[i]:.1e})" if i < 2 else None
            ax.scatter(true_x[i], true_y[i], marker="*", s=350, c="gold",
                       edgecolors="k", linewidths=0.8, zorder=10, label=lbl)
            circle = plt.Circle(
                (true_x[i], true_y[i]), r200_arcsec[i],
                fill=False, linestyle="--", linewidth=1.0,
                edgecolor="gold", alpha=0.6, zorder=4,
            )
            ax.add_patch(circle)

        # Recovered halos
        if len(lenses.x) > 0:
            log_m = np.log10(np.maximum(lenses.mass, 1e10))
            sizes = np.clip((log_m - 10) * 30, 20, 300)
            ax.scatter(lenses.x, lenses.y, s=sizes, c=color,
                       edgecolors="k", linewidths=0.6, alpha=0.85,
                       zorder=8, label=f"recovered (N={len(lenses.x)})")

            for lx, ly, lm in zip(lenses.x, lenses.y, lenses.mass):
                ax.annotate(f"{lm:.1e}", (lx, ly),
                            textcoords="offset points", xytext=(5, 5),
                            fontsize=6.5, color="0.3", zorder=11)

            # Arrows to nearest truth for each true halo
            matches = _match_halos(lenses, true_x, true_y)
            info_lines = []
            for m in matches:
                if m["rec_idx"] is not None and m["distance"] < 60:
                    ax.annotate("",
                        xy=(true_x[m["true_idx"]], true_y[m["true_idx"]]),
                        xytext=(lenses.x[m["rec_idx"]], lenses.y[m["rec_idx"]]),
                        arrowprops=dict(arrowstyle="-|>", color="C1", lw=1.5,
                                        shrinkA=3, shrinkB=3),
                        zorder=9)
                    info_lines.append(
                        f"Halo {chr(65+m['true_idx'])}: "
                        f"Δ={m['distance']:.1f}\", "
                        f"M={m['mass_rec']:.1e}"
                    )

            if rchi2 is not None:
                info_lines.append(f"rχ² = {rchi2:.3f}")
            if info_lines:
                ax.text(0.03, 0.03, "\n".join(info_lines),
                        transform=ax.transAxes, fontsize=7.5,
                        bbox=dict(facecolor="white", edgecolor="0.7",
                                  alpha=0.9, pad=3),
                        zorder=12, verticalalignment="bottom")

        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x  (arcsec)")
        ax.set_ylabel("y  (arcsec)")
        ax.set_title(label, fontsize=13, fontweight="bold")

        handles, labels_leg = ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels_leg)
                  if l not in seen and not seen.add(l)]
        ax.legend(*zip(*unique), loc="upper left", fontsize=7, markerscale=0.8)

    fig.suptitle("NFW Two-Halo Reconstruction:  WL-only  vs  WL + Strong Lensing",
                 fontsize=13)

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
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='NFW', z_lens=halo_true.redshift,
        use_strong_lensing=False,
    )

    print("\nRunning WL+SL NFW pipeline (two halos)...")
    lenses_sl, rchi2_sl = fit_lensing_field(
        src, xmax, flags=True, use_flags=use_flags,
        lens_type='NFW', z_lens=halo_true.redshift,
        use_strong_lensing=True,
    )

    # Summary
    matches_wl = _match_halos(lenses_wl, halo_true.x, halo_true.y)
    matches_sl = _match_halos(lenses_sl, halo_true.x, halo_true.y)

    print(f"\n{'='*60}")
    print(f"  TWO-HALO NFW COMPARISON")
    print(f"{'='*60}")
    for i, (mw, ms) in enumerate(zip(matches_wl, matches_sl)):
        print(f"  Halo {chr(65+i)} (true M={halo_true.mass[i]:.1e}):")
        print(f"    WL:    Δ={mw['distance']:.2f}\"  M_rec={mw['mass_rec']:.2e}")
        print(f"    WL+SL: Δ={ms['distance']:.2f}\"  M_rec={ms['mass_rec']:.2e}")
    print(f"  WL rχ²={rchi2_wl:.4f}   WL+SL rχ²={rchi2_sl:.4f}")
    print(f"{'='*60}")

    fig = plot_two_halo_comparison(
        src, lenses_wl, lenses_sl, halo_true, xmax,
        rchi2_wl=rchi2_wl, rchi2_sl=rchi2_sl,
        savepath=savepath,
    )
    if show:
        plt.show()

    return fig, lenses_wl, lenses_sl


# ═══════════════════════════════════════════════════════════════════════════
#  Task 18 — Monte Carlo statistical validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RealizationResult:
    """Result from one cluster realization."""
    seed: int
    n_true_halos: int
    true_x: np.ndarray
    true_y: np.ndarray
    true_mass: np.ndarray
    true_concentration: np.ndarray
    n_sl_systems: int          # how many SL systems were created (0 if subcritical)
    # WL-only
    n_rec_wl: int
    delta_wl: np.ndarray       # per-true-halo offset (arcsec)
    mass_rec_wl: np.ndarray    # per-true-halo recovered mass
    conc_rec_wl: np.ndarray    # per-true-halo recovered concentration (slaved to M-c)
    rchi2_wl: float
    # WL+SL
    n_rec_sl: int
    delta_sl: np.ndarray
    mass_rec_sl: np.ndarray
    conc_rec_sl: np.ndarray    # per-true-halo recovered concentration (jointly fit)
    rchi2_sl: float
    # Timing
    time_wl: float
    time_sl: float
    # Success flags
    ok_wl: bool
    ok_sl: bool


def _draw_random_cluster(rng, n_halos=1, z_lens=0.3):
    """
    Draw a random cluster configuration.

    Concentrations are drawn from the Duffy et al. (2008) mass-concentration
    relation with log-normal intrinsic scatter (sigma_ln_c ~ 0.25).
    This matches the pipeline's internal M-c relation, so the initial
    guess starts close to truth — the validated WL regime.

    Halos that scatter to high concentration (c > ~5) will be supercritical
    and produce strong lensing.  This is physically correct: real SL clusters
    are concentration outliers.

    Parameters
    ----------
    rng : np.random.Generator
    n_halos : int
        Number of halos (1 or 2).
    z_lens : float

    Returns
    -------
    x, y, mass, concentration : arrays
    """
    # Positions: primary near centre, secondary offset by 30-80"
    x = np.zeros(n_halos)
    y = np.zeros(n_halos)
    x[0] = rng.uniform(-15, 15)
    y[0] = rng.uniform(-15, 15)
    if n_halos > 1:
        angle = rng.uniform(0, 2 * np.pi)
        sep = rng.uniform(30, 80)
        x[1] = x[0] + sep * np.cos(angle)
        y[1] = y[0] + sep * np.sin(angle)

    # Masses: primary 5e14–2e15, secondary 2e14–8e14
    mass = np.zeros(n_halos)
    mass[0] = 10 ** rng.uniform(14.7, 15.3)
    if n_halos > 1:
        mass[1] = 10 ** rng.uniform(14.3, 14.9)

    # Concentration from Duffy et al. (2008) with log-normal scatter
    # c_Duffy = 5.71 * (M / 2e12)^(-0.084) * (1+z)^(-0.47)
    sigma_ln_c = 0.25   # intrinsic scatter in ln(c)
    c_mean = 5.71 * (mass / 2e12) ** (-0.084) * (1 + z_lens) ** (-0.47)
    concentration = c_mean * np.exp(rng.normal(0, sigma_ln_c, size=n_halos))
    concentration = np.maximum(concentration, 1.0)  # floor at c=1

    return x, y, mass, concentration


def run_single_realization(
    seed: int,
    n_halos: int = 1,
    n_sources: int = 150,
    xmax: float = 80.0,
    z_lens: float = 0.3,
    z_source_wl: float = 1.0,
    z_source_sl: float = 2.0,
    use_flags: list | None = None,
    verbose: bool = False,
) -> RealizationResult:
    """
    Run one realization: draw a random cluster, build WL+SL data,
    run both pipelines, collect results.
    """
    if use_flags is None:
        use_flags = [True, True, True]  # shear + flexion + g-flexion

    rng = np.random.default_rng(seed)

    # Draw random cluster
    tx, ty, tmass, tc = _draw_random_cluster(rng, n_halos=n_halos, z_lens=z_lens)

    halo_true = halo_obj.NFW_Lens(
        x=tx, y=ty, z=np.zeros(n_halos),
        concentration=tc, mass=tmass,
        redshift=z_lens,
        chi2=np.zeros(n_halos),
    )
    # Concentration is left at the scattered draw — the data are
    # generated with the true c, while the pipeline assumes c_Duffy(M).
    # This M-c mismatch is what flux ratios should correct.

    # WL catalog
    src = make_weak_lensing_catalog_nfw(
        halo_true, xmax=xmax, n_sources=n_sources,
        z_source=z_source_wl, seed=seed,
    )

    # SL systems (one per halo, built against isolated halos)
    systems = []
    for i in range(n_halos):
        try:
            halo_iso = make_nfw_halo(
                x=tx[i], y=ty[i], mass=tmass[i],
                concentration=tc[i], redshift=z_lens,
            )
            # Use the scattered c directly — supercriticality reflects truth
            halo_iso.calculate_concentration()
            sys_i = make_nfw_strong_system(
                system_id=f"sys_{i}",
                halo=halo_iso,
                beta_offset=0.8,
                z_source=z_source_sl,
                sigma_theta=0.04,
                include_flux=True,
                sigma_flux_frac=0.05,
                flux_noise_seed=seed * 10 + i,  # unique per halo, deterministic
            )
            systems.append(sys_i)
        except ValueError:
            # Halo not supercritical — skip SL for this halo
            if verbose:
                print(f"  Seed {seed}, halo {i}: subcritical, no SL system")
    attach_strong_systems(src, systems)

    # ── WL-only pipeline ──
    ok_wl = True
    t0 = time.time()
    try:
        lenses_wl, rchi2_wl = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=z_lens,
            use_strong_lensing=False,
        )
    except Exception as e:
        if verbose:
            print(f"  Seed {seed} WL EXCEPTION: {e}")
        lenses_wl = halo_obj.NFW_Lens(
            x=[], y=[], z=[], concentration=[], mass=[],
            redshift=z_lens, chi2=[])
        rchi2_wl = np.inf
        ok_wl = False
    time_wl = time.time() - t0

    # ── WL+SL pipeline ──
    ok_sl = True
    t0 = time.time()
    try:
        lenses_sl, rchi2_sl = fit_lensing_field(
            src, xmax, flags=False, use_flags=use_flags,
            lens_type='NFW', z_lens=z_lens,
            use_strong_lensing=True,
        )
    except Exception as e:
        if verbose:
            print(f"  Seed {seed} SL EXCEPTION: {e}")
        lenses_sl = halo_obj.NFW_Lens(
            x=[], y=[], z=[], concentration=[], mass=[],
            redshift=z_lens, chi2=[])
        rchi2_sl = np.inf
        ok_sl = False
    time_sl = time.time() - t0

    # ── Match recovered to true ──
    matches_wl = _match_halos(lenses_wl, tx, ty)
    matches_sl = _match_halos(lenses_sl, tx, ty)

    delta_wl = np.array([m["distance"] for m in matches_wl])
    delta_sl = np.array([m["distance"] for m in matches_sl])
    mass_wl = np.array([m["mass_rec"] for m in matches_wl])
    mass_sl = np.array([m["mass_rec"] for m in matches_sl])
    conc_wl = np.array([m["conc_rec"] for m in matches_wl])
    conc_sl = np.array([m["conc_rec"] for m in matches_sl])

    if verbose:
        for i in range(n_halos):
            print(f"  Seed {seed}, halo {i}: c_true={tc[i]:.2f}  "
                  f"SL={'yes' if len(systems) > i else 'no'}  "
                  f"Δ_WL={delta_wl[i]:.1f}\" Δ_SL={delta_sl[i]:.1f}\" "
                  f"M_WL={mass_wl[i]:.1e}/c={conc_wl[i]:.1f}  "
                  f"M_SL={mass_sl[i]:.1e}/c={conc_sl[i]:.1f}")

    return RealizationResult(
        seed=seed, n_true_halos=n_halos,
        true_x=tx, true_y=ty, true_mass=tmass,
        true_concentration=tc.copy(),
        n_sl_systems=len(systems),
        n_rec_wl=len(lenses_wl.x), delta_wl=delta_wl,
        mass_rec_wl=mass_wl, conc_rec_wl=conc_wl, rchi2_wl=rchi2_wl,
        n_rec_sl=len(lenses_sl.x), delta_sl=delta_sl,
        mass_rec_sl=mass_sl, conc_rec_sl=conc_sl, rchi2_sl=rchi2_sl,
        time_wl=time_wl, time_sl=time_sl,
        ok_wl=ok_wl, ok_sl=ok_sl,
    )


def _mc_worker(args):
    """
    Thin wrapper for multiprocessing — unpacks a dict of kwargs
    and calls run_single_realization.  Module-level function so it's
    picklable by Pool.
    """
    return run_single_realization(**args)


def run_monte_carlo(
    n_realizations: int = 50,
    n_halos: int = 1,
    n_sources: int = 150,
    xmax: float = 80.0,
    seed_offset: int = 1000,
    n_workers: int | None = None,
    savepath: str | None = "nfw_mc_results.pdf",
    show: bool = True,
    verbose: bool = True,
):
    """
    Task 18: Run N random cluster realizations, collect statistics.

    Supports parallel execution via multiprocessing.  Each realization
    is fully independent (different seed, no shared state), so scaling
    is near-linear up to the number of physical cores.

    Parameters
    ----------
    n_realizations : int
        Number of random cluster draws.
    n_halos : int
        Halos per cluster (1 or 2).
    n_sources : int
        WL sources per realization.
    xmax : float
        Field half-width.
    seed_offset : int
        Base seed (realization i uses seed_offset + i).
    n_workers : int or None
        Number of parallel workers.
        None → auto-detect (physical cores minus 1, capped at n_realizations).
        1 → serial execution with verbose per-realization output.
    savepath : str or None
        Path for summary figure.
    show : bool
        Whether to call plt.show().
    verbose : bool
        Print per-realization diagnostics (serial only) and summary.

    Returns
    -------
    results : list of RealizationResult
    fig : matplotlib Figure
    """
    # ── Determine worker count ──
    if n_workers is None:
        n_cpu = os.cpu_count() or 1
        n_workers = min(max(1, n_cpu - 1), n_realizations)
    n_workers = max(1, n_workers)
    parallel = n_workers > 1

    # ── Build argument list ──
    job_args = [
        {
            "seed": seed_offset + i,
            "n_halos": n_halos,
            "n_sources": n_sources,
            "xmax": xmax,
            "verbose": verbose and not parallel,  # suppress per-worker output in parallel
        }
        for i in range(n_realizations)
    ]

    # ── Execute ──
    t_start = time.time()

    if parallel:
        print(f"Running {n_realizations} realizations on {n_workers} workers...")
        results: List[RealizationResult] = []
        with multiprocessing.Pool(processes=n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_mc_worker, job_args)):
                results.append(r)
                done = i + 1
                bar_len = 40
                filled = int(bar_len * done / n_realizations)
                bar = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (n_realizations - done) / rate if rate > 0 else 0
                print(f"\r  [{bar}] {done}/{n_realizations}  "
                      f"{rate:.1f} it/s  ETA {eta:.0f}s", end="", flush=True)
        print()  # newline after progress bar
    else:
        print(f"Running {n_realizations} realizations (serial)...")
        results = []
        for i, args in enumerate(job_args):
            if verbose:
                print(f"\n── Realization {i+1}/{n_realizations} (seed={args['seed']}) ──")
            r = _mc_worker(args)
            results.append(r)

    t_total = time.time() - t_start

    # Sort by seed for reproducible ordering (imap_unordered returns out of order)
    results.sort(key=lambda r: r.seed)

    # ── Collect statistics ──
    all_delta_wl = np.concatenate([r.delta_wl for r in results if r.ok_wl])
    all_delta_sl = np.concatenate([r.delta_sl for r in results if r.ok_sl])
    all_mass_true = np.concatenate([r.true_mass for r in results if r.ok_wl])
    all_mass_wl = np.concatenate([r.mass_rec_wl for r in results if r.ok_wl])
    all_mass_sl = np.concatenate([r.mass_rec_sl for r in results if r.ok_sl])

    finite_wl = np.isfinite(all_delta_wl)
    finite_sl = np.isfinite(all_delta_sl)

    n_ok_wl = np.sum(finite_wl)
    n_ok_sl = np.sum(finite_sl)
    n_fail_wl = np.sum(~np.array([r.ok_wl for r in results]))
    n_fail_sl = np.sum(~np.array([r.ok_sl for r in results]))

    # ── Print summary ──
    n_with_sl = sum(1 for r in results if r.n_sl_systems > 0)
    all_conc = np.concatenate([r.true_concentration for r in results])
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SUMMARY")
    print(f"  {n_realizations} realizations, {n_halos} halo(s) each")
    print(f"  {n_workers} worker(s), {t_total:.1f}s total "
          f"({t_total/n_realizations:.1f}s/realization)")
    print(f"{'='*60}")
    print(f"  True concentration: median={np.median(all_conc):.2f}  "
          f"mean={np.mean(all_conc):.2f}  range=[{np.min(all_conc):.2f}, {np.max(all_conc):.2f}]")
    print(f"  Realizations with SL: {n_with_sl}/{n_realizations} "
          f"({100*n_with_sl/n_realizations:.0f}%)")
    print(f"  Pipeline failures:  WL={n_fail_wl}  WL+SL={n_fail_sl}")
    print(f"  Matched halos:      WL={n_ok_wl}  WL+SL={n_ok_sl}")

    if n_ok_wl > 0:
        d_wl = all_delta_wl[finite_wl]
        print(f"\n  Positional offset (arcsec):")
        print(f"    WL:    median={np.median(d_wl):.1f}  "
              f"mean={np.mean(d_wl):.1f}  std={np.std(d_wl):.1f}")
    if n_ok_sl > 0:
        d_sl = all_delta_sl[finite_sl]
        print(f"    WL+SL: median={np.median(d_sl):.1f}  "
              f"mean={np.mean(d_sl):.1f}  std={np.std(d_sl):.1f}")

    if n_ok_wl > 0 and n_ok_sl > 0:
        # Paired comparison (same realization, same halo)
        paired_wl = []
        paired_sl = []
        paired_mass_ratio_wl = []
        paired_mass_ratio_sl = []
        for r in results:
            if r.ok_wl and r.ok_sl:
                for dw, ds, mt, mw, ms in zip(
                    r.delta_wl, r.delta_sl,
                    r.true_mass, r.mass_rec_wl, r.mass_rec_sl,
                ):
                    if np.isfinite(dw) and np.isfinite(ds):
                        paired_wl.append(dw)
                        paired_sl.append(ds)
                        if mt > 0 and np.isfinite(mw) and np.isfinite(ms):
                            paired_mass_ratio_wl.append(mw / mt)
                            paired_mass_ratio_sl.append(ms / mt)
        paired_wl = np.array(paired_wl)
        paired_sl = np.array(paired_sl)
        if len(paired_wl) > 0:
            improved_pos = np.sum(paired_sl < paired_wl)
            print(f"\n  Paired positional comparison ({len(paired_wl)} halo-pairs):")
            print(f"    SL improved position: {improved_pos}/{len(paired_wl)} "
                  f"({100*improved_pos/len(paired_wl):.0f}%)")
            print(f"    Median Δ improvement: "
                  f"{np.median(paired_wl - paired_sl):.1f}\"")

        paired_mass_ratio_wl = np.array(paired_mass_ratio_wl)
        paired_mass_ratio_sl = np.array(paired_mass_ratio_sl)
        if len(paired_mass_ratio_wl) > 0:
            # Mass accuracy: |log10(M_rec/M_true)| — 0 = perfect
            log_err_wl = np.abs(np.log10(np.maximum(paired_mass_ratio_wl, 1e-10)))
            log_err_sl = np.abs(np.log10(np.maximum(paired_mass_ratio_sl, 1e-10)))
            improved_mass = np.sum(log_err_sl < log_err_wl)
            print(f"\n  Mass recovery (M_rec / M_true):")
            print(f"    WL:    median={np.median(paired_mass_ratio_wl):.2f}  "
                  f"mean={np.mean(paired_mass_ratio_wl):.2f}  "
                  f"std={np.std(paired_mass_ratio_wl):.2f}")
            print(f"    WL+SL: median={np.median(paired_mass_ratio_sl):.2f}  "
                  f"mean={np.mean(paired_mass_ratio_sl):.2f}  "
                  f"std={np.std(paired_mass_ratio_sl):.2f}")
            print(f"    SL improved mass: {improved_mass}/{len(log_err_wl)} "
                  f"({100*improved_mass/len(log_err_wl):.0f}%)")
            print(f"    Median |log10(M_rec/M_true)| : "
                  f"WL={np.median(log_err_wl):.3f}  "
                  f"WL+SL={np.median(log_err_sl):.3f}")

        # ── SL-subset: only realizations where SL was available ──
        sl_paired_wl = []
        sl_paired_sl = []
        sl_mass_ratio_wl = []
        sl_mass_ratio_sl = []
        sl_c_true = []
        sl_c_wl = []
        sl_c_sl = []
        for r in results:
            if r.ok_wl and r.ok_sl and r.n_sl_systems > 0:
                for dw, ds, mt, mw, ms, ct, cw, cs in zip(
                    r.delta_wl, r.delta_sl,
                    r.true_mass, r.mass_rec_wl, r.mass_rec_sl,
                    r.true_concentration, r.conc_rec_wl, r.conc_rec_sl,
                ):
                    if np.isfinite(dw) and np.isfinite(ds):
                        sl_paired_wl.append(dw)
                        sl_paired_sl.append(ds)
                        if mt > 0 and np.isfinite(mw) and np.isfinite(ms):
                            sl_mass_ratio_wl.append(mw / mt)
                            sl_mass_ratio_sl.append(ms / mt)
                        if np.isfinite(cw) and np.isfinite(cs):
                            sl_c_true.append(ct)
                            sl_c_wl.append(cw)
                            sl_c_sl.append(cs)
        if len(sl_paired_wl) > 0:
            sl_pw = np.array(sl_paired_wl)
            sl_ps = np.array(sl_paired_sl)
            sl_mrw = np.array(sl_mass_ratio_wl)
            sl_mrs = np.array(sl_mass_ratio_sl)
            print(f"\n  SL-SUBSET ({len(sl_pw)} halo-pairs with SL data):")
            pos_imp = np.sum(sl_ps < sl_pw)
            print(f"    Position: SL improved {pos_imp}/{len(sl_pw)} "
                  f"({100*pos_imp/len(sl_pw):.0f}%)")
            if len(sl_mrw) > 0:
                le_wl = np.abs(np.log10(np.maximum(sl_mrw, 1e-10)))
                le_sl = np.abs(np.log10(np.maximum(sl_mrs, 1e-10)))
                mass_imp = np.sum(le_sl < le_wl)
                print(f"    Mass M_rec/M_true:  WL median={np.median(sl_mrw):.2f}  "
                      f"WL+SL median={np.median(sl_mrs):.2f}")
                print(f"    Mass: SL improved {mass_imp}/{len(le_wl)} "
                      f"({100*mass_imp/len(le_wl):.0f}%)")
                print(f"    |log10(M_rec/M_true)| : "
                      f"WL={np.median(le_wl):.3f}  "
                      f"WL+SL={np.median(le_sl):.3f}")

            if len(sl_c_true) > 0:
                ct = np.array(sl_c_true)
                cw = np.array(sl_c_wl)
                cs = np.array(sl_c_sl)
                # |c_rec - c_true| as accuracy metric
                err_wl = np.abs(cw - ct)
                err_sl = np.abs(cs - ct)
                conc_imp = np.sum(err_sl < err_wl)
                print(f"    Concentration:  c_true median={np.median(ct):.2f}  "
                      f"WL median={np.median(cw):.2f}  "
                      f"WL+SL median={np.median(cs):.2f}")
                print(f"    Concentration: SL improved {conc_imp}/{len(ct)} "
                      f"({100*conc_imp/len(ct):.0f}%)")
                print(f"    |c_rec - c_true|:  "
                      f"WL={np.median(err_wl):.2f}  "
                      f"WL+SL={np.median(err_sl):.2f}")
                # How many hit the bounds?
                n_lo = int(np.sum(cs <= 2.01))
                n_hi = int(np.sum(cs >= 14.99))
                if n_lo + n_hi > 0:
                    print(f"    c_WL+SL bounds: {n_lo} at lower (2.0), {n_hi} at upper (15.0)")

    # ── Summary figure ──
    fig = _plot_mc_summary(
        results, all_delta_wl, all_delta_sl,
        all_mass_true, all_mass_wl, all_mass_sl,
        finite_wl, finite_sl, n_halos,
        savepath=savepath,
    )
    if show:
        plt.show()

    return results, fig


def _plot_mc_summary(
    results, all_delta_wl, all_delta_sl,
    all_mass_true, all_mass_wl, all_mass_sl,
    finite_wl, finite_sl, n_halos,
    savepath=None,
):
    """
    Three-panel summary focused on mass and concentration recovery.

    Panel 1: |log10(M_rec/M_true)| histogram for the SL subset.
             Distribution of mass-recovery error, WL vs WL+SL overlaid.
    Panel 2: Mass recovery scatter (M_rec vs M_true) — SL subset only,
             paired blue circle (WL) and green diamond (WL+SL) at same
             M_true, colour-coded by whether WL+SL improved.
    Panel 3: Concentration recovery (c_rec vs c_true) — SL subset only,
             paired points showing whether the joint fit moves c toward
             truth.  This is the diagnostic for whether SL flux ratios
             are actually constraining concentration.
    """

    # ── Collect SL-subset data ──
    mt, mw, ms = [], [], []
    ct, cw, cs = [], [], []
    n_no = 0

    for r in results:
        if not (r.ok_wl and r.ok_sl):
            continue
        if r.n_sl_systems == 0:
            n_no += 1
            continue
        for i in range(len(r.true_mass)):
            if not (np.isfinite(r.delta_wl[i]) and np.isfinite(r.delta_sl[i])):
                continue
            if r.true_mass[i] > 0 and np.isfinite(r.mass_rec_wl[i]) and np.isfinite(r.mass_rec_sl[i]):
                mt.append(r.true_mass[i])
                mw.append(r.mass_rec_wl[i])
                ms.append(r.mass_rec_sl[i])
            if np.isfinite(r.conc_rec_wl[i]) and np.isfinite(r.conc_rec_sl[i]):
                ct.append(r.true_concentration[i])
                cw.append(r.conc_rec_wl[i])
                cs.append(r.conc_rec_sl[i])

    mt = np.array(mt); mw = np.array(mw); ms = np.array(ms)
    ct = np.array(ct); cw = np.array(cw); cs = np.array(cs)
    n_sl = len(mt)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=150, constrained_layout=True)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 1: |log10(M_rec/M_true)| histogram
    # ══════════════════════════════════════════════════════════════════════
    ax = axes[0]
    if n_sl > 0:
        le_wl = np.abs(np.log10(np.maximum(mw / mt, 1e-10)))
        le_sl = np.abs(np.log10(np.maximum(ms / mt, 1e-10)))
        e_max = max(le_wl.max(), le_sl.max())
        bins = np.linspace(0, e_max * 1.05, 16)
        ax.hist(le_wl, bins=bins, alpha=0.55, color="C0",
                edgecolor="k", linewidth=0.5,
                label=f"WL  (med={np.median(le_wl):.3f})")
        ax.hist(le_sl, bins=bins, alpha=0.55, color="C2",
                edgecolor="k", linewidth=0.5,
                label=f"WL+SL  (med={np.median(le_sl):.3f})")
        ax.axvline(np.median(le_wl), color="C0", ls="--", lw=1.2, alpha=0.8)
        ax.axvline(np.median(le_sl), color="C2", ls="--", lw=1.2, alpha=0.8)
    ax.set_xlabel(r"$|\log_{10}(M_{\rm rec}/M_{\rm true})|$")
    ax.set_ylabel("Count")
    ax.set_title(f"Mass error distribution (SL subset, N={n_sl})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 2: Mass recovery — paired, colour by improvement
    # ══════════════════════════════════════════════════════════════════════
    ax = axes[1]
    if n_sl > 0:
        improved = np.abs(np.log10(ms/mt)) < np.abs(np.log10(mw/mt))
        for i in range(n_sl):
            colour = "C2" if improved[i] else "C3"
            ax.plot([mt[i], mt[i]], [mw[i], ms[i]],
                    "-", color=colour, alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(mt, mw, s=40, c="C0", alpha=0.85,
                   edgecolors="k", linewidths=0.5, zorder=4,
                   label=f"WL  (med M/M$_t$={np.median(mw/mt):.2f})")
        ax.scatter(mt, ms, s=40, c="C2", alpha=0.85, marker="D",
                   edgecolors="k", linewidths=0.5, zorder=5,
                   label=f"WL+SL  (med M/M$_t$={np.median(ms/mt):.2f})")
        all_v = np.concatenate([mt, mw, ms])
        lo, hi = all_v.min() * 0.7, all_v.max() * 1.4
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$M_{\rm true}$  ($M_\odot$)")
    ax.set_ylabel(r"$M_{\rm rec}$  ($M_\odot$)")
    n_imp_m = int(improved.sum()) if n_sl > 0 else 0
    ax.set_title(f"Mass recovery ({n_imp_m}/{n_sl} improved by SL)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ══════════════════════════════════════════════════════════════════════
    # Panel 3: Concentration recovery — paired
    # ══════════════════════════════════════════════════════════════════════
    ax = axes[2]
    n_c = len(ct)
    if n_c > 0:
        c_improved = np.abs(cs - ct) < np.abs(cw - ct)
        for i in range(n_c):
            colour = "C2" if c_improved[i] else "C3"
            ax.plot([ct[i], ct[i]], [cw[i], cs[i]],
                    "-", color=colour, alpha=0.4, lw=0.8, zorder=2)
        ax.scatter(ct, cw, s=40, c="C0", alpha=0.85,
                   edgecolors="k", linewidths=0.5, zorder=4,
                   label=f"WL  (med={np.median(cw):.2f})")
        ax.scatter(ct, cs, s=40, c="C2", alpha=0.85, marker="D",
                   edgecolors="k", linewidths=0.5, zorder=5,
                   label=f"WL+SL  (med={np.median(cs):.2f})")
        all_c = np.concatenate([ct, cw, cs])
        lo, hi = max(all_c.min() - 0.5, 0), all_c.max() + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        # Show the locked Duffy value as horizontal reference
        ax.axhline(np.median(cw), color="C0", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$c_{\rm true}$")
    ax.set_ylabel(r"$c_{\rm rec}$")
    n_imp_c = int(c_improved.sum()) if n_c > 0 else 0
    ax.set_title(f"Concentration recovery ({n_imp_c}/{n_c} improved by SL)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"NFW Monte Carlo: {len(results)} realizations "
        f"({n_sl} with SL, {n_no} without)",
        fontsize=13,
    )

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "mc":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        w = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_monte_carlo(n_realizations=n, n_workers=w, verbose=True)
    else:
        run_two_halo_comparison()