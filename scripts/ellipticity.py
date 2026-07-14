#!/usr/bin/env python3
"""
ellipticity_analysis.py — Convergence ellipticity of ARCH cluster
reconstructions, with literature-comparison helpers.

Run as
    python -m scripts.ellipticity_analysis <lens_csv> --lens-type POWER_LAW
or import the analyse_kappa_ellipticity() function directly.

Physics
-------
Standard "distortion" ellipticity from the convergence quadrupole:

    M_xx = int (x - x_c)^2 kappa  dA
    M_yy = int (y - y_c)^2 kappa  dA
    M_xy = int (x - x_c)(y - y_c) kappa  dA

    e_1 = (M_xx - M_yy) / (M_xx + M_yy)
    e_2 = 2 M_xy        / (M_xx + M_yy)
    |e| = sqrt(e_1^2 + e_2^2)
    q   = b/a = sqrt((1 - |e|) / (1 + |e|))
    PA  = (1/2) atan2(e_2, e_1)         # major axis from +x, CCW

ARCH-specific note
------------------
ARCH halos are azimuthally symmetric (circular) by construction.  The
ellipticity of the *total* convergence therefore measures the spatial
arrangement of the multiple recovered halos — it is not the intrinsic
ellipticity of any single halo.  This is methodologically distinct
from LENSTOOL-style fits (e.g., Bergamini+2023, Mahler+2018, Jauzac+2015)
which use elliptical PIEMDs and report each halo's intrinsic shape.
A2744 ellipticity comparisons should bear this in mind:  ARCH and
LENSTOOL agree on the cluster-scale κ ellipticity but parameterize the
underlying structure differently.

Conventions
-----------
- Positions in arcsec, original (post-centroid-restoration) frame —
  same frame as the CSV output and the kappa contour PDFs.
- PA_image: angle of major axis from +x (RA-offset axis), measured CCW,
  in degrees, wrapped to [0, 180).
- PA_astro: angle from +y (Dec-offset, North) toward +x (East), measured
  CCW, in degrees, wrapped to [0, 180).
  ⇒ PA_astro = (90 - PA_image) mod 180.
  If your image is sky-mirrored (compass East arrow points LEFT while
  RA-offset increases RIGHT), the astronomical PA is (180 - PA_astro) mod 180.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import Planck18 as COSMO

import arch.halo_obj as halo_obj
import arch.utils as utils


# =============================================================================
# Core moment / ellipticity routines
# =============================================================================

def _weighted_centroid(X, Y, weights, mask=None):
    """
    Compute the centroid using arbitrary weights (e.g. κ or uniform).

    Returns (x_c, y_c, W_total).
    """
    if mask is None:
        m = np.isfinite(weights) & (weights > 0)
    else:
        m = mask & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return float("nan"), float("nan"), 0.0
    w = weights[m]
    W = float(w.sum())
    if W <= 0:
        return float("nan"), float("nan"), 0.0
    return float((X[m] * w).sum() / W), float((Y[m] * w).sum() / W), W


def _weighted_second_moments(X, Y, weights, x_c, y_c, mask=None):
    """Second moments about (x_c, y_c) using arbitrary weights."""
    if mask is None:
        m = np.isfinite(weights) & (weights > 0)
    else:
        m = mask & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return 0.0, 0.0, 0.0, 0.0
    w = weights[m]
    dx = X[m] - x_c
    dy = Y[m] - y_c
    M_xx = float((dx * dx * w).sum())
    M_yy = float((dy * dy * w).sum())
    M_xy = float((dx * dy * w).sum())
    W = float(w.sum())
    return M_xx, M_yy, M_xy, W


def _kappa_centroid(X, Y, kappa, mask=None):
    """κ-weighted centroid (legacy API, retained for callers).  Returns (x_c, y_c, M0)."""
    return _weighted_centroid(X, Y, kappa, mask=mask)


def _kappa_second_moments(X, Y, kappa, x_c, y_c, mask=None):
    """Second moments about (x_c, y_c), κ-weighted (legacy API)."""
    return _weighted_second_moments(X, Y, kappa, x_c, y_c, mask=mask)


def _moments_to_ellipticity(M_xx, M_yy, M_xy):
    """
    Convert second moments to (e1, e2, |e|, q, PA_image_deg, PA_astro_deg).

    Returns a dict.
    """
    denom = M_xx + M_yy
    if denom <= 0:
        return {
            "e1": float("nan"), "e2": float("nan"), "e_mod": float("nan"),
            "q": float("nan"),
            "PA_image_deg": float("nan"), "PA_astro_deg": float("nan"),
        }
    e1 = (M_xx - M_yy) / denom
    e2 = 2.0 * M_xy / denom
    e_mod = float(np.hypot(e1, e2))
    # |e| should be <= 1 numerically; clamp for the q formula.
    e_for_q = min(e_mod, 0.99999)
    q = float(np.sqrt(max((1.0 - e_for_q) / (1.0 + e_for_q), 0.0)))
    PA_im_rad = 0.5 * np.arctan2(e2, e1)
    PA_image_deg = float(np.degrees(PA_im_rad) % 180.0)
    PA_astro_deg = float((90.0 - PA_image_deg) % 180.0)
    return {
        "e1": float(e1), "e2": float(e2), "e_mod": e_mod, "q": q,
        "PA_image_deg": PA_image_deg, "PA_astro_deg": PA_astro_deg,
    }


# =============================================================================
# High-level analysis
# =============================================================================

def analyse_kappa_ellipticity(
    lenses,
    z_source: float,
    lens_type: str = "POWER_LAW",
    extent: tuple = None,
    grid_res: int = 512,
    grid_center: tuple = None,
    grid_half_size_arcsec: float = None,
    aperture_centers: list = None,
    aperture_radii_arcsec: tuple = (np.inf,),
    inner_mask_arcsec: float = 5.0,
    kappa_floor_quantile: float = 0.0,
):
    """
    Build the κ map and compute ellipticity in one or more apertures.

    Parameters
    ----------
    lenses : NFW_Lens or PowerLawHalo
        Recovered halos.
    z_source : float
        Source redshift for κ.
    lens_type : str
        'NFW' or 'POWER_LAW'.
    extent : (xmin, xmax, ymin, ymax) or None
        Explicit grid extent.  If given, overrides grid_center /
        grid_half_size_arcsec.
    grid_res : int
        Grid resolution per side.
    grid_center : (x, y) or None
        Center of the square kappa grid.  If None, uses the midpoint of
        the halo position bounding box.
    grid_half_size_arcsec : float or None
        Half-size of the square grid.  If None, sized to contain the
        largest finite aperture in aperture_radii_arcsec plus a 30″
        margin (or to comfortably wrap the halo positions if no finite
        apertures are requested).  This ensures aperture limits actually
        clip inside the grid.
    aperture_centers : list of (x, y) or None
        Aperture centers.  If None, uses the κ-weighted centroid of the
        inner-masked grid.
    aperture_radii_arcsec : sequence
        Aperture radii.  np.inf means the full grid.
    inner_mask_arcsec : float
        Mask cells within this radius of each halo center.
    kappa_floor_quantile : float in [0, 1)
        If > 0, subtract the κ value at this quantile from the map before
        computing moments.  Useful when a shallow halo profile (e.g.
        POWER_LAW with n near 0.4) produces a near-uniform κ background
        that biases moments toward the grid shape rather than the cluster
        structure.  Try 0.10–0.25.  0 = no subtraction (default).

    Returns
    -------
    dict keyed by (center_index, aperture_label).
    """
    # ----- build grid extent -----
    xs = np.atleast_1d(lenses.x); ys = np.atleast_1d(lenses.y)
    if extent is None:
        if grid_center is None:
            cx = 0.5 * (float(xs.min()) + float(xs.max()))
            cy = 0.5 * (float(ys.min()) + float(ys.max()))
        else:
            cx, cy = float(grid_center[0]), float(grid_center[1])
        if grid_half_size_arcsec is None:
            # Half-size large enough to comfortably contain (a) the halo
            # bounding box, and (b) the largest finite aperture.
            finite_radii = [r for r in aperture_radii_arcsec if np.isfinite(r)]
            r_max = max(finite_radii) if finite_radii else 0.0
            halo_reach = max(
                abs(float(xs.max()) - cx), abs(float(xs.min()) - cx),
                abs(float(ys.max()) - cy), abs(float(ys.min()) - cy),
            )
            half = max(r_max, halo_reach) + 30.0
        else:
            half = float(grid_half_size_arcsec)
        extent = (cx - half, cx + half, cy - half, cy + half)

    # ----- build kappa grid -----
    X, Y, kappa = utils.calculate_kappa(
        lenses, extent=extent, lens_type=lens_type,
        source_redshift=z_source,
    )

    # ----- optional κ-floor subtraction (suppress sheet contamination) -----
    sheet_subtracted = False
    if kappa_floor_quantile > 0.0:
        kfin = kappa[np.isfinite(kappa)]
        if kfin.size > 100:
            floor = float(np.quantile(kfin, kappa_floor_quantile))
            kappa = np.clip(kappa - floor, 0.0, None)
            sheet_subtracted = True

    # ----- inner mask around each halo center -----
    if inner_mask_arcsec > 0:
        inner_mask = np.ones_like(X, dtype=bool)
        for k in range(len(xs)):
            R = np.hypot(X - lenses.x[k], Y - lenses.y[k])
            inner_mask &= (R >= inner_mask_arcsec)
    else:
        inner_mask = np.ones_like(X, dtype=bool)

    # ----- aperture centers -----
    if aperture_centers is None:
        cx0, cy0, _ = _kappa_centroid(X, Y, kappa, mask=inner_mask)
        aperture_centers = [(cx0, cy0)]

    results = {}
    for ic, (cx, cy) in enumerate(aperture_centers):
        for radius in aperture_radii_arcsec:
            if not np.isfinite(radius):
                ap_mask = inner_mask
                label = "full"
            else:
                R = np.hypot(X - cx, Y - cy)
                ap_mask = inner_mask & (R <= radius)
                label = f"r<{radius:.1f}\""
            x_c, y_c, _ = _kappa_centroid(X, Y, kappa, mask=ap_mask)
            M_xx, M_yy, M_xy, M0 = _kappa_second_moments(
                X, Y, kappa, x_c, y_c, mask=ap_mask
            )
            stats = _moments_to_ellipticity(M_xx, M_yy, M_xy)
            stats.update({
                "aperture_center_input": (cx, cy),
                "kappa_centroid": (x_c, y_c),
                "aperture_radius_arcsec": float(radius),
                "aperture_radius_kpc": (
                    float(radius)
                    * COSMO.kpc_proper_per_arcmin(lenses.redshift)
                            .to(u.kpc / u.arcsec).value
                    if np.isfinite(radius) else float("inf")
                ),
                "kappa_total": M0,
                "n_pixels": int(np.sum(ap_mask)),
                "M_xx": M_xx, "M_yy": M_yy, "M_xy": M_xy,
                "grid_extent": extent,
                "sheet_subtracted": sheet_subtracted,
                "kappa_floor_quantile": kappa_floor_quantile,
            })
            results[(ic, label)] = stats

    # ----- aperture-bite diagnostic -----
    # If all finite apertures contain the same number of pixels as the
    # full mask, then the apertures are clipping at the grid boundary
    # rather than inside the grid — moments will all be identical and
    # the comparison is degenerate.
    counts = [(label, v["n_pixels"]) for (_, label), v in results.items()]
    unique_counts = set(c for _, c in counts)
    if len(unique_counts) == 1 and len(counts) > 1:
        print("\n  ⚠  All apertures contain the same pixel count — aperture")
        print("     limits are not biting inside the grid.  Increase grid")
        print("     half-size, or pass --grid-half-size large enough to wrap")
        print("     the requested apertures with margin.")

    return results


# =============================================================================
# Pretty-printing and literature comparison
# =============================================================================

def print_ellipticity_table(results):
    print(f"\n  {'Aperture':18s} {'q (b/a)':>8s} {'|e|':>7s} "
          f"{'e1':>8s} {'e2':>8s} {'PA_im':>8s} {'PA_astro':>9s} "
          f"{'centroid (″)':>18s}")
    print("  " + "-" * 90)
    for (ic, label), s in results.items():
        cx, cy = s["kappa_centroid"]
        rad_kpc = s["aperture_radius_kpc"]
        if np.isfinite(rad_kpc):
            ap_str = f"{label} ({rad_kpc:.0f} kpc)"
        else:
            ap_str = label
        print(
            f"  {ap_str[:18]:18s} "
            f"{s['q']:>8.3f} {s['e_mod']:>7.3f} "
            f"{s['e1']:>+8.3f} {s['e2']:>+8.3f} "
            f"{s['PA_image_deg']:>7.1f}° {s['PA_astro_deg']:>8.1f}° "
            f" ({cx:>+6.1f}, {cy:>+6.1f})"
        )


def compare_to_literature(arch_results, literature_table):
    """
    Compare ARCH ellipticity to a literature table.

    literature_table : list of dict, each with keys
        'reference', 'aperture_kpc', 'q', 'PA_astro_deg',
        optionally 'q_err', 'PA_err', 'comparison_type', 'notes'.
    """
    if not literature_table:
        print("\n  (No literature values supplied — populate A2744_LITERATURE "
              "in this script to enable comparison.)")
        return

    # Filter out entries with missing q or PA (placeholder rows)
    populated = [e for e in literature_table
                 if e.get("q") is not None and e.get("PA_astro_deg") is not None]
    if not populated:
        print("\n  (Literature table has no populated entries.)")
        return

    # Flag the comparison type up front
    types = {e.get("comparison_type", "unknown") for e in populated}
    if "PIEMD_intrinsic" in types:
        print("\n  ⚠  Literature values below are LENSTOOL PIEMD/dPIE INTRINSIC")
        print("     per-halo ellipticities — NOT quadrupole moments of κ.")
        print("     Expect ~0.05 agreement in q and ~10° in PA at best, even")
        print("     when ARCH and LENSTOOL fit the same data.  For paper-")
        print("     quality comparisons, re-run analyse_kappa_ellipticity()")
        print("     on the published LENSTOOL κ maps.")

    print(f"\n  Literature comparison (PA wrapped to [0, 180)):")
    print(f"  {'Reference':30s} {'Aper':>8s} "
          f"{'q_lit':>10s} {'PA_lit':>10s} "
          f"{'q_ARCH':>8s} {'PA_ARCH':>9s} "
          f"{'Δq':>7s} {'ΔPA':>8s}")
    print("  " + "-" * 100)
    for entry in populated:
        lit_ap = entry["aperture_kpc"]
        best = None; best_diff = np.inf
        for _, val in arch_results.items():
            ap = val["aperture_radius_kpc"]
            if np.isfinite(ap) and abs(ap - lit_ap) < best_diff:
                best_diff = abs(ap - lit_ap)
                best = val
        if best is None:
            continue
        arch_q = best["q"]; arch_pa = best["PA_astro_deg"]
        dq = arch_q - entry["q"]
        dpa = arch_pa - entry["PA_astro_deg"]
        # Wrap ΔPA to (-90, 90]
        while dpa > 90:  dpa -= 180
        while dpa <= -90: dpa += 180
        q_err = entry.get("q_err")
        pa_err = entry.get("PA_err")
        q_lit_str = (f"{entry['q']:.3f}±{q_err:.2f}" if q_err is not None
                     else f"{entry['q']:>8.3f}")
        pa_lit_str = (f"{entry['PA_astro_deg']:5.1f}±{pa_err:.1f}°"
                      if pa_err is not None
                      else f"{entry['PA_astro_deg']:>7.1f}°")
        print(
            f"  {entry['reference'][:30]:30s} "
            f"{lit_ap:>4.0f} kpc "
            f"{q_lit_str:>10s} {pa_lit_str:>10s} "
            f"{arch_q:>8.3f} {arch_pa:>8.1f}° "
            f"{dq:>+7.3f} {dpa:>+7.1f}°"
        )
        if entry.get("notes"):
            print(f"      ↳ {entry['notes']}")


# =============================================================================
# Region matching for literature comparison
# =============================================================================

def _match_region_to_arch_halo(lenses, region):
    """
    Return (x, y) of the ARCH halo associated with a literature region tag.

    Conventions:
        BCG-N, North, N : ARCH halo with largest y (most northern)
        BCG-S, South, S : ARCH halo with smallest y (most southern)
        cluster, core   : None (caller substitutes the κ-weighted centroid)
        NW, North-West  : not in current ARCH core reconstruction → None

    Returns (x, y) or None if no match.
    """
    xs = np.atleast_1d(lenses.x)
    ys = np.atleast_1d(lenses.y)
    n = len(xs)
    if n == 0 or not region:
        return None
    r = region.lower().replace("_", "-").strip()
    if r in ("bcg-n", "north", "n"):
        idx = int(np.argmax(ys))
        return float(xs[idx]), float(ys[idx])
    if r in ("bcg-s", "south", "s"):
        idx = int(np.argmin(ys))
        return float(xs[idx]), float(ys[idx])
    # cluster / core / centroid / unknown → caller decides
    return None


# =============================================================================
# Visualization: literature-overlay plot
# =============================================================================


def plot_ellipticity_with_literature(
    lenses, z_source, arch_results, literature_table,
    lens_type="POWER_LAW", out_path=None,
    zoom_arcsec=None, sky_mirrored=False,
    ellipse_size_arcsec=22.0,
    arch_ellipse_aperture_label=None,
):
    """
    Comparison plot: ARCH's κ contour and ellipse(s) overlaid with literature
    ellipses drawn at each literature halo's associated ARCH halo position.

    Each literature entry's `region` field controls placement:
      - "BCG-N"  : most-northern ARCH halo
      - "BCG-S"  : most-southern ARCH halo
      - other / missing : κ-weighted centroid

    Parameters
    ----------
    arch_results : dict
        Output of analyse_kappa_ellipticity().  ARCH ellipses for each
        aperture are drawn in muted gray so the literature comparison
        reads cleanly.
    literature_table : list of dict
        Populated A2744_LITERATURE entries.
    ellipse_size_arcsec : float
        Visual semi-major axis used for ALL ellipses (ARCH and literature)
        so the comparison is purely about shape, not aperture size.
    arch_ellipse_aperture_label : str or None
        Restrict ARCH ellipses drawn to a single aperture label (e.g.,
        "r<53.4\"" or "full").  None → draw all apertures.
    """
    from matplotlib.patches import Ellipse
    from matplotlib.lines import Line2D

    xs = np.atleast_1d(lenses.x); ys = np.atleast_1d(lenses.y)

    # ---- grid extent ----
    try:
        extent = next(iter(arch_results.values()))["grid_extent"]
    except (KeyError, StopIteration):
        cx_h = 0.5 * (float(xs.min()) + float(xs.max()))
        cy_h = 0.5 * (float(ys.min()) + float(ys.max()))
        half = max(
            float(xs.max()) - cx_h, cx_h - float(xs.min()),
            float(ys.max()) - cy_h, cy_h - float(ys.min()),
        ) + 30.0
        extent = (cx_h - half, cx_h + half, cy_h - half, cy_h + half)

    X, Y, kappa = utils.calculate_kappa(
        lenses, extent=extent, lens_type=lens_type,
        source_redshift=z_source,
    )

    # ---- centroid + zoom ----
    first_stats = next(iter(arch_results.values()))
    cx_cen, cy_cen = first_stats["kappa_centroid"]
    if zoom_arcsec is None:
        zoom_arcsec = max(
            float(xs.max()) - cx_cen, cx_cen - float(xs.min()),
            float(ys.max()) - cy_cen, cy_cen - float(ys.min()),
        ) + 20.0

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")

    # κ contour as quiet background
    kfin = kappa[np.isfinite(kappa) & (kappa > 0)]
    if kfin.size > 50:
        levels = np.quantile(kfin, [0.50, 0.70, 0.85, 0.94, 0.98])
        levels = sorted({round(float(lv), 4) for lv in levels})
    else:
        levels = [0.05, 0.10, 0.20, 0.50, 1.00]
    ax.contour(X, Y, kappa, levels=levels, colors="#dddddd",
               linewidths=0.9, zorder=1)

    # ARCH halos
    ax.plot(xs, ys, "k*", markersize=14, zorder=10,
            markeredgecolor="white", markeredgewidth=0.8)
    for hi, (hx, hy) in enumerate(zip(xs, ys)):
        ax.annotate(
            f" H{hi}", (hx, hy), fontsize=9, fontweight="bold",
            color="black", zorder=11,
        )

    # ARCH ellipses (all apertures or one)
    arch_color = "black"
    arch_entries_drawn = []
    for (ic, label), s in arch_results.items():
        if arch_ellipse_aperture_label is not None and label != arch_ellipse_aperture_label:
            continue
        cx, cy = s["kappa_centroid"]
        a = ellipse_size_arcsec
        b = s["q"] * a
        ax.add_patch(Ellipse(
            (cx, cy), width=2 * a, height=2 * b,
            angle=s["PA_image_deg"],
            edgecolor=arch_color, facecolor="none", linewidth=2.4,
            linestyle="-", zorder=5,
        ))
        ax.plot([cx], [cy], "x", color=arch_color, markersize=9, mew=2,
                zorder=6)
        arch_entries_drawn.append((label, s))

    # ---- literature ellipses, grouped by paper for color ----
    paper_colors = {
        "Bergamini": "#d62728",  # red
        "Furtak":    "#1f77b4",  # blue
        "Mahler":    "#2ca02c",  # green
        "Jauzac":    "#9467bd",  # purple
        "Merten":    "#ff7f0e",  # orange
    }

    def _paper_key(reference):
        for k in paper_colors:
            if k in reference:
                return k
        return "Other"

    populated = [e for e in literature_table
                 if e.get("q") is not None and e.get("PA_astro_deg") is not None]

    if not populated:
        print("  No populated literature entries — falling back to ARCH-only plot.")
    else:
        for entry in populated:
            region = entry.get("region")
            pos = _match_region_to_arch_halo(lenses, region)
            if pos is None:
                pos = (cx_cen, cy_cen)
            ex, ey = pos
            color = paper_colors.get(_paper_key(entry["reference"]), "#555555")

            q = entry["q"]
            pa_astro = entry["PA_astro_deg"]
            # Convert PA_astro back to PA_image (CCW from +x) for Ellipse:
            #   PA_astro = (90 - PA_image) mod 180  →  PA_image = (90 - PA_astro) mod 180
            pa_image = (90.0 - pa_astro) % 180.0

            a = ellipse_size_arcsec
            b = q * a
            ax.add_patch(Ellipse(
                (ex, ey), width=2 * a, height=2 * b,
                angle=pa_image,
                edgecolor=color, facecolor="none", linewidth=1.6,
                linestyle="--", alpha=0.95, zorder=4,
            ))

    # ---- legend (grouped: ARCH first, then literature by paper) ----
    legend_elements = []
    for label, s in arch_entries_drawn:
        legend_elements.append(Line2D(
            [0], [0], color=arch_color, lw=2.4, linestyle="-",
            label=f"ARCH {label}: q={s['q']:.2f}, PA={s['PA_astro_deg']:.0f}°",
        ))
    # one legend entry per literature paper, with the paper name + entries
    by_paper = {}
    for entry in populated:
        key = _paper_key(entry["reference"])
        by_paper.setdefault(key, []).append(entry)
    for paper_name, entries in by_paper.items():
        color = paper_colors.get(paper_name, "#555555")
        for entry in entries:
            tag = entry["reference"].replace(f"{paper_name}+23 ", "") \
                                    .replace(f"{paper_name}+18 ", "") \
                                    .replace(f"{paper_name}+ ", "")
            legend_elements.append(Line2D(
                [0], [0], color=color, lw=1.6, linestyle="--",
                label=f"{entry['reference']}: q={entry['q']:.2f}, "
                      f"PA={entry['PA_astro_deg']:.0f}°",
            ))
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left",
              framealpha=0.95)

    ax.set_xlim(cx_cen - zoom_arcsec, cx_cen + zoom_arcsec)
    ax.set_ylim(cy_cen - zoom_arcsec, cy_cen + zoom_arcsec)
    ax.set_xlabel("RA offset (arcsec)")
    ax.set_ylabel("Dec offset (arcsec)")

    title_bits = ["κ ellipticity — ARCH vs literature"]
    if first_stats.get("sheet_subtracted"):
        title_bits.append(
            f"(ARCH sheet subtracted q={first_stats['kappa_floor_quantile']:.2f})"
        )
    ax.set_title(" ".join(title_bits), fontsize=10)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved comparison plot to {out_path}")

    return fig, ax


# =============================================================================
# Single-aperture overlay plot
# =============================================================================

def plot_ellipticity_overlay(
    lenses, z_source, results, lens_type="POWER_LAW",
    extent=None, out_path=None, sky_mirrored=False,
    zoom_arcsec=None,
):
    """
    Render the κ contours with the major axis of each aperture's ellipse
    overlaid as an ELLIPSE through the κ-weighted centroid.

    Improvements over the previous version:
      - Square figure with matching aspect via ax.set_box_aspect(1).
      - Ellipses (not lines) so q and PA are simultaneously visible.
      - Aperture circles drawn as thin outlines.
      - Zoom to the contour-rich region by default rather than the full grid.

    Parameters
    ----------
    zoom_arcsec : float or None
        Half-side of the displayed view, centered on the κ-weighted
        centroid.  If None, derived from the smallest finite aperture
        in results plus a 20″ margin (else from halo bounding box).
    """
    from matplotlib.patches import Ellipse, Circle

    xs = np.atleast_1d(lenses.x); ys = np.atleast_1d(lenses.y)

    # ---- decide on the κ grid extent ----
    if extent is None:
        # Use the same grid used in the analysis if present, otherwise
        # fall back to a square box centered on the halo midpoint.
        try:
            extent = next(iter(results.values()))["grid_extent"]
        except (KeyError, StopIteration):
            cx_h = 0.5 * (float(xs.min()) + float(xs.max()))
            cy_h = 0.5 * (float(ys.min()) + float(ys.max()))
            half = max(
                float(xs.max()) - cx_h, cx_h - float(xs.min()),
                float(ys.max()) - cy_h, cy_h - float(ys.min()),
            ) + 30.0
            extent = (cx_h - half, cx_h + half, cy_h - half, cy_h + half)

    X, Y, kappa = utils.calculate_kappa(
        lenses, extent=extent, lens_type=lens_type,
        source_redshift=z_source,
    )

    # ---- centroid + zoom window ----
    first_stats = next(iter(results.values()))
    cx_view, cy_view = first_stats["kappa_centroid"]

    if zoom_arcsec is None:
        finite_radii = [s["aperture_radius_arcsec"] for s in results.values()
                        if np.isfinite(s["aperture_radius_arcsec"])]
        if finite_radii:
            zoom_arcsec = min(finite_radii) + 20.0
        else:
            zoom_arcsec = max(
                float(xs.max()) - cx_view, cx_view - float(xs.min()),
                float(ys.max()) - cy_view, cy_view - float(ys.min()),
            ) + 20.0

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_box_aspect(1)
    ax.set_aspect("equal", adjustable="box")

    # κ contours at sensible quantile-based levels
    kfin = kappa[np.isfinite(kappa) & (kappa > 0)]
    if kfin.size > 50:
        levels = np.quantile(kfin, [0.50, 0.70, 0.85, 0.94, 0.98])
        levels = sorted({round(float(lv), 4) for lv in levels})
    else:
        levels = [0.05, 0.10, 0.20, 0.50, 1.00]
    ax.contour(X, Y, kappa, levels=levels, colors="C1", linewidths=0.9)

    # ARCH halos
    ax.plot(xs, ys, "k*", markersize=12, label="ARCH halos",
            markeredgecolor="white", markeredgewidth=0.8)

    # Per-aperture overlay: aperture circle + ellipse showing q, PA
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, ((ic, label), s) in enumerate(results.items()):
        c = colors[i % len(colors)]
        cx, cy = s["kappa_centroid"]
        R_ap = s["aperture_radius_arcsec"]

        # Aperture circle outline (only if finite and visible)
        if np.isfinite(R_ap) and R_ap < 3 * zoom_arcsec:
            ax.add_patch(Circle(
                (cx, cy), R_ap,
                edgecolor=c, facecolor="none", linestyle=":",
                linewidth=0.9, alpha=0.7,
            ))

        # Ellipse: pick a visually sensible semi-major a, then b = q*a.
        # Use a fraction of the aperture (or of the zoom) so the ellipse
        # sits inside the view.
        if np.isfinite(R_ap):
            a = 0.6 * R_ap
        else:
            a = 0.4 * zoom_arcsec
        b = s["q"] * a
        # matplotlib Ellipse 'angle' is degrees CCW from +x axis = PA_image
        ax.add_patch(Ellipse(
            (cx, cy), width=2 * a, height=2 * b,
            angle=s["PA_image_deg"],
            edgecolor=c, facecolor="none", linewidth=2.0,
            label=(f"{label}: q={s['q']:.2f}, "
                   f"PA={s['PA_astro_deg']:.0f}°"),
        ))
        ax.plot([cx], [cy], "x", color=c, markersize=8, mew=2)

    ax.set_xlim(cx_view - zoom_arcsec, cx_view + zoom_arcsec)
    ax.set_ylim(cy_view - zoom_arcsec, cy_view + zoom_arcsec)
    ax.set_xlabel("RA offset (arcsec)")
    ax.set_ylabel("Dec offset (arcsec)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    title_bits = ["κ ellipticity"]
    if first_stats.get("sheet_subtracted"):
        title_bits.append(
            f"(sheet subtracted, q={first_stats['kappa_floor_quantile']:.2f})"
        )
    if sky_mirrored:
        title_bits.append("(sky-mirrored: PA→180−PA)")
    ax.set_title(" ".join(title_bits), fontsize=10)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved overlay plot to {out_path}")
    return fig, ax


# =============================================================================
# Literature values for A2744
# =============================================================================
#
# IMPORTANT CAVEAT — read before using these values in a paper.
#
# All three references below (Mahler+2018, Bergamini+2023, Furtak+2023) use
# LENSTOOL with elliptical PIEMD/dPIE building blocks, and report INTRINSIC
# per-halo ellipticities — the ε and PA that the parametric fit assigned to
# each cluster-scale halo, NOT a quadrupole moment of the total convergence.
# ARCH measures the latter from a sum of CIRCULAR halos.  These are different
# geometric statements:
#
#   - When a single PIEMD halo dominates and is centered in the aperture,
#     its intrinsic ε agrees with the kappa-quadrupole ε to within numerical
#     factors set by the profile shape.  Use the dominant LENSTOOL halo's ε
#     and PA as a qualitative reference.
#
#   - For two-component cluster cores like A2744 (BCG-N + BCG-S regions),
#     ARCH's quadrupole-of-total-kappa will lie somewhere between the two
#     LENSTOOL halos' intrinsic ε and PA, weighted by their kappa
#     contributions inside the aperture.  Numerical agreement at the
#     ~0.05 level in q (or ~10° in PA) is the right expectation, not
#     exact match.
#
# A genuinely apples-to-apples comparison requires running
# analyse_kappa_ellipticity() on the published LENSTOOL kappa maps
# (HFF MAST archive for Mahler+2018, Bergamini's SLOT tool for B23).
# That is the right thing to do for paper-quality claims.
#
# Convention conversions applied below (PA wrapped to [0, 180)):
#   ε definition:        all three papers use (a^2 - b^2) / (a^2 + b^2)
#                         (same as ARCH).  Furtak's table caption shows a
#                         typo ((b-a)/(a+b)) — values match quadratic.
#   q = b/a:             sqrt((1 - ε) / (1 + ε))
#   PA_astro_deg:        (90° - θ_paper) mod 180°
#                         (works for both "CCW from East" — Mahler, Furtak —
#                          and "CCW from West" — Bergamini — since PA is
#                          defined mod 180° and E ↔ W differ by 180°)

A2744_LITERATURE = [
    # ------------------------------------------------------------------
    # Bergamini+2023 (A&A 670, A60; arXiv:2207.09416), Table 1
    #   LM-model output parameters, "Cluster-scale halos"
    #   ε = (a^2-b^2)/(a^2+b^2); θ counterclockwise from west
    # ------------------------------------------------------------------
    {
        "reference": "Bergamini+23 Halo-1 (BCG-N)",
        "comparison_type": "PIEMD_intrinsic",
        "region": "BCG-N",
        "aperture_kpc": 250.0,       # rough match to SL field; PIEMD is scale-free
        "q": 0.500,                  # ε = 0.6 ± 0.1
        "q_err": 0.05,
        "PA_astro_deg": 179.7,       # θ = 90.3° CCW from W → ~N-S elongation
        "PA_err": 2.6,
        "notes": "Cluster-scale halo near BCG-N; pos = (-1.5, -0.1)\". "
                 "σ_LT = 522.7 km/s, r_core = 6.8\" (~31 kpc).",
    },
    {
        "reference": "Bergamini+23 Halo-2 (BCG-S)",
        "comparison_type": "PIEMD_intrinsic",
        "region": "BCG-S",
        "aperture_kpc": 250.0,
        "q": 0.655,                  # ε = 0.4 ± 0.1
        "q_err": 0.07,
        "PA_astro_deg": 36.7,        # θ = 53.3° CCW from W → NE
        "PA_err": 2.7,
        "notes": "Cluster-scale halo near BCG-S; pos = (-18.2, -15.7)\". "
                 "σ_LT = 633.9 km/s, r_core = 7.6\" (~34 kpc).",
    },

    # ------------------------------------------------------------------
    # Furtak+2023 (MNRAS 523, 4568; arXiv:2212.04381), Table 3
    #   UNCOVER main cluster core: Main-1, Main-2
    #   ε definition: caption shows (b-a)/(a+b) but values are consistent
    #     with (a^2-b^2)/(a^2+b^2) — using quadratic interpretation.
    #   θ counterclockwise from east-west axis (= CCW from East).
    # ------------------------------------------------------------------
    {
        "reference": "Furtak+23 Main-1",
        "comparison_type": "PIEMD_intrinsic",
        "region": "BCG-S",            # Dec = -30:24:18.32 (more south)
        "aperture_kpc": 250.0,
        "q": 0.647,                  # ε = 0.41 +0.02/-0.03
        "q_err": 0.02,
        "PA_astro_deg": 45.0,        # θ = 45° CCW from E → NE
        "PA_err": 4.0,
        "notes": "Main cluster halo; σ = 681 km/s, r_core = 23 kpc.",
    },
    {
        "reference": "Furtak+23 Main-2",
        "comparison_type": "PIEMD_intrinsic",
        "region": "BCG-N",            # Dec = -30:24:00.85 (less negative = more north)
        "aperture_kpc": 250.0,
        "q": 0.616,                  # ε = 0.45 +0.01/-0.02
        "q_err": 0.015,
        "PA_astro_deg": 17.0,        # θ = 73° CCW from E → mostly N
        "PA_err": 3.0,
        "notes": "Main cluster halo; σ = 807 km/s, r_core = 67 kpc.",
    },

    # ------------------------------------------------------------------
    # Furtak+2023 external clumps (NW and N) — outside ARCH's core SL
    # field but included for completeness.  Likely NOT comparable to
    # ARCH's current A2744 reconstruction which focuses on the main core.
    # Uncomment if running on a wide-field reconstruction.
    # ------------------------------------------------------------------
    # {
    #     "reference": "Furtak+23 NW-1",
    #     "comparison_type": "PIEMD_intrinsic",
    #     "aperture_kpc": 500.0,
    #     "q": 0.539,                # ε = 0.55
    #     "PA_astro_deg": 71.0,      # θ = 19° CCW from E
    #     "notes": "NW external clump.  σ = 407 km/s, r_core = 5 kpc.",
    # },
    # {
    #     "reference": "Furtak+23 NW-2",
    #     "comparison_type": "PIEMD_intrinsic",
    #     "aperture_kpc": 500.0,
    #     "q": 0.829,                # ε = 0.18
    #     "PA_astro_deg": 98.0,      # θ = -8° CCW from E
    #     "notes": "NW external clump.  σ = 906 km/s, r_core = 62 kpc.",
    # },
    # {
    #     "reference": "Furtak+23 North",
    #     "comparison_type": "PIEMD_intrinsic",
    #     "aperture_kpc": 500.0,
    #     "q": 0.905,                # ε = 0.10
    #     "PA_astro_deg": 163.0,     # θ = -73° CCW from E
    #     "notes": "Northern external clump.  σ = 486 km/s, r_core = 2 kpc.",
    # },

    # ------------------------------------------------------------------
    # Mahler+2018 (MNRAS 473, 663; arXiv:1702.06962), Table 4
    #   DM1, DM2 cluster-scale halos.
    #   ε = (a^2-b^2)/(a^2+b^2) (explicit in caption).
    #   Specific Table 4 numerical values not retrieved via search.
    #   Populate q and PA_astro from your paper copy.  PA conversion:
    #     PA_astro_deg = (90 - θ_paper) mod 180.
    # ------------------------------------------------------------------
    # {
    #     "reference": "Mahler+18 DM1",
    #     "comparison_type": "PIEMD_intrinsic",
    #     "aperture_kpc": 250.0,
    #     "q": None,                 # = sqrt((1-ε)/(1+ε))
    #     "PA_astro_deg": None,
    #     "notes": "Northern cluster-scale dPIE clump.",
    # },
    # {
    #     "reference": "Mahler+18 DM2",
    #     "comparison_type": "PIEMD_intrinsic",
    #     "aperture_kpc": 250.0,
    #     "q": None,
    #     "PA_astro_deg": None,
    #     "notes": "Southern cluster-scale dPIE clump.",
    # },
]

# Backwards-compat alias for the CLI dispatch path
A2744_LITERATURE_TEMPLATE = A2744_LITERATURE


# =============================================================================
# CSV loading
# =============================================================================

# =============================================================================
# Threshold-swept ellipticity diagnostic
# =============================================================================
# Why this exists: a single quadrupole-of-κ ellipticity for a multi-halo
# configuration with circular building blocks is dominated by each halo's
# intrinsic spread (M_h * <r^2>_h) when the profile is broad, especially
# for POWER_LAW with n near 0.4.  The OFFSET term M_h * |Δx|^2 that
# carries the inter-halo elongation is then drowned out, giving q → 1
# even when the contours visibly form a dumbbell.
#
# Sweeping a κ threshold and computing q above that threshold separates
# the regimes:
#   - low threshold  : whole field, intrinsic-dominated → q ≈ 1
#   - mid threshold  : keeps the inter-halo bridge, dumbbell → q ≪ 1
#   - high threshold : isolated peak cores, each circular → q ≈ 1
#
# Reporting q(threshold) is more scientifically honest than any single q,
# because the "true" cluster ellipticity depends on which feature of the
# κ map the reader cares about.

def ellipticity_vs_threshold(
    lenses,
    z_source: float,
    lens_type: str = "POWER_LAW",
    extent: tuple = None,
    grid_res: int = 512,
    grid_center: tuple = None,
    grid_half_size_arcsec: float = None,
    inner_mask_arcsec: float = 5.0,
    quantiles=None,
):
    """
    Compute q and PA over pixels with κ above a sequence of thresholds.

    Returns
    -------
    list of dict, one per quantile, with keys:
        quantile, threshold_kappa, n_pixels, fraction_kept,
        q, PA_astro_deg, PA_image_deg, e_mod,
        kappa_centroid (x, y), kappa_total
    """
    if quantiles is None:
        quantiles = np.concatenate([
            np.linspace(0.00, 0.50, 11),  # dense at low end
            np.linspace(0.55, 0.95, 9),   # coarser at high end
        ])

    # ---- build grid (same square-grid logic as analyse_kappa_ellipticity) ----
    xs = np.atleast_1d(lenses.x); ys = np.atleast_1d(lenses.y)
    if extent is None:
        if grid_center is None:
            cx = 0.5 * (float(xs.min()) + float(xs.max()))
            cy = 0.5 * (float(ys.min()) + float(ys.max()))
        else:
            cx, cy = grid_center
        if grid_half_size_arcsec is None:
            halo_reach = max(
                abs(float(xs.max()) - cx), abs(float(xs.min()) - cx),
                abs(float(ys.max()) - cy), abs(float(ys.min()) - cy),
            )
            half = halo_reach + 30.0
        else:
            half = float(grid_half_size_arcsec)
        extent = (cx - half, cx + half, cy - half, cy + half)

    X, Y, kappa = utils.calculate_kappa(
        lenses, extent=extent, lens_type=lens_type,
        source_redshift=z_source,
    )

    # ---- inner mask around halo centers ----
    if inner_mask_arcsec > 0:
        inner_mask = np.ones_like(X, dtype=bool)
        for k in range(len(xs)):
            R = np.hypot(X - lenses.x[k], Y - lenses.y[k])
            inner_mask &= (R >= inner_mask_arcsec)
    else:
        inner_mask = np.ones_like(X, dtype=bool)

    kfin = kappa[np.isfinite(kappa) & inner_mask]
    if kfin.size < 100:
        raise RuntimeError(
            "Too few finite κ pixels for threshold sweep."
        )
    n_total_finite = int(kfin.size)

    rows = []
    for qt in quantiles:
        threshold = float(np.quantile(kfin, qt))
        ap_mask = (
            inner_mask & np.isfinite(kappa) & (kappa >= threshold)
        )
        n_pix = int(np.sum(ap_mask))
        if n_pix < 20:
            continue
        x_c, y_c, M0 = _kappa_centroid(X, Y, kappa, mask=ap_mask)
        M_xx, M_yy, M_xy, _ = _kappa_second_moments(
            X, Y, kappa, x_c, y_c, mask=ap_mask
        )
        stats_k = _moments_to_ellipticity(M_xx, M_yy, M_xy)

        # Uniform-weighted moments: weight = 1.0 inside the mask.
        # Measures the geometric shape of the iso-κ region (what the eye
        # sees in contour maps), independent of κ value distribution.
        uniform_w = np.where(ap_mask, 1.0, 0.0)
        x_c_u, y_c_u, _ = _weighted_centroid(X, Y, uniform_w, mask=ap_mask)
        M_xx_u, M_yy_u, M_xy_u, _ = _weighted_second_moments(
            X, Y, uniform_w, x_c_u, y_c_u, mask=ap_mask
        )
        stats_u = _moments_to_ellipticity(M_xx_u, M_yy_u, M_xy_u)

        rows.append({
            "quantile": float(qt),
            "threshold_kappa": threshold,
            "n_pixels": n_pix,
            "fraction_kept": n_pix / n_total_finite,
            # κ-weighted (mass-weighted) measurement
            "q": stats_k["q"],
            "e_mod": stats_k["e_mod"],
            "PA_astro_deg": stats_k["PA_astro_deg"],
            "PA_image_deg": stats_k["PA_image_deg"],
            "kappa_centroid": (x_c, y_c),
            # Uniform-weighted (geometric) measurement
            "q_uniform": stats_u["q"],
            "e_mod_uniform": stats_u["e_mod"],
            "PA_astro_uniform_deg": stats_u["PA_astro_deg"],
            "PA_image_uniform_deg": stats_u["PA_image_deg"],
            "uniform_centroid": (x_c_u, y_c_u),
            "kappa_total": M0,
            "grid_extent": extent,
        })
    return rows


def print_threshold_table(rows):
    print(f"\n  {'qntile':>6s} {'κ thresh':>9s} {'frac':>6s}  "
          f"{'q_κ':>6s} {'PA_κ':>7s}  {'q_uni':>6s} {'PA_uni':>8s}")
    print("  " + "-" * 60)
    for r in rows:
        print(
            f"  {r['quantile']:>6.2f} {r['threshold_kappa']:>9.4f} "
            f"{r['fraction_kept']:>6.3f}  "
            f"{r['q']:>6.3f} {r['PA_astro_deg']:>6.1f}°  "
            f"{r['q_uniform']:>6.3f} {r['PA_astro_uniform_deg']:>7.1f}°"
        )
    print("\n  q_κ   : κ-weighted (mass-weighted) quadrupole "
          "— what the lensing inertia tensor 'sees'")
    print("  q_uni : uniform-weighted moments of the iso-κ region "
          "— what the eye 'sees' in contour plots")


def plot_ellipticity_vs_threshold(
    rows, out_path=None, mark_quantiles=(0.20, 0.50, 0.80),
    title_suffix="",
):
    """
    Two-panel diagnostic plot: q(threshold) and PA(threshold), showing
    both κ-weighted (mass) and uniform-weighted (geometric) curves.
    """
    if not rows:
        print("  (no threshold rows to plot)")
        return None

    qntiles = np.array([r["quantile"] for r in rows])
    q_k = np.array([r["q"] for r in rows])
    pa_k = np.array([r["PA_astro_deg"] for r in rows])
    q_u = np.array([r["q_uniform"] for r in rows])
    pa_u = np.array([r["PA_astro_uniform_deg"] for r in rows])
    kappa_thr = np.array([r["threshold_kappa"] for r in rows])

    fig, (ax_q, ax_pa) = plt.subplots(
        2, 1, figsize=(7.0, 7.5), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    ax_q.plot(qntiles, q_k, "o-", color="C0", markersize=4, linewidth=1.6,
              label="κ-weighted (mass quadrupole)")
    ax_q.plot(qntiles, q_u, "s-", color="C2", markersize=4, linewidth=1.6,
              label="uniform-weighted (geometric)")
    ax_q.set_ylabel("q = b/a")
    ax_q.set_ylim(0.0, 1.05)
    ax_q.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax_q.set_title(f"ARCH ellipticity vs κ threshold{title_suffix}",
                   fontsize=10)

    ax_pa.plot(qntiles, pa_k, "o-", color="C0", markersize=4, linewidth=1.6,
               label="κ-weighted")
    ax_pa.plot(qntiles, pa_u, "s-", color="C2", markersize=4, linewidth=1.6,
               label="uniform-weighted")
    ax_pa.set_ylabel("PA_astro (deg)")
    ax_pa.set_xlabel("κ-threshold quantile (over inner-masked grid)")
    ax_pa.grid(True, alpha=0.3)
    ax_pa.set_ylim(0, 180)
    ax_pa.legend(loc="upper left", fontsize=9, framealpha=0.95)

    for q_mark in mark_quantiles:
        for ax in (ax_q, ax_pa):
            ax.axvline(q_mark, color="black", linewidth=0.6,
                       linestyle=":", alpha=0.5)

    ax_top = ax_q.secondary_xaxis(
        "top",
        functions=(
            lambda x: np.interp(x, qntiles, kappa_thr),
            lambda k: np.interp(k, kappa_thr, qntiles),
        ),
    )
    ax_top.set_xlabel("κ value", fontsize=9)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved threshold-sweep plot to {out_path}")
    return fig


def threshold_row_to_analysis_result(thr_row, weighting="kappa", label=None):
    """
    Convert a single threshold-sweep row into the same dict shape that
    analyse_kappa_ellipticity() returns, so existing plotters and
    comparators can consume it.

    Parameters
    ----------
    thr_row : dict
        One row from ellipticity_vs_threshold().
    weighting : {'kappa', 'uniform'}
        Which moment definition to extract.
    label : str or None
        Display label.  Defaults to e.g. 'κ>0.572 (q85)'.
    """
    if weighting == "uniform":
        q = thr_row["q_uniform"]
        e_mod = thr_row["e_mod_uniform"]
        PA_im = thr_row["PA_image_uniform_deg"]
        PA_as = thr_row["PA_astro_uniform_deg"]
        cen = thr_row["uniform_centroid"]
    else:
        q = thr_row["q"]
        e_mod = thr_row["e_mod"]
        PA_im = thr_row["PA_image_deg"]
        PA_as = thr_row["PA_astro_deg"]
        cen = thr_row["kappa_centroid"]

    if label is None:
        label = (f"κ>{thr_row['threshold_kappa']:.3f} "
                 f"(q{int(round(thr_row['quantile']*100)):02d})")

    return {
        (0, label): {
            "e1": float("nan"), "e2": float("nan"),
            "e_mod": e_mod, "q": q,
            "PA_image_deg": PA_im, "PA_astro_deg": PA_as,
            "aperture_center_input": cen,
            "kappa_centroid": cen,
            "aperture_radius_arcsec": float("nan"),
            "aperture_radius_kpc": float("nan"),
            "kappa_total": thr_row["kappa_total"],
            "n_pixels": thr_row["n_pixels"],
            "M_xx": float("nan"), "M_yy": float("nan"), "M_xy": float("nan"),
            "grid_extent": thr_row["grid_extent"],
            "sheet_subtracted": False,
            "kappa_floor_quantile": 0.0,
            "threshold_kappa": thr_row["threshold_kappa"],
            "threshold_quantile": thr_row["quantile"],
            "weighting": weighting,
        }
    }


def _select_threshold_row(rows, target_quantile):
    """Find the threshold-sweep row whose quantile is closest to target."""
    if not rows:
        return None
    qs = np.array([r["quantile"] for r in rows])
    idx = int(np.argmin(np.abs(qs - target_quantile)))
    return rows[idx]


def print_headline(thr_row, weighting="kappa"):
    """Pretty-print the single 'headline' ARCH ellipticity for the paper."""
    if weighting == "uniform":
        q = thr_row["q_uniform"]
        e_mod = thr_row["e_mod_uniform"]
        PA = thr_row["PA_astro_uniform_deg"]
    else:
        q = thr_row["q"]
        e_mod = thr_row["e_mod"]
        PA = thr_row["PA_astro_deg"]
    print("\n  " + "=" * 60)
    print(f"  ARCH HEADLINE ellipticity ({weighting}-weighted)")
    print("  " + "-" * 60)
    print(f"    κ threshold      : {thr_row['threshold_kappa']:.4f}")
    print(f"    quantile         : {thr_row['quantile']:.2f}")
    print(f"    fraction of grid : {thr_row['fraction_kept']:.2f}")
    print(f"    q (b/a)          : {q:.3f}")
    print(f"    |e|              : {e_mod:.3f}")
    print(f"    PA_astro         : {PA:.1f}°")
    print("  " + "=" * 60)


# =============================================================================
# External κ map analysis (e.g., MARS, GRALE, LENSTOOL published κ FITS files)
# =============================================================================

def analyse_external_kappa_fits(
    fits_path,
    center_ra_deg: float = None,
    center_dec_deg: float = None,
    half_size_arcsec: float = 100.0,
    inner_mask_arcsec: float = 0.0,
    inner_mask_centers_radec=None,
    hdu_index=None,
    quantiles=None,
):
    """
    Run the threshold-quantile ellipticity sweep on an external published
    κ map (FITS file with a WCS), returning rows in the same format as
    ellipticity_vs_threshold() so the two can be plotted together.

    Parameters
    ----------
    fits_path : str or Path
        Path to FITS file containing a 2D κ map with WCS in the header.
    center_ra_deg, center_dec_deg : float
        Cluster center in J2000 decimal degrees.  If None, defaults to
        A2744 BCG-N: RA = 3.58134, Dec = -30.38866.  The arcsec offset
        grid is built relative to this point.
    half_size_arcsec : float
        Half-side of the analysis box (a square centered on the given
        center).  Default 100″ ≈ 470 kpc at A2744's z = 0.308, comparable
        to ARCH's 500-kpc aperture.
    inner_mask_arcsec : float
        If > 0, mask within this radius of each entry in
        inner_mask_centers_radec.  Default 0 (no inner mask) — external
        κ maps are pixelized reconstructions without analytic cusps, so
        this isn't usually needed.
    inner_mask_centers_radec : list of (ra_deg, dec_deg) or None
        Centers to apply the inner mask around.
    hdu_index : int or None
        Which HDU to read.  If None, scans for the first 2D image HDU.
    quantiles : array-like
        Threshold quantiles to evaluate.  Matches the default of
        ellipticity_vs_threshold() if None.

    Returns
    -------
    (rows, extra) : tuple
        rows  : list of dicts with the same keys as ellipticity_vs_threshold()
        extra : dict with kappa_min, kappa_max, kappa_dynamic_range,
                center_used_radec, and grid info (useful diagnostics).
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord

    # A2744 BCG-N default center
    if center_ra_deg is None:
        center_ra_deg = 3.58134
    if center_dec_deg is None:
        center_dec_deg = -30.38866

    with fits.open(str(fits_path)) as hdul:
        if hdu_index is not None:
            hdu = hdul[hdu_index]
            kappa_map = np.asarray(hdu.data, dtype=float)
            header = hdu.header
        else:
            kappa_map = None
            header = None
            for h in hdul:
                if h.data is not None and getattr(h.data, "ndim", 0) == 2:
                    kappa_map = np.asarray(h.data, dtype=float)
                    header = h.header
                    break
            if kappa_map is None:
                raise ValueError(f"No 2D image HDU found in {fits_path}")

    try:
        wcs = WCS(header).celestial
    except Exception as e:
        raise RuntimeError(
            f"Could not build a celestial WCS from {fits_path}: {e}.  "
            f"If the FITS file has a non-standard header, you may need to "
            f"pass hdu_index explicitly or pre-process it."
        )

    # Pixel grid → world coords → arcsec offsets relative to the center.
    ny, nx = kappa_map.shape
    iy, ix = np.indices((ny, nx))
    px_ra, px_dec = wcs.wcs_pix2world(ix.flatten(), iy.flatten(), 0)
    cos_dec = np.cos(np.deg2rad(center_dec_deg))
    # +x = East (RA increasing); +y = North
    X = (((px_ra - center_ra_deg + 540.0) % 360.0 - 180.0)
         * 3600.0 * cos_dec).reshape(ny, nx)
    Y = ((px_dec - center_dec_deg) * 3600.0).reshape(ny, nx)

    region_mask = (np.abs(X) <= half_size_arcsec) & (np.abs(Y) <= half_size_arcsec)

    # Inner mask (optional)
    inner_mask = np.ones_like(X, dtype=bool)
    if inner_mask_arcsec > 0 and inner_mask_centers_radec:
        for mc_ra, mc_dec in inner_mask_centers_radec:
            mc_x = ((mc_ra - center_ra_deg + 540.0) % 360.0 - 180.0) * 3600.0 * cos_dec
            mc_y = (mc_dec - center_dec_deg) * 3600.0
            R = np.hypot(X - mc_x, Y - mc_y)
            inner_mask &= (R >= inner_mask_arcsec)

    base_mask = (
        region_mask & inner_mask
        & np.isfinite(kappa_map) & (kappa_map > 0)
    )
    n_total = int(np.sum(base_mask))
    if n_total < 200:
        raise RuntimeError(
            f"Too few positive finite κ pixels in cropped region of "
            f"{fits_path} (got {n_total}).  Increase half_size_arcsec or "
            f"check the center coordinates."
        )

    kfin = kappa_map[base_mask]
    kappa_min, kappa_max = float(kfin.min()), float(kfin.max())

    if quantiles is None:
        quantiles = np.concatenate([
            np.linspace(0.00, 0.50, 11),
            np.linspace(0.55, 0.95, 9),
        ])

    rows = []
    for qt in quantiles:
        threshold = float(np.quantile(kfin, qt))
        ap_mask = base_mask & (kappa_map >= threshold)
        n_pix = int(np.sum(ap_mask))
        if n_pix < 20:
            continue
        x_c, y_c, M0 = _kappa_centroid(X, Y, kappa_map, mask=ap_mask)
        M_xx, M_yy, M_xy, _ = _kappa_second_moments(
            X, Y, kappa_map, x_c, y_c, mask=ap_mask
        )
        stats_k = _moments_to_ellipticity(M_xx, M_yy, M_xy)

        uniform_w = np.where(ap_mask, 1.0, 0.0)
        x_c_u, y_c_u, _ = _weighted_centroid(X, Y, uniform_w, mask=ap_mask)
        M_xx_u, M_yy_u, M_xy_u, _ = _weighted_second_moments(
            X, Y, uniform_w, x_c_u, y_c_u, mask=ap_mask
        )
        stats_u = _moments_to_ellipticity(M_xx_u, M_yy_u, M_xy_u)

        rows.append({
            "quantile": float(qt),
            "threshold_kappa": threshold,
            "n_pixels": n_pix,
            "fraction_kept": n_pix / n_total,
            "q": stats_k["q"],
            "e_mod": stats_k["e_mod"],
            "PA_astro_deg": stats_k["PA_astro_deg"],
            "PA_image_deg": stats_k["PA_image_deg"],
            "kappa_centroid": (x_c, y_c),
            "q_uniform": stats_u["q"],
            "e_mod_uniform": stats_u["e_mod"],
            "PA_astro_uniform_deg": stats_u["PA_astro_deg"],
            "PA_image_uniform_deg": stats_u["PA_image_deg"],
            "uniform_centroid": (x_c_u, y_c_u),
            "kappa_total": M0,
        })

    extra = {
        "kappa_min": kappa_min,
        "kappa_max": kappa_max,
        "kappa_dynamic_range": kappa_max / kappa_min if kappa_min > 0 else float("inf"),
        "center_used_radec": (center_ra_deg, center_dec_deg),
        "half_size_arcsec": half_size_arcsec,
        "n_pixels_total": n_total,
    }
    return rows, extra


def plot_threshold_comparison(
    arch_rows, external_rows,
    arch_label="ARCH",
    external_label="MARS",
    out_path=None,
    title=None,
    arch_extra=None,
    external_extra=None,
):
    """
    Single-figure two-panel comparison: q(quantile) and PA(quantile) for
    ARCH and an external κ map on the same axes.
    """
    if not arch_rows or not external_rows:
        print("  (one of the row lists is empty — skipping comparison plot)")
        return None

    aq = np.array([r["quantile"] for r in arch_rows])
    aQ = np.array([r["q"] for r in arch_rows])
    aPA = np.array([r["PA_astro_deg"] for r in arch_rows])
    aK = np.array([r["threshold_kappa"] for r in arch_rows])

    eq = np.array([r["quantile"] for r in external_rows])
    eQ = np.array([r["q"] for r in external_rows])
    ePA = np.array([r["PA_astro_deg"] for r in external_rows])

    fig, (ax_q, ax_pa) = plt.subplots(
        2, 1, figsize=(7.5, 7.5), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    ax_q.plot(aq, aQ, "o-", color="C0", markersize=4, linewidth=1.8,
              label=arch_label)
    ax_q.plot(eq, eQ, "s-", color="C3", markersize=4, linewidth=1.8,
              label=external_label)
    ax_q.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    ax_q.set_ylabel("q = b/a")
    ax_q.set_ylim(0.0, 1.05)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax_q.set_title(
        title or f"{arch_label} vs {external_label}: ellipticity vs κ threshold",
        fontsize=10,
    )

    # Optional κ-value annotation on top axis for ARCH (left as primary).
    if len(aq) > 1 and np.all(np.diff(aq) > 0):
        ax_top = ax_q.secondary_xaxis(
            "top",
            functions=(
                lambda x: np.interp(x, aq, aK),
                lambda k: np.interp(k, aK, aq),
            ),
        )
        ax_top.set_xlabel(f"{arch_label} κ at quantile", fontsize=9)

    ax_pa.plot(aq, aPA, "o-", color="C0", markersize=4, linewidth=1.8,
               label=arch_label)
    ax_pa.plot(eq, ePA, "s-", color="C3", markersize=4, linewidth=1.8,
               label=external_label)
    ax_pa.set_ylabel("PA_astro (deg)")
    ax_pa.set_xlabel("κ-threshold quantile (within analysis region)")
    ax_pa.set_ylim(0, 180)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # Caption block with dynamic-range diagnostics
    cap_lines = []
    if external_extra is not None:
        cap_lines.append(
            f"{external_label}: κ in [{external_extra['kappa_min']:.3f}, "
            f"{external_extra['kappa_max']:.3f}], "
            f"dynamic range {external_extra['kappa_dynamic_range']:.1f}x"
        )
    if arch_extra is not None:
        cap_lines.append(
            f"{arch_label}: κ in [{arch_extra.get('kappa_min', 0):.3f}, "
            f"{arch_extra.get('kappa_max', 0):.3f}], "
            f"dynamic range {arch_extra.get('kappa_dynamic_range', 0):.1f}x"
        )
    if cap_lines:
        fig.text(0.02, 0.005, "\n".join(cap_lines), fontsize=8,
                 color="#444", family="monospace")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved ARCH-vs-{external_label} comparison plot to {out_path}")
    return fig


# =============================================================================
# Analytical dPIE κ field from published LENSTOOL halo parameters
# =============================================================================
# Build a κ map from the dPIE/PIEMD halo parameters published in LENSTOOL
# reconstructions (Bergamini+2023, Furtak+2023, ...), then run the same
# threshold-quantile sweep on it.  This gives a true apples-to-apples
# methodology comparison: same κ definition, same moment operator, same
# threshold quantiles — only the underlying reconstruction differs.
#
# dPIE 2D surface mass density (Eliasdottir+2007 / Limousin+2005):
#
#     Σ(R) = (σ² / 2G) × (r_cut / (r_cut − r_core))
#            × [ 1/√(R² + r_core²) − 1/√(R² + r_cut²) ]
#
# Elliptical generalization (Kassiola & Kovner 1993):
#
#     R_ε² = x'² (1 − ε) + y'² (1 + ε),  ε = (1 − q²)/(1 + q²)
#
# where (x', y') is the halo-frame position (rotated by the position
# angle).  When r_cut → ∞ this reduces to PIEMD; we leave r_cut as a
# parameter (LENSTOOL default for cluster-scale halos is ~800 kpc).

_G_SI = 6.6743e-11   # m³ / (kg s²)
_C_SI = 2.99792458e8  # m / s
_KPC_M = 3.0856775814913673e19  # meters per kpc


def _compute_sigma_crit_kg_m2(z_lens, z_source, cosmo=None):
    """
    Critical surface density Σ_crit in kg / m² for a given lens / source
    redshift pair.
    """
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    D_l  = cosmo.angular_diameter_distance(z_lens).to(u.m).value
    D_s  = cosmo.angular_diameter_distance(z_source).to(u.m).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).to(u.m).value
    return _C_SI ** 2 / (4.0 * np.pi * _G_SI) * (D_s / (D_l * D_ls))


def kappa_dpie_elliptical(
    X, Y,
    x_h_arcsec, y_h_arcsec,
    sigma_kms, r_core_arcsec, r_cut_arcsec,
    q, PA_astro_deg,
    sigma_crit_kg_m2, kpc_per_arcsec,
):
    """
    Convergence κ on the (X, Y) arcsec grid for a single elliptical dPIE
    halo with LENSTOOL-style parameterization.

    Parameters
    ----------
    X, Y : 2D arrays
        Arcsec offsets (X = +East, Y = +North).
    x_h_arcsec, y_h_arcsec : float
        Halo center in the same coordinate frame.
    sigma_kms : float
        Velocity dispersion in km/s.
    r_core_arcsec, r_cut_arcsec : float
        Core and cutoff radii in arcsec.
    q : float
        Axis ratio b/a (q ∈ (0, 1]).
    PA_astro_deg : float
        Major-axis position angle in degrees, measured from N toward E
        (the same convention used elsewhere in this module).
    sigma_crit_kg_m2 : float
        Σ_crit for the chosen (z_lens, z_source) in kg/m².
    kpc_per_arcsec : float
        Proper kpc per arcsec at the lens redshift.

    Returns
    -------
    kappa : 2D array, same shape as X
    """
    q = max(min(float(q), 1.0), 1e-3)  # clamp to avoid div-by-zero in ε
    eps = (1.0 - q ** 2) / (1.0 + q ** 2)

    # PA_astro is measured from +Y (North) toward +X (East), CCW from N
    # PA_image (CCW from +X) = (90 - PA_astro) mod 180
    pa_image = np.deg2rad((90.0 - PA_astro_deg) % 180.0)
    cos_p, sin_p = np.cos(pa_image), np.sin(pa_image)
    dx = X - x_h_arcsec
    dy = Y - y_h_arcsec
    # Halo frame: x' along the major axis
    x_rot =  dx * cos_p + dy * sin_p
    y_rot = -dx * sin_p + dy * cos_p

    R_ell_arcsec = np.sqrt(x_rot ** 2 * (1.0 - eps) + y_rot ** 2 * (1.0 + eps))

    # Convert all lengths to meters
    R_m      = R_ell_arcsec * kpc_per_arcsec * _KPC_M
    r_core_m = r_core_arcsec * kpc_per_arcsec * _KPC_M
    r_cut_m  = r_cut_arcsec  * kpc_per_arcsec * _KPC_M

    # dPIE surface density
    sigma_v_m_s = sigma_kms * 1e3
    prefac = (sigma_v_m_s ** 2 / (2.0 * _G_SI)) * (r_cut_m / (r_cut_m - r_core_m))
    Sigma = prefac * (
        1.0 / np.sqrt(R_m ** 2 + r_core_m ** 2)
        - 1.0 / np.sqrt(R_m ** 2 + r_cut_m ** 2)
    )
    return Sigma / sigma_crit_kg_m2


# Published cluster-scale halo parameters.  All positions are given as
# offsets from a "BCG-N reference" point; the reference is taken to be
# the ARCH halo at the largest +y position at run time so that the two
# reconstructions are compared at the same on-sky location.
# r_cut defaults to 800 kpc (LENSTOOL standard for cluster-scale halos).
LITERATURE_HALO_SETS = {
    "bergamini23": {
        "reference": "Bergamini et al. 2023 (A&A 670, A60); arXiv:2207.09416",
        "r_cut_kpc_default": 800.0,
        "halos": [
            {
                "name": "Halo-1 (BCG-N)",
                "anchor": "N", "pos_offset_arcsec_from_anchor": (-1.5, -0.1),
                "sigma_kms": 522.7, "r_core_arcsec": 6.8,
                "q": 0.500, "PA_astro_deg": 179.7,
            },
            {
                "name": "Halo-2 (BCG-S)",
                # Position offset is relative to BCG-N in Bergamini's table,
                # so we anchor to N here too rather than the S BCG.
                "anchor": "N", "pos_offset_arcsec_from_anchor": (-18.2, -15.7),
                "sigma_kms": 633.9, "r_core_arcsec": 7.6,
                "q": 0.655, "PA_astro_deg": 36.7,
            },
        ],
    },
    "furtak23": {
        "reference": "Furtak et al. 2023 (MNRAS 523, 4568); arXiv:2212.04381",
        "r_cut_kpc_default": 800.0,
        "halos": [
            # Furtak doesn't publish position offsets in a single-table
            # form that's directly translatable; we place each halo at the
            # corresponding ARCH BCG location.
            {
                "name": "Main-1 (BCG-S)",
                "anchor": "S", "pos_offset_arcsec_from_anchor": (0.0, 0.0),
                "sigma_kms": 681.0, "r_core_kpc": 23.0,
                "q": 0.647, "PA_astro_deg": 45.0,
            },
            {
                "name": "Main-2 (BCG-N)",
                "anchor": "N", "pos_offset_arcsec_from_anchor": (0.0, 0.0),
                "sigma_kms": 807.0, "r_core_kpc": 67.0,
                "q": 0.616, "PA_astro_deg": 17.0,
            },
        ],
    },
}


def _arch_bcg_anchors(lenses):
    """
    Return {'N': (x, y), 'S': (x, y)} using ARCH halos: north = max-y, south = min-y.
    For single-halo reconstructions both anchors collapse to the one halo.
    """
    xs = np.atleast_1d(lenses.x).astype(float)
    ys = np.atleast_1d(lenses.y).astype(float)
    if xs.size == 0:
        raise ValueError("No ARCH halos to anchor literature halos to.")
    n_idx = int(np.argmax(ys))
    s_idx = int(np.argmin(ys))
    return {
        "N": (float(xs[n_idx]), float(ys[n_idx])),
        "S": (float(xs[s_idx]), float(ys[s_idx])),
    }


def construct_literature_kappa_map(
    halo_set_name, lenses,
    X, Y,
    z_lens, z_source,
    cosmo=None,
):
    """
    Build a κ map by summing dPIE contributions from a named literature
    halo set on the (X, Y) grid.

    The halo positions are anchored to the ARCH BCG-N / BCG-S halo
    positions (max-y / min-y) so the literature map and the ARCH map are
    centered on the same on-sky structures.
    """
    if halo_set_name not in LITERATURE_HALO_SETS:
        raise ValueError(
            f"Unknown literature set {halo_set_name!r}.  "
            f"Available: {list(LITERATURE_HALO_SETS)}"
        )
    spec = LITERATURE_HALO_SETS[halo_set_name]
    sigma_crit = _compute_sigma_crit_kg_m2(z_lens, z_source, cosmo=cosmo)

    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_lens).to(
        u.kpc / u.arcsec
    ).value

    anchors = _arch_bcg_anchors(lenses)
    r_cut_arcsec_default = spec["r_cut_kpc_default"] / kpc_per_arcsec

    kappa_total = np.zeros_like(X, dtype=float)
    halo_info_for_plot = []
    for halo in spec["halos"]:
        anchor_xy = anchors[halo["anchor"]]
        x_h = anchor_xy[0] + halo["pos_offset_arcsec_from_anchor"][0]
        y_h = anchor_xy[1] + halo["pos_offset_arcsec_from_anchor"][1]

        # Resolve r_core: paper may give arcsec OR kpc; convert if needed
        if "r_core_arcsec" in halo:
            r_core_arcsec = float(halo["r_core_arcsec"])
        elif "r_core_kpc" in halo:
            r_core_arcsec = float(halo["r_core_kpc"]) / kpc_per_arcsec
        else:
            raise KeyError(f"Halo {halo['name']} missing r_core")

        r_cut_arcsec = halo.get("r_cut_arcsec",
                                halo.get("r_cut_kpc", None))
        if r_cut_arcsec is None:
            r_cut_arcsec = r_cut_arcsec_default
        elif "r_cut_kpc" in halo:
            r_cut_arcsec = halo["r_cut_kpc"] / kpc_per_arcsec

        kappa_h = kappa_dpie_elliptical(
            X, Y, x_h, y_h,
            sigma_kms=halo["sigma_kms"],
            r_core_arcsec=r_core_arcsec,
            r_cut_arcsec=r_cut_arcsec,
            q=halo["q"], PA_astro_deg=halo["PA_astro_deg"],
            sigma_crit_kg_m2=sigma_crit,
            kpc_per_arcsec=kpc_per_arcsec,
        )
        kappa_total += kappa_h
        halo_info_for_plot.append({
            "name": halo["name"], "x": x_h, "y": y_h,
            "q": halo["q"], "PA_astro_deg": halo["PA_astro_deg"],
            "sigma_kms": halo["sigma_kms"],
            "r_core_arcsec": r_core_arcsec,
            "r_cut_arcsec": r_cut_arcsec,
        })

    extra = {
        "halo_set_name": halo_set_name,
        "reference": spec["reference"],
        "halos_placed": halo_info_for_plot,
        "sigma_crit_kg_m2": sigma_crit,
        "kpc_per_arcsec": kpc_per_arcsec,
    }
    return kappa_total, extra


def analyse_literature_kappa_set(
    halo_set_name, lenses,
    z_lens, z_source,
    arch_grid_extent,
    grid_res=512,
    inner_mask_arcsec=5.0,
    quantiles=None,
    cosmo=None,
):
    """
    Build the literature κ on the same grid extent that ARCH uses, then
    run the threshold-quantile sweep so the rows match ARCH's exactly.

    Returns (rows, extra), where extra includes κ range and placed-halo
    information suitable for downstream plotting.
    """
    x0, x1, y0, y1 = arch_grid_extent
    xs = np.linspace(x0, x1, grid_res)
    ys = np.linspace(y0, y1, grid_res)
    X, Y = np.meshgrid(xs, ys)

    kappa, extra = construct_literature_kappa_map(
        halo_set_name, lenses, X, Y,
        z_lens=z_lens, z_source=z_source, cosmo=cosmo,
    )

    # Inner mask around the placed halo centers (same idea as ARCH)
    inner_mask = np.ones_like(X, dtype=bool)
    if inner_mask_arcsec > 0:
        for hinfo in extra["halos_placed"]:
            R = np.hypot(X - hinfo["x"], Y - hinfo["y"])
            inner_mask &= (R >= inner_mask_arcsec)

    base_mask = inner_mask & np.isfinite(kappa) & (kappa > 0)
    kfin = kappa[base_mask]
    n_total = int(kfin.size)
    if n_total < 200:
        raise RuntimeError(
            f"Too few positive κ pixels for {halo_set_name} threshold sweep "
            f"(got {n_total})."
        )
    extra["kappa_min"] = float(kfin.min())
    extra["kappa_max"] = float(kfin.max())
    extra["kappa_dynamic_range"] = (
        extra["kappa_max"] / extra["kappa_min"] if extra["kappa_min"] > 0
        else float("inf")
    )
    extra["n_pixels_total"] = n_total

    if quantiles is None:
        quantiles = np.concatenate([
            np.linspace(0.00, 0.50, 11),
            np.linspace(0.55, 0.95, 9),
        ])

    rows = []
    for qt in quantiles:
        threshold = float(np.quantile(kfin, qt))
        ap_mask = base_mask & (kappa >= threshold)
        n_pix = int(np.sum(ap_mask))
        if n_pix < 20:
            continue
        x_c, y_c, M0 = _kappa_centroid(X, Y, kappa, mask=ap_mask)
        M_xx, M_yy, M_xy, _ = _kappa_second_moments(
            X, Y, kappa, x_c, y_c, mask=ap_mask
        )
        stats_k = _moments_to_ellipticity(M_xx, M_yy, M_xy)

        uniform_w = np.where(ap_mask, 1.0, 0.0)
        x_c_u, y_c_u, _ = _weighted_centroid(X, Y, uniform_w, mask=ap_mask)
        M_xx_u, M_yy_u, M_xy_u, _ = _weighted_second_moments(
            X, Y, uniform_w, x_c_u, y_c_u, mask=ap_mask
        )
        stats_u = _moments_to_ellipticity(M_xx_u, M_yy_u, M_xy_u)

        rows.append({
            "quantile": float(qt),
            "threshold_kappa": threshold,
            "n_pixels": n_pix,
            "fraction_kept": n_pix / n_total,
            "q": stats_k["q"],
            "e_mod": stats_k["e_mod"],
            "PA_astro_deg": stats_k["PA_astro_deg"],
            "PA_image_deg": stats_k["PA_image_deg"],
            "kappa_centroid": (x_c, y_c),
            "q_uniform": stats_u["q"],
            "e_mod_uniform": stats_u["e_mod"],
            "PA_astro_uniform_deg": stats_u["PA_astro_deg"],
            "PA_image_uniform_deg": stats_u["PA_image_deg"],
            "uniform_centroid": (x_c_u, y_c_u),
            "kappa_total": M0,
        })
    return rows, extra


def plot_threshold_comparison_multi(
    arch_rows,
    external_curves,   # list of (label, rows, extra) tuples
    out_path=None,
    title=None,
    arch_extra=None,
):
    """
    Compare ARCH's q(quantile) and PA(quantile) curves against one or
    more external reconstructions on a single two-panel figure.
    """
    if not arch_rows or not external_curves:
        print("  (no rows to compare — skipping multi-comparison plot)")
        return None

    fig, (ax_q, ax_pa) = plt.subplots(
        2, 1, figsize=(8.0, 8.0), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    palette = ["C3", "C2", "C4", "C5"]
    markers = ["s", "^", "D", "v"]

    aq = np.array([r["quantile"] for r in arch_rows])
    aQ = np.array([r["q"] for r in arch_rows])
    aPA = np.array([r["PA_astro_deg"] for r in arch_rows])
    aK = np.array([r["threshold_kappa"] for r in arch_rows])

    ax_q.plot(aq, aQ, "o-", color="C0", markersize=4, linewidth=2.0,
              label="ARCH", zorder=5)
    ax_pa.plot(aq, aPA, "o-", color="C0", markersize=4, linewidth=2.0,
               label="ARCH", zorder=5)

    cap_lines = []
    if arch_extra is not None:
        cap_lines.append(
            f"ARCH: κ ∈ [{arch_extra.get('kappa_min', 0):.3f}, "
            f"{arch_extra.get('kappa_max', 0):.3f}], "
            f"range {arch_extra.get('kappa_dynamic_range', 0):.1f}x"
        )

    for i, (label, rows, extra) in enumerate(external_curves):
        c = palette[i % len(palette)]
        m = markers[i % len(markers)]
        eq = np.array([r["quantile"] for r in rows])
        eQ = np.array([r["q"] for r in rows])
        ePA = np.array([r["PA_astro_deg"] for r in rows])
        ax_q.plot(eq, eQ, m + "-", color=c, markersize=4, linewidth=1.8,
                  label=label, alpha=0.9)
        ax_pa.plot(eq, ePA, m + "-", color=c, markersize=4, linewidth=1.8,
                   label=label, alpha=0.9)
        if extra:
            cap_lines.append(
                f"{label}: κ ∈ [{extra.get('kappa_min', 0):.3f}, "
                f"{extra.get('kappa_max', 0):.3f}], "
                f"range {extra.get('kappa_dynamic_range', 0):.1f}x"
            )

    ax_q.axhline(1.0, color="gray", linewidth=0.8, alpha=0.5)
    ax_q.set_ylabel("q = b/a")
    ax_q.set_ylim(0.0, 1.05)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax_q.set_title(
        title or "Ellipticity vs κ threshold: ARCH vs literature dPIE reconstructions",
        fontsize=10,
    )

    if len(aq) > 1 and np.all(np.diff(aq) > 0):
        ax_top = ax_q.secondary_xaxis(
            "top",
            functions=(
                lambda x: np.interp(x, aq, aK),
                lambda k: np.interp(k, aK, aq),
            ),
        )
        ax_top.set_xlabel("ARCH κ at quantile", fontsize=9)

    ax_pa.set_ylabel("PA_astro (deg)")
    ax_pa.set_xlabel("κ-threshold quantile (within analysis region)")
    ax_pa.set_ylim(0, 180)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.legend(loc="upper left", fontsize=10, framealpha=0.95)

    if cap_lines:
        fig.text(0.02, 0.005, "\n".join(cap_lines), fontsize=8,
                 color="#444", family="monospace")

    fig.tight_layout(rect=(0, 0.04 + 0.012 * len(cap_lines), 1, 1))
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"  Saved literature-κ comparison plot to {out_path}")
    return fig


def plot_chapter_comparison_figure(
    lenses, z_source, lens_type,
    arch_rows,
    literature_set_names,
    out_path=None,
    grid_res=400,
    inner_mask_arcsec=5.0,
    contour_levels=(0.2, 0.4, 0.6, 0.8, 1.2),
    headline_quantile=0.85,
    inter_halo_axis_deg=None,
    cosmo=None,
    title=None,
):
    """
    Publication-quality five-panel comparison figure:

        ┌──────────────┬──────────────┬──────────────┐
        │ ARCH κ map   │ Bergamini κ  │ Furtak κ map │
        │ (POWER_LAW)  │ (dPIE)       │ (dPIE)       │
        ├──────────────┴──────┬───────┴──────────────┤
        │  q vs threshold     │ PA vs threshold      │
        └─────────────────────┴──────────────────────┘

    The κ maps share a single color scale so the dynamic-range difference
    between ARCH POWER_LAW and the LENSTOOL dPIE reconstructions is
    visually unmistakable.  Inter-halo axis reference line drawn on the
    PA panel to anchor the geometric interpretation.
    """
    if not arch_rows:
        raise ValueError("arch_rows must be non-empty")
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo

    # ---- Build all κ maps on the same grid ----
    extent = arch_rows[0]["grid_extent"]
    x0, x1, y0, y1 = extent

    # ARCH κ (uses its native calculate_kappa, which handles the profile math)
    Xa, Ya, kappa_arch = utils.calculate_kappa(
        lenses, extent=extent, lens_type=lens_type,
        source_redshift=z_source,
    )

    # Literature κ maps on the same grid as ARCH so visual alignment is exact
    lit_kappa_maps = []
    for set_name in literature_set_names:
        if set_name not in LITERATURE_HALO_SETS:
            print(f"  (skipping unknown set {set_name})")
            continue
        kappa_lit, lit_extra = construct_literature_kappa_map(
            set_name, lenses, Xa, Ya,
            z_lens=lenses.redshift, z_source=z_source, cosmo=cosmo,
        )
        pretty = {"bergamini23": "Bergamini+23 (dPIE)",
                  "furtak23":    "Furtak+23 (dPIE)"}.get(set_name, set_name)
        lit_kappa_maps.append((set_name, pretty, kappa_lit, lit_extra))

    # ---- Common κ color scale: use a robust upper bound across all maps ----
    all_kappa = np.concatenate([
        kappa_arch[np.isfinite(kappa_arch)].ravel()
    ] + [
        k[np.isfinite(k)].ravel() for _, _, k, _ in lit_kappa_maps
    ])
    vmin = 0.0
    vmax = float(np.percentile(all_kappa, 99.5))

    # ---- Threshold-sweep rows for each literature set ----
    lit_rows_lookup = {}
    for set_name, pretty, kappa_lit, lit_extra in lit_kappa_maps:
        try:
            rows_lit, _ = analyse_literature_kappa_set(
                set_name, lenses,
                z_lens=lenses.redshift, z_source=z_source,
                arch_grid_extent=extent, grid_res=grid_res,
                inner_mask_arcsec=inner_mask_arcsec,
                cosmo=cosmo,
            )
            lit_rows_lookup[pretty] = rows_lit
        except Exception as e:
            print(f"  (lit threshold sweep failed for {set_name}: {e})")

    # ---- Figure layout ----
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(
        2, 6,
        height_ratios=[1.15, 1.0],
        hspace=0.38, wspace=0.6,
        left=0.06, right=0.97, top=0.92, bottom=0.10,
    )
    ax_arch = fig.add_subplot(gs[0, 0:2])
    ax_berg = fig.add_subplot(gs[0, 2:4])
    ax_furt = fig.add_subplot(gs[0, 4:6])
    ax_q    = fig.add_subplot(gs[1, 0:3])
    ax_pa   = fig.add_subplot(gs[1, 3:6])

    # Order top-row panels:  ARCH, then whatever lit sets we have
    top_axes = [ax_arch, ax_berg, ax_furt]
    panel_specs = [("ARCH POWER_LAW", kappa_arch)]
    for _, pretty, kappa_lit, _ in lit_kappa_maps:
        panel_specs.append((pretty, kappa_lit))
    # Hide any unused top axes (e.g. only one lit set given)
    for ax in top_axes[len(panel_specs):]:
        ax.set_visible(False)

    # Locations of ARCH halos
    arch_xs = np.atleast_1d(lenses.x).astype(float)
    arch_ys = np.atleast_1d(lenses.y).astype(float)

    def _draw_kappa_panel(ax, title_text, kappa):
        im = ax.imshow(
            kappa, origin="lower",
            extent=[x0, x1, y0, y1],
            vmin=vmin, vmax=vmax,
            cmap="viridis", aspect="equal",
            interpolation="bilinear",
        )
        # Contour lines at fixed levels
        cs = ax.contour(
            Xa, Ya, kappa,
            levels=[lv for lv in contour_levels if vmin < lv < vmax * 1.1],
            colors="white", linewidths=0.7, alpha=0.85,
        )
        ax.clabel(cs, fmt="%.1f", fontsize=7, inline=True)
        # ARCH halo positions for spatial reference
        ax.plot(arch_xs, arch_ys, marker="*", linestyle="",
                color="white", markersize=10, markeredgecolor="black",
                markeredgewidth=0.7)
        ax.set_title(title_text, fontsize=10)
        ax.set_xlabel("RA offset (″)", fontsize=9)
        ax.set_ylabel("Dec offset (″)", fontsize=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.tick_params(labelsize=8)
        return im

    last_im = None
    for ax, (label, k) in zip(top_axes[:len(panel_specs)], panel_specs):
        last_im = _draw_kappa_panel(ax, label, k)

    # Single shared colorbar to the right of the top row
    cbar_ax = fig.add_axes([0.978, 0.55, 0.012, 0.34])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("κ (z_s = %.1f)" % z_source, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ---- Bottom row: q and PA curves ----
    aq  = np.array([r["quantile"] for r in arch_rows])
    aQ  = np.array([r["q"] for r in arch_rows])
    aPA = np.array([r["PA_astro_deg"] for r in arch_rows])

    palette_lit = ["#d62728", "#2ca02c", "#9467bd"]  # red, green, purple
    markers_lit = ["s", "^", "D"]

    ax_q.plot(aq, aQ, "o-", color="#1f77b4", markersize=4.5,
              linewidth=2.2, label="ARCH POWER_LAW", zorder=5)
    ax_pa.plot(aq, aPA, "o-", color="#1f77b4", markersize=4.5,
               linewidth=2.2, label="ARCH POWER_LAW", zorder=5)

    for i, (pretty, rows_lit) in enumerate(lit_rows_lookup.items()):
        c = palette_lit[i % len(palette_lit)]
        m = markers_lit[i % len(markers_lit)]
        eq = np.array([r["quantile"] for r in rows_lit])
        eQ = np.array([r["q"] for r in rows_lit])
        ePA = np.array([r["PA_astro_deg"] for r in rows_lit])
        ax_q.plot(eq, eQ, m + "-", color=c, markersize=4, linewidth=1.7,
                  label=pretty, alpha=0.92)
        ax_pa.plot(eq, ePA, m + "-", color=c, markersize=4, linewidth=1.7,
                   label=pretty, alpha=0.92)

    # Headline-quantile marker
    for ax in (ax_q, ax_pa):
        ax.axvline(headline_quantile, color="black",
                   linewidth=0.7, linestyle=":", alpha=0.55)
    ax_q.text(headline_quantile + 0.005, 0.05,
              f"q = {headline_quantile:.2f}",
              fontsize=8, color="black", alpha=0.7,
              rotation=90, va="bottom")

    ax_q.axhline(1.0, color="gray", linewidth=0.7, alpha=0.5)
    ax_q.set_ylabel("q = b/a", fontsize=10)
    ax_q.set_xlabel("κ-threshold quantile", fontsize=10)
    ax_q.set_ylim(0.3, 1.05)
    ax_q.set_xlim(-0.02, 0.98)
    ax_q.grid(True, alpha=0.3)
    ax_q.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax_q.tick_params(labelsize=9)

    # PA panel: anchor to inter-halo axis
    if inter_halo_axis_deg is None:
        # Compute from ARCH halo positions if 2+ halos
        if len(arch_xs) >= 2:
            n_idx = int(np.argmax(arch_ys))
            s_idx = int(np.argmin(arch_ys))
            dx = arch_xs[n_idx] - arch_xs[s_idx]
            dy = arch_ys[n_idx] - arch_ys[s_idx]
            inter_halo_axis_deg = (90.0 - np.rad2deg(np.arctan2(dy, dx))) % 180.0
    if inter_halo_axis_deg is not None:
        ax_pa.axhline(inter_halo_axis_deg, color="black",
                      linewidth=0.9, linestyle="--", alpha=0.55)
        ax_pa.text(0.01, inter_halo_axis_deg + 1.5,
                   f"inter-halo geometric axis = {inter_halo_axis_deg:.0f}°",
                   fontsize=8, color="black", alpha=0.7)

    ax_pa.set_ylabel("PA_astro (deg)", fontsize=10)
    ax_pa.set_xlabel("κ-threshold quantile", fontsize=10)
    ax_pa.set_xlim(-0.02, 0.98)
    ax_pa.set_ylim(0, 100)
    ax_pa.grid(True, alpha=0.3)
    ax_pa.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax_pa.tick_params(labelsize=9)

    # Title spans both rows
    if title is None:
        title = ("Cluster morphology comparison: ARCH vs LENSTOOL "
                 "parametric reconstructions for Abell 2744")
    fig.suptitle(title, fontsize=11.5, y=0.985)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=180)
        print(f"  Saved chapter-quality comparison figure to {out_path}")
    return fig


def _resolve_ellipticity_output_dir(csv_path: Path) -> Path:
    """
    Decide where to place ellipticity PDFs given an input lens CSV.

    If the CSV lives in a directory named 'NFW' or 'POWER_LAW' (the
    layout produced by pipelines/read_jwst.py with per-mode subdirs),
    place outputs in a sibling 'Ellipticity/' directory under the same
    cluster parent.  Otherwise, fall back to the CSV's own directory
    (legacy behavior).
    """
    parent = csv_path.parent
    if parent.name in ("NFW", "POWER_LAW"):
        out = parent.parent / "Ellipticity"
        out.mkdir(parents=True, exist_ok=True)
        return out
    return parent


def _parse_csv_header_metadata(csv_path):
    """
    Read leading '#' comment lines and extract any 'key=value' metadata.

    Looks for redshift=X and theta_star=X (used by POWER_LAW outputs;
    NFW outputs may omit one or both).  Returns a dict, possibly empty.
    """
    meta = {}
    try:
        with open(csv_path) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                for m in re.finditer(r"([A-Za-z_]+)\s*=\s*([0-9.eE+\-]+)", line):
                    key, val = m.group(1), m.group(2)
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        pass
    except Exception:
        pass
    return meta


def load_lenses_from_csv(
    csv_path, lens_type, theta_star=30.0, z_lens_override=None,
):
    """
    Load a lens object from an ARCH CSV.

    Cluster redshift resolution order:
      1. z_lens_override (CLI / caller, if not None)
      2. 'redshift=X' parsed from CSV header comment
      3. Whatever lens.import_from_csv() sets (zero if not stored)

    Raises a clean error if no usable redshift can be found.
    """
    csv_path = Path(csv_path)
    header_meta = _parse_csv_header_metadata(csv_path)

    # Decide redshift
    if z_lens_override is not None and z_lens_override > 0:
        z_lens = float(z_lens_override)
        redshift_source = "CLI --z-lens"
    elif header_meta.get("redshift", 0.0) > 0:
        z_lens = float(header_meta["redshift"])
        redshift_source = "CSV header"
    else:
        z_lens = 0.0
        redshift_source = "unset"

    # POWER_LAW theta_star pivot
    ts = float(header_meta.get("theta_star", theta_star))

    if lens_type == "NFW":
        lens = halo_obj.NFW_Lens(
            x=[], y=[], z=[], mass=[], concentration=[],
            redshift=z_lens, chi2=[],
        )
    elif lens_type == "POWER_LAW":
        lens = halo_obj.PowerLawHalo(
            x=[], y=[], kappa_star=[], slope=[],
            theta_star=ts, redshift=z_lens, chi2=[],
        )
    else:
        raise ValueError(
            f"lens_type must be 'NFW' or 'POWER_LAW', got {lens_type!r}"
        )

    lens.import_from_csv(str(csv_path))

    # import_from_csv may overwrite lens.redshift with 0 for CSVs
    # whose header doesn't carry redshift (NFW CSVs at present).
    # Restore from our resolved value.
    if (getattr(lens, "redshift", None) in (None, 0, 0.0)) and z_lens > 0:
        lens.redshift = z_lens

    if not (getattr(lens, "redshift", 0) and lens.redshift > 0):
        raise RuntimeError(
            f"Cluster redshift could not be determined for {csv_path}. "
            f"The CSV header has no 'redshift=' field and no --z-lens "
            f"override was provided.  Pass --z-lens 0.308 (or the "
            f"appropriate value for your cluster) on the command line."
        )

    print(f"  z_lens    : {lens.redshift:.4f}  (from {redshift_source})")
    return lens


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        prog="python -m scripts.ellipticity_analysis",
        description=(
            "Compute convergence ellipticity for an ARCH cluster "
            "reconstruction.  Reports axis ratio, ellipticity components, "
            "position angle, and (if literature values are populated) "
            "comparison to published reconstructions."
        ),
    )
    p.add_argument("csv_path", help="Path to lenses_*.csv from ARCH.")
    p.add_argument("--lens-type", choices=["NFW", "POWER_LAW"],
                   default="POWER_LAW")
    p.add_argument("--z-source", type=float, default=2.0,
                   help="Source redshift for κ (default: 2.0).")
    p.add_argument("--z-lens", type=float, default=None,
                   help="Override cluster (lens) redshift.  By default, "
                        "uses the 'redshift=X' field in the CSV header.  "
                        "Required for NFW CSVs whose header omits the field.")
    p.add_argument("--apertures-kpc", type=float, nargs="*",
                   default=[250.0, 500.0],
                   help="Aperture radii in kpc.  Default: 250 500.")
    p.add_argument("--no-full-field", action="store_true",
                   help="Skip the 'full grid' aperture entry.")
    p.add_argument("--inner-mask-arcsec", type=float, default=5.0,
                   help="Mask cells within this radius of each halo center.")
    p.add_argument("--grid-res", type=int, default=512,
                   help="κ grid resolution (default: 512).")
    p.add_argument("--grid-half-size", type=float, default=None,
                   help="Half-size of the square κ grid in arcsec.  If unset, "
                        "sized to comfortably wrap the largest aperture so "
                        "aperture limits actually clip inside the grid.")
    p.add_argument("--kappa-floor-quantile", type=float, default=0.0,
                   help="Subtract the κ value at this quantile from the map "
                        "before computing moments.  Removes uniform-sheet "
                        "contamination.  Try 0.10–0.25 when one POWER_LAW halo "
                        "has slope pinned at the lower bound.  Default: 0 "
                        "(off).")
    p.add_argument("--theta-star", type=float, default=30.0,
                   help="POWER_LAW pivot radius (default: 30 arcsec).")
    p.add_argument("--plot", action="store_true",
                   help="Save a κ-contour overlay PDF beside the CSV.")
    p.add_argument("--vs-threshold", action="store_true",
                   help="Compute and plot q and PA as a function of κ "
                        "threshold quantile.  Diagnostic for whether the "
                        "global quadrupole is hiding intermediate-κ "
                        "elongation in a multi-halo configuration.")
    p.add_argument("--external-fits", type=str, default=None,
                   help="Path to an external κ FITS map (e.g. MARS, GRALE).  "
                        "When set, runs the same threshold sweep on this "
                        "map and produces a single comparison plot vs ARCH.  "
                        "Implies --vs-threshold.")
    p.add_argument("--external-label", type=str, default="MARS",
                   help="Label for the external map in the comparison plot "
                        "(default: MARS).")
    p.add_argument("--external-half-size", type=float, default=100.0,
                   help="Half-side of the analysis box (arcsec) for the "
                        "external map.  Default 100\" ≈ 470 kpc at A2744.")
    p.add_argument("--external-center-ra", type=float, default=None,
                   help="Cluster center RA (deg, J2000) for the external "
                        "map.  Default: A2744 BCG-N = 3.58134.")
    p.add_argument("--external-center-dec", type=float, default=None,
                   help="Cluster center Dec (deg, J2000) for the external "
                        "map.  Default: A2744 BCG-N = -30.38866.")
    p.add_argument("--external-hdu", type=int, default=None,
                   help="HDU index to read from the external FITS (default: "
                        "first 2D image HDU).")
    p.add_argument("--compare-literature-kappa", type=str, default=None,
                   help="Comma-separated list of literature halo sets to "
                        "reconstruct as κ fields and compare against ARCH.  "
                        "Available: bergamini23, furtak23.  "
                        "Example: --compare-literature-kappa bergamini23,furtak23  "
                        "Implies --vs-threshold.  Each set is built from "
                        "published dPIE parameters using ARCH's BCG-N / BCG-S "
                        "halo positions as anchors.")
    p.add_argument("--report-quantile", type=float, default=None,
                   help="Treat the threshold-sweep row at this quantile as "
                        "the ARCH headline ellipticity.  Used for the "
                        "literature-overlay plot.  Implies --vs-threshold.  "
                        "Try 0.85 for the bimodal-cluster regime.")
    p.add_argument("--report-weighting", choices=["kappa", "uniform"],
                   default="kappa",
                   help="Which moment definition to use for the headline "
                        "result.  Default: kappa (mass-weighted).")
    p.add_argument("--ellipse-size-arcsec", type=float, default=22.0,
                   help="Visual semi-major axis of overlaid ellipses in "
                        "the literature-comparison plot.  Default: 22.  "
                        "Try 12-15 for the A2744 two-halo geometry to "
                        "reduce ellipse overlap.")
    p.add_argument("--literature-overlay", action="store_true",
                   help="Also save a comparison plot overlaying ARCH and "
                        "literature ellipses (only meaningful for A2744 with "
                        "populated A2744_LITERATURE).")
    p.add_argument("--sky-mirrored", action="store_true",
                   help="Flag that +x corresponds to West (compass E is "
                        "left) in the image.  Affects PA reporting.")
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    print(f"\n=== ARCH ellipticity analysis ===")
    print(f"  CSV       : {csv_path}")
    print(f"  Lens type : {args.lens_type}")
    print(f"  z_source  : {args.z_source}")

    lenses = load_lenses_from_csv(
        csv_path, args.lens_type,
        theta_star=args.theta_star,
        z_lens_override=args.z_lens,
    )
    n_halos = len(np.atleast_1d(lenses.x))
    print(f"  Halos     : {n_halos}")
    if n_halos == 0:
        sys.exit("No halos in CSV.")

    kpc_per_arcsec = (
        COSMO.kpc_proper_per_arcmin(lenses.redshift)
              .to(u.kpc / u.arcsec).value
    )
    print(f"  1″ = {kpc_per_arcsec:.2f} kpc proper")

    aperture_radii = [r_kpc / kpc_per_arcsec for r_kpc in args.apertures_kpc]
    if not args.no_full_field:
        aperture_radii.append(np.inf)

    results = analyse_kappa_ellipticity(
        lenses, z_source=args.z_source, lens_type=args.lens_type,
        aperture_radii_arcsec=aperture_radii,
        inner_mask_arcsec=args.inner_mask_arcsec,
        grid_res=args.grid_res,
        grid_half_size_arcsec=args.grid_half_size,
        kappa_floor_quantile=args.kappa_floor_quantile,
    )
    print_ellipticity_table(results)

    if args.sky_mirrored:
        print("\n  --sky-mirrored: convert printed PA_astro to "
              "standard astronomical PA via PA' = (180 - PA_astro) mod 180.")

    # Conditional literature comparison
    is_a2744 = "ABELL_2744" in str(csv_path).upper() or "A2744" in str(csv_path).upper()
    if is_a2744:
        compare_to_literature(results, A2744_LITERATURE_TEMPLATE)

    # If a headline quantile is requested, force the threshold sweep on
    do_threshold_sweep = (
        args.vs_threshold
        or (args.report_quantile is not None)
        or (args.external_fits is not None)
        or (args.compare_literature_kappa is not None)
    )

    thr_rows = None
    headline_row = None
    arch_kappa_range = None
    if do_threshold_sweep:
        thr_rows = ellipticity_vs_threshold(
            lenses, z_source=args.z_source, lens_type=args.lens_type,
            grid_half_size_arcsec=args.grid_half_size,
            inner_mask_arcsec=args.inner_mask_arcsec,
            grid_res=args.grid_res,
        )
        print_threshold_table(thr_rows)
        # Record κ range for the diagnostic caption
        thr_kappas = [r["threshold_kappa"] for r in thr_rows]
        if thr_kappas:
            kmin, kmax = float(min(thr_kappas)), float(max(thr_kappas))
            arch_kappa_range = {
                "kappa_min": kmin, "kappa_max": kmax,
                "kappa_dynamic_range": kmax / kmin if kmin > 0 else float("inf"),
            }
        out_dir = _resolve_ellipticity_output_dir(csv_path)
        thr_path = out_dir / (csv_path.stem + "_ellipticity_vs_threshold.png")
        plot_ellipticity_vs_threshold(
            thr_rows, out_path=str(thr_path),
            title_suffix=f"  ({args.lens_type})",
        )

        if args.report_quantile is not None:
            headline_row = _select_threshold_row(thr_rows, args.report_quantile)
            if headline_row is not None:
                print_headline(headline_row, weighting=args.report_weighting)

    # External κ map comparison (e.g. MARS)
    if args.external_fits is not None:
        ext_path = Path(args.external_fits)
        if not ext_path.exists():
            print(f"  --external-fits: file not found at {ext_path}")
        else:
            print(f"\n=== External κ map analysis ({args.external_label}) ===")
            print(f"  FITS path : {ext_path}")
            try:
                ext_rows, ext_extra = analyse_external_kappa_fits(
                    ext_path,
                    center_ra_deg=args.external_center_ra,
                    center_dec_deg=args.external_center_dec,
                    half_size_arcsec=args.external_half_size,
                    hdu_index=args.external_hdu,
                )
                print(f"  Center    : "
                      f"({ext_extra['center_used_radec'][0]:.5f}, "
                      f"{ext_extra['center_used_radec'][1]:.5f}) deg")
                print(f"  Half-size : {ext_extra['half_size_arcsec']:.1f}\"")
                print(f"  κ range   : [{ext_extra['kappa_min']:.3f}, "
                      f"{ext_extra['kappa_max']:.3f}]  "
                      f"(dynamic range {ext_extra['kappa_dynamic_range']:.1f}x)")
                print(f"  N pixels  : {ext_extra['n_pixels_total']}")
                print(f"\n  {args.external_label} threshold sweep:")
                print_threshold_table(ext_rows)

                out_dir = _resolve_ellipticity_output_dir(csv_path)
                comp_path = out_dir / (
                    csv_path.stem
                    + f"_threshold_vs_{args.external_label.lower()}.png"
                )
                plot_threshold_comparison(
                    arch_rows=thr_rows,
                    external_rows=ext_rows,
                    arch_label="ARCH",
                    external_label=args.external_label,
                    out_path=str(comp_path),
                    arch_extra=arch_kappa_range,
                    external_extra=ext_extra,
                )
            except Exception as e:
                print(f"  External analysis failed: {type(e).__name__}: {e}")

    # Literature-κ reconstruction comparison (e.g. Bergamini, Furtak)
    if args.compare_literature_kappa is not None and thr_rows is not None:
        set_names = [s.strip().lower() for s in
                     args.compare_literature_kappa.split(",") if s.strip()]
        external_curves = []
        for set_name in set_names:
            if set_name not in LITERATURE_HALO_SETS:
                print(f"  Unknown literature set: {set_name!r} (skipping). "
                      f"Available: {list(LITERATURE_HALO_SETS)}")
                continue
            print(f"\n=== Literature-κ analysis ({set_name}) ===")
            spec = LITERATURE_HALO_SETS[set_name]
            print(f"  Reference : {spec['reference']}")
            try:
                # Reuse the grid extent ARCH's threshold sweep used.
                arch_extent = thr_rows[0]["grid_extent"]
                lit_rows, lit_extra = analyse_literature_kappa_set(
                    set_name, lenses,
                    z_lens=lenses.redshift,
                    z_source=args.z_source,
                    arch_grid_extent=arch_extent,
                    grid_res=args.grid_res,
                    inner_mask_arcsec=args.inner_mask_arcsec,
                )
                print(f"  Σ_crit    : {lit_extra['sigma_crit_kg_m2']:.3f} kg/m²")
                print(f"  Halos placed (anchored to ARCH BCG positions):")
                for h in lit_extra["halos_placed"]:
                    print(f"    {h['name']:<28s}  "
                          f"x={h['x']:6.1f}\"  y={h['y']:6.1f}\"  "
                          f"σ={h['sigma_kms']:.0f} km/s  "
                          f"r_core={h['r_core_arcsec']:.2f}\"  "
                          f"q={h['q']:.3f}  PA={h['PA_astro_deg']:.1f}°")
                print(f"  κ range   : [{lit_extra['kappa_min']:.3f}, "
                      f"{lit_extra['kappa_max']:.3f}]  "
                      f"(dynamic range {lit_extra['kappa_dynamic_range']:.1f}x)")
                print(f"  Threshold sweep:")
                print_threshold_table(lit_rows)
                # Pretty label for the plot legend
                pretty = {"bergamini23": "Bergamini+23 (dPIE)",
                          "furtak23":    "Furtak+23 (dPIE)"}.get(set_name, set_name)
                external_curves.append((pretty, lit_rows, lit_extra))
            except Exception as e:
                print(f"  Literature-κ analysis failed for {set_name}: "
                      f"{type(e).__name__}: {e}")

        if external_curves:
            out_dir = _resolve_ellipticity_output_dir(csv_path)
            tag = "_vs_".join(s.strip().lower()
                              for s in args.compare_literature_kappa.split(","))
            comp_path = out_dir / (
                csv_path.stem + f"_threshold_vs_{tag}.png"
            )
            plot_threshold_comparison_multi(
                arch_rows=thr_rows,
                external_curves=external_curves,
                out_path=str(comp_path),
                arch_extra=arch_kappa_range,
            )

            # Chapter-quality 5-panel figure: κ maps + curves
            chap_path = out_dir / (
                csv_path.stem + f"_chapter_comparison_{tag}.png"
            )
            try:
                plot_chapter_comparison_figure(
                    lenses, z_source=args.z_source,
                    lens_type=args.lens_type,
                    arch_rows=thr_rows,
                    literature_set_names=set_names,
                    out_path=str(chap_path),
                    grid_res=args.grid_res,
                    inner_mask_arcsec=args.inner_mask_arcsec,
                    headline_quantile=(args.report_quantile
                                       if args.report_quantile is not None
                                       else 0.85),
                )
            except Exception as e:
                print(f"  Chapter-quality figure failed: "
                      f"{type(e).__name__}: {e}")

    if args.plot:
        out_dir = _resolve_ellipticity_output_dir(csv_path)
        plot_path = out_dir / (csv_path.stem + "_ellipticity.png")
        plot_ellipticity_overlay(
            lenses, args.z_source, results, lens_type=args.lens_type,
            out_path=str(plot_path), sky_mirrored=args.sky_mirrored,
        )

    if args.literature_overlay and is_a2744:
        populated = [e for e in A2744_LITERATURE_TEMPLATE
                     if e.get("q") is not None
                     and e.get("PA_astro_deg") is not None]
        if populated:
            out_dir = _resolve_ellipticity_output_dir(csv_path)

            # Decide which ARCH result to use as the comparison ellipse.
            # If --report-quantile was specified, use that single
            # threshold-based result.  Otherwise use the existing global
            # aperture results.
            if headline_row is not None:
                arch_for_overlay = threshold_row_to_analysis_result(
                    headline_row, weighting=args.report_weighting,
                )
                comp_path = out_dir / (
                    csv_path.stem
                    + f"_ellipticity_vs_literature_q"
                    + f"{int(round(headline_row['quantile']*100)):02d}.png"
                )
                source_tag = (f"ARCH at κ-quantile {headline_row['quantile']:.2f} "
                              f"({args.report_weighting}-weighted)")
            else:
                arch_for_overlay = results
                comp_path = out_dir / (
                    csv_path.stem + "_ellipticity_vs_literature.png"
                )
                source_tag = "ARCH global aperture quadrupole"

            print(f"\n  Literature overlay: using {source_tag}")
            plot_ellipticity_with_literature(
                lenses, args.z_source, arch_for_overlay,
                literature_table=A2744_LITERATURE_TEMPLATE,
                lens_type=args.lens_type,
                out_path=str(comp_path),
                sky_mirrored=args.sky_mirrored,
                ellipse_size_arcsec=args.ellipse_size_arcsec,
            )
        else:
            print("  --literature-overlay requested but A2744_LITERATURE "
                  "has no populated entries.")


if __name__ == "__main__":
    main()