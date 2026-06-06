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

def _kappa_centroid(X, Y, kappa, mask=None):
    """κ-weighted centroid.  Returns (x_c, y_c, M0)."""
    if mask is None:
        m = np.isfinite(kappa) & (kappa > 0)
    else:
        m = mask & np.isfinite(kappa) & (kappa > 0)
    if not np.any(m):
        return float("nan"), float("nan"), 0.0
    k = kappa[m]
    M0 = float(k.sum())
    if M0 <= 0:
        return float("nan"), float("nan"), 0.0
    return float((X[m] * k).sum() / M0), float((Y[m] * k).sum() / M0), M0


def _kappa_second_moments(X, Y, kappa, x_c, y_c, mask=None):
    """Second moments about (x_c, y_c)."""
    if mask is None:
        m = np.isfinite(kappa) & (kappa > 0)
    else:
        m = mask & np.isfinite(kappa) & (kappa > 0)
    if not np.any(m):
        return 0.0, 0.0, 0.0, 0.0
    k = kappa[m]
    dx = X[m] - x_c
    dy = Y[m] - y_c
    M_xx = float((dx * dx * k).sum())
    M_yy = float((dy * dy * k).sum())
    M_xy = float((dx * dy * k).sum())
    M0 = float(k.sum())
    return M_xx, M_yy, M_xy, M0


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
# Visualization: contour + major axis overlay
# =============================================================================

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


def load_lenses_from_csv(csv_path, lens_type, theta_star=30.0):
    csv_path = str(csv_path)
    if lens_type == "NFW":
        lens = halo_obj.NFW_Lens(
            x=[], y=[], z=[], mass=[], concentration=[],
            redshift=0.0, chi2=[],
        )
    elif lens_type == "POWER_LAW":
        lens = halo_obj.PowerLawHalo(
            x=[], y=[], kappa_star=[], slope=[],
            theta_star=theta_star, redshift=0.0, chi2=[],
        )
    else:
        raise ValueError(f"lens_type must be 'NFW' or 'POWER_LAW', got {lens_type!r}")
    lens.import_from_csv(csv_path)
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

    lenses = load_lenses_from_csv(csv_path, args.lens_type,
                                  theta_star=args.theta_star)
    n_halos = len(np.atleast_1d(lenses.x))
    print(f"  Halos     : {n_halos}")
    if n_halos == 0:
        sys.exit("No halos in CSV.")

    kpc_per_arcsec = (
        COSMO.kpc_proper_per_arcmin(lenses.redshift)
              .to(u.kpc / u.arcsec).value
    )
    print(f"  z_lens    : {lenses.redshift:.3f}")
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

    if args.plot:
        out_dir = _resolve_ellipticity_output_dir(csv_path)
        plot_path = out_dir / (csv_path.stem + "_ellipticity.pdf")
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
            comp_path = out_dir / (
                csv_path.stem + "_ellipticity_vs_literature.pdf"
            )
            plot_ellipticity_with_literature(
                lenses, args.z_source, results,
                literature_table=A2744_LITERATURE_TEMPLATE,
                lens_type=args.lens_type,
                out_path=str(comp_path),
                sky_mirrored=args.sky_mirrored,
            )
        else:
            print("  --literature-overlay requested but A2744_LITERATURE "
                  "has no populated entries.")


if __name__ == "__main__":
    main()