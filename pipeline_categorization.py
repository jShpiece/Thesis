"""
Thesis-quality taxonomy plot for lensing reconstruction pipelines.

Axes:
  x: Mass representation (Parametric -> Hybrid -> Field-based)
  y: Inference strategy (Iterative/local/staged -> Global simultaneous)

Encoding:
  color: dominant lensing regime (Strong / Weak / Flexion / Multi)
  marker: Multi-signal vs single-regime (diamond vs circle)

Design choices:
- No 3D imports.
- Deterministic jitter to reduce overplotting.
- Thesis-safe labeling: label only a curated set; use leader lines.
- Label placement uses axes-fraction offsets (stable) and clamps labels inside frame.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pipelines


# ----------------------------
# Style (optional)
# ----------------------------
def apply_style(style_name: str = "scientific_presentation.mplstyle") -> None:
    try:
        plt.style.use(style_name)
    except OSError:
        pass


# ----------------------------
# Deterministic jitter
# ----------------------------
def stable_jitter(name: str, scale: float = 0.035) -> Tuple[float, float]:
    """Deterministic jitter from a stable hash; same name -> same jitter."""
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    r1 = int(h[:8], 16) / 0xFFFFFFFF
    r2 = int(h[8:16], 16) / 0xFFFFFFFF
    return (scale * (r1 - 0.5), scale * (r2 - 0.5))


# ----------------------------
# Regime encoding
# ----------------------------
# Colorblind-safe categorical palette (fixed hue order, validated for CVD
# separation): blue / teal / violet / orange / neutral gray. Marker shape is
# varied in tandem so identity never depends on color alone.
COLOR_MAP = {
    "Strong": "#2a78d6",   # blue
    "Weak": "#1baf7a",     # teal
    "Flexion": "#4a3aa7",  # violet
    "Multi": "#eb6834",    # orange
    "Other": "#898781",    # neutral gray
}

MARKER_MAP = {
    "Strong": "o",
    "Weak": "s",
    "Flexion": "^",
    "Multi": "D",
    "Other": "X",
}

RING_COLOR = "black"  # secondary "flexion-enabled" halo; kept neutral so it
                       # never collides with a categorical hue


def dominant_regime(signals: List[str]) -> str:
    s = set(signals)
    has_strong = "Strong Lensing" in s
    has_weak = ("Weak Shear" in s) or ("Magnification" in s)
    has_flex = "Flexion" in s

    n = int(has_strong) + int(has_weak) + int(has_flex)
    if n >= 2:
        return "Multi"
    if has_strong:
        return "Strong"
    if has_weak:
        return "Weak"
    if has_flex:
        return "Flexion"
    return "Other"


# ----------------------------
# Coordinate mapping
# ----------------------------
def build_representation_mapping() -> defaultdict:
    # x: Parametric (-1) -> Hybrid (0) -> Field-based (+1)
    return defaultdict(
        lambda: 0.0,
        {
            # Parametric family
            "Parametric": -1.0,
            "Parametric + WL profile fitting": -0.8,

            # Analytic inversion (not parametric, not free-form inference; place near hybrid but slightly field-leaning)
            "Direct Fourier inversion": 0.3,

            # Hybrid family
            "Hybrid (Grid + Parametric core)": 0.0,
            "Hybrid (Grid + Galaxy halos)": 0.0,

            # ARCH: parametric-leaning with (possible) adaptive corrections
            "Parametric + adaptive corrections (ARCH)": -0.5,

            # Sparse methods (field-leaning but structured)
            "Sparse multiscale (Wavelet)": 0.6,

            # Free-form / field-based
            "Free-form (Grid-based)": 1.0,
            "Free-form (Genetic algorithm)": 1.0,
            "Free-form (Pixel-based)": 1.0,
        },
    )


def build_inference_mapping() -> defaultdict:
    # y: Iterative/local (-1) -> Global (+1)
    return defaultdict(
        lambda: 0.0,
        {
            "Global (MCMC)": 1.0,
            "Global MCMC": 1.0,
            "Global optimization": 0.9,
            "Global inversion": 0.85,
            "Global convex optimization": 0.85,
            "Global linear inversion": 0.8,
            "Global likelihood": 0.8,
            "Global": 0.75,
            "Semi-global": 0.45,
            "Monte Carlo sampling": 0.35,

            "Iterative multiscale": -0.35,
            "Iterative linear inversion": -0.35,
            "Iterative evolutionary": -0.6,
            "Iterative / staged optimization": -0.35,
        },
    )


# ----------------------------
# Label placement helpers
# ----------------------------
def clamp01(v: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, v))


def data_to_axes(xd: float, yd: float, xmin: float, xmax: float, ymin: float, ymax: float) -> Tuple[float, float]:
    xa = (xd - xmin) / (xmax - xmin)
    ya = (yd - ymin) / (ymax - ymin)
    return xa, ya


def axes_to_data(xa: float, ya: float, xmin: float, xmax: float, ymin: float, ymax: float) -> Tuple[float, float]:
    xd = xmin + xa * (xmax - xmin)
    yd = ymin + ya * (ymax - ymin)
    return xd, yd


def label_offset_axes(name: str, base: float = 0.035, direction: Tuple[float, float] | None = None) -> Tuple[float, float]:
    # Direction can be pinned explicitly (to resolve collisions between
    # nearby points); otherwise fall back to a deterministic hash direction.
    if direction is not None:
        dx, dy = direction
    else:
        dx, dy = stable_jitter(name, scale=1.0)
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    return base * (dx / norm), base * (dy / norm)


# Manual routing for labels whose anchor points sit close together in data
# space (deterministic hashing alone can still send neighbors toward each
# other). (offset_scale, (dir_x, dir_y)) - tuned by inspecting the render.
LABEL_OVERRIDES: Dict[str, Tuple[float, Tuple[float, float]]] = {
    "LENSTOOL": (0.11, (0.15, 1.0)),
    "GLAFIC": (0.10, (1.0, -0.15)),
    "Zitrin_LTM": (0.08, (-1.0, -0.5)),
    "WSLAP_plus": (0.075, (-1.0, -0.3)),
    "SaWLens": (0.075, (1.0, 0.6)),
    "Bradac_2005": (0.075, (0.5, 1.0)),
    "Lanusse_2016": (0.08, (-0.8, 1.0)),
    "Kaiser_Squires": (0.045, (0.0, -1.0)),
    "PixeLens": (0.045, (0.7, -0.6)),
    "GRALE": (0.045, (0.7, -0.6)),
    "ARCH": (0.06, (-0.2, -1.0)),
}


# ----------------------------
# Main plotting
# ----------------------------
def main() -> None:
    apply_style()

    # Presentation-scale overrides: the .mplstyle is tuned for a printed page;
    # a defense slide is viewed from across a room, so bump everything up.
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "legend.fontsize": 13,
            "figure.constrained_layout.use": False,
        }
    )

    pipelines: Dict[str, Dict[str, object]] = {
        "LENSTOOL": {
            "representation": "Parametric",
            "inference_strategy": "Global (MCMC)",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "GLAFIC": {
            "representation": "Parametric",
            "inference_strategy": "Global optimization",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "Zitrin_LTM": {
            "representation": "Parametric",
            "inference_strategy": "Semi-global",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "CATS": {
            "representation": "Parametric",
            "inference_strategy": "Global MCMC",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "Sharon_Johnson": {
            "representation": "Parametric",
            "inference_strategy": "Global optimization",
            "signals_used": ["Strong Lensing"],
        },
        "Grillo_Model": {
            "representation": "Parametric",
            "inference_strategy": "Global",
            "signals_used": ["Strong Lensing"],
        },
        "Bradac_2005": {
            "representation": "Free-form (Grid-based)",
            "inference_strategy": "Global inversion",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "SaWLens": {
            "representation": "Hybrid (Grid + Parametric core)",
            "inference_strategy": "Iterative multiscale",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "WSLAP_plus": {
            "representation": "Hybrid (Grid + Galaxy halos)",
            "inference_strategy": "Iterative linear inversion",
            "signals_used": ["Strong Lensing", "Weak Shear"],
        },
        "GRALE": {
            "representation": "Free-form (Genetic algorithm)",
            "inference_strategy": "Iterative evolutionary",
            "signals_used": ["Strong Lensing"],
        },
        "PixeLens": {
            "representation": "Free-form (Pixel-based)",
            "inference_strategy": "Monte Carlo sampling",
            "signals_used": ["Strong Lensing"],
        },
        "Lanusse_2016": {
            "representation": "Sparse multiscale (Wavelet)",
            "inference_strategy": "Global convex optimization",
            "signals_used": ["Weak Shear", "Flexion"],
        },
        "Kaiser_Squires": {
            "representation": "Direct Fourier inversion",
            "inference_strategy": "Global linear inversion",
            "signals_used": ["Weak Shear"],
        },
        "Umetsu_WL": {
            "representation": "Parametric + WL profile fitting",
            "inference_strategy": "Global likelihood",
            "signals_used": ["Weak Shear", "Magnification"],
        },
        "ARCH": {
            "representation": "Parametric + adaptive corrections (ARCH)",
            "inference_strategy": "Iterative / staged optimization",
            "signals_used": ["Weak Shear", "Flexion", "Strong Lensing"],
        },
    }

    rep_map = build_representation_mapping()
    inf_map = build_inference_mapping()

    # Build points
    points = []
    for name, info in pipelines.items():
        rep = str(info["representation"])
        inf = str(info["inference_strategy"])
        signals = list(info["signals_used"])  # type: ignore[arg-type]

        x0 = rep_map[rep]
        y0 = inf_map[inf]
        dx, dy = stable_jitter(name, scale=0.035)

        regime = dominant_regime(signals)
        points.append(
            {
                "name": name,
                "x": x0 + dx,
                "y": y0 + dy,
                "regime": regime,
                "color": COLOR_MAP[regime],
                "marker": MARKER_MAP[regime],
                "has_flexion": ("Flexion" in set(signals)),
            }
        )


    fig, ax = plt.subplots(figsize=(12, 10.5))

    # Background bands for representation categories (subtle)
    ax.axvspan(-1.1, -0.33, alpha=0.06, zorder=0)  # Parametric
    ax.axvspan(-0.33, 0.33, alpha=0.03, zorder=0)  # Hybrid
    ax.axvspan(0.33, 1.1, alpha=0.06, zorder=0)    # Field-based

    # Set limits BEFORE label placement (important)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")

    # Hide numeric axis ticks/labels (subjective map)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    # Reference lines
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5, zorder=1)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5, zorder=1)

    # Scatter by regime
    regime_groups = defaultdict(list)
    for p in points:
        regime_groups[p["regime"]].append(p)

    for regime, group in regime_groups.items():
        ax.scatter(
            [g["x"] for g in group],
            [g["y"] for g in group],
            s=180,
            c=[g["color"] for g in group],
            marker=MARKER_MAP[regime],
            edgecolors="black",
            linewidths=0.9,
            alpha=0.95,
            zorder=3,
        )

    # ----------------------------
    # Flexion ring overlay (secondary encoding)
    # ----------------------------
    flex_points = [p for p in points if p["has_flexion"]]
    if flex_points:
        ax.scatter(
            [p["x"] for p in flex_points],
            [p["y"] for p in flex_points],
            s=340,                 # visibly larger than base markers -> halo
            marker="o",            # ring shape (consistent visual)
            facecolors="none",     # hollow
            edgecolors=RING_COLOR,
            linewidths=2.2,
            alpha=0.9,
            zorder=4,
        )

    # Axis labels/titles
    ax.set_xlabel("Mass Representation", fontsize=18, labelpad=48)
    ax.set_ylabel("Inference Strategy  (Iterative / Local  $\u2192$  Global)", fontsize=18)
    ax.set_title("Taxonomy of Cluster Lensing Reconstruction Pipelines", fontsize=22, weight="bold", pad=22)

    # Category labels beneath x-axis
    ax.text(-0.715, -1.16, "Parametric",   ha="center", va="top", fontsize=15, weight="bold")
    ax.text( 0.00,  -1.16, "Hybrid",       ha="center", va="top", fontsize=15, weight="bold")
    ax.text( 0.715, -1.16, "Field-based",  ha="center", va="top", fontsize=15, weight="bold")

    # ----------------------------
    # Labeling (stable + thesis-friendly)
    # ----------------------------
    # Label a curated set; the complete list belongs in Table 2.1.
    label_set = {
        "ARCH",
        "LENSTOOL", "GLAFIC",
        "SaWLens", "WSLAP_plus",
        "GRALE",
        "Kaiser_Squires", "Lanusse_2016",
        "Bradac_2005", "PixeLens", "Zitrin_LTM",
    }

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    for p in points:
        name = p["name"]
        if name not in label_set:
            continue

        xi, yi = p["x"], p["y"]
        xa, ya = data_to_axes(xi, yi, xmin, xmax, ymin, ymax)

        base, direction = LABEL_OVERRIDES.get(name, (0.045, None))
        ox, oy = label_offset_axes(name, base=base, direction=direction)

        xla = clamp01(xa + ox, lo=0.04, hi=0.92)
        yla = clamp01(ya + oy, lo=0.04, hi=0.92)
        xl, yl = axes_to_data(xla, yla, xmin, xmax, ymin, ymax)

        is_arch = (name == "ARCH")
        ax.annotate(
            name,
            xy=(xi, yi),
            xytext=(xl, yl),
            ha="center",
            va="center",
            fontsize=13 if is_arch else 12,
            weight="bold" if is_arch else "normal",
            bbox=dict(
                boxstyle="round,pad=0.22",
                fc="white",
                ec="black" if is_arch else "none",
                alpha=0.9,
            ),
            arrowprops=dict(arrowstyle="-", lw=0.9, color="black", alpha=0.65),
            zorder=5,
            clip_on=False,
        )

    # ----------------------------
    # Legend
    # ----------------------------
    legend_order = ["Strong", "Weak", "Flexion", "Multi", "Other"]
    handles: List[Line2D] = []
    present = {p["regime"] for p in points}

    for key in legend_order:
        if key not in present:
            continue
        handles.append(
            Line2D(
                [0], [0],
                marker=MARKER_MAP[key],
                color="w",
                label=("Multi-signal" if key == "Multi" else key),
                markerfacecolor=COLOR_MAP[key],
                markeredgecolor="black",
                markersize=12,
            )
        )

    # Flexion ring legend entry (append ONCE)
    if any(p.get("has_flexion", False) for p in points):
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label="Flexion-enabled (ring)",
                markerfacecolor="none",
                markeredgecolor=RING_COLOR,
                markeredgewidth=2.2,
                markersize=13,
            )
        )

    legend = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        fontsize=13,
        borderpad=0.8,
        handletextpad=0.8,
    )
    legend.get_frame().set_edgecolor("black")

    # Footnote pointing to the full inventory (only a curated subset is labeled)
    fig.text(
        0.99, 0.01,
        "See Table 2.1 for the complete pipeline list.",
        ha="right", va="bottom", fontsize=10, color="#595959", style="italic",
    )

    fig.savefig("taxonomy_map.pdf", bbox_inches="tight")
    fig.savefig("taxonomy_map.png", dpi=400, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
