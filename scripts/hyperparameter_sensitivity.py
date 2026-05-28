"""
hyperparameter_sensitivity.py
=============================
Task 1 of the ARCH validation chapter.

Sweeps three pipeline hyperparameters one at a time on the default
two-halo synthetic mock (2e14 core + 3e13 sub at 45" offset), holding
the other two at their defaults, and produces:

    figures/hyperparameter_sensitivity.pdf
    results/hyperparameter_sensitivity.json

The figure has three rows:

    Row 1 — core M_200 vs hyperparameter (truth = 2e14)
    Row 2 — sub  M_200 vs hyperparameter (truth = 3e13)
    Row 3 — N_recovered (blue) and N_spurious (red) overlay
            (truth N_injected = 2)

The plotting axis convention follows the chapter:
    tau_0 column         log x-axis,   default 0.003
    local_radius column  linear x-axis, default 20"
    fov column           linear x-axis, default 0.5 (see note below)

A horizontal "truth" line (solid black) and vertical "default" line
(dashed grey) sit on every panel so bias (offset from truth) and
stability (scatter across the grid) are both visible at a glance.

Important note on fov
---------------------
The chapter's fov is the geometric cut applied to candidate halos in
units of the field half-width W (so fov=0.5 means drop lenses beyond
0.5 W from origin).  In the code, ``filter_lens_positions`` cuts at
``xmax_filter * 1.5``, so the wrapper passes
``xmax_filter = fov * W / 1.5``.

With the production default ``xmax_filter = W``, the effective cut
sits at fov=1.5 — i.e. essentially "no cut".  The chapter's
[0.25, 0.75] sweep therefore investigates whether a TIGHTER cut would
be a more defensible production default, rather than bracketing the
existing default.  The vertical "default" line in the fov panel is
drawn at fov=0.5 to mark the proposed chapter default, NOT the
current production value of 1.5.  If the figure caption needs to
match production, set ``DEFAULT_FOV = 1.5`` here and adjust the
sweep range accordingly.

CLI
---
Run the full sweep, save JSON + figure:
    python hyperparameter_sensitivity.py

Re-plot from a previously cached JSON:
    python hyperparameter_sensitivity.py --plot-only

Quick iteration (300 sources, default seeds, default ranges) for
plot development:
    python hyperparameter_sensitivity.py --quick

Multi-seed for tighter curves (slower):
    python hyperparameter_sensitivity.py --n-seeds 3

Force re-run even if JSON exists:
    python hyperparameter_sensitivity.py --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from scripts.synthetic_mock import (
    DEFAULT_FIELD_HALF_WIDTH,
    DEFAULT_MATCH_RADIUS,
    DEFAULT_N_SOURCES,
    DEFAULT_Z_LENS,
    make_default_halos,
    make_mock_catalog,
    match_recovered_to_truth,
    run_arch_pipeline,
)


# ─── Sweep configuration ───────────────────────────────────────────────────

# Hyperparameter defaults held fixed when sweeping the others.
DEFAULT_TAU_0 = 0.003
DEFAULT_LOCAL_RADIUS = 20.0
DEFAULT_FOV = 0.5  # *Proposed chapter default*; production = 1.5 (no cut). See header.

# Sweep grids.
TAU_0_GRID = np.logspace(-3.5, -1.5, 9)           # 9 logarithmic points
LOCAL_RADIUS_GRID = np.linspace(5.0, 60.0, 8)     # 8 linear points
FOV_GRID = np.linspace(0.25, 0.75, 6)             # 6 linear points

# Quick mode (for plot-development iteration).
QUICK_N_SOURCES = 300
QUICK_TAU_0_GRID = np.logspace(-3.5, -1.5, 5)
QUICK_LOCAL_RADIUS_GRID = np.linspace(5.0, 60.0, 4)
QUICK_FOV_GRID = np.linspace(0.25, 0.75, 3)

# Seeds for multi-seed mode.  Single-seed default = [0].
DEFAULT_SEEDS = [0]


# ─── Single grid-point run ─────────────────────────────────────────────────

def run_one_grid_point(
    tau_0: float,
    local_radius: float,
    fov: float,
    seed: int,
    n_sources: int = DEFAULT_N_SOURCES,
    field_half_width: float = DEFAULT_FIELD_HALF_WIDTH,
    z_lens: float = DEFAULT_Z_LENS,
    match_radius: float = DEFAULT_MATCH_RADIUS,
) -> Dict[str, Any]:
    """
    Build a default two-halo mock at the given seed, run the wrapper with
    the supplied hyperparameters, match the result to truth, and return a
    flat record dict.

    On exception, returns a record with all output fields set to NaN /
    empty plus an ``error`` field carrying the traceback.
    """
    record: Dict[str, Any] = {
        "tau_0": float(tau_0),
        "local_radius": float(local_radius),
        "fov": float(fov),
        "seed": int(seed),
        "core_mass": float("nan"),
        "sub_mass": float("nan"),
        "core_distance": float("nan"),
        "sub_distance": float("nan"),
        "n_recovered": 0,
        "n_matched": 0,
        "n_spurious": 0,
        "spurious_masses": [],
        "reduced_chi2": float("nan"),
        "elapsed_seconds": float("nan"),
        "error": None,
    }
    t0 = time.time()
    try:
        injected = make_default_halos(z_lens=z_lens)
        src = make_mock_catalog(
            seed=seed, halos=injected, n_sources=n_sources,
            field_half_width=field_half_width,
        )
        lenses, rchi2 = run_arch_pipeline(
            src, field_half_width=field_half_width, z_lens=z_lens,
            tau_0=tau_0, local_radius=local_radius, fov=fov,
            verbose=False,
        )
        result = match_recovered_to_truth(
            lenses, injected, match_radius=match_radius,
            injected_names=["core", "sub"],
        )
        record.update({
            "core_mass": result.masses["core"],
            "sub_mass": result.masses["sub"],
            "core_distance": result.distances["core"],
            "sub_distance": result.distances["sub"],
            "n_recovered": result.n_recovered,
            "n_matched": result.n_matched,
            "n_spurious": result.n_spurious,
            "spurious_masses": list(result.spurious_masses),
            "reduced_chi2": float(rchi2) if np.isfinite(rchi2) else float("nan"),
        })
    except Exception:
        record["error"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - t0
    return record


# ─── Sweeps ────────────────────────────────────────────────────────────────

def run_sweep_axis(
    axis_name: str,
    values: np.ndarray,
    seeds: List[int],
    n_sources: int,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Sweep one hyperparameter axis.  At each grid point, run ``len(seeds)``
    pipeline calls with that value substituted into the appropriate slot
    (tau_0, local_radius, or fov) while the other two stay at their
    defaults.

    Returns a flat list of per-(value, seed) record dicts.
    """
    if axis_name not in ("tau_0", "local_radius", "fov"):
        raise ValueError(f"unknown axis: {axis_name}")

    records: List[Dict[str, Any]] = []
    total = len(values) * len(seeds)
    t_axis_start = time.time()

    for i, v in enumerate(values):
        for j, seed in enumerate(seeds):
            tau_0 = float(v) if axis_name == "tau_0" else DEFAULT_TAU_0
            lr = float(v) if axis_name == "local_radius" else DEFAULT_LOCAL_RADIUS
            fov = float(v) if axis_name == "fov" else DEFAULT_FOV

            point_idx = i * len(seeds) + j + 1
            if verbose:
                print(
                    f"  [{axis_name}] point {point_idx:3d}/{total:3d}  "
                    f"{axis_name}={float(v):.4g}  seed={seed} ...",
                    end="", flush=True,
                )
            record = run_one_grid_point(
                tau_0=tau_0, local_radius=lr, fov=fov,
                seed=seed, n_sources=n_sources,
            )
            records.append(record)
            if verbose:
                if record["error"] is None:
                    print(
                        f" done ({record['elapsed_seconds']:5.1f}s)  "
                        f"N={record['n_recovered']}  "
                        f"spur={record['n_spurious']}  "
                        f"core_M={record['core_mass']:.2e}  "
                        f"sub_M={record['sub_mass']:.2e}"
                    )
                else:
                    print(f" ERROR ({record['elapsed_seconds']:.1f}s)")

    if verbose:
        elapsed = time.time() - t_axis_start
        print(f"  [{axis_name}] axis complete in {elapsed:.1f}s "
              f"({elapsed / 60.0:.1f} min)")
    return records


def run_all_sweeps(
    seeds: List[int],
    n_sources: int,
    quick: bool,
    out_path: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all three axis sweeps and assemble the JSON-serialisable results
    dict.  Saves the JSON incrementally after each axis completes so a
    crash in (say) the fov sweep doesn't cost the tau_0 results.
    """
    tau_grid = QUICK_TAU_0_GRID if quick else TAU_0_GRID
    lr_grid = QUICK_LOCAL_RADIUS_GRID if quick else LOCAL_RADIUS_GRID
    fov_grid = QUICK_FOV_GRID if quick else FOV_GRID

    injected = make_default_halos()
    results: Dict[str, Any] = {
        "metadata": {
            "n_seeds": len(seeds),
            "seeds": list(seeds),
            "z_lens": DEFAULT_Z_LENS,
            "n_sources": int(n_sources),
            "field_half_width": float(DEFAULT_FIELD_HALF_WIDTH),
            "match_radius": float(DEFAULT_MATCH_RADIUS),
            "quick": bool(quick),
            "injected": {
                "core": {
                    "x": float(injected.x[0]),
                    "y": float(injected.y[0]),
                    "mass": float(injected.mass[0]),
                    "concentration": float(injected.concentration[0]),
                },
                "sub": {
                    "x": float(injected.x[1]),
                    "y": float(injected.y[1]),
                    "mass": float(injected.mass[1]),
                    "concentration": float(injected.concentration[1]),
                },
            },
            "defaults": {
                "tau_0": DEFAULT_TAU_0,
                "local_radius": DEFAULT_LOCAL_RADIUS,
                "fov": DEFAULT_FOV,
            },
            "grids": {
                "tau_0": tau_grid.tolist(),
                "local_radius": lr_grid.tolist(),
                "fov": fov_grid.tolist(),
            },
        },
        "sweeps": {},
    }

    axes = [
        ("tau_0", tau_grid),
        ("local_radius", lr_grid),
        ("fov", fov_grid),
    ]
    for axis_name, grid in axes:
        if verbose:
            print()
            print("─" * 64)
            print(f"  Sweeping {axis_name}  ({len(grid)} values × {len(seeds)} seeds)")
            print("─" * 64)
        recs = run_sweep_axis(
            axis_name, grid, seeds, n_sources=n_sources, verbose=verbose,
        )
        results["sweeps"][axis_name] = {
            "values": grid.tolist(),
            "records": recs,
        }
        # Incremental save after each axis
        save_results(results, out_path)
        if verbose:
            print(f"  Saved incremental results to {out_path}")

    return results


# ─── JSON I/O ──────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    """Recursively replace NaN with None so the JSON is strictly valid."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def save_results(results: Dict[str, Any], path: Path) -> None:
    """Write the results dict to JSON, replacing NaN with null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(results), f, indent=2)


def load_results(path: Path) -> Dict[str, Any]:
    """Load results JSON.  ``None`` (null) values are returned as NaN floats."""
    with path.open("r") as f:
        raw = json.load(f)

    def _restore(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _restore(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_restore(x) for x in obj]
        if obj is None:
            return float("nan")
        return obj

    # Only restore NaN inside the records — leave the metadata's None-free
    # structure alone.
    out = dict(raw)
    if "sweeps" in raw:
        out["sweeps"] = {
            axis: {
                "values": v["values"],
                "records": [
                    {
                        k: (float("nan") if (k != "error" and val is None) else val)
                        for k, val in rec.items()
                    }
                    for rec in v["records"]
                ],
            }
            for axis, v in raw["sweeps"].items()
        }
    return out


# ─── Plot helpers ──────────────────────────────────────────────────────────

def _collect_per_value(
    records: List[Dict[str, Any]], values: List[float], axis: str, field: str,
) -> np.ndarray:
    """
    Return an (n_values, n_seeds) array of ``field`` for the supplied
    records, where records are grouped by their value of ``axis``.

    Missing values become NaN.  Assumes records contain every (value, seed)
    combination of the sweep.
    """
    by_value: Dict[float, List[float]] = {float(v): [] for v in values}
    for r in records:
        v = float(r[axis])
        # Find the closest value in the grid (floating-point tolerant)
        idx = int(np.argmin([abs(v - float(g)) for g in values]))
        key = float(values[idx])
        val = r.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            by_value[key].append(float("nan"))
        else:
            by_value[key].append(float(val))
    # Pad columns to the same length
    max_len = max(len(by_value[float(v)]) for v in values) if values else 0
    rows = []
    for v in values:
        col = list(by_value[float(v)])
        while len(col) < max_len:
            col.append(float("nan"))
        rows.append(col)
    return np.asarray(rows, dtype=float)


def _plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    label: Optional[str] = None,
    marker: str = "o",
) -> None:
    """Plot per-seed array y (shape n_x × n_seeds) as median + IQR band,
    or as a simple line for single-seed."""
    if y.shape[1] == 1:
        ax.plot(x, y[:, 0], marker=marker, color=color, label=label, lw=1.5, ms=5)
    else:
        med = np.nanmedian(y, axis=1)
        lo = np.nanpercentile(y, 16, axis=1)
        hi = np.nanpercentile(y, 84, axis=1)
        ax.fill_between(x, lo, hi, alpha=0.25, color=color, lw=0)
        ax.plot(x, med, marker=marker, color=color, label=label, lw=1.5, ms=5)


# ─── Figure ────────────────────────────────────────────────────────────────

def make_figure(results: Dict[str, Any]) -> plt.Figure:
    """
    Three-row × three-column hyperparameter sensitivity figure.

    Rows:
        0 — core M_200 (truth line at 2e14)
        1 — sub  M_200 (truth line at 3e13)
        2 — N_recovered (blue) and N_spurious (red), truth N=2

    Columns:
        0 — tau_0          (log x-axis)
        1 — local_radius   (linear x-axis)
        2 — fov            (linear x-axis)
    """
    meta = results["metadata"]
    sweeps = results["sweeps"]

    core_truth = meta["injected"]["core"]["mass"]
    sub_truth = meta["injected"]["sub"]["mass"]
    defaults = meta["defaults"]

    fig, axes = plt.subplots(
        3, 3, figsize=(13, 9),
        sharex="col",
    )

    axis_specs = [
        ("tau_0", r"$\tau_0$", True),
        ("local_radius", r"local radius [arcsec]", False),
        ("fov", r"fov [units of $W$]", False),
    ]

    for col, (axis_name, xlabel, log_x) in enumerate(axis_specs):
        sweep = sweeps[axis_name]
        values = np.asarray(sweep["values"], dtype=float)
        records = sweep["records"]

        core_mass = _collect_per_value(records, values.tolist(), axis_name, "core_mass")
        sub_mass = _collect_per_value(records, values.tolist(), axis_name, "sub_mass")
        n_rec = _collect_per_value(records, values.tolist(), axis_name, "n_recovered")
        n_spur = _collect_per_value(records, values.tolist(), axis_name, "n_spurious")

        default_x = defaults[axis_name]

        # ── Row 0: core mass ──
        ax = axes[0, col]
        _plot_line(ax, values, core_mass, color="C0", label="recovered")
        ax.axhline(core_truth, color="k", lw=1.0, label="truth")
        ax.axvline(default_x, color="grey", ls="--", lw=1.0, label="default")
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"core $M_{200}$ [$M_\odot$]")
        ax.set_title(xlabel)

        # ── Row 1: sub mass ──
        ax = axes[1, col]
        _plot_line(ax, values, sub_mass, color="C1", label="recovered")
        ax.axhline(sub_truth, color="k", lw=1.0, label="truth")
        ax.axvline(default_x, color="grey", ls="--", lw=1.0, label="default")
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"sub $M_{200}$ [$M_\odot$]")

        # ── Row 2: halo counts ──
        ax = axes[2, col]
        _plot_line(ax, values, n_rec, color="C0", label=r"$N_\mathrm{recovered}$")
        _plot_line(ax, values, n_spur, color="C3", label=r"$N_\mathrm{spurious}$",
                   marker="s")
        ax.axhline(2, color="k", lw=1.0, label=r"truth $N=2$")
        ax.axvline(default_x, color="grey", ls="--", lw=1.0)
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"$N_\mathrm{halos}$")
        ax.set_xlabel(xlabel)
        # Integer y-axis for halo counts
        ymax = np.nanmax([np.nanmax(n_rec), np.nanmax(n_spur), 2.0])
        if np.isfinite(ymax):
            ax.set_ylim(-0.5, ymax + 1.0)

    # Single legend in the top-right corner of the figure
    handles_top = axes[0, 0].get_legend_handles_labels()
    handles_bot = axes[2, 0].get_legend_handles_labels()
    # Combine without duplicates
    seen = set()
    combined_handles = []
    combined_labels = []
    for handles, labels in (handles_top, handles_bot):
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            combined_handles.append(h)
            combined_labels.append(l)
    fig.legend(
        combined_handles, combined_labels,
        loc="upper center", ncol=len(combined_labels),
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )

    fig.suptitle(
        f"Hyperparameter sensitivity  "
        f"(n_seeds = {meta['n_seeds']}, "
        f"n_sources = {meta['n_sources']}, "
        f"injected: core {core_truth:.1e} + sub {sub_truth:.1e} $M_\\odot$)",
        y=1.06,
    )
    fig.tight_layout()
    return fig


# ─── Summary printer ───────────────────────────────────────────────────────

def print_default_summary(results: Dict[str, Any]) -> None:
    """
    Walk each axis sweep and pull out the grid point closest to its default,
    printing recovered (core, sub, N, spurious, chi2) — a 30-second sanity
    check that the defaults sit in a stable region.
    """
    meta = results["metadata"]
    defaults = meta["defaults"]
    print()
    print("─" * 64)
    print("  Pipeline behaviour at default hyperparameters (per axis)")
    print("─" * 64)
    for axis_name in ("tau_0", "local_radius", "fov"):
        sweep = results["sweeps"][axis_name]
        values = sweep["values"]
        idx = int(np.argmin([abs(float(v) - defaults[axis_name]) for v in values]))
        v = values[idx]
        # Average per-seed at that grid point
        recs_here = [r for r in sweep["records"] if abs(float(r[axis_name]) - float(v)) < 1e-12]
        if not recs_here:
            print(f"  {axis_name}: no records at v={v}")
            continue
        core_arr = np.array([r["core_mass"] for r in recs_here], dtype=float)
        sub_arr = np.array([r["sub_mass"] for r in recs_here], dtype=float)
        n_arr = np.array([r["n_recovered"] for r in recs_here], dtype=float)
        spur_arr = np.array([r["n_spurious"] for r in recs_here], dtype=float)
        chi2_arr = np.array([r["reduced_chi2"] for r in recs_here], dtype=float)
        print(f"  {axis_name} = {float(v):.4g} (default = {defaults[axis_name]:.4g}):")
        print(f"     core M  = {np.nanmedian(core_arr):.3e}  "
              f"(truth = {meta['injected']['core']['mass']:.2e})")
        print(f"     sub  M  = {np.nanmedian(sub_arr):.3e}  "
              f"(truth = {meta['injected']['sub']['mass']:.2e})")
        print(f"     N_halos = {np.nanmedian(n_arr):.1f}  "
              f"spurious = {np.nanmedian(spur_arr):.1f}  "
              f"chi2_nu = {np.nanmedian(chi2_arr):.3f}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("."),
        help="Directory containing results/ and figures/ subdirs. "
             "Default: current directory.",
    )
    parser.add_argument(
        "--results-name", default="hyperparameter_sensitivity.json",
        help="Filename for the JSON cache (inside <output-dir>/results/).",
    )
    parser.add_argument(
        "--figure-name", default="hyperparameter_sensitivity.pdf",
        help="Filename for the figure (inside <output-dir>/figures/).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Number of seeds per grid point.  Default 1 (matches the "
             "chapter's 23-run budget).  3 gives much smoother curves at "
             "3x the wall time.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run the sweeps even if the JSON cache already exists.",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip the sweeps, load existing JSON, regenerate the figure.",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Use a much smaller grid and 300 sources for fast iteration. "
             "Use this when developing the plotting code, NOT for "
             "publishable results.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-grid-point progress output.",
    )
    args = parser.parse_args(argv)

    # Try the project's matplotlib style; fall back gracefully.
    try:
        plt.style.use("scientific_presentation.mplstyle")
    except (OSError, FileNotFoundError):
        pass

    results_path = args.output_dir / "results" / args.results_name
    figure_path = args.output_dir / "figures" / args.figure_name

    # ── Decide: run, or load existing? ─────────────────────────────
    if args.plot_only:
        if not results_path.exists():
            print(f"--plot-only requested but {results_path} does not exist.",
                  file=sys.stderr)
            return 2
        print(f"Loading cached results from {results_path}")
        results = load_results(results_path)
    elif results_path.exists() and not args.force:
        print(f"Cached results already exist at {results_path}.")
        print(f"Use --force to overwrite, or --plot-only to regenerate the figure.")
        results = load_results(results_path)
    else:
        seeds = list(range(args.n_seeds))
        n_sources = QUICK_N_SOURCES if args.quick else DEFAULT_N_SOURCES
        t_start = time.time()
        print(f"Starting full hyperparameter sweep "
              f"(n_seeds={args.n_seeds}, n_sources={n_sources}, "
              f"quick={args.quick})")
        results = run_all_sweeps(
            seeds=seeds, n_sources=n_sources, quick=args.quick,
            out_path=results_path, verbose=not args.quiet,
        )
        elapsed = time.time() - t_start
        print()
        print(f"Total wall time: {elapsed / 60.0:.1f} min")

    # ── Summary table ──────────────────────────────────────────────
    try:
        print_default_summary(results)
    except Exception as e:
        print(f"Could not print summary: {e}")

    # ── Figure ─────────────────────────────────────────────────────
    print()
    print(f"Generating figure -> {figure_path}")
    fig = make_figure(results)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())