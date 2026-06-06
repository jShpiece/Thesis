"""
hyperparameter_sensitivity_A.py
===============================
Task 1, Option A: SINGLE-HALO hyperparameter-stability figure.

Inject one NFW halo (2e14 at origin), sweep tau_0, local_radius, and
fov one at a time around their defaults, and show that recovered mass,
recovered position, and halo count are stable across hyperparameter
perturbations.

This is the chapter's stability claim for ARCH at its operating point,
isolated from the substructure-deblending question.  Substructure
detection and deblending are characterised separately (Task 3); this
figure says nothing about either, by design.

Figure: 3 rows x 3 cols.
    row 0  recovered M_200            (truth 2e14)
    row 1  recovered position offset  (truth 0", recovered |r-r_true|)
    row 2  N_recovered (blue) and N_spurious (red); truth N=1

Sweeps (one hyperparameter at a time, others at default):
    tau_0         in [10^-3.5, 10^-1.5], 9 log points
    local_radius  in [5", 60"], 8 linear points
    fov           in [0.25, 0.75], 6 linear points

CLI:
    python hyperparameter_sensitivity_A.py
    python hyperparameter_sensitivity_A.py --quick
    python hyperparameter_sensitivity_A.py --plot-only
    python hyperparameter_sensitivity_A.py --n-seeds 3
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

import arch.halo_obj as halo_obj

from scripts.synthetic_mock import (
    DEFAULT_FIELD_HALF_WIDTH,
    DEFAULT_MATCH_RADIUS,
    DEFAULT_N_SOURCES,
    DEFAULT_Z_LENS,
    make_mock_catalog,
    match_recovered_to_truth,
    run_arch_pipeline,
)

# ─── Injection geometry (single halo) ──────────────────────────────────────
CORE_MASS = 2.0e14
CORE_POS = (0.0, 0.0)
INJECTED_NAMES = ["core"]

# ─── Operating-point defaults (must match the precheck) ────────────────────
DEFAULT_TAU_0 = 0.003
DEFAULT_LOCAL_RADIUS = 20.0
DEFAULT_FOV = 0.5

# ─── Sweep grids ───────────────────────────────────────────────────────────
TAU_0_GRID = np.logspace(-3.5, -1.5, 9)
LOCAL_RADIUS_GRID = np.linspace(5.0, 60.0, 8)
FOV_GRID = np.linspace(0.25, 0.75, 6)

QUICK_N_SOURCES = 400
QUICK_TAU_0_GRID = np.logspace(-3.5, -1.5, 5)
QUICK_LOCAL_RADIUS_GRID = np.linspace(5.0, 60.0, 4)
QUICK_FOV_GRID = np.linspace(0.25, 0.75, 3)


# ─── Single-halo construction ──────────────────────────────────────────────

def make_single_core(z_lens: float = DEFAULT_Z_LENS) -> halo_obj.NFW_Lens:
    """Single 2e14 NFW core at origin; concentration from c(M)."""
    halos = halo_obj.NFW_Lens(
        x=np.array([CORE_POS[0]]),
        y=np.array([CORE_POS[1]]),
        z=np.array([0.0]),
        concentration=np.array([5.0]),
        mass=np.array([CORE_MASS]),
        redshift=z_lens,
        chi2=np.array([0.0]),
    )
    halos.calculate_concentration()
    return halos


# ─── Single grid-point run ─────────────────────────────────────────────────

def run_one_grid_point(
    tau_0: float, local_radius: float, fov: float, seed: int,
    n_sources: int = DEFAULT_N_SOURCES,
    field_half_width: float = DEFAULT_FIELD_HALF_WIDTH,
    z_lens: float = DEFAULT_Z_LENS,
    match_radius: float = DEFAULT_MATCH_RADIUS,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "tau_0": float(tau_0), "local_radius": float(local_radius), "fov": float(fov),
        "seed": int(seed),
        "core_mass": float("nan"), "core_distance": float("nan"),
        "n_recovered": 0, "n_matched": 0, "n_spurious": 0,
        "spurious_masses": [], "spurious_distances": [],
        "reduced_chi2": float("nan"),
        "elapsed_seconds": float("nan"), "error": None,
    }
    t0 = time.time()
    try:
        injected = make_single_core(z_lens=z_lens)
        src = make_mock_catalog(
            seed=seed, halos=injected, n_sources=n_sources,
            field_half_width=field_half_width,
        )
        lenses, rchi2 = run_arch_pipeline(
            src, field_half_width=field_half_width, z_lens=z_lens,
            tau_0=tau_0, local_radius=local_radius, fov=fov, verbose=False,
        )
        result = match_recovered_to_truth(
            lenses, injected, match_radius=match_radius,
            injected_names=INJECTED_NAMES,
        )
        # Compute distances for spurious halos (distance from the core injection)
        spur_dists = []
        claimed = set(j for j in result.matched_indices.values() if j >= 0)
        for j in range(len(lenses.x)):
            if j in claimed:
                continue
            d = float(np.hypot(lenses.x[j] - CORE_POS[0],
                               lenses.y[j] - CORE_POS[1]))
            spur_dists.append(d)
        record.update({
            "core_mass": result.masses["core"],
            "core_distance": result.distances["core"],
            "n_recovered": result.n_recovered,
            "n_matched": result.n_matched,
            "n_spurious": result.n_spurious,
            "spurious_masses": list(result.spurious_masses),
            "spurious_distances": spur_dists,
            "reduced_chi2": float(rchi2) if np.isfinite(rchi2) else float("nan"),
        })
    except Exception:
        record["error"] = traceback.format_exc()
    record["elapsed_seconds"] = time.time() - t0
    return record


# ─── Sweeps ────────────────────────────────────────────────────────────────

def run_sweep_axis(axis_name, values, seeds, n_sources, verbose=True):
    if axis_name not in ("tau_0", "local_radius", "fov"):
        raise ValueError(f"unknown axis: {axis_name}")
    records, total = [], len(values) * len(seeds)
    t_axis = time.time()
    for i, v in enumerate(values):
        for j, seed in enumerate(seeds):
            tau_0 = float(v) if axis_name == "tau_0" else DEFAULT_TAU_0
            lr = float(v) if axis_name == "local_radius" else DEFAULT_LOCAL_RADIUS
            fov = float(v) if axis_name == "fov" else DEFAULT_FOV
            n = i * len(seeds) + j + 1
            if verbose:
                print(f"  [{axis_name}] {n:3d}/{total:3d}  "
                      f"{axis_name}={float(v):.4g}  seed={seed} ...",
                      end="", flush=True)
            rec = run_one_grid_point(tau_0, lr, fov, seed, n_sources=n_sources)
            records.append(rec)
            if verbose:
                if rec["error"] is None:
                    print(f" {rec['elapsed_seconds']:5.1f}s  "
                          f"N={rec['n_recovered']} spur={rec['n_spurious']}  "
                          f"core_M={rec['core_mass']:.2e}  "
                          f"d_core={rec['core_distance']:.2f}")
                else:
                    print(f" ERROR ({rec['elapsed_seconds']:.1f}s)")
    if verbose:
        el = time.time() - t_axis
        print(f"  [{axis_name}] done in {el:.1f}s ({el/60:.1f} min)")
    return records


def run_all_sweeps(seeds, n_sources, quick, out_path, verbose=True):
    tau_grid = QUICK_TAU_0_GRID if quick else TAU_0_GRID
    lr_grid = QUICK_LOCAL_RADIUS_GRID if quick else LOCAL_RADIUS_GRID
    fov_grid = QUICK_FOV_GRID if quick else FOV_GRID

    injected = make_single_core()
    results: Dict[str, Any] = {
        "metadata": {
            "option": "A",
            "description": (
                "Single-halo hyperparameter-stability test.  One 2e14 NFW "
                "core at the field centre, no substructure injected.  The "
                "figure tests whether recovered mass, position, and halo "
                "count are stable against perturbations of tau_0, "
                "local_radius, and fov around their defaults.  Substructure "
                "detection and deblending are characterised separately."
            ),
            "n_seeds": len(seeds), "seeds": list(seeds),
            "z_lens": DEFAULT_Z_LENS,
            "n_sources": int(n_sources),
            "field_half_width": float(DEFAULT_FIELD_HALF_WIDTH),
            "match_radius": float(DEFAULT_MATCH_RADIUS), "quick": bool(quick),
            "injected": {
                "core": {
                    "x": CORE_POS[0], "y": CORE_POS[1], "mass": CORE_MASS,
                    "concentration": float(injected.concentration[0]),
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

    for axis_name, grid in [
        ("tau_0", tau_grid),
        ("local_radius", lr_grid),
        ("fov", fov_grid),
    ]:
        if verbose:
            print(f"\n{'-'*64}\n  Sweeping {axis_name} "
                  f"({len(grid)} values x {len(seeds)} seeds)\n{'-'*64}")
        recs = run_sweep_axis(axis_name, grid, seeds, n_sources, verbose=verbose)
        results["sweeps"][axis_name] = {"values": grid.tolist(), "records": recs}
        save_results(results, out_path)
        if verbose:
            print(f"  Saved incremental results to {out_path}")
    return results


# ─── JSON I/O ──────────────────────────────────────────────────────────────

def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    return obj


def save_results(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(results), f, indent=2)


def load_results(path: Path):
    with path.open("r") as f:
        raw = json.load(f)
    out = dict(raw)
    if "sweeps" in raw:
        out["sweeps"] = {
            axis: {"values": v["values"], "records": [
                {k: (float("nan") if (k != "error" and val is None) else val)
                 for k, val in rec.items()} for rec in v["records"]
            ]} for axis, v in raw["sweeps"].items()
        }
    return out


# ─── Plot helpers ──────────────────────────────────────────────────────────

def _collect_per_value(records, values, axis, field):
    by_value = {float(v): [] for v in values}
    for r in records:
        v = float(r[axis])
        idx = int(np.argmin([abs(v - float(g)) for g in values]))
        key = float(values[idx])
        val = r.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            by_value[key].append(float("nan"))
        else:
            by_value[key].append(float(val))
    max_len = max((len(by_value[float(v)]) for v in values), default=0)
    rows = []
    for v in values:
        col = list(by_value[float(v)])
        while len(col) < max_len:
            col.append(float("nan"))
        rows.append(col)
    return np.asarray(rows, dtype=float)


def _plot_line(ax, x, y, *, color, label=None, marker="o"):
    if y.shape[1] == 1:
        ax.plot(x, y[:, 0], marker=marker, color=color, label=label, lw=1.5, ms=5)
    else:
        med = np.nanmedian(y, axis=1)
        lo = np.nanpercentile(y, 16, axis=1)
        hi = np.nanpercentile(y, 84, axis=1)
        ax.fill_between(x, lo, hi, alpha=0.25, color=color, lw=0)
        ax.plot(x, med, marker=marker, color=color, label=label, lw=1.5, ms=5)


# ─── Figure ────────────────────────────────────────────────────────────────

def make_figure(results) -> plt.Figure:
    meta, sweeps = results["metadata"], results["sweeps"]
    core_truth = meta["injected"]["core"]["mass"]
    defaults = meta["defaults"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9.5), sharex="col")
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
        core_dist = _collect_per_value(records, values.tolist(), axis_name, "core_distance")
        n_rec = _collect_per_value(records, values.tolist(), axis_name, "n_recovered")
        n_spur = _collect_per_value(records, values.tolist(), axis_name, "n_spurious")
        default_x = defaults[axis_name]

        # Row 0: recovered core mass
        ax = axes[0, col]
        _plot_line(ax, values, core_mass, color="C0", label="recovered")
        ax.axhline(core_truth, color="k", lw=1.0, label="truth")
        ax.axvline(default_x, color="grey", ls="--", lw=1.0, label="default")
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"core $M_{200}$ [$M_\odot$]")
        ax.set_title(xlabel)

        # Row 1: recovered core position offset
        ax = axes[1, col]
        _plot_line(ax, values, core_dist, color="C2", label="recovered offset")
        ax.axhline(0.0, color="k", lw=1.0)
        ax.axvline(default_x, color="grey", ls="--", lw=1.0)
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"position offset [arcsec]")

        # Row 2: halo counts
        ax = axes[2, col]
        _plot_line(ax, values, n_rec, color="C0", label=r"$N_\mathrm{recovered}$")
        _plot_line(ax, values, n_spur, color="C3", label=r"$N_\mathrm{spurious}$",
                   marker="s")
        ax.axhline(1, color="k", lw=1.0, label=r"truth $N=1$")
        ax.axvline(default_x, color="grey", ls="--", lw=1.0)
        if log_x:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"$N_\mathrm{halos}$")
        ax.set_xlabel(xlabel)
        ymax = np.nanmax([np.nanmax(n_rec), np.nanmax(n_spur), 1.0])
        if np.isfinite(ymax):
            ax.set_ylim(-0.5, ymax + 1.0)

    # Legend (combine unique labels from rows 0 and 2)
    seen, ch, cl = set(), [], []
    for handles, labels in (axes[0, 0].get_legend_handles_labels(),
                            axes[1, 0].get_legend_handles_labels(),
                            axes[2, 0].get_legend_handles_labels()):
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                ch.append(h)
                cl.append(l)
    fig.legend(ch, cl, loc="upper center", ncol=len(cl),
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(
        f"Hyperparameter stability (Option A): single {core_truth:.1e} "
        f"$M_\\odot$ NFW core, "
        f"n_seeds={meta['n_seeds']}, n_src={meta['n_sources']}",
        y=1.07,
    )
    fig.text(
        0.5, -0.02,
        "Single-halo test isolating stability of mass and position recovery; "
        "substructure detection is characterised separately.",
        ha="center", va="top", fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout()
    return fig


# ─── Summary ────────────────────────────────────────────────────────────────

def print_default_summary(results):
    meta, defaults = results["metadata"], results["metadata"]["defaults"]
    print(f"\n{'-'*64}\n  Behaviour at default hyperparameters (per axis)\n{'-'*64}")
    for axis_name in ("tau_0", "local_radius", "fov"):
        sweep = results["sweeps"][axis_name]
        values = sweep["values"]
        idx = int(np.argmin([abs(float(v) - defaults[axis_name]) for v in values]))
        v = values[idx]
        recs = [r for r in sweep["records"]
                if abs(float(r[axis_name]) - float(v)) < 1e-12]
        if not recs:
            continue
        core = np.array([r["core_mass"] for r in recs], dtype=float)
        dist = np.array([r["core_distance"] for r in recs], dtype=float)
        n = np.array([r["n_recovered"] for r in recs], dtype=float)
        spur = np.array([r["n_spurious"] for r in recs], dtype=float)
        bias_med = np.nanmedian(core) / meta["injected"]["core"]["mass"]
        print(f"  {axis_name} = {float(v):.4g}: "
              f"M={np.nanmedian(core):.3e} (bias {bias_med:.2f}x)  "
              f"d={np.nanmedian(dist):.2f}\"  "
              f"N={np.nanmedian(n):.1f}  spur={np.nanmedian(spur):.1f}")


# ─── Main ──────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", type=Path, default=Path("."))
    p.add_argument("--results-name", default="hyperparameter_sensitivity_A.json")
    p.add_argument("--figure-name", default="hyperparameter_sensitivity_A.pdf")
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    try:
        plt.style.use("scientific_presentation.mplstyle")
    except (OSError, FileNotFoundError):
        pass

    results_path = args.output_dir / "results" / args.results_name
    figure_path = args.output_dir / "figures" / args.figure_name

    if args.plot_only:
        if not results_path.exists():
            print(f"--plot-only but {results_path} missing.", file=sys.stderr)
            return 2
        print(f"Loading cached results from {results_path}")
        results = load_results(results_path)
    elif results_path.exists() and not args.force:
        print(f"Cached results exist at {results_path}. "
              f"Use --force or --plot-only.")
        results = load_results(results_path)
    else:
        seeds = list(range(args.n_seeds))
        n_sources = QUICK_N_SOURCES if args.quick else DEFAULT_N_SOURCES
        print(f"Option A sweep: single {CORE_MASS:.1e} M_sun core, "
              f"n_seeds={args.n_seeds}, n_sources={n_sources}, "
              f"quick={args.quick}")
        t0 = time.time()
        results = run_all_sweeps(
            seeds=seeds, n_sources=n_sources, quick=args.quick,
            out_path=results_path, verbose=not args.quiet,
        )
        print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min")

    try:
        print_default_summary(results)
    except Exception as e:
        print(f"Could not print summary: {e}")

    print(f"\nGenerating figure -> {figure_path}")
    fig = make_figure(results)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())