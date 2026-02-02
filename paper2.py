# toy_plot_stages_sis_sl.py
# Run the toy pipeline twice (lambda_sl=0 and lambda_sl>0) and plot stages in 3x2 grids.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import arch.pipeline as pipeline
import arch.halo_obj as halo_obj
import arch.source_obj as source_obj


# ----------------------------
# 1) Toy data construction
# ----------------------------

def make_weak_lensing_catalog(
    te_true: float,
    xmax: float,
    n_sources: int,
    z_source: float = 1.0,
    sig_shear: float = 0.10,
    sig_flex: float = 0.02,
    sig_gflex: float = 0.03,
    rmin: float = 1.0,
    seed: int = 7,
) -> source_obj.Source:
    rng = np.random.default_rng(seed)

    x = rng.uniform(-xmax, xmax, size=n_sources)
    y = rng.uniform(-xmax, xmax, size=n_sources)
    r = np.hypot(x, y)
    keep = r > rmin
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

    lens_true = halo_obj.SIS_Lens(
        x=np.array([0.0]),
        y=np.array([0.0]),
        te=np.array([te_true]),
        chi2=np.array([0.0]),
    )

    src.apply_lensing(lens_true, lens_type="SIS")
    src.apply_noise()

    return src


def make_two_image_sis_system(
    system_id: str,
    te_true: float,
    beta_xy: tuple[float, float],
    sigma_theta: float = 0.05,
    z_source: float = 2.0,
):
    # NOTE: this generator guarantees consistency for an axisymmetric SIS toy lens at (0,0)
    bx, by = beta_xy
    b = float(np.hypot(bx, by))
    if not (0.0 < b < te_true):
        raise ValueError("Need 0 < |beta| < te_true for a 2-image SIS system.")

    ehatx, ehaty = bx / b, by / b
    r1 = te_true + b
    r2 = te_true - b

    theta_x = np.array([r1 * ehatx, r2 * ehatx], dtype=float)
    theta_y = np.array([r1 * ehaty, r2 * ehaty], dtype=float)

    StrongLensingSystem = getattr(source_obj, "StrongLensingSystem")
    return StrongLensingSystem(
        system_id=system_id,
        theta_x=theta_x,
        theta_y=theta_y,
        z_source=float(z_source),
        sigma_theta=float(sigma_theta),
        meta={"toy": True, "beta_true": (bx, by), "te_true": te_true},
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
    # Make a lightweight copy so later mutations don't overwrite earlier snapshots
    return halo_obj.SIS_Lens(
        x=np.array(lenses.x, dtype=float).copy(),
        y=np.array(lenses.y, dtype=float).copy(),
        te=np.array(lenses.te, dtype=float).copy(),
        chi2=np.array(getattr(lenses, "chi2", np.zeros_like(np.atleast_1d(lenses.x))), dtype=float).copy(),
    )


def run_pipeline_capture_stages(
    src: source_obj.Source,
    xmax: float,
    lambda_sl: float,
    z_lens: float = 0.5,
):
    """
    Runs the SIS pipeline through the major stages and returns a dict stage->lenses snapshot.
    Assumes you patched pipeline funcs to accept lambda_sl where relevant.
    """
    use_flags = [True, True, False]  # shear + flexion; adjust to match your conventions

    stages = {}

    lenses = pipeline.generate_initial_guess(src, lens_type="SIS", z_l=z_lens)
    stages["initial_guess"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_positions(src, lenses, xmax, use_flags, lens_type="SIS", lambda_sl=lambda_sl)
    stages["optimization"] = _copy_lenses_sis(lenses)

    lenses = pipeline.filter_lens_positions(src, lenses, xmax, lens_type="SIS")
    stages["filter"] = _copy_lenses_sis(lenses)

    lenses, _best = pipeline.forward_lens_selection(src, lenses, use_flags, lens_type="SIS", lambda_sl=lambda_sl)
    stages["forward_selection"] = _copy_lenses_sis(lenses)

    # same merge threshold heuristic used in your main driver
    merger_threshold = (len(src.x) / (2 * xmax) ** 2) ** (-0.5) if len(src.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, "SIS")
    stages["merging"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_strength(src, lenses, use_flags, lens_type="SIS", lambda_sl=lambda_sl)
    stages["opt_strength"] = _copy_lenses_sis(lenses)

    return stages


# ----------------------------
# 3) Plotting (3x2 grid)
# ----------------------------

def plot_stage_grid(
    stages: dict,
    true_xy: tuple[float, float],
    xmax: float,
    title: str,
    savepath: str | None = None,
):
    """
    3x2 grid; each panel shows:
      - true lens position (star)
      - all candidate/selected lenses at that stage (circles)
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 12), dpi=150, constrained_layout=True)
    axes = axes.ravel()

    true_x, true_y = true_xy

    for ax, stage_name in zip(axes, STAGE_NAMES):
        L = stages.get(stage_name, None)

        ax.scatter([true_x], [true_y], marker="*", s=180, label="true lens")

        if L is not None and len(np.atleast_1d(L.x)) > 0:
            ax.scatter(L.x, L.y, s=35, alpha=0.9, label="candidates")
            nL = len(np.atleast_1d(L.x))
            # Label the lenses with their Einstein radii
            # Only for final stage
            if stage_name == STAGE_NAMES[-1]:
                for (lx, ly, lte) in zip(L.x, L.y, L.te):
                    ax.text(lx, ly, f"{lte:.2f}", fontsize=8, ha="center", va="center")
        else:
            nL = 0

        ax.set_title(f"{stage_name}  (N={nL})")
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title, fontsize=14)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig


def main():
    # Truth
    te_true = 10.0
    xmax = 50.0
    n_sources = 50

    # Build WL catalog
    src = make_weak_lensing_catalog(te_true=te_true, xmax=xmax, n_sources=n_sources, seed=12)

    # Add one SL system
    sys1 = make_two_image_sis_system(
        system_id="toy_sys1",
        te_true=te_true,
        beta_xy=(0.6, 0.2),
        sigma_theta=0.03,
        z_source=2.0,
    )
    attach_strong_systems(src, [sys1])

    # Run both variants
    stages_nosl = run_pipeline_capture_stages(src, xmax=xmax, lambda_sl=0.0)
    stages_sl   = run_pipeline_capture_stages(src, xmax=xmax, lambda_sl=1.0)

    # Plot: two separate 3x2 figures (cleanest interpretation of your “3x2 grid” request)
    plot_stage_grid(
        stages_nosl, true_xy=(0.0, 0.0), xmax=xmax,
        title="ARCH SIS toy run (no strong lensing term, lambda_sl=0)",
        savepath="stages_nosl.png",
    )
    plot_stage_grid(
        stages_sl, true_xy=(0.0, 0.0), xmax=xmax,
        title="ARCH SIS toy run (with strong lensing term, lambda_sl=1)",
        savepath="stages_sl.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
