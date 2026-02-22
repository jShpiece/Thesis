from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import arch.pipeline as pipeline
import arch.halo_obj as halo_obj
import arch.source_obj as source_obj


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
    for (x0, y0, _te) in true_lens_xyte:
        r = np.hypot(x - x0, y - y0)
        keep &= (r > rmin)
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

    theta_x = np.array([x0 + r1 * ehatx, x0 + r2 * ehatx], dtype=float)
    theta_y = np.array([y0 + r1 * ehaty, y0 + r2 * ehaty], dtype=float)

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
        chi2=np.array(getattr(lenses, "chi2", np.zeros_like(np.atleast_1d(lenses.x))), dtype=float).copy(),
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
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["initial_guess"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_positions(src, lenses, xmax, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["optimization"] = _copy_lenses_sis(lenses)

    lenses = pipeline.filter_lens_positions(src, lenses, xmax, lens_type="SIS")
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["filter"] = _copy_lenses_sis(lenses)

    lenses, _best = pipeline.forward_lens_selection(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["forward_selection"] = _copy_lenses_sis(lenses)

    merger_threshold = (len(src.x) / (2 * xmax) ** 2) ** (-0.5) if len(src.x) > 0 else 1.0
    lenses = pipeline.merge_close_lenses(lenses, merger_threshold, "SIS")
    _ = pipeline.update_chi2_values(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    stages["merging"] = _copy_lenses_sis(lenses)

    lenses = pipeline.optimize_lens_strength(src, lenses, use_flags, lens_type="SIS", use_strong_lensing=use_strong_lensing)
    _ = pipeline.update_chi2_values(src, lenses, [True, True, True], lens_type="SIS", use_strong_lensing=use_strong_lensing)
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
                for (lx, ly, lte) in zip(L.x, L.y, L.te):
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


def main():
    # ----------------------------
    # Truth: TWO SIS lenses
    # ----------------------------
    true_lens_xyte = [
        (-15.0,  0.0, 5.0),   # lens A: (x, y, te)
        ( 18.0, 12.0, 3.5),   # lens B
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
        title=f"ARCH SIS toy run",
        savepath="stages_two_lens_sl.png",
    )
    plt.show()


if __name__ == "__main__":
    main()
