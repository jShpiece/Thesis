"""
power_law_synthetic_tests.py
=============================
Synthetic-field tests and Monte Carlo validation for the ARCH POWER_LAW
extension.  Mirrors the structure of tests_NFW.py:

    1. build_standardized_field  - construct a controlled synthetic field
    2. plot_results              - visualize a single recovered fit vs truth
    3. pipeline_breakdown        - run the pipeline step-by-step and plot
                                   the result of each stage in a 2x3 grid
                                   (the gold-standard diagnostic plot)
    4. run_random_realizations   - run N trials with different noise seeds
                                   and collect recovered (x, y, kappa_star, n)
    5. plot_random_realizations  - histograms of position, kappa_star, n
                                   recovery across the MC

CLI:

    python power_law_synthetic_tests.py
        --mode   single | breakdown | mc
        --nlens  1 | 2 | 3
        --nsrc   100
        --xmax   150
        --kappa-star  0.25
        --slope       0.7
        --z-l         0.308
        --noise       (flag)
        --ntrials     500          (mc only)
        --seed        42
        --outdir      Output/POWER_LAW_tests

Examples:

    # Quick visual smoke test on a noisy 2-halo field
    python power_law_synthetic_tests.py --mode single --nlens 2 --noise

    # Full pipeline breakdown plot for a 1-halo field
    python power_law_synthetic_tests.py --mode breakdown --nlens 1 --noise

    # MC over 200 realizations of a single halo at default mass
    python power_law_synthetic_tests.py --mode mc --ntrials 200 --noise
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import arch.halo_obj as halo_obj
import arch.utils as utils
import arch.metric as metric
import arch.pipeline as pipeline
import arch.source_obj as source_obj
import arch.main as main


# ============================================================
# Defaults  (matching arch convention)
# ============================================================

DEFAULT_THETA_STAR = 30.0
DEFAULT_Z_L = 0.308
DEFAULT_Z_S = 1.5
DEFAULT_KAPPA_STAR = 0.25
DEFAULT_SLOPE = 0.7
DEFAULT_NSOURCES = 100
DEFAULT_XMAX = 50.0
DEFAULT_USE_FLAGS = [True, True, True]
DEFAULT_OUTDIR = "Output/POWER_LAW_tests"


# Try to use the project's mplstyle if it sits beside this file
_HERE = os.path.dirname(os.path.abspath(__file__))
_STYLE = os.path.join(_HERE, "scientific_presentation.mplstyle")
if os.path.exists(_STYLE):
    plt.style.use(_STYLE)


# ============================================================
# Field construction
# ============================================================

def build_standardized_field(
    Nlens, Nsource, kappa_star, slope, xmax,
    theta_star=DEFAULT_THETA_STAR,
    z_l=DEFAULT_Z_L, z_s=DEFAULT_Z_S,
    use_noise=False,
    substructure=False,
):
    """
    Build a synthetic POWER_LAW field with deterministic geometry.

    Geometry mirrors tests_NFW.build_standardized_field:
        Nlens=1: single halo at (0, 0)  (same convention used in the
                 random-realization MC for clean position histograms)
        Nlens=2: two halos along y=0 at (-xmax/2, +xmax/2)
        Nlens>=3: lenses laid on a diagonal grid

    Sources are placed on a sqrt(Nsource) x sqrt(Nsource) grid spanning
    [-xmax, xmax] in both directions (matching tests_NFW convention).

    Parameters
    ----------
    Nlens : int
    Nsource : int
        Approximate; rounded to the nearest perfect square.
    kappa_star, slope : float
        Used as the truth values for every halo.  If `substructure=True`,
        the first halo's kappa_star is reduced by 10x (primary halo
        with weak substructure).
    xmax : float
        Half-width of the source field (arcsec).
    theta_star : float
        Pivot radius (arcsec).  Default 30.
    z_l, z_s : float
        Lens and source redshifts.
    use_noise : bool
        Apply Gaussian noise via Source.apply_noise() after lensing.
    substructure : bool
        If True and Nlens >= 2, halo 0 is downweighted to 0.1*kappa_star.

    Returns
    -------
    halos : PowerLawHalo
    sources : Source
    noisy : str ('noisy' or 'noiseless')
    """
    # Lens positions (mirrors tests_NFW convention but using halo
    # coords centered on the field, since POWER_LAW seeds expect
    # positive coordinates from cast_votes).
    if Nlens == 1:
        x = np.array([0.0])
        y = np.array([0.0])
    elif Nlens == 2:
        x = np.linspace(-xmax / 2, xmax / 2, Nlens)
        y = np.zeros(Nlens)
    else:
        x = np.linspace(-xmax / 2, xmax / 2, Nlens)
        y = np.linspace(-xmax / 2, xmax / 2, Nlens)

    k_arr = np.full(Nlens, float(kappa_star))
    n_arr = np.full(Nlens, float(slope))
    if substructure and Nlens >= 2:
        k_arr[0] /= 10.0  # the FIRST halo becomes the substructure

    halos = halo_obj.PowerLawHalo(
        x=x, y=y,
        kappa_star=k_arr, slope=n_arr,
        theta_star=theta_star, redshift=z_l,
        chi2=np.zeros(Nlens),
    )

    # Source positions: uniform-random on a disk of radius xmax.  This
    # is intentionally NOT a square grid — random placement injects
    # source-distribution noise on top of measurement noise, so MC
    # variance reflects realistic survey-cadence uncertainty rather
    # than the artificial regularity of a meshgrid.  Mirrors the
    # spherical sampling in tests_NFW.run_random_realizations:
    #     r = sqrt(U) * R   (uniform-area)
    #     theta = 2*pi*U
    if Nsource <= 1:
        Nsource = 1
        xs = np.random.uniform(-xmax, xmax, 1)
        ys = np.random.uniform(-xmax, xmax, 1)
    else:
        r_s = np.sqrt(np.random.random(Nsource)) * xmax
        theta_s = np.random.random(Nsource) * 2 * np.pi
        xs = r_s * np.cos(theta_s)
        ys = r_s * np.sin(theta_s)

    sig_s = np.full(Nsource, 0.1)
    sig_f = np.full(Nsource, 0.01)
    sig_g = np.full(Nsource, 0.02)
    sources = source_obj.Source(
        xs, ys,
        np.zeros_like(xs), np.zeros_like(xs),
        np.zeros_like(xs), np.zeros_like(xs),
        np.zeros_like(xs), np.zeros_like(xs),
        sig_s, sig_f, sig_g,
        z_s,
    )

    sources.apply_lensing(halos, lens_type='POWER_LAW', z_source=z_s)
    if use_noise:
        sources.apply_noise()
        noisy = 'noisy'
    else:
        noisy = 'noiseless'
    sources.filter_sources()
    return halos, sources, noisy


# ============================================================
# Single-realization plotting
# ============================================================

def plot_results(lens, true_lens, title, reduced_chi2, xmax,
                 ax=None, legend=False, show_strength=False,
                 show_chi2=False):
    """
    Plot a single recovered POWER_LAW fit vs truth.

    Mirrors tests_NFW.plot_results: blue stars for truth (sized by
    kappa_star), red circles for recovered (sized by kappa_star),
    optional annotations of (kappa_star, n) and reduced chi^2.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Truth: blue stars
    if true_lens is not None and getattr(true_lens, 'x', np.array([])).size > 0:
        ax.scatter(
            true_lens.x, true_lens.y,
            marker='*',
            s=400 + 2000 * np.asarray(true_lens.kappa_star),
            facecolors='royalblue', edgecolors='white', linewidths=1.4,
            zorder=10,
            label='Truth',
        )
        if show_strength:
            for i in range(true_lens.x.size):
                ax.annotate(
                    fr"$\kappa_\star={true_lens.kappa_star[i]:.2f}$" "\n"
                    fr"$n={true_lens.slope[i]:.2f}$",
                    xy=(true_lens.x[i], true_lens.y[i]),
                    xytext=(8, 8), textcoords='offset points',
                    color='royalblue', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', edgecolor='royalblue',
                              alpha=0.8),
                )

    # Recovered: red open circles
    if lens is not None and getattr(lens, 'x', np.array([])).size > 0:
        ax.scatter(
            lens.x, lens.y,
            marker='o',
            s=200 + 2000 * np.asarray(lens.kappa_star),
            facecolors='none', edgecolors='crimson', linewidths=1.7,
            zorder=11,
            label='Recovered',
        )
        if show_strength:
            for j in range(lens.x.size):
                ax.annotate(
                    fr"$\hat\kappa_\star={lens.kappa_star[j]:.3f}$" "\n"
                    fr"$\hat n={lens.slope[j]:.3f}$",
                    xy=(lens.x[j], lens.y[j]),
                    xytext=(-95, -42), textcoords='offset points',
                    color='crimson', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='white', edgecolor='crimson',
                              alpha=0.8),
                )

    if show_chi2:
        title = title + fr"   ($\chi^2_\nu={reduced_chi2:.2f}$)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    if legend:
        ax.legend(loc='lower right', fontsize=9)


# ============================================================
# Pipeline breakdown (gold-standard diagnostic plot)
# ============================================================

def pipeline_breakdown(sources, true_lenses, xmax, use_flags, noisy,
                       theta_star=DEFAULT_THETA_STAR,
                       z_l=DEFAULT_Z_L,
                       outdir=DEFAULT_OUTDIR,
                       name=None,
                       print_steps=False):
    """
    Run the POWER_LAW pipeline step by step and plot each intermediate
    state in a 2x3 grid.

    Mirrors tests_NFW.pipeline_breakdown.  Saves to:
        {outdir}/breakdown/{name or auto-generated}.png

    Parameters
    ----------
    sources : Source
    true_lenses : PowerLawHalo
        For overlay only.
    xmax : float
    use_flags : list of bool
    noisy : str
        'noisy' or 'noiseless', for plot naming.
    theta_star : float
    z_l : float
    outdir : str
    name : str or None
    print_steps : bool
    """
    fig, axarr = plt.subplots(2, 3, figsize=(20, 14), sharex=True, sharey=True)
    axarr = axarr.flatten()

    # Step 1: Initial candidate generation
    lenses = pipeline.generate_initial_guess(
        sources, lens_type='POWER_LAW',
        z_l=z_l, theta_star=theta_star,
    )
    print(f"Step 1: generated {lenses.x.size} candidates")
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Candidate Lens Generation',
                 rchi2, xmax, ax=axarr[0], legend=True)
    if print_steps:
        print(f"Step 1: generated {lenses.x.size} candidates "
              f"(rchi2={rchi2:.3f})")

    # Step 2: Optimize positions
    lenses = pipeline.optimize_lens_positions(
        sources, lenses, xmax, use_flags, lens_type='POWER_LAW')
    print(f"Step 2: post-optimization {lenses.x.size} lenses")
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Individual Lens Optimization',
                 rchi2, xmax, ax=axarr[1])
    if print_steps:
        print(f"Step 2: post-optimization {lenses.x.size} lenses "
              f"(rchi2={rchi2:.3f})")

    # Step 3: Filter
    lenses = pipeline.filter_lens_positions(
        sources, lenses, xmax, lens_type='POWER_LAW')
    print(f"Step 3: post-filter {lenses.x.size} lenses")
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Physical Criteria Filtering',
                 rchi2, xmax, ax=axarr[2])
    if print_steps:
        print(f"Step 3: post-filter {lenses.x.size} lenses "
              f"(rchi2={rchi2:.3f})")

    # Step 4: Forward selection
    selected, _ = pipeline.forward_lens_selection(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    if selected is None or selected.x.size == 0:
        fig.suptitle('No Halos Recovered')
        savepath = _make_breakdown_path(
            true_lenses, noisy, use_flags, outdir, name=name)
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)
        if print_steps:
            print(f"  no halos selected — plot saved to {savepath}")
        return None
    lenses = selected
    print(f"Step 4: forward selection -> {lenses.x.size} lenses")
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Forward Lens Selection',
                 rchi2, xmax, ax=axarr[3])
    if print_steps:
        print(f"Step 4: forward selection -> {lenses.x.size} lenses "
              f"(rchi2={rchi2:.3f})")

    # Step 5: Merge
    area = (2 * xmax) ** 2
    ns = len(sources.x) / area
    merger_threshold = (1.0 / np.sqrt(ns)) if ns > 0 else 1.0
    lenses = pipeline.merge_close_lenses(
        lenses, merger_threshold=merger_threshold,
        lens_type='POWER_LAW')
    print(f"Step 5: merge -> {lenses.x.size} lenses")
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Lens Merging',
                 rchi2, xmax, ax=axarr[4], show_strength=True)
    if print_steps:
        print(f"Step 5: merge -> {lenses.x.size} lenses "
              f"(rchi2={rchi2:.3f})")

    # Step 6: Strength optimization
    lenses = pipeline.optimize_lens_strength(
        sources, lenses, use_flags, lens_type='POWER_LAW')
    print(f"Step 6: strength refine -> {lenses.x.size} lenses") 
    rchi2 = pipeline.update_chi2_values(
        sources, lenses, [True, True, True], lens_type='POWER_LAW')
    plot_results(lenses, true_lenses, 'Strength Refinement',
                 rchi2, xmax, ax=axarr[5], show_strength=True, show_chi2=True)
    if print_steps:
        print(f"Step 6: strength refine -> {lenses.x.size} lenses "
              f"(rchi2={rchi2:.3f})")

    # Title summarizing recovery
    n_truth = true_lenses.x.size
    n_recov = lenses.x.size
    fig.suptitle(
        f"POWER_LAW pipeline breakdown — "
        f"truth halos: {n_truth}, recovered: {n_recov}, "
        fr"$\chi^2_\nu={rchi2:.2f}$",
        fontsize=12, y=0.995,
    )

    savepath = _make_breakdown_path(
        true_lenses, noisy, use_flags, outdir, name=name)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches='tight')
    plt.close(fig)
    if print_steps:
        print(f"  plot saved to {savepath}")
    return lenses


def _make_breakdown_path(true_lenses, noisy, use_flags, outdir, name=None):
    """Generate a structured filename like the NFW convention."""
    breakdown_dir = os.path.join(outdir, "breakdown")
    os.makedirs(breakdown_dir, exist_ok=True)

    n_lens = int(true_lenses.x.size)
    k = float(np.atleast_1d(true_lenses.kappa_star)[0])
    n = float(np.atleast_1d(true_lenses.slope)[0])
    flag_tag = "".join("1" if f else "0" for f in use_flags)
    auto = (f"{n_lens}halo_k{k:.2f}_n{n:.2f}_{noisy}_flags{flag_tag}"
            .replace('.', 'p'))
    if name:
        auto = f"{auto}_{name}"
    return os.path.join(breakdown_dir, f"{auto}.png")


# ============================================================
# Single-shot runner (for `--mode single`)
# ============================================================

def run_single(args):
    """Run a single realization end-to-end and produce a 1-panel summary plot."""
    rng_seed = args.seed
    if rng_seed is not None:
        np.random.seed(rng_seed)

    halos, sources, noisy = build_standardized_field(
        Nlens=args.nlens, Nsource=args.nsrc,
        kappa_star=args.kappa_star, slope=args.slope, xmax=args.xmax,
        theta_star=args.theta_star, z_l=args.z_l, z_s=args.z_s,
        use_noise=args.noise, substructure=args.substructure,
    )

    print(f"Field built: {args.nlens} truth halos, "
          f"{sources.x.size} sources after filter, {noisy}")

    recovered, rchi2 = main.fit_lensing_field(
        sources, args.xmax, flags=args.print_steps,
        use_flags=DEFAULT_USE_FLAGS,
        lens_type='POWER_LAW',
        z_lens=args.z_l,
    )
    print(f"Recovered: {recovered.x.size if recovered else 0} lenses, "
          f"rchi2={rchi2:.3f}")

    out_dir = os.path.join(args.outdir, "single")
    os.makedirs(out_dir, exist_ok=True)
    flag_tag = "".join("1" if f else "0" for f in DEFAULT_USE_FLAGS)
    fname = (f"{args.nlens}halo_k{args.kappa_star:.2f}_n{args.slope:.2f}_"
             f"{noisy}_flags{flag_tag}_seed{rng_seed}.png").replace('.', 'p', 2)
    savepath = os.path.join(out_dir, fname)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_results(recovered, halos, "POWER_LAW single-shot recovery",
                 rchi2, args.xmax,
                 ax=ax, legend=True, show_strength=True, show_chi2=True)
    # Background: source field colored by |F|
    F_amp = np.hypot(sources.f1, sources.f2)
    sc = ax.scatter(sources.x, sources.y, c=F_amp, s=14, cmap='viridis',
                    alpha=0.5, edgecolors='none', zorder=1)
    fig.tight_layout()
    fig.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {savepath}")


def run_breakdown(args):
    """Run a single realization with full step-by-step plotting."""
    rng_seed = args.seed
    if rng_seed is not None:
        np.random.seed(rng_seed)

    halos, sources, noisy = build_standardized_field(
        Nlens=args.nlens, Nsource=args.nsrc,
        kappa_star=args.kappa_star, slope=args.slope, xmax=args.xmax,
        theta_star=args.theta_star, z_l=args.z_l, z_s=args.z_s,
        use_noise=args.noise, substructure=args.substructure,
    )
    print(f"Field built: {args.nlens} truth halos, "
          f"{sources.x.size} sources after filter, {noisy}")

    pipeline_breakdown(
        sources, halos, args.xmax, DEFAULT_USE_FLAGS, noisy,
        theta_star=args.theta_star, z_l=args.z_l,
        outdir=args.outdir,
        name=f"seed{rng_seed}" if rng_seed is not None else None,
        print_steps=args.print_steps,
    )


# ============================================================
# Monte Carlo
# ============================================================

def run_random_realizations(
    Ntrials,
    Nlenses=1,
    Nsources=DEFAULT_NSOURCES,
    xmax=DEFAULT_XMAX,
    kappa_star=DEFAULT_KAPPA_STAR,
    slope=DEFAULT_SLOPE,
    theta_star=DEFAULT_THETA_STAR,
    z_l=DEFAULT_Z_L,
    z_s=DEFAULT_Z_S,
    use_flags=None,
    use_noise=True,
    substructure=False,
    random_seed=None,
):
    """
    Run Ntrials realizations of the POWER_LAW pipeline, collecting recovered
    parameters at each.

    The truth halos are FIXED across trials (deterministic geometry from
    build_standardized_field); only the source noise realization changes.

    Returns
    -------
    recovered_params : dict
        {'x': ndarray, 'y': ndarray, 'kappa_star': ndarray, 'slope': ndarray,
         'rchi2': ndarray, 'n_recovered_per_trial': ndarray}
    true_params : dict
        {'x': ndarray, 'y': ndarray, 'kappa_star': ndarray, 'slope': ndarray}
    """
    if use_flags is None:
        use_flags = DEFAULT_USE_FLAGS

    if random_seed is not None:
        np.random.seed(random_seed)

    # Truth (fixed across trials)
    truth, _, _ = build_standardized_field(
        Nlens=Nlenses, Nsource=Nsources,
        kappa_star=kappa_star, slope=slope, xmax=xmax,
        theta_star=theta_star, z_l=z_l, z_s=z_s,
        use_noise=False, substructure=substructure,
    )
    true_params = {
        'x': truth.x.copy(),
        'y': truth.y.copy(),
        'kappa_star': truth.kappa_star.copy(),
        'slope': truth.slope.copy(),
    }

    rec_x, rec_y, rec_k, rec_n, rec_rchi2 = [], [], [], [], []
    n_per_trial = []

    print(f"Running {Ntrials} POWER_LAW realizations...")
    for trial in range(Ntrials):
        # Build a fresh field with a new noise draw.  build_standardized_field
        # uses np.random under the hood (via Source.apply_noise), so the
        # trial-by-trial seeding here is enough.
        _, sources, _ = build_standardized_field(
            Nlens=Nlenses, Nsource=Nsources,
            kappa_star=kappa_star, slope=slope, xmax=xmax,
            theta_star=theta_star, z_l=z_l, z_s=z_s,
            use_noise=use_noise, substructure=substructure,
        )

        try:
            lenses, rchi2 = main.fit_lensing_field(
                sources, xmax, flags=False,
                use_flags=use_flags, lens_type='POWER_LAW',
                z_lens=z_l,
            )
        except Exception as e:
            print(f"  trial {trial}: FAILED ({type(e).__name__}: {e})")
            n_per_trial.append(0)
            continue

        if lenses is None or lenses.x.size == 0:
            n_per_trial.append(0)
            continue

        rec_x.extend(lenses.x.tolist())
        rec_y.extend(lenses.y.tolist())
        rec_k.extend(lenses.kappa_star.tolist())
        rec_n.extend(lenses.slope.tolist())
        rec_rchi2.extend([rchi2] * lenses.x.size)
        n_per_trial.append(int(lenses.x.size))

        if (trial + 1) % max(Ntrials // 20, 1) == 0:
            done_pct = 100 * (trial + 1) / Ntrials
            print(f"  ... {trial+1}/{Ntrials} ({done_pct:.0f}%)")

    recovered_params = {
        'x': np.array(rec_x),
        'y': np.array(rec_y),
        'kappa_star': np.array(rec_k),
        'slope': np.array(rec_n),
        'rchi2': np.array(rec_rchi2),
        'n_recovered_per_trial': np.array(n_per_trial),
    }
    return recovered_params, true_params


def plot_random_realizations(
    recovered, true_params, title, xmax, outdir=DEFAULT_OUTDIR,
):
    """
    Plot the MC-recovered parameter distributions in a 2x2 grid:

        Top-left:    2D position histogram (log scale) with truth overlay
        Top-right:   kappa_star distribution + Gaussian fit + truth line
        Bottom-left: slope distribution + Gaussian fit + truth line
        Bottom-right: number-of-halos-per-trial histogram

    Mirrors tests_NFW.plot_random_realizations conventions but adapted
    for the POWER_LAW (kappa_star, slope) parameter pair.
    """
    out_dir = os.path.join(outdir, "random_realization")
    os.makedirs(out_dir, exist_ok=True)
    savepath = os.path.join(out_dir, f"{title}.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(title, fontsize=13)

    # --- (0,0)  position 2D histogram ---
    ax = axes[0, 0]
    if recovered['x'].size > 0:
        h = ax.hist2d(
            recovered['x'], recovered['y'],
            bins=30, cmap='viridis',
            norm=LogNorm(),
        )
    ax.scatter(true_params['x'], true_params['y'],
               s=300, c='red', marker='*',
               edgecolors='white', linewidths=1.2,
               label='Truth', zorder=10)
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    ax.set_title('Recovered position distribution')
    ax.set_xlim(-xmax, xmax); ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)

    # --- helper: Gaussian fit ---
    def _gaussian(x, A, mu, sigma):
        return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def _fit_and_plot(values, ax_, truth_value, xlabel, color):
        if values.size < 5:
            ax_.text(0.5, 0.5, "insufficient data",
                     ha='center', va='center', transform=ax_.transAxes)
            ax_.set_xlabel(xlabel)
            return
        bins = max(min(40, values.size // 3), 8)
        counts, edges, _ = ax_.hist(
            values, bins=bins, density=True,
            histtype='stepfilled', color=color, alpha=0.55,
            edgecolor='black', linewidth=0.5,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Gaussian fit
        try:
            p0 = [counts.max(), float(np.median(values)), float(np.std(values))]
            popt, _ = curve_fit(_gaussian, centers, counts, p0=p0,
                                maxfev=2000)
            xs_fit = np.linspace(values.min(), values.max(), 200)
            ax_.plot(xs_fit, _gaussian(xs_fit, *popt),
                     'k--', lw=1.5,
                     label=fr"Gauss fit: $\mu={popt[1]:.3f}$, "
                           fr"$\sigma={abs(popt[2]):.3f}$")
        except Exception:
            pass
        if np.isscalar(truth_value):
            truth_arr = [truth_value]
        else:
            truth_arr = np.atleast_1d(truth_value)
        for t in np.unique(truth_arr):
            ax_.axvline(t, color='red', linestyle=':', lw=1.5,
                        label=f'truth = {t:.3f}')
        ax_.axvline(np.median(values), color='green', linestyle='--', lw=1.2,
                    label=f'median = {np.median(values):.3f}')
        ax_.set_xlabel(xlabel)
        ax_.set_ylabel('density')
        ax_.legend(loc='best', fontsize=8)

    # --- (0,1)  kappa_star distribution ---
    _fit_and_plot(recovered['kappa_star'], axes[0, 1],
                  truth_value=true_params['kappa_star'],
                  xlabel=r'$\kappa_\star$ recovered',
                  color='steelblue')
    axes[0, 1].set_title(r'$\kappa_\star$ recovery')

    # --- (1,0)  slope distribution ---
    _fit_and_plot(recovered['slope'], axes[1, 0],
                  truth_value=true_params['slope'],
                  xlabel=r'$n$ recovered',
                  color='seagreen')
    axes[1, 0].set_title(r'slope $n$ recovery')

    # --- (1,1)  number-of-halos-per-trial histogram ---
    ax = axes[1, 1]
    n_arr = recovered['n_recovered_per_trial']
    if n_arr.size > 0:
        bins = np.arange(-0.5, max(n_arr.max(), 1) + 1.5, 1)
        ax.hist(n_arr, bins=bins, color='goldenrod', alpha=0.7,
                edgecolor='black')
        ax.axvline(true_params['x'].size, color='red', linestyle=':',
                   lw=1.5, label=f"truth = {true_params['x'].size}")
        ax.set_xlabel('halos recovered per trial')
        ax.set_ylabel('count')
        ax.set_title('Detection rate')
        ax.legend(loc='upper right', fontsize=9)

    fig.tight_layout()
    fig.savefig(savepath, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"MC plot saved to {savepath}")
    return savepath


def run_mc(args):
    """Run a Monte Carlo and produce the recovery plots."""
    rec, truth = run_random_realizations(
        Ntrials=args.ntrials,
        Nlenses=args.nlens,
        Nsources=args.nsrc,
        xmax=args.xmax,
        kappa_star=args.kappa_star,
        slope=args.slope,
        theta_star=args.theta_star,
        z_l=args.z_l,
        z_s=args.z_s,
        use_flags=DEFAULT_USE_FLAGS,
        use_noise=args.noise,
        substructure=args.substructure,
        random_seed=args.seed,
    )

    flag_tag = "".join("1" if f else "0" for f in DEFAULT_USE_FLAGS)
    title = (f"MC_{args.nlens}halo_k{args.kappa_star:.2f}_"
             f"n{args.slope:.2f}_{'noisy' if args.noise else 'noiseless'}_"
             f"flags{flag_tag}_N{args.ntrials}").replace('.', 'p')
    plot_random_realizations(rec, truth, title, args.xmax,
                             outdir=args.outdir)

    # Also dump the raw arrays so the MC can be replayed without rerunning
    npy_path = os.path.join(args.outdir, "random_realization", f"{title}.npy")
    np.save(npy_path,
            {'recovered': rec, 'truth': truth, 'args': vars(args)},
            allow_pickle=True)
    print(f"Raw data saved to {npy_path}")

    # Summary stats
    n_total = rec['x'].size
    n_trials = rec['n_recovered_per_trial'].size
    detection_rate = (n_total / (n_trials * args.nlens)) if n_trials > 0 else 0.0
    print()
    print(f"=== MC SUMMARY ({args.ntrials} trials) ===")
    print(f"  total recovered halos    : {n_total}")
    print(f"  expected (no FP/FN)      : {n_trials * args.nlens}")
    print(f"  detection rate           : {detection_rate*100:.1f}%")
    if n_total > 0:
        print(f"  median kappa_star        : {np.median(rec['kappa_star']):.4f}  "
              f"(truth {args.kappa_star:.4f})")
        print(f"  median slope             : {np.median(rec['slope']):.4f}  "
              f"(truth {args.slope:.4f})")
        if args.nlens == 1:
            r = np.hypot(rec['x'] - truth['x'][0],
                         rec['y'] - truth['y'][0])
            print(f"  median position err      : {np.median(r):.2f} arcsec")


# ============================================================
# CLI
# ============================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="Synthetic + MC tests for ARCH POWER_LAW.")
    p.add_argument('--mode', choices=['single', 'breakdown', 'mc'],
                   default='single')
    p.add_argument('--nlens', type=int, default=1)
    p.add_argument('--nsrc', type=int, default=DEFAULT_NSOURCES)
    p.add_argument('--xmax', type=float, default=DEFAULT_XMAX)
    p.add_argument('--kappa-star', type=float, default=DEFAULT_KAPPA_STAR,
                   dest='kappa_star')
    p.add_argument('--slope', type=float, default=DEFAULT_SLOPE)
    p.add_argument('--theta-star', type=float, default=DEFAULT_THETA_STAR,
                   dest='theta_star')
    p.add_argument('--z-l', type=float, default=DEFAULT_Z_L, dest='z_l')
    p.add_argument('--z-s', type=float, default=DEFAULT_Z_S, dest='z_s')
    p.add_argument('--noise', action='store_true',
                   help='Add Gaussian noise to source signals')
    p.add_argument('--substructure', action='store_true',
                   help='Make halo 0 a substructure (0.1x kappa_star)')
    p.add_argument('--ntrials', type=int, default=200,
                   help='Monte Carlo trial count (--mode mc)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--outdir', type=str, default=DEFAULT_OUTDIR)
    p.add_argument('--print-steps', action='store_true',
                   help='Print per-step diagnostics during pipeline runs')
    return p


def main_entry():
    args = _build_parser().parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.mode == 'single':
        run_single(args)
    elif args.mode == 'breakdown':
        run_breakdown(args)
    elif args.mode == 'mc':
        run_mc(args)


if __name__ == "__main__":
    main_entry()