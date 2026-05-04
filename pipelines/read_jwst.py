import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from scipy.ndimage import gaussian_filter
from pathlib import Path
import warnings
import csv
import concurrent.futures
import tqdm
from astropy.cosmology import Planck18 as COSMO

# Import ARCH modules
import arch.main as main
import arch.source_obj as source_obj
import arch.halo_obj as halo_obj
import arch.utils as utils

# Set matplotlib style
plt.style.use('scientific_presentation.mplstyle')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Redshifts
hubble_param = 0.67  # Hubble constant


def unpack_and_run_jackknife(args):
    return jackknife_worker(*args)


def run_fitting_with_sources(self_obj, sources):
    original_sources = self_obj.sources
    self_obj.sources = sources
    self_obj.run_lens_fitting(flags=False)
    result = self_obj.lenses.copy()
    self_obj.sources = original_sources
    return result


def jackknife_worker(i, self_obj, sources):
    """
    Per-trial jackknife worker.  Returns the recovered lens parameters
    in a list of rows; columns differ by lens_type:
        NFW:       [i, x, y, M200, concentration]
        POWER_LAW: [i, x, y, kappa_star, slope]
    """
    modified_sources = sources.copy()
    modified_sources.remove(i)
    lenses = run_fitting_with_sources(self_obj, modified_sources)

    if self_obj.lens_type == 'NFW':
        return [
            [i, x, y, m, c]
            for x, y, m, c in zip(lenses.x, lenses.y,
                                  lenses.mass, lenses.concentration)
        ]
    elif self_obj.lens_type == 'POWER_LAW':
        return [
            [i, x, y, k, n]
            for x, y, k, n in zip(lenses.x, lenses.y,
                                  lenses.kappa_star, lenses.slope)
        ]
    else:
        raise ValueError(f"Unsupported lens_type for jackknife: {self_obj.lens_type}")


class JWSTPipeline:
    """
    JWST lensing pipeline supporting both NFW and POWER_LAW reconstructions.

    The lens_type field in the config selects which profile to fit:
        'NFW'        - default, behavior identical to the prior runner
        'POWER_LAW'  - uses Phase 4 power-law pipeline; recovered halos
                       carry (kappa_star, slope) instead of (mass,
                       concentration), and theta_star is fixed at 30
                       arcsec (the project default; override via config
                       key 'theta_star' if desired).
    """

    def __init__(self, config):
        self.config = config
        self.CDELT = 8.54006306703281e-6 * 3600  # arcsec/pixel

        # Paths
        self.flexion_catalog_path = Path(config['flexion_catalog_path'])
        self.source_catalog_path = Path(config['source_catalog_path'])
        self.image_path = Path(config['image_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_name = config['cluster_name']
        self.signal_choice = config['signal_choice']

        if self.signal_choice == 'all':
            self.use_flags = [True, True, True]
        elif self.signal_choice == 'shear_f':
            self.use_flags = [True, True, False]
        elif self.signal_choice == 'f_g':
            self.use_flags = [False, True, True]
        elif self.signal_choice == 'shear_g':
            self.use_flags = [True, False, True]
        else:
            raise ValueError(f"Invalid signal choice: {self.signal_choice}")

        # Lens type — defaults to NFW for backward compatibility
        self.lens_type = config.get('lens_type', 'NFW')
        if self.lens_type not in ('NFW', 'POWER_LAW'):
            raise ValueError(
                f"Invalid lens_type {self.lens_type!r}. "
                f"Must be 'NFW' or 'POWER_LAW'.")

        # POWER_LAW pivot radius (only used if lens_type == 'POWER_LAW')
        self.theta_star = float(config.get('theta_star', 30.0))

        # Redshifts
        self.z_source = config['source_redshift']
        self.z_cluster = config['cluster_redshift']

        # Data placeholders (unchanged)
        self.IDs = None
        self.q = None
        self.phi = None
        self.psi11 = None
        self.psi12 = None
        self.psi22 = None
        self.F1_fit = None
        self.F2_fit = None
        self.G1_fit = None
        self.G2_fit = None
        self.a = None
        self.chi2 = None
        self.xc = None
        self.yc = None
        self.lenses = None
        self.sources = None
        self.centroid_x = None
        self.centroid_y = None

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(self):
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()
        self.run_lens_fitting()

    def visualize(self):
        self.read_source_catalog()
        self.read_flexion_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()
        self.sources.x += self.centroid_x
        self.sources.y += self.centroid_y
        self.import_lenses()
        self.plot_results()

    def compute_error_bars(self):
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()

        output_path = (f"jackknife_results_{self.cluster_name}_"
                       f"{self.signal_choice}_{self.lens_type}.csv")
        # Per-lens-type column header
        if self.lens_type == 'NFW':
            header = ["i", "x", "y", "M200", "concentration"]
        elif self.lens_type == 'POWER_LAW':
            header = ["i", "x", "y", "kappa_star", "slope"]
        with open(output_path, "w", newline='') as f:
            csv.writer(f).writerow(header)

        sources_copy = self.sources.copy()
        n_sources = len(sources_copy.x)
        indices_to_use = np.random.choice(n_sources, size=200, replace=False)

        args_list = [(i, self, sources_copy) for i in indices_to_use]

        with open(output_path, "a", newline='') as f:
            writer = csv.writer(f)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in tqdm.tqdm(
                    executor.map(unpack_and_run_jackknife, args_list,
                                 chunksize=3),
                    total=len(indices_to_use),
                    desc=f"Computing error bars ({self.lens_type})"
                ):
                    writer.writerows(result)

    # ------------------------------------------------------------------
    # Catalog I/O (unchanged from prior runner)
    # ------------------------------------------------------------------

    def read_source_catalog(self):
        table = Table.read(self.source_catalog_path)
        self.x_centroids = np.array(table['xcentroid'])
        self.y_centroids = np.array(table['ycentroid'])
        self.labels = np.array(table['label'])

    def read_flexion_catalog(self):
        df = pd.read_pickle(self.flexion_catalog_path)
        self.IDs = df['label'].to_numpy()
        self.q = df['q'].to_numpy()
        self.phi = df['phi'].to_numpy()
        self.F1_fit = df['F1_fit'].to_numpy() / self.CDELT
        self.F2_fit = df['F2_fit'].to_numpy() / self.CDELT
        self.G1_fit = df['G1_fit'].to_numpy() / self.CDELT
        self.G2_fit = df['G2_fit'].to_numpy() / self.CDELT
        self.a = df['a'].to_numpy() * self.CDELT
        self.chi2 = df['rchi2'].to_numpy()
        self.rs = df['rs'].to_numpy() * self.CDELT
        print(f"Read {len(self.IDs)} entries from flexion catalog.")

    def cut_flexion_catalog(self):
        F = np.hypot(self.F1_fit, self.F2_fit)
        aF = self.a * F
        max_aF = 0.5
        max_rs = 5.0
        max_chi2 = 1.5
        min_a, max_a = 0.01, 2.0

        bad_flexion = aF > max_aF
        bad_rs = self.rs > max_rs
        bad_chi2 = self.chi2 > max_chi2
        bad_a = (self.a < min_a) | (self.a > max_a)
        nan_indices = np.isnan(self.IDs) | np.isnan(self.q) | np.isnan(self.phi) | \
                      np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | \
                      np.isnan(self.G1_fit) | np.isnan(self.G2_fit) | \
                      np.isnan(self.a) | np.isnan(self.rs) | np.isnan(self.chi2)
        bad_indices = bad_flexion | bad_rs | bad_chi2 | bad_a | nan_indices

        self.IDs = self.IDs[~bad_indices]
        self.q = self.q[~bad_indices]
        self.phi = self.phi[~bad_indices]
        self.F1_fit = self.F1_fit[~bad_indices]
        self.F2_fit = self.F2_fit[~bad_indices]
        self.G1_fit = self.G1_fit[~bad_indices]
        self.G2_fit = self.G2_fit[~bad_indices]
        self.a = self.a[~bad_indices]
        self.rs = self.rs[~bad_indices]
        self.chi2 = self.chi2[~bad_indices]

        print(f"Filtered flexion catalog to {len(self.IDs)} entries "
              f"after applying updated cuts.")

    def match_sources(self):
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        xc_list, yc_list = [], []
        for ID in self.IDs:
            idx = label_to_index.get(ID)
            if idx is not None:
                xc_list.append(self.x_centroids[idx])
                yc_list.append(self.y_centroids[idx])
            else:
                warnings.warn(f"ID '{ID}' not found in source catalog. "
                              f"Assigning NaN.")
                xc_list.append(np.nan)
                yc_list.append(np.nan)
        self.xc = np.array(xc_list) * self.CDELT
        self.yc = np.array(yc_list) * self.CDELT

    def initialize_sources(self):
        shear_magnitude = (self.q - 1) / (self.q + 1)
        e1 = shear_magnitude * np.cos(2 * self.phi)
        e2 = shear_magnitude * np.sin(2 * self.phi)
        sigs = np.full_like(e1, np.mean([np.std(e1), np.std(e2)]))
        sigaf = np.mean([np.std(self.a * self.F1_fit), np.std(self.a * self.F2_fit)])
        sigag = np.mean([np.std(self.a * self.G1_fit), np.std(self.a * self.G2_fit)])
        sigf, sigg = sigaf / self.a, sigag / self.a

        self.sources = source_obj.Source(
            x=self.xc, y=self.yc,
            e1=e1, e2=e2,
            f1=self.F1_fit, f2=self.F2_fit,
            g1=self.G1_fit, g2=self.G2_fit,
            sigs=sigs, sigf=sigf, sigg=sigg,
            redshift=self.z_source,
        )
        self.centroid_x = np.mean(self.xc)
        self.centroid_y = np.mean(self.yc)
        self.sources.x -= self.centroid_x
        self.sources.y -= self.centroid_y

    # ------------------------------------------------------------------
    # Lens fitting — dispatches by lens_type
    # ------------------------------------------------------------------

    def run_lens_fitting(self, flags=True):
        xmax = np.max(np.hypot(self.sources.x, self.sources.y))

        if self.lens_type == 'NFW':
            self.lenses, _ = main.fit_lensing_field(
                self.sources, xmax, flags=flags, use_flags=self.use_flags,
                lens_type='NFW', z_lens=self.z_cluster,
            )
        elif self.lens_type == 'POWER_LAW':
            # main.fit_lensing_field as currently shipped does not accept
            # theta_star; the pipeline branches on lens_type internally
            # and uses the default 30" pivot from generate_initial_guess.
            # If you've extended fit_lensing_field to take theta_star,
            # uncomment the kwarg.
            self.lenses, _ = main.fit_lensing_field(
                self.sources, xmax, flags=flags, use_flags=self.use_flags,
                lens_type='POWER_LAW', z_lens=self.z_cluster,
                # theta_star=self.theta_star,
            )

        check = self.lenses.check_for_nan_properties()
        if check:
            print("Warning: Some lens properties are NaN or Inf. "
                  "Check the lens fitting process.")

        # Restore lens & source positions to the original (un-centered) frame
        self.lenses.x += self.centroid_x
        self.lenses.y += self.centroid_y
        self.sources.x += self.centroid_x
        self.sources.y += self.centroid_y

        # NFW-only post-processing: convert mass to h^-1 M_sun
        if self.lens_type == 'NFW':
            self.lenses.mass *= hubble_param

        # Save
        file_name = (self.output_dir
                     / f"lenses_{self.cluster_name}_{self.signal_choice}"
                     f"_{self.lens_type}.csv")
        self.lenses.export_to_csv(file_name)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_results(self):
        """
        Publication-ready figures with grayscale JWST background, kappa
        contours, numbered peak markers, compass and scale bar.  Works
        for both NFW and POWER_LAW lens types — calculate_kappa handles
        the dispatch internally; this method only differs in the title
        annotation and (for NFW) the total-mass label.
        """

        def _fmt_sum_mass_hinv(m_hinv):
            if not np.isfinite(m_hinv):
                return r"$\sum M =$ N/A"
            return rf"$\sum M = {(m_hinv/1e14):.2f}\times 10^{{14}}\ h^{{-1}}\,M_\odot$"

        def _compute_levels_from_kappa(kappa, positive=True):
            finite = np.isfinite(kappa)
            if not np.any(finite):
                return [0.02, 0.04, 0.06, 0.08, 0.10]
            vals = kappa[finite]
            if positive:
                vals = vals[vals > 0]
            if vals.size < 20:
                kmax = np.nanmax(kappa)
                if not np.isfinite(kmax) or kmax <= 0:
                    return [0.02, 0.04, 0.06]
                return list(np.linspace(0.35, 0.9, 5) * kmax)
            qs = np.quantile(vals, [0.70, 0.82, 0.90, 0.96, 0.985])
            levels = [float(lv) for lv in qs if np.isfinite(lv) and lv > 0]
            if len(levels) < 3:
                kmax = np.nanmax(vals)
                levels = list(np.linspace(0.4, 0.85, 4)
                              * (kmax if np.isfinite(kmax) else 0.1))
            return sorted(set(levels))

        def _draw_compass_and_scalebar(ax, extent, bar_arcsec=50.0, z_lens=None):
            xmin, xmax, ymin, ymax = extent
            dx, dy = (xmax - xmin), (ymax - ymin)
            cx = xmin + 0.12 * dx
            cy = ymin + 0.12 * dy
            alen = 0.08 * max(dx, dy)
            ax.add_patch(FancyArrow(cx, cy, 0, alen, width=0.0,
                                    head_width=0.035 * dx,
                                    head_length=0.04 * dy,
                                    length_includes_head=True, color='k'))
            ax.text(cx, cy + alen + 0.02 * dy, "N",
                    ha="center", va="bottom", fontsize=8)
            ax.add_patch(FancyArrow(cx, cy, -alen, 0, width=0.0,
                                    head_width=0.035 * dy,
                                    head_length=0.04 * dx,
                                    length_includes_head=True, color='k'))
            ax.text(cx - alen - 0.02 * dx, cy, "E",
                    ha="right", va="center", fontsize=8)
            sx1 = xmax - 0.12 * dx
            sx0 = sx1 - bar_arcsec
            sy = ymin + 0.10 * dy
            ax.plot([sx0, sx1], [sy, sy], color='k', lw=1.8)
            if z_lens is not None and COSMO is not None:
                kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(z_lens).value / 60.0
                label = (rf"{int(round(bar_arcsec))}″ "
                         rf"({int(round(bar_arcsec * kpc_per_arcsec))} kpc)")
            else:
                label = rf"{int(round(bar_arcsec))}″"
            ax.text(0.5 * (sx0 + sx1), sy + 0.02 * dy, label,
                    ha="center", va="bottom", fontsize=8)

        def _plot_single_panel(img_data, img_extent, X, Y, kappa, levels,
                               peaks, title, sum_mass_hinv,
                               z_lens=None, smooth_sigma=1.0,
                               save_pdf_path=None, cluster_name=None):

            def _rotate_cw90_panel(img_data, img_extent, X, Y, kappa, peaks=None):
                xmin, xmax, ymin, ymax = img_extent
                img_r = np.rot90(img_data, k=-1)
                kappa_r = np.rot90(kappa, k=-1)
                X_r = np.rot90(Y, k=-1)
                Y_r = np.rot90(-X, k=-1)
                extent_r = (ymin, ymax, -xmax, -xmin)
                peaks_r = None
                if peaks:
                    peaks_r = [(py, -px) for (px, py) in peaks]
                return img_r, extent_r, X_r, Y_r, kappa_r, peaks_r

            kappa_disp = (gaussian_filter(kappa, smooth_sigma)
                          if smooth_sigma else kappa)

            if cluster_name == "EL_GORDO":
                img_data, img_extent, X, Y, kappa_disp, peaks = _rotate_cw90_panel(
                    img_data, img_extent, X, Y, kappa_disp, peaks=peaks)

            fig, ax = plt.subplots(figsize=(4.8, 4.9), dpi=600)

            norm = ImageNormalize(img_data,
                                  vmin=np.percentile(img_data, 1),
                                  vmax=np.percentile(img_data, 99.7),
                                  stretch=LogStretch())
            ax.imshow(img_data, cmap="gray_r", origin="lower",
                      extent=img_extent, norm=norm)

            try:
                ax.contour(X, Y, kappa_disp, levels=levels,
                           colors='C1', linewidths=1.1)
            except Exception:
                kmax = np.nanmax(kappa_disp)
                lvls = ([0.3 * kmax, 0.6 * kmax]
                        if np.isfinite(kmax) and kmax > 0
                        else [0.02, 0.04])
                ax.contour(X, Y, kappa_disp, levels=lvls,
                           colors='C1', linewidths=1.1)

            if peaks:
                for i, (px, py) in enumerate(peaks, start=1):
                    ax.plot(px, py, marker='o', ms=3.5,
                            mfc='none', mec='r', mew=1.1)
                    t = ax.text(px, py, f"{i}",
                                ha='center', va='center',
                                fontsize=7, color='r', fontweight='bold')
                    t.set_path_effects(
                        [pe.withStroke(linewidth=1.5, foreground='w')])

            ax.set_xlabel("RA offset (arcsec)")
            ax.set_ylabel("Dec offset (arcsec)")

            lv_str = (", ".join([f"{lv:.2f}" for lv in levels[:5]])
                      + ("…" if len(levels) > 5 else ""))
            ax.text(0.02, 0.98, rf"$\kappa$ levels: {lv_str}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2',
                              fc='white', ec='0.2', lw=0.8))

            _draw_compass_and_scalebar(ax, img_extent, z_lens=z_lens)

            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            ax.grid(False)

            fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.98))
            if save_pdf_path is not None:
                fig.savefig(save_pdf_path, bbox_inches="tight", format="pdf")
                plt.close(fig)
            else:
                return fig, ax

        # Background
        img_data = self.get_image_data()
        img_extent = (0.0, img_data.shape[1] * self.CDELT,
                      0.0, img_data.shape[0] * self.CDELT)

        # Model kappa from current lenses
        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, lens_type=self.lens_type,
            source_redshift=self.z_source,
        )

        # Peaks within 300 kpc
        peaks, _ = utils.find_peaks_and_masses(
            kappa, z_lens=self.z_cluster, z_source=self.z_source,
            radius_kpc=300,
        )

        # Fixed kappa levels per cluster (cached so all signal choices
        # share the same color scale)
        if not hasattr(self, "_kappa_levels"):
            self._kappa_levels = {}
        cache_key = (self.cluster_name, self.lens_type)
        if cache_key not in self._kappa_levels:
            self._kappa_levels[cache_key] = _compute_levels_from_kappa(
                kappa, positive=True)
        levels = self._kappa_levels[cache_key]

        # Title and total-mass label depend on lens_type
        if self.lens_type == 'NFW':
            total_mass_hinv = float(np.nansum(
                getattr(self.lenses, "mass", np.array([np.nan]))))
            title_main = (rf"{self.cluster_name}: JWST WL Reconstruction "
                          rf"(NFW, {self.signal_choice})")
        else:  # POWER_LAW
            total_mass_hinv = np.nan  # not directly comparable
            slope_med = float(np.median(getattr(self.lenses, "slope",
                                                np.array([np.nan]))))
            title_main = (rf"{self.cluster_name}: JWST WL Reconstruction "
                          rf"(power-law, $\langle n\rangle={slope_med:.2f}$, "
                          rf"{self.signal_choice})")

        save_main = (Path(self.output_dir)
                     / f"{self.cluster_name}_clu_{self.signal_choice}"
                     f"_{self.lens_type}.pdf")

        _plot_single_panel(
            img_data=img_data, img_extent=img_extent,
            X=X, Y=Y, kappa=kappa, levels=levels, peaks=peaks,
            title=title_main, sum_mass_hinv=total_mass_hinv,
            z_lens=self.z_cluster, smooth_sigma=1.0,
            save_pdf_path=str(save_main), cluster_name=self.cluster_name,
        )

        # Mass comparison (NFW only — the utility expects .mass)
        if self.lens_type == 'NFW':
            utils.compare_mass_estimates(
                self.lenses,
                Path(self.output_dir) / (f"mass_{self.cluster_name}"
                                          f"_{self.signal_choice}.pdf"),
                f"Mass Comparison: {self.cluster_name} "
                f"(signals: {self.signal_choice})",
                self.cluster_name,
            )

    def get_image_data(self):
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data

    def import_lenses(self):
        file_name = (self.output_dir
                     / f"lenses_{self.cluster_name}_{self.signal_choice}"
                     f"_{self.lens_type}.csv")
        if self.lens_type == 'NFW':
            lens = halo_obj.NFW_Lens(
                x=[], y=[], z=[], mass=[], concentration=[],
                redshift=self.z_cluster, chi2=[],
            )
        elif self.lens_type == 'POWER_LAW':
            lens = halo_obj.PowerLawHalo(
                x=[], y=[], kappa_star=[], slope=[],
                theta_star=self.theta_star,
                redshift=self.z_cluster, chi2=[],
            )
        lens.import_from_csv(file_name)
        self.lenses = lens


# ======================================================================
# NFW vs POWER_LAW comparison helper
# ======================================================================

def run_nfw_vs_power_law(base_config):
    """
    Run the full pipeline twice on the same cluster — once with NFW,
    once with POWER_LAW — and produce both reconstructions for direct
    comparison.

    Both runs share the same source catalog and the same signal_choice,
    so any differences in the recovered mass map come purely from the
    profile assumption.

    Parameters
    ----------
    base_config : dict
        A standard JWSTPipeline config (without the `lens_type` key).
        The function injects `lens_type` for each branch.

    Returns
    -------
    nfw_pipeline, pl_pipeline : JWSTPipeline
        The two finished pipeline objects, each holding their fitted
        lenses.  Useful for further side-by-side analysis.
    """
    print(f"\n{'='*64}\n  NFW reconstruction\n{'='*64}")
    nfw_config = dict(base_config); nfw_config['lens_type'] = 'NFW'
    nfw_pipe = JWSTPipeline(nfw_config)
    nfw_pipe.run()
    nfw_pipe.visualize()

    print(f"\n{'='*64}\n  POWER_LAW reconstruction\n{'='*64}")
    pl_config = dict(base_config); pl_config['lens_type'] = 'POWER_LAW'
    pl_pipe = JWSTPipeline(pl_config)
    pl_pipe.run()
    pl_pipe.visualize()

    # Print a brief comparison summary
    print(f"\n{'='*64}\n  COMPARISON SUMMARY\n{'='*64}")
    print(f"  Cluster      : {base_config['cluster_name']}")
    print(f"  Signal choice: {base_config['signal_choice']}")
    print(f"  NFW halos    : {nfw_pipe.lenses.x.size}  "
          f"(median M = {np.nanmedian(nfw_pipe.lenses.mass):.2e} h^-1 M_sun)")
    print(f"  PL halos     : {pl_pipe.lenses.x.size}  "
          f"(median k* = {np.nanmedian(pl_pipe.lenses.kappa_star):.3f}, "
          f"<n> = {np.nanmedian(pl_pipe.lenses.slope):.3f})")

    # Match recovered halos by position to flag agreement / disagreement
    if nfw_pipe.lenses.x.size > 0 and pl_pipe.lenses.x.size > 0:
        used = set()
        max_match_dist = 30.0  # arcsec — generous tolerance
        print(f"\n  Position matching (within {max_match_dist}\"):")
        for i in range(nfw_pipe.lenses.x.size):
            best_d, best_j = np.inf, -1
            for j in range(pl_pipe.lenses.x.size):
                if j in used:
                    continue
                d = np.hypot(nfw_pipe.lenses.x[i] - pl_pipe.lenses.x[j],
                             nfw_pipe.lenses.y[i] - pl_pipe.lenses.y[j])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < max_match_dist:
                used.add(best_j)
                print(f"    NFW halo {i}  ({nfw_pipe.lenses.x[i]:.1f}, "
                      f"{nfw_pipe.lenses.y[i]:.1f})  <->  "
                      f"PL halo {best_j}  ({pl_pipe.lenses.x[best_j]:.1f}, "
                      f"{pl_pipe.lenses.y[best_j]:.1f})  "
                      f"sep={best_d:.2f}\"")
            else:
                print(f"    NFW halo {i}  ({nfw_pipe.lenses.x[i]:.1f}, "
                      f"{nfw_pipe.lenses.y[i]:.1f})  -> no PL match")
        unmatched = [j for j in range(pl_pipe.lenses.x.size) if j not in used]
        for j in unmatched:
            print(f"    PL halo {j}   ({pl_pipe.lenses.x[j]:.1f}, "
                  f"{pl_pipe.lenses.y[j]:.1f})  -> no NFW match")

    return nfw_pipe, pl_pipe


# ======================================================================
# CLI entry
# ======================================================================

if __name__ == '__main__':
    signals = ['all', 'shear_f', 'f_g', 'shear_g']

    # Set to 'NFW', 'POWER_LAW', or 'COMPARE' to run both
    run_mode = 'COMPARE'

    for signal in signals:
        abell_config = {
            'flexion_catalog_path': 'Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
            'image_path': 'Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
            'output_dir': 'Output/JWST/ABELL/',
            'cluster_name': 'ABELL_2744',
            'cluster_redshift': 0.308,
            'source_redshift': 0.8,
            'signal_choice': signal,
        }

        el_gordo_config = {
            'flexion_catalog_path': 'Data/JWST/EL_GORDO/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'Data/JWST/EL_GORDO/Catalogs/stacked_cat.ecsv',
            'image_path': 'Data/JWST/EL_GORDO/Image_Data/stacked.fits',
            'output_dir': 'Output/JWST/EL_GORDO/',
            'cluster_name': 'EL_GORDO',
            'cluster_redshift': 0.873,
            'source_redshift': 1.2,
            'signal_choice': signal,
        }

        for cfg in (abell_config, el_gordo_config):
            if run_mode == 'COMPARE':
                run_nfw_vs_power_law(cfg)
            else:
                cfg['lens_type'] = run_mode
                pipe = JWSTPipeline(cfg)
                pipe.run()
                pipe.visualize()