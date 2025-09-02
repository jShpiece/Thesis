import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
from matplotlib.gridspec import GridSpec

# Import custom modules 
import main
import source_obj
import halo_obj
import utils

# Set matplotlib style
plt.style.use('scientific_presentation.mplstyle')  # Ensure this style file exists

# Suppress specific warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)

# Redshifts
hubble_param = 0.67 # Hubble constant

def unpack_and_run_jackknife(args):
    # Unpack the arguments and run the jackknife worker function
    return jackknife_worker(*args)

def run_fitting_with_sources(self_obj, sources):
    # This function runs the lens fitting with a modified set of sources.
    # Used in the jackknife procedure.
    original_sources = self_obj.sources
    self_obj.sources = sources
    self_obj.run_lens_fitting(flags=False)
    result = self_obj.lenses.copy()
    self_obj.sources = original_sources
    return result

# Top-level worker function: accepts self explicitly
def jackknife_worker(i, self_obj, sources):
    modified_sources = sources.copy()
    modified_sources.remove(i)
    lenses = run_fitting_with_sources(self_obj, modified_sources)
    return [
        [i, x, y, m, c]
        for x, y, m, c in zip(lenses.x, lenses.y, lenses.mass, lenses.concentration)
    ]

class JWSTPipeline:
    """
    A class to encapsulate the JWST lensing pipeline.
    """
    def __init__(self, config):
        """
        Initializes the pipeline with the given configuration.

        Parameters:
            config (dict): Configuration parameters for the pipeline.
        """
        self.config = config
        self.CDELT = 8.54006306703281e-6 * 3600  # degrees/pixel converted to arcsec/pixel

        # Paths
        self.flexion_catalog_path = Path(config['flexion_catalog_path'])
        self.source_catalog_path = Path(config['source_catalog_path'])
        self.image_path = Path(config['image_path'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cluster_name = config['cluster_name']
        self.signal_choice = config['signal_choice'] # Need to interpret this after reading in
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

        # Create redshifts as instance variables
        self.z_source = config['source_redshift']
        self.z_cluster = config['cluster_redshift']

        # Data placeholders
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
        self.xc = None # Source positions
        self.yc = None
        self.lenses = None
        self.sources = None
        self.centroid_x = None # Centroid among all sources
        self.centroid_y = None

    def run(self):
        """    
        Runs the entire pipeline.
        """
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()
        self.run_lens_fitting()

    def visualize(self):
        '''
        Visualizes the lensing results.
        '''
        self.read_source_catalog()
        self.read_flexion_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()
        self.sources.x += self.centroid_x # Move sources back to original coordinates
        self.sources.y += self.centroid_y
        self.import_lenses()
        self.plot_results()

    def compute_error_bars(self):
        """
        Performs jackknife resampling to compute error bars in parallel.
        """
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()

        output_path = f"jackknife_results_{self.cluster_name}_{self.signal_choice}.csv"
        with open(output_path, "w", newline='') as f:
            csv.writer(f).writerow(["i", "x", "y", "M200", "concentration"])

        sources_copy = self.sources.copy()  # Create a copy of the sources to avoid modifying the original
        n_sources = len(sources_copy.x)
        indices_to_use = np.random.choice(n_sources, size=200, replace=False)  # Randomly select 200 indices for jackknife

        # Prepare args list for executor
        args_list = [(i, self, sources_copy) for i in indices_to_use]

        with open(output_path, "a", newline='') as f:
            writer = csv.writer(f)
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for result in tqdm.tqdm(executor.map(unpack_and_run_jackknife, args_list, chunksize = 3), total=len(indices_to_use), desc="Computing error bars"):
                    writer.writerows(result)

    def read_source_catalog(self):
        """
        Reads the source catalog.
        """
        table = Table.read(self.source_catalog_path)
        self.x_centroids = np.array(table['xcentroid'])
        self.y_centroids = np.array(table['ycentroid'])
        self.labels = np.array(table['label'])

    def read_flexion_catalog(self):
            """
            Reads the flexion catalog.
            """
            # Read the flexion catalog
            df = pd.read_pickle(self.flexion_catalog_path)
            # Extract relevant columns
            self.IDs = df['label'].to_numpy()
            self.q = df['q'].to_numpy()
            self.phi = df['phi'].to_numpy()
            self.F1_fit = df['F1_fit'].to_numpy()
            self.F2_fit = df['F2_fit'].to_numpy() 
            self.G1_fit = df['G1_fit'].to_numpy() 
            self.G2_fit = df['G2_fit'].to_numpy()
            self.a = df['a'].to_numpy()
            self.chi2 = df['rchi2'].to_numpy()
            self.rs = df['rs'].to_numpy() # We don't need to carry this past this function

            # Convert to arcseconds / inverse arcseconds - do this step *immediately*, there are no circumstances where we want to keep the data in pixels
            self.a *= self.CDELT
            self.F1_fit /= self.CDELT
            self.F2_fit /= self.CDELT
            self.G1_fit /= self.CDELT
            self.G2_fit /= self.CDELT
            self.rs *= self.CDELT
            print(f"Read {len(self.IDs)} entries from flexion catalog.")

    def cut_flexion_catalog(self):
        # Compute flexion magnitude and dimensionless flexion
        F = np.hypot(self.F1_fit, self.F2_fit)
        aF = self.a * F  # Dimensionless flexion

        '''
        # Updated thresholds - use for ABELL_2744
        max_aF = 0.5                 # More conservative flexion threshold
        max_rs = 5.0                 # Tighter Sérsic radius cutoff
        max_chi2 = 1.5               # Chi2 fit quality
        min_a, max_a = 0.01, 2.0     # Size range
        '''

        # Updated thresholds - use for EL_GORDO
        max_aF = 0.5                 # More conservative flexion threshold
        max_rs = 5.0                # Tighter Sérsic radius cutoff
        max_chi2 = 1.5               # Chi2 fit quality
        min_a, max_a = 0.01, 2.0     # Size range

        # Boolean masks for bad data
        bad_flexion = aF > max_aF
        bad_rs = self.rs > max_rs
        bad_chi2 = self.chi2 > max_chi2
        bad_a = (self.a < min_a) | (self.a > max_a)
        nan_indices = np.isnan(self.IDs) | np.isnan(self.q) | np.isnan(self.phi) | \
                    np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | \
                    np.isnan(self.G1_fit) | np.isnan(self.G2_fit) | \
                    np.isnan(self.a) | np.isnan(self.rs) | np.isnan(self.chi2)

        bad_indices = bad_flexion | bad_rs | bad_chi2 | bad_a | nan_indices 
        # Apply cuts
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

        # Measures should carry the things we want to plot, and have a name attribute for plotting
        # Recalculate aF
        '''
        aF = self.a * np.hypot(self.F1_fit, self.F2_fit)
        measures = [
            self.q, self.phi, self.F1_fit, self.F2_fit,
            self.G1_fit, self.G2_fit, self.a, self.rs, self.chi2, aF
        ]
        names = [
            'q', 'phi', 'F1_fit', 'F2_fit',
            'G1_fit', 'G2_fit', 'a', 'rs', 'chi2', 'aF'
        ]

        for measure, name in zip(measures, names):
            plt.figure(figsize=(10, 6))
            plt.hist(measure, bins=50, color='blue', alpha=0.7)
            plt.title(f'Distribution of {name}')
            plt.xlabel(name)
            plt.ylabel("Frequency")
            plt.grid()
            plt.savefig(self.output_dir / f'distribution_{name}.png')
        plt.close('all')
        '''
        print(f"Filtered flexion catalog to {len(self.IDs)} entries after applying updated cuts.")

    def match_sources(self):
        """
        Matches each object in self.IDs to a position in the source catalog
        defined by self.labels, self.x_centroids, and self.y_centroids.

        Populates:
            self.xc: array of x-centroid positions for matched IDs
            self.yc: array of y-centroid positions for matched IDs

        Notes:
            - If an ID is not found in the source catalog, its position is set to NaN.
            - Note to self - this functions to make the cuts in the source catalog that we made in the flexion catalog.
        """
        # Build a mapping from label → index
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        # Match IDs and extract centroids
        xc_list = []
        yc_list = []

        for ID in self.IDs:
            idx = label_to_index.get(ID)
            if idx is not None:
                xc_list.append(self.x_centroids[idx])
                yc_list.append(self.y_centroids[idx])
            else:
                # Necessary to assign NaN if ID not found - otherwise, xc_list and yc_list will be of different lengths
                # This is a warning, not an error, as we want to continue processing
                # Also this has never happened in the use of this script - it's unlikely to happen in practice
                warnings.warn(f"ID '{ID}' not found in source catalog. Assigning NaN.")
                xc_list.append(np.nan)
                yc_list.append(np.nan)

        # Assign results to attributes
        self.xc = np.array(xc_list)
        self.yc = np.array(yc_list)

        # Convert positions to arcseconds
        self.xc *= self.CDELT 
        self.yc *= self.CDELT

    def initialize_sources(self):
        """
        Prepares the Source object with calculated lensing signals and uncertainties.
        """
        # Calculate shear components
        shear_magnitude = (self.q - 1) / (self.q + 1)
        e1 = shear_magnitude * np.cos(2 * self.phi)
        e2 = shear_magnitude * np.sin(2 * self.phi)

        # Center coordinates (necessary for pipeline)
        self.centroid_x = np.mean(self.xc) # Store centroid for later use
        self.centroid_y = np.mean(self.yc)

        # Use dummy values for uncertainties to initialize Source object
        dummy = np.ones_like(e1) 

        # Create Source object
        self.sources = source_obj.Source(
            x=self.xc, y=self.yc,
            e1=e1, e2=e2,
            f1=self.F1_fit, f2=self.F2_fit,
            g1=self.G1_fit, g2=self.G2_fit,  
            sigs=dummy, sigf=dummy, sigg=dummy, 
            redshift=self.z_source
        )

        sigs = np.full_like(self.sources.e1, np.mean([np.std(self.sources.e1), np.std(self.sources.e2)]))
        sigaf = np.mean([np.std(self.a * self.sources.f1), np.std(self.a * self.sources.f2)]) 
        sigag = np.mean([np.std(self.a * self.sources.g1), np.std(self.a * self.sources.g2)])
        sigf, sigg = sigaf / self.a, sigag / self.a

        # Update Source object with new uncertainties
        self.sources.sigs = sigs
        self.sources.sigf = sigf
        self.sources.sigg = sigg

        self.sources.x -= self.centroid_x # Center sources around (0, 0)
        self.sources.y -= self.centroid_y

    def run_lens_fitting(self, flags=True):
        """
        Runs the lens fitting pipeline.
        """
        xmax = np.max(np.hypot(self.sources.x, self.sources.y))

        self.lenses, _ = main.fit_lensing_field(
            self.sources, xmax, flags=flags, use_flags=self.use_flags, lens_type='NFW', z_lens=self.z_cluster
        )

        check = self.lenses.check_for_nan_properties() # Ensure no NaN properties in lenses
        if check:
            print("Warning: Some lens properties are NaN or Inf. Check the lens fitting process.")

        self.lenses.x += self.centroid_x # Adjust lens positions back to original coordinates
        self.lenses.y += self.centroid_y
        self.lenses.mass *= hubble_param # Convert mass to M_sun h^-1
        self.sources.x += self.centroid_x # Move sources back to original coordinates
        self.sources.y += self.centroid_y
        # Save the lens
        file_name = self.output_dir / f"lenses_{self.cluster_name}_{self.signal_choice}.csv"
        self.lenses.export_to_csv(file_name)

    def plot_results(self):
        """
        Create publication-ready figures (no side table):
        • Grayscale JWST background
        • κ-contours with fixed levels (consistent across panels)
        • Numbered peak markers only (no mass text on the image)
        • Compass (N/E) and scale bar
        • Title and footer caption with ΣM in h^{-1} M_⊙
        Saves vector PDFs into self.output_dir.
        """

        # ---------------- helpers ----------------

        def _fmt_sum_mass_hinv(m_hinv):
            """ΣM = a × 10^{14} h^{-1} M_⊙ (mathtext) or N/A."""
            if not np.isfinite(m_hinv):
                return r"$\sum M =$ N/A"
            return rf"$\sum M = {(m_hinv/1e14):.2f}\times 10^{{14}}\ h^{{-1}}\,M_\odot$"

        def _compute_levels_from_kappa(kappa, positive=True):
            """Stable κ-levels from quantiles of positive κ; fallback if sparse."""
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
                levels = list(np.linspace(0.4, 0.85, 4) * (kmax if np.isfinite(kmax) else 0.1))
            return sorted(set(levels))

        def _draw_compass_and_scalebar(ax, extent, bar_arcsec=50.0, z_lens=None, cosmo=None):
            """Draw N(up)/E(left) arrows and a 50″ scale bar (+kpc if cosmology provided)."""
            xmin, xmax, ymin, ymax = extent
            dx, dy = (xmax - xmin), (ymax - ymin)

            # Compass (bottom-left)
            cx = xmin + 0.12*dx
            cy = ymin + 0.12*dy
            alen = 0.08*max(dx, dy)
            ax.add_patch(FancyArrow(cx, cy, 0,     alen, width=0.0,
                                    head_width=0.035*dx, head_length=0.04*dy,
                                    length_includes_head=True, color='k'))
            ax.text(cx, cy + alen + 0.02*dy, "N", ha="center", va="bottom", fontsize=8)
            ax.add_patch(FancyArrow(cx, cy, -alen, 0,    width=0.0,
                                    head_width=0.035*dy, head_length=0.04*dx,
                                    length_includes_head=True, color='k'))
            ax.text(cx - alen - 0.02*dx, cy, "E", ha="right", va="center", fontsize=8)

            # Scale bar (bottom-right)
            sx1 = xmax - 0.12*dx
            sx0 = sx1 - bar_arcsec
            sy  = ymin + 0.10*dy
            ax.plot([sx0, sx1], [sy, sy], color='k', lw=1.8)
            if (z_lens is not None) and ('COSMO' in globals()) and (COSMO is not None):
                kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(z_lens).value / 60.0
                label = rf"{int(round(bar_arcsec))}″ ({int(round(bar_arcsec*kpc_per_arcsec))} kpc)"
            else:
                label = rf"{int(round(bar_arcsec))}″"
            ax.text(0.5*(sx0+sx1), sy + 0.02*dy, label, ha="center", va="bottom", fontsize=8)

        def _plot_single_panel(
            img_data, img_extent, X, Y, kappa, levels, peaks, title, sum_mass_hinv,
            z_lens=None, smooth_sigma=1.0, save_pdf_path=None
        ):
            """Single clean panel with background, κ-contours, numbered peaks, compass & scalebar."""
            # Smooth κ for display/contours (optional)
            kappa_disp = gaussian_filter(kappa, smooth_sigma) if smooth_sigma else kappa

            # Figure
            fig, ax = plt.subplots(figsize=(4.8, 4.9), dpi=600)  # single-column friendly

            # Background (JWST grayscale)
            norm = ImageNormalize(img_data, vmin=np.percentile(img_data, 1),
                                vmax=np.percentile(img_data, 99.7), stretch=LogStretch())
            ax.imshow(img_data, cmap="gray_r", origin="lower", extent=img_extent, norm=norm)

            # κ-contours (use precomputed fixed levels if available)
            try:
                cs = ax.contour(X, Y, kappa_disp, levels=levels, colors='C1', linewidths=1.1)
            except Exception:
                # Robust fallback
                kmax = np.nanmax(kappa_disp)
                lvls = [0.3*kmax, 0.6*kmax] if np.isfinite(kmax) and kmax > 0 else [0.02, 0.04]
                ax.contour(X, Y, kappa_disp, levels=lvls, colors='C1', linewidths=1.1)

            # Peaks: numeric markers only (no mass text on-figure)
            if peaks:
                for i, (px, py) in enumerate(peaks, start=1):
                    ax.plot(px, py, marker='o', ms=3.5, mfc='none', mec='r', mew=1.1)
                    t = ax.text(px, py, f"{i}", ha='center', va='center', fontsize=7,
                                color='r', fontweight='bold')
                    t.set_path_effects([pe.withStroke(linewidth=1.5, foreground='w')])

            # Labels & title
            ax.set_xlabel("RA offset (arcsec)")
            ax.set_ylabel("Dec offset (arcsec)")
            # ax.set_title(title, fontsize=11)

            # κ-level legend (compact)
            lv_str = ", ".join([f"{lv:.2f}" for lv in levels[:5]]) + ("…" if len(levels) > 5 else "")
            ax.text(0.02, 0.98, rf"$\kappa$ levels: {lv_str}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.2', lw=0.8))

            # Compass & scale bar
            _draw_compass_and_scalebar(ax, img_extent, z_lens=z_lens, cosmo=globals().get('COSMO', None))

            # Footer caption with ΣM (kept off the map area)
            '''
            if sum_mass_hinv is not None:
                fig.text(0.5, 0.01, _fmt_sum_mass_hinv(sum_mass_hinv),
                        ha="center", va="bottom", fontsize=9)
            '''
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
            ax.grid(False)

            fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.98))
            if save_pdf_path is not None:
                fig.savefig(save_pdf_path, bbox_inches="tight", format="pdf")
                plt.close(fig)
            else:
                return fig, ax

        # ---------------- main body ----------------

        # Background image + extent (arcsec)
        img_data = self.get_image_data()
        img_extent = (0.0, img_data.shape[1] * self.CDELT,
                    0.0, img_data.shape[0] * self.CDELT)

        # Model κ from current lenses
        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, lens_type='NFW', source_redshift=self.z_source
        )

        # Peaks within 300 kpc (positions only for plotting)
        peaks, masses = utils.find_peaks_and_masses(
            kappa, z_lens=self.z_cluster, z_source=self.z_source, radius_kpc=300
        )

        # Fixed κ-levels per cluster (cached for consistency)
        if not hasattr(self, "_kappa_levels"):
            self._kappa_levels = {}
        if self.cluster_name not in self._kappa_levels:
            self._kappa_levels[self.cluster_name] = _compute_levels_from_kappa(kappa, positive=True)
        levels = self._kappa_levels[self.cluster_name]

        # Titles, formatting, and save paths
        total_mass_hinv = float(np.nansum(getattr(self.lenses, "mass", np.array([np.nan]))))
        title_main = rf"{self.cluster_name}: JWST Weak-Lensing Mass Reconstruction ({self.signal_choice})"
        save_main  = Path(self.output_dir) / f"{self.cluster_name}_clu_{self.signal_choice}.pdf"

        # Main panel (no table)
        _plot_single_panel(
            img_data=img_data, img_extent=img_extent,
            X=X, Y=Y, kappa=kappa, levels=levels, peaks=peaks,
            title=title_main, sum_mass_hinv=total_mass_hinv,
            z_lens=self.z_cluster, smooth_sigma=1.0, save_pdf_path=str(save_main)
        )

        # Mass comparison (existing utility)
        utils.compare_mass_estimates(
            self.lenses,
            Path(self.output_dir) / f"mass_{self.cluster_name}_{self.signal_choice}.pdf",
            f"Mass Comparison: {self.cluster_name} (signals: {self.signal_choice})",
            self.cluster_name
        )

        # Kaiser–Squires panels (only when combining all signals)
        if self.signal_choice == 'all':
            # Extent from source footprint (arcsec)
            x_min, x_max = float(np.min(self.sources.x)), float(np.max(self.sources.x))
            y_min, y_max = float(np.min(self.sources.y)), float(np.max(self.sources.y))
            ks_extent = [x_min, x_max, y_min, y_max]

            # Simple density-based smoothing scale (arcsec)
            max_r = np.max(np.hypot(self.sources.x, self.sources.y))
            avg_density = len(self.sources.x) / (np.pi/4.0 * max_r**2 + 1e-12)
            smoothing_scale = (1.0 / max(avg_density, 1e-6))**0.5

            # Flexion-only KS
            w_f = getattr(self.sources, "sigf", None)
            w_f = None if w_f is None else (w_f**-2)
            Xf, Yf, kappa_f = utils.perform_kaiser_squire_reconstruction(
                self.sources, extent=ks_extent, signal='flexion',
                smoothing_scale=smoothing_scale, weights=w_f, apodize=True
            )
            kappa_f = utils.mass_sheet_transformation(kappa_f, k=utils.estimate_mass_sheet_factor(kappa_f))
            peaks_f, _m_f = utils.find_peaks_and_masses(
                kappa_f, z_lens=self.z_cluster, z_source=self.z_source, radius_kpc=300
            )
            save_f = Path(self.output_dir) / f"ks_flex_{self.cluster_name}.pdf"
            _plot_single_panel(
                img_data=img_data, img_extent=img_extent,
                X=Xf, Y=Yf, kappa=kappa_f, levels=levels, peaks=peaks_f,
                title=f"{self.cluster_name}: Kaiser–Squires (flexion)",
                sum_mass_hinv=total_mass_hinv, z_lens=self.z_cluster,
                smooth_sigma=1.0, save_pdf_path=str(save_f)
            )

            # Shear-only KS
            Xs, Ys, kappa_s = utils.perform_kaiser_squire_reconstruction(
                self.sources, extent=ks_extent, signal='shear',
                smoothing_scale=smoothing_scale, weights=None, apodize=True
            )
            kappa_s = utils.mass_sheet_transformation(kappa_s, k=utils.estimate_mass_sheet_factor(kappa_s))
            peaks_s, _m_s = utils.find_peaks_and_masses(
                kappa_s, z_lens=self.z_cluster, z_source=self.z_source, radius_kpc=300
            )
            save_s = Path(self.output_dir) / f"ks_shear_{self.cluster_name}.pdf"
            _plot_single_panel(
                img_data=img_data, img_extent=img_extent,
                X=Xs, Y=Ys, kappa=kappa_s, levels=levels, peaks=peaks_s,
                title=f"{self.cluster_name}: Kaiser–Squires (shear)",
                sum_mass_hinv=total_mass_hinv, z_lens=self.z_cluster,
                smooth_sigma=1.0, save_pdf_path=str(save_s)
            )
        else:
            print("Skipping Kaiser–Squires panels (only generated for signal_choice='all').")


    def get_image_data(self):
        """
        Reads image data from a FITS file.
        """
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data

    def import_lenses(self):
        """
        Imports lens data from a CSV file.
        """
        file_name = self.output_dir / f"lenses_{self.cluster_name}_{self.signal_choice}.csv"
        # Create an empty lens
        lens = halo_obj.NFW_Lens(x=[], y=[], z=[], mass=[], concentration=[], redshift=self.z_cluster, chi2=[])
        lens.import_from_csv(file_name)
        self.lenses = lens

if __name__ == '__main__':
    # Configuration dictionary
    signals = ['shear_f', 'f_g', 'shear_g']
    # Create an output file to store all the results

    for signal in signals:
        abell_config = {
            'flexion_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
            'image_path': 'JWST_Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
            'output_dir': 'Output/JWST/ABELL/',
            'cluster_name': 'ABELL_2744',
            'cluster_redshift': 0.308,
            'source_redshift': 0.8,
            'signal_choice': signal
        }

        el_gordo_config = {
            'flexion_catalog_path': 'JWST_Data/JWST/EL_GORDO/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'JWST_Data/JWST/EL_GORDO/Catalogs/stacked_cat.ecsv',
            'image_path': 'JWST_Data/JWST/EL_GORDO/Image_Data/stacked.fits',
            'output_dir': 'Output/JWST/EL_GORDO/',
            'cluster_name': 'EL_GORDO',
            'cluster_redshift': 0.873,
            'source_redshift': 1.2,
            'signal_choice': signal
        }

        # Initialize and run the pipeline
        pipeline_el_gordo = JWSTPipeline(el_gordo_config)
        pipeline_abell = JWSTPipeline(abell_config)

        #pipeline_el_gordo.run()
        #pipeline_el_gordo.visualize()
        #pipeline_abell.run()
        #pipeline_abell.visualize()
        pipeline_el_gordo.compute_error_bars()
        print(f"Finished running pipeline for signal choice: {signal}")