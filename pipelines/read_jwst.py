import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import matplotlib.patheffects as pe
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from astropy import units as u
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

        # Strong-lensing configuration (off by default — backward
        # compatible with existing NFW configs).  When the catalog
        # path is provided, the pipeline loads the systems via
        # load_a2744_sl_catalog.load_strong_lensing_systems and
        # passes use_strong_lensing=True to fit_lensing_field.
        self.strong_lensing_catalog_path = config.get(
            'strong_lensing_catalog_path', None)
        self.sl_qf_min = int(config.get('sl_qf_min', 2))
        self.sl_qf_max = int(config.get('sl_qf_max', 3))
        self.sl_keep_locations = tuple(
            config.get('sl_keep_locations', ('BGCs',)))
        self.sl_wl_reference_radec = config.get(
            'sl_wl_reference_radec', None)  # (RA, Dec) fallback if no WCS
        self.use_strong_lensing = bool(self.strong_lensing_catalog_path)

        # File-naming suffix to disambiguate WL-only vs WL+SL outputs.
        # Used in CSV / PDF filenames so the same (cluster, signal,
        # lens_type) can have separate outputs for SL ablations.
        self.sl_suffix = 'WLSL' if self.use_strong_lensing else 'WL'

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
                       f"{self.signal_choice}_{self.lens_type}_"
                       f"{self.sl_suffix}.csv")
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

        # Load strong-lensing systems if requested.  Must be done AFTER
        # the centroid is computed because SL positions need to be
        # co-registered with the WL frame (i.e., centroid-subtracted
        # the same way Source.x/y are).
        if self.use_strong_lensing:
            self._load_strong_lensing_systems()

    def _load_strong_lensing_systems(self):
        """
        Parse the SL catalog and attach systems to self.sources.
        Co-registers SL positions with the WL frame using either FITS
        WCS (if available) or a tangent-plane projection around a
        provided reference (RA, Dec).
        """
        # Defer the import so users without the loader file installed
        # can still use the WL-only paths.
        try:
            from pipelines.load_a2744_sl_catalog import (
                load_strong_lensing_systems, print_sl_summary,
            )
        except ImportError:
            try:
                # Same directory as read_jwst.py (when invoked as a script)
                import sys as _sys
                from pathlib import Path as _Path
                _sys.path.insert(0, str(_Path(__file__).parent))
                from load_a2744_sl_catalog import (
                    load_strong_lensing_systems, print_sl_summary,
                )
            except ImportError as e:
                raise ImportError(
                    "Could not import load_a2744_sl_catalog.  Place "
                    "load_a2744_sl_catalog.py in pipelines/ next to "
                    "read_jwst.py, or install it on the Python path."
                ) from e

        systems, diag = load_strong_lensing_systems(
            catalog_path=self.strong_lensing_catalog_path,
            cdelt_arcsec_per_pix=self.CDELT,
            centroid_x_arcsec=self.centroid_x,
            centroid_y_arcsec=self.centroid_y,
            fits_path=self.image_path,
            wl_reference_radec=self.sl_wl_reference_radec,
            qf_min=self.sl_qf_min,
            qf_max=self.sl_qf_max,
            keep_locations=self.sl_keep_locations,
            require_zspec=True,
        )
        print_sl_summary(systems, diag,
                         header_text=f"SL catalog ({self.cluster_name})")

        # Attach to Source so fit_lensing_field can find them
        self.sources.strong_systems = systems

        # Coordinate-alignment sanity check.  Print the spatial overlap
        # between the WL source distribution and the SL image positions.
        # If they don't overlap (e.g., misaligned by tens of arcsec),
        # the SL constraints will pull the recovered halos away from
        # the actual cluster center and inflate masses to match — a
        # silent failure mode that's hard to diagnose post-hoc.
        self._print_sl_alignment_check(systems)

    def _print_sl_alignment_check(self, systems):
        """
        Diagnostic printout of WL vs SL spatial extent in the centered
        frame.  If SL positions don't overlap WL sources, the fit will
        be wrong; this prints the ranges so you can verify visually.
        """
        if not systems:
            return

        wl_x = np.asarray(self.sources.x)
        wl_y = np.asarray(self.sources.y)
        sl_x = np.concatenate([np.atleast_1d(s.theta_x) for s in systems])
        sl_y = np.concatenate([np.atleast_1d(s.theta_y) for s in systems])

        # WL field extent (already centroid-subtracted)
        wl_xmin, wl_xmax = float(wl_x.min()), float(wl_x.max())
        wl_ymin, wl_ymax = float(wl_y.min()), float(wl_y.max())
        sl_xmin, sl_xmax = float(sl_x.min()), float(sl_x.max())
        sl_ymin, sl_ymax = float(sl_y.min()), float(sl_y.max())

        # Compute centroid offset between SL and WL
        wl_cx, wl_cy = float(wl_x.mean()), float(wl_y.mean())
        sl_cx, sl_cy = float(sl_x.mean()), float(sl_y.mean())
        offset = np.hypot(sl_cx - wl_cx, sl_cy - wl_cy)

        # SL image fraction within WL extent
        in_field = ((sl_x >= wl_xmin) & (sl_x <= wl_xmax)
                    & (sl_y >= wl_ymin) & (sl_y <= wl_ymax))
        n_in_field = int(in_field.sum())
        n_total = sl_x.size

        print(f"\n=== Coordinate alignment check ({self.cluster_name}) ===")
        print(f"  WL source extent (arcsec): "
              f"x=[{wl_xmin:+7.1f}, {wl_xmax:+7.1f}], "
              f"y=[{wl_ymin:+7.1f}, {wl_ymax:+7.1f}]")
        print(f"  SL image extent  (arcsec): "
              f"x=[{sl_xmin:+7.1f}, {sl_xmax:+7.1f}], "
              f"y=[{sl_ymin:+7.1f}, {sl_ymax:+7.1f}]")
        print(f"  WL centroid: ({wl_cx:+.2f}, {wl_cy:+.2f})  (should be ~ 0,0)")
        print(f"  SL centroid: ({sl_cx:+.2f}, {sl_cy:+.2f})")
        print(f"  Centroid offset: {offset:.1f} arcsec")
        print(f"  SL images inside WL field: {n_in_field}/{n_total} "
              f"({100 * n_in_field / n_total:.0f}%)")

        # Diagnose alignment status.  Two failure modes:
        #   1. Few SL images inside WL field (< 50%): genuine
        #      coordinate misalignment.
        #   2. Most SL images in field but centroid offset is large:
        #      could be either misalignment OR a real cluster offset
        #      (the WL source-distribution centroid does NOT have to
        #      coincide with the cluster mass centroid; sources are
        #      background galaxies whose distribution depends on
        #      photometric depth, not cluster mass).
        if n_in_field < 0.5 * n_total:
            print(f"  *** WARNING: < 50% of SL images fall within the "
                  f"WL source field.  Likely coordinate misalignment.  "
                  f"Check the FITS WCS or sl_wl_reference_radec.")
        elif offset > 30.0:
            print(f"  Note: SL centroid is offset {offset:.0f} arcsec "
                  f"from the WL source centroid, but {100*n_in_field/n_total:.0f}% "
                  f"of SL images sit inside the WL field.  This is "
                  f"normal — the WL source centroid is set by photometric "
                  f"depth, not cluster mass, and the cluster's actual "
                  f"BCG/mass center can sit several tens of arcsec from "
                  f"the source-distribution centroid.")
        else:
            print(f"  Alignment looks reasonable.")

    # ------------------------------------------------------------------
    # Lens fitting — dispatches by lens_type
    # ------------------------------------------------------------------

    def run_lens_fitting(self, flags=True):
        xmax = np.max(np.hypot(self.sources.x, self.sources.y))

        if self.lens_type == 'NFW':
            self.lenses, _ = main.fit_lensing_field(
                self.sources, xmax, flags=flags, use_flags=self.use_flags,
                lens_type='NFW', z_lens=self.z_cluster,
                use_strong_lensing=self.use_strong_lensing,
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
                use_strong_lensing=self.use_strong_lensing,
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
                     f"_{self.lens_type}_{self.sl_suffix}.csv")
        self.lenses.export_to_csv(file_name)

        # Sanity-check diagnostic — print recovered parameters per halo
        # so the user can see if anything is bound-pinned or
        # implausibly large.
        self._print_summary()

    def _print_summary(self):
        """Print a per-halo summary of the fitted lenses."""
        if self.lenses is None or self.lenses.x.size == 0:
            print("  (no halos recovered)")
            return

        if self.lens_type == 'NFW':
            print(f"\n  NFW fit summary ({self.lenses.x.size} halos):")
            print(f"  {'idx':>3} {'x':>8} {'y':>8} "
                  f"{'mass':>13} {'conc':>6}")
            print("  " + "-" * 50)
            for i in range(self.lenses.x.size):
                print(f"  {i:>3d} "
                      f"{self.lenses.x[i]:>8.2f} {self.lenses.y[i]:>8.2f} "
                      f"{self.lenses.mass[i]:>13.3e} "
                      f"{self.lenses.concentration[i]:>6.2f}")
            print(f"  Sum mass: {np.nansum(self.lenses.mass):.3e} M_sun")

        elif self.lens_type == 'POWER_LAW':
            # Bound-pinning detection — uses the tightened production
            # bounds (slope in (0.4, 1.7), kappa_star in (1e-6, 10)).
            bound_tol = 1.0e-2
            slope_lo, slope_hi = 0.4, 1.7
            kappa_lo, kappa_hi = 1.0e-6, 10.0
            k_pinned = (
                (self.lenses.kappa_star < kappa_lo * (1 + bound_tol))
                | (self.lenses.kappa_star > kappa_hi * (1 - bound_tol))
            )
            n_pinned = (
                (self.lenses.slope < slope_lo + bound_tol)
                | (self.lenses.slope > slope_hi - bound_tol)
            )

            theta_E_arr = self.lenses.calc_theta_E()

            # M(<250 kpc) per halo — this radius is more useful than
            # 100 kpc because it sits in the regime where weak lensing
            # actually constrains the profile (sources at JWST cluster
            # distances typically span ~30-200 arcsec from a halo,
            # which is roughly 100-800 kpc at z~0.3-0.9).  The 100 kpc
            # reference would extrapolate inward of most sources and
            # give artificially-tight error estimates.
            kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(
                float(self.lenses.redshift)).to(u.kpc / u.arcsec).value
            ref_radius_kpc = 250.0
            theta_ref = ref_radius_kpc / kpc_per_arcsec
            M_ref_arr = np.array([
                float(self.lenses.calc_mass_2d(theta_ref, self.z_source)[i])
                for i in range(self.lenses.x.size)
            ])

            print(f"\n  POWER_LAW fit summary ({self.lenses.x.size} halos):")
            print(f"  {'idx':>3} {'x':>8} {'y':>8} {'kappa*':>8} "
                  f"{'n':>6} {'theta_E':>10} {'M(<250kpc)':>13}  flag")
            print("  " + "-" * 75)
            for i in range(self.lenses.x.size):
                flags = []
                if k_pinned[i]:
                    flags.append("k_pinned")
                if n_pinned[i]:
                    flags.append("n_pinned")
                flag_str = "+".join(flags) if flags else ""
                print(f"  {i:>3d} "
                      f"{self.lenses.x[i]:>8.2f} {self.lenses.y[i]:>8.2f} "
                      f"{self.lenses.kappa_star[i]:>8.4f} "
                      f"{self.lenses.slope[i]:>6.3f} "
                      f"{theta_E_arr[i]:>9.2f}\" "
                      f"{M_ref_arr[i]:>13.3e}  {flag_str}")
            print("  " + "-" * 75)
            print(f"  Total: kappa_star pinned: {int(k_pinned.sum())}/{self.lenses.x.size}, "
                  f"slope pinned: {int(n_pinned.sum())}/{self.lenses.x.size}")
            print(f"  Median: kappa_star = {np.median(self.lenses.kappa_star):.4f}, "
                  f"slope = {np.median(self.lenses.slope):.3f}, "
                  f"theta_E = {np.median(theta_E_arr):.2f}\"")
            print(f"  Sum M(<{ref_radius_kpc:.0f}kpc): "
                  f"{np.nansum(M_ref_arr):.3e} M_sun")

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

        # Model kappa from current lenses.  We mask the inner
        # `inner_mask_radius` arcsec around each halo center because
        # neither profile is data-constrained closer than the typical
        # source distance from a halo (~5 arcsec for JWST).  For
        # POWER_LAW especially, the kappa profile diverges as r^(-n) so
        # the inner pixels are pure model extrapolation that would
        # otherwise dominate the contour-level computation.  Both NFW
        # and POWER_LAW are masked equally for a fair comparison.
        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, lens_type=self.lens_type,
            source_redshift=self.z_source,
        )
        inner_mask_radius = float(getattr(self, 'inner_mask_radius', 5.0))
        if inner_mask_radius > 0 and self.lenses.x.size > 0:
            for k in range(self.lenses.x.size):
                R = np.hypot(X - self.lenses.x[k], Y - self.lenses.y[k])
                kappa = np.where(R < inner_mask_radius, 0.0, kappa)

        # Peaks within 300 kpc.  Compute these BEFORE masking would
        # affect them (they're already past the masked inner region
        # since peak detection finds large-scale halo centers).
        peaks, _ = utils.find_peaks_and_masses(
            kappa, z_lens=self.z_cluster, z_source=self.z_source,
            radius_kpc=300,
        )

        # Kappa contour levels.  We use QUANTILE-based levels of the
        # unmasked, positive kappa values: each contour separates a
        # specific percentile band, so the contours always show the
        # actual structure of the reconstruction regardless of the
        # absolute kappa range.
        #
        # Why not absolute levels [0.05, 0.10, 0.20, 0.50, 1.00]:
        # produced uninformative contour distributions on real data —
        # the field is dominated by low kappa with localized spikes
        # near halos, so most of the absolute range is empty and only
        # 1-2 contours render.  Why not max-fractional levels:
        # similar issue — kappa_max is set by the inner spike near a
        # halo, and the rest of the field has kappa << kappa_max, so
        # the lower-fractional contours all sit below where any
        # extended structure lives.
        #
        # Quantile levels guarantee each contour band corresponds to
        # a meaningful fraction of pixels, giving 5 visible contours
        # that span the actual data distribution.  Defaults: 50%, 70%,
        # 85%, 94%, 98% percentiles (concentrating the bands toward
        # the high-kappa tail where structure lives).
        #
        # Override per-cluster: set self.kappa_levels in the config or
        # instance before calling visualize() to use absolute levels.
        if not hasattr(self, "_kappa_levels"):
            self._kappa_levels = {}
        cache_key = (self.cluster_name, self.lens_type)
        if cache_key not in self._kappa_levels:
            if hasattr(self, 'kappa_levels') and self.kappa_levels is not None:
                # User-supplied absolute levels
                self._kappa_levels[cache_key] = list(self.kappa_levels)
            else:
                kappa_finite = kappa[np.isfinite(kappa) & (kappa > 0)]
                if kappa_finite.size > 20:
                    quantiles = [0.50, 0.70, 0.85, 0.94, 0.98]
                    levels_q = [
                        float(np.quantile(kappa_finite, q)) for q in quantiles]
                    # De-duplicate (in case the field is nearly uniform)
                    levels_q = sorted(set(round(lv, 4) for lv in levels_q))
                    if len(levels_q) >= 3:
                        self._kappa_levels[cache_key] = levels_q
                    else:
                        # Field too uniform for quantiles; fall back to
                        # max-fractional
                        kappa_max = float(np.max(kappa_finite))
                        self._kappa_levels[cache_key] = [
                            f * kappa_max
                            for f in [0.20, 0.40, 0.60, 0.80, 0.95]]
                else:
                    # Field nearly empty (heavy masking); fallback
                    self._kappa_levels[cache_key] = [
                        0.05, 0.10, 0.20, 0.50, 1.00]
        levels = self._kappa_levels[cache_key]

        # Title and total-mass label depend on lens_type
        # Title prefix indicating WL-only vs WL+SL reconstruction
        sl_label = "WL+SL" if self.use_strong_lensing else "WL-only"

        if self.lens_type == 'NFW':
            total_mass_hinv = float(np.nansum(
                getattr(self.lenses, "mass", np.array([np.nan]))))
            title_main = (rf"{self.cluster_name}: JWST {sl_label} Reconstruction "
                          rf"(NFW, {self.signal_choice})")
        else:  # POWER_LAW
            total_mass_hinv = np.nan
            slope_med = float(np.median(getattr(self.lenses, "slope",
                                                np.array([np.nan]))))
            title_main = (rf"{self.cluster_name}: JWST {sl_label} Reconstruction "
                          rf"(power-law, $\langle n\rangle={slope_med:.2f}$, "
                          rf"{self.signal_choice})")

        save_main = (Path(self.output_dir)
                     / f"{self.cluster_name}_clu_{self.signal_choice}"
                     f"_{self.lens_type}_{self.sl_suffix}.pdf")

        _plot_single_panel(
            img_data=img_data, img_extent=img_extent,
            X=X, Y=Y, kappa=kappa, levels=levels, peaks=peaks,
            title=title_main, sum_mass_hinv=total_mass_hinv,
            z_lens=self.z_cluster, smooth_sigma=1.0,
            save_pdf_path=str(save_main), cluster_name=self.cluster_name,
        )

        # Mass comparison — now supports both NFW and POWER_LAW
        utils.compare_mass_estimates(
            self.lenses,
            Path(self.output_dir) / (f"mass_{self.cluster_name}"
                                      f"_{self.signal_choice}"
                                      f"_{self.lens_type}"
                                      f"_{self.sl_suffix}.pdf"),
            f"Mass Comparison: {self.cluster_name} "
            f"({self.lens_type}, {sl_label}, signals: {self.signal_choice})",
            self.cluster_name,
            lens_type=self.lens_type,
        )

    def get_image_data(self):
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data

    def import_lenses(self):
        file_name = (self.output_dir
                     / f"lenses_{self.cluster_name}_{self.signal_choice}"
                     f"_{self.lens_type}_{self.sl_suffix}.csv")
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

# Cluster definitions — paths are relative to the repository root
# (the parent of `pipelines/` and `arch/`).  Override with --repo-root
# if invoking from elsewhere.
CLUSTERS = {
    'ABELL_2744': {
        'flexion_catalog_path': 'Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl',
        'source_catalog_path':  'Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
        'image_path':           'Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
        'output_dir':           'Output/JWST/ABELL/',
        'cluster_redshift':     0.308,
        'source_redshift':      0.8,
        # Strong-lensing catalog: Bergamini+2023 Table A.1
        # Set strong_lensing_catalog_path=None to disable SL.
        'strong_lensing_catalog_path':
            'Data/JWST/ABELL_2744/Catalogs/apjacd643t2_mrt.txt',
        # Fallback (RA, Dec) for tangent-plane projection if FITS WCS
        # is unavailable.  This is the approximate cluster center
        # (BCG-S region) and only used as a fallback.
        'sl_wl_reference_radec': (3.58833, -30.40014),
    },
    'EL_GORDO': {
        'flexion_catalog_path': 'Data/JWST/EL_GORDO/Catalogs/multiband_flexion.pkl',
        'source_catalog_path':  'Data/JWST/EL_GORDO/Catalogs/stacked_cat.ecsv',
        'image_path':           'Data/JWST/EL_GORDO/Image_Data/stacked.fits',
        'output_dir':           'Output/JWST/EL_GORDO/',
        'cluster_redshift':     0.873,
        'source_redshift':      1.2,
        # No SL catalog for El Gordo yet
        'strong_lensing_catalog_path': None,
    },
}

VALID_SIGNALS = ['all', 'shear_f', 'f_g', 'shear_g']
VALID_MODES = ['NFW', 'POWER_LAW', 'COMPARE']
VALID_ACTIONS = ['fit', 'visualize', 'errorbars']


def _build_config(cluster_name, signal, theta_star, repo_root,
                  use_strong_lensing=True):
    """Construct a JWSTPipeline config dict for a (cluster, signal) pair.

    Strong lensing is enabled when (a) the cluster has a SL catalog
    in CLUSTERS, AND (b) use_strong_lensing=True.  Set
    use_strong_lensing=False to force a WL-only run regardless of
    catalog availability (useful for ablation studies).
    """
    base = CLUSTERS[cluster_name]
    cfg = {
        'flexion_catalog_path': str(repo_root / base['flexion_catalog_path']),
        'source_catalog_path':  str(repo_root / base['source_catalog_path']),
        'image_path':           str(repo_root / base['image_path']),
        'output_dir':           str(repo_root / base['output_dir']),
        'cluster_name':         cluster_name,
        'cluster_redshift':     base['cluster_redshift'],
        'source_redshift':      base['source_redshift'],
        'signal_choice':        signal,
        'theta_star':           theta_star,
    }
    # SL config: forward path + reference RA/Dec from the cluster
    # definition.  When use_strong_lensing=False, force the path to
    # None so the pipeline runs WL-only.
    sl_path = base.get('strong_lensing_catalog_path', None)
    if use_strong_lensing and sl_path:
        cfg['strong_lensing_catalog_path'] = str(repo_root / sl_path)
        cfg['sl_wl_reference_radec'] = base.get(
            'sl_wl_reference_radec', None)
    else:
        cfg['strong_lensing_catalog_path'] = None
    return cfg


def _run_one(cfg, mode, action):
    """Run a single (cluster, signal, mode, action) job."""
    if mode == 'COMPARE':
        if action != 'fit':
            raise ValueError(
                "--mode COMPARE only supported with --action fit (it runs both "
                "NFW and POWER_LAW from scratch).  For visualize/errorbars, "
                "pick a specific lens type.")
        return run_nfw_vs_power_law(cfg)

    cfg = dict(cfg)
    cfg['lens_type'] = mode
    pipe = JWSTPipeline(cfg)
    if action == 'fit':
        pipe.run()
        pipe.visualize()
    elif action == 'visualize':
        pipe.visualize()
    elif action == 'errorbars':
        pipe.compute_error_bars()
    return pipe


def main_cli():
    import argparse
    p = argparse.ArgumentParser(
        prog='python -m pipelines.read_jwst',
        description=(
            "JWST cluster lensing reconstruction with NFW and/or POWER_LAW "
            "profiles.  Run from the repository root (the parent of "
            "`pipelines/` and `arch/`)."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Default sweep — both clusters, all 4 signals, NFW vs POWER_LAW comparison\n"
            "  python -m pipelines.read_jwst\n"
            "\n"
            "  # Just Abell 2744, all signals, POWER_LAW only\n"
            "  python -m pipelines.read_jwst --cluster ABELL_2744 --mode POWER_LAW\n"
            "\n"
            "  # One run end to end (Abell, 'all', power-law, theta_star=50\")\n"
            "  python -m pipelines.read_jwst --cluster ABELL_2744 \\\n"
            "      --signal all --mode POWER_LAW --theta-star 50\n"
            "\n"
            "  # Re-render plots from saved CSVs (no re-fitting)\n"
            "  python -m pipelines.read_jwst --cluster ABELL_2744 \\\n"
            "      --signal all --mode NFW --action visualize\n"
            "\n"
            "  # Jackknife error bars on a previously fit reconstruction\n"
            "  python -m pipelines.read_jwst --cluster ABELL_2744 \\\n"
            "      --signal all --mode NFW --action errorbars\n"
        ),
    )
    p.add_argument('--cluster', choices=list(CLUSTERS.keys()) + ['ALL'],
                   default='ALL',
                   help="Which cluster to run.  Default: ALL.")
    p.add_argument('--signal', choices=VALID_SIGNALS + ['ALL'],
                   default='ALL',
                   help="Which signal combination to use.  Default: ALL "
                        "(sweeps over all 4).")
    p.add_argument('--mode', choices=VALID_MODES, default='COMPARE',
                   help="Which profile to fit.  COMPARE runs both NFW and "
                        "POWER_LAW.  Default: COMPARE.")
    p.add_argument('--action', choices=VALID_ACTIONS, default='fit',
                   help="What to do: 'fit' runs the pipeline and renders "
                        "plots; 'visualize' re-renders plots from saved "
                        "CSV; 'errorbars' runs the jackknife.  Default: fit.")
    p.add_argument('--theta-star', type=float, default=30.0,
                   dest='theta_star',
                   help="POWER_LAW pivot radius in arcsec.  Default: 30.")
    p.add_argument('--no-sl', action='store_true', dest='no_sl',
                   help="Disable strong lensing even for clusters that "
                        "have an SL catalog configured.  Useful for "
                        "WL-only ablation studies.")
    p.add_argument('--repo-root', type=str, default=None, dest='repo_root',
                   help="Repository root path (the parent of `pipelines/` "
                        "and `arch/`).  Default: auto-detected from this "
                        "file's location.")
    args = p.parse_args()

    # Resolve repository root
    if args.repo_root is None:
        # This file lives at <repo_root>/pipelines/read_jwst.py
        repo_root = Path(__file__).resolve().parent.parent
    else:
        repo_root = Path(args.repo_root).resolve()

    if not (repo_root / 'arch').is_dir():
        warnings.warn(
            f"Repository root {repo_root!s} does not contain an `arch/` "
            f"directory.  This may indicate the path is wrong.  Set "
            f"--repo-root explicitly if needed.")

    # Resolve cluster and signal lists
    clusters = list(CLUSTERS.keys()) if args.cluster == 'ALL' else [args.cluster]
    signals  = VALID_SIGNALS         if args.signal  == 'ALL' else [args.signal]
    use_strong_lensing = not args.no_sl

    # Print a short header so the user can see what they're about to do
    n_jobs = len(clusters) * len(signals)
    print(f"\n{'='*64}")
    print(f"  ARCH JWST runner")
    print(f"{'='*64}")
    print(f"  repo root  : {repo_root}")
    print(f"  clusters   : {clusters}")
    print(f"  signals    : {signals}")
    print(f"  mode       : {args.mode}")
    print(f"  action     : {args.action}")
    if args.mode == 'POWER_LAW' or args.mode == 'COMPARE':
        print(f"  theta_star : {args.theta_star}")
    print(f"  strong-lensing: {'enabled' if use_strong_lensing else 'DISABLED'}")
    print(f"  total jobs : {n_jobs}")
    print()

    for cluster_name in clusters:
        for signal in signals:
            print(f"\n--- {cluster_name} / signal={signal} / "
                  f"mode={args.mode} / action={args.action} ---")
            cfg = _build_config(cluster_name, signal,
                                args.theta_star, repo_root,
                                use_strong_lensing=use_strong_lensing)
            try:
                _run_one(cfg, args.mode, args.action)
            except FileNotFoundError as e:
                print(f"  SKIPPED: missing input data ({e})")
            except Exception as e:
                print(f"  FAILED ({type(e).__name__}): {e}")
                # Continue with the next job rather than aborting the whole sweep
                continue


if __name__ == '__main__':
    main_cli() 