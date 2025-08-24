import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from matplotlib.patches import FancyArrow
from scipy.ndimage import gaussian_filter
from pathlib import Path
import warnings
import csv
import concurrent.futures
import tqdm

# Import custom modules (ensure these are in your Python path)
import main
import source_obj
import utils


# Set matplotlib style
plt.style.use('scientific_presentation.mplstyle')  # Ensure this style file exists

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
        '''
        # Override real lensing with simulated signal based on literature results for el gordo
        
        simulated_cluster_x = [110.0, 30.0] # First entry SE, second entry NW
        simulated_cluster_y = [35.0, 120.0] 
        simulated_cluster_mass = [5e14, 5e14]
        simulated_lenses = halo_obj.NFW_Lens(
            x=simulated_cluster_x, y=simulated_cluster_y, z=np.zeros_like(simulated_cluster_x), 
            concentration=np.ones_like(simulated_cluster_x), mass=simulated_cluster_mass, 
            redshift=self.z_cluster, chi2 = np.zeros_like(simulated_cluster_x)
            )
        
        simulated_lenses.x -= self.centroid_x # Center simulated lenses around (0, 0)
        simulated_lenses.y -= self.centroid_y
        simulated_lenses.calculate_concentration()
        self.lenses = simulated_lenses
        self.sources.zero_lensing_signals()
        self.sources.apply_noise()
        self.sources.apply_lensing(simulated_lenses, lens_type='NFW', z_source=self.z_source)
        
        self.output_dir = Path(self.config['output_dir'] + "Simulated/")
        '''

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

        print(np.log10(self.lenses.mass))

    def plot_results(self):
        """
        Plots the convergence map overlaid on the JWST image.
        """
        img_data = self.get_image_data()
        img_extent = [
            0, img_data.shape[1] * self.CDELT,
            0, img_data.shape[0] * self.CDELT
        ]
        
        # Plot settings
        def plot_cluster_kappa(
            img_data, img_extent, kx, ky, kappa,
            title, save_name_pdf,
            peaks=None, masses=None,
            z_lens=None, cosmo=None,
            levels=None,  # e.g., levels=[0.02,0.04,0.06,0.08,0.10]
            zero_contour=True,
            smooth_sigma=None  # e.g., 1.0 pixels
        ):
            """
            Create MNRAS-ready figure: JWST grayscale background + kappa contours,
            with fixed contour levels, scalebar, and compass arrows.

            Parameters
            ----------
            img_data : 2D array
                Background image (JWST).
            img_extent : (xmin, xmax, ymin, ymax) in arcsec offsets
                Extent for imshow.
            kx, ky : 2D arrays (arcsec offsets)
                Meshgrid coordinates for kappa.
            kappa : 2D array
                Convergence field.
            title : str
                Figure title (short).
            save_name_pdf : str
                Path to save vector output (PDF).
            peaks : list of (x, y) arcsec offsets, optional
            masses : list of floats in Msun, optional
            z_lens : float, optional
                Cluster redshift for physical scale conversion.
            cosmo : astropy.cosmology, optional
            levels : list of floats, optional
                Fixed contour levels in kappa; required for inter-panel consistency.
            zero_contour : bool
                Draw bold dashed kappa=0 contour.
            smooth_sigma : float, optional
                Gaussian sigma in pixels for smoothing kappa before contouring.
            """

            # Prepare kappa field
            kappa_disp = gaussian_filter(kappa, smooth_sigma) if smooth_sigma else kappa

            # Figure
            fig, ax = plt.subplots(figsize=(4.25, 4.25))  # ~8.9 cm width typical MNRAS column

            # Background image
            norm_img = ImageNormalize(img_data, vmin=np.percentile(img_data, 1),
                                    vmax=np.percentile(img_data, 99.7),
                                    stretch=LogStretch())
            ax.imshow(img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm_img)

            # Contours
            if levels is None:
                # Fallback (mildly robust), but prefer fixed levels passed in
                finite = np.isfinite(kappa_disp)
                q = np.quantile(kappa_disp[finite], [0.80, 0.88, 0.92, 0.96, 0.98])
                levels = [lv for lv in q if lv > 0]
                if len(levels) == 0:
                    levels = [np.nanmax(kappa_disp)*0.2, np.nanmax(kappa_disp)*0.4]

            cs = ax.contour(kx, ky, kappa_disp, levels=levels, colors='C1', linewidths=1.2)
            # Optional labels: uncomment if not too busy
            # ax.clabel(cs, inline=True, fontsize=8, fmt='%.3f')

            if zero_contour:
                try:
                    ax.contour(kx, ky, kappa_disp, levels=[0], colors='k', linewidths=2.0, linestyles='--')
                except Exception:
                    pass  # if kappa never crosses 0

            # Peaks
            if peaks is not None and masses is not None and len(peaks) == len(masses):
                for i, ((x0, y0), m) in enumerate(zip(peaks, masses), start=1):
                    ax.plot(x0, y0, marker='o', ms=3.5, mfc='none', mec='r', mew=1.1)
                    label = rf"{i}: $M_{{<300\,\mathrm{{kpc}}}}={(m/1e13):.2f}\times 10^{{13}}\,M_\odot$"
                    ax.annotate(label, xy=(x0, y0), xytext=(x0+8, y0+8),
                                textcoords='data',
                                fontsize=8, color='k',
                                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.2', lw=0.8),
                                arrowprops=dict(arrowstyle='->', lw=0.8, color='0.2'))

            # Axes
            ax.set_xlabel('RA offset (arcsec)')
            ax.set_ylabel('Dec offset (arcsec)')
            ax.set_title(title, fontsize=11)

            # Compass (N/E)
            # Small arrows at ~10% in from edge
            xmin, xmax, ymin, ymax = img_extent
            dx = xmax - xmin
            dy = ymax - ymin
            base_x = xmin + 0.1 * dx
            base_y = ymin + 0.1 * dy
            ax.add_patch(FancyArrow(base_x, base_y, 0, 0.08*dy, width=0.0, length_includes_head=True, head_width=0.02*dx, head_length=0.04*dy, color='k'))
            ax.text(base_x, base_y + 0.09*dy, 'N', ha='center', va='bottom', fontsize=8)
            ax.add_patch(FancyArrow(base_x, base_y, 0.08*dx, 0, width=0.0, length_includes_head=True, head_width=0.02*dy, head_length=0.04*dx, color='k'))
            ax.text(base_x + 0.09*dx, base_y, 'E', ha='left', va='center', fontsize=8)

            # Scale bar (arcsec and kpc)
            bar_len_arcsec = 50.0  # adjust to field scale
            bar_x0 = xmax - 0.1*dx - bar_len_arcsec
            bar_y0 = ymin + 0.08*dy
            ax.plot([bar_x0, bar_x0 + bar_len_arcsec], [bar_y0, bar_y0], color='k', lw=1.8)
            if z_lens is not None and cosmo is not None:
                # Physical scale (kpc/arcsec)
                kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_lens).value / 60.0
                ax.text(bar_x0 + bar_len_arcsec/2, bar_y0 + 0.02*dy,
                        f"{bar_len_arcsec:.0f}\" ({bar_len_arcsec*kpc_per_arcsec:.0f} kpc)",
                        ha='center', va='bottom', fontsize=8)
            else:
                ax.text(bar_x0 + bar_len_arcsec/2, bar_y0 + 0.02*dy,
                        f"{bar_len_arcsec:.0f}\"", ha='center', va='bottom', fontsize=8)

            # Inset legend for levels (optional)
            # Create a small text box listing contour levels
            levels_str = ", ".join([f"{lv:.2f}" for lv in levels[:4]]) + ("…" if len(levels) > 4 else "")
            ax.text(0.02, 0.98, rf"$\kappa$ levels: {levels_str}",
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.2', lw=0.8))

            # Final polish
            ax.grid(False)  # no grid on imaging
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

            fig.tight_layout(pad=0.5)
            fig.savefig(save_name_pdf, dpi=600, bbox_inches='tight', format='pdf', transparent=False)
            plt.close(fig)

        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, lens_type='NFW', source_redshift=self.z_source
        )

        peaks, masses = utils.find_peaks_and_masses(
            kappa, 
            z_lens=self.z_cluster, z_source=self.z_source,
            radius_kpc=300
        )
        title = r'Mass Reconstruction of {} with JWST - {}'.format(self.cluster_name, self.signal_choice) + '\n' + r'Total Mass = {:.2e} $h^{{-1}} M_\odot$'.format(np.sum(self.lenses.mass))
        plot_cluster_kappa([X,Y,kappa], title, self.output_dir / '{}_clu_{}.png'.format(self.cluster_name, self.signal_choice), peaks=peaks, masses=masses)

        # Compare mass estimates
        utils.compare_mass_estimates(self.lenses, self.output_dir / 'mass_{}_{}.png'.format(self.cluster_name, self.signal_choice), 
                                'Mass Comparison of {} with JWST Data \n Signal used: - {}'.format(self.cluster_name, self.signal_choice), self.cluster_name)

        # Create a comparison by doing a kaiser squires transformation to get kappa from the flexion
        # only do this for all signals - it won't vary with the signal choice
        if self.signal_choice != 'all':
            print("Skipping Kaiser-Squires reconstruction for non-'all' signal choices.")
            return

        avg_source_density = len(self.sources.x) / (np.pi/4 * np.max(np.hypot(self.sources.x, self.sources.y))**2)
        smoothing_scale = 1 / (avg_source_density)**0.5
        kappa_extent = [min(self.sources.x), max(self.sources.x), min(self.sources.y), max(self.sources.y)]

        weights_flexion = self.sources.sigf**-2
        X, Y, kappa_flexion = utils.perform_kaiser_squire_reconstruction(
            self.sources, extent=kappa_extent, signal='flexion', 
            smoothing_scale=smoothing_scale, weights=weights_flexion, apodize=True)
        k_val = utils.estimate_mass_sheet_factor(kappa_flexion)  # Mass sheet transformation parameter
        kappa_flexion = utils.mass_sheet_transformation(kappa_flexion, k=k_val)
        peaks, masses = utils.find_peaks_and_masses(
            kappa_flexion,
            z_lens=self.z_cluster, z_source=self.z_source,
            radius_kpc=300
        )
        title = 'Kaiser-Squires Flexion Reconstruction of {} with JWST'.format(self.cluster_name)
        save_title = self.output_dir / 'ks_flex_{}.png'.format(self.cluster_name)
        plot_cluster_kappa([X,Y,kappa_flexion], title, save_title, peaks=peaks, masses=masses)

        # Do this for the shear as well
        X, Y, kappa_shear = utils.perform_kaiser_squire_reconstruction(
            self.sources, extent=kappa_extent, signal='shear', 
            smoothing_scale=smoothing_scale)
        k_val = utils.estimate_mass_sheet_factor(kappa_shear)  # Mass sheet transformation parameter
        kappa_shear = utils.mass_sheet_transformation(kappa_shear, k=k_val)
        peaks, masses = utils.find_peaks_and_masses(
            kappa_shear,
            z_lens=self.z_cluster, z_source=self.z_source,
            radius_kpc=300
        )
        title = 'Kaiser-Squires Shear Reconstruction of {} with JWST'.format(self.cluster_name)
        save_title = self.output_dir / 'ks_shear_{}.png'.format(self.cluster_name)
        plot_cluster_kappa([X,Y,kappa_shear], title, save_title, peaks=peaks, masses=masses)


    def get_image_data(self):
        """
        Reads image data from a FITS file.
        """
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data


if __name__ == '__main__':
    # Configuration dictionary
    signals = ['all', 'shear_f', 'f_g', 'shear_g']
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

        pipeline_el_gordo.run()
        pipeline_abell.run()
        print(f"Finished running pipeline for signal choice: {signal}")