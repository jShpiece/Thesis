import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from pathlib import Path
import warnings

import csv
import concurrent.futures
from functools import partial
import tqdm

# Import custom modules (ensure these are in your Python path)
import main
import source_obj
import halo_obj
import utils
import pipeline

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
        def plot_cluster(convergence, title, save_name, peaks=None, masses=None):
            fig, ax = plt.subplots(figsize=(10, 10))
            
            norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

            # Display JWST image
            ax.imshow(
                img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
            )
            
            # Overlay convergence contours
            contour_levels = np.percentile(convergence[2], np.linspace(70, 99, 5))
            contours = ax.contour(
                convergence[0], convergence[1], convergence[2], levels=contour_levels,
                cmap='plasma', linewidths=1.5, alpha=0.8
            )
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')

            # Draw a countour where kappa = 0
            ax.contour(
                convergence[0], convergence[1], convergence[2],
                levels=[0], colors='black', linewidths=1.5, linestyles='dashed'
            )

            # Overlay mass peaks if provided
            if peaks is not None and masses is not None:
                for i, ((ra_offset, dec_offset), mass) in enumerate(zip(peaks, masses)):
                    ax.plot(ra_offset, dec_offset, 'ro', markersize=5, label='Mass Peak' if i == 0 else "")
                    ax.text(
                        ra_offset + 1, dec_offset + 1,
                        r"$M_{<300 kpc} = %.2f \times 10^{13}\ M_\odot$" % (mass / 1e13),
                        color='black', fontsize=10, weight='bold', ha='left', va='bottom'
                    )
            
            # Axes, title, legend
            ax.set_xlabel('RA Offset (arcsec)')
            ax.set_ylabel('Dec Offset (arcsec)')
            ax.set_title(title)
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            
            plt.tight_layout()
            # plt.show()
            plt.savefig(save_name, dpi=300)
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
        plot_cluster([X,Y,kappa], title, self.output_dir / '{}_clu_{}.png'.format(self.cluster_name, self.signal_choice), peaks=peaks, masses=masses)

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
        plot_cluster([X,Y,kappa_flexion], title, save_title, peaks=peaks, masses=masses)

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
        plot_cluster([X,Y,kappa_shear], title, save_title, peaks=peaks, masses=masses)
        '''
        # Look at the first lens, see how chi2 changes with mass
        lens_SE = halo_obj.NFW_Lens(
            x=self.lenses.x[0],
            y=self.lenses.y[0],
            z=self.z_cluster,
            concentration=self.lenses.concentration[0],
            mass=self.lenses.mass[0],
            redshift=self.z_cluster,
            chi2=self.lenses.chi2[0]
        )
        lens_NW = halo_obj.NFW_Lens(
            x=self.lenses.x[1],
            y=self.lenses.y[1],
            z=self.z_cluster,
            concentration=self.lenses.concentration[1],
            mass=self.lenses.mass[1],
            redshift=self.z_cluster,
            chi2=self.lenses.chi2[1]
        )

        # We're going to vary the mass of this lens and see how the chi2 value changes
        mass_values = np.logspace(13,16,1000)
        chi2_values_SE = []
        chi2_values_NW = []
        for mass_value in mass_values:
            lens_SE.mass[0] = mass_value
            lens_NW.mass[0] = mass_value
            chi2_SE = pipeline.update_chi2_values(self.sources, lens_SE, use_flags=self.use_flags, lens_type='NFW',z_source=self.z_source)
            chi2_NW = pipeline.update_chi2_values(self.sources, lens_NW, use_flags=self.use_flags, lens_type='NFW',z_source=self.z_source)
            chi2_values_SE.append(chi2_SE)
            chi2_values_NW.append(chi2_NW)

        # Plot
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # For SE lens
        chi2_SE = np.array(chi2_values_SE)
        mean_SE = np.mean(chi2_SE)
        std_SE = np.std(chi2_SE)
        # Mask out extreme spikes above mean + 5*std for plotting
        mask_SE = chi2_SE < (mean_SE + 5 * std_SE)
        axs[0].plot(mass_values[mask_SE], chi2_SE[mask_SE])
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Mass [M_sun]')
        axs[0].set_ylabel('Chi2')
        axs[0].set_title('Chi2 vs Mass for SE Lens')
        axs[0].grid()
        # Let's add a point at the minimum value
        min_index = np.argmin(chi2_SE)
        axs[0].scatter(mass_values[min_index], chi2_SE[min_index], color='red')
        # Also add a line where the actual lens is located
        axs[0].axvline(x=self.lenses.mass[0], color='blue', linestyle='--')

        # For NW lens
        chi2_NW = np.array(chi2_values_NW)
        mean_NW = np.mean(chi2_NW)
        std_NW = np.std(chi2_NW)
        mask_NW = chi2_NW < (mean_NW + 5 * std_NW)
        axs[1].plot(mass_values[mask_NW], chi2_NW[mask_NW])
        axs[1].set_xscale('log')
        axs[1].set_xlabel('Mass [M_sun]')
        axs[1].set_ylabel('Chi2')
        axs[1].set_title('Chi2 vs Mass for NW Lens')
        axs[1].grid()
        # Let's add a point at the minimum value
        min_index = np.argmin(chi2_NW)
        axs[1].scatter(mass_values[min_index], chi2_NW[min_index], color='red')
        # Also add a line where the actual lens is located
        axs[1].axvline(x=self.lenses.mass[1], color='blue', linestyle='--')

        plt.tight_layout()
        plt.show()
        '''

    def get_image_data(self):
        """
        Reads image data from a FITS file.
        """
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data


if __name__ == '__main__':
    #while True:
        #print('This machine is a useless piece of shit and is lucky Jacob doesnt have a hammer nearby')
    # Configuration dictionary
    signals = ['all', 'shear_f', 'f_g', 'shear_g']
    # Create an output file to store all the results
    '''
    output_file = Path('Output/JWST/ABELL/combined_results.csv')
    with open(output_file, 'w') as f:
        f.write("Signal\tCluster\tX\tY\tMass\n")
    '''

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
        # pipeline_abell.run()
        # pipeline_abell.compute_error_bars()
        # Save results to the output file
        '''
        with open(output_file, 'a') as f:
            for i in range(len(pipeline_abell.lenses.x)):
                f.write(f"{signal}\t{pipeline_abell.cluster_name}\t{pipeline_abell.lenses.x[i]:.2f}\t{pipeline_abell.lenses.y[i]:.2f}\t{pipeline_abell.lenses.mass[i]:.2e}\n")
            for i in range(len(pipeline_el_gordo.lenses.x)):
                f.write(f"{signal}\t{pipeline_el_gordo.cluster_name}\t{pipeline_el_gordo.lenses.x[i]:.2f}\t{pipeline_el_gordo.lenses.y[i]:.2f}\t{pipeline_el_gordo.lenses.mass[i]:.2e}\n")
        # Print completion message
        '''
        print(f"Finished running pipeline for signal choice: {signal}")