import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from pathlib import Path
import warnings

# Import custom modules (ensure these are in your Python path)
import main
import source_obj
import halo_obj
import utils

# Set matplotlib style
plt.style.use('scientific_presentation.mplstyle')  # Ensure this style file exists

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Redshifts
hubble_param = 0.67 # Hubble constant

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
        
        # Create redshifts as a global variable
        global z_source
        z_source = config['source_redshift']
        global z_cluster
        z_cluster = config['cluster_redshift']

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
        self.match_sources()
        self.initialize_sources()
        # self.run_lens_fitting()
        # self.save_results()
        self.lenses = halo_obj.NFW_Lens([0], [0], [0], [0], [0], [0], [0])
        self.plot_results()

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
        Reads the flexion catalog and filters out bad data.
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
        rs = df['rs'].to_numpy() # We don't need to carry this past this function
        print(f"Read {len(self.IDs)} entries from flexion catalog.")

        phi_correction = np.where(self.phi < 0, self.phi + 2*np.pi, self.phi)
        # Apply the correction to phi
        self.phi = phi_correction

        # Eliminate all entries with chi2 > 1.5, that are inappropriate sizes, or with rs > 10 - also remove NaNs
        bad_rs = (rs > 10)
        bad_chi2 = self.chi2 > 1.5
        bad_a = (self.a < 0.01) | (self.a > 20) #if self.cluster_name == 'EL_GORDO' else (self.a < 0.1) | (self.a > 10)
        nan_indices = np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | np.isnan(self.a) | np.isnan(self.chi2) | np.isnan(self.q) | np.isnan(self.phi) | np.isnan(self.G1_fit) | np.isnan(self.G2_fit)
        # Remove NaN entries

        bad_indices = (bad_chi2) | (bad_a) | (bad_rs) | (nan_indices) #| bad_q # Combine all bad indices
        
        self.IDs = self.IDs[~bad_indices]
        self.q = self.q[~bad_indices]
        self.phi = self.phi[~bad_indices]
        self.F1_fit = self.F1_fit[~bad_indices]
        self.F2_fit = self.F2_fit[~bad_indices]
        self.G1_fit = self.G1_fit[~bad_indices]
        self.G2_fit = self.G2_fit[~bad_indices]
        self.a = self.a[~bad_indices]
        self.chi2 = self.chi2[~bad_indices]

    def initialize_sources(self):
        """
        Prepares the Source object with calculated lensing signals and uncertainties.
        """
        # Convert positions to arcseconds
        self.xc *= self.CDELT
        self.yc *= self.CDELT

        # Calculate shear components
        shear_magnitude = (self.q - 1) / (self.q + 1)
        e1 = shear_magnitude * np.cos(2 * self.phi)
        e2 = shear_magnitude * np.sin(2 * self.phi)

        # Center coordinates
        self.centroid_x = np.mean(self.xc) # Store centroid for later use
        self.centroid_y = np.mean(self.yc)
        self.xc -= self.centroid_x
        self.yc -= self.centroid_y

        # Use dummy values for uncertainties to initialize Source object
        dummy = np.ones_like(e1) 

        # Create Source object
        self.sources = source_obj.Source(
            x=self.xc, y=self.yc,
            e1=e1, e2=e2,
            f1=self.F1_fit, f2=self.F2_fit,
            g1=self.G1_fit, g2=self.G2_fit,  
            sigs=dummy, sigf=dummy, sigg=dummy
        )
        
        max_flexion = 0.5
        # Remove sources with flexion > max_flexion
        bad_indices = self.sources.filter_sources(max_flexion=max_flexion)
        self.a = np.delete(self.a, bad_indices) # We also need to remove the bad indices from the a array (not part of the source object)

        sigs = np.full_like(self.sources.e1, np.mean([np.std(self.sources.e1), np.std(self.sources.e2)]))
        sigaf = np.mean([np.std(self.a * self.sources.f1), np.std(self.a * self.sources.f2)])
        sigag = np.mean([np.std(self.a * self.sources.g1), np.std(self.a * self.sources.g2)])
        sigf, sigg = sigaf / (self.a + 1e-10), sigag / (self.a + 1e-10)

        # Update Source object with new uncertainties
        self.sources.sigs = sigs
        self.sources.sigf = sigf
        self.sources.sigg = sigg

    def match_sources(self):
        """
        Matches flexion data with source positions based on IDs.
        This is necessary to get the positions of the sources on the image.
        """
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        matched_indices = []
        for ID in self.IDs:
            idx = label_to_index.get(ID)
            if idx is not None:
                matched_indices.append(idx)
            else:
                warnings.warn(f"ID {ID} not found in source catalog.")
                matched_indices.append(None)

        # Extract matched positions
        self.xc = np.array([
            self.x_centroids[idx] if idx is not None else np.nan
            for idx in matched_indices
        ])
        self.yc = np.array([
            self.y_centroids[idx] if idx is not None else np.nan
            for idx in matched_indices
        ])

    def run_lens_fitting(self):
        """
        Runs the lens fitting pipeline.
        """
        xmax = np.max(np.hypot(self.sources.x, self.sources.y))
        
        self.lenses, _ = main.fit_lensing_field(
            self.sources, xmax, flags=True, use_flags=self.use_flags, lens_type='NFW', z_lens=z_cluster, z_source=z_source
        )
        
        self.lenses.x += self.centroid_x    # Adjust lens positions back to original coordinates
        self.lenses.y += self.centroid_y
        self.lenses.mass *= hubble_param # Convert mass to M_sun h^-1
        self.sources.x += self.centroid_x # Move sources back to original coordinates
        self.sources.y += self.centroid_y

    def save_results(self):
        """
        Saves the lenses and sources to files.
        """
        self.lenses.export_to_csv(self.output_dir / 'lenses.csv')
        self.sources.export_to_csv(self.output_dir / 'sources.csv')

    def plot_results(self):
        """
        Plots the convergence map overlaid on the JWST image.
        """
        img_data = self.get_image_data()
        img_extent = [
            0, img_data.shape[1] * self.CDELT,
            0, img_data.shape[0] * self.CDELT
        ]

        self.sources.x += self.centroid_x
        self.sources.y += self.centroid_y
        # Adjust source positions to match image coordinates

        # Load lens positions
        if self.lenses is None:
            self.lenses = halo_obj.NFW_Lens([0], [0], [0], [0], [0], [0], [0])
            self.lenses.import_from_csv(self.output_dir / 'lenses.csv')
        # Load source positions
        if self.sources is None:
            self.sources = source_obj.Source([0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0])
            self.sources.import_from_csv(self.output_dir / 'sources.csv')

        # Calculate convergence map
        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, lens_type='NFW', source_redshift=z_source
        )

        # Plot settings
        def plot_cluster(convergence, title, save_name):
            fig, ax = plt.subplots(figsize=(10, 10))
            norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

            # Display image
            ax.imshow(
                img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
            )
            
            # Overlay convergence contours
            contour_levels = np.linspace(0, 0.1, 10)
            contours = ax.contour(
                convergence[0], convergence[1], convergence[2], levels=contour_levels, cmap='plasma', linewidths=1.5
            )
            # Add colorbar for contours
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')
            cbar = plt.colorbar(contours, ax=ax)

            # Plot lens positions
            # ax.scatter(self.lenses.x, self.lenses.y, s=50, facecolors='none', edgecolors='red', label='Lenses')

            # Labels and title
            ax.set_xlabel('RA Offset (arcsec)')
            ax.set_ylabel('Dec Offset (arcsec)')
            ax.set_title(title)
            # Save and display
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_name, dpi=300)

        # Plot the cluster
        title = r'Mass Reconstruction of {} with JWST - {}'.format(self.cluster_name, self.signal_choice) + '\n' + r'Total Mass = {:.2e} $h^{{-1}} M_\odot$'.format(np.sum(self.lenses.mass))
        # plot_cluster([X,Y,kappa], title, self.output_dir / '{}_clu_{}.png'.format(self.cluster_name, self.signal_choice))
        
        plot_name = self.output_dir / 'mass_{}_{}.png'.format(self.cluster_name, self.signal_choice)
        plot_title = 'Mass Comparison of {} with JWST Data \n Signal used: - {}'.format(self.cluster_name, self.signal_choice)

        # Compare mass estimates
        # utils.compare_mass_estimates(self.lenses, plot_name, plot_title, self.cluster_name)

        # Create a comparison by doing a kaiser squires transformation to get kappa from the flexion
        avg_source_density = len(self.sources.x) / (np.pi/4 * np.max(np.hypot(self.sources.x, self.sources.y))**2)
        smoothing_scale = 1 / (avg_source_density)**0.5

        kappa_extent = [min(self.sources.x), max(self.sources.x), min(self.sources.y), max(self.sources.y)]
        X, Y, kappa_ks = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='flexion', smoothing_sigma=smoothing_scale)
        title = 'Kaiser-Squires Flexion Reconstruction of {} with JWST'.format(self.cluster_name)
        save_title = self.output_dir / 'ks_flex_{}.png'.format(self.cluster_name)
        plot_cluster([X,Y,kappa_ks], title, save_title)

        # Do this for the shear as well
        X, Y, kappa_shear = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='shear', smoothing_sigma=smoothing_scale)
        title = 'Kaiser-Squires Shear Reconstruction of {} with JWST'.format(self.cluster_name)
        save_title = self.output_dir / 'ks_shear_{}.png'.format(self.cluster_name)
        plot_cluster([X,Y,kappa_shear], title, save_title)
        

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
    output_file = Path('Output/JWST/ABELL/combined_results.csv')
    with open(output_file, 'w') as f:
        f.write("Signal\tCluster\tX\tY\tMass\n")

    for signal in signals:
        abell_config = {
            'flexion_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
            'image_path': 'JWST_Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
            'output_dir': 'Output/JWST/ABELL/',
            'cluster_name': 'ABELL_2744',
            'cluster_redshift': 0.308,
            'source_redshift': 1,
            'signal_choice': signal
        }

        el_gordo_config = {
            'flexion_catalog_path': 'JWST_Data/JWST/EL_GORDO/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'JWST_Data/JWST/EL_GORDO/Catalogs/stacked_cat.ecsv',
            'image_path': 'JWST_Data/JWST/EL_GORDO/Image_Data/stacked.fits',
            'output_dir': 'Output/JWST/EL_GORDO/',
            'cluster_name': 'EL_GORDO',
            'cluster_redshift': 0.870,
            'source_redshift': 4.25,
            'signal_choice': signal
        }

        # Initialize and run the pipeline
        pipeline_el_gordo = JWSTPipeline(el_gordo_config)
        pipeline_abell = JWSTPipeline(abell_config)

        pipeline_el_gordo.run()
        pipeline_abell.run()
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
        raise SystemExit("Exiting after running for all signals. Remove this line to run for each signal separately.")