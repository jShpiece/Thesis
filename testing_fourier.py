import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from pathlib import Path
import warnings
from astropy.visualization import hist as fancy_hist


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
        self.cut_flexion_catalog()
        self.match_sources()
        self.find_outliers()
        self.initialize_sources()
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
        # Make cuts in the data based on aF, rs, chi2, and a
        F = np.hypot(self.F1_fit, self.F2_fit) 
        aF = self.a * F # Dimensionless flexion

        # Set thresholds for bad data
        max_flexion = 1.5 # Maximum flexion threshold (dimensionless)
        bad_flexion = (np.abs(F) > max_flexion) # Flexion threshold (no need for absolute value, aF must be positive)
        bad_rs = (self.rs > 10) # Maximum sersic radius threshold (in arcseconds)
        bad_chi2 = self.chi2 > 1.5 # Maximum reduced chi2 threshold - this is a goodness of fit indicator
        bad_a = (self.a > 1) | (self.a < 0.01) # Maximum and minimum scale (in arcseconds) - this is a measure of the size of the source
        nan_indices = np.isnan(self.IDs) | np.isnan(self.q) | np.isnan(self.phi) | np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | np.isnan(self.G1_fit) | np.isnan(self.G2_fit) | np.isnan(self.a) | np.isnan(self.chi2)
        bad_indices = bad_flexion | bad_chi2 | bad_a | nan_indices | bad_rs

        # Remove bad data
        self.IDs = self.IDs[~bad_indices]
        self.q = self.q[~bad_indices]
        self.phi = self.phi[~bad_indices]
        self.F1_fit = self.F1_fit[~bad_indices]
        self.F2_fit = self.F2_fit[~bad_indices]
        self.G1_fit = self.G1_fit[~bad_indices]
        self.G2_fit = self.G2_fit[~bad_indices]
        self.a = self.a[~bad_indices]
        self.chi2 = self.chi2[~bad_indices]

        print(f"Filtered flexion catalog to {len(self.IDs)} entries after applying cuts.")

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
            arcsec_per_pixel=self.CDELT
        )

        sigs = np.full_like(self.sources.e1, np.mean([np.std(self.sources.e1), np.std(self.sources.e2)]))
        sigaf = np.mean([np.std(self.a * self.sources.f1), np.std(self.a * self.sources.f2)]) 
        sigag = np.mean([np.std(self.a * self.sources.g1), np.std(self.a * self.sources.g2)])
        sigf, sigg = sigaf / self.a, sigag / self.a
        print(f"Using uncertainties: sigs = {np.mean(sigs)}, sigf = {np.median(sigf)}, sigg = {np.median(sigg)}")
        print(f"Sigaf = {sigaf}, Sigag = {sigag}")

        # Update Source object with new uncertainties
        self.sources.sigs = sigs
        self.sources.sigf = sigf
        self.sources.sigg = sigg

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
        def plot_cluster(convergence, title, save_name):
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

            # Display image
            ax.imshow(
                img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
            )
            
            # Overlay convergence contours
            contour_levels = np.percentile(convergence[2], np.linspace(70, 99, 5))
            contours = ax.contour(
                convergence[0], convergence[1], convergence[2], levels=contour_levels, cmap='plasma', linewidths=1.5
            )
            # Add colorbar for contours
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')

            # Labels and title
            ax.set_xlabel('RA Offset (arcsec)')
            ax.set_ylabel('Dec Offset (arcsec)')
            ax.set_title(title)
            # Save and display
            # Only create a legend if there are labels to display
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_name, dpi=300)

        # Create a comparison by doing a kaiser squires transformation to get kappa from the flexion
        # only do this for all signals (there's no point in doing it for shear_f or flexion_g)
        avg_source_density = len(self.sources.x) / (np.pi/4 * np.max(np.hypot(self.sources.x, self.sources.y))**2)
        smoothing_scale = 1 / (avg_source_density)**0.5
        kappa_extent = [min(self.sources.x), max(self.sources.x), min(self.sources.y), max(self.sources.y)]

        
        # Override the sources with a custom set of lenses, to see if the reconstruction works with known mass
        # Do this for a range of noise choices
        
        # NFW version
        '''        
        xl = [60,80,100]
        yl = [30,40,50]
        ml = [1e14, 1e15, 1e14] # Masses in M_sun h^-1
        self.lenses = halo_obj.NFW_Lens(
            x=np.array(xl), y=np.array(yl), z=np.zeros(len(xl)),
            concentration=np.ones(len(xl)), mass=np.array(ml),
            redshift=z_cluster, chi2=np.zeros(len(xl))
        )
        self.lenses.calculate_concentration()
        '''
        
        # SIS version
        # '''
        xl = [80]
        yl = [40]
        eR = [23]
        self.lenses = halo_obj.SIS_Lens(
            x=np.array(xl), y=np.array(yl),
            te=np.array(eR), chi2=np.zeros(len(xl))
        )

        # Question: What is the scatter in aF as a function of distance from the cluster center?
        # Compare real vs simulated data in radial annuli

        aF_real = []
        aF_simulated = []
        sigma_aF_real = []
        sigma_aF_simulated = []
        gamma_1_real = []
        gamma_2_real = []
        gamma_1_simulated = []
        gamma_2_simulated = []
        sigma_gamma_1_real = []
        sigma_gamma_2_real = []
        sigma_gamma_1_simulated = []
        sigma_gamma_2_simulated = []
        r_edges = np.linspace(0, 120, 50)  # Annuli edges in arcseconds
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])  # Midpoints of annuli

        # Compute radial distances from cluster center
        radii = np.hypot(self.xc - self.centroid_x, self.yc - self.centroid_y)

        # Compute aF from real data
        aF_real_vals = self.a * np.hypot(self.sources.f1, self.sources.f2)
        for r_min, r_max in zip(r_edges[:-1], r_edges[1:]):
            annulus_mask = (radii >= r_min) & (radii < r_max)
            aF_real.append(np.mean(aF_real_vals[annulus_mask]))
            sigma_aF_real.append(np.std(aF_real_vals[annulus_mask]))
            gamma_1_real.append(np.mean(self.sources.g1[annulus_mask]))
            gamma_2_real.append(np.mean(self.sources.g2[annulus_mask]))
            sigma_gamma_1_real.append(np.std(self.sources.g1[annulus_mask]))
            sigma_gamma_2_real.append(np.std(self.sources.g2[annulus_mask]))


        # Simulate data: apply noise and lensing model
        self.sources.zero_lensing_signals()  # Reset lensing signals to zero for reconstruction
        self.sources.apply_noise()
        self.sources.apply_lensing(self.lenses, lens_type='SIS', z_source=z_source)

        # Compute aF from simulated data
        aF_simulated_vals = self.a * np.hypot(self.sources.f1, self.sources.f2)
        for r_min, r_max in zip(r_edges[:-1], r_edges[1:]):
            annulus_mask = (radii >= r_min) & (radii < r_max)
            aF_simulated.append(np.mean(aF_simulated_vals[annulus_mask]))
            sigma_aF_simulated.append(np.std(aF_simulated_vals[annulus_mask]))
            gamma_1_simulated.append(np.mean(self.sources.g1[annulus_mask]))
            gamma_2_simulated.append(np.mean(self.sources.g2[annulus_mask]))
            sigma_gamma_1_simulated.append(np.std(self.sources.g1[annulus_mask]))
            sigma_gamma_2_simulated.append(np.std(self.sources.g2[annulus_mask]))
        
        # Plot the scatter in aF as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, sigma_aF_real, label='Real Data', color='blue')
        plt.plot(r_centers, sigma_aF_simulated, label='Simulated Data', color='orange')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Scatter in aF')
        plt.title('Scatter in aF as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'scatter_aF_vs_distance.png', dpi=300)

        # Plot the mean of aF as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, aF_real, label='Mean aF (Real Data)', color='blue')
        plt.plot(r_centers, aF_simulated, label='Mean aF (Simulated Data)', color='orange')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Mean aF')
        plt.title('Mean aF as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'mean_aF_vs_distance.png', dpi=300)

        real_signal_to_noise = aF_real / np.array(sigma_aF_real)
        simulated_signal_to_noise = aF_simulated / np.array(sigma_aF_simulated)
        # Plot the signal to noise ratio as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, real_signal_to_noise, label='SNR (Real Data)', color='blue')
        plt.plot(r_centers, simulated_signal_to_noise, label='SNR (Simulated Data)', color='orange')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Signal to Noise Ratio (SNR)')
        plt.title('Signal to Noise Ratio as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'snr_vs_distance.png', dpi=300)

        # Also plot number of sources as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.hist(radii, bins=r_edges, color='gray', alpha=0.7, edgecolor='black')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Number of Sources')
        plt.title('Number of Sources as a Function of Distance from Cluster Center')
        plt.grid(True)
        plt.savefig(self.output_dir / 'num_sources_vs_distance.png', dpi=300)

        # Plot the shear components as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, gamma_1_real, label='Gamma 1 (Real Data)', color='blue')
        plt.plot(r_centers, gamma_1_simulated, label='Gamma 1 (Simulated Data)', color='green')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Shear Components')
        plt.title('Shear Components as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'shear_vs_distance.png', dpi=300)

        # Plot the shear noise as a function of distance from the cluster center
        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, sigma_gamma_1_real, label='Sigma Gamma 1 (Real Data)', color='blue')
        plt.plot(r_centers, sigma_gamma_1_simulated, label='Sigma Gamma 1 (Simulated Data)', color='green')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Shear Noise')
        plt.title('Shear Noise as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'shear_noise_vs_distance.png', dpi=300)

        gamma_SNR_real_1 = np.array(gamma_1_real) / np.array(sigma_gamma_1_real)
        gamma_SNR_real_2 = np.array(gamma_2_real) / np.array(sigma_gamma_2_real)
        gamma_SNR_simulated_1 = np.array(gamma_1_simulated) / np.array(sigma_gamma_1_simulated)
        gamma_SNR_simulated_2 = np.array(gamma_2_simulated) / np.array(sigma_gamma_2_simulated)
        # Plot the shear signal to noise ratio as a function of distance from the cluster center

        plt.figure(figsize=(10, 6))
        plt.plot(r_centers, gamma_SNR_real_1, label='SNR Gamma 1 (Real Data)', color='blue')
        plt.plot(r_centers, gamma_SNR_simulated_1, label='SNR Gamma 1 (Simulated Data)', color='green')
        plt.xlabel('Distance from Cluster Center (arcsec)')
        plt.ylabel('Shear Signal to Noise Ratio (SNR)')
        plt.title('Shear Signal to Noise Ratio as a Function of Distance from Cluster Center')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'shear_snr_vs_distance.png', dpi=300)
        # '''

        X, Y, kappa_flexion = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='flexion', smoothing_scale=smoothing_scale, resolution_scale=1.0)
        title = 'Kaiser-Squires Flexion Reconstruction of {} with JWST'.format(self.cluster_name)
        save_title = self.output_dir / 'ks_flex_{}.png'.format(self.cluster_name)
        plot_cluster([X,Y,kappa_flexion], title, save_title)

        # Do this for the shear as well
        X, Y, kappa_shear = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='shear', smoothing_scale=smoothing_scale, resolution_scale=1.0)
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

    def find_outliers(self, cutoff_distance=120):
        """
        Identifies bad sources based on flexion and shear signals.
        This method attempts to identify outliers that will not contribute to the lensing analysis, or that may skew the results.
        """
        # Define a center for the cluster
        cluster_center = [80, 40]  # Approx center for Abell 2744
        # Calculate distance from cluster center
        distances = np.hypot(self.xc - cluster_center[0], self.yc - cluster_center[1])

        # Throw out sources more than cutoff_distance arcseconds from the cluster center
        outlier_mask = distances > cutoff_distance
        self.IDs = self.IDs[~outlier_mask]
        self.q = self.q[~outlier_mask]
        self.phi = self.phi[~outlier_mask]
        self.F1_fit = self.F1_fit[~outlier_mask]
        self.F2_fit = self.F2_fit[~outlier_mask]
        self.G1_fit = self.G1_fit[~outlier_mask]
        self.G2_fit = self.G2_fit[~outlier_mask]
        self.a = self.a[~outlier_mask]
        self.chi2 = self.chi2[~outlier_mask]
        self.xc = self.xc[~outlier_mask]
        self.yc = self.yc[~outlier_mask]
        distances = distances[~outlier_mask]

        F = np.hypot(self.F1_fit, self.F2_fit)
        aF = self.a * F

        # Change the output directory to the cutoff distance
        #self.output_dir = self.output_dir / f'cutoff_{cutoff_distance}'
        # Create the output directory if it doesn't exist
        #self.output_dir.mkdir(parents=True, exist_ok=True)

        # Look at the properties of the remaining sources
        # Histogram of flexion, shear, size, and distance from cluster center
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        fancy_hist(aF, bins='freedman', ax=axs[0], color='blue', alpha=0.7)
        axs[0].set_title('aF Distribution')
        axs[0].set_xlabel('aF')
        axs[0].set_ylabel('Count')
        fancy_hist(F, bins='freedman', ax=axs[1], color='orange', alpha=0.7)
        axs[1].set_title('F Distribution')
        axs[1].set_xlabel('F')
        axs[1].set_ylabel('Count')
        fancy_hist(self.a, bins='freedman', ax=axs[2], color='green', alpha=0.7)
        axs[2].set_title('Size (a) Distribution')
        axs[2].set_xlabel('a (arcsec)')
        axs[2].set_ylabel('Count')
        fancy_hist(distances, bins='freedman', ax=axs[3], color='red', alpha=0.7)
        axs[3].set_title('Distance from Cluster Center Distribution')
        axs[3].set_xlabel('Distance (arcsec)')
        axs[3].set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'outlier_analysis.png', dpi=300)
        plt.close()
        print("Outlier analysis complete. Plots saved to output directory.")


    def drastic_test(self):
        '''
        Since I haven't been able to find the issue otherwise, let's try a drastic test.
        My hypothesis is that there is a single source that is dramatically affecting the results.
        Let's try removing sources one by one, performing the reconstruction each time, and seeing if the results change.
        '''

        # Get the catalog
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.cut_flexion_catalog()
        self.match_sources()
        self.initialize_sources()

        n_sources = len(self.sources.x)
        print(f"Starting drastic test with {n_sources} sources.")
        kappa_extent = [min(self.sources.x), max(self.sources.x), min(self.sources.y), max(self.sources.y)]
        smoothing_scale = 1 / (len(self.sources.x) / (np.pi/4 * np.max(np.hypot(self.sources.x, self.sources.y))**2))**0.5

        # Get the image data and extent
        # This is necessary to plot the image over the convergence map
        img_data = self.get_image_data()
        img_extent = [
            0, img_data.shape[1] * self.CDELT,
            0, img_data.shape[0] * self.CDELT
        ]

        def plot_cluster(convergence, title, save_name):
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

            # Display image
            ax.imshow(
                img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
            )
            
            # Overlay convergence contours
            contour_levels = np.percentile(convergence[2], np.linspace(70, 99, 5))
            contours = ax.contour(
                convergence[0], convergence[1], convergence[2], levels=contour_levels, cmap='plasma', linewidths=1.5
            )
            # Add colorbar for contours
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.3f')

            # Labels and title
            ax.set_xlabel('RA Offset (arcsec)')
            ax.set_ylabel('Dec Offset (arcsec)')
            ax.set_title(title)
            # Save and display
            # Only create a legend if there are labels to display
            if len(ax.get_legend_handles_labels()[0]) > 0:
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_name, dpi=300)

        # Initialize progress bar
        utils.print_progress_bar(0, n_sources, prefix='Drastic Test Progress:', suffix='Complete', length=50)

        for i in range(n_sources):
            new_sources = self.sources.copy()
            new_sources.remove(i)
            # Perform the reconstruction
            X, Y, kappa_flexion = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='flexion', smoothing_scale=smoothing_scale, resolution_scale=1.0)
            title = 'Kaiser-Squires Flexion Reconstruction of {} with JWST \n Removed Source {}'.format(self.cluster_name, i)
            save_title = self.output_dir / 'flex_{}.png'.format(i)
            plot_cluster([X,Y,kappa_flexion], title, save_title)

            # Do this for the shear as well
            X, Y, kappa_shear = utils.perform_kaiser_squire_reconstruction(self.sources, extent=kappa_extent, signal='shear', smoothing_scale=smoothing_scale, resolution_scale=1.0)
            title = 'Kaiser-Squires Shear Reconstruction of {} with JWST \n Removed Source {}'.format(self.cluster_name, i)
            save_title = self.output_dir / 'shear_{}.png'.format(i)
            plot_cluster([X,Y,kappa_shear], title, save_title)
            plt.close() 
            # Update progress bar
            utils.print_progress_bar(i + 1, n_sources, prefix='Drastic Test Progress:', suffix='Complete', length=50)

if __name__ == '__main__':
    signals = ['all']

    for signal in signals:

        abell_config = {
        'flexion_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl',
        'source_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
        'image_path': 'JWST_Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
        'output_dir': 'Output/JWST/ABELL/drastic_test/',
        'cluster_name': 'ABELL_2744',
        'cluster_redshift': 0.308,
        'source_redshift': 0.8,
        'signal_choice': signal
        }

        # Initialize and run the pipeline
        pipeline_abell = JWSTPipeline(abell_config)
        pipeline_abell.drastic_test()