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
        print(self.CDELT)

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

        '''
        self.F1_fit /= self.CDELT
        self.F2_fit /= self.CDELT
        self.G1_fit /= self.CDELT
        self.G2_fit /= self.CDELT
        self.a *= self.CDELT
        '''
        
        # Make cuts in the data based on flexion value, rs, chi2, and a
        max_flexion = 0.2
        bad_flexion = (np.abs(self.F1_fit) > max_flexion) | (np.abs(self.F2_fit) > max_flexion)
        bad_rs = (rs > 10)
        bad_chi2 = self.chi2 > 2
        bad_a = (self.a > 100) | (self.a < 0.1) # a should be between 0.1 and 20
        nan_indices = np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | np.isnan(self.a) | np.isnan(self.chi2) | np.isnan(self.q) | np.isnan(self.phi) | np.isnan(self.G1_fit) | np.isnan(self.G2_fit)

        bad_indices = (bad_flexion) | (bad_chi2) | (bad_a) | (bad_rs) | (nan_indices) # Combine all bad indices
        
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
        self.xc *= self.CDELT
        self.yc *= self.CDELT

        # Calculate shear components
        shear_magnitude = (self.q - 1) / (self.q + 1)
        e1 = shear_magnitude * np.cos(2 * self.phi)
        e2 = shear_magnitude * np.sin(2 * self.phi)

        # Center coordinates (necessary for pipeline)
        self.centroid_x = np.mean(self.xc) # Store centroid for later use
        self.centroid_y = np.mean(self.yc)
        #self.xc -= self.centroid_x
        #self.yc -= self.centroid_y

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
        
        sigs = np.full_like(self.sources.e1, np.mean([np.std(self.sources.e1), np.std(self.sources.e2)]))
        sigaf = np.mean([np.std(self.a * self.sources.f1), np.std(self.a * self.sources.f2)])
        sigag = np.mean([np.std(self.a * self.sources.g1), np.std(self.a * self.sources.g2)])
        sigf, sigg = sigaf / self.a, sigag / self.a

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
        def plot_cluster(title, save_name):
            fig, ax = plt.subplots(figsize=(10, 10))
            norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

            # Display image
            ax.imshow(
                img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
            )
            

            # Plot sources and flexion arrows
            ax.quiver(
                self.xc, self.yc,
                self.F1_fit, self.F2_fit,
                angles='xy', scale_units='xy', scale=0.1, color='red', alpha=0.5
            )

            # Labels and title
            ax.set_xlabel('RA Offset (arcsec)')
            ax.set_ylabel('Dec Offset (arcsec)')
            ax.set_title(title)
            # Save and display
            #plt.legend()
            plt.tight_layout()
            plt.savefig(save_name, dpi=300)
            plt.close()

        plot_cluster('Flexion in Abell 2744', 'Flexion_Abell_2744.png')

        # Define a center of 2744
        center_x = 80
        center_y = 40

        # Define a radial flexion - the component of the flexion that points towards the center
        dx = self.xc - center_x
        dy = self.yc - center_y
        # Calculate the radius and angle
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # Angle in radians
        
        radial_flexion = self.F1_fit * np.cos(theta) + self.F2_fit * np.sin(theta)
        # Cut down to a maximum radius of 10 arcsec
        max_radius = 100
        mask = r < max_radius
        r = r[mask]
        radial_flexion = radial_flexion[mask]

        # Bin the flexion 
        bin_width = 5
        bin_num = int(np.max(r) / bin_width)
        binned_flexion = np.zeros(bin_num)
        binned_r = np.zeros(bin_num)
        one_sigma = np.zeros(bin_num)
        for i in range(bin_num):
            # Find indices of points in the current bin
            bin_indices = np.where((r >= i * bin_width) & (r < (i + 1) * bin_width))[0]
            if len(bin_indices) > 0:
                binned_flexion[i] = np.median(radial_flexion[bin_indices])
                binned_r[i] = np.median(r[bin_indices])
                one_sigma[i] = np.std(radial_flexion[bin_indices])
            else:
                binned_flexion[i] = np.nan
                binned_r[i] = np.nan
                one_sigma[i] = np.nan
        # Remove NaN values from the binned arrays
        binned_flexion = binned_flexion[~np.isnan(binned_flexion)]
        binned_r = binned_r[~np.isnan(binned_r)]
        one_sigma = one_sigma[~np.isnan(one_sigma)]

        # Plot radial flexion
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(r, radial_flexion, 'o', markersize=2, color='blue', alpha=0.5)
        ax.errorbar(binned_r, binned_flexion, yerr=one_sigma, fmt='o', color='red', label='Binned Flexion')
        ax.set_xlabel('Radius (arcsec)')
        ax.set_ylabel('Radial Flexion')
        ax.set_title('Radial Flexion in Abell 2744')
        plt.tight_layout()
        plt.savefig('Radial_Flexion_Abell_2744.png', dpi=300)
        plt.show()



    def get_image_data(self):
        """
        Reads image data from a FITS file.
        """
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
        return img_data

if __name__ == '__main__':
    # Configuration dictionary
    signals = ['all']


    abell_config = {
        'flexion_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/og_flexion.pkl',
        'source_catalog_path': 'JWST_Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv',
        'image_path': 'JWST_Data/JWST/ABELL_2744/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
        'output_dir': 'Output/JWST/ABELL/',
        'cluster_name': 'ABELL_2744',
        'cluster_redshift': 0.308,
        'source_redshift': 1,
        'signal_choice': 'all'
    }

    pipeline_abell = JWSTPipeline(abell_config)

    pipeline_abell.run()