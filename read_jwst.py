import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import ImageNormalize, LogStretch
from astropy.visualization import hist as fancyhist
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
z_cluster = 0.308
z_source = 0.5
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
        # Interpret signal choice
        # can be all, shear_f, f_g, or shear_g
        # corresponds to [True, True, True], [True, True, False], [False, True, True], [True, False, True]
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
        self.xc = None
        self.yc = None
        self.lenses = None
        self.sources = None
        self.centroid_x = None
        self.centroid_y = None

    def run(self):
        """
        Runs the entire pipeline.
        """
        self.read_flexion_catalog()
        self.read_source_catalog()
        self.match_sources()
        self.initialize_sources()
        self.run_lens_fitting()
        self.save_results()
        self.plot_results()

    def read_source_catalog(self):
        """
        Reads the source catalog.
        """
        table = Table.read(self.source_catalog_path)
        self.x_centroids = np.array(table['xcentroid'])
        self.y_centroids = np.array(table['ycentroid'])
        self.labels = np.array(table['label'])
        print(f"Read {len(self.labels)} sources from catalog.")

    def read_flexion_catalog(self):
        """
        Reads the JWST flexion catalog.
        """
        df = pd.read_pickle(self.flexion_catalog_path)
        self.IDs = df['label'].to_numpy()
        self.q = df['q'].to_numpy()
        self.phi = df['phi'].to_numpy()
        #self.phi = np.deg2rad(self.phi)  # Convert phi to radians if necessary
        self.F1_fit = df['F1_fit'].to_numpy()
        self.F2_fit = df['F2_fit'].to_numpy() 
        self.G1_fit = df['G1_fit'].to_numpy() 
        self.G2_fit = df['G2_fit'].to_numpy()
        self.a = df['a'].to_numpy()
        self.chi2 = df['rchi2'].to_numpy()
        print(f"Read {len(self.IDs)} entries from flexion catalog.")

        # Eliminate all entries with chi2 > 10
        # Add any indice where a < 0.1 - this is too small to get a good reading 
        bad_chi2 = self.chi2 > 10
        bad_a = (self.a < 0.1) | (self.a > 10)
        print('There are {} entries with a < 0.1 or a > 10'.format(np.sum(bad_a)))
        print('There are {} entries with chi2 > 10'.format(np.sum(bad_chi2)))
        bad_indices = (bad_chi2) | (bad_a)
        print(f"Removing {np.sum(bad_indices)} entries with chi2 > 10 or a < 0.1.")
        
        self.IDs = self.IDs[~bad_indices]
        self.q = self.q[~bad_indices]
        self.phi = self.phi[~bad_indices]
        self.F1_fit = self.F1_fit[~bad_indices]
        self.F2_fit = self.F2_fit[~bad_indices]
        self.G1_fit = self.G1_fit[~bad_indices]
        self.G2_fit = self.G2_fit[~bad_indices]
        self.a = self.a[~bad_indices]
        self.chi2 = self.chi2[~bad_indices]
        
        # Check for NaN values
        
        nan_indices = np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | np.isnan(self.a) | np.isnan(self.chi2) | np.isnan(self.q) | np.isnan(self.phi)
        if np.any(nan_indices):
            warnings.warn(f"Found NaN values in flexion catalog. Removing {np.sum(nan_indices)} entries.")
            self.IDs = self.IDs[~nan_indices]
            self.q = self.q[~nan_indices]
            self.phi = self.phi[~nan_indices]
            self.F1_fit = self.F1_fit[~nan_indices]
            self.F2_fit = self.F2_fit[~nan_indices]
            self.G1_fit = self.G1_fit[~nan_indices]
            self.G2_fit = self.G2_fit[~nan_indices]
            self.a = self.a[~nan_indices]
            self.chi2 = self.chi2[~nan_indices]
        
    def initialize_sources(self):
        """
        Prepares the Source object with calculated lensing signals and uncertainties.
        """
        # Convert positions to arcseconds
        self.xc_arcsec = self.xc * self.CDELT
        self.yc_arcsec = self.yc * self.CDELT

        # Calculate shear components
        shear_magnitude = (self.q - 1) / (self.q + 1)
        e1 = shear_magnitude * np.cos(2 * self.phi)
        e2 = shear_magnitude * np.sin(2 * self.phi)

        # Center coordinates
        self.centroid_x = np.mean(self.xc_arcsec) # Store centroid for later use
        self.centroid_y = np.mean(self.yc_arcsec)
        self.xc_centered = self.xc_arcsec - self.centroid_x
        self.yc_centered = self.yc_arcsec - self.centroid_y

        # Use dummy values for uncertainties to initialize Source object
        sigs = np.ones_like(e1) 
        sigf = np.ones_like(e1) 
        sigg = np.ones_like(e1) 

        # Create Source object
        self.sources = source_obj.Source(
            x=self.xc_centered,
            y=self.yc_centered,
            e1=e1,
            e2=e2,
            f1=self.F1_fit,
            f2=self.F2_fit,
            g1=self.G1_fit,  
            g2=self.G2_fit,  
            sigs=sigs,
            sigf=sigf,
            sigg=sigg  # Adjust if necessary
        )

        bad_indices = self.sources.filter_sources()
        print(f"Removing {len(bad_indices)} sources with flexion > 0.1.")

        # Calculate the noise levels for each signal
        a = self.a
        # We also need to remove the bad indices from the a array
        a = np.delete(a, bad_indices)
        sigs = np.full_like(self.sources.e1, np.mean([np.std(self.sources.e1), np.std(self.sources.e2)]))
        sigaf = np.mean([np.std(a * self.sources.f1), np.std(a * self.sources.f2)])
        epsilon = 1e-8 # Small value to avoid division by zero
        sigf = sigaf / (a + epsilon)
        sigag = np.mean([np.std(a * self.sources.g1), np.std(a * self.sources.g2)])
        sigg = sigag / (a + epsilon)

        assert len(sigs) == len(sigf) == len(sigg) == len(self.sources.x), 'Lengths of sigs, sigf, and sigg do not match the number of sources: {} vs {} vs {} vs {}'.format(len(sigs), len(sigf), len(sigg), len(self.sources.x))
        
        # Update Source object with new uncertainties
        self.sources.sigs = sigs
        self.sources.sigf = sigf
        self.sources.sigg = sigg

        '''
        # Do histograms of q, phi, f1, f2, a, and chi2
        
        signals = [self.sources.e1, self.sources.e2, self.sources.f1, self.sources.f2, a, self.chi2, self.phi]
        signal_names = ['e1', 'e2', 'F1', 'F2', 'a', 'chi2', 'phi']
        for signal, name in zip(signals, signal_names):
            fig, ax = plt.subplots()
            fancyhist(signal, bins='scott', ax=ax, histtype='step', density=True, label = 'Range: {} - {}'.format(np.min(signal), np.max(signal)))
            ax.set_xlabel(name)
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title(f'{name} Distribution - With Cuts')
            plt.savefig(self.output_dir / f'{name}_distribution_with_cuts.png', dpi=300)
        plt.close('all')
        '''
        

    def match_sources(self):
        """
        Matches flexion data with source positions based on IDs.
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

        # Remove entries with no matching source
        valid = ~np.isnan(self.xc) & ~np.isnan(self.yc)
        self.IDs = self.IDs[valid]
        self.q = self.q[valid]
        self.phi = self.phi[valid]
        self.F1_fit = self.F1_fit[valid]
        self.F2_fit = self.F2_fit[valid]
        self.a = self.a[valid]
        self.chi2 = self.chi2[valid]
        self.xc = self.xc[valid]
        self.yc = self.yc[valid]

    def run_lens_fitting(self):
        """
        Runs the lens fitting pipeline.
        """
        xmax = np.max(np.hypot(self.sources.x, self.sources.y))
        print(f"Running lens fitting with {len(self.sources.x)} sources.")
        print(f"Maximum source distance: {xmax:.2f} arcsec.")
        self.lenses, _ = main.fit_lensing_field(
            self.sources, xmax, flags=True, use_flags=self.use_flags, lens_type='NFW', z_lens=z_cluster, z_source=z_source
        )
        # Adjust lens positions back to original coordinates
        self.lenses.x += self.centroid_x
        self.lenses.y += self.centroid_y
        self.lenses.mass *= hubble_param # Convert mass to M_sun h^-1

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
        img_data, img_header = self.get_image_data()
        img_extent = [
            0, img_data.shape[1] * self.CDELT,
            0, img_data.shape[0] * self.CDELT
        ]

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
            self.lenses, extent=img_extent, smoothing_scale=5, lens_type='NFW'
        )


        # Plot settings
        '''
        fig, ax = plt.subplots(figsize=(10, 10))
        norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

        # Display image
        ax.imshow(
            img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
        )

        
        # Overlay convergence contours
        contour_levels = np.percentile(kappa, np.linspace(60, 100, 6))
        contours = ax.contour(
            X, Y, kappa, levels=contour_levels, cmap='plasma', linewidths=1.5
        )
        # Add colorbar for contours
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        cbar = plt.colorbar(contours, ax=ax)

        # Plot lens positions
        ax.scatter(self.lenses.x, self.lenses.y, s=50, facecolors='none', edgecolors='red', label='Lenses')

        # Labels and title
        ax.set_xlabel('RA Offset (arcsec)')
        ax.set_ylabel('Dec Offset (arcsec)')
        ax.set_title(r'Mass Reconstruction of A2744 with JWST - {}'.format(self.signal_choice) + '\n' + r'Total Mass = {:.2e} $h^{{-1}} M_\odot$'.format(np.sum(self.lenses.mass)))
        # Save and display
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'A2744_clu_{}.png'.format(self.signal_choice), dpi=300)
        '''

        utils.compare_mass_estimates_a2744(self.lenses, 'Output/JWST/mass_comparison_{}.png'.format(self.signal_choice))


    def get_image_data(self):
        """
        Reads image data from a FITS file.
        """
        with fits.open(self.image_path) as hdul:
            img_data = hdul['SCI'].data
            header = hdul['SCI'].header
        return img_data, header

if __name__ == '__main__':
    # Configuration dictionary
    signals = ['all', 'shear_f', 'f_g', 'shear_g']

    for signal in signals:
        config = {
            'flexion_catalog_path': 'JWST_Data/JWST/Cluster_Field/Catalogs/multiband_flexion.pkl',
            'source_catalog_path': 'JWST_Data/JWST/Cluster_Field/Catalogs/stacked_cat.ecsv',
            'image_path': 'JWST_Data/JWST/Cluster_Field/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
            'output_dir': 'Output/JWST',
            'signal_choice': signal
        }

        # Initialize and run the pipeline
        pipeline = JWSTPipeline(config)
        pipeline.run()