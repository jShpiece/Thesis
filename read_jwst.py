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

        # Data placeholders
        self.IDs = None
        self.q = None
        self.phi = None
        self.F1_fit = None
        self.F2_fit = None
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

    def read_flexion_catalog(self):
        """
        Reads the JWST flexion catalog.
        """
        df = pd.read_pickle(self.flexion_catalog_path)
        self.IDs = df['label'].to_numpy()
        self.q = df['q'].to_numpy()
        self.phi = df['phi'].to_numpy()
        self.phi = np.deg2rad(self.phi)  # Convert phi to radians if necessary
        self.F1_fit = df['F1_fit'].to_numpy() / self.CDELT  # Convert flexion to arcsec^-1
        self.F2_fit = df['F2_fit'].to_numpy() / self.CDELT  
        self.a = df['a'].to_numpy() * self.CDELT # Convert scale to arcsec
        self.chi2 = df['rchi2'].to_numpy()

        # Check for NaN values
        nan_indices = np.isnan(self.F1_fit) | np.isnan(self.F2_fit) | np.isnan(self.a) | np.isnan(self.chi2) | np.isnan(self.q) | np.isnan(self.phi)
        if np.any(nan_indices):
            warnings.warn(f"Found NaN values in flexion catalog. Removing {np.sum(nan_indices)} entries.")
            self.IDs = self.IDs[~nan_indices]
            self.q = self.q[~nan_indices]
            self.phi = self.phi[~nan_indices]
            self.F1_fit = self.F1_fit[~nan_indices]
            self.F2_fit = self.F2_fit[~nan_indices]
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

        # Estimate noise
        sigs = np.full_like(e1, np.mean([np.std(e1), np.std(e2)]))
        sigaf = np.mean([np.std(self.a * self.F1_fit), np.std(self.a * self.F2_fit)])
        epsilon = 1e-8 # Small value to avoid division by zero
        sigf = sigaf / (self.a + epsilon)

        # Create Source object
        self.sources = source_obj.Source(
            x=self.xc_centered,
            y=self.yc_centered,
            e1=e1,
            e2=e2,
            f1=self.F1_fit,
            f2=self.F2_fit,
            g1=self.F1_fit,  
            g2=self.F2_fit,  
            sigs=sigs,
            sigf=sigf,
            sigg=sigf  # Adjust if necessary
        )

        self.sources.filter_sources()

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
        self.lenses, _ = main.fit_lensing_field(
            self.sources, xmax, flags=True, use_flags=[True, True, False], lens_type='NFW'
        )
        # Adjust lens positions back to original coordinates
        self.lenses.x += self.centroid_x
        self.lenses.y += self.centroid_y

    def save_results(self):
        """
        Saves the lenses and sources to files.
        """
        np.save(self.output_dir / 'lenses.npy', [self.lenses.x, self.lenses.y, self.lenses.mass, self.lenses.chi2])
        np.save(self.output_dir / 'sources.npy', [
            self.sources.x, self.sources.y, 
            self.sources.e1, self.sources.e2,
            self.sources.f1, self.sources.f2, 
            self.sources.sigs, self.sources.sigf
        ])

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
            x, y, mass, chi2 = np.load(self.output_dir / 'lenses.npy')
            self.lenses = halo_obj.NFW_Lens(x, y, np.zeros_like(x), np.zeros_like(x), mass, 0.2, chi2)
            self.lenses.calculate_concentration()
        
        # Load source positions
        if self.sources is None:
            x, y, e1, e2, f1, f2, sigs, sigf = np.load(self.output_dir / 'sources.npy')
            x += 69 # Adjust for centroid shift
            y += 69
            self.sources = source_obj.Source(x, y, e1, e2, f1, f2, f1, f2, sigs, sigf, sigf)

        # Calculate convergence map
        X, Y, kappa = utils.calculate_kappa(
            self.lenses, extent=img_extent, smoothing_scale=5, lens_type='NFW'
        )


        # Plot settings
        fig, ax = plt.subplots(figsize=(10, 10))
        norm = ImageNormalize(img_data, vmin=0, vmax=100, stretch=LogStretch())

        # Display image
        ax.imshow(
            img_data, cmap='gray_r', origin='lower', extent=img_extent, norm=norm
        )

        
        # Overlay convergence contours
        contour_levels = np.linspace(np.min(kappa), np.max(kappa), 10)
        contours = ax.contour(
            X, Y, kappa, levels=contour_levels, cmap='plasma', linewidths=1.5
        )

        # Add colorbar for convergence
        cbar = plt.colorbar(contours, ax=ax)
        cbar.set_label(r'Convergence $\kappa$', rotation=270, labelpad=15)
        
        # Plot lens positions
        ax.scatter(self.lenses.x, self.lenses.y, s=50, facecolors='none', edgecolors='red', label='Lenses')

        # Plot source positions
        # ax.scatter(self.sources.x, self.sources.y, s=50, c='blue', label='Sources', marker='.')

        # Labels and title
        ax.set_xlabel('RA Offset (arcsec)')
        ax.set_ylabel('Dec Offset (arcsec)')
        ax.set_title('Convergence Map Overlaid on JWST Image')

        # Save and display
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'convergence_map.png', dpi=300)
        plt.show()

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
    config = {
        'flexion_catalog_path': 'JWST_Data/JWST/Cluster_Field/Catalogs/F115W_flexion.pkl',
        'source_catalog_path': 'JWST_Data/JWST/Cluster_Field/Catalogs/stacked_cat.ecsv',
        'image_path': 'JWST_Data/JWST/Cluster_Field/Image_Data/jw02756-o003_t001_nircam_clear-f115w_i2d.fits',
        'output_dir': 'Output/JWST',
    }

    # Initialize and run the pipeline
    pipeline = JWSTPipeline(config)
    # pipeline.run()
    pipeline.plot_results()