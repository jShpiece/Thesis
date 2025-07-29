import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
import warnings
import main
import source_obj
import halo_obj
import utils

plt.style.use('scientific_presentation.mplstyle')

class HST_Pipeline:
    arcsec_per_pixel = 0.03
    hubble_param = 0.67

    def __init__(self, config):
        """Initialize the pipeline with a configuration dictionary."""
        self.config = config
        self.field = config['field']
        self.full = config['full']
        self.use_flags = config['use_flags']
        self.lens_type = config['lens_type']
        self.z_cluster = config['cluster_redshift']
        self.z_source = config['source_redshift']

        self.img_path = config['image_path']
        self.cat_path = config['catalog_path']
        self.vmax = config['vmax']
        self.dx = config['dx'] * self.arcsec_per_pixel
        self.dy = config['dy'] * self.arcsec_per_pixel
        self.out_dir = config['output_dir']

        self.img_data, _ = self.get_image_data()
        if self.field == 'cluster':
            self.img_data = [self.img_data[0], self.img_data[1]]
        else:
            self.img_data = [self.img_data]

        self.file_stub = f"{config['cluster_name']}_{'par' if self.field == 'parallel' else 'clu'}" + self.get_signal_tag()
        self.sources = None
        self.lenses = None
        self.centroid = None
        self.xmax = None

    def get_signal_tag(self):
        """Generate a tag to append to filenames based on which signals are used."""
        labels = ['_shear', '_F', '_G']
        return 'all' if all(self.use_flags) else ''.join(sig for flag, sig in zip(self.use_flags, labels) if flag)

    def get_image_data(self):
        """Load image data from a FITS file and return data and header."""
        return fits.getdata(self.img_path, header=True)

    def get_catalog(self):
        """Read the lensing source catalog CSV and extract required columns."""
        data = np.genfromtxt(self.cat_path, delimiter=',')
        with open(self.cat_path, 'r') as f:
            header = f.readline().strip().split(',')
        idx = {key: header.index(key) for key in ['xs','ys','a','q','phi','f1','f2','g1','g2']}
        cols = [data[1:, idx[key]] for key in idx]
        return cols

    def init_sources(self):
        """Initialize the Source object with coordinates, ellipticities, and uncertainties."""
        x, y, a, q, phi, f1, f2, g1, g2 = self.get_catalog()
        x += self.dx; y += self.dy # Adjust coordinates to match the image center
        self.centroid = np.mean(x), np.mean(y)
        self.xmax = np.max(np.hypot(x,y))
        e1 = (q - 1)/(q + 1) * np.cos(2 * phi)
        e2 = (q - 1)/(q + 1) * np.sin(2 * phi)
        sigs = np.full_like(e1, np.mean([np.std(e1), np.std(e2)]))
        sigaf = np.mean([np.std(a*f1), np.std(a*f2)])
        sigag = np.mean([np.std(a*g1), np.std(a*g2)])
        sigf, sigg = sigaf / a, sigag / a
        x -= self.centroid[0]; y -= self.centroid[1] # Center the coordinates
        self.sources = source_obj.Source(x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg)

    def fit_lenses(self):
        """Fit lens model to the source catalog using specified flags and model."""
        self.lenses, _ = main.fit_lensing_field(self.sources, self.xmax, flags=True, use_flags=self.use_flags,
                                                lens_type=self.lens_type, z_lens=self.z_cluster, z_source=self.z_source)
        self.sources.x += self.centroid[0]; self.sources.y += self.centroid[1]
        self.lenses.x += self.centroid[0]; self.lenses.y += self.centroid[1]
        self.lenses.mass *= self.hubble_param

    def plot_overlay(self, ax, X, Y, kappa, lenses=None, sources=None, peaks=None, masses=None):
        """Overlay image data with optional convergence contours, lens, and source positions."""
        ax.set_xlabel('x (arcsec)'); ax.set_ylabel('y (arcsec)')
        for img in self.img_data:
            norm = ImageNormalize(img, vmin=0, vmax=self.vmax, stretch=LogStretch())
            ax.imshow(img, origin='lower', extent=self.extent, norm=norm, cmap='gray_r')
        if kappa is not None:
            levels = np.percentile(kappa, np.linspace(60, 100, 5))
            contour = ax.contour(X, Y, kappa, levels=levels, cmap='plasma', linewidths=1.5)
            ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
        if lenses is not None:
            ax.scatter(lenses.x, lenses.y, c='red', label='Lenses')
        if sources is not None:
            ax.scatter(sources.x, sources.y, c='blue', alpha=0.5, s=2, label='Sources')
        if peaks is not None and masses is not None:
            for i, ((px, py), mass) in enumerate(zip(peaks, masses)):
                ax.plot(px, py, 'ro', markersize=5, label='Mass Peak' if i == 0 else '')
                ax.text(px + 1, py + 1, f"{mass/1e13:.2f}e13 $M_\\odot$", color='black', fontsize=8, weight='bold')

    def run(self):
        """Run the full pipeline: load data, fit lenses, plot and save output."""
        if self.full:
            self.init_sources()
            self.fit_lenses()
            self.lenses.export_to_csv(self.out_dir + self.file_stub + '_lenses.csv')
            self.sources.export_to_csv(self.out_dir + self.file_stub + '_sources.csv')
        else:
            if self.lens_type == 'SIS':
                self.lenses = halo_obj.SIS_Lens.load_from_csv(self.out_dir + self.file_stub + '_lenses.csv')
            else:
                self.lenses = halo_obj.NFW_Lens([0], [0], [0], [0], [0], self.z_cluster, [0])
                self.lenses.import_from_csv(self.out_dir + self.file_stub + '_lenses.csv')
            self.sources = source_obj.Source([0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0])
            self.sources.import_from_csv(self.out_dir + self.file_stub + '_sources.csv')

        self.extent = [0, self.img_data[0].shape[1]*self.arcsec_per_pixel,
                        0, self.img_data[0].shape[0]*self.arcsec_per_pixel]
        X, Y, kappa = utils.calculate_kappa(self.lenses, self.extent, lens_type=self.lens_type, source_redshift=self.z_source)
        k_val = utils.estimate_mass_sheet_factor(kappa)
        kappa = utils.mass_sheet_transformation(kappa, k=k_val)
        peaks, masses = utils.find_peaks_and_masses(
            kappa, z_lens=self.z_cluster, z_source=self.z_source, radius_kpc=250
            )

        fig, ax = plt.subplots(figsize=(8, 10))
        self.plot_overlay(ax, X, Y, kappa, peaks=peaks, masses=masses)
        ax.set_title(f"{self.config['cluster_name']} Convergence Map - {'Parallel' if self.field=='parallel' else 'Cluster'} Field\nM = {np.sum(self.lenses.mass):.3e} $h^{{-1}} M_\\odot$\nSignals: " + self.get_signal_tag())
        plt.savefig(self.out_dir + self.file_stub + '.png')

        # Set the smoothing scale as the average distance between sources
        source_density = len(self.sources.x) / (np.pi * self.xmax**2)
        smoothing_scale = 1 / np.sqrt(source_density) 
        weights_shear = np.ones_like(self.sources.x) 
        weights_flexion = self.sources.sigf**-2
        if self.use_flags == [True]*3:
            for sig in ['flexion', 'shear']:
                Xk, Yk, kappak = utils.perform_kaiser_squire_reconstruction(
                    self.sources, self.extent, signal=sig, 
                    smoothing_scale=smoothing_scale, 
                    weights=weights_shear if sig == 'shear' else weights_flexion, 
                    apodize=True if sig == 'flexion' else False
                    )
                k_val = utils.estimate_mass_sheet_factor(kappak)
                kappak = utils.mass_sheet_transformation(kappak, k=k_val)
                peaks_ks, masses_ks = utils.find_peaks_and_masses(
                    kappak, z_lens=self.z_cluster, z_source=self.z_source, radius_kpc=250
                    )
                fig, ax = plt.subplots()
                self.plot_overlay(ax, Xk, Yk, kappak, peaks=peaks_ks, masses=masses_ks)
                ax.set_title(f"Kaiser-Squires {sig.capitalize()} Map")
                plt.savefig(self.out_dir + f'ABELL2744_ks_{sig}.png')

if __name__ == '__main__':
    signal_sets = [
        [True, True, True],
        [True, True, False],
        [False, True, True],
        [True, False, True]
    ]
    for use_flags in signal_sets:
        config = {
            'field': 'cluster',
            'full': True,
            'use_flags': use_flags,
            'lens_type': 'NFW',
            'cluster_redshift': 0.308,
            'source_redshift': 0.52,
            'image_path': 'JWST_Data/HST/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits',
            'catalog_path': 'JWST_Data/HST/a2744_clu_lenser.csv',
            'vmax': 1,
            'dx': 115,
            'dy': 55,
            'output_dir': 'Output/JWST/ABELL/abel/',
            'cluster_name': 'ABELL_2744'
        }
        pipeline = HST_Pipeline(config)
        pipeline.run()
