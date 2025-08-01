import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import halo_obj
from astropy.io import fits
import read_jwst

if __name__ == '__main__':
    signals = ['shear_f', 'f_g', 'shear_g']
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

        pipeline = read_jwst.JWSTPipeline(abell_config)
        pipeline.compute_error_bars()

        file_name = f'jackknife_results_{abell_config["cluster_name"]}_{signal}.csv'

        df = pd.read_csv(file_name)
        
        # df contains the results of a jackknife analysis
        # specifically the lenses produced by the pipeline
        # columns are:
        # i, x, y, mass, concentration

        # Read through the trials, construct the halo objects, and get the core mass (250 kpc). store that value. 
        # At the end, get the mean and standard deviation of the core mass.
        with fits.open(abell_config['image_path']) as hdul:
            img_data = hdul['SCI'].data

        CDELT = 8.54006306703281e-6 * 3600  # degrees/pixel converted to arcsec/pixel

        img_extent = [
            0, img_data.shape[1] * CDELT,
            0, img_data.shape[0] * CDELT
        ]

        core_masses = []
        
        for id in df['i']:
            # Grab the rows corresponding to the current id
            # These are not in order
            id = int(id)
            rows = df[df['i'] == id]
            halos = halo_obj.NFW_Lens(
                rows['x'].values, 
                rows['y'].values, 
                np.zeros_like(rows['x'].values), 
                rows['concentration'].values, 
                rows['M200'].values, 
                abell_config['cluster_redshift'], 
                np.zeros_like(rows['x'].values)
            )

            X, Y, kappa = utils.calculate_kappa(
                halos, extent=img_extent, lens_type='NFW', source_redshift=0.8
            )
            k_val = utils.estimate_mass_sheet_factor(kappa) # Mass sheet transformation parameter
            kappa = utils.mass_sheet_transformation(kappa, k=k_val)
            peaks, masses = utils.find_peaks_and_masses(
                kappa, 
                z_lens=0.308, z_source=0.8,
                radius_kpc=250
            )

            core_mass = 0
            for peak in peaks:
                if np.sqrt((peak[0] - 80)**2 + (peak[1] - 40)**2) < 10:
                    core_mass += masses[peaks.index(peak)]
            core_masses.append(core_mass)

        failed_detections = np.sum(np.array(core_masses) == 0)
        print(f'Number of failed core detections: {failed_detections}')
        # Remove failed detections from the core_masses list - the number of failed detections is printed above, and we don't want to skew the results
        core_masses = [mass for mass in core_masses if mass > 0]
        core_masses = np.array(core_masses)
        mean_core_mass = np.mean(core_masses)
        std_core_mass = np.std(core_masses)
        print(f'Mean core mass: {mean_core_mass:.2f} M_sun')
        print(f'Standard deviation of core mass: {std_core_mass:.2f} M_sun')

        # Also plot the core masses
        plt.figure(figsize=(10, 6))
        plt.hist(core_masses, bins='scott', color='blue', alpha=0.7, fill=False, label='Core Values - mean: {:.3e}, std: {:.3e}'.format(mean_core_mass, std_core_mass))
        plt.xscale('log')
        plt.xlabel('Core Mass (M_sun)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Core Masses from Jackknife Analysis')
        plt.legend()
        plt.savefig(f'core_masses_jackknife_{abell_config["cluster_name"]}_{signal}.png')