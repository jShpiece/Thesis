import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
import warnings
import main
import source_obj
import halo_obj
import utils


plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

# Define some constants
arcsec_per_pixel = 0.03 # From the instrumentation documentation
hubble_param = 0.67 # Hubble constant
z_cluster = 0.308 # Redshift of the cluster
z_source = 0.52 # Mean redshift of the sources

def plot_cluster(ax, img_data, X, Y, conv, lenses, sources, extent, vmax=1, legend=True):
    # Plotting function to overlay lensing reconstruction - in the form of lenses or convergence contours, 
    # ontop of an optical image of the cluster
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    for img in img_data:
        # Allow for multiple images to be overlayed - allows for band or epoch stacking
        norm = ImageNormalize(img, vmin=0, vmax=vmax, stretch=LogStretch())
        ax.imshow(img, origin='lower', extent=extent, norm=norm, cmap='gray_r')

    if conv is not None:
        # Adjusted contour levels for better feature representation.
        contour_levels = np.percentile(conv, np.linspace(60, 100, 5))

        # Only overlay the contours of the convergence map - the color map will be too busy
        # Overlay the contours of the convergence map.
        contour = ax.contour(X, Y, conv, levels=contour_levels, cmap='plasma', linewidths=1.5)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    if lenses is not None:
        ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
    if sources is not None:
        ax.scatter(sources.x, sources.y, marker='.', color='blue', alpha=0.5, label='Sources')

    if legend:
        ax.legend()


def get_img_data(fits_file_path) -> np.ndarray:
    # Get the image data from the fits file
    fits_file = fits.open(fits_file_path)
    img_data = fits_file[0].data
    header = fits_file[0].header
    return img_data, header


def get_file_paths(cluster='a2744', field='cluster'):
    if cluster == 'a2744':
        if field == 'cluster':
            fits_file_path = 'JWST_Data/HST/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
            csv_file_path = 'JWST_Data/HST/a2744_clu_lenser.csv'
            vmax = 1 # Set the maximum value for the image normalization
            dx = 115
            dy = 55
        elif field == 'parallel':
            fits_file_path = 'JWST_Data/HST/hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz.fits'
            csv_file_path = 'JWST_Data/HST/a2744_par_lenser.csv'
            vmax = 0.1 # Set the maximum value for the image normalization
            dx = 865
            dy = 400
    
    return fits_file_path, csv_file_path, vmax, dx, dy


def get_catalog_data(file):
    # Get the catalog data from the csv file
    data = np.genfromtxt(file, delimiter=',')
    # Read the header to get the column names
    with open(file, 'r') as f:
        header = f.readline().strip().split(',')
    # Get the column indices
    xcol, ycol, acol = header.index('xs'), header.index('ys'), header.index('a') # Position terms
    qcol, phicol = header.index('q'), header.index('phi') # Shape terms
    f1col, f2col = header.index('f1'), header.index('f2') # Flexion terms
    g1col, g2col = header.index('g1'), header.index('g2') # Shear terms
    # Extract the data - omit the header
    x, y, a = data[1:, xcol], data[1:, ycol], data[1:, acol]
    q, phi = data[1:, qcol], data[1:, phicol]
    f1, f2 = data[1:, f1col], data[1:, f2col]
    g1, g2 = data[1:, g1col], data[1:, g2col]

    return x, y, a, q, phi, f1, f2, g1, g2


def reconstruct_system(file, dx, dy, flags=False, use_flags=[True, True, True], lens_type='NFW'):
    # Take in a file of sources and reconstruct the lensing system

    # Get the catalog data
    x, y, a, q, phi, f1, f2, g1, g2 = get_catalog_data(file)

    # Apply offsets to the x and y coordinates - this corrects for miscommunication between the pipeline and image
    x += dx 
    y += dy 

    # Set xmax to be the largest distance from the center
    centroid = np.mean(x), np.mean(y)
    xmax = np.max(np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2))

    # Calculate the shear from the shape parameters
    shear_mag = (q-1)/(q+1)
    e1, e2 = shear_mag * np.cos(2*phi), shear_mag * np.sin(2*phi)

    # Compute noise parameters
    sigs = np.full_like(e1, np.mean([np.std(e1), np.std(e2)]))
    sigaf = np.mean([np.std(a * f1), np.std(a * f2)])
    sigag = np.mean([np.std(a * g1), np.std(a * g2)])
    sigf, sigg = sigaf / a, sigag / a

    # Set the centroid to be the origin
    x -= centroid[0]
    y -= centroid[1]

    # Create source object and perform the lensing fit
    sources = source_obj.Source(x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg)
    recovered_lenses, reducedchi2 = main.fit_lensing_field(sources, xmax=xmax, flags=flags, use_flags=use_flags, lens_type=lens_type, z_lens=z_cluster, z_source=z_source)

    # Move our sourecs and lenses back to the original centroid
    sources.x += centroid[0]
    sources.y += centroid[1]
    recovered_lenses.x += centroid[0]
    recovered_lenses.y += centroid[1]
    recovered_lenses.mass *= hubble_param # Convert the mass to h^-1 solar masses

    return recovered_lenses, sources, xmax, reducedchi2


def reconstruct_a2744(field='cluster', full_reconstruction=False, use_flags=[True, True, True], lens_type='SIS'):
    '''
    A handler function to plot the lensing field of Abell 2744 - either the cluster or parallel field.
    --------------------
    Parameters:
        field: 'cluster' or 'parallel' - which field to plot
        full_reconstruction: whether or not to perform a full reconstruction of the lensing field, or to load in a precomputed one
    '''

    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings

    fits_file_path, csv_file_path, vmax, dx, dy = get_file_paths(field=field)

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    # Define output directory and file name
    dir = 'Output//JWST//ABELL//abel//'
    file_name = f"A2744{'_par' if field == 'parallel' else '_clu'}"

    # Append signal flags to file name
    signals = ['_gamma', '_F', '_G']
    if use_flags == [True, True, True]:
        file_name += '_all'
    else:
        file_name += ''.join(sig for flag, sig in zip(use_flags, signals) if flag)

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, dx * arcsec_per_pixel, dy * arcsec_per_pixel, flags=True, use_flags=use_flags, lens_type=lens_type)
        print('Completed reconstruction')

        lenses.export_to_csv(dir + file_name + '_lenses.csv')
        sources.export_to_csv(dir + file_name + '_sources.csv')
    else:
        # If we're not doing a full reconstruction, we need to load in the data
        if lens_type == 'SIS':
            lenses = halo_obj.SIS_Lens.load_from_csv(dir + file_name + '_lenses.csv')
        else:
            # lenses = halo_obj.NFW_Lens.load_from_csv(dir + file_name + '_lenses.csv')
            lenses = halo_obj.NFW_Lens([0], [0], [0], [0], [0], 0, [0])
            lenses.redshift = z_cluster
            lenses.import_from_csv(dir + file_name + '_lenses.csv')
        sources = source_obj.Source([0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0])
        sources.import_from_csv(dir + file_name + '_sources.csv')
        print('Loaded in data')

    # Generate a convergence map that spans the same area as the image
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X, Y, kappa = utils.calculate_kappa(lenses, extent, lens_type=lens_type, source_redshift=z_source)
    print('Calculated kappa')
    
    if lens_type=='SIS':
        mass = utils.calculate_mass(kappa, z_cluster, z_source, 1) # Calculate the mass within the convergence map
    else:
        mass = np.sum(lenses.mass) # Calculate the mass of the NFW lenses
    
    # Build the title
    title = 'Abell 2744 Convergence Map - ' + 'Parallel Field' if field == 'parallel' else 'Cluster Field' + '\n' + f'M = {mass:.3e} $h^{{-1}} M_\\odot$'

    if use_flags != [True, True, True]:
        title += '\n Signals Used: '
        signals = []
        if use_flags[0]:
            signals.append(r'$\gamma$')
        if use_flags[1]:
            signals.append('F')
        if use_flags[2]:
            signals.append('G')
        title += ' '.join(signals)
    else:
        title += '\n All Signals Used'
    
    # Create the figure 
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111)
    plot_cluster(ax, img_data, X, Y, kappa, None, None, extent, vmax, legend=False)
    ax.set_title(title)
    plt.savefig(dir + file_name + '.png')

    # If signal choice is all, also do a kaiser squires reconstruction
    if use_flags == [True, True, True]:
        '''
        lenses = halo_obj.NFW_Lens([130], [130], [0], [10], [1e14], z_cluster, [0])
        sources.zero_lensing_signals()
        sources.apply_lensing(lenses, lens_type=lens_type, z_source=z_source)
        sources.apply_noise()
        '''

        # Remove all sources more than 30 arcsec from the center
        x_c, y_c = 130, 130
        r = np.sqrt((sources.x - x_c)**2 + (sources.y - y_c)**2)
        max_r = 30
        sources.x = sources.x[r < max_r]
        print(len(sources.x))
        sources.y = sources.y[r < max_r]
        sources.f1 = sources.f1[r < max_r]
        sources.f2 = sources.f2[r < max_r]
        sources.g1 = sources.g1[r < max_r]
        sources.g2 = sources.g2[r < max_r]
        sources.sigf = sources.sigf[r < max_r]
        sources.sigs = sources.sigs[r < max_r]
        sources.sigg = sources.sigg[r < max_r]
        sources.e1 = sources.e1[r < max_r]
        sources.e2 = sources.e2[r < max_r]

        X1, Y1, kappa_f = utils.perform_kaiser_squire_reconstruction(sources, extent, signal = 'flexion')
        X2, Y2, kappa_g = utils.perform_kaiser_squire_reconstruction(sources, extent, signal = 'shear')

        # Plot the flexion and shear maps
        plt.close('all')
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Kaiser-Squires Reconstruction with HST Data')
        ax[0].set_title('Flexion Map')
        ax[0].contour(X1, Y1, kappa_f, levels=np.percentile(kappa_f, np.linspace(70, 99, 5)), cmap='plasma', linewidths=1.5)
        ax[1].set_title('Shear Map')
        ax[1].contour(X2, Y2, kappa_g, levels=np.percentile(kappa_g, np.linspace(70, 99, 5)), cmap='plasma', linewidths=1.5)
        
        for img in img_data:
            # Allow for multiple images to be overlayed - allows for band or epoch stacking
            norm = ImageNormalize(img, vmin=0, vmax=vmax, stretch=LogStretch())
            ax[0].imshow(img, origin='lower', extent=extent, norm=norm, cmap='gray_r')
            ax[1].imshow(img, origin='lower', extent=extent, norm=norm, cmap='gray_r')
        plt.savefig(dir + file_name + '_ks_reconstruction.png')
        plt.show()
        raise ValueError('Kaiser-Squires reconstruction complete')

if __name__ == '__main__':
    use_all_signals = [True, True, True] # Use all signals
    shear_flex = [True, True, False] # Use shear and flexion
    all_flex = [False, True, True] # Use flexion and g-flexion
    global_signals = [True, False, True] # Use shear and g-flexion (global signals)
    signal_choices = [use_all_signals, shear_flex, all_flex, global_signals]
    
    for field in ['cluster']:
        for use_flags in signal_choices:
            reconstruct_a2744(field=field, full_reconstruction=False, use_flags=use_flags, lens_type='NFW')