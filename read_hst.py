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
h = 0.7 # Hubble constant
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
        for i, eR in enumerate(lenses.te):
            ax.annotate(np.round(eR, 2), (lenses.x[i], lenses.y[i]))
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


def reconstruct_system(file, dx, dy, flags=False, use_flags=[True, True, True], lens_type='NFW'):
    # Take in a file of sources and reconstruct the lensing system

    # Read in the data (csv)
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
    sigf = np.full_like(f1, np.mean([np.std(a*f1), np.std(a*f2)]) / a)
    sigg = np.full_like(g1, np.mean([np.std(g1), np.std(g2)]))

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

    # Define some file paths
    dir = 'Output//abel//'
    file_name = 'A2744'
    file_name += '_par' if field == 'parallel' else '_clu'

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, dx * arcsec_per_pixel, dy * arcsec_per_pixel, flags=True, use_flags=use_flags, lens_type=lens_type)
        lenses.mass /= h # Convert the mass to h^-1 solar masses
        print('Completed reconstruction')

        lenses.export_to_csv(dir + file_name + '_lenses.csv')
        sources.export_to_csv(dir + file_name + '_sources.csv')
    else:
        # If we're not doing a full reconstruction, we need to load in the data
        if lens_type == 'SIS':
            lenses = halo_obj.SIS_Lens.load_from_csv(dir + file_name + '_lenses.csv')
        else:
            lenses = halo_obj.NFW_Lens.load_from_csv(dir + file_name + '_lenses.csv')
        sources = source_obj.Source.load_from_csv(dir + file_name + '_sources.csv')
        print('Loaded in data')

    # Generate a convergence map that spans the same area as the image
    # I'd like the kappa scale to be 1 arcsecond per pixel. This means each kappa pixel is 20 image pixels
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X, Y, kappa = utils.calculate_kappa(lenses, extent, smoothing_scale=1, lens_type=lens_type)
    print('Calculated kappa')
    
    if lens_type=='SIS':
        mass = utils.calculate_mass(kappa, z_cluster, z_source, 1) # Calculate the mass within the convergence map
    else:
        mass = np.sum(lenses.mass) # Calculate the mass of the NFW lenses
    
    # Build the file name
    
    if use_flags == [True, True, True]:
        file_name += '_all'
    else:
        signals = []
        if use_flags[0]:
            signals.append('_gamma')
        if use_flags[1]:
            signals.append('_F')
        if use_flags[2]:
            signals.append('_G')
        file_name += ''.join(signals)
    
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
    
    # Now compare the mass estimates - for now, only do this for all signals
    plot_name = dir + file_name + '_mass.png'
    utils.compare_mass_estimates(lenses, plot_name)
    plt.close()
    print('Plotted and saved')

if __name__ == '__main__':
    # lenses = np.load('Output//a2744_clu_lenses.npy')
    # compare_mass_estimates(lenses, 'Output//abel//a2744_clu_mass.png')

    use_all_signals = [True, True, True] # Use all signals
    shear_flex = [True, True, False] # Use shear and flexion
    all_flex = [False, True, True] # Use flexion and g-flexion
    global_signals = [True, False, True] # Use shear and g-flexion (global signals)
    signal_choices = [use_all_signals, shear_flex, all_flex, global_signals]
        
    for field in ['cluster']:
        for use_flags in signal_choices:
            reconstruct_a2744(field=field, full_reconstruction=True, use_flags=use_flags, lens_type='NFW')