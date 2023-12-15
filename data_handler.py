import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
import warnings
import os
import pipeline
from utils import calculate_mass

plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots


def plot_cluster(ax, img_data, X, Y, conv, lenses, sources, extent, vmax=1, legend=True):
    # Plotting function to overlay lensing reconstruction - in the form of lenses or convergence contours, 
    # ontop of an optical image of the cluster
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    for img in img_data:
        # Allow for multiple images to be overlayed - allows for band or epoch stacking
        norm = ImageNormalize(img, vmin=0, vmax=vmax, stretch=LogStretch())
        ax.imshow(img, cmap='gray_r', origin='lower', extent=extent, norm=norm)

    if conv is not None:
        # Plot the convergence contours
        # Construct levels via percentiles
        percentiles = np.percentile(conv, np.linspace(60, 100, 7)) # Set the levels to grab the interesting features
        contours = ax.contour(X, Y, conv, levels=percentiles, cmap='viridis', linestyles='solid', linewidths=1) 
        ax.clabel(contours, inline=True, fontsize=10, fmt='%2.1e', colors='blue')

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


def get_coords(csv_file_path, coord_type='pixels') -> np.ndarray:
    # coord_type can be 'pixels' or 'degrees'
    # Read in the data (csv)
    data = np.genfromtxt(csv_file_path, delimiter=',')
    # Read the header to get the column names
    with open(csv_file_path, 'r') as f:
        header = f.readline().strip().split(',')
    # Get the column indices
    if coord_type == 'pixels':
        xcol, ycol = header.index('X_IMAGE'), header.index('Y_IMAGE')
    elif coord_type == 'degrees':
        xcol, ycol = header.index('X_WORLD'), header.index('Y_WORLD')
    else:
        raise ValueError('coord_type must be "pixels" or "degrees"')
    xcol, ycol = data[1:, xcol], data[1:, ycol]
    return np.array([xcol, ycol]).T


def reconstruct_system(file, flags=False, randomize = False):
    # Take in a file of sources and reconstruct the lensing system

    # Read in the data (csv)
    data = np.genfromtxt(file, delimiter=',')
    # Read the header to get the column names
    with open(file, 'r') as f:
        header = f.readline().strip().split(',')
    # Get the column indices
    xcol, ycol, qcol, phicol, f1col, f2col = header.index('xs'), header.index('ys'), header.index('q'), header.index('phi'), header.index('f1'), header.index('f2')
    # Extract the data - omit the header
    x, y, q, phi, f1, f2 = data[1:, xcol], data[1:, ycol], data[1:, qcol], data[1:, phicol], data[1:, f1col], data[1:, f2col]
    # Set these to be arrays 
    x, y, q, phi, f1, f2 = np.array(x), np.array(y), np.array(q), np.array(phi), np.array(f1), np.array(f2)

    # Calculate the shear from q and phi
    shear_mag = (q-1)/(q+1)
    e1 = shear_mag * np.cos(2*phi)
    e2 = shear_mag * np.sin(2*phi)
  
    # Set xmax to be the largest distance from the center
    centroid = np.mean(x), np.mean(y)
    xmax = np.max(np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2))

    # Create a source object
    acol = header.index('a')
    a = np.array(data[1:, acol])

    test_source_object = pipeline.Source(x, y, e1, e2, f1, f2, 0, 0)

    valid_indices = test_source_object.filter_sources(a)

    x, y, e1, e2, f1, f2, a = x[valid_indices], y[valid_indices], e1[valid_indices], e2[valid_indices], f1[valid_indices], f2[valid_indices], a[valid_indices]

    x += 115 # Offset between image and catalog - cluster field
    y += 55 # Offset between image and catalog - cluster field
    # x += 865 # Offset between image and catalog - parallel field
    # y += 400 # Offset between image and catalog - parallel field

    # Compute noise parameters
    sigs_mag = np.mean([np.std(e1), np.std(e2)])
    sigs = np.ones_like(x) * sigs_mag

    sigaf = np.mean([np.std(a*f1), np.std(a*f2)])
    sigf = sigaf / a 

    if randomize:
        # randomize the e1, e2 and f1, f2 vectors by rotating them by a random angle
        rand_angle = np.random.uniform(0, 2*np.pi, len(e1))
        e1, e2 = e1 * np.cos(2*rand_angle) - e2 * np.sin(2*rand_angle), e1 * np.sin(2*rand_angle) + e2 * np.cos(2*rand_angle)
        f1, f2 = f1 * np.cos(rand_angle) - f2 * np.sin(rand_angle), f1 * np.sin(rand_angle) + f2 * np.cos(rand_angle)

    # Set the centroid to be the origin
    x -= centroid[0]
    y -= centroid[1]

    # Create source object
    sources = pipeline.Source(x, y, e1, e2, f1, f2, sigs, sigf)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, xmax=xmax, lens_floor = 1, flags=flags)

    # Move our sourecs and lenses back to the original centroid
    sources.x += centroid[0]
    sources.y += centroid[1]
    recovered_lenses.x += centroid[0]
    recovered_lenses.y += centroid[1]

    return recovered_lenses, sources, xmax, reducedchi2


def reconstruct_a2744(field='cluster', randomize=False, full_reconstruction=False):
    '''
    A handler function to plot the lensing field of Abell 2744 - either the cluster or parallel field.
    --------------------
    Parameters:
        field: 'cluster' or 'parallel' - which field to plot
        randomize: whether or not to randomize the source orientations, which helps test the algorithm's tendency to overfit
        full_reconstruction: whether or not to perform a full reconstruction of the lensing field, or to load in a precomputed one
    '''
    z_cluster = 0.308 # Redshift of the cluster
    z_source = 0.52 # Mean redshift of the sources

    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings

    if field == 'cluster':
        fits_file_path = 'Data/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
        csv_file_path = 'Data/a2744_clu_lenser.csv'
        vmax = 1 # Set the maximum value for the image normalization
    elif field == 'parallel':
        fits_file_path = 'Data/hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz.fits'
        csv_file_path = 'Data/a2744_par_lenser.csv'
        vmax = 0.1 # Set the maximum value for the image normalization
    else:
        raise ValueError('field must be "cluster" or "parallel"')

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, flags=True, randomize=randomize)

        # Save the class objects so that we can replot without having to rerun the code
        if randomize:
            # If we're randomizing, I don't need to save the data
            pass
        else:
            dir = 'Data//'
            file_name = 'a2744' 
            file_name += '_par' if field == 'parallel' else '_clu' 
            np.save(dir + file_name + '_lenses', np.array([lenses.x, lenses.y, lenses.te, lenses.chi2]))
            np.save(dir + file_name + '_sources', np.array([sources.x, sources.y, sources.e1, sources.e2, sources.f1, sources.f2, sources.sigs, sources.sigf]))
    else:
        # If we're not doing a full reconstruction, we need to load in the data
        dir = 'Data//'
        file_name = 'a2744' 
        file_name += '_par' if field == 'parallel' else '_clu' 
        lenses = pipeline.Lens(*np.load(dir + file_name + '_lenses.npy'))
        sources = pipeline.Source(*np.load(dir + file_name + '_sources.npy'))

    # Generate a convergence map that spans the same area as the image
    arcsec_per_pixel = 0.05 # From the intrumentation documentation
    # I'd like the kappa scale to be 1 arcsecond per pixel. This means each kappa pixel is 20 image pixels
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X = np.linspace(0, extent[1], int(extent[1]))
    Y = np.linspace(0, extent[3], int(extent[3]))
    X, Y = np.meshgrid(X, Y)
    kappa = np.zeros_like(X)

    for k in range(len(lenses.x)):
        r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2 + 0.5**2) # Add 0.5 to avoid division by 0 
        kappa += lenses.te[k] / (2 * r)
    
    # Calculate the mass within the convergence map
    # Kappa has the same extent as the image, which means that the scale of the convergence map is the same as the scale of the image
    mass = calculate_mass(kappa, z_cluster, z_source, 1) # Calculate the mass within the convergence map

    # Now, perform a mass sheet transformation
    def mass_sheet(kappa, k):
        return k*kappa + (1 - k)
    
    kappa = mass_sheet(kappa, (1-np.mean(kappa))**-1) # Set the mean kappa to 0

    # Let's also smooth the convergence map - we don't expect to recover information on small scales
    # kernel = create_gaussian_kernel(100, 1) # For now, lets smooth on the scale of one arcsecond
    # kappa = convolve_image(kappa, kernel)

    # Create labels for the plot
    dir = 'Images//abel//'
    file_name = 'A2744_kappa_'
    file_name += 'par' if field == 'parallel' else 'clu'
    file_name += '_rand' if randomize else ''

    title = 'Abell 2744 Convergence Map - '
    title += 'Parallel Field' if field == 'parallel' else 'Cluster Field' 
    title += ' - Randomized' if randomize else '' 
    if field == 'cluster':
        title += '\n Reconstructed Mass:' + r' $M = {:.2e} M_\odot$'.format(mass)

    # Plot the convergence map
    fig, ax = plt.subplots()
    plot_cluster(ax, img_data, X, Y, kappa, None, None, extent, vmax, legend=False)
    ax.set_title(title)
    if randomize:
        # If there's already a randomized plot, I don't want to overwrite it
        # Can we add a number to the end of the file name?
        # First, check if the file exists
        i = 1
        while True:
            if os.path.isfile(dir + file_name + f'_{i}.png'):
                i += 1
            else:
                break
        file_name += f'_{i}'
    plt.savefig(dir + file_name + '.png')
    # plt.show()


if __name__ == '__main__':
    reconstruct_a2774(field='cluster', randomize=False, full_reconstruction=False)
