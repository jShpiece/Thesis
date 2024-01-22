import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
import warnings
import os
import pipeline
from utils import calculate_mass, mass_sheet_transformation, print_progress_bar
from astropy.visualization import hist as fancy_hist
import scipy.ndimage
from matplotlib.path import Path
from scipy import ndimage

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
        # Adjusted contour levels for better feature representation.
        contour_levels = np.percentile(conv, np.linspace(60, 100, 5))

        # Contour lines with enhanced visibility.
        contours = ax.contour(
            X, Y, conv, 
            levels=contour_levels, 
            cmap='plasma', 
            linestyles='-', 
            linewidths=1.5
        )

        # Fine-tuned alpha value for better overlay visibility.
        color_map_overlay = ax.imshow(
            conv, 
            cmap='viridis', 
            origin='lower', 
            extent=extent, 
            alpha=0.4, 
            vmin=0, 
            vmax=np.max(contour_levels)
        )

        # Customized color bar for clarity.
        color_bar = plt.colorbar(color_map_overlay, ax=ax)
        color_bar.set_label(r'$\kappa$', rotation=0, labelpad=10)


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
            fits_file_path = 'Data/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
            csv_file_path = 'Data/a2744_clu_lenser.csv'
            vmax = 1 # Set the maximum value for the image normalization
            dx = 115
            dy = 55
        elif field == 'parallel':
            fits_file_path = 'Data/hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz.fits'
            csv_file_path = 'Data/a2744_par_lenser.csv'
            vmax = 0.1 # Set the maximum value for the image normalization
            dx = 865
            dy = 400
    
    return fits_file_path, csv_file_path, vmax, dx, dy


def reconstruct_system(file, dx, dy, flags=False, randomize = False, use_shear=True, use_flexion=True):
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
    sigs_mag = np.mean([np.std(e1), np.std(e2)])
    sigs = np.ones_like(x) * sigs_mag

    sigaf = np.mean([np.std(a*f1), np.std(a*f2)])
    sigf = sigaf / a 

    sigg = np.mean([np.std(g1), np.std(g2)])
    sigg = np.ones_like(x) * sigg

    if randomize:
        # randomize the e1, e2 and f1, f2 vectors by rotating them by a random angle
        rand_angle = np.random.uniform(0, 2*np.pi, len(e1))
        e1, e2 = e1 * np.cos(rand_angle) - e2 * np.sin(rand_angle), e1 * np.sin(rand_angle) + e2 * np.cos(rand_angle)
        f1, f2 = f1 * np.cos(rand_angle) - f2 * np.sin(rand_angle), f1 * np.sin(rand_angle) + f2 * np.cos(rand_angle)

    # Set the centroid to be the origin
    x -= centroid[0]
    y -= centroid[1]

    # Create source object and perform the lensing fit
    sources = pipeline.Source(x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg)
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, xmax=xmax, flags=flags, use_shear=use_shear, use_flexion=use_flexion)

    # Move our sourecs and lenses back to the original centroid
    sources.x += centroid[0]
    sources.y += centroid[1]
    recovered_lenses.x += centroid[0]
    recovered_lenses.y += centroid[1]

    return recovered_lenses, sources, xmax, reducedchi2


def reconstruct_a2744(field='cluster', randomize=False, full_reconstruction=False, use_shear=True, use_flexion=True):
    '''
    A handler function to plot the lensing field of Abell 2744 - either the cluster or parallel field.
    --------------------
    Parameters:
        field: 'cluster' or 'parallel' - which field to plot
        randomize: whether or not to randomize the source orientations, which helps test the algorithm's tendency to overfit
        full_reconstruction: whether or not to perform a full reconstruction of the lensing field, or to load in a precomputed one
    '''
    # Define some constants
    z_cluster = 0.308 # Redshift of the cluster
    z_source = 0.52 # Mean redshift of the sources
    arcsec_per_pixel = 0.03 # From the intrumentation documentation

    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings


    fits_file_path, csv_file_path, vmax, dx, dy = get_file_paths(field=field)

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, dx * arcsec_per_pixel, dy * arcsec_per_pixel, flags=True, randomize=randomize, use_shear=use_shear, use_flexion=use_flexion)

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
    # I'd like the kappa scale to be 1 arcsecond per pixel. This means each kappa pixel is 20 image pixels
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X = np.linspace(0, extent[1], int(extent[1]))
    Y = np.linspace(0, extent[3], int(extent[3]))
    X, Y = np.meshgrid(X, Y)
    kappa = np.zeros_like(X)

    for k in range(len(lenses.x)):
        r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2 + 0.5**2) # Add 0.5 to avoid division by 0 
        kappa += lenses.te[k] / (2 * r)
    
    # Smooth the convergence map with a gaussian kernel (use an external function)
    kappa = scipy.ndimage.gaussian_filter(kappa, sigma=4)
    
    # kappa = mass_sheet_transformation(kappa, (1-np.mean(kappa))**-1) # Set the mean kappa to 0
    mass = calculate_mass(kappa, z_cluster, z_source, 1) # Calculate the mass within the convergence map

    # Create labels for the plot
    dir = 'Images//abel//'
    file_name = 'A2744_kappa_'
    file_name += 'par' if field == 'parallel' else 'clu'
    file_name += '_rand' if randomize else ''
    file_name += '_shear' if use_shear and not use_flexion else ''
    file_name += '_flexion' if use_flexion and not use_shear else ''

    title = 'Abell 2744 Convergence Map - '
    title += 'Parallel Field' if field == 'parallel' else 'Cluster Field'
    title += ' - Randomized' if randomize else ''
    if field == 'cluster':
        title += '\n M = ' + f'{mass:.3e}' + r' $h^{-1} M_\odot$'
    title += '\n Shear only' if use_shear and not use_flexion else ''
    title += '\n Flexion only' if not use_shear and use_flexion else ''

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
    plt.show()


def build_avg_kappa_field(Ntrial = 10, field='cluster', use_shear=True, use_flexion=True):
    # Build an average kappa field by randomizing the source orientations multiple times and averaging the resulting kappa fields
    # This can be used to subtract out features that are due to 'overfitting'
    arcsec_per_pixel = 0.03 # From the intrumentation documentation

    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings

    if field == 'cluster':
        fits_file_path = 'Data/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
        csv_file_path = 'Data/a2744_clu_lenser.csv'
        dx = 115
        dy = 55
    elif field == 'parallel':
        fits_file_path = 'Data/hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz.fits'
        csv_file_path = 'Data/a2744_par_lenser.csv'
        dx = 865
        dy = 400
    else:
        raise ValueError('field must be "cluster" or "parallel"')

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    # Generate a convergence map that spans the same area as the image
    # I'd like the kappa scale to be 1 arcsecond per pixel. This means each kappa pixel is 20 image pixels
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X = np.linspace(0, extent[1], int(extent[1]))
    Y = np.linspace(0, extent[3], int(extent[3]))
    X, Y = np.meshgrid(X, Y)

    avg_kappa_field = np.zeros_like(X)

    # Initialize progress bar
    print_progress_bar(0, Ntrial, prefix='Building Kappa:', suffix='Complete', length=50)

    for n in range(Ntrial):
        lenses, *_ = reconstruct_system(csv_file_path, dx * arcsec_per_pixel, dy * arcsec_per_pixel, flags=False, randomize=True, use_shear=use_shear, use_flexion=use_flexion)
        kappa = np.zeros_like(X)
        for k in range(len(lenses.x)):
            r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2 + 0.5**2) # Add 0.5 to avoid division by 0
            kappa += lenses.te[k] / (2 * r)
        avg_kappa_field += kappa
        print_progress_bar(n+1, Ntrial, prefix='Building Kappa:', suffix='Complete', length=50)
    avg_kappa_field /= Ntrial
    # Save the average kappa field
    dir = 'Data//'
    file_name = 'a2744'
    file_name += '_par' if field == 'parallel' else '_clu'
    np.save(dir + file_name + '_avg_kappa_field', np.array([X, Y, avg_kappa_field]))


def plot_er_dist(merge_radius=1):
    lenses = pipeline.Lens(*np.load('Data//a2744_clu_lenses.npy'))
    # Remove lenses with negative Einstein radii
    lenses = pipeline.Lens(lenses.x[lenses.te > 0], lenses.y[lenses.te > 0], lenses.te[lenses.te > 0], lenses.chi2[lenses.te > 0])
    lenses.merge_close_lenses(merge_radius)
    # Create a plot of the lensing field 
    
    fits_file_path = 'Data/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
    img_data = get_img_data(fits_file_path)[0]
    img_data = [img_data[0], img_data[1]] # Stack the two epochs
    arcsec_per_pixel = 0.03 # From the intrumentation documentation
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    fig, ax = plt.subplots()
    for img in img_data:
        # Allow for multiple images to be overlayed - allows for band or epoch stacking
        norm = ImageNormalize(img, vmin=0, vmax=1, stretch=LogStretch())
        ax.imshow(img, cmap='gray_r', origin='lower', extent=extent, norm=norm)
    cmap = ax.scatter(lenses.x, lenses.y, c = lenses.te, cmap='plasma', s = lenses.te * 100, alpha=0.5)
    color_bar = plt.colorbar(cmap, ax=ax)
    color_bar.set_label(r'$\theta_E$', rotation=270, labelpad=10)
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    ax.set_title('Abell 2744 - Recovered Lenses \n Merge Radius = {} arcsec'.format(merge_radius))
    ax.set_aspect('equal')
    plt.savefig('Images//abel//mergers//field_{}.png'.format(merge_radius))

    fig, ax = plt.subplots()
    fancy_hist(lenses.te, bins='freedman', histtype='step', ax=ax)
    ax.set_xlabel(r'$\theta_E$ (arcsec)')
    ax.set_ylabel('Count')
    ax.set_title('Abell 2744 - Einstein Radii Distribution \n Merge Radius = {} arcsec'.format(merge_radius))
    plt.savefig('Images//abel//mergers//hist_{}.png'.format(merge_radius))


if __name__ == '__main__':
    reconstruct_a2744(field='cluster', randomize=False, full_reconstruction=True, use_shear=True, use_flexion=True)