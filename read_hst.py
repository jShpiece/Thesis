import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
import warnings
import os
import main
import source_obj
import halo_obj
import utils
from astropy.visualization import hist as fancy_hist
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd
from matplotlib import gridspec

plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots
h = 0.7 # Hubble constant
z_cluster = 0.308 # Redshift of the cluster


def plot_cluster(ax, img_data, X, Y, conv, lenses, sources, extent, vmax=1, legend=True):
    # Plotting function to overlay lensing reconstruction - in the form of lenses or convergence contours, 
    # ontop of an optical image of the cluster
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    for img in img_data:
        # Allow for multiple images to be overlayed - allows for band or epoch stacking
        norm = ImageNormalize(img, vmin=0, vmax=vmax, stretch=LogStretch())
        ax.imshow(img, origin='lower', extent=extent, norm=norm, cmap='gray')

    if conv is not None:
        # Adjusted contour levels for better feature representation.
        contour_levels = np.percentile(conv, np.linspace(60, 100, 5))

        # Fine-tuned alpha value for better overlay visibility.
        '''
        color_map_overlay = ax.imshow(
            conv, 
            cmap='viridis', 
            origin='lower', 
            extent=extent, 
            alpha=0.6, 
            vmin=0, 
            vmax=np.max(contour_levels)
        )

        # Customized color bar for clarity.
        color_bar = plt.colorbar(color_map_overlay, ax=ax)
        color_bar.set_label(r'$\kappa$', rotation=0, labelpad=10)
        '''
        # Only overlay the contours of the convergence map - the color map will be too busy
        # Overlay the contours of the convergence map.
        ax.contour(X, Y, conv, levels=contour_levels, cmap='plasma', linewidths=1.5)

    if lenses is not None:
        ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
        for i, eR in enumerate(lenses.te):
            ax.annotate(np.round(eR, 2), (lenses.x[i], lenses.y[i]))
    if sources is not None:
        ax.scatter(sources.x, sources.y, marker='.', color='blue', alpha=0.5, label='Sources')

    if legend:
        ax.legend()


def compare_mass_estimates(halos, plot_name):
    # Radii in kpc
    r = np.linspace(100, 300, 1000)  
    
    # Literature mass estimates: dictionary of form {label: (mass, radius)}
    mass_estimates = {
        'MARS': (1.73e14, 200, ), 
        'Bird': (1.93e14, 200), 
        'GRALE': (2.25e14, 250),
        'Merten et al.': (2.24e14, 250)
    }
    
    # 2D grid setup (in kpc)
    nx, ny = 200, 200
    x_range, y_range = (-500, 500), (-500, 500)  # Adjust as appropriate
    x_vals = np.linspace(x_range[0], x_range[1], nx)
    y_vals = np.linspace(y_range[0], y_range[1], ny)
    M_2D_total = np.zeros((ny, nx), dtype=float)
    
    # Sum the mass grids from all halos
    for i in range(len(halos.x)):
        # Compute 2D mass for this halo, centered at (halo.x, halo.y) if needed
        # Construct the halo (non-iterable)
        halo = halo_obj.NFW_Lens(halos.x[i], halos.y[i], [0], halos.concentration[i], halos.mass[i], z_cluster, halos.chi2[i])
        M_2D_halo = utils.nfw_projected_mass(
            halo,
            r_p=0,          # Dummy placeholder since we want the 2D grid
            return_2d=True,
            nx=nx,
            ny=ny,
            x_range=x_range,
            y_range=y_range,
            x_center=halo.x,  # or halo.x_pos if your object stores it differently
            y_center=halo.y   # or halo.y_pos if your object stores it differently
        )
        M_2D_total += M_2D_halo
    
    # Coordinates of each pixel in the grid
    XX, YY = np.meshgrid(x_vals, y_vals)
    RR = np.sqrt(XX**2 + YY**2)  # Distance of each pixel from (0,0), adjust if needed
    
    # Compute the enclosed mass at each radius in r
    mass_enclosed = np.zeros_like(r)
    for i, radius in enumerate(r):
        mask = (RR <= radius)
        mass_enclosed[i] = np.sum(M_2D_total[mask])
    
    # Plot results
    fig, ax = plt.subplots()
    fig.suptitle('Mass Estimates')
    
    # Our reconstruction
    ax.plot(r, mass_enclosed, label='Reconstruction')
    
    # Literature estimates
    markers = ['o', 's', 'D', '^']
    for i, (label, (mass_lit, r_lit)) in enumerate(mass_estimates.items()):
        ax.scatter(r_lit, mass_lit, label=label, marker=markers[i % len(markers)])
    
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel(r'Mass ($M_\odot$)')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(plot_name)



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


def reconstruct_system(file, dx, dy, flags=False, randomize = False, use_flags=[True, True, True], lens_type='SIS'):
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
    sources = source_obj.Source(x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg)
    recovered_lenses, reducedchi2 = main.fit_lensing_field(sources, xmax=xmax, flags=flags, use_flags=use_flags, lens_type=lens_type)

    # Move our sourecs and lenses back to the original centroid
    sources.x += centroid[0]
    sources.y += centroid[1]
    recovered_lenses.x += centroid[0]
    recovered_lenses.y += centroid[1]

    return recovered_lenses, sources, xmax, reducedchi2


def reconstruct_a2744(field='cluster', randomize=False, full_reconstruction=False, use_flags=[True, True, True], lens_type='SIS'):
    '''
    A handler function to plot the lensing field of Abell 2744 - either the cluster or parallel field.
    --------------------
    Parameters:
        field: 'cluster' or 'parallel' - which field to plot
        randomize: whether or not to randomize the source orientations, which helps test the algorithm's tendency to overfit
        full_reconstruction: whether or not to perform a full reconstruction of the lensing field, or to load in a precomputed one
    '''
    # Define some constants
    z_source = 0.52 # Mean redshift of the sources
    arcsec_per_pixel = 0.03 # From the instrumentation documentation

    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings

    fits_file_path, csv_file_path, vmax, dx, dy = get_file_paths(field=field)

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, dx * arcsec_per_pixel, dy * arcsec_per_pixel, flags=True, randomize=randomize, use_flags=use_flags, lens_type=lens_type)
        lenses.mass /= h # Convert the mass to h^-1 solar masses
        print('Completed reconstruction')

        # Save the class objects so that we can replot without having to rerun the code
        if randomize:
            # If we're randomizing, I don't need to save the data
            pass
        else:
            dir = 'Output//abel//'
            file_name = 'a2744' 
            file_name += '_par' if field == 'parallel' else '_clu' 
            if lens_type == 'SIS':
                np.save(dir + file_name + '_lenses', np.array([lenses.x, lenses.y, lenses.te, lenses.chi2]))
            else:
                np.save(dir + file_name + '_lenses', np.array([lenses.x, lenses.y, lenses.mass, lenses.chi2]))
            np.save(dir + file_name + '_sources', np.array([sources.x, sources.y, sources.e1, sources.e2, sources.f1, sources.f2, sources.sigs, sources.sigf]))
            print('Saved data')
    else:
        # If we're not doing a full reconstruction, we need to load in the data
        dir = 'Output//abel//'
        file_name = 'a2744' 
        file_name += '_par' if field == 'parallel' else '_clu' 
        lenses = halo_obj.SIS_Lens(*np.load(dir + file_name + '_lenses.npy')) if lens_type == 'SIS' else halo_obj.NFW_Lens(*np.load(dir + file_name + '_lenses.npy'))
        sources = source_obj.Source(*np.load(dir + file_name + '_sources.npy'))

    # Generate a convergence map that spans the same area as the image
    # I'd like the kappa scale to be 1 arcsecond per pixel. This means each kappa pixel is 20 image pixels
    extent = [0, img_data[0].shape[1] * arcsec_per_pixel, 0, img_data[0].shape[0] * arcsec_per_pixel]

    X, Y, kappa = utils.calculate_kappa(lenses, extent, smoothing_scale=1, lens_type=lens_type)
    print('Calculated kappa')
    
    # kappa = mass_sheet_transformation(kappa, (1-np.mean(kappa))**-1) # Set the mean kappa to 0
    if lens_type=='SIS':
        mass = utils.calculate_mass(kappa, z_cluster, z_source, 1) # Calculate the mass within the convergence map
    else:
        mass = np.sum(lenses.mass) # Calculate the mass of the NFW lenses

    # Create labels for the plot
    dir = 'Output/abel/'

    # Build the file name
    file_name = 'A2744_kappa_'
    file_name += 'par' if field == 'parallel' else 'clu'

    if randomize:
        file_name += '_rand'

    file_name += '_SIS' if lens_type == 'SIS' else '_NFW'

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
    title = 'Abell 2744 Convergence Map - '
    title += 'Parallel Field' if field == 'parallel' else 'Cluster Field'

    if randomize:
        title += ' - Randomized'

    title += f'\n M = {mass:.3e} $h^{{-1}} M_\\odot$'

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
    
    # Create the figure and gridspec
    
    fig = plt.figure(figsize=(8, 10))
    # gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])

    # Create the main plot axis
    # ax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(111)
    plot_cluster(ax, img_data, X, Y, kappa, None, None, extent, vmax, legend=False)
    ax.set_title(title)

    # Save the figure
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
    
    # Now compare the mass estimates - for now, only do this for all signals
    plot_name = dir + file_name + '_mass.png'
    compare_mass_estimates(lenses, plot_name)
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
            reconstruct_a2744(field=field, randomize=False, full_reconstruction=True, use_flags=use_flags, lens_type='NFW')