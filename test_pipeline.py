import numpy as np
import matplotlib.pyplot as plt
import pipeline
from utils import print_progress_bar, create_gaussian_kernel, convolve_image
import time
from astropy.visualization import hist as fancyhist
import warnings
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
from matplotlib import cm
from astropy import units as u

plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

# Define default noise parameters
sigf = 0.01
sigs = 0.1 

# ------------------------
# Testing Plotting Functions
# ------------------------

def _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, title, ax=None, legend=True):
    """Private helper function to plot the results of lensing reconstruction."""
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
    for i, eR in enumerate(lenses.te):
        ax.annotate(np.round(eR, 2), (lenses.x[i], lenses.y[i]))
    ax.scatter(sources.x, sources.y, marker='.', color='blue', alpha=0.5, label='Sources')
    if true_lenses is not None:
        ax.scatter(true_lenses.x, true_lenses.y, marker='x', color='green', label='True Lenses')
    if legend:
        ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if xmax is not None:
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
    ax.set_aspect('equal')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))


def plot_random_realizations(xsol, ysol, er, true_lenses, Nlens, Nsource, Ntrials, xmax, distinguish_lenses=False):
    '''
    Plot histograms of the random realizations.
    '''
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    param_labels = [r'$x$', r'$y$', r'$\theta_E$']

    # Plot histograms of the recovered lens positions
    for lens_num in range(Nlens):
        color = colors[lens_num % len(colors)]
        for a, data, param_label in zip(ax, [xsol, ysol, er], param_labels):
            range_val = None
            if param_label in [r'$x$', r'$y$']:
                range_val = (-xmax, xmax)
            elif param_label == r'$\theta_E$':
                range_val = (0, 5)
            if distinguish_lenses:
                fancyhist(data[:, lens_num], ax=a, bins='scott', histtype='step', density=True, color=color, label=f'Lens {lens_num + 1}', range=range_val)
            else:
                fancyhist(data.flatten(), ax=a, bins='scott', histtype='step', density=True, color=colors[0], label=f'Lens {lens_num + 1}', range=range_val)
            a.set_xlabel(param_label)
            a.set_ylabel('Probability Density')

    eR_median = np.median(er.flatten())
    ax[2].vlines(eR_median, 0, ax[2].get_ylim()[1], color='blue', label='Median = {:.2f}'.format(eR_median))

    # Plot true lens positions
    for a, true_value in zip(ax, [true_lenses.x, true_lenses.y, true_lenses.te]):
        a.vlines(true_value, 0, a.get_ylim()[1], color='red', label='True Lenses')
        a.legend(loc='upper right')

    fig.suptitle(f'Random Realization Test \n {Nlens} lenses, {Nsource} sources \n {Ntrials} trials')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Images//rand_real//rr_test_{Nlens}_lens_{Nsource}_source_{Ntrials}.png')
    plt.close()


def plot_cluster(ax, img_data, X, Y, conv, lenses, sources, extent, legend=True):
    # Plotting function to overlay lensing reconstruction - in the form of lenses or convergence contours, 
    # ontop of an optical image of the cluster
    ax.set_xlabel('x (arcsec)')
    ax.set_ylabel('y (arcsec)')
    for img in img_data:
        # Allow for multiple images to be overlayed - allows for band or epoch stacking
        norm = ImageNormalize(img, vmin=0, vmax=1, stretch=LogStretch())
        ax.imshow(img, cmap='gray_r', origin='lower', extent=extent, norm=norm)

    if conv is not None:
        percentiles = np.percentile(conv, np.linspace(0, 100, 7))
        contours = ax.contour(X, Y, conv, levels=percentiles, cmap='viridis', linestyles='solid') # plot contours
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

# ------------------------
# Test Implementation Functions
# ------------------------

def run_simple_test(Nlens, Nsource, xmax, flags=False, lens_random=False, source_random=True):
    """Runs a basic test of the algorithm.
    params:
        Nlens: number of lenses
        Nsource: number of sources
        xmax: range of lensing field
        flags: if True, print out intermediate results
    """
    start_time = time.time()

    # Initialize true lens and source configurations
    lenses = pipeline.createLenses(nlens=Nlens, randompos=lens_random, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=source_random, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = Nlens, flags=flags)
    
    print('Time elapsed:', time.time() - start_time)
    
    _plot_results(xmax, recovered_lenses, sources, lenses.x, lenses.y, reducedchi2, 'Lensing Reconstruction')
    plt.savefig('Images//tests//simple_test_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def visualize_pipeline_steps(Nlens, Nsource, xmax):
    """Visualizes each step of the lensing reconstruction pipeline.
    params:
        nlens: number of lenses
        nsource: number of sources
        xmax: range of lensing field
    """
    # Setup lensing and source configurations
    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)
    sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
    xl, yl, _ = true_lenses.x, true_lenses.y, true_lenses.te

    # Arrange a plot with 5 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lensing Reconstruction Pipeline', fontsize=16)

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Initial Lens Positions', ax=axarr[0,0])


    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, sigf, sigs)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Optimized Lens Positions', ax=axarr[0,1])


    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Filtered Lens Positions', ax=axarr[0,2])

    # Step 4: Merge lenses that are too close to each other
    lenses.merge_close_lenses()
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Merged Lens Positions', ax=axarr[1,0])

    # Step 5: Iterative elimination
    lens_floors = np.arange(1, len(lenses.x) + 1)
    best_dist = np.abs(reducedchi2 - 1)
    for lens_floor in lens_floors:
        # Clone the lenses object
        test_lenses = pipeline.Lens(lenses.x, lenses.y, lenses.te, lenses.chi2)
        test_lenses.iterative_elimination(lens_floor=lens_floor)
        reducedchi2 = test_lenses.update_chi2_values(sources, sigf, sigs)
        new_dist = np.abs(reducedchi2 - 1)
        if new_dist < best_dist:
            best_dist = new_dist
            best_lenses = test_lenses
        lenses = best_lenses
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Iterative Elimination', ax=axarr[1,1])

    # Step 6: Final minimization
    lenses.full_minimization(sources, sigf, sigs)
    reducedchi2 = lenses.update_chi2_values(sources, sigf, sigs)
    _plot_results(xmax, lenses, sources, xl, yl, reducedchi2, 'Final Minimization', ax=axarr[1,2])

    # Save and show the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images//tests//breakdown_{}_lens_{}_source.png'.format(Nlens, Nsource))
    plt.show()


def generate_random_realizations(Ntrials, Nlens=1, Nsource=1, xmax=10, sigf=0.01, sigs=0.1):
    '''
    Generate random realizations of the lensing field.
    '''
    # True lens configuration
    true_lenses = pipeline.createLenses(nlens=Nlens, randompos=False, xmax=xmax)

    # Store solution arrays
    xsol, ysol, er = np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens)), np.empty((Ntrials, Nlens))

    # Initialize a progress bar
    print_progress_bar(0, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)
    
    for trial in range(Ntrials):
        sources = pipeline.createSources(true_lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=True, xmax=xmax)
        lenses, _ = pipeline.fit_lensing_field(sources, sigs=sigs, sigf=sigf, xmax=xmax, lens_floor=Nlens, flags=False)
        
        num_lenses_recovered = len(lenses.x)
        # Take the minimum of the recovered lenses and the specified number of lenses
        for i in range(min(num_lenses_recovered, Nlens)):
            xsol[trial, i], ysol[trial, i], er[trial, i] = lenses.x[i], lenses.y[i], lenses.te[i]
    
        # Update progress bar
        print_progress_bar(trial, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)
    
    # Finalize the progress bar so that I don't lose my mind
    print_progress_bar(Ntrials, Ntrials, prefix='Random Realization Progress:', suffix='Complete', length=50)

    np.save('Data//xsol_{}_lens_{}_source'.format(Nlens, Nsource), xsol)
    np.save('Data//ysol_{}_lens_{}_source'.format(Nlens, Nsource), ysol)
    np.save('Data//er_{}_lens_{}_source'.format(Nlens, Nsource), er)

    return xsol, ysol, er, true_lenses


def assess_number_recovered(Nlens, Nsource, xmax, sigf=0.01, sigs=0.1, lens_random=False, source_random=True):
    '''
    This runs a simple test of the algorithm, and returns the number of lenses recovered.
    Use this to determine whether the algorithm is able to recover the correct number of lenses when 
    the number of lenses is not known a priori by the algorithm.
    '''
    # Initialize true lens and source configurations
    lenses = pipeline.createLenses(nlens=Nlens, randompos=lens_random, xmax=xmax)
    sources = pipeline.createSources(lenses, ns=Nsource, sigf=sigf, sigs=sigs, randompos=source_random, xmax=xmax)

    # Perform lens position optimization
    recovered_lenses, reducedchi2 = pipeline.fit_lensing_field(sources, sigf=sigf, sigs=sigs, xmax=xmax, lens_floor = Nlens, flags=False)
    # Don't count any lenses with an einstein radius of 0
    num_recovered = 0
    for eR in recovered_lenses.te:
        if eR > 0:
            num_recovered += 1
    return num_recovered


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


def overlay_real_img(image_path:str, ax, conv:np.ndarray, nlevels:int=7) -> None:
    """
    Method to overlay **arr** map contours onto the **image**

    @image: Path to image containing cluster to overlay contours
    @ax: matplotlib axis object where the image is to be displayed
    @conv: a 2D numpy array containing the contours to overlay
    @nlevels: Number of contours
   
    """
    # open fits file
    img = fits.open(image_path)
    img_data = img['SCI'].data

    CDELT = img['SCI'].header['CDELT1'] * u.deg
    ROWS, COLS = img['SCI'].data.shape
    fsize = max(ROWS*CDELT, COLS*CDELT)

    unit = u.arcsec

    # Code to overlay contours associated with a (N1 X N2) image onto a window of physical size (fize X fsize)
    ROWS, COLS = conv.shape
    X = np.linspace(0, fsize.to(unit).value, ROWS)
    Y = np.linspace(0, fsize.to(unit).value, COLS)

    # contour levels
    levels = np.linspace(conv.min(), conv.max(), nlevels)

    norm = ImageNormalize(img_data, vmin=0, vmax=10, stretch=LogStretch())
    extent = [0, fsize.to(unit).value, 0, fsize.to(unit).value] # to display appropriate frame size in matplotlib window

    im = ax.imshow(img['SCI'].data, norm=norm, origin='lower', cmap='gray_r', extent=extent) # plot the science image
    contours = ax.contour(X, Y, conv, levels, cmap='plasma', linestyles='solid') # plot contours
    ax.clabel(contours, inline=False, colors='blue') # attaches label to each contour

    return None


def plot_a2774_field(field='cluster', randomize=False, full_reconstruction=False):
    '''
    A handler function to plot the lensing field of Abell 2744 - either the cluster or parallel field.
    --------------------
    Parameters:
        field: 'cluster' or 'parallel' - which field to plot
        randomize: whether or not to randomize the source orientations, which helps test the algorithm's tendency to overfit
        full_reconstruction: whether or not to perform a full reconstruction of the lensing field, or to load in a precomputed one
    '''
    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings

    if field == 'cluster':
        fits_file_path = 'Data/color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
        csv_file_path = 'a2744_clu_lenser.csv'
    elif field == 'parallel':
        fits_file_path = 'Data/hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz.fits'
        csv_file_path = 'a2744_par_lenser.csv'

    img_data, _ = get_img_data(fits_file_path)
    if field == 'cluster':
        img_data = [img_data[0], img_data[1]] # Stack the two epochs
    elif field == 'parallel':
        img_data = [img_data]

    if full_reconstruction:
        lenses, sources, _, _ = reconstruct_system(csv_file_path, flags=True, randomize=randomize)

        # Save the class objects so that we can replot without having to rerun the code
        dir = 'Data//'
        file_name = 'a2744' 
        file_name += '_par' if field == 'parallel' else '_clu' 
        file_name += '_rand' if randomize else ''
        np.save(dir + file_name + '_lenses', np.array([lenses.x, lenses.y, lenses.te, lenses.chi2]))
        np.save(dir + file_name + '_sources', np.array([sources.x, sources.y, sources.e1, sources.e2, sources.f1, sources.f2, sources.sigs, sources.sigf]))
    else:
        # Put code here for loading our saved class objects
        dir = 'Data//'
        file_name = 'a2744' 
        file_name += '_par' if field == 'parallel' else '_clu' 
        file_name += '_rand' if randomize else ''
        lenses = pipeline.Lens(*np.load(dir + file_name + '_lenses.npy'))
        sources = pipeline.Source(*np.load(dir + file_name + '_sources.npy'))

    # Generate a convergence map of the lensing field, spanning the range of the sources
    x = np.linspace(min(sources.x)-20, max(sources.x)+20, 100)
    y = np.linspace(min(sources.y)-20, max(sources.y)+20, 100)

    extent = [min(x), max(x), min(y), max(y)]

    X, Y = np.meshgrid(x, y)
    kappa = np.zeros_like(X)
    for k in range(len(lenses.x)):
        r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2 + (0.5)**2)
        kappa += lenses.te[k] / (2 * r)
    
    # Now, perform a mass sheet transformation
    def mass_sheet(kappa, k):
        return k*kappa + (1 - k)
    
    kappa = mass_sheet(kappa, (1-np.mean(kappa))**-1) # Set the mean kappa to 0

    # Let's also smooth the convergence map - we don't expect to recover information on small scales
    kernel = create_gaussian_kernel(100, 1) # For now, lets smooth on the scale of a single pixel
    kappa = convolve_image(kappa, kernel)

    # Create labels for the plot
    dir = 'Images//abel//'
    file_name = 'A2744_kappa_'
    file_name += 'par' if field == 'parallel' else 'clu'
    file_name += '_rand' if randomize else ''

    title = 'Abell 2744 Convergence Map - '
    title += 'Parallel Field' if field == 'parallel' else 'Cluster Field' 
    title += ' - Randomized' if randomize else '' 

    # Plot the convergence map
    fig, ax = plt.subplots()
    plot_cluster(ax, img_data, X, Y, kappa, None, None, extent, legend=False)
    ax.set_title(title)
    plt.savefig(dir + file_name + '.png')
    plt.show()


if __name__ == '__main__':
    plot_a2774_field(field='cluster', randomize=False, full_reconstruction=False)
