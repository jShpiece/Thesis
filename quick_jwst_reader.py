import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch
from astropy import units as u
from astropy.table import Table
import pipeline
import utils
import warnings

plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

def read_file(path):
    """
    Read in the JWST flexion catalog and return a pandas dataframe
    """
    df = pd.read_pickle(path)
    # Column names are as follows
    # ID, q, phi, F1_fit, F2_fit, a 
    # Read these in
    ID = df['label']
    q = df['q']
    phi = df['phi']
    F1_fit = df['F1_fit']
    F2_fit = df['F2_fit']
    a = df['a']

    # Convert to numpy arrays, remove nans
    ID = np.array(ID)
    q = np.array(q)
    phi = np.array(phi)
    F1_fit = np.array(F1_fit)
    F2_fit = np.array(F2_fit)
    a = np.array(a)

    # Remove nans
    cuts = np.where((np.isfinite(q)) & (np.isfinite(phi)) & (np.isfinite(F1_fit)) & (np.isfinite(F2_fit)) & (np.isfinite(a)))[0]
    ID = ID[cuts]
    q = q[cuts]
    phi = phi[cuts]
    F1_fit = F1_fit[cuts]
    F2_fit = F2_fit[cuts]
    a = a[cuts]

    return ID, q, phi, F1_fit, F2_fit, a


def get_img_data(fits_file_path) -> np.ndarray:
    # Get the image data from the fits file
    fits_file = fits.open(fits_file_path)
    img_data = fits_file['SCI'].data
    header = fits_file['SCI'].header
    return img_data, header


def filter_data(ID, xc, yc, q, phi, f1, f2, a):
    # Remove flexions that are too large and q values that are not finite
    cuts = np.where((np.abs(f1) < 10) & (np.abs(f2) < 10) & ((a) < 5) & (np.isfinite(q)) & (a > 0.1))[0]
    ID = ID[cuts]
    xc = xc[cuts]
    yc = yc[cuts]
    q = q[cuts]
    phi = phi[cuts]
    f1 = f1[cuts]
    f2 = f2[cuts]
    a = a[cuts]

    return ID, xc, yc, q, phi, f1, f2, a


def naive_run(lenser_path, cat_path, img_path):
    # Ignore warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings
    ID, q, phi, f1, f2, a = read_file(lenser_path)
    img, header = get_img_data(img_path)
    t = Table.read(cat_path)
    # print(t.colnames)

    xc = t['xcentroid']
    yc = t['ycentroid']

    xc = np.array(xc)
    yc = np.array(yc)

    # Convert to arcsec
    cdelt = header['CDELT2'] * u.deg
    cdelt = cdelt.to(u.arcsec).value
    xc = xc * cdelt
    yc = yc * cdelt
    a = a * cdelt
    f1 = f1 / cdelt 
    f2 = f2 / cdelt

    ID, xc, yc, q, phi, f1, f2, a = filter_data(ID, xc, yc, q, phi, f1, f2, a)


    shear_mag = (q-1)/(q+1)
    e1, e2 = shear_mag * np.cos(2*phi), shear_mag * np.sin(2*phi)

    # Okay...lets try the pipeline
    # Set xmax to be the largest distance from the center
    centroid = np.mean(xc), np.mean(yc)
    xmax = np.max(np.sqrt((xc - centroid[0])**2 + (yc - centroid[1])**2))

    # Get the noise
    sigs_mag = np.mean([np.std(e1), np.std(e2)])
    sigs = np.ones_like(xc) * sigs_mag

    sigaf = np.mean([np.std(a*f1), np.std(a*f2)])
    sigf = sigaf / a 

    sources = pipeline.Source(xc, yc, e1, e2, f1, f2, sigs, sigf)
    lenses, _ = pipeline.fit_lensing_field(sources, xmax, flags = True)

    # Save these
    np.save('Data/JWST/lenses.npy', np.array([lenses.x, lenses.y, lenses.te, lenses.chi2]))
    np.save('Data/JWST/sources.npy', np.array([sources.x, sources.y, sources.e1, sources.e2, sources.f1, sources.f2, sources.sigs, sources.sigf]))


def reconstructor():
    lenser_path = 'Data/JWST/Cluster Field/Catalogs/F115W_flexion.pkl'
    img_path = 'Data\JWST\Cluster Field\Image Data\jw02756-o003_t001_nircam_clear-f115w_i2d.fits'
    cat_path = 'Data\JWST\Cluster Field\Catalogs\stacked_cat.ecsv'

    z_lens = 0.308
    z_source = 0.5
    
    warnings.filterwarnings("ignore", category=RuntimeWarning) # Beginning of pipeline will generate expected RuntimeWarnings
    # naive_run(lenser_path, cat_path, img_path)

    # Load in the data
    lenses = pipeline.Lens(*np.load('Data/JWST/lenses.npy', allow_pickle=True))
    # sources = pipeline.Source(*np.load('Data/JWST/sources.npy', allow_pickle=True))
    img, header = get_img_data(img_path)
    cdelt = header['CDELT2'] * u.deg 
    cdelt = cdelt.to(u.arcsec).value

    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [0, img.shape[1] * cdelt, 0, img.shape[0] * cdelt]
    X, Y, kappa = utils.calculate_kappa(lenses, extent, 5)
    ax.imshow(img, cmap='gray_r', origin='lower', extent=extent, norm=ImageNormalize(img, vmin=0, vmax=100, stretch=LogStretch()))

    # Adjusted contour levels for better feature representation.
    contour_levels = np.percentile(kappa, np.linspace(60, 100, 5))

    # Contour lines with enhanced visibility.
    contours = ax.contour(
        X, Y, kappa, 
        levels=contour_levels, 
        cmap='plasma', 
        linestyles='-', 
        linewidths=1.5
    )

    # Fine-tuned alpha value for better overlay visibility.
    color_map_overlay = ax.imshow(
        kappa, 
        cmap='viridis', 
        origin='lower', 
        extent=extent, 
        alpha=0.2, 
        vmin=0, 
        vmax=np.max(contour_levels)
    )

    # Customized color bar for clarity.
    color_bar = plt.colorbar(color_map_overlay, ax=ax)
    color_bar.set_label(r'$\kappa$', rotation=0, labelpad=10)

    ax.set_xlabel('RA (arcsec)')
    ax.set_ylabel('Dec (arcsec)')
    mass = utils.calculate_mass(kappa, z_lens, z_source, 1)

    ax.set_title('Abell 2744 Convergence Map - JWST Data \n' + f'{mass:.3e}' + r' $h^{-1} M_\odot$')
    plt.savefig('Images/JWST_flexion_lens_map.png', dpi=300)
    plt.show()


def match_sources_and_correlate():
    lenser_path = 'Data/JWST/Cluster Field/Catalogs/F115W_flexion.pkl'
    img_path = 'Data\JWST\Cluster Field\Image Data\jw02756-o003_t001_nircam_clear-f115w_i2d.fits'
    cat_path = 'Data\JWST\Cluster Field\Catalogs\stacked_cat.ecsv'
    
    ''' READ IN JWST DATA '''
    t = Table.read(cat_path)
    # Sky Centroid gives the centroid of the source in the sky
    # in units of RA and Dec
    ra_jwst = t['sky_centroid'].ra
    dec_jwst = t['sky_centroid'].dec
    # Turn these into numpy arrays
    ra_jwst = np.array(ra_jwst)
    dec_jwst = np.array(dec_jwst)

    # Also get the lensing data
    ID_jwst, q_jwst, phi_jwst, f1_jwst, f2_jwst, a_jwst = read_file(lenser_path)
    # Convert to arcsec
    cdelt = 0.031 # arcsec/pixel
    f1_jwst = f1_jwst / cdelt
    f2_jwst = f2_jwst / cdelt
    a_jwst = a_jwst * cdelt


    ID_jwst, ra_jwst, dec_jwst, q_jwst, phi_jwst, f1_jwst, f2_jwst, a_jwst = filter_data(ID_jwst, ra_jwst, dec_jwst, q_jwst, phi_jwst, f1_jwst, f2_jwst, a_jwst)

    ''' READ IN HST DATA '''
    hubble_path = 'Data/a2744_clu_lenser.csv'
    data = np.genfromtxt(hubble_path, delimiter=',')
    with open(hubble_path, 'r') as f:
        header = f.readline().strip().split(',')

    ra_hst_col = header.index('X_WORLD')
    dec_hst_col = header.index('Y_WORLD')
    
    # Get lensing data
    acol = header.index('a') # Shape term
    qcol, phicol = header.index('q'), header.index('phi') # Shape terms
    f1col, f2col = header.index('f1'), header.index('f2') # Flexion terms

    a_hst = data[1:, acol]
    q_hst, phi_hst = data[1:, qcol], data[1:, phicol]
    f1_hst, f2_hst = data[1:, f1col], data[1:, f2col]
    
    ra_hst = data[1:, ra_hst_col]
    dec_hst = data[1:, dec_hst_col]

    ''' FIND MATCHES '''
    match_count = 0
    JWST_IDs = []
    HST_IDs = []
    for i in range(len(ra_hst)):
        for j in range(len(ra_jwst)):
            if np.abs(ra_hst[i] - ra_jwst[j]) < 0.0001 and np.abs(dec_hst[i] - dec_jwst[j]) < 0.0001:
                match_count += 1
                JWST_IDs.append(j)
                HST_IDs.append(i)

    ''' ONLY KEEP MATCHED DATA '''
    # HST
    ra_hst = ra_hst[HST_IDs]
    dec_hst = dec_hst[HST_IDs]
    a_hst = a_hst[HST_IDs]
    q_hst = q_hst[HST_IDs]
    phi_hst = phi_hst[HST_IDs]
    f1_hst = f1_hst[HST_IDs]
    f2_hst = f2_hst[HST_IDs]

    # JWST
    ra_jwst = ra_jwst[JWST_IDs]
    dec_jwst = dec_jwst[JWST_IDs]
    a_jwst = a_jwst[JWST_IDs]
    q_jwst = q_jwst[JWST_IDs]
    phi_jwst = phi_jwst[JWST_IDs]
    f1_jwst = f1_jwst[JWST_IDs]
    f2_jwst = f2_jwst[JWST_IDs]

    # Also remember to compute shear
    shear_mag_hst = (q_hst-1)/(q_hst+1)
    e1_hst, e2_hst = shear_mag_hst * np.cos(2*phi_hst), shear_mag_hst * np.sin(2*phi_hst)

    shear_mag_jwst = (q_jwst-1)/(q_jwst+1)
    e1_jwst, e2_jwst = shear_mag_jwst * np.cos(2*phi_jwst), shear_mag_jwst * np.sin(2*phi_jwst)

    ''' PLOT THE RESULTS '''
    # Want flexion flexion and shear shear plots between hst and jwst
    print('Number of matches: ', match_count)
    # Flexion flexion
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(f1_hst, f1_jwst, s=10, label='F1: correlation = {}'.format(np.round(np.corrcoef(f1_hst, f1_jwst)[0,1], 2)))
    ax.scatter(f2_hst, f2_jwst, s=10, label='F2: correlation = {}'.format(np.round(np.corrcoef(f2_hst, f2_jwst)[0,1], 2)))
    # Create an agreement line
    x = np.linspace(-1, 1, 100)
    ax.plot(x, x, color='black', linestyle='--', label='Agreement')
    ax.set_xlabel('HST')
    ax.set_ylabel('JWST')
    ax.set_title('Flexion Flexion Comparison')
    ax.legend()
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.savefig('Images/flexion_flexion.png', dpi=300)

    # Shear shear
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(e1_hst, e1_jwst, s=10, label=r'$\gamma_1$: correlation = {}'.format(np.round(np.corrcoef(e1_hst, e1_jwst)[0,1], 2)))
    ax.scatter(e2_hst, e2_jwst, s=10, label=r'$\gamma_2$: correlation = {}'.format(np.round(np.corrcoef(e2_hst, e2_jwst)[0,1], 2)))
    # Create an agreement line
    x = np.linspace(-1, 1, 100)
    ax.plot(x, x, color='black', linestyle='--', label='Agreement')
    ax.set_xlabel('HST')
    ax.set_ylabel('JWST')
    ax.set_title('Shear Shear Comparison')
    ax.legend()
    plt.savefig('Images/shear_shear.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(f1_hst*a_hst, f1_jwst*a_jwst, s=10, label='a*F1: correlation = {}'.format(np.round(np.corrcoef(f1_hst*a_hst, f1_jwst*a_jwst)[0,1], 2)))
    ax.scatter(f2_hst*a_hst, f2_jwst*a_jwst, s=10, label='a*F2: correlation = {}'.format(np.round(np.corrcoef(f2_hst*a_hst, f2_jwst*a_jwst)[0,1], 2)))
    # Create an agreement line
    x = np.linspace(-0.5, 0.5, 100)
    print(np.corrcoef(x, x)[0,1])
    ax.plot(x, x, color='black', linestyle='--', label='Agreement')
    ax.set_xlabel('HST')
    ax.set_ylabel('JWST')
    ax.set_title('aF Comparison')
    ax.legend()
    plt.savefig('Images/af.png', dpi=300)
 
if __name__ == '__main__':
    lenser_path = 'Data/JWST/Cluster Field/Catalogs/F115W_flexion.pkl'
    img_path = 'Data\JWST\Cluster Field\Image Data\jw02756-o003_t001_nircam_clear-f115w_i2d.fits'
    cat_path = 'Data\JWST\Cluster Field\Catalogs\stacked_cat.ecsv'
   
    match_sources_and_correlate()
    raise SystemExit

    # Lets look at the distribution of the different terms in the JWST data
    ''' READ IN JWST DATA '''
    t = Table.read(cat_path)
    # Sky Centroid gives the centroid of the source in the sky
    # in units of RA and Dec
    ra_jwst = t['sky_centroid'].ra
    dec_jwst = t['sky_centroid'].dec
    # Turn these into numpy arrays
    ra_jwst = np.array(ra_jwst)
    dec_jwst = np.array(dec_jwst)

    # Also get the lensing data
    ID_jwst, q_jwst, phi_jwst, f1_jwst, f2_jwst, a_jwst = read_file(lenser_path)
    # Convert to arcsec
    cdelt = 0.031 # arcsec/pixel
    f1_jwst = f1_jwst / cdelt
    f2_jwst = f2_jwst / cdelt
    a_jwst = a_jwst * cdelt

    # I'm interested in the distribution of shear, flexion and a right now. 
    # First, lets check out the major outliers
    # Lets define an outlier as anything where a > 5, or |f1| > 10, or |f2| > 10
    outliers_loc = np.where((np.abs(f1_jwst) > 10) | (np.abs(f2_jwst) > 10) | (np.abs(a_jwst) > 5) | (a_jwst < 0.1))[0]
    print('Number of outliers: ', len(outliers_loc))

    # Remove these outliers
    ID_jwst = np.delete(ID_jwst, outliers_loc)
    ra_jwst = np.delete(ra_jwst, outliers_loc)
    dec_jwst = np.delete(dec_jwst, outliers_loc)
    q_jwst = np.delete(q_jwst, outliers_loc)
    phi_jwst = np.delete(phi_jwst, outliers_loc)
    f1_jwst = np.delete(f1_jwst, outliers_loc)
    f2_jwst = np.delete(f2_jwst, outliers_loc)
    a_jwst = np.delete(a_jwst, outliers_loc)

    # Shear
    shear_mag_jwst = (q_jwst-1)/(q_jwst+1)
    e1_jwst, e2_jwst = shear_mag_jwst * np.cos(2*phi_jwst), shear_mag_jwst * np.sin(2*phi_jwst)

    print('Number of sources: ', len(e1_jwst))

    # Shear
    print('Mean shear: ', np.mean(e1_jwst), np.mean(e2_jwst))
    print('Std shear: ', np.std(e1_jwst), np.std(e2_jwst))
    print('Range of shear: ', np.min(e1_jwst), np.max(e1_jwst), np.min(e2_jwst), np.max(e2_jwst))

    # Flexion
    print('Mean flexion: ', np.mean(f1_jwst), np.mean(f2_jwst))
    print('Std flexion: ', np.std(f1_jwst), np.std(f2_jwst))
    print('Range of flexion: ', np.min(f1_jwst), np.max(f1_jwst), np.min(f2_jwst), np.max(f2_jwst))
    
    # a
    print('Mean a: ', np.mean(a_jwst))
    print('Std a: ', np.std(a_jwst))
    print('Range of a: ', np.min(a_jwst), np.max(a_jwst))

    # Lets plot these
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(e1_jwst, e2_jwst, s=10)
    ax.set_xlabel(r'$\gamma_1$')
    ax.set_ylabel(r'$\gamma_2$')
    ax.set_title('Shear Distribution')
    plt.savefig('Images/JWST/shear_dist.png', dpi=300)

    # Flexion
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(f1_jwst, f2_jwst, s=10)
    ax.set_xlabel('F1')
    ax.set_ylabel('F2')
    ax.set_title('Flexion Distribution')
    plt.savefig('Images/JWST/flexion_dist.png', dpi=300)

    # af
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(f1_jwst*a_jwst, f2_jwst*a_jwst, s=10)    
    ax.set_xlabel('aF1')
    ax.set_ylabel('aF2')
    ax.set_title('aF Distribution')
    plt.savefig('Images/JWST/af_dist.png', dpi=300)

    # Lets also make histograms of these
    fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    fig.suptitle('JWST Shear Distribution')
    ax[0].hist(e1_jwst, bins=50)
    ax[0].set_xlabel(r'$\gamma_1$')
    ax[0].set_ylabel('Count')
    ax[1].hist(e2_jwst, bins=50)
    ax[1].set_xlabel(r'$\gamma_2$')
    ax[1].set_ylabel('Count')
    plt.savefig('Images/JWST/shear_dist_hist.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    fig.suptitle('JWST Flexion Distribution')
    ax[0].hist(f1_jwst, bins=100)
    ax[0].set_xlabel('F1')
    ax[0].set_ylabel('Count')
    ax[1].hist(f2_jwst, bins=100)
    ax[1].set_xlabel('F2')
    ax[1].set_ylabel('Count')
    plt.savefig('Images/JWST/flexion_dist_hist.png', dpi=300)

    fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=1)
    fig.suptitle('JWST aF Distribution')
    ax[0].hist(f1_jwst*a_jwst, bins=50)
    ax[0].set_xlabel('aF1')
    ax[0].set_ylabel('Count')
    ax[1].hist(f2_jwst*a_jwst, bins=50)
    ax[1].set_xlabel('aF2')
    ax[1].set_ylabel('Count')
    plt.savefig('Images/JWST/af_dist_hist.png', dpi=300)

    # Lets also look at the distribution of a
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.hist(a_jwst, bins=50)
    ax.set_xlabel('a')
    ax.set_ylabel('Count')
    ax.set_title('a Distribution')
    plt.savefig('Images/JWST/a_dist_hist.png', dpi=300)
