import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.visualization import ImageNormalize, LogStretch
import pipeline

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


def get_wcs(header) -> WCS:
    # Get the WCS from the header
    wcs = WCS(header)
    return wcs


def plotter(img_data, extent, coords, wcs):
    '''
    Plots the image and the objects on the image, given the image data, 
    the extent of the image, and the coordinates of the objects.
    '''
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the image
    norm = ImageNormalize(img_data, vmin=0, vmax=1, stretch=LogStretch())
    ax.imshow(img_data, cmap='gray_r', norm=norm, extent=extent, origin='lower')

    # Plot the objects
    ax.scatter(coords[:, 0], coords[:, 1], s=5, c='r', label = 'Catalogue Objects')
    ax.legend()
    ax.set_xlabel('RA (arcsec)')
    ax.set_ylabel('DEC (arcsec)')
    ax.set_title('A2744 Cluster Lenser Data')
    
    plt.show()



# fits_file_path = 'Data\hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f435w_v1.0_drz_sci.fits'
# fits_file_path = 'Data\hlsp_frontier_hst_acs-30mas-selfcal_abell2744-hffpar_f606w_v1.0_drz_sci.fits'
# fits_file_path = 'Data\hlsp_frontier_hst_acs-30mas_abell2744-hffpar_f606w_v1.0_drz_sci.fits'
fits_file_path = 'Data\color_hlsp_frontier_hst_acs-30mas_abell2744_f814w_v1.0-epoch2_f606w_v1.0_f435w_v1.0_drz_sci.fits'
csv_file_path = 'a2744_clu_lenser.csv'

img_data, header = get_img_data(fits_file_path)
# This image data is 3D - choose the first layer
img_data1 = img_data[0, :, :]
img_data2 = img_data[1, :, :]
img_data3 = img_data[2, :, :]
coords = get_coords(csv_file_path, coord_type='pixels')
x_pix = coords[:, 0]
y_pix = coords[:, 1]

# Okay, lets try adding the contours to this
'''
lenses = pipeline.Lens(*np.load('Data//a2744_lenses.npy'))
sources = pipeline.Source(*np.load('Data//a2744_sources.npy'))
chi2 = lenses.update_chi2_values(sources)

# Generate a convergence map of the lensing field, spanning the range of the sources
x = np.linspace(min(x_pix), max(x_pix), 100)
y = np.linspace(min(y_pix), max(y_pix), 100)

X, Y = np.meshgrid(x, y)
kappa = np.zeros_like(X)
for k in range(len(lenses.x)):
    r = np.sqrt((X - lenses.x[k])**2 + (Y - lenses.y[k])**2)
    kappa += lenses.te[k] / (2 * r)
'''
x_offset = 1474-1359
y_offset = 8364-8309
print(x_offset, y_offset)
plt.figure()
norm = ImageNormalize(img_data1, vmin=0, vmax=1, stretch=LogStretch())
plt.imshow(img_data1, cmap='gray_r', origin='lower', norm=norm)
plt.imshow(img_data2, cmap='gray_r', origin='lower', norm=norm)
# plt.imshow(img_data3, cmap='gray_r', origin='lower', norm=norm)
plt.scatter(x_pix + x_offset, y_pix + y_offset, s=2, c='r', label = 'Catalogue Objects', alpha=0.5)
# plt.contour(X, Y, kappa, 20, colors='k')
plt.show()

# plotter(img_data, None, coords, wcs)
