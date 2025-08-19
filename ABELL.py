import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LogStretch
from pathlib import Path
import warnings

import csv
import concurrent.futures
from functools import partial
import tqdm

# Import custom modules (ensure these are in your Python path)
import main
import source_obj
import halo_obj
import utils
import pipeline

flexion_catalog_path = 'JWST_Data/JWST/ABELL_2744/Catalogs/multiband_flexion.pkl'
source_catalog_path = 'JWST_Data/JWST/ABELL_2744/Catalogs/stacked_cat.ecsv'
redshift_path = 'JWST_Data/JWST/ABELL_2744/Catalogs/UNCOVER_DR2_LW_SUPER_catalog.fits'

from scipy.optimize import linear_sum_assignment

def match_one_to_one_by_position(
    xc_list, yc_list,                  # arrays (N,)
    xc_redshift, yc_redshift, z_phot,  # arrays (M,)
    tol=0.01                           # matching radius (same units as coords)
):
    """
    Returns:
        redshifts:   (N,) float array; matched z or np.nan
        match_idx:   (N,) int array; index into redshift catalog or -1
        separations: (N,) float array; distance to matched object or np.inf
        stats:       dict with counts {'matched': k, 'unmatched': N-k}
    """
    # Convert to arrays
    xc_list = np.asarray(xc_list, float)
    yc_list = np.asarray(yc_list, float)
    xc_r    = np.asarray(xc_redshift, float)
    yc_r    = np.asarray(yc_redshift, float)
    z_phot  = np.asarray(z_phot, float)

    N = xc_list.size
    M = xc_r.size

    # Build full pairwise distance matrix (N x M); Euclidean in the given plane
    # If these are sky coords in arcsec already, this is appropriate.
    dx = xc_list[:, None] - xc_r[None, :]
    dy = yc_list[:, None] - yc_r[None, :]
    D  = np.hypot(dx, dy)

    # Disallow pairs beyond tolerance by assigning a very large cost
    BIG = 1e9
    cost = D.copy()
    cost[D > tol] = BIG

    # Hungarian assignment minimizes total cost over all rows/cols
    # It returns one column for each row (size = min(N, M)) but conceptually
    # we map each source row to at most one redshift column.
    row_ind, col_ind = linear_sum_assignment(cost)

    # Initialize outputs
    match_idx   = np.full(N, -1, dtype=int)
    separations = np.full(N, np.inf, dtype=float)
    redshifts   = np.full(N, np.nan, dtype=float)

    # Accept only assignments within tolerance
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < BIG:
            match_idx[r]   = c
            separations[r] = D[r, c]
            redshifts[r]   = z_phot[c]

    stats = {
        'matched': int(np.isfinite(separations).sum()),
        'unmatched': int(np.isinf(separations).sum())
    }
    return redshifts, match_idx, separations, stats



class ABELL:
    def __init__(self):
        self.flexion_catalog_path = flexion_catalog_path
        self.source_catalog_path = source_catalog_path
        self.redshift_path = redshift_path
        self.CDELT = 8.54006306703281e-6 * 3600  # degrees/pixel converted to arcsec/pixel
        self.z_source = 0.8
        self.z_cluster = 0.308
    
    def read_source_catalog(self):
        """
        Reads the source catalog.
        """
        table = Table.read(self.source_catalog_path)
        self.x_centroids = np.array(table['xcentroid'])
        self.y_centroids = np.array(table['ycentroid'])
        self.labels = np.array(table['label'])
        sky_centroids = np.array(table['sky_centroid'])
        arr = np.asarray(sky_centroids, dtype=object).ravel()
        coords = SkyCoord(arr, unit='deg')
        ra_deg = coords.ra.deg
        dec_deg = coords.dec.deg
        self.RA = ra_deg
        self.Dec = dec_deg


    def read_flexion_catalog(self):
        """
        Reads the flexion catalog and filters out bad data.
        """
        # Read the flexion catalog
        df = pd.read_pickle(self.flexion_catalog_path)
        print(df.columns)
        # Extract relevant columns
        self.IDs = df['label'].to_numpy()
        self.q = df['q'].to_numpy()
        self.phi = df['phi'].to_numpy()
        self.F1_fit = df['F1_fit'].to_numpy()
        self.F2_fit = df['F2_fit'].to_numpy() 
        self.G1_fit = df['G1_fit'].to_numpy() 
        self.G2_fit = df['G2_fit'].to_numpy()
        self.a = df['a'].to_numpy()
        self.chi2 = df['rchi2'].to_numpy()
        self.rs = df['rs'].to_numpy() # We don't need to carry this past this function

        # Convert to arcseconds / inverse arcseconds - do this step *immediately*, there are no circumstances where we want to keep the data in pixels
        self.a *= self.CDELT
        self.F1_fit /= self.CDELT
        self.F2_fit /= self.CDELT
        self.G1_fit /= self.CDELT
        self.G2_fit /= self.CDELT
        self.rs *= self.CDELT


    def read_redshift_data(self):
        """
        Reads the redshift data.
        """
        self.redshift_data = fits.open(self.redshift_path)[1].data
        #print(self.redshift_data.columns)

        self.xc_redshift = self.redshift_data['ra']
        self.yc_redshift = self.redshift_data['dec']
        self.z_phot = self.redshift_data['z_phot']
        z_spec = self.redshift_data['z_spec']
        use_phot = self.redshift_data['use_phot']

        # Create a mask, only use z_phot when use_phot = 1
        # Else, set z_phot to nan
        self.z_phot[use_phot != 1] = np.nan


    def match_sources(self):
        """
        Matches each object in self.IDs to a position in the source catalog
        defined by self.labels, self.x_centroids, and self.y_centroids.

        Populates:
            self.xc: array of x-centroid positions for matched IDs
            self.yc: array of y-centroid positions for matched IDs

        Notes:
            - If an ID is not found in the source catalog, its position is set to NaN.
            - Note to self - this functions to make the cuts in the source catalog that we made in the flexion catalog.
        """
        # Build a mapping from label → index
        label_to_index = {label: idx for idx, label in enumerate(self.labels)}

        # Match IDs and extract centroids
        xc_list = []
        yc_list = []
        RA_list = []
        Dec_list = []

        for ID in self.IDs:
            idx = label_to_index.get(ID)
            if idx is not None:
                xc_list.append(self.x_centroids[idx])
                yc_list.append(self.y_centroids[idx])
                RA_list.append(self.RA[idx])
                Dec_list.append(self.Dec[idx])
            else:
                # Necessary to assign NaN if ID not found - otherwise, xc_list and yc_list will be of different lengths
                # This is a warning, not an error, as we want to continue processing
                # Also this has never happened in the use of this script - it's unlikely to happen in practice
                warnings.warn(f"ID '{ID}' not found in source catalog. Assigning NaN.")
                xc_list.append(np.nan)
                yc_list.append(np.nan)
                RA_list.append(np.nan)
                Dec_list.append(np.nan)

        plt.figure()
        plt.scatter(RA_list, Dec_list, s=5, alpha=0.5, color='blue')
        plt.scatter(self.xc_redshift, self.yc_redshift, s=5, alpha=0.5, color='red')
        plt.xlabel('X Centroid')
        plt.ylabel('Y Centroid')
        plt.title('Matched Source Catalog')
        plt.show()

        # Match RA_list / Dec_list to xc_redshift and yc_redshift
        redshift = np.zeros_like(xc_list, dtype=float)

        for i in range(len(xc_list)):
            if not np.isnan(xc_list[i]) and not np.isnan(yc_list[i]):
                # Find the closest redshift
                distance = np.sqrt((self.xc_redshift - xc_list[i])**2 + (self.yc_redshift - yc_list[i])**2)
                matching_redshift = self.z_phot[distance == distance.min()]
                if matching_redshift.size > 0:
                    redshift[i] = matching_redshift.mean()
                else:
                    redshift[i] = np.nan

        self.redshift = redshift
        print('Matched redshifts: {}, Unmatched redshifts: {}'.format(np.sum(~np.isnan(redshift)), np.sum(np.isnan(redshift))))

if __name__ == '__main__':
    abell_2744 = ABELL()
    abell_2744.read_source_catalog()
    abell_2744.read_redshift_data()
    abell_2744.read_flexion_catalog()
    abell_2744.match_sources()
    print("Redshift data read successfully.")

