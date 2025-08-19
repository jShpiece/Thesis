import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
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

flexion_catalog_path = 'JWST_Data/JWST/EL_GORDO/Catalogs/multiband_flexion.pkl'
source_catalog_path = 'JWST_Data/JWST/EL_GORDO/Catalogs/stacked_cat.ecsv'
redshift_path = 'JWST_Data/JWST/EL_GORDO/Catalogs/MUSE.dat'


class ELGORDO:
    def __init__(self):
        self.flexion_catalog_path = flexion_catalog_path
        self.source_catalog_path = source_catalog_path
        self.redshift_path = redshift_path
        self.CDELT = 8.54006306703281e-6 * 3600  # degrees/pixel converted to arcsec/pixel
        self.z_source = 1.0
        self.z_cluster = 0.87
    
    def read_source_catalog(self):
        """
        Reads the source catalog.
        """
        table = Table.read(self.source_catalog_path)
        print(table.columns)
        print(table['label'])
        # print the first row
        print(table[0])
        self.x_centroids = np.array(table['xcentroid'])
        self.y_centroids = np.array(table['ycentroid'])
        self.labels = np.array(table['label'])


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
        self.source_RA = df['RA'].to_numpy()
        self.source_DEC = df['DEC'].to_numpy()

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
        self.redshift_data = pd.read_csv(self.redshift_path, delim_whitespace=True)
        # No column names
        # Columns are as follows
        # Unique ID, RA, DEC, Redshift, Redshift quality flag, Multiple lensed sources flag
        # Name these columns in the DataFrame
        self.redshift_data.columns = ['ID', 'RA', 'DEC', 'Redshift', 'Redshift_Quality', 'Multiple_Lensed_Sources']
        print(self.redshift_data.columns)

        # Plot a histogram of redshifts
        plt.figure(figsize=(8, 6))
        plt.hist(self.redshift_data['Redshift'], bins=30, color='blue', alpha=0.7)
        plt.xlabel('Redshift')
        plt.ylabel('Number of Sources')
        plt.title('Histogram of Redshifts')
        plt.grid()
        plt.show()

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

        for ID in self.IDs:
            idx = label_to_index.get(ID)
            if idx is not None:
                xc_list.append(self.x_centroids[idx])
                yc_list.append(self.y_centroids[idx])
            else:
                # Necessary to assign NaN if ID not found - otherwise, xc_list and yc_list will be of different lengths
                # This is a warning, not an error, as we want to continue processing
                # Also this has never happened in the use of this script - it's unlikely to happen in practice
                warnings.warn(f"ID '{ID}' not found in source catalog. Assigning NaN.")
                xc_list.append(np.nan)
                yc_list.append(np.nan)

        # Now, match these sources to the redshift catalog by position (we'll need to match RA and DEC)
        table = 

if __name__ == '__main__':
    el_gordo = ELGORDO()
    el_gordo.read_redshift_data()
    print("Redshift data read successfully.")

