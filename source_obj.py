import numpy as np
import utils
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

class Source:
    """
    Represents a catalog of sources with lensing properties.

    Attributes:
        x (np.ndarray): x-positions of the sources.
        y (np.ndarray): y-positions of the sources.
        e1 (np.ndarray): First component of ellipticity (shear) of the sources.
        e2 (np.ndarray): Second component of ellipticity (shear) of the sources.
        f1 (np.ndarray): First component of flexion of the sources.
        f2 (np.ndarray): Second component of flexion of the sources.
        g1 (np.ndarray): First component of g-flexion of the sources.
        g2 (np.ndarray): Second component of g-flexion of the sources.
        sigs (np.ndarray): Standard deviations of the shear components.
        sigf (np.ndarray): Standard deviations of the flexion components.
        sigg (np.ndarray): Standard deviations of the g-flexion components.
    """

    def __init__(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg):
        # Ensure all inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.e1 = np.atleast_1d(e1)
        self.e2 = np.atleast_1d(e2)
        self.f1 = np.atleast_1d(f1)
        self.f2 = np.atleast_1d(f2)
        self.g1 = np.atleast_1d(g1)
        self.g2 = np.atleast_1d(g2)
        self.sigs = np.atleast_1d(sigs)
        self.sigf = np.atleast_1d(sigf)
        self.sigg = np.atleast_1d(sigg)

    def copy(self):
        """
        Creates a deep copy of the Source object.

        Returns:
            Source: Deep copy of the Source object.
        """
        return Source(
            x=self.x.copy(),
            y=self.y.copy(),
            e1=self.e1.copy(),
            e2=self.e2.copy(),
            f1=self.f1.copy(),
            f2=self.f2.copy(),
            g1=self.g1.copy(),
            g2=self.g2.copy(),
            sigs=self.sigs.copy(),
            sigf=self.sigf.copy(),
            sigg=self.sigg.copy()
        )

    def add(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg):
        """
        Adds a new source to the catalog.

        Parameters:
            x (float): x-position of the new source.
            y (float): y-position of the new source.
            e1 (float): First component of ellipticity (shear) of the new source.
            e2 (float): Second component of ellipticity (shear) of the new source.
            f1 (float): First component of flexion of the new source.
            f2 (float): Second component of flexion of the new source.
            g1 (float): First component of g-flexion of the new source.
            g2 (float): Second component of g-flexion of the new source.
            sigs (float): Standard deviation of the shear components.
            sigf (float): Standard deviation of the flexion components.
            sigg (float): Standard deviation of the g-flexion components.
        """
        # Append new source properties to existing arrays
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.e1 = np.append(self.e1, e1)
        self.e2 = np.append(self.e2, e2)
        self.f1 = np.append(self.f1, f1)
        self.f2 = np.append(self.f2, f2)
        self.g1 = np.append(self.g1, g1)
        self.g2 = np.append(self.g2, g2)
        self.sigs = np.append(self.sigs, sigs)
        self.sigf = np.append(self.sigf, sigf)
        self.sigg = np.append(self.sigg, sigg)

    def remove(self, indices):
        """
        Removes sources from the catalog based on given indices.

        Parameters:
            indices (array_like): Indices of the sources to remove.
        """
        for attr in [
            'x', 'y', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'sigs', 'sigf', 'sigg'
        ]:
            setattr(self, attr, np.delete(getattr(self, attr), indices))

    def zero_lensing_signals(self):
        """
        Resets all lensing signals (shear, flexion, g-flexion) to zero.
        """
        for attr in ['e1', 'e2', 'f1', 'f2', 'g1', 'g2']:
            setattr(self, attr, np.zeros_like(getattr(self, attr)))

    def filter_sources(self, max_flexion=0.1):
        """
        Filters out sources with flexion values exceeding the specified threshold.

        Parameters:
            max_flexion (float): Maximum allowed absolute flexion value. Default is 0.1.

        Returns:
            np.ndarray: Boolean array indicating the valid sources after filtering.
        """
        # Identify valid sources where both f1 and f2 are within the allowed flexion
        valid_indices = (np.abs(self.f1) <= max_flexion) & (np.abs(self.f2) <= max_flexion)
        
        # Identify bad indices where the condition is not met
        bad_indices = np.where(~valid_indices)[0]
        
        # Remove the bad indices using the class's remove method
        self.remove(bad_indices)

    def apply_noise(self):
        """
        Applies random noise to the lensing properties based on their standard deviations.
        """
        # Apply noise to shear components
        for attr, sigma in zip(['e1', 'e2'], [self.sigs, self.sigs]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)
        # Apply noise to flexion components
        for attr, sigma in zip(['f1', 'f2'], [self.sigf, self.sigf]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)
        # Apply noise to g-flexion components
        for attr, sigma in zip(['g1', 'g2'], [self.sigg, self.sigg]):
            noise = np.random.normal(0, sigma, size=getattr(self, attr).shape)
            setattr(self, attr, getattr(self, attr) + noise)

    def apply_lensing(self, lenses, lens_type='SIS', z_source=0.8):
        """
        Applies lensing effects to the sources using the specified lens model.

        Parameters:
            lenses: An object containing lens properties (e.g., positions, masses).
            lens_type (str): The type of lens model to use ('SIS' or 'NFW'). Default is 'SIS'.
            z_source (float): Redshift of the sources (used for NFW lensing). Default is 0.8.
        """
        if lens_type == 'SIS':
            shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_sis(
                lenses, self
            )
        elif lens_type == 'NFW':
            shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(
                lenses, self, z_source
            )
        else:
            raise ValueError("Invalid lens type. Use 'SIS' or 'NFW'.")

        # Update lensing properties by adding the calculated signals
        for attr, delta in zip(
            ['e1', 'e2', 'f1', 'f2', 'g1', 'g2'],
            [shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2]
        ):
            setattr(self, attr, getattr(self, attr) + delta)

    def export_to_csv(self, filename):
        """
        Exports the source catalog to a CSV file.

        Parameters:
            filename (str): Name of the file to export to.
        """
        import pandas as pd

        # Create a DataFrame from the Source object
        df = pd.DataFrame({
            'x': self.x,
            'y': self.y,
            'e1': self.e1,
            'e2': self.e2,
            'f1': self.f1,
            'f2': self.f2,
            'g1': self.g1,
            'g2': self.g2,
            'sigs': self.sigs,
            'sigf': self.sigf,
            'sigg': self.sigg
        })

        # Export the DataFrame to a CSV file
        df.to_csv(filename, index=False)
    
    def import_from_csv(self, filename):
        """
        Imports a source catalog from a CSV file.

        Parameters:
            filename (str): Name of the file to import from.
        """
        import pandas as pd

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)

        # Assign the DataFrame columns to the Source object attributes
        for attr in ['x', 'y', 'e1', 'e2', 'f1', 'f2', 'g1', 'g2', 'sigs', 'sigf', 'sigg']:
            setattr(self, attr, df[attr].values)