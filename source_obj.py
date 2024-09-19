import numpy as np
import utils

class Source:
    # Class to store source information. Each source has a position (x, y) 
    # and ellipticity (e1, e2), flexion (f1, f2), and g_flexion (g1, g2) signals
    # as well as the standard deviations of these signals (sigs, sigf, sigg)
    def __init__(self, x, y, e1, e2, f1, f2, g1, g2, sigs, sigf, sigg):
        # Make sure all inputs are numpy arrays
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


    def remove_sources(self, indices):
        # Remove sources from the source catalog based on indices
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.e1 = np.delete(self.e1, indices)
        self.e2 = np.delete(self.e2, indices)
        self.f1 = np.delete(self.f1, indices)
        self.f2 = np.delete(self.f2, indices)
        self.g1 = np.delete(self.g1, indices)
        self.g2 = np.delete(self.g2, indices)
        self.sigs = np.delete(self.sigs, indices)
        self.sigf = np.delete(self.sigf, indices)
        self.sigg = np.delete(self.sigg, indices)


    def zero_lensing_signals(self):
        # Set all lensing signals to zero
        self.e1 = np.zeros_like(self.e1)
        self.e2 = np.zeros_like(self.e2)
        self.f1 = np.zeros_like(self.f1)
        self.f2 = np.zeros_like(self.f2)
        self.g1 = np.zeros_like(self.g1)
        self.g2 = np.zeros_like(self.g2)


    def filter_sources(self, max_flexion=0.1):
        # Make cuts in the source data based on size and flexion
        valid_indices = (np.abs(self.f1) <= max_flexion) & (np.abs(self.f2) <= max_flexion)
        self.x, self.y = self.x[valid_indices], self.y[valid_indices]
        self.e1, self.e2 = self.e1[valid_indices], self.e2[valid_indices]
        self.f1, self.f2 = self.f1[valid_indices], self.f2[valid_indices]
        self.g1, self.g2 = self.g1[valid_indices], self.g2[valid_indices]
        self.sigs = self.sigs[valid_indices]
        self.sigf = self.sigf[valid_indices]
        self.sigg = self.sigg[valid_indices]
        return valid_indices


    def apply_noise(self):
        # Apply noise to the source - lensing properties
        self.e1 += np.random.normal(0, self.sigs)
        self.e2 += np.random.normal(0, self.sigs)
        self.f1 += np.random.normal(0, self.sigf)
        self.f2 += np.random.normal(0, self.sigf)
        self.g1 += np.random.normal(0, self.sigg)
        self.g2 += np.random.normal(0, self.sigg)


    def apply_SIS_lensing(self, lenses):
        """
        Apply the lensing effects to the source using the Singular Isothermal Sphere (SIS) model. 
        This model primarily utilizes the Einstein radii of each lens to determine its effect.

        Parameters:
        - lenses: An object containing the lens properties (x, y, Einstein radii 'te').
                Expected to be arrays but can handle single values.
        """
        
        shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_sis(lenses, self)
        self.e1 += shear_1
        self.e2 += shear_2
        self.f1 += flex_1
        self.f2 += flex_2
        self.g1 += gflex_1
        self.g2 += gflex_2


    def apply_NFW_lensing(self, lenses, z_source=0.8):
        """
        Apply the lensing effects to the source using the Navarro-Frenk-White (NFW) model.
        This model utilizes the mass, concentration, and redshift of each halo to determine its effect.

        Parameters:
        - halos: An object containing the halo properties (x, y, mass, concentration, redshift).
                Expected to be arrays but can handle single values.
        - z_source: The redshift of the source.
        """

        shear_1, shear_2, flex_1, flex_2, gflex_1, gflex_2 = utils.calculate_lensing_signals_nfw(lenses, self, z_source)
        self.e1 += shear_1
        self.e2 += shear_2
        self.f1 += flex_1
        self.f2 += flex_2
        self.g1 += gflex_1
        self.g2 += gflex_2