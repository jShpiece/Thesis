import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u

# Constants
M_solar = 1.989e30 # kg
G = 6.67430e-11 # m^3 kg^-1 s^-2
c = 299792458 # m/s

class SIS_Lens:
    # Class to store lens information. Each lens has a position (x, y) and an Einstein radius (te)
    def __init__(self, x, y, te, chi2):
        # When initializing the Lens object, make sure all inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.te = np.atleast_1d(te)
        self.chi2 = np.atleast_1d(chi2)


class NFW_Lens:
    # Class to store halo information. Each halo has a position (x, y, z), concentration, mass, redshift, and chi^2 value


    def __init__(self, x, y, z, concentration, mass, redshift, chi2):
        # Initialize the halo object with the given parameters
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.z = np.atleast_1d(z)
        self.concentration = np.atleast_1d(concentration)
        self.mass = np.atleast_1d(mass)
        # Ensure the mass array is not empty
        if mass.size == 0:
            print(mass)
            raise ValueError('Mass cannot be empty')
        self.mass = np.atleast_1d(np.abs(mass)) # Masses must be positive
        self.redshift = redshift # Redshift of the cluster, assumed to be the same for all halos
        self.chi2 = np.atleast_1d(chi2)

    # --------------------------------------------
    # Halo Calculation Functions
    # --------------------------------------------

    def project_to_2D(self):
        """
        Projects a set of 3D points onto the plane formed by the first two principal eigenvectors.
        This will shift from our halos being in object 3D space to being in a projected 2D space.
        """

        # Sanity checks
        assert len(self.x) == len(self.y) == len(self.z), "The x, y, and z arrays must have the same length."
        assert self.x.ndim == self.y.ndim == self.z.ndim == 1, "The x, y, and z arrays must be 1D."
        assert len(self.x) > 1, "At least two points are required."

        # Combine the x, y, z coordinates into a single matrix
        points = np.vstack((self.x, self.y, self.z)).T

        # Calculate the covariance matrix
        cov_matrix = np.cov(points, rowvar=False)

        # Compute the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort the eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project the points onto the plane formed by the first two principal eigenvectors
        projected_points = np.dot(points, eigenvectors[:, :2])

        x = projected_points[:, 0]
        y = projected_points[:, 1]
        # Make sure these are arrays
        if np.isscalar(x):
            x = np.array([x])
        if np.isscalar(y):
            y = np.array([y])

        self.x, self.y = x, y
        self.z = np.zeros(len(self.x)) # Set the z values to zero now that we are in 2D


    def calc_R200(self):
        # Compute the R200 radius for each halo
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value
        mass = np.abs(self.mass) * M_solar # Convert to kg - mass is allowed to be negative, but it should be positive for this calculation
        R200 = ((3 / (800 * np.pi)) * (mass / rho_c))**(1/3) # In meters
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        return (200/3) * (self.concentration**3) / (np.log(1 + self.concentration) - (self.concentration / (1 + self.concentration)))


    def calculate_concentration(self):
        # Compute the concentration parameter for each halo
        # This is done with the Duffy et al. (2008) relation
        # This relation is valid for 0 < z < 2 - this covers the range of redshifts we are interested in
        self.mass += 1e-10 # Add a small value to the mass to avoid division by zero
        self.concentration = 5.71 * (np.abs(self.mass) / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47) 