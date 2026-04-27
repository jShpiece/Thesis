import numpy as np
from astropy.cosmology import Planck15 as cosmo
from astropy import units as u

# Physical constants
M_SUN = 1.989e30  # Mass of the sun in kg
G = 6.67430e-11   # Gravitational constant in m^3 kg^-1 s^-2
C = 299_792_458   # Speed of light in m/s

class SIS_Lens:
    """
    Represents a Singular Isothermal Sphere (SIS) lens.

    Attributes:
        x (np.ndarray): x-positions of the lens(es).
        y (np.ndarray): y-positions of the lens(es).
        te (np.ndarray): Einstein radius (theta_E) of the lens(es).
        chi2 (np.ndarray): Chi-squared values associated with the lens(es).
    """

    def __init__(self, x, y, te, chi2):
        """
        Initializes the SIS_Lens object.

        Parameters:
            x (array_like): x-positions of the lens(es).
            y (array_like): y-positions of the lens(es).
            te (array_like): Einstein radius (theta_E) of the lens(es).
            chi2 (array_like): Chi-squared values associated with the lens(es).
        """
        # Ensure inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.te = np.atleast_1d(te)
        self.chi2 = np.atleast_1d(chi2)
    
    def copy(self):
        """
        Creates a deep copy of the SIS_Lens object.

        Returns:
            SIS_Lens: Deep copy of the SIS_Lens object.
        """
        return SIS_Lens(
            x=self.x.copy(),
            y=self.y.copy(),
            te=self.te.copy(),
            chi2=self.chi2.copy()
        )
    
    def merge(self, other):
        """
        Merges another SIS_Lens object into this one.

        Parameters:
            other (SIS_Lens): Another SIS_Lens object to merge into this one.
        """
        self.x = np.concatenate((self.x, other.x))
        self.y = np.concatenate((self.y, other.y))
        self.te = np.concatenate((self.te, other.te))
        self.chi2 = np.concatenate((self.chi2, other.chi2))

    def remove(self, indices):
        """
        Removes lenses at the specified indices.

        Parameters:
            indices (array_like): Indices of lenses to remove.
        """
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.te = np.delete(self.te, indices)
        self.chi2 = np.delete(self.chi2, indices)

    def export_to_csv(self, filename):
        """
        Exports the SIS_Lens object to a CSV file.

        Parameters:
            filename (str): Name of the CSV file to create.
        """
        data = np.vstack((self.x, self.y, self.te, self.chi2)).T
        np.savetxt(filename, data, delimiter=",", header="x,y,te,chi2")

    def import_from_csv(self, filename):
        """
        Imports data from a CSV file into the SIS_Lens object.

        Parameters:
            filename (str): Name of the CSV file to read.
        """
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.te = data[:, 2]
        self.chi2 = data[:, 3]

class NFW_Lens:
    """
    Represents a Navarro-Frenk-White (NFW) lens or halo.

    Attributes:
        x (np.ndarray): x-positions of the halo(s).
        y (np.ndarray): y-positions of the halo(s).
        z (np.ndarray): z-positions of the halo(s).
        concentration (np.ndarray): Concentration parameters of the halo(s).
        mass (np.ndarray): Masses of the halo(s) in solar masses.
        redshift (float): Redshift of the cluster (assumed the same for all halos).
        chi2 (np.ndarray): Chi-squared values associated with the halo(s).
    """

    def __init__(self, x, y, z, concentration, mass, redshift, chi2):
        """
        Initializes the NFW_Lens object.

        Parameters:
            x (array_like): x-positions of the halo(s).
            y (array_like): y-positions of the halo(s).
            z (array_like): z-positions of the halo(s).
            concentration (array_like): Concentration parameters of the halo(s).
            mass (array_like): Masses of the halo(s) in solar masses.
            redshift (float): Redshift of the cluster (assumed the same for all halos).
            chi2 (array_like): Chi-squared values associated with the halo(s).
        """
        # Ensure inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.z = np.atleast_1d(z)
        self.concentration = np.atleast_1d(concentration)
        self.mass = np.atleast_1d(mass)
        self.chi2 = np.atleast_1d(chi2)
        self.redshift = redshift

        # Ensure masses are positive
        self.mass = np.abs(self.mass)

    def copy(self):
        """
        Creates a deep copy of the NFW_Lens object.

        Returns:
            NFW_Lens: Deep copy of the NFW_Lens object.
        """
        return NFW_Lens(
            x=self.x.copy(),
            y=self.y.copy(),
            z=self.z.copy(),
            concentration=self.concentration.copy(),
            mass=self.mass.copy(),
            redshift=self.redshift,
            chi2=self.chi2.copy()
        )
    
    def merge(self, other):
        """
        Merges another NFW_Lens object into this one.

        Parameters:
            other (NFW_Lens): Another NFW_Lens object to merge into this one.
        """
        assert self.redshift == other.redshift, "Redshifts must match for merging."
        self.x = np.concatenate((self.x, other.x))
        self.y = np.concatenate((self.y, other.y))
        self.z = np.concatenate((self.z, other.z))
        self.concentration = np.concatenate((self.concentration, other.concentration))
        self.mass = np.concatenate((self.mass, other.mass))
        self.chi2 = np.concatenate((self.chi2, other.chi2))

    def remove(self, indices):
        """
        Removes halos at the specified indices.

        Parameters:
            indices (array_like): Indices of halos to remove.
        """
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.z = np.delete(self.z, indices)
        self.concentration = np.delete(self.concentration, indices)
        self.mass = np.delete(self.mass, indices)
        self.chi2 = np.delete(self.chi2, indices)
    
    def export_to_csv(self, filename):
        """
        Exports the NFW_Lens object to a CSV file.

        Parameters:
            filename (str): Name of the CSV file to create.
        """
        data = np.vstack((self.x, self.y, self.z, self.concentration, self.mass, self.chi2)).T
        np.savetxt(filename, data, delimiter=",", header="x,y,z,concentration,mass,chi2")

    def import_from_csv(self, filename):
        """
        Imports data from a CSV file into the NFW_Lens object.

        Parameters:
            filename (str): Name of the CSV file to read.
        """
        data = np.loadtxt(filename, delimiter=",", skiprows=1, ndmin=2)
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.z = data[:, 2]
        self.concentration = data[:, 3]
        self.mass = data[:, 4]
        self.chi2 = data[:, 5]

    def project_to_2D(self):
        """
        Projects 3D positions onto a 2D plane formed by the first two principal components.

        This method uses Principal Component Analysis (PCA) to project the halos from
        3D space onto a 2D plane, effectively reducing dimensionality while preserving
        as much variance as possible. After projection, the z-values are set to zero.
        """
        # Validate input dimensions
        if not (len(self.x) == len(self.y) == len(self.z)):
            raise ValueError("The x, y, and z arrays must have the same length.")
        if not (self.x.ndim == self.y.ndim == self.z.ndim == 1):
            raise ValueError("The x, y, and z arrays must be 1D.")
        if len(self.x) <= 1:
            raise ValueError("At least two points are required for projection.")

        # Stack coordinates into an (N, 3) array
        points = np.vstack((self.x, self.y, self.z)).T

        # Compute the covariance matrix and its eigenvalues and eigenvectors
        cov_matrix = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvectors by descending eigenvalues
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project points onto the plane of the first two principal components
        projected_points = points @ eigenvectors[:, :2]

        # Update positions and set z-values to zero
        self.x = projected_points[:, 0]
        self.y = projected_points[:, 1]
        self.z = np.zeros(len(self.x))

    def calc_R200(self):
        """
        Calculates the R200 radius for each halo.

        Returns:
            tuple:
                R200 (np.ndarray): R200 radii in meters.
                R200_arcsec (np.ndarray): R200 radii in arcseconds.
        """
        # Critical density at the cluster's redshift
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value

        # Mass in kg
        mass_kg = self.mass * M_SUN

        # Calculate R200 in meters
        R200 = ((3 * mass_kg) / (800 * np.pi * rho_c)) ** (1 / 3)

        # Convert R200 to arcseconds
        D_A = cosmo.angular_diameter_distance(self.redshift).to(u.m).value
        R200_arcsec = (R200 / D_A) * 206_265  # Radians to arcseconds

        return R200, R200_arcsec

    def calc_delta_c(self):
        """
        Calculates the characteristic density contrast (delta_c) for each halo.

        Returns:
            np.ndarray: Characteristic density contrast values.
        """
        c = self.concentration # Concentration parameter - rename for brevity
        delta_c = (200 / 3) * (c ** 3) / (np.log(1 + c) - c / (1 + c))
        return delta_c

    def calculate_concentration(self):
        """
        Calculates the concentration parameter for each halo based on mass and redshift.
        Uses result from Duffy et al. (2008) for the mass-concentration relation.
        Updates:
            self.concentration (np.ndarray): Updated concentration parameters.
        """
        mass_corrected = self.mass + 1e-10 # Avoid division by zero
        self.concentration = (5.71 * (mass_corrected / 2e12) ** (-0.084) * (1 + self.redshift) ** (-0.47))
        '''
        Or raganin et al
        A = 6.02
        B = -0.12
        C = 0.16
        self.concentration = (A * (mass_corrected / 1e13) ** B * ((1.47)/(1 + self.redshift)) ** (C))
        '''

    def check_for_nan_properties(self):
        """
        Checks for NaN values in the halo properties.

        Returns:
            bool: True if any property contains NaN values, False otherwise.
            Let's also check for inf values while we're at it.
        """
        return (np.isnan(self.x).any() or np.isnan(self.y).any() or np.isnan(self.z).any() or
                np.isnan(self.concentration).any() or np.isnan(self.mass).any() or
                np.isnan(self.chi2).any() or np.isinf(self.x).any() or np.isinf(self.y).any() or
                np.isinf(self.z).any() or np.isinf(self.concentration).any() or
                np.isinf(self.mass).any() or np.isinf(self.chi2).any())

class PowerLawHalo:
    """
    Represents a power-law convergence halo.

    Profile:
        kappa(theta) = kappa_star * (theta / theta_star) ** (-slope)

    valid for 0 < slope < 2.  At slope = 1 with
    kappa_star = theta_E / (2 * theta_star), the profile reduces to a
    singular isothermal sphere (SIS).

    Attributes:
        x (np.ndarray): x-positions of the halo(s) (arcsec).
        y (np.ndarray): y-positions of the halo(s) (arcsec).
        kappa_star (np.ndarray): Normalization of kappa at theta = theta_star
            (dimensionless).
        slope (np.ndarray): Power-law slope `n`, with 0 < n < 2.
        theta_star (float): Pivot radius (arcsec). Fixed globally; shared
            across all halos in a single PowerLawHalo collection (the
            (kappa_star, theta_star) pair is exactly degenerate, so
            theta_star is treated as a unit of length, not a free parameter).
        redshift (float): Redshift of the cluster (assumed the same for
            all halos), used for physical mass computations.
        chi2 (np.ndarray): Chi-squared values associated with the halo(s).
    """

    def __init__(self, x, y, kappa_star, slope, theta_star, redshift, chi2):
        """
        Initializes the PowerLawHalo object.

        Parameters:
            x (array_like): x-positions of the halo(s) (arcsec).
            y (array_like): y-positions of the halo(s) (arcsec).
            kappa_star (array_like): Normalization at the pivot radius.
            slope (array_like): Power-law slope `n`, must be in (0, 2).
            theta_star (float): Pivot radius (arcsec). Global, halo-shared.
            redshift (float): Redshift of the cluster.
            chi2 (array_like): Chi-squared values associated with the halo(s).
        """
        # Ensure inputs are numpy arrays
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.kappa_star = np.atleast_1d(kappa_star)
        self.slope = np.atleast_1d(slope)
        self.chi2 = np.atleast_1d(chi2)
        self.theta_star = float(theta_star)
        self.redshift = redshift

        # Ensure normalizations are positive (matches NFW_Lens mass convention)
        self.kappa_star = np.abs(self.kappa_star)

    def copy(self):
        """
        Creates a deep copy of the PowerLawHalo object.

        Returns:
            PowerLawHalo: Deep copy of the PowerLawHalo object.
        """
        return PowerLawHalo(
            x=self.x.copy(),
            y=self.y.copy(),
            kappa_star=self.kappa_star.copy(),
            slope=self.slope.copy(),
            theta_star=self.theta_star,
            redshift=self.redshift,
            chi2=self.chi2.copy()
        )

    def merge(self, other):
        """
        Merges another PowerLawHalo object into this one.

        Parameters:
            other (PowerLawHalo): Another PowerLawHalo object to merge into this one.
        """
        assert self.redshift == other.redshift, "Redshifts must match for merging."
        assert np.isclose(self.theta_star, other.theta_star), \
            "Pivot radii (theta_star) must match for merging."
        self.x = np.concatenate((self.x, other.x))
        self.y = np.concatenate((self.y, other.y))
        self.kappa_star = np.concatenate((self.kappa_star, other.kappa_star))
        self.slope = np.concatenate((self.slope, other.slope))
        self.chi2 = np.concatenate((self.chi2, other.chi2))

    def remove(self, indices):
        """
        Removes halos at the specified indices.

        Parameters:
            indices (array_like): Indices of halos to remove.
        """
        self.x = np.delete(self.x, indices)
        self.y = np.delete(self.y, indices)
        self.kappa_star = np.delete(self.kappa_star, indices)
        self.slope = np.delete(self.slope, indices)
        self.chi2 = np.delete(self.chi2, indices)

    def export_to_csv(self, filename):
        """
        Exports the PowerLawHalo object to a CSV file.

        Parameters:
            filename (str): Name of the CSV file to create.
        """
        data = np.vstack((self.x, self.y, self.kappa_star, self.slope, self.chi2)).T
        header = (f"theta_star={self.theta_star},redshift={self.redshift}\n"
                  f"x,y,kappa_star,slope,chi2")
        np.savetxt(filename, data, delimiter=",", header=header)

    def import_from_csv(self, filename):
        """
        Imports data from a CSV file into the PowerLawHalo object.

        The first commented header line is expected to contain
        `theta_star=...,redshift=...` and is parsed to set those
        global attributes.

        Parameters:
            filename (str): Name of the CSV file to read.
        """
        # Parse the metadata header
        with open(filename, "r") as f:
            first = f.readline().strip().lstrip("#").strip()
        for token in first.split(","):
            if "theta_star=" in token:
                self.theta_star = float(token.split("=")[1])
            if "redshift=" in token:
                self.redshift = float(token.split("=")[1])

        data = np.loadtxt(filename, delimiter=",", skiprows=2, ndmin=2)
        self.x = data[:, 0]
        self.y = data[:, 1]
        self.kappa_star = data[:, 2]
        self.slope = data[:, 3]
        self.chi2 = data[:, 4]

    def calc_theta_E(self):
        """
        Calculates the Einstein radius for each halo.

        Derived from alpha(theta_E) = theta_E with the closed-form deflection:
            theta_E = [2 * kappa_star / (2 - n)] ** (1/n) * theta_star

        Returns:
            np.ndarray: Einstein radii in arcseconds.
        """
        # Guard against the formal n=2 divergence
        denom = 2.0 - self.slope
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        coeff = 2.0 * self.kappa_star / denom
        # Avoid 0**(1/n) edge case
        coeff = np.where(coeff > 0, coeff, 1e-12)
        theta_E = coeff ** (1.0 / self.slope) * self.theta_star
        return theta_E

    def calc_mass_2d(self, theta_arcsec, z_source):
        """
        Calculates the 2D projected mass enclosed within angular radius
        theta_arcsec, for a source at redshift z_source.

        From Phase 0:
            M_2D(<theta) = [2*pi*Sigma_cr*D_l^2*kappa_star*theta_star^n / (2-n)] * theta^(2-n)

        Parameters:
            theta_arcsec (array_like): Angular radius (arcsec). Either a
                scalar (broadcast to all halos) or an array matching the
                number of halos.
            z_source (float): Source redshift, used to compute Sigma_cr.

        Returns:
            np.ndarray: Projected mass within theta in solar masses.
        """
        theta_arcsec = np.atleast_1d(theta_arcsec)
        n = self.slope

        # Cosmological distances
        D_l = cosmo.angular_diameter_distance(self.redshift).to(u.m).value
        D_s = cosmo.angular_diameter_distance(z_source).to(u.m).value
        D_ls = cosmo.angular_diameter_distance_z1z2(
            self.redshift, z_source).to(u.m).value

        # Critical surface density (kg / m^2)
        Sigma_cr = (C ** 2 / (4.0 * np.pi * G)) * (D_s / (D_l * D_ls))

        # Convert angular radii to physical (arcsec -> rad -> m at the lens)
        ARCSEC_TO_RAD = 1.0 / 206_265.0
        theta_phys = theta_arcsec * ARCSEC_TO_RAD * D_l            # meters
        theta_star_phys = self.theta_star * ARCSEC_TO_RAD * D_l    # meters

        # M_2D (kg) — note: integration in angular units gives the same
        # closed form, just multiplied by D_l^2 to convert area to physical.
        denom = 2.0 - n
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
        prefactor = (2.0 * np.pi * Sigma_cr * self.kappa_star
                     * theta_star_phys ** n / denom)
        M_2D_kg = prefactor * theta_phys ** denom

        # Return in solar masses for consistency with NFW_Lens.mass
        return M_2D_kg / M_SUN

    def set_default_slope(self, n0=1.2):
        """
        Sets the slope to a physically motivated default value.

        The default n0 = 1.2 corresponds to the local NFW 2D-convergence
        slope at the scale radius x = R/r_s = 1, providing a sensible
        initialization for halos with no prior slope information. This is
        the analog of `calculate_concentration` for the NFW class.

        Parameters:
            n0 (float): Default slope value. Must satisfy 0 < n0 < 2.
                Default: 1.2 (NFW slope at r_s).
        """
        if not (0.0 < n0 < 2.0):
            raise ValueError(f"Default slope n0 must be in (0, 2); got {n0}")
        self.slope = np.full_like(self.kappa_star, n0, dtype=float)

    def set_kappa_star_from_theta_E(self, theta_E):
        """
        Sets kappa_star from a known Einstein radius and the current slope.

        Inverts the closed-form theta_E expression:
            kappa_star = (2 - n) / 2 * (theta_E / theta_star) ** n

        Useful for seeding a power-law halo from an SIS-style estimate.

        Parameters:
            theta_E (array_like): Einstein radii (arcsec), one per halo.
        """
        theta_E = np.atleast_1d(theta_E)
        n = self.slope
        self.kappa_star = (2.0 - n) / 2.0 * (theta_E / self.theta_star) ** n

    def check_for_nan_properties(self):
        """
        Checks for NaN or Inf values in the halo properties.

        Returns:
            bool: True if any property contains NaN or Inf values, False otherwise.
        """
        return (np.isnan(self.x).any() or np.isnan(self.y).any() or
                np.isnan(self.kappa_star).any() or np.isnan(self.slope).any() or
                np.isnan(self.chi2).any() or
                np.isinf(self.x).any() or np.isinf(self.y).any() or
                np.isinf(self.kappa_star).any() or np.isinf(self.slope).any() or
                np.isinf(self.chi2).any())

    def check_physical_bounds(self):
        """
        Checks that slope values are within the physical range (0, 2).

        Returns:
            np.ndarray: Boolean array, True where halo slope is physical.
        """
        return (self.slope > 0.0) & (self.slope < 2.0)