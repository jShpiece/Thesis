import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import hist as fancy_hist
import pipeline
import utils
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
import scipy.ndimage
from multiprocessing import Pool
import time
import os
import scipy.optimize as opt
import copy

dir = 'MDARK/'
column_names = ['MainHaloID', ' Total Mass', ' Redshift', 'Halo Number', ' Mass Fraction', ' Characteristic Size'] # Column names for the key files
# Important Notes!
# Total Mass is in units of M_sun/h
# Characteristic Size is in units of arcseconds
plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots

# Physical constants
c = 3 * 10**8 # Speed of light in m/s
G = 6.674 * 10**-11 # Gravitational constant in m^3/kg/s^2
h = 0.677 # Hubble constant
M_solar = 1.989 * 10**30 # Solar mass in kg
z_source = 0.8 # Redshift of the source galaxies


class Halo:
    def __init__(self, x, y, z, concentration, mass, redshift, chi2):
        # Initialize the halo object with the given parameters
        self.x = np.atleast_1d(x)
        self.y = np.atleast_1d(y)
        self.z = np.atleast_1d(z)
        self.concentration = np.atleast_1d(concentration)
        self.mass = np.atleast_1d(mass)
        # Ensure the mass array is not empty
        if mass.size == 0:
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
        R200 = (3 / (800 * np.pi) * (np.abs(self.mass) * M_solar / rho_c))**(1/3) # In meters
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        return (200/3) * (self.concentration**3) / (np.log(1 + self.concentration) - self.concentration / (1 + self.concentration))


    def calc_corresponding_einstein_radius(self, source_redshift):
        # Compute the Einstein radius for a given source redshift
        Dl, Ds, Dls = utils.angular_diameter_distances(self.redshift, source_redshift)
        eR = np.sqrt((4 * G * self.mass * M_solar) / (c**2) * (Dls / (Ds * Dl))) * 206265 # Convert to arcseconds
        # Return the Einstein radius - make sure its an array
        if np.isscalar(eR):
            eR = np.array([eR])
        return eR


    def calculate_concentration(self):
        # Compute the concentration parameter for each halo
        # This is done with the Duffy et al. (2008) relation
        # This relation is valid for 0 < z < 2
        # Note - numpy doesn't like negative powers on lists, even if the answer isn't complex
        # Get around this by taking taking the absolute value of arrays (then multiplying by -1 if necessary) (actually don't do this - mass and concentration should be positive)
        self.concentration = 5.71 * (np.abs(self.mass) / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47) 
        # self.concentration = 5.71 * (self.mass / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47)

    # --------------------------------------------
    # Pipeline Functions
    # --------------------------------------------

    def update_chi2_values(self, sources, use_flags):
        # Given a set of sources, update the chi^2 values for each lens
        global_chi2 = pipeline.calculate_chi_squared(sources, self, use_flags, lensing='NFW')
        chi2_values = np.zeros(len(self.x))
        if len(self.x) == 1:
            # Only one lens - calculate the chi2 value for this lens
            chi2_values[0] = assign_lens_chi2_values(self, sources, use_flags)
        else:
            for i in range(len(self.x)):
                # Only pass in the i-th lens
                one_halo = Halo(self.x[i], self.y[i], self.z[i], self.concentration[i], self.mass[i], self.redshift, [0])
                chi2_values[i] = assign_lens_chi2_values(one_halo, sources, use_flags)
        self.chi2 = chi2_values
        dof = pipeline.calc_degrees_of_freedom(sources, self, use_flags)
        if dof == 0:
            print('Degrees of freedom is zero')
            return global_chi2
        return global_chi2 / dof


    def optimize_lens_positions(self, sources, use_flags):
        # Given a set of initial guesses for lens positions, find the optimal lens positions
        # via local minimization
        max_attempts = 1
        for i in range(len(self.x)):
            one_source = pipeline.Source(sources.x[i], sources.y[i], 
                                sources.e1[i], sources.e2[i], 
                                sources.f1[i], sources.f2[i], 
                                sources.g1[i], sources.g2[i],
                                sources.sigs[i], sources.sigf[i], sources.sigg[i])
            guess = [self.x[i], self.y[i], max(0, self.mass[i])] # Class is already initialized with initial guesses
            best_result = None
            best_params = guess
            for _ in range(max_attempts):
                
                result = opt.minimize(
                    chi2wrapper3, guess, args=([one_source, use_flags, self.concentration[i], self.redshift]), 
                    method='Nelder-Mead', 
                    tol=1e-8, 
                    options={'maxiter': 1000}
                )
                '''
                # Lets try a BFGS method, to avoid getting stuck in local minima
                result = opt.minimize(
                    chi2wrapper3, guess, args=([one_source, use_flags, self.concentration[i], self.redshift]), 
                    method='BFGS', 
                    tol=1e-8, 
                    options={'maxiter': 1000}
                )
                '''
                if best_result is None or result.fun < best_result.fun:
                    best_result = result
                    best_params = result.x

            self.x[i], self.y[i], self.mass[i] = best_params[0], best_params[1], best_params[2]
            # Now update the concentrations
            self.calculate_concentration()


    def filter_lens_positions(self, sources, xmax, threshold_distance=0.1):
        # Filter out halos that are too close to sources or too far from the center

        # Compute the distance of each lens from each source
        distances_to_sources = np.sqrt((self.x[:, None] - sources.x)**2 + (self.y[:, None] - sources.y)**2)
        # Identify lenses that are too close to sources
        too_close_to_sources = np.any(distances_to_sources < threshold_distance, axis=1)
        # Identify lenses that are too far from the center
        too_far_from_center = np.sqrt(self.x**2 + self.y**2) > 1 * xmax
        # Identify lenses with zero mass (or too small to be considered lenses)
        zero_mass_lenses = np.abs(self.mass) < 10**10

        # Remove lenses that are too close to sources, too far from the center, or have zero mass
        valid_indices = ~too_close_to_sources & ~too_far_from_center & ~zero_mass_lenses
        # Update the lens positions
        self.x, self.y = self.x[valid_indices], self.y[valid_indices]
        self.mass = self.mass[valid_indices]
        self.chi2 = self.chi2[valid_indices]
        self.concentration = self.concentration[valid_indices]


    def merge_close_lenses(self, merger_threshold=5):
        def perform_merger(i, j):
            # Given two lenses, merge them and place the new lens at the weighted average position
            # and with the average Einstein radius of the pair
            weight_i, weight_j = np.abs(self.mass[i]), np.abs(self.mass[j]) # Weights must be positive
            self.x[i] = (self.x[i]*weight_i + self.x[j]*weight_j) / (weight_i + weight_j)
            self.y[i] = (self.y[i]*weight_i + self.y[j]*weight_j) / (weight_i + weight_j)
            self.mass[i] = (weight_i + weight_j) / 2
            self.x, self.y, self.mass, self.chi2 = np.delete(self.x, j), np.delete(self.y, j), np.delete(self.mass, j), np.delete(self.chi2, j)
            self.concentration = np.delete(self.concentration, j)

        #Merge lenses that are too close to each other
        i = 0
        while i < len(self.x):
            for j in range(i+1, len(self.x)):
                # Check every pair of lenses
                # If they are too close, merge them
                distance = np.sqrt((self.x[i] - self.x[j])**2 + (self.y[i] - self.y[j])**2)
                if distance < merger_threshold:
                    perform_merger(i, j)
                    break
            else:
                i += 1


    def select_lowest_chi2(self, lens_floor=1):
        # Function that enables the iterative elimination of lenses
        # Select the 'lens_floor' lenses with the lowest chi^2 values
        
        # Sort the lenses by chi^2 value
        sorted_indices = np.argsort(self.chi2)
        self.x, self.y, self.mass, self.chi2 = self.x[sorted_indices], self.y[sorted_indices], self.mass[sorted_indices], self.chi2[sorted_indices]

        # Select the 'lens_floor' lenses with the lowest chi^2 values
        if len(self.x) > lens_floor:
            self.x, self.y, self.z, self.mass, self.concentration, self.chi2 = self.x[:lens_floor], self.y[:lens_floor], self.z[:lens_floor], self.mass[:lens_floor], self.concentration[:lens_floor], self.chi2[:lens_floor]


    def iterative_elimination(self, sources, reducedchi2, use_flags):
        # Iteratively eliminate lenses that do not improve the chi^2 value
        # lens_floors = np.arange(1, len(self.x) + 1)
        lens_floors = np.arange(1, len(self.x) + 1)
        best_dist = np.abs(reducedchi2 - 1)
        best_lenses = self
        for lens_floor in lens_floors:
            # Clone the lenses object
            test_lenses = Halo(self.x, self.y, self.z, self.concentration, self.mass, self.redshift, self.chi2)
            test_lenses.select_lowest_chi2(lens_floor=lens_floor)
            reducedchi2 = test_lenses.update_chi2_values(sources, use_flags)
            new_dist = np.abs(reducedchi2 - 1)
            if new_dist < best_dist:
                best_dist = new_dist
                best_lenses = test_lenses
        # Update the lenses object with the best set of lenses
        self.x, self.y, self.mass, self.chi2 = best_lenses.x, best_lenses.y, best_lenses.mass, best_lenses.chi2
        self.calculate_concentration()


    def full_minimization(self, sources, use_flags):
        guess = self.mass
        params = [self.x, self.y, self.redshift, self.concentration, sources, use_flags]
        max_attempts = 5  # Number of optimization attempts with different initial guesses
        best_result = None
        best_params = guess

        for _ in range(max_attempts):
            result = opt.minimize(
                chi2wrapper4, guess, args=params,
                method='Powell',  
                tol=1e-8,  
                options={'maxiter': 1000}
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_params = result.x
        self.mass = best_params

# --------------------------------------------
# Pipeline Functions
# --------------------------------------------

def chi2wrapper3(guess, params):
    # Wrapper function for chi2 to allow for minimization for a single lens object
    # Guess = [x, y, mass]
    # Params = [source, use_flags, concentration, redshift]
    lenses = Halo(guess[0], guess[1], np.zeros_like(guess[0]), params[2], guess[2], params[3], [0])
    return assign_lens_chi2_values(lenses, params[0], params[1])


def chi2wrapper4(guess, params):
    # Wrapper function for chi2 to allow constrained minimization - 
    #    only the masses are allowed to vary
    # This time, the lens object contains the full set of lenses
    # Guess = [mass1, mass2, ...]
    # Params = [x, y, redshift, concentration, sources, use_flags]
    lenses = Halo(params[0], params[1], np.zeros_like(params[0]), params[3], guess, params[2], [0])
    dof = pipeline.calc_degrees_of_freedom(params[4], lenses, params[5])
    # Return the reduced chi^2 value - 1, to be minimized
    return np.abs(pipeline.calculate_chi_squared(params[4], lenses, params[5], lensing='NFW') / dof - 1)


def assign_lens_chi2_values(lenses, sources, use_flags):
    # Given a single lens object, assign a weighted chi2 value based on source distances
    xl = lenses.x
    yl = lenses.y
    assert len(xl) == len(yl), "The x and y arrays must have the same length."
    assert xl.ndim == yl.ndim == 1, "The x and y arrays must be 1D."
    assert len(xl) == 1, "Only one lens is allowed."

    xs = sources.x
    ys = sources.y

    r = np.sqrt((xl[:, None] - xs)**2 + (yl[:, None] - ys)**2)
    # Choose a characteristic distance such that 1/5 of the sources are within this distance
    # Calculate distances from the current lens to all sources
    distances = np.sqrt((xl - xs)**2 + (yl - ys)**2)
    
    # Sort distances
    sorted_distances = np.sort(distances)
    
    # Determine the index for the desired fraction
    index = int(len(sorted_distances) * (0.4))
    
    # Set r0 as the distance at the calculated index
    r0 = sorted_distances[index]

    weights = np.exp(-r**2 / r0**2)
    # Normalize the weights
    weights /= np.sum(weights, axis=1)[:, None]
    assert np.allclose(np.sum(weights, axis=1), 1), "Weights must sum to 1."
    assert weights.shape == (len(xl), len(xs)), "Weights must have shape (len(xl), len(xs))."

    # Unpack flags for clarity
    use_shear, use_flexion, use_g_flexion = use_flags

    # Initialize a clone of sources with zeroed lensing signals
    source_clone = pipeline.Source(
        x=sources.x, y=sources.y,
        e1=np.zeros_like(sources.e1), e2=np.zeros_like(sources.e2),
        f1=np.zeros_like(sources.f1), f2=np.zeros_like(sources.f2),
        g1=np.zeros_like(sources.g1), g2=np.zeros_like(sources.g2),
        sigs=sources.sigs, sigf=sources.sigf, sigg=sources.sigg
    )

    source_clone.apply_NFW_lensing(lenses)

    # Weigh the chi^2 values by the distance of each source from each lens
    total_chi_squared = 0
    for i in range(len(source_clone.x)):
        # Get the chi2 values for each source, weighted by the source-lens distance
        shear_chi2 = ((sources.e1[i] - source_clone.e1[i]) / sources.sigs[i])**2 + ((sources.e2[i] - source_clone.e2[i]) / sources.sigs[i])**2
        flexion_chi2 = ((sources.f1[i] - source_clone.f1[i]) / sources.sigf[i])**2 + ((sources.f2[i] - source_clone.f2[i]) / sources.sigf[i])**2
        g_flexion_chi2 = ((sources.g1[i] - source_clone.g1[i]) / sources.sigg[i])**2 + ((sources.g2[i] - source_clone.g2[i]) / sources.sigg[i])**2
        total_chi_squared += weights[:, i].dot(shear_chi2*use_shear + flexion_chi2*use_flexion + g_flexion_chi2*use_g_flexion)
    '''
    print('e1', sources.e1, source_clone.e1)
    print('e2', sources.e2, source_clone.e2)
    print('f1', sources.f1, source_clone.f1)
    print('f2', sources.f2, source_clone.f2)
    print('g1', sources.g1, source_clone.g1)
    print('g2', sources.g2, source_clone.g2)
    print(shear_chi2, flexion_chi2, g_flexion_chi2, total_chi_squared[0])
    '''
    return total_chi_squared[0]

# --------------------------------------------
# File Management Functions
# --------------------------------------------

def chunk_data(file):
    # Define the data types for each column to improve performance
    data_types = {f'column_{i}': 'float' for i in range(6)}
    # Read the CSV file in chunks
    chunk_size = 50000  # Adjust based on your memory constraints
    chunks = pd.read_csv(file, dtype=data_types, chunksize=chunk_size)
    return chunks


def GOF_file_reader(IDs):
    # Given a list of IDs, read in the goodness of fit data
    # and return the chi2 values and mass values
    chi2_values = []
    mass_values = []

    for ID in IDs:
        file = 'Data/MDARK_Test/Goodness_of_Fit/Goodness_of_Fit_{}.csv'.format(ID)
        data = pd.read_csv(file)
        chi2_values.append(data[[' chi2_all_signals', ' chi2_gamma_F', ' chi2_F_G', ' chi2_gamma_G']].values)
        mass_values.append(data[[' mass_all_signals', ' mass_gamma_F', ' mass_F_G', ' mass_gamma_G']].values)

    return chi2_values, mass_values


# --------------------------------------------
# Plotting Functions
# --------------------------------------------

def _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, title, ax=None, legend=True):
        """Private helper function to plot the results of lensing reconstruction."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
        for i, mass in enumerate(lenses.mass):
            # Put mass in scientific notation
            ax.annotate('{:.2e}'.format(mass), (lenses.x[i], lenses.y[i]))
        ax.scatter(sources.x, sources.y, marker='.', color='blue', alpha=0.5, label='Sources')
        if true_lenses is not None:
            log_masses = np.log10(true_lenses.mass)
            size_scale = (log_masses - min(log_masses)) / (max(log_masses) - min(log_masses)) * 100
            ax.scatter(true_lenses.x, true_lenses.y, marker='o', alpha=0.5, color='green', label='True Lenses', s=size_scale)
        if legend:
            ax.legend(loc='upper right')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if xmax is not None:
            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-xmax, xmax)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))


def plot_cluster_properties(z):
    # For a key file, read in every cluster and plot the distribution of 
    # mass, size, number of halos and substructure mass fraction
    file = dir + 'fixed_key_{}.MDARK'.format(z) 
    chunks = chunk_data(file)
    print('Reading data...')

    Mass = []
    size = []
    N_halo = []
    M_frac = []
    for chunk in chunks:
        Mass.append(chunk[' Total Mass'].values)
        size.append(chunk[' Characteristic Size'].values)
        N_halo.append(chunk[' Halo Number'].values)
        M_frac.append(chunk[' Mass Fraction'].values)
    
    Mass = np.concatenate(Mass)
    size = np.concatenate(size)
    N_halo = np.concatenate(N_halo)
    M_frac = np.concatenate(M_frac)
    # Plot the mass distribution (log scale)
    # Use 1000 bins
    # Remove nan values (these should mostly be from m_frac, where we had 0/0 = nan)
    Mass = Mass[~np.isnan(Mass)]
    size = size[~np.isnan(size)]
    N_halo = N_halo[~np.isnan(N_halo)]
    M_frac = M_frac[~np.isnan(M_frac)]
    
    fig, ax = plt.subplots()
    fancy_hist(Mass, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\rm dark}$ [$M_{\odot}$]')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig('Images/mass_dist_{}.png'.format(z))

    fig, ax = plt.subplots()
    fancy_hist(size, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R_{\rm dark}$ [Mpc]')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig('Images/size_dist_{}.png'.format(z))

    fig, ax = plt.subplots()
    fancy_hist(N_halo, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel(r'$N_{\rm dark}$')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig('Images/N_halo_dist_{}.png'.format(z))

    fig, ax = plt.subplots()
    fancy_hist(M_frac, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_yscale('log')
    ax.set_xlabel(r'$f_{\rm dark}$')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig('Images/M_frac_dist_{}.png'.format(z))


def build_mass_correlation_plot(file_name, plot_name):
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    # Get the mass and true mass
    true_mass = results[' True Mass'].values
    mass = results[' Mass_all_signals'].values
    mass_gamma_f = results[' Mass_gamma_F'].values
    mass_f_g = results[' Mass_F_G'].values
    mass_gamma_g = results[' Mass_gamma_G'].values
    masses = [mass, mass_gamma_f, mass_f_g, mass_gamma_g]
    # True mass is in units of M_sun / h - convert others to the same units
    masses = [mass / h for mass in masses]
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    '''
    # Remove outliers
    outliers = (mass > 1e15) | (mass < 1e12)
    mass = mass[~outliers]
    true_mass = true_mass[~outliers]
    '''
    # Plot the results for each signal combination
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(4):
        # true_mass_temp = true_mass[masses[i] > 0]
        # masses[i] = masses[i][masses[i] > 0]
        true_mass_temp = true_mass
        masses[i] = np.abs(masses[i])
        if masses[i].min() == 0:
            true_mass_temp = true_mass_temp[masses[i] > 0]
            masses[i] = masses[i][masses[i] > 0]

        ax[i].scatter(true_mass_temp, masses[i], s=10, color='black')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        # Add a line of best fit
        x = np.linspace(1e13, 1e15, 100)
        try:
            # Remove any mass values that are zero
            # true_mass = true_mass[masses[i] > 0]
            # masses[i] = masses[i][masses[i] > 0]
            m, b = np.polyfit(np.log10(true_mass_temp), np.log10(masses[i]), 1)
            ax[i].plot(x, 10**(m*np.log10(x) + b), color='red', label='Best Fit: m = {:.2f}'.format(m))
        except:
            print('RuntimeWarning: Skipping line of best fit')
            continue
        # Plot the line of best fit and an agreement line
        ax[i].plot(x, x, color='blue', label='Agreement Line', linestyle='--') # Agreement line - use a different linestyle because the paper won't be in color
        ax[i].legend()
        ax[i].set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
        ax[i].set_ylabel(r'$M_{\rm inferred}$ [$M_{\odot}$]')
        ax[i].set_title('Signal Combination: {} \n Correlation Coefficient: {:.2f}'.format(signals[i], np.corrcoef(true_mass_temp, masses[i])[0, 1]))

    fig.tight_layout()
    fig.savefig(plot_name)
    plt.show()


def build_mass_correlation_plot_errors(ID_file, results_file, plot_name):
    IDs = []
    true_mass = []
    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            mass = line.split(',')[1]
            IDs.append(int(ID))
            true_mass.append(float(mass))

    IDs = np.array(IDs)
    mass_val = np.empty((len(IDs), 4))
    mass_err = np.empty((len(IDs), 4))

    # Now read in the results file
    results = pd.read_csv(results_file)
    # There are 4 columns for mass
    mass_columns = [' Mass_all_signals', ' Mass_gamma_F', ' Mass_F_G', ' Mass_gamma_G']
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']
    # Each ID will appear multiple times, depending on the number of trials run
    # For each ID, collect all of the mass values and calculate the mean and standard deviation
    for i in range(len(IDs)):
        mass_values = []
        for j in range(4):
            mass_values.append(results[results['ID'] == int(IDs[i])][mass_columns[j]].values)
        mass_values = (np.abs(np.array(mass_values)) ** (3/2)) * 10**-6
        mass_val[i] = np.mean(mass_values, axis=1)
        mass_err[i] = np.std(mass_values, axis=1)

    # Plot the results for each signal combination
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    for i in range(4):
        mass = mass_val[:, i]
        err = mass_err[:, i]

        # Remove NaN values - these occur when we only run a subset of the trials
        bad_arrays = np.isnan(mass) | np.isnan(err)
        mass = mass[~bad_arrays]
        err = err[~bad_arrays]
        true_mass = true_mass[:len(mass)]

        ax[i].errorbar(true_mass, mass, yerr=err, fmt='o', color='black')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        # Add a line of best fit
        x = np.linspace(1e13, 1e15, 100)
        try:
            m, b = np.polyfit(np.log10(true_mass), np.log10(mass), 1)
            ax[i].plot(x, 10**(m*np.log10(x) + b), color='red', label='Best Fit: m = {:.2f}'.format(m))
        except:
            print('RuntimeWarning: Skipping line of best fit')
            continue
        # Plot the line of best fit and an agreement line
        ax[i].plot(x, x, color='blue', label='Agreement Line', linestyle='--') # Agreement line - use a different linestyle because the paper won't be in color
        ax[i].legend()
        ax[i].set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
        ax[i].set_ylabel(r'$M_{\rm inferred}$ [$M_{\odot}$]')
        ax[i].set_title('Signal Combination: {} \n Correlation Coefficient: {:.2f}'.format(signals[i], np.corrcoef(true_mass, mass)[0, 1]))

    fig.suptitle('Mass Correlation')
    fig.tight_layout()
    fig.savefig(plot_name)
    # plt.show()


def build_chi2_plot(file_name, ID_file, test_number):
    IDs = []
    true_mass = []
    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            mass = line.split(',')[1]
            IDs.append(int(ID))
            true_mass.append(float(mass))
    
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    trial_IDs = results['ID'].values
    # Get the mass values
    mass_all_signals = results[' Mass_all_signals'].values
    mass_gamma_f = results[' Mass_gamma_F'].values
    mass_f_g = results[' Mass_F_G'].values
    mass_gamma_g = results[' Mass_gamma_G'].values
    mass_values = [mass_all_signals, mass_gamma_f, mass_f_g, mass_gamma_g]
    # Get the chi2 values
    chi2_all_signals = results[' Chi2_all_signals'].values
    chi2_gamma_f = results[' Chi2_gamma_F'].values
    chi2_f_g = results[' Chi2_F_G'].values
    chi2_gamma_g = results[' Chi2_gamma_G'].values
    chi2_values = [chi2_all_signals, chi2_gamma_f, chi2_f_g, chi2_gamma_g]
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    # Now get the true mass for each cluster
    true_mass_trials = []
    for ID in trial_IDs:
        # Get the mass where ID matches the ID in the ID file
        true_mass_trials.append(true_mass[IDs.index(ID)])

    # Plot the results for each signal combination
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(4):
        ax[i].scatter(true_mass_trials, chi2_values[i], s=10, color='black')
        ax[i].set_xscale('log')
        #ax[i].set_yscale('log')
        ax[i].set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
        ax[i].set_ylabel(r'$\chi^2$')
        ax[i].set_title('Signal Combination: {}'.format(signals[i]))
    
    fig.tight_layout()
    fig.savefig('Images/chi2_plot_{}.png'.format(test_number))
    plt.show()

# --------------------------------------------
# MDARK Processing Functions
# --------------------------------------------

def count_clusters(z):
    # Count the number of clusters with within a given mass range
    file = dir + 'fixed_key_{}.MDARK'.format(z)
    chunks = chunk_data(file)

    count1 = 0
    count2 = 0
    count3 = 0

    for chunk in chunks:
        # Count each cluster with mass > 10^13 M_sun
        count1 += np.sum(chunk[' Total Mass'].values > 1e13)
        # Count each cluster of mass > 10^14 M_sun 
        count2 += np.sum((chunk[' Total Mass'].values > 1e14))
        # Count each cluster of mass > 10^15 M_sun 
        count2 += np.sum((chunk[' Total Mass'].values > 1e15))

    print('Number of clusters with mass > 10^13 M_sun: {}'.format(count1))
    print('Number of clusters with mass > 10^14 M_sun: {}'.format(count2))
    print('Number of clusters with mass > 10^15 M_sun: {}'.format(count3))


def find_halos(ids, z):
    # Read a large data file and filter rows based on multiple IDs, 
    # creating a separate Halo object for each ID
    
    file_path = f'{dir}Halos_{z}.MDARK'  # Construct the file path
    cols_to_use = ['MainHaloID', 'x', 'y', 'z', 'concentration_NFW', 'HaloMass', 'GalaxyType']
    
    # Dictionary to hold accumulated data for each ID
    data_accumulator = {id: [] for id in ids}
    
    # Read the file in chunks
    iterator = pd.read_csv(file_path, chunksize=10000, usecols=cols_to_use)
    for chunk in iterator:
        filtered_chunk = chunk[chunk['MainHaloID'].isin(ids) & (chunk['GalaxyType'] != 2)]
        for id in ids:
            # Accumulate data for each ID
            data_accumulator[id].append(filtered_chunk[filtered_chunk['MainHaloID'] == id])
    
    # Dictionary to hold Halo objects, keyed by ID
    halos = {}
    
    # Create Halo objects after all chunks have been processed
    for id, data_list in data_accumulator.items():
        if data_list:
            complete_data = pd.concat(data_list)
            xhalo = complete_data['x'].to_numpy()
            yhalo = complete_data['y'].to_numpy()
            zhalo = complete_data['z'].to_numpy()
            chalo = complete_data['concentration_NFW'].to_numpy()
            masshalo = complete_data['HaloMass'].to_numpy()
            
            # Create and store the Halo object
            halos[id] = Halo(xhalo, yhalo, zhalo, chalo, masshalo, z, np.zeros_like(xhalo))
        else:
            print(f'No data found for ID {id}')

    return halos


def choose_ID(z, mass_range, substructure_range):
    # Given a set of criteria, choose a cluster ID
    file = 'MDARK/fixed_key_{}.MDARK'.format(z)
    chunks = chunk_data(file)

    # Choose a random cluster with mass in the given range
    # and more than 1 halo
    rows = []
    for chunk in chunks:
        IDs = chunk['MainHaloID'].values
        masses = chunk[' Total Mass'].values
        substructure = chunk[' Mass Fraction'].values

        # Apply the criteria
        mass_criteria = (masses > mass_range[0]) & (masses < mass_range[1])
        substructure_criteria = (substructure > substructure_range[0]) & (substructure < substructure_range[1])

        # Find all clusters that satisfy all criteria
        combined_criteria = mass_criteria & substructure_criteria

        if np.any(combined_criteria):
            valid_ids = IDs[combined_criteria]
            # Add these to the list of rows
            rows.append(chunk[chunk['MainHaloID'].isin(valid_ids)])
    # Choose a random cluster from the list of valid rows
    if len(rows) > 0:
        rows = pd.concat(rows) # Concatenate the rows
        row = rows.sample(n=1) # Choose a random row
        return row
    else:
        return None


# --------------------------------------------
# Testing Functions
# --------------------------------------------

def build_lensing_field(halos, z, Nsource = None):
    '''
    Given a set of halos, run the analysis
    This involves the following steps

    1. Convert the halo coordinates to a 2D projection
    2. Convert the coordinates to arcseconds
    3. Generate a set of background galaxies
    4. Center the lenses at (0, 0)
    5. Set the maximum extent of the field of view
    6. Return the lenses and sources
    '''

    # Convert the halo coordinates to a 2D projection
    halos.project_to_2D()
    d = cosmo.angular_diameter_distance(z).to(u.meter).value
    halos.x *= (3.086 * 10**22 / d) * 206265
    halos.y *= (3.086 * 10**22 / d) * 206265

    lenses = halos # Placeholder for the lenses

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # Let the centroid be the location of the most massive halo
    # This will be where we expect to see the most ligwht, which
    # means it will be where observations are centered

    largest_halo = np.argmax(halos.mass)
    centroid = [halos.x[largest_halo], halos.y[largest_halo]]
    lenses.x -= centroid[0] 
    lenses.y -= centroid[1] 

    xmax = np.max((lenses.x**2 + lenses.y**2)**0.5)
    
    # Don't allow the field of view to be larger than 2 arcminutes - or smaller than 1 arcminute
    xmax = np.min([xmax, 3*60])
    xmax = np.max([xmax, 1*60])

    # Set the maximum extent of the field of view
    # to be the maximum extent of the lenses

    # Generate a set of background galaxies
    ns = 0.01
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    sources = utils.createSources(lenses, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')

    return lenses, sources, xmax


def build_test_set(Nclusters, z, file_name):
    # Select a number of clusters, spaced evenly in log space across the mass range
    # This set will be used to test the pipeline
    # For each cluster, we will save the following information
    # ID, Mass, Halo Number, Mass Fraction, Size

    # Establish criteria for cluster selection
    M_min = 1e13
    M_max = 1e15
    substructure_min = 0
    substructure_max = 0.1
    mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), Nclusters+1)

    rows = []
    for i in range(Nclusters):
        mass_range = [mass_bins[i], mass_bins[i+1]]
        substructure_range = [substructure_min, substructure_max]

        row = choose_ID(z, mass_range, substructure_range)
        if row is not None:
            print('Found a cluster in mass bin {}'.format(i))
            rows.append(row)
        else:
            print('No cluster found in mass bin {}'.format(i))
    
    # Let's make sure that the cluster properties are correct
    IDs = [row['MainHaloID'].values[i] for row in rows for i in range(len(row))]
    halos = find_halos(IDs, z)
    mass = []
    halo_number = []
    mass_fraction = []

    for i in range(len(rows)):
        halo = halos[IDs[i]]
        mass.append(halo.mass.sum())
        halo_number.append(len(halo.mass))
        mass_fraction.append(1 - np.max(halo.mass) / halo.mass.sum())

    # Save the rows to a file
    with open(file_name, 'w') as f:
        f.write('ID, Mass, Halo Number, Mass Fraction, Size\n')
        for i in range(len(rows)):
            f.write('{}, {}, {}, {} \n'.format(IDs[i], mass[i], halo_number[i], mass_fraction[i]))
    return 


def run_single_test(args):
    ID, z, signal_choices, halos, sources, xmax = args
    # Run the pipeline for a single cluster, with a given set of signal choices
    # N_test times. Save the results to a file

    xs = sources.x
    ys = sources.y

    sig_s = np.ones(len(xs)) * 0.1
    sig_f = np.ones(len(xs)) * 0.01
    sig_g = np.ones(len(xs)) * 0.02

    masses = []
    candidate_number = []
    chi_scores = []

    # Recreate the sources each time - keeping the positions the same, 
    # But allowing the noises to change

    # Choose a random seed
    np.random.seed()
    noisy_sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    noisy_sources.apply_NFW_lensing(halos)

    # Apply noise
    noisy_sources.e1 += np.random.normal(0, 0.1, len(noisy_sources.x))
    noisy_sources.e2 += np.random.normal(0, 0.1, len(noisy_sources.x))
    noisy_sources.f1 += np.random.normal(0, 0.01, len(noisy_sources.x))
    noisy_sources.f2 += np.random.normal(0, 0.01, len(noisy_sources.x))
    noisy_sources.g1 += np.random.normal(0, 0.02, len(noisy_sources.x))
    noisy_sources.g2 += np.random.normal(0, 0.02, len(noisy_sources.x))

    # Create guesses from each source
    shear = np.sqrt(sources.e1**2 + sources.e2**2)
    flexion = np.sqrt(sources.f1**2 + sources.f2**2)
    r = shear / flexion
    phi = np.arctan2(sources.f2, sources.f1)
    xl = sources.x + r * np.cos(phi)
    yl = sources.y + r * np.sin(phi)
    eR = 2 * shear * r

    Dl, Ds, Dls = utils.angular_diameter_distances(z, 0.8)
    mass = (eR/206265)**2 * (Ds * Dl / Dls) / (4 * G * M_solar) * c**2
    lenses = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
    lenses.calculate_concentration()

    for signal_choice in signal_choices:
        # Write code to run the pipeline for each signal choice using halos
        # WORK IN PROGRESS
        candidate_lenses = copy.deepcopy(lenses) # Copy the lenses object for each signal choice
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)

        # Now run through the pipeline
        # Step 2 - Minimization
        candidate_lenses.optimize_lens_positions(noisy_sources, signal_choice)
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)
        # Step 3 - Filtering
        candidate_lenses.filter_lens_positions(noisy_sources, xmax)
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)
        # Step 4 - Lens Selection
        candidate_lenses.iterative_elimination(noisy_sources, chi2, signal_choice)
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)
        # Step 5 - Merging
        candidate_lenses.merge_close_lenses()
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)
        # Step 6 - Global Minimization
        candidate_lenses.full_minimization(noisy_sources, signal_choice)
        chi2 = candidate_lenses.update_chi2_values(noisy_sources, signal_choice)

        mass = np.sum(candidate_lenses.mass)
        candidate_num = len(candidate_lenses.x)

        chi_scores.append(chi2)
        masses.append(mass)
        candidate_number.append(candidate_num)

    # Save the results to a file
    results = [ID, masses[0], masses[1], masses[2], masses[3], chi_scores[0], chi_scores[1], chi_scores[2], chi_scores[3], candidate_number[0], candidate_number[1], candidate_number[2], candidate_number[3]]
    print('Finished test for cluster {}'.format(ID))
    return results


def run_test_parallel(ID_file, result_file, z, N_test, lensing_type='NFW'):
    IDs = []

    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(int(ID))
    
    IDs = np.array(IDs)
    # IDs = IDs[:5] # For testing purposes, only use the first 5 clusters

    signal_choices = [
        [True, True, True], 
        [True, True, False], 
        [False, True, True], 
        [True, False, True]
    ]

    # Build halos
    halos = find_halos(IDs, z) # halos is a dictionary of Halo objects, keyed by ID
    source_catalogue = {} # Dictionary to hold the source catalogues
    xmax_values = [] # List to hold the maximum extent of each field

    for ID in IDs:
        # Build the lenses and sources
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)

        # Create a new source object, without noise
        xs = sources.x
        ys = sources.y
        clean_source = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.ones(len(xs)) * 0.1, np.ones(len(xs)) * 0.01, np.ones(len(xs)) * 0.02)
        if lensing_type == 'NFW':
            clean_source.apply_NFW_lensing(halos[ID])
        elif lensing_type == 'SIS':
            tE = halos[ID].calc_corresponding_einstein_radius(z_source)
            print('Einstein radius: {}'.format(tE))
            lens = pipeline.Lens(halos[ID].x, halos[ID].y, tE, 0)
            clean_source.apply_SIS_lensing(lens)
        else:
            raise ValueError('Invalid lensing type specified')

        source_catalogue[ID] = clean_source
        xmax_values.append(xmax)

    print('Halo and Source objects loaded...')

    # Prepare the arguments for each task
    tasks = [(ID, z, signal_choices, halos[ID], source_catalogue[ID], xmax_values[i]) for i, ID in enumerate(IDs)]
    # Repeat each task N_test times
    tasks = [task for task in tasks for _ in range(N_test)]

    # Process pool
    with Pool() as pool:
        results = pool.map(run_single_test, tasks)

    # Save the results to a file
    results = np.array(results).reshape(-1, 13)
    with open(result_file, 'w') as f:
        f.write('ID, Mass_all_signals, Mass_gamma_F, Mass_F_G, Mass_gamma_G, Chi2_all_signals, Chi2_gamma_F, Chi2_F_G, Chi2_gamma_G, Nfound_all_signals, Nfound_gamma_F, Nfound_F_G, Nfound_gamma_G\n')
        for i in range(len(results)):
            f.write('{}\n'.format(', '.join(results[i].astype(str))))

    return


# --------------------------------------------
# Data Processing Functions
# --------------------------------------------

def compute_masses(candidate_lenses, z, xmax):
    # Quick helper function to compute the mass of a system

    # Get the true mass of the system
    extent = [-xmax, xmax, -xmax, xmax]
    _,_,kappa = utils.calculate_kappa(candidate_lenses, extent, 5)
    mass = utils.calculate_mass(kappa, z, 0.5, 1)
    return mass


# --------------------------------------------
# Main Functions
# --------------------------------------------

def visualize_fits(ID_file, lensing_type='NFW'):

    # Choose a random cluster ID from the ID file
    IDs = pd.read_csv(ID_file)['ID'].values
    # Make sure the IDs are integers
    IDs = [int(ID) for ID in IDs]
    # Just grab the first 5
    IDs = IDs[:5]
    #ID = np.random.choice(IDs)
    zs = [0.194, 0.391, 0.586, 0.782, 0.977]
    start = time.time()
    halos = find_halos(IDs, zs[0])
    stop = time.time()
    print('Time taken to load halos: {}'.format(stop - start))

    for ID in IDs:
        global_start = time.time()
        halo = halos[ID]
        # Simplify for tests - lets just look at the 3 largest halos
        indices = np.argsort(halo.mass)[::-1][:3]
        halo.x = halo.x[indices]
        halo.y = halo.y[indices]
        halo.mass = halo.mass[indices]
        halo.z = halo.z[indices]
        halo.concentration = halo.concentration[indices]
        
        _, sources, xmax = build_lensing_field(halo, zs[0])
        print('Built lenses and sources...')

        # Copy the halo object, so that one can be altered without affecting the other
        lenses = copy.deepcopy(halo)

        # Arrange a plot with 6 subplots in 2 rows
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))

        use_flags = [True, True, True]  # Use all signals

        # Step 1: Generate initial list of lenses from source guesses
        # lenses = sources.generate_initial_guess()
        start = time.time()
        shear = np.sqrt(sources.e1**2 + sources.e2**2)
        flexion = np.sqrt(sources.f1**2 + sources.f2**2)
        r = shear / flexion
        phi = np.arctan2(sources.f2, sources.f1)
        xl = sources.x + r * np.cos(phi)
        yl = sources.y + r * np.sin(phi)
        eR = 2 * shear * r

        Dl, Ds, Dls = utils.angular_diameter_distances(zs[0], 0.8)
        mass = (eR/206265)**2 * (Ds * Dl / Dls) * c**2 / (4 * G * M_solar) * 10**3
        lenses = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, zs[0], np.zeros_like(xl))
        lenses.calculate_concentration()

        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        # reducedchi2 = 1
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Initial Guesses', ax=axarr[0,0])
        stop = time.time()
        print('Time taken to generate initial guesses: {}'.format(stop - start))
        print('Initial chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))
        
        # Step 2: Optimize guesses with local minimization
        start = time.time()
        lenses.optimize_lens_positions(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        # Check what the lens looks like if the chi2 is nan
        if np.isnan(reducedchi2):
            print('Chi2 is NaN')
            global_chi2 = pipeline.calculate_chi_squared(sources, lenses, use_flags, lensing='NFW')
            print('Global chi2: {}'.format(global_chi2))

        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Initial Optimization', ax=axarr[0,1], legend=False)
        stop = time.time()
        print('Time taken to optimize initial guesses: {}'.format(stop - start))
        print('Optimized chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

        # Step 3: Filter out lenses that are too far from the source population
        start = time.time()
        lenses.filter_lens_positions(sources, xmax)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Filter', ax=axarr[0,2], legend=False)
        stop = time.time()
        print('Time taken to filter lenses: {}'.format(stop - start))
        print('Filtered chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

        # Step 4: Iterative elimination
        # Okay, now lets test this step by injecting the true halos into the candidate lens population
        # This ensures that the true answers are present, which means that the pipeline should find them
        # If it doesn't, then the step is broken
        # If it does, it means that the step is working as intended, and that the issue with the pipeline is an earlier step
        #lenses.x = np.concatenate((lenses.x, halo.x))
        #lenses.y = np.concatenate((lenses.y, halo.y))
        #lenses.mass = np.concatenate((lenses.mass, halo.mass))
        #lenses.z = np.concatenate((lenses.z, halo.z))
        #lenses.concentration = np.concatenate((lenses.concentration, halo.concentration))
        # Update the chi2 values
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        start = time.time()
        lenses.iterative_elimination(sources, reducedchi2, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Iterative Elimination', ax=axarr[1,0], legend=False)
        stop = time.time()
        print('Time taken for iterative elimination: {}'.format(stop - start))
        print('Iterative elimination chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

        # Step 5: Merge lenses that are too close to each other
        start = time.time()
        ns = len(sources.x) / (np.pi * xmax**2)
        merger_threshold = (1/np.sqrt(ns))
        lenses.merge_close_lenses(merger_threshold=merger_threshold)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Merging', ax=axarr[1,1], legend=False)
        stop = time.time()
        print('Time taken for merging: {}'.format(stop - start))
        print('Merging chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

        # Step 6: Final minimization
        start = time.time()
        lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Final Minimization', ax=axarr[1,2], legend=False)
        stop = time.time()
        print('Time taken for final minimization: {}'.format(stop - start))
        print('Final chi2: {:.2f}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

        # Compute mass
        try:
            mass = np.sum(lenses.mass)
        except AttributeError:
            print('Mass not computed')
            mass = 0

        # Save and show the plot
        fig.suptitle('Lensing Reconstruction of Cluster ID {} \n True Mass: {:.2e} $M_\odot$ \n Inferred Mass: {:.2e} $M_\odot$'.format(ID, np.sum(halo.mass), mass))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images/MDARK/pipeline_visualization/Halo_Fit_{}.png'.format(ID))
        # plt.show()
        global_stop = time.time()
        print('Time taken for cluster {}: {}'.format(ID, global_stop - global_start))


def chi2_heatmap():
    # Build a simple heatmap of chi2 values for a very simple cluster
    xmax = 10
    xl = np.array([0])
    yl = np.array([0])
    mass = np.array([1e14])
    z = 0.2
    halo = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
    halo.calculate_concentration()

    # Create a source catalogue
    ns = 0.01
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    sources = utils.createSources(halo, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')

    xgrid = np.linspace(-xmax, xmax, 100)
    ygrid = np.linspace(-xmax, xmax, 100)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)
    chi2 = np.zeros_like(xgrid)
    mass_range = np.logspace(10, 15, 100)
    best_mass = np.zeros_like(xgrid)

    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            for m in mass_range:
                # Find the mass that gives the best chi2 value at this point
                # Initialize the chi2 value to be arbitrarily large
                chi2[i, j] = 1e10
                test_halo = Halo(np.array([xgrid[i, j]]), np.array([ygrid[i, j]]), np.zeros_like(xl), np.zeros_like(xl), np.array([m]), z, np.zeros_like(xl))
                test_halo.calculate_concentration()
                test_chi2 = test_halo.update_chi2_values(sources, [True, True, True])
                if test_chi2 < chi2[i, j]:
                    chi2[i, j] = test_chi2
                    best_mass[i, j] = m
    
    # Plot a heatmap of chi2 and of mass
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(chi2, extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title(r'$\chi^2$ Heatmap')
    fig.colorbar(im, ax=ax[0])
    
    im = ax[1].imshow(np.log10(best_mass), extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title(r'Best Fit Mass (log scale)')
    fig.colorbar(im, ax=ax[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize_fits('Data/MDARK_Test/Test14/ID_file_14.csv', lensing_type='NFW')

    # raise ValueError('This script is not meant to be run as a standalone script')
    # chi2_heatmap()
    # Create a simple system to test this
    xmax = 10
    xl = np.array([0.0])
    yl = np.array([0.0])
    mass = np.array([1e14])
    z = 0.2
    halo = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
    halo.calculate_concentration()

    ns = 0.01
    # Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    Nsource = 1
    # sources = utils.createSources(halo, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')
    xs = np.array([5.0])
    ys = np.array([5.0])
    sources = pipeline.Source(xs, ys, 
                              np.zeros_like(xs), np.zeros_like(xs), 
                              np.zeros_like(xs), np.zeros_like(xs), 
                              np.zeros_like(xs), np.zeros_like(xs), 
                              np.ones(len(xs)) * 0.1, np.ones(len(xs)) * 0.01, np.ones(len(xs)) * 0.02)
    sources.apply_NFW_lensing(halo)
    true_chi2 = halo.update_chi2_values(sources, [True, True, False])
    print('True chi2: {}'.format(true_chi2))

    # Build the guessed lenses
    shear = np.sqrt(sources.e1**2 + sources.e2**2)
    flexion = np.sqrt(sources.f1**2 + sources.f2**2)
    r = shear / flexion
    phi = np.arctan2(sources.f2, sources.f1)
    # r is the distance from the source to the lens
    # phi is the angle pointing from the source to the lens
    xl = sources.x + r * np.cos(phi)
    yl = sources.y + r * np.sin(phi)
    eR = 2 * shear * r

    Dl, Ds, Dls = utils.angular_diameter_distances(z, 0.8)
    mass = (eR/206265)**2 * (Ds * Dl / Dls) * c**2 / (4 * G * M_solar) * 10**3
    lenses = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
    lenses.calculate_concentration()

    # Okay look, the fail points are the minimization and the lens number selection - the rest is fine (or should be)
    # So let's start with the minimization
    # Place a couple of lenses near the true lenses and see if the minimization gets closer or further away from the true values
    # lenses = copy.deepcopy(halo)
    # Now perturb the fit parameters (position and mass)
    # First, the position (correcting for different data types) 
    # lenses.x = xl + np.random.normal(0, 2, len(lenses.x))
    # lenses.y = np.random.normal(0, 2, len(lenses.y))
    # Now the mass
    # lenses.mass += np.random.normal(0, 1e13, len(lenses.mass))
    chi2 = lenses.update_chi2_values(sources, [True, True, True])
    print('Initial chi2: {}'.format(chi2))
    print('Parameter | True Value | Initial Guess')
    print('x | {} | {}'.format(halo.x, lenses.x))
    print('y | {} | {}'.format(halo.y, lenses.y))
    print('Mass | {} | {}'.format(halo.mass, lenses.mass))

    # Now run the minimization
    use_flags = [True, True, True]
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    print('Reduced chi2: {}'.format(reducedchi2))
    print('Parameter | True Value | Minimized Value')
    print('x | {} | {}'.format(halo.x, lenses.x))
    print('y | {} | {}'.format(halo.y, lenses.y))
    print('Mass | {} | {}'.format(halo.mass, lenses.mass))
    # Now plot the results
    plt.figure()
    plt.scatter(halo.x, halo.y, color='red', label='True Lenses')
    plt.scatter(xl, yl, color='blue', label='Initial Guess')
    plt.scatter(lenses.x, lenses.y, color='blue', marker='x', label='Minimized Lenses')
    plt.scatter(sources.x, sources.y, color='green', marker='.', label='Sources')
    # Create a flexion vector pointing from the source to the lens
    plt.quiver(sources.x, sources.y, sources.f1, sources.f2, color='green', label='Source Flexion')
    plt.xlim(-xmax, xmax)
    plt.ylim(-xmax, xmax)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

    # Now let's try the lens number selection
    # Make a bunch of randomly placed lenses, then add the two true lenses. We should find the two true lenses
    # and eliminate the rest
    lenses = copy.deepcopy(halo)
    Nlenses = Nsource # Number of lenses to add, this is our upper limit
    xl = np.random.uniform(-xmax, xmax, Nlenses)
    yl = np.random.uniform(-xmax, xmax, Nlenses)
    mass = np.random.uniform(1e13, 1e15, Nlenses)
    xl = np.append(xl, halo.x)
    yl = np.append(yl, halo.y)
    mass = np.append(mass, halo.mass)
    lenses = Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
    lenses.calculate_concentration()
    chi2 = lenses.update_chi2_values(sources, [True, True, True])
    print('Initial chi2: {}, With {} candidate lenses'.format(chi2, len(lenses.x)))
    lenses.iterative_elimination(sources, chi2, [True, True, True])
    reducedchi2 = lenses.update_chi2_values(sources, [True, True, True])
    print('Reduced chi2: {}, With {} candidate lenses'.format(reducedchi2, len(lenses.x)))

    # Now plot the results
    plt.figure()
    plt.scatter(halo.x, halo.y, color='red', label='True Lenses')
    plt.scatter(xl, yl, color='blue', marker='.', label='Initial Guess')
    plt.scatter(lenses.x, lenses.y, color='green', marker='x', label='Found Lenses')
    plt.xlim(-xmax, xmax)
    plt.ylim(-xmax, xmax)
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()



    raise ValueError('This script is not meant to be run as a standalone script')
    visualize_fits('Data/MDARK_Test/Test14/ID_file_14.csv', lensing_type='NFW')

    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[0]
    start = time.time()
    Ntrials = 1 # Number of trials to run for each cluster in the test set

    test_number = 14
    test_dir = 'Data/MDARK_Test/Test{}'.format(test_number)
    halos_file = 'MDARK/Halos_{}.MDARK'.format(z_chosen)
    ID_file = test_dir + '/ID_file_{}.csv'.format(test_number)
    result_file = test_dir + '/results_{}.csv'.format(test_number)
    plot_name = 'Images/MDARK/mass_correlations/mass_correlation_{}.png'.format(test_number)

    #Check that the ID file exists - if not, create the directory and build the test set
    if not os.path.exists(ID_file):
        os.makedirs('Data/MDARK_Test/Test{}'.format(test_number))
        build_test_set(30, z_chosen, ID_file)
    # raise ValueError('This script is not meant to be run as a standalone script')

    # Choose a random cluster ID from the ID file
    IDs = pd.read_csv(ID_file)['ID'].values
    # Make sure the IDs are integers
    IDs = [int(ID) for ID in IDs]
    zs = [0.194, 0.391, 0.586, 0.782, 0.977]
    start = time.time()
    halos = find_halos(IDs, zs[0])
    stop = time.time()


    run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot_errors(ID_file, result_file, plot_name)
    build_chi2_plot(result_file, ID_file, test_number)