import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import hist as fancy_hist
import pipeline
import utils
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
import pickle
from multiprocessing import Pool
import time
import os

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
    def __init__(self, x, y, z, concentration, mass, redshift):
        self.x = x
        self.y = y
        self.z = z
        self.concentration = concentration
        self.mass = mass
        self.redshift = redshift
    

    def calc_R200(self):
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value
        R200 = (3 / (800 * np.pi) * (self.mass * M_solar / rho_c))**(1/3) # In meters
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        delta_c = (200/3) * (self.concentration**3) / (np.log(1 + self.concentration) - self.concentration / (1 + self.concentration))
        return delta_c


    def calc_corresponding_einstein_radius(self, source_redshift):
        # Compute the Einstein radius for a given source redshift
        Ds = cosmo.angular_diameter_distance(source_redshift).to(u.meter).value
        Dls = cosmo.angular_diameter_distance_z1z2(self.redshift, source_redshift).to(u.meter).value
        Dl = cosmo.angular_diameter_distance(self.redshift).to(u.meter).value
        eR = np.sqrt((4 * G * self.mass * M_solar) / (c**2) * (Dls / (Ds * Dl)))
        #rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value
        #eR = (2 * np.pi * G / c**2) * (Dls / Ds) * (800 * np.pi * rho_c / 3)**(1/3) * (self.mass * M_solar)**(2/3)
        eR *= 206265 # Convert to arcseconds
        # Return the Einstein radius - make sure its an array
        if np.isscalar(eR):
            eR = np.array([eR])
        return eR


    def calc_shear_signal(self, xs, ys):
        # Compute the NFW shear signal at a given position (xs, ys), for the entire set of halos

        def radial_term_2(x):
            # Compute the radial term - this is called g(x) in theory
            if x < 1:
                term1 = 8 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / (x**2 * np.sqrt(1 - x**2))
                term2 = 4 / x**2 * np.log(x / 2)
                term3 = -2 / (x**2 - 1)
                term4 = 4 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / ((x**2 - 1) * np.sqrt(1 - x**2))
                sol = term1 + term2 + term3 + term4
            elif x == 1:
                sol = 10 / 3 + 4 * np.log(1 / 2)
            elif x > 1:
                term1 = 8 * np.arctan(np.sqrt((x - 1) / (1 + x))) / (x**2 * np.sqrt(x**2 - 1))
                term2 = 4 / x**2 * np.log(x / 2)
                term3 = -2 / (x**2 - 1)
                term4 = 4 * np.arctan(np.sqrt((x - 1) / (1 + x))) / ((x**2 - 1)**(3/2))
                sol = term1 + term2 + term3 + term4
            else:
                raise ValueError('Invalid value of x')
            return sol
        
        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.concentration # In meters

        # Compute angular diameter distances
        Ds = cosmo.angular_diameter_distance(z_source).to(u.meter).value
        Dl = cosmo.angular_diameter_distance(self.redshift).to(u.meter).value
        Dls = cosmo.angular_diameter_distance_z1z2(self.redshift, z_source).to(u.meter).value

        # Compute the critical surface density
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value 
        rho_s = rho_c * self.calc_delta_c() 
        sigma_c = (c**2 / (4 * np.pi * G)) * (Ds / (Dl * Dls))
        kappa_s = rho_s * rs / sigma_c

        # Initialize shear signals
        shear_1 = np.zeros(len(xs))
        shear_2 = np.zeros(len(xs))

        for i in range(len(xs)):
            # Compute shear signal at each position
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5

            x = r / (r200_arcsec / self.concentration)
            radial_term = np.array([radial_term_2(val) for val in x])

            shear_mag = kappa_s * radial_term

            cos_phi = dx / r
            sin_phi = dy / r
            cos2phi = cos_phi * cos_phi - sin_phi * sin_phi
            sin2phi = 2 * cos_phi * sin_phi

            shear_1[i] -= np.sum(shear_mag * cos2phi)
            shear_2[i] -= np.sum(shear_mag * sin2phi)

        return shear_1, shear_2


    def calc_F_signal(self, xs, ys):
        # Compute the NFW first flexion signal at a given position (xs, ys), for the entire set of halos

        def radial_term_1(x):
            # Compute the radial term - this is called f(x) in theory
            if x < 1:
                sol = 1 - (2 / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            elif x == 1:
                # This is a special case, unlikely to occur in practice
                sol = 1 - np.pi / 2
            elif x > 1:
                sol = 1 - (2 / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (x + 1)))
            return sol

        def radial_term_3(x):
            # Compute the radial term - this is called h(x) in theory
            if x < 1:
                sol = (2 * x) / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x))) - 1 / x
            elif x == 1:
                sol = 1 / 3
            elif x > 1:
                sol = (2 * x) / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x))) - 1 / x
            return sol

        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.concentration # In meters

        # Compute the angular diameter distances
        Ds = cosmo.angular_diameter_distance(z_source).to(u.meter).value
        Dl = cosmo.angular_diameter_distance(self.redshift).to(u.meter).value
        Dls = cosmo.angular_diameter_distance_z1z2(self.redshift, z_source).to(u.meter).value

        # Compute the critical surface density
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value 
        rho_s = rho_c * self.calc_delta_c()
        sigma_c = (c**2 / (4 * np.pi * G)) * (Ds / (Dl * Dls))
        kappa_s = rho_s * rs / sigma_c
        F_s = kappa_s * Dl / rs

        # Initialize flexion signals
        f1 = np.zeros(len(xs))
        f2 = np.zeros(len(xs))

        for i in range(len(xs)):
            # Compute flexion signal at each position
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5

            x = r / (r200_arcsec / self.concentration) # In arcseconds
            term_1 = np.zeros(len(r))
            term_3 = np.zeros(len(r))
            for val in range(len(x)):
                term_1[val] = radial_term_1(x[val])
                term_3[val] = radial_term_3(x[val])

            F_mag = (-2 * F_s / (x**2 - 1)**2) * (2 * x * term_1 - term_3) # In units of inverse radians
            F_mag /= 206265 # Convert to inverse arcseconds

            f1[i] += np.sum(F_mag * dx / r)
            f2[i] += np.sum(F_mag * dy / r)

        return f1, f2


    def calc_G_signal(self, xs, ys):
        # Compute the NFW second flexion signal at a given position (xs, ys), for the entire set of halos

        def radial_term_4(x):
            # Compute the radial term - this is called i(x) in theory
            leading_term = (8 / x**3 - 20 / x + 15*x)
            if x < 1:
                sol = leading_term * (2 / np.sqrt(1 - x**2)) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            elif x == 1:
                sol = leading_term
            elif x > 1:
                sol = leading_term * (2 / np.sqrt(x**2 - 1)) * np.arctan(np.sqrt((x - 1) / (1 + x)))
            return sol

        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.concentration

        # Compute the angular diameter distances
        Ds = cosmo.angular_diameter_distance(z_source).to(u.meter).value
        Dl = cosmo.angular_diameter_distance(self.redshift).to(u.meter).value
        Dls = cosmo.angular_diameter_distance_z1z2(self.redshift, z_source).to(u.meter).value

        # Compute the critical surface density
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value 
        rho_s = rho_c * self.calc_delta_c()
        sigma_c = (c**2 / (4 * np.pi * G)) * (Ds / (Dl * Dls))
        kappa_s = rho_s * rs / sigma_c
        F_s = kappa_s * Dl / rs

        g1 = np.zeros(len(xs))
        g2 = np.zeros(len(xs))

        for i in range(len(xs)):
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5

            x = r / (r200_arcsec / self.concentration) # In arcseconds
            term_4 = np.zeros(len(r))
            for val in range(len(x)):
                term_4[val] = radial_term_4(x[val])

            cos_phi = dx / r
            sin_phi = dy / r
            cos2phi = cos_phi * cos_phi - sin_phi * sin_phi
            sin2phi = 2 * cos_phi * sin_phi
            cos3phi = cos2phi * cos_phi - sin2phi * sin_phi
            sin3phi = sin2phi * cos_phi + cos2phi * sin_phi

            log_term = np.empty(len(x))
            for val in range(len(x)):
                # Quick hack to avoid error in log term
                log_term[val] = (8 / x[val]**3) * np.log(x[val] / 2)
            
            G_mag = 2 * F_s * (log_term + ((3/x)*(1 - 2*x**2) + term_4) / (x**2 - 1)**2)
            G_mag /= 206265 # Convert to inverse arcseconds 

            g1[i] += np.sum(G_mag * cos3phi)
            g2[i] += np.sum(G_mag * sin3phi)
        
        return g1, g2


# --------------------------------------------
# File Management Functions
# --------------------------------------------

def fix_file(z):
    # Fix the key files
    # THIS HAS BEEN COMPLETED - IT SHOULD NOT BE RUN AGAIN
    # IT IS INCLUDED FOR ARCHIVAL PURPOSES

    file = dir + 'Key_{}.MDARK'.format(z) 

    # Define the data types for each column to improve performance
    data_types = {f'column_{i}': 'float' for i in range(6)}

    df = pd.read_csv(file, nrows=0)

    # Accessing the column names
    column_names = df.columns.tolist()
    # The 6th column name is 'Characteristic Size10898716021' - split it into 'Characteristic Size' and '10898716021'
    problem_column_name = column_names[5]
    # We know that 'Characteristic Size' is 19 characters long, so we can split the string at that point
    column_names[5] = problem_column_name[:20] # (0-19), remember that the upper bound is not included
    # The remaining characters get added to the next column name
    column_names.insert(6, problem_column_name[20:])
    # Now grab column names from 7 - 12, turn this into a row of data
    # and append it to the end of the dataframe
    header = column_names[6:12]
    
    # Corrected column names
    column_names = column_names[0:6]
    
    # Read the CSV file in chunks
    chunk_size = 10000  # Adjust based on your memory constraints
    chunks = pd.read_csv(file, dtype=data_types, chunksize=chunk_size)


    # Write the corrected file
    new_file = dir + 'fixed_key_{}.MDARK'.format(z)
    with open(new_file, 'w') as f:
        f.write(','.join(column_names) + '\n')
        f.write(','.join(header) + '\n')
        # Close here
        f.close()
        for chunk in chunks:
            chunk_subset = chunk.iloc[:, :6]
            chunk_subset.to_csv(new_file, mode='a', header=False, index=False)

    # close the file
    f.close()


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
            halos[id] = Halo(xhalo, yhalo, zhalo, chalo, masshalo, z)
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
    x,y = utils.project_onto_principal_axis(halos.x, halos.y, halos.z)

    # Convert coordinates to arcseconds (from Mpc)
    d = cosmo.angular_diameter_distance(z).to(u.meter)
    # Convert to arcseconds (first to meters as intermediate step)
    x = (x * 3.086 * 10**22 / d).value * 206265
    y = (y * 3.086 * 10**22 / d).value * 206265

    # Update the halos object
    halos.x = x
    halos.y = y

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
    
    for signal_choice in signal_choices:
        candidate_lenses, chi2 = pipeline.fit_lensing_field(noisy_sources, xmax, flags=False, use_flags=signal_choice)

        mass = compute_masses(candidate_lenses, z, xmax)
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
    def _plot_results(xmax, lenses, sources, true_lenses, reducedchi2, title, ax=None, legend=True):
        """Private helper function to plot the results of lensing reconstruction."""
        if ax is None:
            fig, ax = plt.subplots()
        ax.scatter(lenses.x, lenses.y, color='red', label='Recovered Lenses')
        for i, eR in enumerate(lenses.te):
            ax.annotate(np.round(eR, 2), (lenses.x[i], lenses.y[i]))
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

    # Choose a random cluster ID from the ID file
    IDs = pd.read_csv(ID_file)['ID'].values
    # Make sure the IDs are integers
    IDs = [int(ID) for ID in IDs]
    # Grab the first five
    # IDs = IDs[:5]
    # ID = np.random.choice(IDs)
    zs = [0.194, 0.391, 0.586, 0.782, 0.977]

    start = time.time()

    halos = find_halos(IDs, zs[0])

    stop = time.time()
    print('Time taken to load halos: {}'.format(stop - start))

    for ID in IDs:
        # Load the halos
        halo = halos[ID]
        lenses, sources, xmax = build_lensing_field(halo, zs[0])

        # Control the kind of lensing we're looking at
        xs = sources.x
        ys = sources.y

        e1 = np.random.normal(0, 0.1, len(xs))
        e2 = np.random.normal(0, 0.1, len(xs))
        f1 = np.random.normal(0, 0.01, len(xs))
        f2 = np.random.normal(0, 0.01, len(xs))
        g1 = np.random.normal(0, 0.02, len(xs))
        g2 = np.random.normal(0, 0.02, len(xs))

        clean_source = pipeline.Source(xs, ys, e1, e2, f1, f2, g1, g2, np.ones(len(xs)) * 0.1, np.ones(len(xs)) * 0.01, np.ones(len(xs)) * 0.02)
        if lensing_type == 'NFW':
            clean_source.apply_NFW_lensing(halos[ID])
        elif lensing_type == 'SIS':
            tE = halos[ID].calc_corresponding_einstein_radius(z_source)            
            lens = pipeline.Lens(halos[ID].x, halos[ID].y, tE, 0)
            clean_source.apply_SIS_lensing(lens)
        else:
            raise ValueError('Invalid lensing type specified')
        
        sources = clean_source

        # Arrange a plot with 6 subplots in 2 rows
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10))

        use_flags = [True, True, True]  # Use all signals

        # Step 1: Generate initial list of lenses from source guesses
        lenses = sources.generate_initial_guess()
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Initial Guesses', ax=axarr[0,0])

        # Step 2: Optimize guesses with local minimization
        lenses.optimize_lens_positions(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Initial Optimization', ax=axarr[0,1], legend=False)

        # Step 3: Filter out lenses that are too far from the source population
        lenses.filter_lens_positions(sources, xmax)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Filter', ax=axarr[0,2], legend=False)

        # Step 4: Iterative elimination
        lenses.iterative_elimination(sources, reducedchi2, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Iterative Elimination', ax=axarr[1,0], legend=False)

        # Step 5: Merge lenses that are too close to each other
        start = time.time()
        ns = len(sources.x) / (np.pi * xmax**2)
        merger_threshold = (1/np.sqrt(ns))
        lenses.merge_close_lenses(merger_threshold=merger_threshold)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Merging', ax=axarr[1,1], legend=False)

        # Step 6: Final minimization
        lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(xmax, lenses, sources, halo, reducedchi2, 'Final Minimization', ax=axarr[1,2], legend=False)

        # Compute mass
        mass = compute_masses(lenses, zs[0], xmax)

        # Save and show the plot
        fig.suptitle('Lensing Reconstruction of Cluster ID {} \n True Mass: {:.2e} $M_\odot$ \n Inferred Mass: {:.2e} $M_\odot$'.format(ID, np.sum(halo.mass), mass))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images/MDARK/pipeline_visualization/ID_{}.png'.format(ID))
        # plt.show()

        # Now lets create a lens object that perfectly matches the true lenses
        true_eR = halo.calc_corresponding_einstein_radius(z_source)
        perfect_lens_fit = pipeline.Lens(halo.x, halo.y, true_eR, np.zeros_like(true_eR))
        # Update the chi2 values - if the fit is perfect, the reduced chi2 should be 1
        perfect_lens_fit.update_chi2_values(sources, use_flags)
        reducedchi2 = perfect_lens_fit.update_chi2_values(sources, use_flags)

        # Plot the perfect fit
        fig, ax = plt.subplots()
        _plot_results(xmax, perfect_lens_fit, sources, halo, reducedchi2, None, ax=ax)

        # Get the mass of the perfect fit
        mass = compute_masses(perfect_lens_fit, zs[0], xmax)
        
        plt.suptitle('chi2 = {} \n True Mass: {:.2e} $M_\odot$ \n Inferred Mass: {:.2e} $M_\odot$'.format(reducedchi2, np.sum(halo.mass), mass))
        plt.savefig('Images/MDARK/pipeline_visualization/Perfect_Fit_{}.png'.format(ID))
        plt.close()


if __name__ == '__main__':
    # visualize_fits('Data/MDARK_Test/Test14/ID_file_14.csv', lensing_type='SIS')
    # raise ValueError('This script is not meant to be run as a standalone script')

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

    run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='SIS')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot_errors(ID_file, result_file, plot_name)
    build_chi2_plot(result_file, ID_file, test_number)