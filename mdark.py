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
h = 0.7 # Hubble constant
M_solar = 1.989 * 10**30 # Solar mass in kg


class Halo:
    def __init__(self, x, y, z, c, mass, redshift):
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.mass = mass
        self.redshift = redshift
    

    def calc_R200(self):
        z = self.redshift
        omega_crit = 0.27 * (1 + z)**3
        omega_z = 0.27 * (1 + z)**3 / (0.27 * (1 + z)**3 + 0.73)
        R200 = (1.63 * 10**-2) * (self.mass / h**-1)**(1/3) * (omega_crit / omega_z)**(-1/3) * (1 + z)**(-1) * h**-1 # In Kpc
        # Convert to meters
        R200 = R200 * 3.086 * 10**19
        # Convert to arcseconds
        R200_arcsec = (R200 / cosmo.angular_diameter_distance(self.redshift).to(u.meter).value) * 206265
        return R200, R200_arcsec


    def calc_delta_c(self):
        # Compute the characteristic density contrast for each halo
        delta_c = (200/3) * (self.c**3) / (np.log(1 + self.c) - self.c / (1 + self.c))
        return delta_c


    def calc_shear_signal(self, xs, ys):
        # Compute the NFW shear signal at a given position (xs, ys), for the entire set of halos

        def radial_term_2(x):
            # Compute the radial term - this is called g(x) in theory
            if x < 1:
                term1 = 8 * np.arctanh(np.sqrt((1 - x) / (1 + x))) / (x**2 * np.sqrt(1 - x**2))
                term2 = 4 / x**2 * np.log(x / 2)
                term3 = -2 / (x**2 * np.sqrt(1 - x**2))
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
                sol = 0
                print('Error: x is less than zero')
            return sol
        
        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.c # In meters

        z_source = 0.5
        # z_source gives the redshift of the source galaxies, redshift of the lens is self.redshift
        # Compute the angular diameter distances
        Ds = cosmo.angular_diameter_distance(z_source).to(u.meter).value
        Dl = cosmo.angular_diameter_distance(self.redshift).to(u.meter).value
        Dls = cosmo.angular_diameter_distance_z1z2(self.redshift, z_source).to(u.meter).value

        # Compute the critical surface density
        rho_c = cosmo.critical_density(self.redshift).to(u.kg / u.m**3).value 
        rho_s = rho_c * self.calc_delta_c() 
        sigma_c = (c**2 / (4 * np.pi * G)) * (Ds / (Dl * Dls))
        kappa_s = rho_s * rs / sigma_c

        shear_1 = np.zeros(len(xs))
        shear_2 = np.zeros(len(xs))

        for i in range(len(xs)):
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5

            x = r / (r200_arcsec / self.c)
            term_2 = np.zeros(len(x))
            for val in range(len(x)):
                term_2[val] = radial_term_2(x[val])

            shear_mag = kappa_s * term_2
            phi = np.empty(len(dx))
            for j in range(len(dx)):
                phi[j] = np.arctan2(dy[j], dx[j])

            shear_1[i] += np.sum(shear_mag * np.cos(2 * phi))
            shear_2[i] += np.sum(shear_mag * np.sin(2 * phi))

        return shear_1, shear_2


    def calc_F_signal(self, xs, ys):
        # Compute the NFW first flexion signal at a given position (xs, ys), for the entire set of halos

        def radial_term_1(x):
            # Compute the radial term - this is called f(x) in theory
            if x < 1:
                sol = 1 - 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            elif x == 1:
                # This is a special case, unlikely to occur in practice
                sol = 1 - np.pi / 2
            elif x > 1:
                sol = 1 - 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))
            return sol

        def radial_term_3(x):
            # Compute the radial term - this is called h(x) in theory
            if x < 1:
                sol = 1/x - (2 * x) / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            elif x == 1:
                sol = 1 / 3
            elif x > 1:
                sol = (2 * x) / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x))) - 1 / x
            return sol

        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.c # In meters

        z_source = 0.5
        # z_source gives the redshift of the source galaxies, redshift of the lens is self.redshift
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

        f1 = np.zeros(len(xs))
        f2 = np.zeros(len(ys))

        for i in range(len(xs)):
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5
            phi = np.empty(len(dx))
            for j in range(len(dx)):
                phi[j] = np.arctan2(dy[j], dx[j])

            x = r / (r200_arcsec / self.c) # In arcseconds
            term_1 = np.zeros(len(r))
            term_3 = np.zeros(len(r))
            for val in range(len(x)):
                term_1[val] = radial_term_1(x[val])
                term_3[val] = radial_term_3(x[val])

            F_mag = (-2 * F_s / (x**2 - 1)**2) * (2 * x * term_1 - term_3) # In units of inverse radians
            F_mag /= 206265 # Convert to inverse arcseconds
            f1[i] += np.sum(F_mag * np.cos(phi))
            f2[i] += np.sum(F_mag * np.sin(phi))

        return f1, f2


    def calc_G_signal(self, xs, ys):
        # Compute the NFW second flexion signal at a given position (xs, ys), for the entire set of halos

        def radial_term_4(x):
            # Compute the radial term - this is called i(x) in theory
            leading_term = (8 / x**3 - 20 / x + 15*x)
            if x < 1:
                sol = leading_term * 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
            elif x == 1:
                sol = leading_term
            elif x > 1:
                sol = leading_term * 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))
            return sol

        r200, r200_arcsec = self.calc_R200()
        rs = r200 / self.c

        z_source = 0.5
        # z_source gives the redshift of the source galaxies, redshift of the lens is self.redshift
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
        g2 = np.zeros(len(ys))

        for i in range(len(xs)):
            dx = xs[i] - self.x
            dy = ys[i] - self.y
            r = (dx**2 + dy**2)**0.5
            phi = np.empty(len(dx))
            for j in range(len(dx)):
                phi[j] = np.arctan2(dy[j], dx[j])

            x = r / (r200_arcsec / self.c)
            term_4 = np.zeros(len(r))
            for val in range(len(x)):
                term_4[val] = radial_term_4(x[val])

            log_term = np.empty(len(x))
            for val in range(len(x)):
                # Quick hack to avoid error in log term
                log_term[val] = 8 / x[val]**3 * np.log(x[val] / 2)
            
            G_mag = 2 * F_s * (log_term + ((3/x)*(1 - 2*x**2) + term_4) / (x**2 - 1)**2)
            G_mag /= 206265 # Convert to inverse arcseconds 
            g1[i] += np.sum(G_mag * np.cos(3 * phi))
            g2[i] += np.sum(G_mag * np.sin(3 * phi))
        
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


def build_mass_correlation_plot_errors(file_name, plot_name):
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    # Get the mass and true mass
    true_mass = results[' True Mass'].values
    mean_masses = results[[' Mean Mass_all_signals', ' Mean Mass_gamma_F', ' Mean Mass_F_G', ' Mean Mass_gamma_G']].values
    std_masses = results[[' Std Mass_all_signals', ' Std Mass_gamma_F', ' Std Mass_F_G', ' Std Mass_gamma_G']].values
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    # Plot the results for each signal combination
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    for i in range(4):
        mass_values = np.abs(mean_masses[:, i])
        mass_errors = std_masses[:, i]

        # Remove NaN values
        nan_values = np.isnan(mass_values)
        true_mass_temp = true_mass[~nan_values]
        mass_values = mass_values[~nan_values]
        mass_errors = mass_errors[~nan_values]


        '''
        if mean_masses_temp.min() == 0:
            true_mass_temp = true_mass_temp[mean_masses_temp > 0]
            mean_masses_temp = mean_masses_temp[mean_masses_temp > 0]
            std_masses_temp = std_masses_temp[mean_masses_temp > 0]
        '''
        ax[i].errorbar(true_mass_temp, mass_values, yerr=mass_errors, fmt='o', color='black')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        # Add a line of best fit
        x = np.linspace(1e13, 1e15, 100)
        try:
            m, b = np.polyfit(np.log10(true_mass_temp), np.log10(mass_values), 1)
            ax[i].plot(x, 10**(m*np.log10(x) + b), color='red', label='Best Fit: m = {:.2f}'.format(m))
        except:
            print('RuntimeWarning: Skipping line of best fit')
            continue
        # Plot the line of best fit and an agreement line
        ax[i].plot(x, x, color='blue', label='Agreement Line', linestyle='--') # Agreement line - use a different linestyle because the paper won't be in color
        ax[i].legend()
        ax[i].set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
        ax[i].set_ylabel(r'$M_{\rm inferred}$ [$M_{\odot}$]')
        ax[i].set_title('Signal Combination: {} \n Correlation Coefficient: {:.2f}'.format(signals[i], np.corrcoef(true_mass_temp, mass_values)[0, 1]))

    fig.suptitle('Mass Correlation')
    fig.tight_layout()
    fig.savefig(plot_name)
    # plt.show()



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


def find_halos(ID, z):
    # Given a cluster ID, locate the cluster in the data
    # and return the relevant information
    # This needs to be fast 

    file = dir + 'Halos_{}.MDARK'.format(z)
    # Find all the halos with the given ID

    # Create an empty DataFrame to store the results
    filtered_data = pd.DataFrame()

    # Read the CSV file in chunks
    for chunk in pd.read_csv(file, chunksize=10000):
        # Filter the chunk for the specified ObjectID
        filtered_chunk = chunk[chunk['MainHaloID'] == ID]
        filtered_data = pd.concat([filtered_data, filtered_chunk])

    # Now we have the correct objects, we can return the relevant information
    xhalo = filtered_data['x'].values
    yhalo = filtered_data['y'].values
    zhalo = filtered_data['z'].values
    chalo = filtered_data['concentration_NFW'].values
    masshalo = filtered_data['HaloMass'].values
    halo_type = filtered_data['GalaxyType'].values

    # Remove any halos of type 2 - these are 'orphan' halos that could not be properly tracked
    xhalo = xhalo[halo_type != 2]
    yhalo = yhalo[halo_type != 2]
    zhalo = zhalo[halo_type != 2]
    chalo = chalo[halo_type != 2]
    masshalo = masshalo[halo_type != 2]

    # Set these to be numpy arrays
    xhalo = np.array(xhalo)
    yhalo = np.array(yhalo)
    zhalo = np.array(zhalo)
    chalo = np.array(chalo)
    masshalo = np.array(masshalo)

    halos = Halo(xhalo, yhalo, zhalo, chalo, masshalo, z)

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
    # Offset by a small amount, so that we are looking at the
    # center of the cluster, but not directly at the most massive halo

    largest_halo = np.argmax(lenses.mass)
    centroid = [lenses.x[largest_halo], lenses.y[largest_halo]]
    lenses.x -= centroid[0] + np.random.uniform(-20, 20)
    lenses.y -= centroid[1] + np.random.uniform(-20, 20)

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


def build_test_set(Nhalos, z, file_name):
    # Select a number of clusters, spaced evenly in log space across the mass range
    # This set will be used to test the pipeline
    # For each cluster, we will save the following information
    # ID, Mass, Halo Number, Mass Fraction, Size

    # Establish criteria for cluster selection
    M_min = 1e13
    M_max = 1e15
    substructure_min = 0
    substructure_max = 0.1
    mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), Nhalos+1)

    rows = []
    for i in range(Nhalos):
        mass_range = [mass_bins[i], mass_bins[i+1]]
        substructure_range = [substructure_min, substructure_max]

        row = choose_ID(z, mass_range, substructure_range)
        if row is not None:
            print('Found a cluster in mass bin {}'.format(i))
            rows.append(row)
        else:
            print('No cluster found in mass bin {}'.format(i))

    # Save the rows to a file
    with open(file_name, 'w') as f:
        f.write('ID, Mass, Halo Number, Mass Fraction, Size\n')
        for row in rows:
            for i in range(len(row)):
                f.write('{}, {}, {}, {}, {}\n'.format(row['MainHaloID'].values[i], row[' Total Mass'].values[i], row[' Halo Number'].values[i], row[' Mass Fraction'].values[i], row[' Characteristic Size'].values[i]))
    
    return 


def make_catalogue(sources, name):
    # Given a set of sources, build a catalogue as a csv file, with the following columns
    # x, y, e1, e2, f1, f2, g1, g2
    # This is the format expected by the pipeline
    with open(name, 'w') as f:
        f.write('x, y, e1, e2, f1, f2, g1, g2\n')
        for i in range(len(sources.x)):
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(sources.x[i], sources.y[i], sources.e1[i], sources.e2[i], sources.f1[i], sources.f2[i], sources.g1[i], sources.g2[i]))
    f.close()


def run_test(ID_file, result_file, z):
    IDs = []
    # Load the list of IDs - this is a csv file
    with open(ID_file, 'r') as f:
        lines = f.readlines() # Read the lines
        # Skip the first line
        lines = lines[1:] 
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(ID)

    # Create a CSV file to hold the results
    with open(result_file, 'w') as f:
        f.write('ID, True Mass, Mass_all_signals, Mass_gamma_F, Mass_F_G, Mass_gamma_G, N_halos, Nfound_all_signals, Nfound_gamma_F, Nfound_F_G, Nfound_gamma_G\n')
    f.close()

    use_all_signals = [True, True, True] # Use all signals
    shear_flex = [True, True, False] # Use shear and flexion
    all_flex = [False, True, True] # Use flexion and g-flexion
    global_signals = [True, False, True] # Use shear and g-flexion (global signals)
    signal_choices = [use_all_signals, shear_flex, all_flex, global_signals]

    for ID in IDs:
        label = 'Data/MDARK_Test/{}_test'.format(ID)

        halos = find_halos(int(ID), z)
        true_mass = np.sum(halos.mass)

        halos, sources, xmax = build_lensing_field(halos, z)
        masses = []
        candidate_number = []

        for signal_choice in signal_choices:
            # Run the pipeline with each possible choice of signals
            candidate_lenses, _ = pipeline.fit_lensing_field(sources, xmax, flags=False, use_flags=signal_choice)
            mass = compute_masses(candidate_lenses, z) # Compute the mass of the resulting system
            masses.append(mass)
            candidate_number.append(len(candidate_lenses.x))

        # Save the results to a file
        with open(result_file, 'a') as f:
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(ID, true_mass, masses[0], masses[1], masses[2], masses[3], len(halos.mass), candidate_number[0], candidate_number[1], candidate_number[2], candidate_number[3]))
        # Save the sources - these can be passed off to other pipelines for comparison
        make_catalogue(sources, label+'_sources.csv')

        # Also save the lenses - use pickle to save these objects
        with open(label+'_halos.pkl', 'wb') as f:
            pickle.dump(halos, f)
        with open(label+'_candidate_lenses.pkl', 'wb') as f:
            pickle.dump(candidate_lenses, f)

    print('Done!')


def run_single_test(args):
    ID, z, N_test, signal_choices = args
    # Run the pipeline for a single cluster, with a given set of signal choices
    # N_test times. Save the results to a file

    all_masses = []
    all_candidate_numbers = []

    halos = find_halos(int(ID), z)
    halos, sources, xmax = build_lensing_field(halos, z)
    chi_scores = []

    xs = sources.x
    ys = sources.y

    noiseless_sources = pipeline.Source(xs, ys, np.empty_like(xs), np.empty_like(xs), np.empty_like(xs), np.empty_like(xs), np.empty_like(xs), np.empty_like(xs), 0.1, 0.01, 0.02)
    noiseless_sources.apply_NFW_lensing(halos, z, 0.5, xmax)

    for _ in range(N_test):
        masses = []
        candidate_number = []

        # Recreate the sources each time - keeping the positions the same, 
        # But allowing the noises to change

        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
        noisy_sources = sources # Copy the sources

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

        all_masses.append(masses)
        all_candidate_numbers.append(candidate_number)
        print('Finished test {} for ID {}'.format(_, ID))

    # Calculate mean and standard deviation for each type of mass and candidate number
    all_masses = np.array(all_masses)
    all_candidate_numbers = np.array(all_candidate_numbers)

    # This is crude, but not slow
    masses_1 = all_masses[:, 0]
    masses_2 = all_masses[:, 1]
    masses_3 = all_masses[:, 2]
    masses_4 = all_masses[:, 3]

    candidate_numbers_1 = all_candidate_numbers[:, 0]
    candidate_numbers_2 = all_candidate_numbers[:, 1]
    candidate_numbers_3 = all_candidate_numbers[:, 2]
    candidate_numbers_4 = all_candidate_numbers[:, 3]
    
    mean_masses = [np.mean(masses_1), np.mean(masses_2), np.mean(masses_3), np.mean(masses_4)]
    std_masses = [np.std(masses_1), np.std(masses_2), np.std(masses_3), np.std(masses_4)]

    mean_candidate_numbers = [np.mean(candidate_numbers_1), np.mean(candidate_numbers_2), np.mean(candidate_numbers_3), np.mean(candidate_numbers_4)]
    std_candidate_numbers = [np.std(candidate_numbers_1), np.std(candidate_numbers_2), np.std(candidate_numbers_3), np.std(candidate_numbers_4)]

    # Now sort the results into a list, with the mean and std of each signal in order
    results = [[np.sum(halos.mass), mean_masses[0], std_masses[0], mean_masses[1], std_masses[1], mean_masses[2], std_masses[2], mean_masses[3], std_masses[3], len(halos.mass), mean_candidate_numbers[0], std_candidate_numbers[0], mean_candidate_numbers[1], std_candidate_numbers[1], mean_candidate_numbers[2], std_candidate_numbers[2], mean_candidate_numbers[3], std_candidate_numbers[3]], chi_scores]

    return results


def run_test_parallel(ID_file, result_file, z, N_test):
    IDs = []

    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(ID)
    
    signal_choices = [
        [True, True, True], 
        [True, True, False], 
        [False, True, True], 
        [True, False, True]
    ]

    # Prepare the arguments for each task
    tasks = [(ID, z, N_test, signal_choices) for ID in IDs]

    # Process pool
    with Pool() as pool:
        results = pool.map(run_single_test, tasks)
    
    # Extract the results and chi2 scores
    results, chi2_scores = zip(*results)
    
    # Turn the results into a list of strings
    results = [','.join([ID] + [str(val) for val in result]) + '\n' for ID, result in zip(IDs, results)]
    # Turn the chi2 scores into a list of strings
    chi2_scores = [','.join([ID] + [str(val) for val in chi2]) + '\n' for ID, chi2 in zip(IDs, chi2_scores)]


    # Write results to file
    with open(result_file, 'w') as f:
        f.write('ID, True Mass, Mean Mass_all_signals, Std Mass_all_signals, Mean Mass_gamma_F, Std Mass_gamma_F, Mean Mass_F_G, Std Mass_F_G, Mean Mass_gamma_G, Std Mass_gamma_G, N_halos, Mean Nfound_all_signals, Std Nfound_all_signals, Mean Nfound_gamma_F, Std Nfound_gamma_F, Mean Nfound_F_G, Std Nfound_F_G, Mean Nfound_gamma_G, Std Nfound_gamma_G\n')
        for result in results:
            f.write(result)

    # Save the chi2 scores to a file
    with open(result_file.replace('results', 'chi2_scores'), 'w') as f:
        f.write('ID, chi2_all_signals, chi2_gamma_F, chi2_F_G, chi2_gamma_G\n')
        for chi2 in chi2_scores:
            f.write(chi2)
        

# --------------------------------------------
# Data Processing Functions
# --------------------------------------------

def compute_masses(candidate_lenses, z, xmax):
    # Given the results of the pipeline, process the results
    # We can directly compare the input lenses to the candidate lenses
    # We can also compare the true mass from the halos to the inferred mass
    # from the candidate lenses

    # Get the true mass of the system
    extent = [-xmax, xmax, -xmax, xmax]
    _,_,kappa = utils.calculate_kappa(candidate_lenses, extent, 5)
    mass = utils.calculate_mass(kappa, z, 0.5, 1)
    return mass


def goodness_of_fit_test(ID):
    # Given a cluster ID, run the pipeline for many different source numbers, 
    # recording the chi2 value and mass estimate for each run

    z = 0.194
    signal_choices = [
        [True, True, True],
        [True, True, False],
        [False, True, True],
        [True, False, True]
    ]

    labels = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    halos = find_halos(ID, z)
    true_mass = np.sum(halos.mass)

    Nsource_low = 100
    Nsource_high = 1000
    Nsources = np.logspace(np.log10(Nsource_low), np.log10(Nsource_high), 20).astype(int)
    chi2_values = []
    mass_values = []

    halos, sources, xmax = build_lensing_field(halos, z, Nsource=Nsources[0])

    for Nsource in Nsources:
        # I don't need to recreate the halos each time, just the sources
        sources = utils.createSources(halos, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')
        for signal_choice in signal_choices:
            candidate_lenses, chi2 = pipeline.fit_lensing_field(sources, xmax, flags=False, use_flags=signal_choice)
            mass = compute_masses(candidate_lenses, z, xmax)
            chi2_values.append(chi2)
            mass_values.append(mass)
    
    # There will be 4 chi2 and mass values for each source number
    chi2_values = np.array(chi2_values).reshape(-1, 4)
    mass_values = np.array(mass_values).reshape(-1, 4)

    # Save this data to a file
    with open('Data/MDARK_Test/Goodness_of_Fit_{}.csv'.format(ID), 'w') as f:
        f.write('Nsources, chi2_all_signals, chi2_gamma_F, chi2_F_G, chi2_gamma_G, mass_all_signals, mass_gamma_F, mass_F_G, mass_gamma_G\n')
        for i in range(len(Nsources)):
            f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(Nsources[i], chi2_values[i, 0], chi2_values[i, 1], chi2_values[i, 2], chi2_values[i, 3], mass_values[i, 0], mass_values[i, 1], mass_values[i, 2], mass_values[i, 3]))

    # Create a plot of chi2 values and mass values
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # plot the chi2 values, remembering their shape
    for i in range(4):
        ax[0].plot(Nsources, chi2_values[:, i], marker='o', label='{}'.format(labels[i]))
    ax[0].set_xlabel('Number of Sources')
    ax[0].set_ylabel(r'$\chi^2$')
    ax[0].set_title(r'$\chi^2$ vs Number of Sources for Cluster ID {}'.format(ID))
    ax[0].set_yscale('log')
    ax[0].legend()

    for i in range(4):
        ax[1].plot(Nsources, np.abs(mass_values[:, i]), marker='o', label='{}'.format(labels[i]))
    ax[1].set_xlabel('Number of Sources')
    ax[1].set_ylabel(r'$M_{\odot}$')
    ax[1].set_yscale('log')
    ax[1].axhline(np.sum(true_mass), color='black', linestyle='--', label='True Mass')
    ax[1].set_title(r'Mass Estimate vs Number of Sources for Cluster ID {}'.format(ID))
    ax[1].legend()

    fig.tight_layout()
    plt.savefig('Images/MDARK/Goodness_of_Fit/{}.png'.format(ID))
    plt.close


def GOF_correlation_coefficient(test_number):
    ID_file = 'Data/MDARK_Test/Test{}/ID_file_{}.csv'.format(test_number, test_number)
    IDs = pd.read_csv(ID_file)['ID'].values
    true_masses = pd.read_csv(ID_file)[' Mass'].values

    Nsource_low = 100
    Nsource_high = 1000
    Nsources = np.logspace(np.log10(Nsource_low), np.log10(Nsource_high), 20).astype(int)
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    correlation_all_signals = []
    correlation_gamma_f = []
    correlation_f_g = []
    correlation_gamma_g = []

    for Nsource in Nsources:
        true_mass = []
        mass_all_signals = []
        mass_gamma_f = []
        mass_f_g = []
        mass_gamma_g = []

        for ID in IDs:
            file = 'Data/MDARK_Test/Goodness_of_Fit/Goodness_of_Fit_{}.csv'.format(ID)
            data = pd.read_csv(file)
            ID_data = pd.read_csv(ID_file)
            true_mass.append(ID_data[ID_data['ID'] == ID][' Mass'].values[0])

            mass_all_signals.append(data[data['Nsources'] == Nsource][' mass_all_signals'].values[0])
            mass_gamma_f.append(data[data['Nsources'] == Nsource][' mass_gamma_F'].values[0])
            mass_f_g.append(data[data['Nsources'] == Nsource][' mass_F_G'].values[0])
            mass_gamma_g.append(data[data['Nsources'] == Nsource][' mass_gamma_G'].values[0])
        

        correlation_all_signals.append(np.corrcoef(true_mass, mass_all_signals)[0, 1])
        correlation_gamma_f.append(np.corrcoef(true_mass, mass_gamma_f)[0, 1])
        correlation_f_g.append(np.corrcoef(true_mass, mass_f_g)[0, 1])
        correlation_gamma_g.append(np.corrcoef(true_mass, mass_gamma_g)[0, 1])

    # Plot the correlation coefficients
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    correlations = [correlation_all_signals, correlation_gamma_f, correlation_f_g, correlation_gamma_g]
    for i in range(4):
        ax[i].plot(Nsources, correlations[i], marker='o')
        ax[i].set_xlabel('Number of Sources')
        ax[i].set_ylabel('Correlation Coefficient')
        ax[i].set_title('{}'.format(signals[i]))

    fig.tight_layout()
    fig.savefig('Images/MDARK/Correlation_Coefficients_{}.png'.format(test_number))
    plt.show()


def analyze_chi2_file(chi_2_file, result_file, test_num):
    # Given a file with chi2 values, analyze the data
    # and return the number of "good" fits
    chi2_scores = []
    with open(chi_2_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            chi2 = line.split(',')[1:]
            chi2 = [float(val) for val in chi2]
            chi2_scores.extend(chi2)
    
    # Now, let's organize this into a couple of different sets
    # There are 4 chi2 scores for each cluster, based on signal combination
    # We would like to look at the distribution of chi2 scores
    # based on the cluster associated with them, and the signal combination
            
    signal_choices = [
        'All Signals',
        'Shear and Flexion',
        'Flexion and G-Flexion',
        'Shear and G-Flexion'
    ]

    all_signals_chi2 = []
    gamma_f_chi2 = []
    f_g_chi2 = []
    gamma_g_chi2 = []

    # The chi2 scores are organized as [chi2_all_signals, chi2_gamma_f, chi2_f_g, chi2_gamma_g]
    # This repeats in every row for the number of trials run for each cluster
    for i in range(0, len(chi2_scores), 4):
        all_signals_chi2.append(chi2_scores[i])
        gamma_f_chi2.append(chi2_scores[i+1])
        f_g_chi2.append(chi2_scores[i+2])
        gamma_g_chi2.append(chi2_scores[i+3])

    
    # Additionally, look at the chi2 for each cluster (keep all signal combinations)
    cluster_chi2 = []
    for i in range(0, len(chi2_scores), 4):
        cluster_chi2.append(chi2_scores[i:i+4])
    
    # Now plot these distributions
        
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    signal_chi2 = [all_signals_chi2, gamma_f_chi2, f_g_chi2, gamma_g_chi2]

    for i in range(4):
        # Remove outliers
        signal_chi2[i] = np.array(signal_chi2[i])
        # signal_chi2[i] = signal_chi2[i][signal_chi2[i] < 100]
        fancy_hist(signal_chi2[i], bins=1000, ax=ax[i], color='black', histtype='step', label='Median = {}'.format(np.median(signal_chi2[i])))
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel(r'$\chi^2$')
        ax[i].set_ylabel('Number of Clusters')
        # ax[i].set_xlim([0, 100])
        ax[i].vlines(5, 0, 100, color='red', linestyle='--', label='Good Fit Threshold: 5')
        ax[i].legend()
        ax[i].set_title('{}'.format(signal_choices[i]) + ' Chi2 Distribution \n' + r'{}% of Clusters have $\chi^2$ < 5'.format(np.sum(np.array(signal_chi2[i]) < 5) / len(signal_chi2[i]) * 100))
    
    fig.tight_layout()
    plt.savefig('Images/MDARK/Chi2_Distributions_Signal_{}.png'.format(test_num))

    true_masses = pd.read_csv(result_file)[' True Mass'].values
    # Now, let's look at the distribution of chi2 scores for each cluster
    # Just do this as a scatter plot (need 4 cluster_nums per chi2), ie [1,1,1,1,2,2,2,2,...]
    # Associate each signal with a color
    true_masses = np.repeat(true_masses, 30)
    colors = ['red', 'blue', 'green', 'purple']
    colors = colors * 5
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(r'$\chi^2$ Scores for Each Cluster')
    for i in range(4):
        ax.scatter(true_masses, signal_chi2[i], color=colors[i], alpha=0.5, s=5)
    ax.hlines(5, 1e13, 1e15, color='red', linestyle='--', label='Good Fit Threshold: 5')
    ax.set_xlabel(r'Cluster Mass $M_{\odot}$')
    ax.set_ylabel(r'$\chi^2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Create a legend for the colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=signal) for color, signal in zip(colors, signal_choices)]
    ax.legend(handles=handles, title='Signal Combination')
    plt.savefig('Images/MDARK/Chi2_Scores_Cluster_{}.png'.format(test_num))
    plt.show()

# --------------------------------------------
# Main Functions
# --------------------------------------------

if __name__ == '__main__':
    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[0]
    start = time.time()

    test_number = 12
    halos_file = 'MDARK/Halos_{}.MDARK'.format(z_chosen)
    ID_file = 'Data/MDARK_Test/Test{}/ID_file_{}.csv'.format(test_number, test_number)
    result_file = 'Data/MDARK_Test/Test{}/results_{}.csv'.format(test_number, test_number)
    plot_name = 'Images/MDARK/mass_correlations/mass_correlation_{}.png'.format(test_number)
    chi2_file = 'Data/MDARK_Test/Test{}/chi2_scores_{}.csv'.format(test_number, test_number)

    #Check that the ID file exists - if not, create the directory and build the test set
    if not os.path.exists(ID_file):
        os.makedirs('Data/MDARK_Test/Test{}'.format(test_number))
        build_test_set(30, z_chosen, ID_file)

    # run_test_parallel(ID_file, result_file, z_chosen, 30)
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    # build_mass_correlation_plot_errors(result_file, plot_name)
    analyze_chi2_file(chi2_file, result_file, test_number)

    raise ValueError('Done!')

    # I'd like to examine the actual quality of these tests

    ID_file = 'Data/MDARK_Test/Test10/ID_file_10.csv'
    ID_data = pd.read_csv(ID_file)
    ID = ID_data['ID'].sample(n=1).values[0]
    halos = find_halos(int(ID), zs[0])
    halos, sources, xmax = build_lensing_field(halos, zs[0])


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
        ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))


    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10))

    use_flags = [True, True, True]  # Use all signals

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess()
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Initial Guesses', ax=axarr[0,0])

    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Initial Optimization', ax=axarr[0,1], legend=False)

    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax, threshold_distance=5)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Filter', ax=axarr[0,2], legend=False)

    # Step 4: Merge lenses that are too close to each other
    ns = len(sources.x) / (np.pi * xmax**2)
    merger_threshold = 1/np.sqrt(ns)
    lenses.merge_close_lenses(merger_threshold=merger_threshold)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Merging', ax=axarr[1,0], legend=False)

    # Step 5: Iterative elimination
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Iterative Elimination', ax=axarr[1,1], legend=False)

    # Step 6: Final minimization
    lenses.full_minimization(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(xmax, lenses, sources, halos, reducedchi2, 'Final Minimization', ax=axarr[1,2], legend=False)

    # Compute mass
    extent = [-xmax, xmax, -xmax, xmax]
    _, _, kappa = utils.calculate_kappa(lenses, extent, 5)
    mass = utils.calculate_mass(kappa, zs[0], 0.5, 1)

    # Save and show the plot
    fig.suptitle('Lensing Reconstruction of Cluster ID {} \n True Mass: {:.2e} $M_\odot$ \n Inferred Mass: {:.2e} $M_\odot$'.format(ID, np.sum(halos.mass), mass))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
    plt.savefig('Images/MDARK/pipeline_visualization/ID_{}.png'.format(ID))
    plt.show()