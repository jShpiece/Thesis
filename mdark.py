import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import hist as fancy_hist
import pipeline
import utils
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool
import time
import os
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

def _plot_results(halo, true_halo, title, reducedchi2, xmax, ax=None, legend=True, show_mass = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # If the halo is empty, plot the true halo only
    if len(halo.x) == 0:
        x_true, y_true, mass_true = [true_halo.x, true_halo.y, true_halo.mass]
        mass_true_log = np.log10(mass_true)
        true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 50
        ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=0.8, label='True Halos', edgecolors='w', marker='*')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(title + '\n' + 'No Halos Recovered')
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect('equal')
        if legend:
            ax.legend()
        return


    # Extract positions and masses for both sets
    x_true, y_true, mass_true = [true_halo.x, true_halo.y, true_halo.mass]
    x_recon, y_recon, mass_recon = [halo.x, halo.y, halo.mass]

    # Normalize the masses for better visualization in a logarithmic range
    mass_true_log = np.log10(mass_true)
    mass_recon_log = np.log10(np.abs(mass_recon) + 1e-10)  # Add a small value to avoid log(0)

    # Plot true properties with distinct markers and sizes
    true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 200  # Scale sizes
    ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=1, label='True Halos', edgecolors='w', marker='*')

    # Plot reconstructed properties with distinct markers and sizes
    recon_sizes = (mass_recon_log - np.min(mass_recon_log) + 1) * 50  # Scale sizes
    ax.scatter(x_recon, y_recon, s=recon_sizes, c='red', alpha=0.3, label='Recovered Halos', edgecolors='k', marker='o')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.5f}'.format(reducedchi2))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    if legend:
        ax.legend()
    
    if show_mass:
    # Label the mass of each of the lenses (in log scale)
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], '{:.2f}'.format(mass_recon_log[i]), fontsize=12, color='black')


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


def build_mass_correlation_plot(ID_file, file_name, plot_name):
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    # Get the mass and true mass
    # True mass is stored in the ID file
    ID_results = pd.read_csv(ID_file)
    true_mass = ID_results[' Mass'].values
    mass = results[' Mass_all_signals'].values
    mass_gamma_f = results[' Mass_gamma_F'].values
    mass_f_g = results[' Mass_F_G'].values
    mass_gamma_g = results[' Mass_gamma_G'].values

    # Convert masses to floats (currently being read in as strings)
    true_mass = np.array([float(mass) for mass in true_mass])
    mass = np.array([float(mass) for mass in mass])
    mass_gamma_f = np.array([float(mass) for mass in mass_gamma_f])
    mass_f_g = np.array([float(mass) for mass in mass_f_g])
    mass_gamma_g = np.array([float(mass) for mass in mass_gamma_g])

    # NOW replace any Nan values with 0
    mass = np.nan_to_num(mass)
    mass_gamma_f = np.nan_to_num(mass_gamma_f)
    mass_f_g = np.nan_to_num(mass_f_g)
    mass_gamma_g = np.nan_to_num(mass_gamma_g)


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
    # All of the results are floats, so we can convert them to numpy arrays
    # There are 4 columns for mass
    mass_columns = [' Mass_all_signals', ' Mass_gamma_F', ' Mass_F_G', ' Mass_gamma_G']
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']
    # Each ID will appear multiple times, depending on the number of trials run
    # For each ID, collect all of the mass values and calculate the mean and standard deviation
    for i in range(len(IDs)):
        mass_values = []
        for j in range(4):
            mass_values.append(results[results['ID'] == int(IDs[i])][mass_columns[j]].values)
        # Convert strings to floats
        mass_values = [np.array([float(mass) for mass in masses]) for masses in mass_values]
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


def plot_source_signals(sources, test_sources):
    rmse_shear_1 = np.sqrt(np.mean((sources.e1 - test_sources.e1)**2))
    rmse_shear_2 = np.sqrt(np.mean((sources.e2 - test_sources.e2)**2))
    rmse_flexion_1 = np.sqrt(np.mean((sources.f1 - test_sources.f1)**2))
    rmse_flexion_2 = np.sqrt(np.mean((sources.f2 - test_sources.f2)**2))
    rmse_gflexion_1 = np.sqrt(np.mean((sources.g1 - test_sources.g1)**2))
    rmse_gflexion_2 = np.sqrt(np.mean((sources.g2 - test_sources.g2)**2))

    # Create rms histograms for each signal
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    signals = np.array([[rmse_shear_1, rmse_shear_2], [rmse_flexion_1, rmse_flexion_2], [rmse_gflexion_1, rmse_gflexion_2]])
    signal_name = ['Shear', 'Flexion', 'G-Flexion']
    ax = ax.flatten()
    for i in range(3):
        signal = signals[i]
        ax[i].hist(signal[0], bins=100, histtype='step', density=True, label='1', color='blue')
        ax[i].hist(signal[1], bins=100, histtype='step', density=True, label='2', color='red')
        ax[i].set_xlabel('Signal Amplitude')
        ax[i].set_ylabel('Probability Density')
        ax[i].set_title('Signal {}'.format(signal_name[i]))
    fig.tight_layout()
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
            halos[id] = pipeline.Halo(xhalo, yhalo, zhalo, chalo, masshalo, z, np.zeros_like(xhalo))
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

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # Let the centroid be the location of the most massive halo
    # This will be where we expect to see the most light, which
    # means it will be where observations are centered

    largest_halo = np.argmax(halos.mass)
    centroid = [halos.x[largest_halo], halos.y[largest_halo]]
    halos.x -= centroid[0] 
    halos.y -= centroid[1] 

    xmax = np.max((halos.x**2 + halos.y**2)**0.5)
    
    # Don't allow the field of view to be larger than 2 arcminutes - or smaller than 1 arcminute
    xmax = np.min([xmax, 2*60])
    xmax = np.max([xmax, 1*60])

    # Generate a set of background galaxies
    ns = 0.01
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    sources = utils.createSources(halos, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')

    return halos, sources, xmax


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

    np.random.seed()
    noisy_sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    noisy_sources.apply_NFW_lensing(halos)

    # Apply noise
    noisy_sources.apply_noise()

    for signal_choice in signal_choices:
        candidate_lenses, candidate_chi2 = pipeline.fit_lensing_field(noisy_sources, xmax, False, signal_choice, lens_type='NFW')

        mass = np.sum(candidate_lenses.mass)
        candidate_num = len(candidate_lenses.x)

        chi_scores.append(candidate_chi2)
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


def single_realization(args):
    lenses, Nsource, xmax = args

    # Generate a set of background galaxies
    sources = utils.createSources(lenses, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')

    # Now run the pipeline
    candidate_lenses, candidate_chi2 = pipeline.fit_lensing_field(sources, xmax, False, [True, True, True], lens_type='NFW')
    return candidate_lenses, candidate_chi2


def random_realization_test(Ntrials, Nlens, Nsource, mass, xmax, file_name):
    # Run a test with a random realization of lenses and sources
    # This is a test of the pipeline, to see how well it can recover the true lenses

    # Generate the true lenses - place the first lens in the center, the rest place around it
    xl = np.zeros(Nlens)
    yl = np.zeros(Nlens)
    ml = np.zeros(Nlens) + mass # All lenses have the same mass

    for i in range(1, Nlens):
        theta = 2 * np.pi * i / Nlens
        xl[i] = xmax/2 * np.cos(theta)
        yl[i] = xmax/2 * np.sin(theta)

    halos = pipeline.Halo(xl, yl, np.zeros(Nlens), np.zeros(Nlens), ml, 0.194, np.zeros(Nlens))
    halos.calculate_concentration()

    # Run the test in parallel
    tasks = [(halos, Nsource, xmax) for _ in range(Ntrials)]
    with Pool() as pool:
        results = pool.map(single_realization, tasks)
    
    # Unpack the results
    xlens = []
    ylens = []
    log_mass = []
    chi2 = []
    
    for result in results:
        candidate_lenses, candidate_chi2 = result
        for x in candidate_lenses.x:
            xlens.append(x)
        for y in candidate_lenses.y:
            ylens.append(y)
        for mass in candidate_lenses.mass:
            mass = np.abs(mass)
            log_mass.append(np.log10(mass))

        chi2.append(candidate_chi2)

    # Save the results to a file
    # Keep in mind, arrays are not the same length. 
    # We need to pad the arrays with NANs to make them the same length
    max_length = max(len(xlens), len(ylens), len(log_mass), len(chi2))
    xlens = np.pad(xlens, (0, max_length - len(xlens)), 'constant', constant_values=np.nan)
    ylens = np.pad(ylens, (0, max_length - len(ylens)), 'constant', constant_values=np.nan)
    log_mass = np.pad(log_mass, (0, max_length - len(log_mass)), 'constant', constant_values=np.nan)
    chi2 = np.pad(chi2, (0, max_length - len(chi2)), 'constant', constant_values=np.nan)

    results = pd.DataFrame({'x': xlens, 'y': ylens, 'log_mass': log_mass, 'chi2': chi2})
    results.to_csv(file_name, index=False)
    
    return


def interpret_rr_results(results_file, xmax, Nlens, mass):
    # Read in the results file and interpret the results
    results = pd.read_csv(results_file)
    xlens = results['x'].values
    ylens = results['y'].values
    log_mass = results['log_mass'].values
    chi2 = results['chi2'].values
    
    # Check for NANs, if so, remove them
    xlens = xlens[~np.isnan(xlens)]
    ylens = ylens[~np.isnan(ylens)]
    log_mass = log_mass[~np.isnan(log_mass)]

    # Reproduce the true lenses
    xl = np.zeros(Nlens)
    yl = np.zeros(Nlens)
    ml = np.zeros(Nlens) + mass # All lenses have the same mass

    for i in range(1, Nlens):
        theta = 2 * np.pi * i / Nlens
        xl[i] = xmax/2 * np.cos(theta)
        yl[i] = xmax/2 * np.sin(theta)

    if mass == 1e14:
        size = 'large'
    elif mass == 1e13:
        size = 'medium'
    elif mass == 1e12:
        size = 'small'
    else:
        size = 'other'

    # Now plot the results - 4 histograms

    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle('Random Realization Test: Nlens = {}, Mass = {:.2e}'.format(Nlens, mass))
    ax = ax.flatten()
    data = [xlens, ylens, log_mass]
    xlabels = ['x', 'y', 'log_mass']
    titles = ['Lens x Position', 'Lens y Position', 'Lens Mass',]
    true_vals = [xl, yl, np.log10(ml)]
    xrange = [(-xmax, xmax), (-xmax, xmax), (10, 16)]

    for i in range(3):
        fancy_hist(data[i], ax=ax[i], bins='freedman', color='black', histtype='step', density=True)
        if i < 3:
            if len(true_vals[i]) > 1:
                for val in true_vals[i]:
                    ax[i].axvline(val, color='red', linestyle='--', label='True Value', alpha=0.5)
            else:
                ax[i].axvline(true_vals[i], color='red', linestyle='--', label='True Value')
        else:
            ax[i].axvline(1, color='red', linestyle='--', label='True Value')
        ax[i].set_xlim(xrange[i])
        ax[i].set_xlabel(xlabels[i])
        ax[i].set_ylabel('Probability Density')
        ax[i].set_title(titles[i])
        
    fig.tight_layout()
    plt.savefig('Images/NFW_tests/random_realization/Nlens_{}_{}.png'.format(Nlens, size))


# --------------------------------------------
# Debugging Functions
# --------------------------------------------

def map_chi2_space(mass):
    # Create a single halo, a set of sources, and place a test lens everywhere in space, getting the chi2 value for each lens. 
    # Get a sense of how chi2 varies around the halo

    # Create a single halo
    halo = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([5]), np.array([mass]), 0.194, np.array([0]))
    halo.calculate_concentration()
    xmax = 50
    N = 100
    rs = np.sqrt(np.random.random(N)) * xmax  
    thetas = np.random.random(N) * 2 * np.pi
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    sources = pipeline.Source(xs, ys,
                            np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                            np.ones_like(xs) * 0.1, np.ones_like(xs) * 0.01, np.ones_like(xs) * 0.02)
    sources.apply_noise()
    sources.apply_NFW_lensing(halo)
    sources.filter_sources(xmax)

    xgrid = np.linspace(-xmax/10, xmax/10, 50)
    ygrid = np.linspace(-xmax/10, xmax/10, 50)
    mass_grid = np.linspace(np.log10(mass) - 1, np.log10(mass) + 1, 50)

    chi2_values = np.zeros((len(xgrid), len(ygrid), len(mass_grid)))
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            for k in range(len(mass_grid)):
                lens = pipeline.Halo(np.array([xgrid[i]]), np.array([ygrid[j]]), np.array([0]), np.array([5]), np.array([10**mass_grid[k]]), 0.194, np.array([0]))
                lens.calculate_concentration()
                reduced_chi2 = lens.update_chi2_values(sources, [True, True, True])
                chi2_values[i, j, k] = reduced_chi2

    # Clip the chi2 values - we don't care about the really high values
    chi2_values = np.clip(chi2_values, 0, 2)
    # Locate the minimum chi2 value
    min_index = np.unravel_index(np.argmin(chi2_values), chi2_values.shape)
    # Get the true chi2 value
    true_chi2 = halo.update_chi2_values(sources, [True, True, True])
    # Now plot the location of the true lens and the minimum chi2 value, in 3D
    fig = plt.figure()
    fig.suptitle('Chi2 Space around a Single Halo: Mass = {:.2e}'.format(mass))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xgrid[min_index[0]], ygrid[min_index[1]], mass_grid[min_index[2]], color='red', label='Minimum Chi2: {:.3f}'.format(chi2_values[min_index]))
    ax.scatter(0, 0, np.log10(mass), color='blue', label='True Lens: {:.3f}'.format(true_chi2))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('log(Mass)')
    ax.legend()
    # Choose a default view
    ax.view_init(elev=30, azim=30)
    plt.savefig('Images/chi2_space_{}.png'.format(np.round(np.log10(mass), 2)))
    plt.show()


def visualize_fits(ID_file):
    IDs = pd.read_csv(ID_file)['ID'].values
    # Make sure the IDs are integers
    IDs = [int(ID) for ID in IDs]
    # Just grab the last 5 IDs
    IDs = IDs[-5:]
    zs = [0.194, 0.391, 0.586, 0.782, 0.977]
    start = time.time()
    halos = find_halos(IDs, zs[0])
    stop = time.time()
    print('Time taken to load halos: {}'.format(stop - start))

    counter = 0
    for ID in IDs:
        start = time.time()
        halo = halos[ID]
        indices = np.argsort(halo.mass)[::-1][:3]  # Simplify for tests - lets just look at the 3 largest halos
        halo.x = halo.x[indices]
        halo.y = halo.y[indices]
        halo.mass = halo.mass[indices]
        halo.z = halo.z[indices]
        halo.concentration = halo.concentration[indices]
        
        _, sources, xmax = build_lensing_field(halo, zs[0])

        use_flags = [True, True, True]  # Use all signals

        # Run the pipeline
        lenses, reducedchi2 = pipeline.fit_lensing_field(sources, xmax, False, use_flags, lens_type='NFW')

        # Compute mass
        try:
            mass = np.sum(lenses.mass)
        except AttributeError:
            print('Mass not computed')
            mass = 0

        # Save and show the plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        title = 'Cluster ID: {} \n True Mass: {:.2e} \n Inferred Mass: {:.2e}'.format(ID, halo.mass.sum(), mass)
        _plot_results(lenses, halo, title, reducedchi2, xmax, ax)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images/MDARK/pipeline_visualization/Halo_Fit_{}.png'.format(ID))
        plt.close()
        stop = time.time()
        print('Time taken for cluster {}: {}'.format(counter, stop - start))
        counter += 1

    return


def visualize_initial_optimization():
    # Create a single halo, a set of initial guesses around that halo, and see how the optimization step does
    rmax = 30
    N = 30
    
    # Create a single halo
    halo = pipeline.Halo(np.array([0, -10]), np.array([0,-10]), np.array([0,0]), np.array([5,2]), np.array([1e14,1e14]), 0.194, np.array([0,0]))
    # halo = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([5]), np.array([1e14]), 0.194, np.array([0]))
    halo.calculate_concentration()

    # Create a set of guesses
    # Distribute the guesses in a spiral pattern, moving outwards

    r = np.linspace(1, rmax, N)
    theta = np.linspace(0, 2*np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    mass = 10**np.random.normal(14, 1, N) # A little less of a cheat

    # Create a halo object
    lenses = pipeline.Halo(x, y, np.zeros_like(x), np.zeros_like(x), mass, 0.194, np.zeros_like(x))
    lenses.calculate_concentration()
    # Clone the halo object for plotting
    lenses_clone = copy.deepcopy(lenses)

    # Create background sources, which we need in order to perform the optimization
    # Distribute the sources in a wider spiral pattern

    r_s = 2 * r
    xs = r_s * np.cos(theta) + np.random.normal(0, 0.1, N)
    ys = r_s * np.sin(theta) + np.random.normal(0, 0.1, N) # Add some noise to the source positions

    sources = pipeline.Source(xs, ys,
                                np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), 
                                np.ones_like(xs) * 0.1, np.ones_like(xs) * 0.01, np.ones_like(xs) * 0.02)
    sources.apply_noise()
    sources.apply_NFW_lensing(halo)

    
    # Optimize 
    points = lenses.optimize_lens_positions(sources, [True, True, True])
    print('Optimization complete...')
    def plot_optimization_path(points, true_value):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Convert points to numpy array for easier plotting
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker='o', label='Optimization Path')
        # Highlight the first and last points
        ax.scatter(points[0, 0], points[0, 1], points[0, 2], color='g', label='Initial Guess', s=100)
        ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color='b', label='Final Guess', s=100)
        ax.scatter(*true_value, color='r', label='True Value', s=100)  # Plot true value
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.legend()
        plt.show()
    
    # print(points)
    # plot_optimization_path(points, [0, 0, np.log10(1e14)])

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(lenses_clone.x, lenses_clone.y, color='black', label='Initial Guesses', alpha=0.5)
    ax.scatter(halo.x, halo.y, color='red', label='True Halo', marker='*', s=100)
    ax.scatter(lenses.x, lenses.y, color='blue', label='Optimized Guesses', marker='x', alpha=0.75)
    # Draw an arrow from the source to the lens
    for i in range(len(lenses.x)):
        ax.arrow(lenses_clone.x[i], lenses_clone.y[i], lenses.x[i] - lenses_clone.x[i], lenses.y[i] - lenses_clone.y[i], head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, rmax])
    plt.tight_layout()
    # Draw a circle that contains half of our optimization guesses
    # This is a rough estimate of the distance at which the optimization stops working
    r_found = (lenses.x**2 + lenses.y**2)**0.5 
    halo_found = r_found < 5
    print("{}% of the lenses were brought within 5 arcseconds of a halo".format(np.sum(halo_found) / N * 100))
    r_found = np.sort(r_found)
    r_found = r[int(len(r_found)/2)]
    circle = plt.Circle((0, 0), 5, color='red', fill=False, label='Detection Threshold', linestyle='--')
    circle2 = plt.Circle((-10, -10), 5, color='red', fill=False, linestyle='--')
    ax.add_artist(circle)
    ax.add_artist(circle2)

    ax.legend(loc='best')
    ax.set_title('Evaluation of Initial Optimization \n Of {} lenses, {} were brought within 5'' of a halo'.format(N, np.sum(halo_found)), fontsize=10)
    plt.savefig('Images/initial_optimization_2lenses.png')
    plt.show()


def visualize_final_minimization(halo_mass, use_noise=True, offset=0):
    # Create a simple lensing field and test the pipeline on it
    Nlens = 1
    Nsource = 100
    xmax = 50

    # Create a set of lenses
    x = np.array([0])
    y = np.array([0])
    mass = np.ones(Nlens) * halo_mass

    halos = pipeline.Halo(x, y, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
    halos.calculate_concentration()

    # Create a set of sources (in grid form)
    n = int(np.sqrt(Nsource))
    xs = np.linspace(-xmax, xmax, n)
    ys = np.linspace(-xmax, xmax, n)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()
    Nsource = len(xs)
    sig_s = np.ones(Nsource) * 0.1
    sig_f = np.ones(Nsource) * 0.01
    sig_g = np.ones(Nsource) * 0.02
    sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    if use_noise:
        sources.apply_noise()
    sources.apply_NFW_lensing(halos)
    sources.filter_sources()

    # Now, create a set of lenses at the correct positions, with a range of masses
    # We will then test the final minimization step on these lenses, and see the resultant mass
    input_masses = np.logspace(11, 16, 100)
    output_masses = np.zeros_like(input_masses)
    chi2_vals = np.zeros_like(input_masses)
    errors = np.zeros_like(input_masses)
    for i in range(len(input_masses)):
        mass = np.ones(Nlens) * input_masses[i]
        lenses = pipeline.Halo(x+offset, y+offset, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
        lenses.calculate_concentration()
        chi2_reduced = lenses.update_chi2_values(sources, [True, True, True])
        lenses.full_minimization(sources, [True, True, True])
        output_masses[i] = np.sum(lenses.mass)
        chi2_vals[i] = chi2_reduced
        errors[i] = np.abs(output_masses[i] - halo_mass) / halo_mass * 100

    # Make labels for the plot
    if halo_mass == 1e14:
        size = 'large'
    elif halo_mass == 1e13:
        size = 'medium'
    elif halo_mass == 1e12:
        size = 'small'
    else:
        size = 'other'
    
    if use_noise:
        noise = 'noisy'
    else:
        noise = 'noiseless'

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    fig.suptitle('Testing the Final Minimization: {}'.format(noise))
    ax[0].plot(input_masses, output_masses, label='Recovered Mass')
    ax[0].axhline(halo_mass, color='red', linestyle='--', label='True Mass')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('True Mass')
    ax[0].set_ylabel('Recovered Mass')
    ax[0].set_title(r'Average Mass = {:.2e} $\pm$ {:.2e}'.format(np.mean(output_masses), np.std(output_masses)))
    ax[0].legend()

    # Plot the error
    ax[1].plot(input_masses, errors, label='Error')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('Input Mass')
    ax[1].set_ylabel('Error (%)')
    ax[1].set_title('Error Space')
    ax[1].legend()

    # Plot the chi2 values
    ax[2].plot(input_masses, chi2_vals, label='Reduced Chi2')
    ax[2].axvline(halo_mass, color='red', linestyle='--', label='True Mass')
    ax[2].set_xscale('log')
    ax[2].set_xlabel('Input Mass')
    ax[2].set_ylabel('Reduced Chi2')
    ax[2].set_title('Chi2 Space')
    ax[2].legend()

    plt.tight_layout()

    plt.savefig('Images/NFW_tests/final_opt/{}_{}_offset_{}.png'.format(size, noise, offset))
    print('Finished test for halo mass: {}'.format(halo_mass))
    plt.close()


def visualize_final_minimization_path(halo_mass):
    # Create a simple lensing field and test the pipeline on it
    Nlens = 1
    Nsource = 100
    xmax = 50

    # Create a set of lenses
    x = np.array([0])
    y = np.array([0])
    mass = np.ones(Nlens) * halo_mass

    halos = pipeline.Halo(x, y, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
    halos.calculate_concentration()

    # Create a set of sources (in grid form)
    n = int(np.sqrt(Nsource))
    xs = np.linspace(-xmax, xmax, n)
    ys = np.linspace(-xmax, xmax, n)
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.flatten()
    ys = ys.flatten()
    Nsource = len(xs)
    sig_s = np.ones(Nsource) * 0.1
    sig_f = np.ones(Nsource) * 0.01
    sig_g = np.ones(Nsource) * 0.02
    sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    sources.apply_noise()
    sources.apply_NFW_lensing(halos)
    sources.filter_sources()

    # Create a single lens at the correct position, with x% of the true mass
    A = np.random.uniform(0.1, 10)
    lens = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([A*halo_mass]), 0.194, np.array([0]))
    lens.calculate_concentration()
    true_chi2 = lens.update_chi2_values(sources, [True, True, True])

    # Perform the final minimization
    path = lens.full_minimization(sources, [True, True, True])
    # Turn path into a 1d array
    path = np.array(path).flatten()
    chi2 = np.empty(len(path))
    for i in range(len(path)):
        sub_lens = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([0]), np.array([10**path[i]]), 0.194, np.array([0]))
        sub_lens.calculate_concentration()
        chi2[i] = sub_lens.update_chi2_values(sources, [True, True, True])

    # Plot the path in mass space and chi2 space
    # Right now the path is just a list of masses
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Final Minimization Path')
    ax[0].plot(10**path, label='Mass Path')
    ax[0].axhline(halo_mass, color='red', linestyle='--', label='True Mass')
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Mass')
    ax[0].set_title('Mass Path')
    ax[0].set_yscale('log')
    ax[0].legend()

    ax[1].plot(10**path, chi2, label='Chi2 Path')
    ax[1].scatter(halo_mass, true_chi2, color='red', label='True Mass')
    ax[1].set_xlabel('Mass')
    ax[1].set_ylabel('Reduced Chi2')
    ax[1].set_title('Chi2 Path')
    ax[1].set_xscale('log') 
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('Images/NFW_tests/final_opt/{}_path.png'.format(np.log10(halo_mass)))
    plt.show()


def simple_nfw_test(Nlens, Nsource, xmax, halo_mass, use_noise=True):
    # Create a simple lensing field and test the pipeline on it
    start = time.time()
    # Create a set of lenses
    if Nlens == 1:
        x = np.array([0])
        y = np.array([0])
    elif Nlens == 2:
        x = np.linspace(-xmax/2, xmax/2, Nlens)
        y = np.array([0, 0])
    else:
        x = np.linspace(-xmax/2, xmax/2, Nlens)
        y = np.linspace(-xmax/2, xmax/2, Nlens)
    mass = np.ones(Nlens) * halo_mass

    halos = pipeline.Halo(x, y, np.zeros_like(x), np.zeros(Nlens), mass, 0.194, np.zeros_like(x))
    halos.calculate_concentration()

    if Nsource == 1:
        xs = np.random.uniform(-xmax, xmax, Nsource)
        ys = np.random.uniform(-xmax, xmax, Nsource)
    else:
        n = int(np.sqrt(Nsource))
        x = np.linspace(-xmax, xmax, n)
        y = np.linspace(-xmax, xmax, n)
        xs, ys = np.meshgrid(x, y)
        xs = xs.flatten()
        ys = ys.flatten()
        Nsource = len(xs)
    
    sig_s = np.ones(Nsource) * 0.1
    sig_f = np.ones(Nsource) * 0.01
    sig_g = np.ones(Nsource) * 0.02
    sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    if use_noise:
        sources.apply_noise()
        noisy = 'noisy'
    else:
        noisy = 'noiseless'
    sources.apply_NFW_lensing(halos)
    sources.filter_sources()

    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

    use_flags = [True, True, True]  # Use all signals

    # Step 1: Generate initial list of lenses from source guesses
    lenses = sources.generate_initial_guess(z_l = 0.194, lens_type='NFW')
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Initial Guesses', reducedchi2, xmax, ax=axarr[0,0], legend=True)

    # Step 2: Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Initial Optimization', reducedchi2, xmax, ax=axarr[0,1], legend=False)

    # Step 3: Filter out lenses that are too far from the source population
    lenses.filter_lens_positions(sources, xmax)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Filtering', reducedchi2, xmax, ax=axarr[0,2], legend=False)

    # Step 4: Iterative elimination
    lenses.iterative_elimination(sources, reducedchi2, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Lens Number Selection', reducedchi2, xmax, ax=axarr[1,0], legend=False)

    # Step 5: Merge lenses that are too close to each other
    ns = len(sources.x) / (np.pi * xmax**2)
    merger_threshold = (1/np.sqrt(ns))
    lenses.merge_close_lenses(merger_threshold=merger_threshold)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Merging', reducedchi2, xmax, ax=axarr[1,1], legend=False)

    # Step 6: Final minimization
    lenses.full_minimization(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Final Minimization', reducedchi2, xmax, ax=axarr[1,2], legend=False, show_mass=True)


    fig.suptitle('True Mass: {:.2e} $M_\odot$ \n Recovered Mass: {:.2e} $M_\odot$'.format(np.sum(halos.mass), np.sum(lenses.mass)))
    if halo_mass == 1e14:
        size = 'large'
    elif halo_mass == 1e13:
        size = 'medium'
    elif halo_mass == 1e12:
        size = 'small'
    else:
        size = 'other'
    plot_name = 'Images/NFW_tests/standard_tests/{}_Nlens_{}_{}.png'.format(size,Nlens,noisy)
    plt.savefig(plot_name)
    stop = time.time()

    print('Finished test: {} seconds'.format(stop - start))
    return


# --------------------------------------------
# Main Functions
# --------------------------------------------

def run_simple_tests():
    masses = [1e14, 1e13, 1e12]
    lens_numbers = [1,2]
    noise_use = [True, False]

    for mass in masses:
        for Nlens in lens_numbers:
            for noise in noise_use:
                simple_nfw_test(Nlens, 100, 50, mass, use_noise=noise)


def run_rr_tests():
    N_lens = [1,2]
    Nsource = 100
    Ntrials = 100
    xmax = 50
    masses = [1e14, 1e13, 1e12]
    for Nlens in N_lens:
        for mass in masses:           
            start = time.time()
            size = 'large' if mass == 1e14 else 'medium' if mass == 1e13 else 'small' if mass == 1e12 else 'other'
            file_name = 'Images/NFW_tests/random_realization/Nlens_{}_Nsource_{}_size_{}.txt'.format(Nlens, Nsource, size)
            # random_realization_test(Ntrials, Nlens, Nsource, mass, xmax, file_name)
            interpret_rr_results(file_name, xmax, Nlens, mass)
            stop = time.time()
            print('Time taken for {} Nlens = {}: {}'.format(size, Nlens, stop - start))


def process_md_set():
    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[0]
    start = time.time()
    Ntrials = 1 # Number of trials to run for each cluster in the test set

    test_number = 15
    test_dir = 'Data/MDARK_Test/Test{}'.format(test_number)
    halos_file = 'MDARK/Halos_{}.MDARK'.format(z_chosen)
    ID_file = test_dir + '/ID_file_{}.csv'.format(test_number)
    result_file = test_dir + '/results_{}.csv'.format(test_number)
    plot_name = 'Images/MDARK/mass_correlations/mass_correlation_{}.png'.format(test_number)

    #Check that the ID file exists - if not, create the directory and build the test set
    if not os.path.exists(ID_file):
        os.makedirs('Data/MDARK_Test/Test{}'.format(test_number))
        build_test_set(30, z_chosen, ID_file)

    run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot(ID_file, result_file, plot_name)
    # build_chi2_plot(result_file, ID_file, test_number) # No longer of interest


def estimate_error_in_minimization(true_mass):
    halo = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([5]), np.array([true_mass]), 0.194, np.array([0]))
    halo.calculate_concentration()
    xmax = 50
    Nsource = 100
    sources = pipeline.Source(np.random.uniform(-xmax, xmax, Nsource), np.random.uniform(-xmax, xmax, Nsource),
                            np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource), np.zeros(Nsource),
                            np.ones(Nsource) * 0.1, np.ones(Nsource) * 0.01, np.ones(Nsource) * 0.02)
    sources.apply_noise()
    sources.apply_NFW_lensing(halo)
    sources.filter_sources(xmax)

    # Now create a single lens at the correct position, with a random starting mass
    # Actually, let's offset the lens position by a small amount, to account for the fact that our pipeline will not get the exact position
    xl = np.random.normal(0, 0)
    yl = np.random.normal(0, 0)
    starting_mass = 10**np.random.uniform(11, 16)
    lens = pipeline.Halo(np.array(xl), np.array(yl), np.array([0]), np.array([0]), np.array(starting_mass), 0.194, np.array([0]))
    lens.calculate_concentration()
    true_chi2 = halo.update_chi2_values(sources, [True, True, True])
    
    # Perform the final minimization
    lens.full_minimization(sources, [True, True, True])
    reduced_chi2 = lens.update_chi2_values(sources, [True, True, True])

    return reduced_chi2, true_chi2, starting_mass, lens.mass[0]


if __name__ == '__main__':
    # Hypothesis - I am not actually getting a bad normalization - the mass overestimates come from false positives
    # The pipeline is not actually overestimating the mass, but rather detecting multiple halos
    # How can I test this hypothesis?
    # First idea - try final minimization one halo at a time, instead of all halos at once
    # Reasoning: the pipeline should drive false positives towards a mass of 0, but isn't. Could this be because I'm fitting the good halos at the same time as the bad, 
    # which causes bad fits to be overlooked? If I fit the bad halo on its own, will it be driven to a mass of 0?
    run_simple_tests()

    raise ValueError('This script is not meant to be run directly. Please run the main.py script instead.')
    N = 1000
    masses = [1e14, 1e13, 1e12]

    for mass in masses:
        chi2, true_chi2, starting_mass, final_mass = np.empty(N), np.empty(N), np.empty(N), np.empty(N)

        for i in range(N):
            chi2[i], true_chi2[i], starting_mass[i], final_mass[i] = estimate_error_in_minimization(mass)

        # Plot the results
        fig, ax = plt.subplots(1,2, figsize=(15, 10))
        fig.suptitle('Minimization Errors for Mass = {:.2e}'.format(mass))

        # Just need mass and error
        # Need to plot a histogram of the final mass - but the mass is log-normal, so we need to plot it on a log scale
        fancy_hist(np.log10(final_mass), ax=ax[0], bins='freedman', color='black', histtype='step', density=True)
        avg_mass = np.median(np.log10(final_mass))
        mean_mass = np.mean(np.log10(final_mass))
        ax[0].set_title('Final Mass')
        ax[0].set_xlabel('log(Mass)')
        ax[0].set_ylabel('Probability Density')
        ax[0].legend()  

        # Now calculate the percent error and plot a single histogram of that
        ratio = final_mass / mass
        max_ratio = np.min([np.max(ratio), 50])
        ratio = ratio[ratio < max_ratio]
        fancy_hist(ratio, ax=ax[1], bins='freedman', color='black', histtype='step', density=True)
        # Identify the mean and std of the error
        mean_error = np.mean(ratio)
        median_error = np.median(ratio)
        std_error = np.std(ratio)
        ax[1].axvline(mean_error, color='red', linestyle='--', label='Mean ratio = {:.2f}'.format(mean_error))
        ax[1].set_title('Ratio of Final Mass to True Mass')
        ax[1].set_xlabel(r'$\frac{M_f}{M_0}$')
        ax[1].set_ylabel('Probability Density')
        ax[1].legend()

        plt.tight_layout()
        plt.savefig('Images/NFW_tests/final_opt/minimization_errors_{}.png'.format(np.log10(mass)))