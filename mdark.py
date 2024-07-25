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

def _plot_results(halo, true_halo, title, reducedchi2, xmax, ax=None, legend=True):
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
    true_sizes = (mass_true_log - np.min(mass_true_log) + 1) * 100  # Scale sizes
    ax.scatter(x_true, y_true, s=true_sizes, c='blue', alpha=0.8, label='True Halos', edgecolors='w', marker='*')

    # Plot reconstructed properties with distinct markers and sizes
    recon_sizes = (mass_recon_log - np.min(mass_recon_log) + 1) * 50  # Scale sizes
    ax.scatter(x_recon, y_recon, s=recon_sizes, c='red', alpha=0.3, label='Recovered Halos', edgecolors='k', marker='o')

    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title + '\n' + r' $\chi_\nu^2$ = {:.2f}'.format(reducedchi2))
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Add legend
    if legend:
        ax.legend()


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

    lenses = halos # Placeholder for the lenses

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # Let the centroid be the location of the most massive halo
    # This will be where we expect to see the most light, which
    # means it will be where observations are centered

    largest_halo = np.argmax(halos.mass)
    centroid = [halos.x[largest_halo], halos.y[largest_halo]]
    lenses.x -= centroid[0] 
    lenses.y -= centroid[1] 

    xmax = np.max((lenses.x**2 + lenses.y**2)**0.5)
    
    # Don't allow the field of view to be larger than 2 arcminutes - or smaller than 1 arcminute
    xmax = np.min([xmax, 3*60])
    xmax = np.max([xmax, 1*60])

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
    lenses = pipeline.Halo(xl, yl, np.zeros_like(xl), np.zeros_like(xl), mass, z, np.zeros_like(xl))
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


def simple_nfw_test(Nlens, Nsource, xmax):
    # Create a simple lensing field and test the pipeline on it

    # Create a set of lenses
    x = np.random.uniform(-xmax, xmax, Nlens)
    y = np.random.uniform(-xmax, xmax, Nlens)
    mass = np.random.uniform(1e13, 1e15, Nlens)
    concentration = np.random.uniform(1, 10, Nlens)
    # Set the first lens to be at the center
    x[0] = 0
    y[0] = 0
    mass[0] = 1e14

    halos = pipeline.Halo(x, y, np.zeros_like(x), concentration, mass, 0.194, np.zeros_like(x))

    # Create a set of sources
    # xs = np.random.uniform(-xmax, xmax, Nsource)
    # ys = np.random.uniform(-xmax, xmax, Nsource)
    # Distribute sources uniformly in a grid (if Nsource is 1, place it randomly)
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
        # Update Nsource
        Nsource = len(xs)

    sig_s = np.ones(Nsource) * 0.1
    sig_f = np.ones(Nsource) * 0.01
    sig_g = np.ones(Nsource) * 0.02
    sources = pipeline.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), sig_s, sig_f, sig_g)
    sources.apply_noise()
    sources.apply_NFW_lensing(halos)
    sources.filter_sources()

    # Now run the pipeline (and plot it)

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
    # Add the true lenses back into the candidates
    # lenses.x = np.concatenate((lenses.x, halos.x))
    # lenses.y = np.concatenate((lenses.y, halos.y))
    # lenses.mass = np.concatenate((lenses.mass, halos.mass))
    # lenses.concentration = np.concatenate((lenses.concentration, halos.concentration))
    # lenses.calculate_concentration()
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
    _plot_results(lenses, halos, 'Final Minimization', reducedchi2, xmax, ax=axarr[1,2], legend=False)


    fig.suptitle('True Mass: {:.2e} $M_\odot$ \n Recovered Mass: {:.2e} $M_\odot$'.format(np.sum(halos.mass), np.sum(lenses.mass)))
    plt.savefig('Images/test_cluster_Nlens_{}_Nsource_{}.png'.format(Nlens, Nsource))

    print('Finished test...')


# --------------------------------------------
# Main Functions
# --------------------------------------------

def visualize_fits(ID_file):
    IDs = pd.read_csv(ID_file)['ID'].values
    # Make sure the IDs are integers
    IDs = [int(ID) for ID in IDs]
    # Just grab the first N
    IDs = IDs[:5]
    zs = [0.194, 0.391, 0.586, 0.782, 0.977]
    start = time.time()
    halos = find_halos(IDs, zs[0])
    stop = time.time()
    print('Time taken to load halos: {}'.format(stop - start))

    # Create a results file
    # This will tell me the following things
    # For each of the three lenses (primary, lens2, lens3) - did we find them? and what was the mass?
    # For the overall system - what was the total mass?
    # How many candidate lenses did we find?
    # What was the chi2 value?
    # What was the total mass of the system, vs the true mass?

    # Create the file

    located_primary = []
    located_lens2 = []
    located_lens3 = []
    total_mass = []
    candidate_number = []
    chi_scores = []
    true_mass = []

    primary_mass = []
    lens2_mass = []
    lens3_mass = []

    primary_true_mass = []
    lens2_true_mass = []
    lens3_true_mass = []

    counter = 0
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

        # Arrange a plot with 6 subplots in 2 rows
        fig, axarr = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

        use_flags = [True, True, True]  # Use all signals

        # Step 1: Generate initial list of lenses from source guesses
        lenses = sources.generate_initial_guess(z_l = zs[0], lens_type='NFW')
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Initial Guesses', reducedchi2, xmax, ax=axarr[0,0], legend=True)
        
        # Step 2: Optimize guesses with local minimization
        lenses.optimize_lens_positions(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Initial Optimization', reducedchi2, xmax, ax=axarr[0,1], legend=False)
        
        # Step 3: Filter out lenses that are too far from the source population
        lenses.filter_lens_positions(sources, xmax)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Filtering', reducedchi2, xmax, ax=axarr[0,2], legend=False)
        
        # Step 4: Iterative elimination
        # Add the true lenses back into the candidates
        #lenses.x = np.concatenate((lenses.x, halo.x))
        #lenses.y = np.concatenate((lenses.y, halo.y))
        #lenses.mass = np.concatenate((lenses.mass, halo.mass))
        #lenses.concentration = np.concatenate((lenses.concentration, halo.concentration))
        
        lenses.iterative_elimination(sources, reducedchi2, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Iterative Elimination', reducedchi2, xmax, ax=axarr[1,0], legend=False)

        # Step 5: Merge lenses that are too close to each other
        ns = len(sources.x) / (np.pi * xmax**2)
        merger_threshold = (1/np.sqrt(ns))
        lenses.merge_close_lenses(merger_threshold=merger_threshold)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Merging', reducedchi2, xmax, ax=axarr[1,1], legend=False)

        # Step 6: Final minimization
        lenses.full_minimization(sources, use_flags)
        reducedchi2 = lenses.update_chi2_values(sources, use_flags)
        _plot_results(lenses, halo, 'Final Minimization', reducedchi2, xmax, ax=axarr[1,2], legend=False)

        # Compute mass
        try:
            mass = np.sum(lenses.mass)
        except AttributeError:
            print('Mass not computed')
            mass = 0

        # Save and show the plot
        fig.suptitle('Lensing Reconstruction of Cluster ID {} \n True Mass: {:.2e} $M_\odot$ \n Inferred Mass: {:.2e} $M_\odot$'.format(ID, np.sum(halo.mass), mass))

        # Save the results to a file
        primary_found = np.min((lenses.x - halo.x[0])**2 + (lenses.y - halo.y[0])**2) < 10
        lens2_found = np.min((lenses.x - halo.x[1])**2 + (lenses.y - halo.y[1])**2) < 10
        lens3_found = np.min((lenses.x - halo.x[2])**2 + (lenses.y - halo.y[2])**2) < 10

        located_primary.append(primary_found)
        located_lens2.append(lens2_found)
        located_lens3.append(lens3_found)
        total_mass.append(mass)
        candidate_number.append(len(lenses.x))
        chi_scores.append(reducedchi2)

        true_mass.append(np.sum(halo.mass))

        if primary_found:
            primary_index = np.argmin((lenses.x - halo.x[0])**2 + (lenses.y - halo.y[0])**2)
            primary_mass.append(lenses.mass[primary_index])
        else:
            primary_mass.append(0)
        if lens2_found:
            lens2_index = np.argmin((lenses.x - halo.x[1])**2 + (lenses.y - halo.y[1])**2)
            lens2_mass.append(lenses.mass[lens2_index])
        else:
            lens2_mass.append(0)
        if lens3_found:
            lens3_index = np.argmin((lenses.x - halo.x[2])**2 + (lenses.y - halo.y[2])**2)
            lens3_mass.append(lenses.mass[lens3_index])
        else:
            lens3_mass.append(0)
        
        primary_true_mass.append(halo.mass[0])
        lens2_true_mass.append(halo.mass[1])
        lens3_true_mass.append(halo.mass[2])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for better visualization
        plt.savefig('Images/MDARK/pipeline_visualization/Halo_Fit_{}.png'.format(ID))
        plt.close()
        global_stop = time.time()
        print('Time taken for cluster {}: {}'.format(counter, global_stop - global_start))
        counter += 1

    # Save the results to a file
    with open('Images/MDARK/pipeline_visualization/results.csv', 'w') as f:
        f.write('ID, Located Primary, Located Lens 2, Located Lens 3, Total Mass, Candidate Number, Chi2, True Mass, Primary Mass, Lens 2 Mass, Lens 3 Mass, Primary True Mass, Lens 2 True Mass, Lens 3 True Mass\n')
        for i in range(len(IDs)):
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{} \n'.format(IDs[i], 
                                                                located_primary[i], located_lens2[i], located_lens3[i], 
                                                                total_mass[i], candidate_number[i], chi_scores[i], 
                                                                true_mass[i], primary_mass[i], lens2_mass[i], 
                                                                lens3_mass[i], primary_true_mass[i], lens2_true_mass[i], 
                                                                lens3_true_mass[i]))
    
    # Plot the results
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()

    # Plot the total mass
    ax[0].scatter(true_mass, total_mass, color='black')
    ax[0].plot(true_mass, true_mass, color='red', linestyle='--')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('True Mass [$M_\odot$]')
    ax[0].set_ylabel('Inferred Mass [$M_\odot$]')
    ax[0].set_title('Total Mass Inference')
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_xlim([1e13, 1e15])
    ax[0].set_ylim([1e13, 1e15])

    # Plot correlation between true and inferred mass for each lens
    ax[1].scatter(primary_true_mass, primary_mass, color='black')
    ax[1].plot(primary_true_mass, primary_true_mass, color='red', linestyle='--')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('True Mass [$M_\odot$]')
    ax[1].set_ylabel('Inferred Mass [$M_\odot$]')
    ax[1].set_title('Primary Lens Mass Inference')
    ax[1].set_aspect('equal', adjustable='box')

    ax[2].scatter(lens2_true_mass, lens2_mass, color='black')
    ax[2].plot(lens2_true_mass, lens2_true_mass, color='red', linestyle='--')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('True Mass [$M_\odot$]')
    ax[2].set_ylabel('Inferred Mass [$M_\odot$]')
    ax[2].set_title('Lens 2 Mass Inference')
    ax[2].set_aspect('equal', adjustable='box')

    ax[3].scatter(lens3_true_mass, lens3_mass, color='black')
    ax[3].plot(lens3_true_mass, lens3_true_mass, color='red', linestyle='--')
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('True Mass [$M_\odot$]')
    ax[3].set_ylabel('Inferred Mass [$M_\odot$]')
    ax[3].set_title('Lens 3 Mass Inference')
    ax[3].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig('Images/MDARK/pipeline_visualization/Mass_Inference.png')
    plt.show()

    return


def test_initial_steps(lens_type='NFW'):
    # Pick a cluster from the MDARK data, select the three largest halos, and test the NFW lensing
    ID = 11426787731
    zs = 0.194
    use_flags = [True, True, True]

    xmax = 20 # Default xmax

    # Build the halos
    if lens_type == 'NFW':
        halos = find_halos([ID], zs)
        halo = halos[ID]
        # Correct positioning
        halo.project_to_2D()
        d = cosmo.angular_diameter_distance(zs).to(u.meter).value
        halo.x *= (3.086 * 10**22 / d) * 206265
        halo.y *= (3.086 * 10**22 / d) * 206265

        # Center the lenses at (0, 0)
        # This is a necessary step for the pipeline
        # Let the centroid be the location of the most massive halo
        # This will be where we expect to see the most ligwht, which
        # means it will be where observations are centered

        largest_halo = np.argmax(halo.mass)
        centroid = [halo.x[largest_halo], halo.y[largest_halo]]
        halo.x -= centroid[0] 
        halo.y -= centroid[1] 
    
        indices = np.argsort(halo.mass)[::-1][:3]
        halo.x = halo.x[indices]
        halo.y = halo.y[indices]
        halo.mass = halo.mass[indices]
        halo.z = halo.z[indices]
        halo.concentration = halo.concentration[indices]

        xmax = np.max((halo.x**2 + halo.y**2)**0.5)
        # Don't allow the field of view to be larger than 2 arcminutes - or smaller than 1 arcminute
        xmax = np.min([xmax, 3*60])
        xmax = np.max([xmax, 1*60])
        xmax = int(xmax)

        # Rename for consistency
        true_lenses = halo
    elif lens_type == 'SIS':
        # Create 3 lenses
        xl = np.array([0])
        yl = np.array([0])
        # Add two randomly placed lenses
        xl = np.concatenate((xl, np.random.uniform(-10, 10, 2)))
        yl = np.concatenate((yl, np.random.uniform(-10, 10, 2)))
        tE = [1, 1, 1]
        lenses = pipeline.Lens(xl, yl, tE, np.zeros_like(xl))
        true_lenses = lenses

    # Lay sources down in a uniform grid
    xgrid = np.linspace(-xmax, xmax, 2*xmax)
    ygrid = np.linspace(-xmax, xmax, 2*xmax)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)

    # First, lets run each source through the pipeline on its own
    print('Starting test...')
    start = time.time()

    # Can easily vectorize the first two steps
    sources = pipeline.Source(xgrid.flatten(), ygrid.flatten(),
                                np.zeros_like(xgrid.flatten()), np.zeros_like(xgrid.flatten()), 
                                np.zeros_like(xgrid.flatten()), np.zeros_like(xgrid.flatten()), 
                                np.zeros_like(xgrid.flatten()), np.zeros_like(xgrid.flatten()), 
                                np.ones_like(xgrid.flatten()) * 0.1, np.ones_like(xgrid.flatten()) * 0.01, np.ones_like(xgrid.flatten()) * 0.02)
    sources.apply_noise()

    # Apply lensing and generate initial guesses
    if lens_type == 'NFW':
        sources.apply_NFW_lensing(true_lenses)
        lenses = sources.generate_initial_guess(z_l = zs, lens_type='NFW')
    elif lens_type == 'SIS':
        sources.apply_SIS_lensing(true_lenses)
        lenses = sources.generate_initial_guess(lens_type='SIS')

    # Check the initial guesses
    lens_halo_distance = np.zeros(len(lenses.x))
    for i in range(len(lenses.x)):
        lens_halo_distance[i] = np.min((lenses.x[i] - halo.x)**2 + (lenses.y[i] - halo.y)**2)
    halo_found_guess = lens_halo_distance < 5
    # Reshape halo_found to match the shape of chi2_values
    halo_found_guess = halo_found_guess.reshape(xgrid.shape)

    print('Initial guesses calculated...')
    # Step 2 - Optimize guesses with local minimization
    lenses.optimize_lens_positions(sources, use_flags)

    # Check the optimized guesses
    lens_halo_distance = np.zeros(len(lenses.x))
    for i in range(len(lenses.x)):
        lens_halo_distance[i] = np.min((lenses.x[i] - halo.x)**2 + (lenses.y[i] - halo.y)**2)
    halo_found_optimized = lens_halo_distance < 5
    # Reshape halo_found to match the shape of chi2_values
    halo_found_optimized = halo_found_optimized.reshape(xgrid.shape)
    print('Initial optimization complete...')
    
    # Now, lets plot the results
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    if lens_type == 'NFW':
        title = 'Initial Steps - NFW Lensing'
    elif lens_type == 'SIS':
        title = 'Initial Steps - SIS Lensing'
    fig.suptitle(title)

    im = ax[0].imshow(halo_found_guess, extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[0].scatter(true_lenses.x, true_lenses.y, color='red', marker='x', label='True Halos')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Halo Found (Initial Guess)')
    fig.colorbar(im, ax=ax[0])

    im = ax[1].imshow(halo_found_optimized, extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[1].scatter(true_lenses.x, true_lenses.y, color='red', marker='x', label='True Halos')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Halo Found (Optimized)')
    fig.colorbar(im, ax=ax[1])
    
    plt.tight_layout()
    if lens_type == 'NFW':
        file_name = 'Images/MDARK/pipeline_initial_steps_NFW.png'
    elif lens_type == 'SIS':
        file_name = 'Images/MDARK/pipeline_initial_steps_SIS.png'
    plt.savefig(file_name)
    stop = time.time()
    time_taken = stop - start
    time_taken_readable = '{:.2f} s'.format(time_taken)
    # Change to minutes if time taken is greater than 60 seconds
    if time_taken > 60:
        time_taken = time_taken / 60
        time_taken_readable = '{:.2f} m'.format(time_taken)
    print('Time taken for {} lensing: {}'.format(lens_type, time_taken_readable))
    plt.show()


def visualize_initial_guesses():
    # Create a single halo, a set of sources around that halo, and see how the initial guesses do as a function of source-halo distance

    # Create a single halo
    halo = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([5]), np.array([1e14]), 0.194, np.array([0]))
    halo.calculate_concentration()

    # Create a set of sources
    # Distribute the sources in a spiral pattern, moving outwards
    rmax = 50
    N = 200
    r = np.linspace(1, rmax, N)
    theta = np.linspace(0, 2*np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Create a source object
    sources = pipeline.Source(x, y, 
                            np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x), 
                            np.ones_like(x) * 0.1, np.ones_like(x) * 0.01, np.ones_like(x) * 0.02)
    sources.apply_noise()
    sources.apply_NFW_lensing(halo)

    # Create initial guesses
    lenses = sources.generate_initial_guess(z_l = 0.194, lens_type='NFW')

    # Create a plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(sources.x, sources.y, color='black', label='Sources', alpha=0.5)
    ax.scatter(halo.x, halo.y, color='red', label='True Halo', marker='*', s=100)
    ax.scatter(lenses.x, lenses.y, color='blue', label='Initial Guesses', marker='x', alpha=0.75)
    # Draw an arrow from the source to the lens
    for i in range(len(lenses.x)):
        ax.arrow(sources.x[i], sources.y[i], lenses.x[i] - sources.x[i], lenses.y[i] - sources.y[i], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-rmax, rmax])
    ax.set_ylim([-rmax, rmax])
    plt.tight_layout()

    # Evaluate the initial guesses
    lens_halo_distance = np.zeros(len(lenses.x))
    for i in range(len(lenses.x)):
        lens_halo_distance[i] = np.min((lenses.x[i] - halo.x)**2 + (lenses.y[i] - halo.y)**2)
    lens_halo_distance = np.sqrt(lens_halo_distance)
    halo_found = lens_halo_distance < 5
    # Reshape halo_found to match the shape of chi2_values
    halo_found = halo_found.reshape(x.shape)
    print('{}% of the sources were correctly identified'.format(np.sum(halo_found) / N * 100))

    # Lets try to draw a circle at the distance where sources stop identifying the halo
    # This is a rough estimate - using halo_found, what's the point at which the values stop being True?
    # We shouldn't just take the first index where halo_found is False, but rather find the first sequence of multiple False values
    # This is because there may be a few false positives
    for i in range(len(halo_found)):
        if np.sum(halo_found[i:i+5]) == 0:
            break
    # The distance at which the halo is no longer found is r[i]
    # Draw a circle at this distance
    circle = plt.Circle((0, 0), r[i], color='red', fill=False, label='Halo Detection Limit: {:.2f}'.format(r[i]), linestyle='--')
    ax.add_artist(circle)
    ax.legend(loc='best')

    ax.set_title('Evaluation of Initial Guesses \n Of {} sources, {} came close to guessing the halo location'.format(N, np.sum(halo_found)), fontsize=10)


    plt.savefig('Images/initial_guesses_large.png')


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
    '''
    chi_2_vals = []
    halo.update_chi2_values(sources, [True, True, True])
    true_chi2 = halo.chi2
    chi_2_vals.append(true_chi2[0])

    for i in range(len(points)):
        current_point = points[i]
        final_point = current_point[-1]
        lens = pipeline.Halo(final_point[0], final_point[1], np.zeros_like(final_point[0]), np.zeros_like(final_point[0]), final_point[2], 0.194, np.zeros_like(final_point[0]))
        lens.calculate_concentration()
        one_source = pipeline.Source(xs[i], ys[i], 
                                    np.zeros_like(xs[i]), np.zeros_like(xs[i]), np.zeros_like(xs[i]), np.zeros_like(xs[i]), np.zeros_like(xs[i]), np.zeros_like(xs[i]),
                                    np.ones_like(xs[i]) * 0.1, np.ones_like(xs[i]) * 0.01, np.ones_like(xs[i]) * 0.02)
        one_source.apply_NFW_lensing(halo)
        lens.update_chi2_values(sources, [True, True, True])
        chi_2_vals.append(lens.chi2[0])
    
    print('Chi2 values: ', chi_2_vals)
    plt.figure()
    x = np.linspace(0, len(chi_2_vals), len(chi_2_vals))
    plt.plot(x, chi_2_vals, marker='o')
    plt.xlabel('Lens Num')
    plt.ylabel('Chi2 Value')
    plt.title('Chi2 Value after Optimization')
    plt.show()
    '''
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
    
    
    assert len(lenses.mass) == N, 'Masses not computed for all lenses'
    assert np.all(lenses.mass > 0), 'Masses are not positive'
    '''
    # Plot the mass distribution of the lenses as a histogram
    plt.figure()
    plt.hist(np.log10(lenses.mass), bins=10, color='blue', alpha=0.5, label='Optimized Masses')
    plt.axvline(x=14, color='red', linestyle='--', label='True Mass')
    plt.xlabel('Mass (log scale)')
    plt.ylabel('Frequency')
    plt.title('Mass Distribution of Lenses')
    plt.legend()
    plt.savefig('Images/initial_optimization_masses.png')

    # Plot mass distribution as a function of distance from the halo
    plt.figure()
    plt.scatter(r, np.log10(lenses.mass), color='blue', label='Optimized Masses')
    plt.axhline(y=np.log10(1e14), color='red', linestyle='--', label='True Mass')
    plt.xlabel('Distance from Halo')
    plt.ylabel('Mass (log scale)')
    plt.title('Mass Distribution of Lenses as a Function of Distance from Halo')
    plt.legend()
    plt.savefig('Images/initial_optimization_masses_distance.png')
    '''
    plt.show()


def map_chi2_space():
    # Create a single halo, a set of sources, and place a test lens everywhere in space, getting the chi2 value for each lens. 
    # Get a sense of how chi2 varies around the halo

    # Create a single halo
    halo = pipeline.Halo(np.array([0]), np.array([0]), np.array([0]), np.array([5]), np.array([1e14]), 0.194, np.array([0]))
    halo.calculate_concentration()

    # Create a set of sources
    xmax = 50
    N = 100
    rs = np.sqrt(np.random.random(N)) * xmax  
    thetas = np.random.random(N) * 2 * np.pi
    xs = rs * np.cos(thetas)
    ys = rs * np.sin(thetas)

    sources = pipeline.Source(xs, ys,
                              np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs),
                                np.ones_like(xs) * 0.1, np.ones_like(xs) * 0.01, np.ones_like(xs) * 0.02)
    # sources.apply_noise()
    sources.apply_NFW_lensing(halo)
    sources.filter_sources(xmax)

    xgrid = np.linspace(-xmax, xmax, 2*xmax)
    ygrid = np.linspace(-xmax, xmax, 2*xmax)

    chi2_values = np.zeros((len(xgrid), len(ygrid)))
    for i in range(len(xgrid)):
        for j in range(len(ygrid)):
            lens = pipeline.Halo(np.array([xgrid[i]]), np.array([ygrid[j]]), np.array([0]), np.array([5]), np.array([1e14]), 0.194, np.array([0]))
            lens.calculate_concentration()
            lens.update_chi2_values(sources, [True, True, True])
            chi2_values[i, j] = lens.chi2[0]

    # Pick one lens, optimize it, and plot the trail across our chi2 space
    xl = 10
    yl = 10
    mass = 1e14
    test_lens = pipeline.Halo(np.array([xl]), np.array([yl]), np.array([0]), np.array([5]), np.array([mass]), 0.194, np.array([0]))
    test_lens.calculate_concentration()
    test_lens.update_chi2_values(sources, [True, True, True])
    trail = test_lens.optimize_lens_positions(sources, [True, True, True])

    # Get the gradient of the chi2 values (numerically)
    grad_x, grad_y = np.gradient(chi2_values)
    
    # Smooth and clip the gradient
    # grad_x = np.clip(grad_x, -1, 1)
    # grad_y = np.clip(grad_y, -1, 1)
    from scipy.ndimage import gaussian_filter
    # grad_x = gaussian_filter(grad_x, sigma=5)
    # grad_y = gaussian_filter(grad_y, sigma=5)
    
    # Create a plot out of this
    
    # Plot the chi2 values
    fig, ax = plt.subplots(1, 3, figsize=(5, 5))
    fig.suptitle('Chi2 Space')

    ax[0].imshow((chi2_values), extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[0].scatter(halo.x, halo.y, color='red', label='True Halo', marker='*', s=100)
    for i in range(len(trail)):
        points = trail[i]
        x = points[0]
        y = points[1]
        # ax[0].scatter(x, y, color='yellow', alpha=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Chi2 Values for Test Lenses')
    
    # Plot the gradient
    ax[1].imshow(((grad_x)), extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[1].scatter(halo.x, halo.y, color='red', label='True Halo', marker='*', s=100)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Gradient of Chi2 Values (x)')

    ax[2].imshow((grad_y), extent=(-xmax, xmax, -xmax, xmax), origin='lower', cmap='viridis')
    ax[2].scatter(halo.x, halo.y, color='red', label='True Halo', marker='*', s=100)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('y')
    ax[2].set_title('Gradient of Chi2 Values (y)')

    # plt.tight_layout()
    plt.savefig('Images/chi2_space.png')
    plt.show()


if __name__ == '__main__':
    # map_chi2_space()
    # visualize_initial_guesses()
    # visualize_initial_optimization()
    # raise ValueError('This script is not meant to be run as a standalone script')
    
    # visualize_fits('Data/MDARK_Test/Test15/ID_file_15.csv')
    # Run a set of tests with varying scales and lens/source numbers
    #simple_nfw_test(1, 10, 10)
    # simple_nfw_test(2, 10, 10)
    start = time.time()
    simple_nfw_test(1, 100, 50)
    simple_nfw_test(2, 100, 50)
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    # plt.show()
    raise ValueError('This script is not meant to be run as a standalone script')
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