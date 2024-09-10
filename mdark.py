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

def _plot_results(halo, true_halo, title, reducedchi2, xmax, ax=None, legend=True, show_mass = False, show_chi2 = False):
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
    x_recon, y_recon, mass_recon, chi2_recon = [halo.x, halo.y, halo.mass, halo.chi2]

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
    
    if show_chi2:
        # Label the chi2 value of each of the lenses
        for i in range(len(x_recon)):
            ax.text(x_recon[i], y_recon[i], '{:.2f}'.format(chi2_recon[i]), fontsize=12, color='black')


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
    masses = [mass / h for mass in masses]
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

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


# --------------------------------------------
# MDARK Processing Functions
# --------------------------------------------

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
            rows.append(chunk[chunk['MainHaloID'].isin(valid_ids)])
    # Choose a random cluster from the list of valid rows
    if len(rows) > 0:
        rows = pd.concat(rows) # Concatenate the rows
        row = rows.sample(n=1) # Choose a random row
        return row
    else:
        return None


# --------------------------------------------
# Random Realization Functions
# --------------------------------------------

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
    sources = utils.createSources(lenses, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax, lens_type='NFW')
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
    log_detection_mass = []
    chi2 = []
    
    for result in results:
        x_vals = []
        y_vals = []
        mass_vals = []
        candidate_lenses, candidate_chi2 = result
        for x in candidate_lenses.x:
            x_vals.append(x)
        for y in candidate_lenses.y:
            y_vals.append(y)
        for mass in candidate_lenses.mass:
            mass = np.abs(mass)
            mass_vals.append(mass)
        
        x_vals, y_vals, mass_vals = np.array(x_vals), np.array(y_vals), np.array(mass_vals)

        chi2.append(candidate_chi2)
        detection_mass = 10**-10 # Small value to avoid log(0)

        for i in range(len(x_vals)):
            for j in range(len(xl)):
                if np.sqrt((x_vals[i] - xl[j])**2 + (y_vals[i] - yl[j])**2) < 5:
                    detection_mass += mass_vals[i]
        
        xlens.append(x_vals)
        ylens.append(y_vals)
        log_mass.append(np.log10(mass_vals))
        log_detection_mass.append(np.log10(detection_mass))

    # Concatenate the results
    xlens = np.concatenate(xlens)
    ylens = np.concatenate(ylens)
    log_mass = np.concatenate(log_mass)

    # Save the results to a file
    # Keep in mind, arrays are not the same length. 
    # We need to pad the arrays with NANs to make them the same length
    max_length = max(len(xlens), len(ylens), len(log_mass), len(chi2))
    xlens = np.pad(xlens, (0, max_length - len(xlens)), 'constant', constant_values=np.nan)
    ylens = np.pad(ylens, (0, max_length - len(ylens)), 'constant', constant_values=np.nan)
    log_mass = np.pad(log_mass, (0, max_length - len(log_mass)), 'constant', constant_values=np.nan)
    log_detection_mass = np.pad(log_detection_mass, (0, max_length - len(log_detection_mass)), 'constant', constant_values=np.nan)

    chi2 = np.pad(chi2, (0, max_length - len(chi2)), 'constant', constant_values=np.nan)

    results = pd.DataFrame({'x': xlens, 'y': ylens, 'log_mass': log_mass, 'log_detection_mass': log_detection_mass, 'chi2': chi2})
    results.to_csv(file_name, index=False)
    
    return


def interpret_rr_results(results_file, xmax, Nlens, mass):
    # Read in the results file and interpret the results
    results = pd.read_csv(results_file)
    xlens = results['x'].values
    ylens = results['y'].values
    log_mass = results['log_mass'].values
    log_detection_mass = results['log_detection_mass'].values
    
    # Check for NANs, if so, remove them
    xlens = xlens[~np.isnan(xlens)]
    ylens = ylens[~np.isnan(ylens)]
    log_mass = log_mass[~np.isnan(log_mass)]
    log_detection_mass = log_detection_mass[~np.isnan(log_detection_mass)]

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

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Random Realization Test: Nlens = {}, Mass = {:.2e}'.format(Nlens, mass))
    ax = ax.flatten()
    data = [xlens, ylens, log_mass, log_detection_mass]
    xlabels = ['x', 'y', 'log_mass', 'log_detection_mass']
    titles = ['Lens x Position', 'Lens y Position', 'Lens Mass','Detection Mass']
    true_vals = [xl, yl, np.log10(ml), np.log10(ml)]
    xrange = [(-xmax, xmax), (-xmax, xmax), (10, 16), (10, 16)]

    for i in range(4):
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
    plt.savefig('Images/NFW_tests/random_realization/detection_Nlens_{}_{}.png'.format(Nlens, size))


# --------------------------------------------
# Helper Functions
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

# --------------------------------------------
# Testing Functions
# --------------------------------------------

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


def simple_nfw_test(Nlens, Nsource, xmax, halo_mass, use_noise=True, use_flags=[True, True, True]):
    # Create a simple lensing field and test the pipeline on it
    start = time.time()
    halos, sources, noisy = utils.build_standardized_field(Nlens, Nsource, halo_mass, xmax, use_noise)

    # Arrange a plot with 6 subplots in 2 rows
    fig, axarr = plt.subplots(2, 3, figsize=(20, 15), sharex=True, sharey=True)

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
    _plot_results(lenses, halos, 'Merging', reducedchi2, xmax, ax=axarr[1,1], legend=False, show_chi2=True)

    # Step 6: Final minimization
    # Grab only the best lens (if there are multiple)
    '''
    if len(lenses.x) > 1:
        best_lens = np.argmin(lenses.chi2)
        lenses.x = np.array([lenses.x[best_lens]])
        lenses.y = np.array([lenses.y[best_lens]])
        lenses.mass = np.array([lenses.mass[best_lens]])
        lenses.concentration = np.array([lenses.concentration[best_lens]])
        lenses.z = np.array([lenses.z[best_lens]])
        lenses.chi2 = np.array([lenses.chi2[best_lens]])
    '''
    lenses.optimize_lens_positions(sources, use_flags) # Try this instead - see if the offset is the problem
    results = lenses.two_param_minimization(sources, use_flags)
    reducedchi2 = lenses.update_chi2_values(sources, use_flags)
    _plot_results(lenses, halos, 'Final Minimization', reducedchi2, xmax, ax=axarr[1,2], legend=False, show_mass=True)

    # Get the 'ideal' mass - only counting the mass of lenses that are close to the true halo
    ideal_mass = 0
    for i in range(len(lenses.x)):
        for j in range(len(halos.x)):
            if np.sqrt((lenses.x[i] - halos.x[j])**2 + (lenses.y[i] - halos.y[j])**2) < 5:
                ideal_mass += lenses.mass[i]
                break
    
    fig.suptitle('True Mass: {:.2e} $M_\odot$ \n Recovered Mass: {:.2e} $M_\odot$ \n Detection Mass: {:.2e} $M_\odot$'.format(np.sum(halos.mass), np.sum(lenses.mass), ideal_mass))
    if halo_mass == 1e14:
        size = 'large'
    elif halo_mass == 1e13:
        size = 'medium'
    elif halo_mass == 1e12:
        size = 'small'
    else:
        size = 'other'
    
    if use_flags == [True, True, True]:
        directory = 'all'
    elif use_flags == [True, True, False]:
        directory = 'shear_f'
    elif use_flags == [False, True, True]:
        directory = 'f_g'
    elif use_flags == [True, False, True]:
        directory = 'shear_g'
    plot_name = 'Images/NFW_tests/standard_tests/{}/{}_Nlens_{}_{}.png'.format(directory, size,Nlens,noisy)
    plt.savefig(plot_name)
    stop = time.time()
    
    # Plot the minimized path
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    results = np.array(results) # Convert to numpy array
    results[:,0] = 10**results[:,0] # Convert mass values to linear scale
    colors = ['red'] + ['blue']*(len(results) - 2) + ['green'] # Color the first point red and the last point green
    ax.scatter(results[:, 0], results[:, 1], color = colors, label='Minimization Path', alpha=0.5)
    ax.axvline(halos.mass[0], color='black', linestyle='--', label='True Halo')

    # Let's also get a sense of what the chi2 space looks like here
    mass_range = np.logspace(np.log10(halo_mass)-1, np.log10(halo_mass)+1, 100)
    test_lens = pipeline.Halo(lenses.x, lenses.y, lenses.z, lenses.concentration, lenses.mass, 0.194, np.zeros_like(lenses.x))
    chi2_values = np.zeros_like(mass_range)
    for i in range(len(mass_range)):
        if len(lenses.x) > 1:
            for j in range(len(lenses.x)):
                test_lens.mass[j] = mass_range[i]
        else:
            test_lens.mass[0] = mass_range[i]
        test_lens.update_chi2_values(sources, use_flags)
        chi2_values[i] = test_lens.chi2[0] # Get the raw chi2, not the reduced chi2
    ax.plot(mass_range, chi2_values, label='Chi2 Path', linestyle='--', c='red')
    # Identify the lowest point
    min_chi2 = np.min(chi2_values)
    ax.axhline(min_chi2, color='blue', linestyle='--', label='Lowest Chi2', alpha=0.5)
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Chi2')
    ax.set_title('Minimization Path')
    plt.savefig('Images/NFW_tests/standard_tests/{}/{}_Nlens_{}_{}_minimization.png'.format(directory, size, Nlens, noisy))
    plt.close()
    

    print('Finished test: {} seconds'.format(stop - start))
    true_chi2 = halos.update_chi2_values(sources, use_flags)
    print('We beat the true chi2: {}'.format(true_chi2 > reducedchi2))
    print('True chi2: {}, Reduced chi2: {}'.format(true_chi2, reducedchi2))
    return


def run_simple_tests():
    ns = 0.01
    xmax = 50
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    masses = [1e14, 1e13, 1e12]
    lens_numbers = [1]
    noise_use = [True, False]
    # use_flags_choices = [[True, True, True], [True, True, False], [False, True, True], [True, False, True]]

    for mass in masses:
        for Nlens in lens_numbers:
            for noise in noise_use:
                # for use_flags in use_flags_choices:
                simple_nfw_test(Nlens, Nsource, xmax, mass, use_noise=noise, use_flags=[True, True, True])


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
            file_name = 'Images/NFW_tests/random_realization/detection_Nlens_{}_Nsource_{}_size_{}.txt'.format(Nlens, Nsource, size)
            random_realization_test(Ntrials, Nlens, Nsource, mass, xmax, file_name)
            interpret_rr_results(file_name, xmax, Nlens, mass)
            stop = time.time()
            print('Time taken for {} Nlens = {}: {}'.format(size, Nlens, stop - start))


def process_md_set(test_number):
    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[0]
    start = time.time()
    Ntrials = 1 # Number of trials to run for each cluster in the test set

    test_dir = 'Data/MDARK_Test/Test{}'.format(test_number)
    halos_file = 'MDARK/Halos_{}.MDARK'.format(z_chosen)
    ID_file = test_dir + '/ID_file_{}.csv'.format(test_number)
    result_file = test_dir + '/results_{}.csv'.format(test_number)
    plot_name = 'Images/MDARK/mass_correlations/mass_correlation_{}.png'.format(test_number)


    run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot(ID_file, result_file, plot_name)


if __name__ == '__main__':
    # run_simple_tests()
    # How does the error in mass inference change as a function of separation?
    '''
    Nlens = 1
    Nsource = 100
    xmax = 50
    halo_mass = 1e14
    use_noise = False
    # Create a simple lensing field and test the pipeline on it
    start = time.time()
    
    Ntrials = 1000
    final_mass = []
    mass_guess = np.array([1e13])
    # Initialize progress bar
    
    utils.print_progress_bar(0, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    for n in range(Ntrials):
        halos, sources, noisy = utils.build_standardized_field(Nlens, Nsource, halo_mass, xmax, use_noise)
        x_guess = np.array([0])
        y_guess = np.array([0])
        halos_guess = pipeline.Halo(x_guess, y_guess, np.zeros_like(x_guess), np.zeros_like(x_guess), mass_guess, 0.194, np.zeros_like(x_guess))
        halos_guess.calculate_concentration()
        # Perform mass minimization
        results = halos_guess.two_param_minimization(sources, [True, True, True])
        final_mass.append(halos_guess.mass[0])
        # Update the progress bar
        utils.print_progress_bar(n, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    # Complete the progress bar
    utils.print_progress_bar(Ntrials, Ntrials, prefix='Progress:', suffix='Complete', length=50)
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    final_mass = np.array(final_mass)
    # Save the data to a file
    np.save('Data/nfw_mass_distribution_noiseless.npy', final_mass)
    
    # Create a histogram of the final masses
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fancy_hist(final_mass, ax=ax, bins='freedman', color='black', histtype='step', density=True)
    ax.axvline(halo_mass, color='black', linestyle='--', label='True Mass')
    mean = np.mean(final_mass)
    std = np.std(final_mass)
    ax.axvline(mean, color='red', linestyle='--', label='Mean = {:.2e}'.format(mean))
    ax.axvline(mean - std, color='blue', linestyle='--', label='Mean - Std = {:.2e}'.format(mean - std))
    ax.axvline(mean + std, color='blue', linestyle='--', label='Mean + Std = {:.2e}'.format(mean + std))
    ax.legend()
    ax.set_xscale('log')
    ax.set_xlabel('Final Mass ($M_\odot$)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Mass Distribution - Ntrials = {}'.format(Ntrials))
    plt.savefig('Images/NFW_tests/mass_error_noiseless.png')
    plt.show()

    '''
    halos, sources, noisy = utils.build_standardized_field(Nlens, Nsource, xmax, halo_mass, use_noise)
    r = np.linspace(0, 10, 50)
    phi = np.linspace(0, 2*np.pi, 10)
    mass_guess = 10**np.random.uniform(13, 15, 1)

    mass_results = np.zeros_like(r)
    errors = np.zeros_like(r)

    # Create a progress bar
    utils.print_progress_bar(0, len(r), prefix='Progress:', suffix='Complete', length=50)
    for i in range(len(r)):
        masses = []
        for j in range(len(phi)):
            # Create a guess lens, with a separation of r
            x_guess = np.array([r[i] * np.cos(phi[j])])
            y_guess = np.array([r[i] * np.sin(phi[j])])
            halos_guess = pipeline.Halo(x_guess, y_guess, np.zeros_like(x_guess), np.zeros_like(x_guess), mass_guess, 0.194, np.zeros_like(x_guess))
            halos_guess.calculate_concentration()
            # Perform mass minimization
            results = halos_guess.two_param_minimization(sources, [True, True, True])
            masses.append(halos_guess.mass[0])
        # Update the progress bar
        utils.print_progress_bar(i, len(r), prefix='Progress:', suffix='Complete', length=50)
        mass_results[i] = np.mean(masses)
        errors[i] = np.std(masses)
    # Complete the progress bar
    utils.print_progress_bar(len(r), len(r), prefix='Progress:', suffix='Complete', length=50)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.errorbar(r, mass_results, yerr=errors, fmt='o', color='black', label='Final Mass')
    ax.set_xlabel('Separation (arcseconds)')
    ax.set_ylabel('Inferred Mass ($M_\odot$)')
    ax.set_yscale('log')
    ax.set_title('Mass Inference as a Function of Separation (Noisy)')
    ax.axhline(halo_mass, color='red', linestyle='--', label='True Mass')
    plt.savefig('Images/NFW_tests/error_as_separation-close-noisy.png')
    plt.show()
    '''