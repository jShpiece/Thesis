import numpy as np 
import matplotlib.pyplot as plt
from astropy.visualization import hist as fancyhist
import pandas as pd
import halo_obj
import source_obj
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool
import time
import main
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import ast

# Physical constants
c = 3e8  # Speed of light in m/s
G = 6.674e-11  # Gravitational constant in m^3/kg/s^2
h = 0.677  # Hubble constant
M_solar = 1.989e30  # Solar mass in kg
z_source = 0.8  # Redshift of the source galaxies

# Directory paths
data_dir = 'MDARK/'
gof_dir = 'Output/MDARK_Test/Goodness_of_Fit/'

# Column names
column_names = [
    'MainHaloID', 'Total Mass', 'Redshift', 'Halo Number',
    'Mass Fraction', 'Characteristic Size'
]

# Important Notes!
# Total Mass is in units of M_sun/h
# Characteristic Size is in units of arcseconds

# Use the scientific presentation style sheet for all plots
plt.style.use('scientific_presentation.mplstyle')



def build_mass_correlation_plot(ID_file, file_name, plot_name):
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    # Get the mass and true mass
    # True mass is stored in the ID file
    ID_results = pd.read_csv(ID_file)
    '''
    true_mass = ID_results['Mass'].values 
    mass = results['Mass_all_signals'].values 
    mass_gamma_f = results['Mass_gamma_F'].values 
    mass_f_g = results['Mass_F_G'].values 
    mass_gamma_g = results['Mass_gamma_G'].values 

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
    mass_gamma_f = mass_gamma_f 
    mass_f_g = mass_f_g 
    mass_gamma_g = mass_gamma_g 
    masses = [mass, mass_gamma_f, mass_f_g, mass_gamma_g]
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
    '''
    # Now read in the location and masses of the primary and secondary halos, and compare to the true locations
    primary_coords = results[['Primary_Coord_all_signals', 'Primary_Coord_gamma_F', 'Primary_Coord_F_G', 'Primary_Coord_gamma_G']].values
    secondary_coords = results[['Secondary_Coord_all_signals', 'Secondary_Coord_gamma_F', 'Secondary_Coord_F_G', 'Secondary_Coord_gamma_G']].values
    primary_masses = results[['Primary_Mass_all_signals', 'Primary_Mass_gamma_F', 'Primary_Mass_F_G', 'Primary_Mass_gamma_G']].values
    secondary_masses = results[['Secondary_Mass_all_signals', 'Secondary_Mass_gamma_F', 'Secondary_Mass_F_G', 'Secondary_Mass_gamma_G']].values

    for i in range(4): # Iterate over each signal combination
        for j in range(len(primary_coords)): # Iterate over each cluster
            entry = primary_coords[:, i][j]
            if entry == 'nan':
                continue
            entry = entry.strip('[]').split()
            primary_coords[:, i][j] = [float(entry[0]), float(entry[1])]

            entry = secondary_coords[:, i][j]

            try:
                entry = entry.strip('[]').split()
                secondary_coords[:, i][j] = [float(entry[0]), float(entry[1])]
            except:
                secondary_coords[:, i][j] = [np.nan, np.nan]


    # raise ValueError('Stop here')
    # Get the true locations of the primary and secondary halos
    # This will require us to use the find_halos function, since we only have the IDs
    IDs = ID_results['MainHaloID'].values
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']
    halos = find_halos(IDs, 0.194)
    # Need to send halos through 'build lensing field' function in order to get the halo coordinates in arcseconds and centered at (0, 0)
    for ID in IDs:
        halos[ID], _, _ = build_lensing_field(halos[ID], 0.194)
    
    true_primary_coords = []
    true_secondary_coords = []
    true_primary_masses = []
    true_secondary_masses = []
    for ID in IDs:
        halo = halos[ID]
        primary_loc = np.argmax(halo.mass)
        true_primary_coords.append([halo.x[primary_loc], halo.y[primary_loc]])
        true_primary_masses.append(halo.mass[primary_loc])
        if len(halo.mass) >= 2:
            sorted_indices = np.argsort(halo.mass)
            secondary_loc = sorted_indices[-2]
            true_secondary_coords.append([halo.x[secondary_loc], halo.y[secondary_loc]])
            true_secondary_masses.append(halo.mass[secondary_loc])
        else:
            true_secondary_coords.append([np.nan, np.nan])
            true_secondary_masses.append(np.nan)
    
    
    # Now plot the results for the primary and secondary halos - do this as a histogram, plotting the distance from the true location, and the distance from the true mass
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(4):
        primary_coords_temp = primary_coords[:, i]
        secondary_coords_temp = secondary_coords[:, i]

        true_primary_coords_temp = np.array(true_primary_coords)
        true_secondary_coords_temp = np.array(true_secondary_coords)

        # Calculate the distance from the true location
        primary_distance = np.zeros(len(primary_coords_temp))
        secondary_distance = np.zeros(len(primary_coords_temp))
        for n in range(len(primary_coords_temp)):
            primary_distance[n] = np.sqrt((primary_coords_temp[n][0])**2 + (primary_coords_temp[n][1])**2)
            secondary_distance[n] = np.sqrt((secondary_coords_temp[n][0] - true_secondary_coords_temp[n][0])**2 + (secondary_coords_temp[n][1] - true_secondary_coords_temp[n][1])**2)
        # Clean up any nan values
        primary_distance = primary_distance[~np.isnan(primary_distance)]
        secondary_distance = secondary_distance[~np.isnan(secondary_distance)]
        print(secondary_distance)
        # Calculate the distance from the true mass
        # secondary_mass_distance = np.abs(secondary_masses_temp - true_secondary_masses_temp)

        fancyhist(primary_distance, bins='freedman', ax=ax[i], color='blue', alpha=0.5, label='Primary Halo')
        # fancyhist(secondary_distance, bins='freedman', ax=ax[i], color='red', alpha=0.5, label='Secondary Halo')
        ax[i].set_xlabel('Distance from True Location [arcseconds]')
        ax[i].set_ylabel('Frequency')
        ax[i].set_title('Signal Combination: {}'.format(signals[i]))
        ax[i].legend()
    
    fig.tight_layout()
    fig.savefig('Output/MDARK/mass_correlations/primary_secondary_distance_{}.png'.format(ID_file.split('_')[-1].split('.')[0]))
    
    # Now do the same for mass
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(4):
        primary_masses_temp = primary_masses[:, i]
        secondary_masses_temp = secondary_masses[:, i]

        true_primary_masses_temp = np.array(true_primary_masses)
        true_secondary_masses_temp = np.array(true_secondary_masses)

        # Calculate the distance from the true mass
        # Need to do this as a fraction of the true mass
        primary_mass_distance = (primary_masses_temp - true_primary_masses_temp) / true_primary_masses_temp
        secondary_mass_distance = (secondary_masses_temp - true_secondary_masses_temp) / true_secondary_masses_temp
        # handle nan values
        primary_mass_distance = primary_mass_distance[~np.isnan(primary_mass_distance)]
        secondary_mass_distance = secondary_mass_distance[~np.isnan(secondary_mass_distance)]

        fancyhist(primary_mass_distance, bins='freedman', ax=ax[i], color='blue', alpha=0.5, label='Primary Halo')
        try: 
            fancyhist(secondary_mass_distance, bins='freedman', ax=ax[i], color='red', alpha=0.5, label='Secondary Halo')
        except:
            print('RuntimeWarning: Skipping secondary mass distance plot')
        ax[i].set_xlabel('Fractional Mass Difference')
        ax[i].set_ylabel('Frequency')
        ax[i].set_title('Signal Combination: {}'.format(signals[i]))
        ax[i].legend()
    
    fig.tight_layout()
    fig.savefig('Output/MDARK/mass_correlations/primary_secondary_mass_distance_{}.png'.format(ID_file.split('_')[-1].split('.')[0]))

# --------------------------------------------
# File Management Functions
# --------------------------------------------

def read_csv_in_chunks(file_path, usecols=None, dtype=None, chunksize=50000):
    """
    Reads a CSV file in chunks and returns an iterator.

    Parameters:
        file_path (str): Path to the CSV file.
        usecols (list or None): List of columns to read.
        dtype (dict or None): Dictionary specifying data types for columns.
        chunksize (int): Number of rows per chunk.

    Returns:
        Iterator over DataFrame chunks.
    """
    try:
        return pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            chunksize=chunksize
        )
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return None


def GOF_file_reader(IDs):
    """
    Reads goodness of fit data for a list of IDs in parallel.

    Parameters:
        IDs (list): List of IDs.

    Returns:
        chi2_values (dict): Dictionary of chi2 values keyed by ID.
        mass_values (dict): Dictionary of mass values keyed by ID.
    """
    chi2_columns = ['chi2_all_signals', 'chi2_gamma_F', 'chi2_F_G', 'chi2_gamma_G']
    mass_columns = ['mass_all_signals', 'mass_gamma_F', 'mass_F_G', 'mass_gamma_G']

    chi2_values = {}
    mass_values = {}

    def read_file(ID):
        file_path = f'{gof_dir}Goodness_of_Fit_{ID}.csv'
        try:
            data = pd.read_csv(file_path, usecols=chi2_columns + mass_columns)
            return ID, data[chi2_columns].values, data[mass_columns].values
        except FileNotFoundError:
            print(f'File not found: {file_path}')
            return ID, None, None

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_file, ID): ID for ID in IDs}
        for future in as_completed(futures):
            ID, chi2_vals, mass_vals = future.result()
            if chi2_vals is not None and mass_vals is not None:
                chi2_values[ID] = chi2_vals
                mass_values[ID] = mass_vals

    return chi2_values, mass_values


def process_chunk(chunk, id_set):
    """
    Processes a single chunk of data.

    Parameters:
        chunk (DataFrame): The data chunk.
        id_set (set): Set of IDs to filter.

    Returns:
        dict: Filtered data grouped by 'MainHaloID'.
    """
    filtered_chunk = chunk[(chunk['GalaxyType'] != 2) & (chunk['MainHaloID'].isin(id_set))]
    grouped = filtered_chunk.groupby('MainHaloID')
    return {id_: group for id_, group in grouped}


# --------------------------------------------
# MDARK Processing Functions
# --------------------------------------------

def find_halos(ids, z):
    """
    Reads halo data for specified IDs and constructs Halo objects.

    Parameters:
        ids (list): List of halo IDs.
        z (float): Redshift.

    Returns:
        dict: Halo objects keyed by ID.
    """
    file_path = f'{data_dir}Halos_{z}.MDARK'
    cols_to_use = ['MainHaloID', 'x', 'y', 'z', 'concentration_NFW', 'HaloMass', 'GalaxyType']
    id_set = set(ids) # Convert to set for faster lookup
    data_accumulator = defaultdict(list)

    # Read the file in chunks
    iterator = read_csv_in_chunks(file_path, usecols=cols_to_use, chunksize=100000)

    if iterator is None:
        return {}

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, chunk, id_set) for chunk in iterator]

        for future in as_completed(futures):
            chunk_data = future.result()
            for id_, group in chunk_data.items():
                data_accumulator[id_].append(group)

    halos = {}

    for id_ in ids:
        data_list = data_accumulator.get(id_)
        if data_list:
            complete_data = pd.concat(data_list, ignore_index=True)
            xhalo = complete_data['x'].values
            yhalo = complete_data['y'].values
            zhalo = complete_data['z'].values
            chalo = complete_data['concentration_NFW'].values
            masshalo = complete_data['HaloMass'].values
            galaxy_type = complete_data['GalaxyType'].values

            # Filter out orphan halos
            keep_indices = np.where(galaxy_type != 2)[0]
            xhalo = xhalo[keep_indices]
            yhalo = yhalo[keep_indices]
            zhalo = zhalo[keep_indices]
            chalo = chalo[keep_indices]
            masshalo = masshalo[keep_indices]

            # Create and store the Halo object
            halos[id_] = halo_obj.NFW_Lens(
                xhalo, yhalo, zhalo, chalo, masshalo, z, np.zeros_like(masshalo)
            )
        else:
            print(f'No data found for ID {id_}')

    return halos


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
    halos.x *= (3.086 * 10**22 / d) * 206265 # Convert to arcseconds
    halos.y *= (3.086 * 10**22 / d) * 206265

    # Center the lenses at (0, 0)
    # This is done by finding the largest halo and setting its coordinates to (0, 0)
    # This is allowed because observations will center on the region of highest light, which should be the center of mass

    largest_halo = np.argmax(halos.mass)
    centroid = [halos.x[largest_halo], halos.y[largest_halo]]
    halos.x -= centroid[0] 
    halos.y -= centroid[1] 

    # Reassign concentration values, instead of using MDARK values
    # halos.calculate_concentration()
    # Remove any halos with concentrations greater than 10
    halos.remove(halos.concentration > 10)

    xmax = np.max((halos.x**2 + halos.y**2)**0.5)

    # Set a maximum size for the field of view of 5 arcminutes
    # And a minimum size of 1 arcminute
    xmax = np.min([xmax, 300]) # arcseconds
    xmax = np.max([xmax, 60]) # arcseconds
    
    # Generate a set of background galaxies
    ns = 0.01
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    rs = np.sqrt(np.random.random(Nsource)) * xmax
    theta = np.random.random(Nsource) * 2 * np.pi
    xs = rs * np.cos(theta)
    ys = rs * np.sin(theta)
    sources = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.ones(len(xs)) * 0.1, np.ones(len(xs)) * 0.01, np.ones(len(xs)) * 0.02)
    sources.apply_lensing(halos, lens_type='NFW', z_source=z_source)
    sources.apply_noise()
    sources.filter_sources(xmax)

    return halos, sources, xmax


def build_ID_list(test_number, Ncluster, redshift):
    # Define mass, halo number, and mass fraction ranges
    minimum_mass = 10**13
    maximum_mass = 10**15

    # Set up the file paths
    key_file = 'MDARK/fixed_key_{}.MDARK'.format(redshift)
    output_file = 'Output/MDARK/Test{}/ID_options.csv'.format(test_number, test_number)

    # Space the clusters out evenly in mass (when spacing evenly, do it in log space stupid!)   
    mass_range = np.logspace(np.log10(minimum_mass), np.log10(maximum_mass), Ncluster + 1)

    # Initialize the results array
    results = []
    # Iterate over each mass range and try to find a cluster in each chunk
    for i in range(Ncluster):
        IDs = []
        mass_min = mass_range[i]
        mass_max = mass_range[i + 1]

        # Process the key file in chunks
        chunk_iter = pd.read_csv(key_file, chunksize=100000)

        for chunk in chunk_iter:
            # Clean the column names to remove any leading/trailing spaces
            chunk.columns = chunk.columns.str.strip()

            # Filter the chunk based on the criteria
            filtered_cluster = chunk[(chunk['Total Mass'] > mass_min) & 
                                    (chunk['Total Mass'] < mass_max)]

            if not filtered_cluster.empty:
                # Randomly sample one cluster from the filtered chunk
                cluster = filtered_cluster.sample(1)

                # Extract relevant values
                ID = cluster['MainHaloID'].values[0]
                IDs.append(ID)
        results.append(IDs) # Maintain a list of lists
    
    # Write the results to a file without changing the structure (a list of lists)
    with open(output_file, 'w') as f:
        f.write('MainHaloID\n')
        for ID_set in results:
            f.write('{}\n'.format(', '.join([str(ID) for ID in ID_set])))


def build_ID_file(Ncluster, IDs_path, test_number, redshift):
    # Read in the IDs
    # Remember that the file is a list of lists
    IDs = []
    '''
    with open(IDs_path, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID_set = line.split(',')
            IDs.append([int(ID) for ID in ID_set])
    
    keep_IDs = []

    for i in range(Ncluster):
        # Read in the IDs in this slice of the array
        ID_set = IDs[i]
        
        # Initial criteria settings
        min_fraction = 0.0
        max_fraction = 0.1
        min_halo_number = 1
        max_halo_number = 10

        halos = find_halos(ID_set, redshift)
        found_cluster = False  # Flag to track if a suitable cluster is found

        # Iterate over all IDs in the current set with initial criteria
        for ID in ID_set:
            halo = halos[ID]
            mass_fraction = 1 - np.max(halo.mass) / np.sum(halo.mass)
            halo_number = len(halo.mass)
            
            if (mass_fraction > min_fraction) and (mass_fraction < max_fraction) and (halo_number > min_halo_number) and (halo_number < max_halo_number):
                print('Found a cluster that meets the criteria: {}'.format(ID))
                keep_IDs.append(ID)
                found_cluster = True
                break

        # If no suitable cluster is found, loosen the criteria
        while not found_cluster:
            # Allow for more halos (but the mass fraction must always be less than 0.1)
            max_halo_number *= 10
            if max_halo_number > 10**6:
                print('No suitable clusters found for this mass range - choosing a random cluster')
                random_ID = np.random.choice(ID_set)
                keep_IDs.append(random_ID)
                found_cluster = True
                break
            
            # Check the IDs again with the relaxed criteria
            for ID in ID_set:
                halo = halos[ID]
                mass_fraction = 1 - np.max(halo.mass) / np.sum(halo.mass)
                halo_number = len(halo.mass)
                
                if (mass_fraction > min_fraction) and (mass_fraction < max_fraction) and (halo_number > min_halo_number) and (halo_number < max_halo_number):
                    print('Found a cluster with relaxed criteria: {}'.format(ID))
                    keep_IDs.append(ID)
                    found_cluster = True
                    break
    
    '''
    # Use IDs found in previous run
    keep_IDs = [
        11364694586,
        11364617712,
        11364769410,
        11365743903,
        11364731572,
        11367394754,
        11364879575,
        11364948500,
        11394951535,
        11428660863,
        11365089045,
        11364946026,
        11364767375,
        11365235173,
        11365019088,
        11364729865,
        11364766172,
        11364653016,
        11365594609,
        11365888282,
        11370096473,
        11398214747,
        11370019351,
        11370834252,
        11373404570,
        11389713568,
        11370518133,
        11400494318,
        11380127241,
        11410267432
    ]

    # Now, we have a list of IDs that meet our criteria. Lets get the additional information from the key file, then save it to a file
    key_file = 'MDARK/fixed_key_{}.MDARK'.format(redshift)
    output_file = f'Output/MDARK/Test{test_number}/ID_file_{test_number}.csv'
    
    # Open the output file in append mode
    with open(output_file, 'w') as outfile:
        # Iterate over the key file in chunks
        chunk_iter = pd.read_csv(key_file, chunksize=100000)
        for chunk in chunk_iter:
            # Clean the column names to remove any leading/trailing spaces
            chunk.columns = chunk.columns.str.strip()

            # Filter the chunk based on the keep_IDs list
            filtered_cluster = chunk[chunk['MainHaloID'].isin(keep_IDs)]
            if not filtered_cluster.empty:
                # Append to the output file
                filtered_cluster.to_csv(outfile, index=False, header=outfile.tell() == 0)
    
    # Repair the output file - remove empty lines and change the header 
    with open(output_file, 'r') as f:
        lines = f.readlines()
    # Remove the first line
    lines = lines[1:]
    with open(output_file, 'w') as f:
        f.write('MainHaloID,Mass,Redshift,Halo Number,Mass Fraction,Characteristic Size\n')
        for line in lines:
            if line != '\n':
                f.write(line)


# --------------------------------------------
# Test Functions
# --------------------------------------------

def run_single_test(args):
    ID, signal_choices, sources, xmax, N_test = args
    # Run the pipeline for a single cluster, repeating N_test times.

    # Initialize arrays to store results
    masses = np.zeros((N_test, len(signal_choices)))
    candidate_numbers = np.zeros((N_test, len(signal_choices)), dtype=int)
    primary_coords = np.zeros((N_test, len(signal_choices), 2))
    secondary_coords = np.zeros((N_test, len(signal_choices), 2))
    primary_masses = np.zeros((N_test, len(signal_choices)))
    secondary_masses = np.zeros((N_test, len(signal_choices)))

    for test_idx in range(N_test):
        for idx, signal_choice in enumerate(signal_choices):
            candidate_lenses, _ = main.fit_lensing_field(
                sources, xmax, False, signal_choice, lens_type='NFW'
            )

            candidate_masses = candidate_lenses.mass
            candidate_x = candidate_lenses.x
            candidate_y = candidate_lenses.y

            masses[test_idx, idx] = np.sum(candidate_masses)
            candidate_numbers[test_idx, idx] = len(candidate_x)

            if len(candidate_masses) > 0:
                primary_loc = np.argmax(candidate_masses)
                primary_coords[test_idx, idx] = [candidate_x[primary_loc], candidate_y[primary_loc]]
                primary_masses[test_idx, idx] = candidate_masses[primary_loc]
            else:
                primary_coords[test_idx, idx] = [np.nan, np.nan]
                primary_masses[test_idx, idx] = np.nan

            if len(candidate_masses) >= 2:
                sorted_indices = np.argsort(candidate_masses)
                secondary_loc = sorted_indices[-2]
                secondary_coords[test_idx, idx] = [candidate_x[secondary_loc], candidate_y[secondary_loc]]
                secondary_masses[test_idx, idx] = candidate_masses[secondary_loc]
            else:
                secondary_coords[test_idx, idx] = [np.nan, np.nan]
                secondary_masses[test_idx, idx] = np.nan

    # You can compute averages or other statistics here
    # For example, compute the mean over N_test repetitions
    mean_masses = np.mean(masses, axis=0)
    mean_candidate_numbers = np.mean(candidate_numbers, axis=0)
    mean_primary_coords = np.mean(primary_coords, axis=0)
    mean_secondary_coords = np.mean(secondary_coords, axis=0)
    mean_primary_masses = np.mean(primary_masses, axis=0)
    mean_secondary_masses = np.mean(secondary_masses, axis=0)

    # Prepare the results
    results = [ID]
    results.extend(mean_masses.tolist())
    results.extend(mean_candidate_numbers.tolist())
    results.extend(mean_primary_coords.tolist())
    results.extend(mean_primary_masses.tolist())
    results.extend(mean_secondary_coords.tolist())
    results.extend(mean_secondary_masses.tolist())

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

    signal_choices = [
        [True, True, True],    # All signals
        [True, True, False],   # Shear and Flexion
        [False, True, True],   # Flexion and G-Flexion
        [True, False, True]    # Shear and G-Flexion
    ]

    # Build halos
    halos = find_halos(IDs, z)  # halos is a dictionary of Halo objects, keyed by ID
    source_catalogue = {}       # Dictionary to hold the source catalogues
    xmax_values = []            # List to hold the maximum extent of each field

    for ID in IDs:
        # Build the lenses and sources
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)
        source_catalogue[ID] = sources
        xmax_values.append(xmax)

    print('Halo and Source objects loaded...')

    # Prepare the arguments for each task
    tasks = [
        (ID, signal_choices, source_catalogue[ID], xmax_values[i], N_test)
        for i, ID in enumerate(IDs)
    ]

    # Process pool
    with Pool() as pool:
        results = pool.map(run_single_test, tasks)

    # Save the results to a file
    with open(result_file, 'w') as f:
        headers = [
            'ID',
            'Mass_all_signals', 'Mass_gamma_F', 'Mass_F_G', 'Mass_gamma_G',
            'Nfound_all_signals', 'Nfound_gamma_F', 'Nfound_F_G', 'Nfound_gamma_G',
            'Primary_Coord_all_signals', 'Primary_Coord_gamma_F', 'Primary_Coord_F_G', 'Primary_Coord_gamma_G',
            'Primary_Mass_all_signals', 'Primary_Mass_gamma_F', 'Primary_Mass_F_G', 'Primary_Mass_gamma_G',
            'Secondary_Coord_all_signals', 'Secondary_Coord_gamma_F', 'Secondary_Coord_F_G', 'Secondary_Coord_gamma_G',
            'Secondary_Mass_all_signals', 'Secondary_Mass_gamma_F', 'Secondary_Mass_F_G', 'Secondary_Mass_gamma_G'
        ]
        f.write(','.join(headers) + '\n')
        for result in results:
            f.write(','.join(map(str, result)) + '\n')

    return


def process_md_set(test_number):
    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[0]
    start = time.time()
    Ntrials = 1 # Number of trials to run for each cluster in the test set

    test_dir = 'Output/MDARK/Test{}'.format(test_number)
    halos_file = 'MDARK/Halos_{}.MDARK'.format(z_chosen)
    ID_file = test_dir + '/ID_file_{}.csv'.format(test_number)
    result_file = test_dir + '/results_{}.csv'.format(test_number)
    plot_name = 'Output/MDARK/mass_correlations/mass_correlation_{}.png'.format(test_number)

    run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot(ID_file, result_file, plot_name)


if __name__ == '__main__':
    # build_ID_list(17, 30, 0.194)
    # build_ID_file(30, 'Output/MDARK/Test17/ID_options.csv', 17, 0.194)
    # process_md_set(17)
    build_mass_correlation_plot('Output/MDARK/Test17/ID_file_17.csv', 'Output/MDARK/Test17/results_17.csv', 'Output/MDARK/mass_correlations/mass_correlation_17.png')
    # Pick out a halo, run the pipeline, look at the results

    raise ValueError('Stop here')
    # IDs = [IDs[0]] # Pick out the first cluster to study
    
    z = 0.194
    counter = 0
    for ID in IDs:
        halos = find_halos([ID], z)
        halos_copy = halos[ID].copy()
        halos_copy.calculate_concentration()
        print(halos_copy.mass)

        # Compare the concentration of the MDARK halo to our linear fit
        plt.figure()
        plt.scatter(halos_copy.concentration, halos[ID].concentration, color='black')
        plt.xlabel('Concentration (Linear Fit)')
        plt.ylabel('Concentration (MDARK)')
        plt.title('Cluster ID: {}'.format(ID))
        plt.savefig('Output/MDARK/pipeline_visualization/concentration_comparison_{}.png'.format(counter))
        plt.close()

        '''
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fancyhist(np.log10(halos[ID].mass), bins='freedman', ax=ax, color='blue')
        ax.set_xlabel('log10(Mass)')
        ax.set_ylabel('Frequency')
        ax.set_title('Cluster ID: {}'.format(ID))
        plt.savefig('Output/MDARK/pipeline_visualization/mass_histogram_{}.png'.format(counter))
        plt.close()
        '''

        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)
        candidate_lenses, _ = main.fit_lensing_field(sources, xmax, True, [True, True, False], lens_type='NFW')
        # Determine size of lenses in the plot based on their mass
        true_sizes = (np.log10(halos[ID].mass) - 12) * 10
        recon_sizes = (np.log10(candidate_lenses.mass) - 12) * 10

        plt.figure()
        plt.scatter(sources.x, sources.y, s=10, color='black', alpha=0.5, label='Sources')
        plt.scatter(halos[ID].x, halos[ID].y, s=true_sizes, color='red', label='Lenses', marker='x')
        plt.scatter(candidate_lenses.x, candidate_lenses.y, s=recon_sizes, color='blue', label='Candidates', marker='o')
        # Label candidates with their masses
        for i in range(len(candidate_lenses.x)):
            plt.text(candidate_lenses.x[i], candidate_lenses.y[i], '{:.2e}'.format(candidate_lenses.mass[i]), fontsize=8)
        plt.legend()
        plt.xlabel('x [arcseconds]')
        plt.ylabel('y [arcseconds]')
        plt.title('Cluster ID: {} \n True Mass: {:.2e} $M_{{\odot}}$ \n Candidate Mass: {:.2e} $M_{{\odot}}$'.format(ID, np.sum(halos[ID].mass), np.sum(candidate_lenses.mass)))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('Output/MDARK/pipeline_visualization/cluster_{}.png'.format(counter))
        plt.close()

        counter += 1