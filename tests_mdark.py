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
import json

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

# --------------------------------------------

def build_mass_correlation_plot(ID_file, file_name, plot_name):
    # Load the results (inferred properties) from the CSV file
    results = pd.read_csv(file_name)
    # Load the true mass data
    ID_results = pd.read_csv(ID_file)

    # Extract true mass
    true_mass = ID_results['Mass'].astype(float).values

    # Extract inferred masses
    # Adjust as needed for the columns available in the CSV produced by run_test_parallel
    mass = results['Mass_all_signals'].astype(float).values
    mass_gamma_f = results['Mass_gamma_F'].astype(float).values
    mass_f_g = results['Mass_F_G'].astype(float).values
    mass_gamma_g = results['Mass_gamma_G'].astype(float).values

    # Replace NaNs with zeros if needed
    mass = np.nan_to_num(mass)
    mass_gamma_f = np.nan_to_num(mass_gamma_f)
    mass_f_g = np.nan_to_num(mass_f_g)
    mass_gamma_g = np.nan_to_num(mass_gamma_g)

    masses = [mass, mass_gamma_f, mass_f_g, mass_gamma_g]
    signals = ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']

    # Plot the correlation between true and inferred masses
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(4):
        # Filter out zero or invalid values if needed
        valid_indices = (masses[i] > 0) & (true_mass > 0)
        true_mass_temp = true_mass[valid_indices]
        inferred_mass_temp = masses[i][valid_indices]

        ax[i].scatter(true_mass_temp, inferred_mass_temp, s=10, color='black')
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')

        # Add a line of best fit
        x = np.linspace(1e13, 1e15, 100)
        if len(true_mass_temp) > 1 and len(inferred_mass_temp) > 1:
            try:
                m, b = np.polyfit(np.log10(true_mass_temp), np.log10(inferred_mass_temp), 1)
                ax[i].plot(x, 10**(m*np.log10(x) + b), color='red', label=f'Best Fit: m = {m:.2f}')
            except:
                print('Skipping line of best fit')
        # Add an agreement line
        ax[i].plot(x, x, color='blue', linestyle='--', label='Agreement Line')
        ax[i].legend()
        ax[i].set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
        ax[i].set_ylabel(r'$M_{\rm inferred}$ [$M_{\odot}$]')
        if len(true_mass_temp) > 1:
            corr = np.corrcoef(true_mass_temp, inferred_mass_temp)[0, 1]
        else:
            corr = np.nan
        ax[i].set_title(f'Signal Combination: {signals[i]} \n Correlation Coefficient: {corr:.2f}')

    fig.tight_layout()
    fig.savefig(plot_name)

    # If you need to parse and plot primary/secondary coordinates and masses,
    # you can parse them here similarly. For example:
    # For primary coordinates:
    primary_coord_cols = [
        'Primary_Coord_all_signals', 'Primary_Coord_gamma_F',
        'Primary_Coord_F_G', 'Primary_Coord_gamma_G'
    ]
    # Parsing coordinate strings like "[x y]"
    primary_coords = []
    for col in primary_coord_cols:
        coord_strs = results[col].values
        parsed = []
        for cs in coord_strs:
            if isinstance(cs, str) and cs.strip().startswith('[') and cs.strip().endswith(']'):
                vals = cs.strip('[]').split()
                if len(vals) == 2:
                    x, y = vals
                    x, y = float(x), float(y)
                    parsed.append([x, y])
                else:
                    parsed.append([np.nan, np.nan])
            else:
                parsed.append([np.nan, np.nan])
        primary_coords.append(np.array(parsed))

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
    minimum_mass = 10**14
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
    ID, signal_choices, signal_map, sources, xmax, N_test = args
    # Initialize arrays to store results
    masses = np.zeros((N_test, len(signal_choices)))
    candidate_numbers = np.zeros((N_test, len(signal_choices)), dtype=int)
    primary_coords = np.zeros((N_test, len(signal_choices), 2))
    secondary_coords = np.zeros((N_test, len(signal_choices), 2))
    primary_masses = np.zeros((N_test, len(signal_choices)))
    secondary_masses = np.zeros((N_test, len(signal_choices)))

    for test_idx in range(N_test):
        for idx, signal_choice in enumerate(signal_choices):
            candidate_lenses, _ = main.fit_lensing_field(sources, xmax, False, signal_choice, lens_type='NFW')

            candidate_masses = candidate_lenses.mass
            candidate_x = candidate_lenses.x
            candidate_y = candidate_lenses.y

            masses[test_idx, idx] = np.sum(candidate_masses)
            candidate_numbers[test_idx, idx] = len(candidate_x)

            # Primary lens
            if len(candidate_masses) > 0:
                primary_loc = np.argmax(candidate_masses)
                primary_coords[test_idx, idx] = [candidate_x[primary_loc], candidate_y[primary_loc]]
                primary_masses[test_idx, idx] = candidate_masses[primary_loc]
            else:
                primary_coords[test_idx, idx] = [np.nan, np.nan]
                primary_masses[test_idx, idx] = np.nan

            # Secondary lens
            if len(candidate_masses) >= 2:
                sorted_indices = np.argsort(candidate_masses)
                secondary_loc = sorted_indices[-2]
                secondary_coords[test_idx, idx] = [candidate_x[secondary_loc], candidate_y[secondary_loc]]
                secondary_masses[test_idx, idx] = candidate_masses[secondary_loc]
            else:
                secondary_coords[test_idx, idx] = [np.nan, np.nan]
                secondary_masses[test_idx, idx] = np.nan

    # Compute averages
    mean_masses = np.mean(masses, axis=0)
    mean_candidate_numbers = np.mean(candidate_numbers, axis=0)
    mean_primary_coords = np.mean(primary_coords, axis=0)
    mean_secondary_coords = np.mean(secondary_coords, axis=0)
    mean_primary_masses = np.mean(primary_masses, axis=0)
    mean_secondary_masses = np.mean(secondary_masses, axis=0)

    # Store results in a structured dictionary for easy reading
    results_dict = {'ID': ID, 'results': {}}

    for i, sc in enumerate(signal_choices):
        sc_key = signal_map[sc]
        results_dict['results'][sc_key] = {
            'mean_mass': mean_masses[i],
            'mean_candidate_number': mean_candidate_numbers[i],
            'mean_primary_coord': mean_primary_coords[i].tolist(),
            'mean_primary_mass': mean_primary_masses[i],
            'mean_secondary_coord': mean_secondary_coords[i].tolist(),
            'mean_secondary_mass': mean_secondary_masses[i]
        }

    print(f'Finished test for cluster {ID}')
    return results_dict


def run_test_parallel(ID_file, result_file, z, N_test, lensing_type='NFW'):
    IDs = []
    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(int(ID))
    IDs = np.array(IDs)

    signal_choices = [
        (True, True, True),    # All signals
        (True, True, False),   # Shear and Flexion
        (False, True, True),   # Flexion and G-Flexion
        (True, False, True)    # Shear and G-Flexion
    ]

    signal_map = {
        (True, True, True): 'All Signals',
        (True, True, False): 'Shear and Flexion',
        (False, True, True): 'Flexion and G-Flexion',
        (True, False, True): 'Shear and G-Flexion'
    }

    halos = find_halos(IDs, z)
    source_catalogue = {}
    xmax_values = []

    for ID in IDs:
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)
        source_catalogue[ID] = sources
        xmax_values.append(xmax)

    print('Halo and Source objects loaded...')

    # Do not convert signal_choices tuples into lists
    tasks = [(ID, signal_choices, signal_map, source_catalogue[ID], xmax_values[i], N_test)
            for i, ID in enumerate(IDs)]

    with Pool() as pool:
        results = pool.map(run_single_test, tasks)

    headers = [
        'ID',
        'Mass_all_signals', 'Mass_gamma_F', 'Mass_F_G', 'Mass_gamma_G',
        'Nfound_all_signals', 'Nfound_gamma_F', 'Nfound_F_G', 'Nfound_gamma_G',
        'Primary_Coord_all_signals', 'Primary_Coord_gamma_F', 'Primary_Coord_F_G', 'Primary_Coord_gamma_G',
        'Primary_Mass_all_signals', 'Primary_Mass_gamma_F', 'Primary_Mass_F_G', 'Primary_Mass_gamma_G',
        'Secondary_Coord_all_signals', 'Secondary_Coord_gamma_F', 'Secondary_Coord_F_G', 'Secondary_Coord_gamma_G',
        'Secondary_Mass_all_signals', 'Secondary_Mass_gamma_F', 'Secondary_Mass_F_G', 'Secondary_Mass_gamma_G'
    ]

    with open(result_file, 'w') as f:
        f.write(','.join(headers) + '\n')

        for res_dict in results:
            ID = res_dict['ID']
            row = [ID]

            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                row.append(data['mean_mass'])
            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                row.append(data['mean_candidate_number'])
            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                x, y = data['mean_primary_coord']
                row.append(f"[{x} {y}]")
            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                row.append(data['mean_primary_mass'])
            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                x, y = data['mean_secondary_coord']
                row.append(f"[{x} {y}]")
            for sc in signal_choices:
                sc_key = signal_map[sc]
                data = res_dict['results'][sc_key]
                row.append(data['mean_secondary_mass'])

            f.write(','.join(map(str, row)) + '\n')


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
    # build_ID_list(18, 30, 0.194)
    build_ID_file(30, 'Output/MDARK/Test19/ID_options.csv', 19, 0.194)
    process_md_set(19)
    # build_mass_correlation_plot('Output/MDARK/Test18/ID_file_18.csv', 'Output/MDARK/Test18/results_18.csv', 'Output/MDARK/mass_correlations/mass_correlation')
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