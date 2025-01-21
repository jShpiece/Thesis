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
from utils import print_progress_bar

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


def build_mass_correlation_plot(test_dir, test_num):
    """
    This function reads true and inferred halo information from CSV files,
    plots mass correlations, and optionally plots subhalo (primary/secondary)
    mass and coordinate correlations.
    """
    # ------------------------- Helper Functions -------------------------
    def parse_coordinates(df, column_names):
        coord_arrays = []
        for col in column_names:
            parsed = []
            for cs in df[col].values:
                if isinstance(cs, str) and cs.strip().startswith('[') and cs.strip().endswith(']'):
                    vals = cs.strip('[]').split()
                    parsed.append([float(vals[0]), float(vals[1])] if len(vals) == 2 else [np.nan, np.nan])
                else:
                    parsed.append([np.nan, np.nan])
            coord_arrays.append(np.array(parsed))
        return coord_arrays

    def plot_correlation(ax, true_vals, inferred_vals, signals_label, 
                        x_label, y_label, x_range, y_range=None):
        valid_idx = inferred_vals > 0
        x_data = true_vals[valid_idx]
        y_data = inferred_vals[valid_idx]
        ax.scatter(x_data, y_data, s=10, color='black')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        x_line = np.linspace(x_range[0], x_range[1], 100)
        try:
            if len(x_data) > 1 and len(y_data) > 1:
                m, b = np.polyfit(np.log10(x_data), np.log10(y_data), 1)
                ax.plot(x_line, 10**(m*np.log10(x_line) + b),
                        color='red', label=f'Best Fit: m = {m:.2f}')
        except:
            pass

        ax.plot(x_line, x_line, color='blue', linestyle='--', label='Agreement Line')
        ax.legend()
        corr = np.corrcoef(x_data, y_data)[0, 1] if len(x_data) > 1 else np.nan
        ax.set_title(f'Signal: {signals_label}\nCorr Coef: {corr:.2f}')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    def plot_2d_hist(ax, coords, signals_label, bins, x_label='x [arcsec]', y_label='y [arcsec]'):
        valid_idx = ~np.isnan(coords[:, 0]) & ~np.isnan(coords[:, 1]) # Filter out NaNs
        x = coords[valid_idx, 0]
        y = coords[valid_idx, 1]
        ax.hist2d(x, y, bins=bins, cmap='viridis')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Signal: {signals_label}')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Number of Halos')

    def plot_clusters(ax, true_halos, found_halos, x_label='x [arcsec]', y_label='y [arcsec]'):
        if len(true_halos.mass) == 1:
            sizes = np.ones(len(true_halos.mass))
            sizes[0] = 50
        else:
            sizes = np.ones(len(true_halos.mass))
            sizes[np.argsort(true_halos.mass)[-1]] = 50
            sizes[np.argsort(true_halos.mass)[-2]] = 25
        ax.scatter(true_halos.x, true_halos.y, s=sizes, color='blue', label='True Halos', alpha = 0.5, marker='x')
        if len(found_halos.mass) == 1:
            sizes = np.ones(len(found_halos.mass))
            sizes[0] = 50
        else:
            sizes = np.ones(len(found_halos.mass))
            sizes[np.argsort(found_halos.mass)[-1]] = 50
            sizes[np.argsort(found_halos.mass)[-2]] = 25
        ax.scatter(found_halos.x, found_halos.y, s=sizes, color='red', label='Found Halos', alpha = 0.5, marker='o')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.axis('equal')

    # ------------------------- Define Path Names----------------------
    results_file = f'{test_dir}/results_{test_num}.csv'
    ID_file = f'{test_dir}/ID_file_{test_num}.csv'

    # ------------------------- Name Mapping -------------------------
    signal_map = {
        'All Signals': (True, True, True),
        'Shear and Flexion': (True, True, False),
        'Flexion and G-Flexion': (False, True, True),
        'Shear and G-Flexion': (True, False, True)
    }

    # ------------------------- Load CSV Data -------------------------
    ID_results = pd.read_csv(ID_file)
    true_mass = ID_results['Mass'].values
    true_halos = find_halos(ID_results['MainHaloID'].values, 0.194)
    for ID in true_halos:
        true_halos[ID], _, _ = build_lensing_field(true_halos[ID], 0.194, 0)

    # Load the reconstructed halos 
    # The halos are stored across multiple csv files in the test_dir
    # In the format candidate_lenses_{ID}_{test_idx}_{signal}.csv
    # Read these all in and store them in a dictionary keyed by ID and signal type
    # Open up the results file and read in the data
    results_file = f'{test_dir}/results_{test_num}.csv'
    x, y, z, concentration, mass, chi2, ID, signal = [], [], [], [], [], [], [], []
    # Read in the data from the results file
    with open(results_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            vals = line.split(',')
            x.append(float(vals[0]))
            y.append(float(vals[1]))
            z.append(float(vals[2]))
            concentration.append(float(vals[3]))
            mass.append(float(vals[4]))
            chi2.append(float(vals[5]))
            ID.append(int(vals[6]))
            signal.append(vals[7].strip())
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    concentration = np.array(concentration)
    mass = np.array(mass)
    chi2 = np.array(chi2)
    ID = np.array(ID)
    signal = np.array(signal)

    found_halos = {}
    for i in range(len(ID_results)):
        IDs = ID_results['MainHaloID'].values[i]
        # Initialize nested dictionary if it does not exist
        if IDs not in found_halos:
            found_halos[IDs] = {}

        for signal_type in ['All Signals', 'Shear and Flexion', 'Flexion and G-Flexion', 'Shear and G-Flexion']:
            idx = np.where((ID == IDs) & (signal == signal_type))[0]
            found_halos[IDs][signal_type] = halo_obj.NFW_Lens(
                x[idx],
                y[idx],
                z[idx],
                concentration[idx],
                mass[idx],
                0.197,
                chi2[idx]
            )

    # ------------------------- Plot clusters -------------------------
    # Just plot every cluster with all signals
    for ID in ID_results['MainHaloID'].values:
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_clusters(ax, true_halos[ID], found_halos[ID]['All Signals'])
        plt.savefig(f'{test_dir}/cluster_{ID}.png')
        plt.close()
    
    # ------------------------- Plot Mass Correlations -------------------------
    # Write this out later

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

    # Remove halos that are more than 1 Mpc away from the centroid - these are not actual members of the cluster
    centroid = np.argmax(halos.mass)
    halos.remove(((halos.x - halos.x[centroid])**2 + (halos.y - halos.y[centroid])**2 + (halos.z - halos.z[centroid])**2)**0.5 > 1)

    # Convert the halo coordinates to a 2D projection
    halos.project_to_2D()
    d = cosmo.angular_diameter_distance(z).to(u.meter).value
    halos.x *= (3.086 * 10**22 / d) * 206265 # Convert to arcseconds
    halos.y *= (3.086 * 10**22 / d) * 206265

    # Center the lenses at (0, 0)
    # This is allowed because observations will center on the region of highest light, which should be the center of mass

    largest_halo = np.argmax(halos.mass)
    centroid = [halos.x[largest_halo], halos.y[largest_halo]]
    halos.x -= centroid[0] 
    halos.y -= centroid[1] 

    # Remove any halos with concentrations greater than 10 - these are not realistic
    halos.remove(halos.concentration > 10)
    # TEST - Get rid of all but the three largest halos
    halos.remove(np.argsort(halos.mass)[:-3])
    # CHEATING TEST - Force the second largest halo to be 1/10th the mass of the largest halo
    halos.mass[np.argsort(halos.mass)[-2]] = halos.mass[np.argsort(halos.mass)[-1]] / 10

    # Set the maximum extent of the field of view 
    xmax = np.max((halos.x**2 + halos.y**2)**0.5)
    xmax = np.min([xmax, 180]) # arcseconds
    xmax = np.max([xmax, 60]) # arcseconds

    # Generate a set of background galaxies
    ns = 0.01
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources
    rs = np.sqrt(np.random.random(Nsource)) * xmax # Place sources randomly but symmetrically within a circle
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
    key_file = 'MDARK/test_key_{}.MDARK'.format(redshift)
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
            IDs.append([float(ID) for ID in ID_set])
    
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
    test_number, ID, signal_choices, signal_map, sources, xmax, N_test = args

    for test_idx in range(N_test):
        for idx, signal_choice in enumerate(signal_choices):
            candidate_lenses, _ = main.fit_lensing_field(sources, xmax, False, signal_choice, lens_type='NFW')
            data = np.vstack((candidate_lenses.x, candidate_lenses.y, candidate_lenses.z, candidate_lenses.concentration, candidate_lenses.mass, candidate_lenses.chi2)).T
            with open('Output/MDARK/Test{}/results_{}.csv'.format(test_number, test_number), 'a') as f:
                for row in data:
                    f.write('{},{},{},{},{},{},{},{}\n'.format(row[0], row[1], row[2], row[3], row[4], row[5], ID, signal_map[signal_choice]))
    print('ID {} completed...'.format(ID))
    return True


def run_test_parallel(ID_file, z, N_test, test_number, lensing_type='NFW'):
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
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z, ID)
        source_catalogue[ID] = sources
        xmax_values.append(xmax)

    # Check the halo substructure content
    for ID in IDs:
        print('ID: {}'.format(ID))
        print('Number of halos: {}'.format(len(halos[ID].mass)))
        print('Primary halo mass: {:.2e}'.format(np.max(halos[ID].mass)))
        print('Secondary halo mass: {:.2e}'.format(np.sort(halos[ID].mass)[-2]))
        print('Mass fraction: {:.3f}'.format(1 - np.max(halos[ID].mass) / np.sum(halos[ID].mass)))
        print('-'*50)

    # raise ValueError('Check the halo substructure content')
    print('Halo and Source objects loaded...')

    '''
    for ID in IDs:
        mass_primary = np.max(halos[ID].mass)
        mass_secondary = np.sort(halos[ID].mass)[-2]
        print('Primary Mass: {:.2e} Solar Masses'.format(mass_primary))
        print('Secondary Mass: {:.2e} Solar Masses'.format(mass_secondary))
        print('Mass Ratio: {:.2f}'.format(mass_secondary / mass_primary))
        print('-' * 50)
    raise ValueError
    '''

    # Do not convert signal_choices tuples into lists
    tasks = [(test_number, ID, signal_choices, signal_map, source_catalogue[ID], xmax_values[i], N_test)
            for i, ID in enumerate(IDs)]
    
    # Create a results csv file to store the results
    results_file = 'Output/MDARK/Test{}/results_{}.csv'.format(test_number, test_number)
    with open(results_file, 'w') as f:
        f.write("x,y,z,concentration,mass,chi2,ID,signal\n")
    
    with Pool() as pool:
        results = pool.map(run_single_test, tasks)

    # Put the results in a file
    

    print('All tests completed...')


def process_md_set(test_number):
    '''
    Function to process a test set of MDARK clusters
    '''
    # Initialize file paths
    zs = [0.194, 0.221, 0.248, 0.276]
    z_chosen = zs[1]
    start = time.time()
    Ntrials = 1 # Number of trials to run for each cluster in the test set

    test_dir = 'Output/MDARK/Test{}'.format(test_number)
    ID_file = test_dir + '/ID_file_{}.csv'.format(test_number)

    run_test_parallel(ID_file, z_chosen, Ntrials, test_number, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))


def create_improved_key_optimized():
    key_file = 'MDARK/fixed_key_0.221.MDARK'
    halo_file = 'MDARK/Halos_0.221.MDARK'
    output_file = 'MDARK/test_key_0.221.MDARK'

    # Write the header to the output file
    with open(output_file, 'w') as f:
        f.write('MainHaloID,Total Mass,Redshift,Halo Number,Mass Fraction,Characteristic Size\n')

    print("Step 1: Identifying relevant clusters from key file...")
    
    # --- 1) Identify relevant clusters from the key file ---
    relevant_clusters = {}
    key_chunks = read_csv_in_chunks(
        key_file,
        usecols=[
            'MainHaloID',
            ' Total Mass',
            ' Redshift',
            ' Halo Number',
            ' Mass Fraction',
            ' Characteristic Size'
        ],
        chunksize=10000
    )

    # Example chunk count for display (adjust if known or derive dynamically)
    chunk_count_key = 11271  
    chunk_index_key = 0

    for key_chunk in key_chunks:
        chunk_index_key += 1
        print_progress_bar(chunk_index_key, chunk_count_key, prefix='Reading Key Chunks:', suffix='Complete', length=50)
        
        for _, row in key_chunk.iterrows():
            mass = row[' Total Mass']
            halo_num = row[' Halo Number']
            if mass >= 1e12 and halo_num >= 2:
                main_id = row['MainHaloID']
                relevant_clusters[main_id] = {
                    'mass': mass,
                    'redshift': row[' Redshift'],
                    'halo_num': halo_num,
                    'mass_fraction': row[' Mass Fraction'],
                    'char_size': row[' Characteristic Size']
                }

    print(f"\nFound {len(relevant_clusters)} relevant clusters.\n")

    print("Step 2: Gathering halo data in one pass...")

    # --- 2) Gather halos for the relevant clusters in one pass ---
    cluster_halo_data = {cid: [] for cid in relevant_clusters.keys()}

    halo_chunks = read_csv_in_chunks(
        halo_file,
        usecols=[
            'MainHaloID',
            'GalaxyType',
            'HaloMass',
            'concentration_NFW',
            'x',
            'y',
            'z'
        ],
        chunksize=10000
    )

    # Example chunk count for halos (adjust if known or derive dynamically)
    chunk_count_halo = 50000  
    chunk_index_halo = 0

    for halo_chunk in halo_chunks:
        chunk_index_halo += 1
        print_progress_bar(chunk_index_halo, chunk_count_halo, prefix='Reading Halo Chunks:', suffix='Complete', length=50)
        
        halo_chunk.columns = halo_chunk.columns.str.strip()
        filtered_halos = halo_chunk[halo_chunk['MainHaloID'].isin(cluster_halo_data.keys())]
        
        if not filtered_halos.empty:
            for cid, sub_df in filtered_halos.groupby('MainHaloID'):
                cluster_halo_data[cid].append(sub_df)

    print("\nFinished gathering halo data.\n")
    print("Step 3: Processing each cluster...")

    # --- 3) Process each relevant cluster ---
    cluster_ids = list(cluster_halo_data.keys())
    total_clusters = len(cluster_ids)
    processed_clusters = 0

    for cid in cluster_ids:
        processed_clusters += 1
        print_progress_bar(processed_clusters, total_clusters, prefix='Processing Clusters:', suffix='Complete', length=50)

        # If we have no halo data for this cluster, skip
        if not cluster_halo_data[cid]:
            continue

        cluster_df = pd.concat(cluster_halo_data[cid], ignore_index=True)
        
        # Find the largest halo by mass to define the centroid
        centroid_idx = np.argmax(cluster_df['HaloMass'].values)
        x_centroid = cluster_df.iloc[centroid_idx]['x']
        y_centroid = cluster_df.iloc[centroid_idx]['y']
        z_centroid = cluster_df.iloc[centroid_idx]['z']

        # Filter out halos that are more than 1 Mpc from the largest halo
        r = np.sqrt(
            (cluster_df['x'] - x_centroid)**2 +
            (cluster_df['y'] - y_centroid)**2 +
            (cluster_df['z'] - z_centroid)**2
        )
        cluster_df = cluster_df[r < 1]

        # Remove orphan halos (GalaxyType == 2)
        cluster_df = cluster_df[cluster_df['GalaxyType'] != 2]
        
        if len(cluster_df) == 0:
            continue
        
        # Compute properties
        total_mass = cluster_df['HaloMass'].sum()
        halo_number = len(cluster_df)
        mass_fraction = 1 - np.max(cluster_df['HaloMass']) / total_mass
        
        # Use stored values from relevant_clusters, except redshift is forced to 0.221 as in your example
        char_size = relevant_clusters[cid]['char_size']
        redshift = 0.221

        with open(output_file, 'a') as f:
            f.write(
                '{},{},{},{},{},{}\n'.format(
                    cid,
                    total_mass,
                    redshift,
                    halo_number,
                    mass_fraction,
                    char_size
                )
            )

    print("\nAll processing complete.")


if __name__ == '__main__':
    '''
    test_number = 20
    test_dir = 'Output/MDARK/Test{}/'.format(test_number)
    ID_name = test_dir + 'ID_file_{}.csv'.format(test_number)
    process_md_set(test_number)
    build_mass_correlation_plot(test_dir, test_number)
    '''

    # create_improved_key_optimized()
    # build_ID_list(21, 30, 0.221)
    # build_ID_file(30, 'Output/MDARK/Test21/ID_options.csv', 21, 0.221)
    process_md_set(21)
    build_mass_correlation_plot('Output/MDARK/Test21', 21)