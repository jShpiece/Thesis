import numpy as np 
import matplotlib.pyplot as plt
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
    true_mass = ID_results[' Mass'].values
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
    id_set = set(ids)
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

    xmax = np.max((halos.x**2 + halos.y**2)**0.5)

    # Set a maximum size for the field of view of 2 arcminutes
    xmax = np.min([xmax, 120]) # arcseconds
    
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


# --------------------------------------------
# Test Functions
# --------------------------------------------

def run_single_test(args):
    ID, signal_choices, sources, xmax = args
    # Run the pipeline for a single cluster, with a given set of signal choices
    # N_test times. Save the results to a file

    masses = []
    candidate_number = []

    for signal_choice in signal_choices:
        candidate_lenses, _ = main.fit_lensing_field(sources, xmax, False, signal_choice, lens_type='NFW')

        mass = np.sum(candidate_lenses.mass)
        candidate_num = len(candidate_lenses.x)
        masses.append(mass)
        candidate_number.append(candidate_num)

    # Save the results to a file
    results = [ID, masses[0], masses[1], masses[2], masses[3], candidate_number[0], candidate_number[1], candidate_number[2], candidate_number[3]]
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
        [True, True, True], # All signals
        [True, True, False], # Shear and Flexion
        [False, True, True], # Flexion and G-Flexion
        [True, False, True] # Shear and G-Flexion
    ]

    # Build halos
    halos = find_halos(IDs, z) # halos is a dictionary of Halo objects, keyed by ID
    source_catalogue = {} # Dictionary to hold the source catalogues
    xmax_values = [] # List to hold the maximum extent of each field

    for ID in IDs:
        # Build the lenses and sources
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)
        '''
        # Create a new source object, without noise (Note from the future - why?)
        xs = sources.x
        ys = sources.y
        clean_source = source_obj.Source(xs, ys, np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.zeros_like(xs), np.ones(len(xs)) * 0.1, np.ones(len(xs)) * 0.01, np.ones(len(xs)) * 0.02)
        if lensing_type == 'NFW':
            clean_source.apply_lensing(halos[ID], lens_type='NFW', z_source=z_source)
        elif lensing_type == 'SIS':
            tE = halos[ID].calc_corresponding_einstein_radius(z_source)
            print('Einstein radius: {}'.format(tE))
            lens = halo_obj.SIS_Lens(halos[ID].x, halos[ID].y, tE, 0)
            clean_source.apply_lensing(lens, lens_type='SIS')
        else:
            raise ValueError('Invalid lensing type specified')

        source_catalogue[ID] = clean_source
        xmax_values.append(xmax)
        '''
        source_catalogue[ID] = sources
        xmax_values.append(xmax)

    print('Halo and Source objects loaded...')

    # Prepare the arguments for each task
    tasks = [(ID, signal_choices, source_catalogue[ID], xmax_values[i]) for i, ID in enumerate(IDs)]
    # Repeat each task N_test times
    tasks = [task for task in tasks for _ in range(N_test)]

    # Process pool
    with Pool() as pool:
        results = pool.map(run_single_test, tasks)

    # Save the results to a file
    results = np.array(results).reshape(-1, 9)
    with open(result_file, 'w') as f:
        f.write('ID,Mass_all_signals,Mass_gamma_F,Mass_F_G,Mass_gamma_G,Nfound_all_signals,Nfound_gamma_F,Nfound_F_G,Nfound_gamma_G\n')
        for i in range(len(results)):
            f.write('{}\n'.format(', '.join(results[i].astype(str))))

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

    # run_test_parallel(ID_file, result_file, z_chosen, Ntrials, lensing_type='NFW')
    stop = time.time()
    print('Time taken: {}'.format(stop - start))
    build_mass_correlation_plot(ID_file, result_file, plot_name)


if __name__ == '__main__':
    # process_md_set(15)
    # raise SystemExit
    # Pick out a halo, run the pipeline, look at the results

    ID_file = 'Output/MDARK/Test15/ID_file_15.csv'
    IDs = []
    with open(ID_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(int(ID))
    
    IDs = np.array(IDs)
    z = 0.194
    
    for ID in IDs:
        halos = find_halos([ID], z)
        halos[ID], sources, xmax = build_lensing_field(halos[ID], z)
        candidate_lenses, _ = main.fit_lensing_field(sources, xmax, True, [True, True, False], lens_type='NFW')

        plt.figure()
        plt.scatter(sources.x, sources.y, s=10, color='black', alpha=0.5, label='Sources')
        plt.scatter(halos[ID].x, halos[ID].y, s=100, color='red', label='Lenses', marker='x')
        plt.scatter(candidate_lenses.x, candidate_lenses.y, s=100, color='blue', label='Candidates', marker='o')
        # Label candidates with their masses
        for i in range(len(candidate_lenses.x)):
            plt.text(candidate_lenses.x[i], candidate_lenses.y[i], '{:.2e}'.format(candidate_lenses.mass[i]), fontsize=8)
        plt.legend()
        plt.xlabel('x [arcseconds]')
        plt.ylabel('y [arcseconds]')
        plt.title('Cluster ID: {} \n True Mass: {:.2e} $M_{{\odot}}$ \n Candidate Mass: {:.2e} $M_{{\odot}}$'.format(ID, np.sum(halos[ID].mass), np.sum(candidate_lenses.mass)))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('Output/MDARK/pipeline_visualization/cluster_{}.png'.format(ID))