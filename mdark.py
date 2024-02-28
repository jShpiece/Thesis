import numpy as np 
import matplotlib.pyplot as plt
import time
import pandas as pd
from astropy.visualization import hist as fancy_hist
import pipeline
import utils
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
import sklearn.metrics
import pickle

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
    def __init__(self, x, y, z, c, mass):
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.mass = mass
    
    def calc_R200(self, z):
        omega_crit = 0.27
        omega_z = 0.27 * (1 + z)**3 / (0.27 * (1 + z)**3 + 0.73)
        R200 = (1.63 * 10**-2) * (self.mass / h**-1)**(1/3) * (omega_crit / omega_z)**(-1/3) * (1 + z)**(-1) * h**-1 # In Kpc
        return R200
    
    def build_lenses(self, z, R200):
        z_s = 0.8
        # Given a set of halos, build the lenses
        # Main computational step is getting the einstein radius for each lens
        D_s = cosmo.angular_diameter_distance(z_s).to(u.meter)
        D_ls = cosmo.angular_diameter_distance_z1z2(z, z_s).to(u.meter)

        mass = self.mass * M_solar # Mass in kilograms
        R200 = (R200 / 206265) * D_s.value # Convert to meters

        eR = ((2 * np.pi * G) / (c**2) * (D_ls / D_s) * (mass / R200)).value * 206265 # Convert to arcseconds

        return pipeline.Lens(np.array(self.x), np.array(self.y), np.array(eR), np.zeros_like(self.x))


def fix_file(z):
    # Fix the key files

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

    df = pd.read_csv(file, nrows=0)

    # Read the CSV file in chunks
    chunk_size = 50000  # Adjust based on your memory constraints
    chunks = pd.read_csv(file, dtype=data_types, chunksize=chunk_size)
    return chunks


def plot_mass_dist(z):
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
    # Remove nan values
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


def count_clusters(z):
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
    chunks = chunk_data(file)
    # Find all the halos with the given ID

    # Create an empty DataFrame to store the results
    filtered_data = pd.DataFrame()

    # Read the CSV file in chunks
    for chunk in pd.read_csv(file, chunksize=10000):
        # Filter the chunk for the specified ObjectID
        filtered_chunk = chunk[chunk['MainHaloID'] == ID]

        # Concatenate the filtered chunk to the final DataFrame
        filtered_data = pd.concat([filtered_data, filtered_chunk])

    # Now we have the correct objects, we can return the relevant information
    xhalo = filtered_data['x'].values
    yhalo = filtered_data['y'].values
    zhalo = filtered_data['z'].values
    chalo = filtered_data['concentration_NFW'].values
    masshalo = filtered_data['HaloMass'].values

    halos = Halo(xhalo, yhalo, zhalo, chalo, masshalo)

    print('Found {} halos'.format(len(halos.x)))
    print('Mass: {:.2e}'.format(np.sum(halos.mass)))

    return halos


def choose_ID(z, mass_range, halo_range, substructure_range, size_range):
    # Given a set of criteria, choose a cluster ID
    file = dir + 'fixed_key_{}.MDARK'.format(z)
    chunks = chunk_data(file)

    # Choose a random cluster with mass in the given range
    # and more than 1 halo
    rows = []
    for chunk in chunks:
        IDs = chunk['MainHaloID'].values
        masses = chunk[' Total Mass'].values
        halos = chunk[' Halo Number'].values
        substructure = chunk[' Mass Fraction'].values
        size = chunk[' Characteristic Size'].values

        # Apply the criteria
        mass_criteria = (masses > mass_range[0]) & (masses < mass_range[1])
        halo_criteria = (halos > halo_range[0]) & (halos < halo_range[1])
        substructure_criteria = (substructure > substructure_range[0]) & (substructure < substructure_range[1])
        size_criteria = (size > size_range[0]) & (size < size_range[1])

        # Find all clusters that satisfy all criteria
        combined_criteria = mass_criteria & substructure_criteria

        if np.any(combined_criteria):
            valid_ids = IDs[combined_criteria]
            # Add these to the list of rows
            rows.append(chunk[chunk['MainHaloID'].isin(valid_ids)])
    # Turn this into a single dataframe
    rows = pd.concat(rows)

    # Choose a random cluster from the list of valid rows
    if len(rows) > 0:
        # Choose a single row
        row = rows.sample(n=1)
        print('Found a cluster with mass in range')
        return row
    else:
        return None



def run_analysis(halos, z):
    '''
    Given a set of halos, run the analysis
    This involves the following steps

    1. Compute the einstein radius for each cluster, from 
        M200 and R200
    2. Generate a set of background galaxies, with 
        random positions and redshifts. Apply a shear
        and flexion to each galaxy. 
    3. Run these source galaxies through the pipeline. This
        will generate a list of candidate lenses, which 
        will be compared to the input list of halos. 
    '''

    # Convert coordinates to arcseconds
    # Currently, coordinates are in Mpc
    # We need to convert to arcseconds
    d = cosmo.angular_diameter_distance(z).to(u.meter)
    # Convert to meters
    x = halos.x * 3.086 * 10**22
    y = halos.y * 3.086 * 10**22
    # Convert to arcseconds
    x = (x / d).value * 206265
    y = (y / d).value * 206265

    halos.x = x
    halos.y = y

    lenses = halos.build_lenses(z, halos.calc_R200(z))

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # to work correctly
    centroid = np.mean(lenses.x), np.mean(lenses.y)
    lenses.x -= centroid[0]
    lenses.y -= centroid[1]
    print('Centroid: {}'.format(centroid))

    xmax = np.max((lenses.x**2 + lenses.y**2)**0.5)
    print('xmax: {}'.format(xmax))
    xmax = np.min([xmax, 200])

    # Set the maximum extent of the field of view
    # to be the maximum extent of the lenses

    # Generate a set of background galaxies
    ns = 0.005
    Nsource = int(ns * np.pi * (xmax)**2) # Number of sources

    sources = utils.createSources(lenses, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax)
    candidate_lenses, _ = pipeline.fit_lensing_field(sources, xmax, flags=False, use_flags= [True, True, True])

    return lenses, candidate_lenses, sources


def build_test_set(Nhalos, z, file_name):
    # Select a number of halos, spaced evenly in log space across the mass range
    M_min = 1e12
    M_max = 1e16
    Nhalo_min = 1
    Nhalo_max = 20
    substructure_min = 0.01
    substructure_max = 0.1
    size_min = 50
    size_max = 500
    mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), Nhalos+1)

    # Choose a cluster in each mass bin
    rows = []
    for i in range(Nhalos):
        mass_range = [mass_bins[i], mass_bins[i+1]]
        halo_range = [Nhalo_min, Nhalo_max]
        substructure_range = [substructure_min, substructure_max]
        size_range = [size_min, size_max]
        row = choose_ID(z, mass_range, halo_range, substructure_range, size_range)
        if row is not None:
            print('Found a cluster in mass bin {}'.format(i))
            rows.append(row)

    # Save the rows to a file
    with open(file_name, 'w') as f:
        f.write('ID, Mass, Halo Number, Mass Fraction, Size\n')
        for row in rows:
            for i in range(len(row)):
                f.write('{}, {}, {}, {}, {}\n'.format(row['MainHaloID'].values[i], row[' Total Mass'].values[i], row[' Halo Number'].values[i], row[' Mass Fraction'].values[i], row[' Characteristic Size'].values[i]))
    
    return 


def process_results(halos, lenses, candidate_lenses, z, label):
    # Given the results of the pipeline, process the results
    # We can directly compare the input lenses to the candidate lenses
    # We can also compare the true mass from the halos to the inferred mass
    # from the candidate lenses

    # Get the true mass of the system
    xmax = np.max([np.max(candidate_lenses.x), np.max(candidate_lenses.y)])
    extent = [-xmax, xmax, -xmax, xmax]
    _,_,kappa = utils.calculate_kappa(candidate_lenses, extent, 5)
    mass = utils.calculate_mass(kappa, z, 0.5, 1)
    true_mass = np.sum(halos.mass)
    return mass, true_mass


def make_catalogue(sources, name):
    # Given a set of sources, save the catalogue to a file
    # Build this as a csv file, with the following columns
    # x, y, e1, e2, f1, f2, g1, g2
    # This is the format expected by the pipeline

    # Save the catalogue to a file
    with open(name, 'w') as f:
        f.write('x, y, e1, e2, f1, f2, g1, g2\n')
        for i in range(len(sources.x)):
            f.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format(sources.x[i], sources.y[i], sources.e1[i], sources.e2[i], sources.f1[i], sources.f2[i], sources.g1[i], sources.g2[i]))
    f.close()


def run_test(ID_file):
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
    with open('Data/MDARK_Test/results_2.csv', 'w') as f:
        f.write('ID, Mass, True Mass, N_halos, N_candidates\n')
    f.close()

    for ID in IDs:
        z = 0.194
        label = 'Data/MDARK_Test/{}_test'.format(ID)

        halos = find_halos(ID, z)

        lenses, candidate_lenses, sources = run_analysis(halos, z)
        mass, true_mass = process_results(halos, lenses, candidate_lenses, z, label+'_results.txt')
        with open('Data/MDARK_Test/results.csv', 'a') as f:
            f.write('{}, {}, {}, {}, {}\n'.format(ID, mass, true_mass, len(lenses.x), len(candidate_lenses.x)))
        f.close()

        # Save the sources - these can be passed off to other pipelines for comparison
        make_catalogue(sources, label+'_sources.csv')

        # Also save the lenses - use pickle to save these objects
        with open(label+'_lenses.pkl', 'wb') as f:
            pickle.dump(lenses, f)
        with open(label+'_candidate_lenses.pkl', 'wb') as f:
            pickle.dump(candidate_lenses, f)

    print('Done!')


def build_mass_correlation_plot(file_name):
    # Open the results file and read in the data
    results = pd.read_csv(file_name)
    # Get the mass and true mass
    mass = results[' Mass'].values
    true_mass = results[' True Mass'].values

    # Plot the results, calculate the correlation coefficient
    fig, ax = plt.subplots()
    ax.scatter(true_mass, mass, s=10, color='black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Add a line of best fit
    m, b = np.polyfit(np.log10(true_mass), np.log10(mass), 1)
    x = np.linspace(1e13, 1e15, 100)
    y = 10**b * x**m
    ax.plot(x, y, color='red', label='Best Fit: {:.2f}'.format(m))
    # Also add a 1:1 line
    ax.plot(x, x, color='blue', label='1:1')
    ax.legend()
    ax.set_xlabel(r'$M_{\rm true}$ [$M_{\odot}$]')
    ax.set_ylabel(r'$M_{\rm inferred}$ [$M_{\odot}$]')
    ax.set_title('Multidark: Mass Inference \n Correlation Coefficient: {:.2f}'.format(np.corrcoef(true_mass, mass)[0, 1]))
    fig.tight_layout()
    fig.savefig('Data/MDARK_Test/mass_inference.png')


if __name__ == '__main__':
    zs = [0.194, 0.221, 0.248, 0.276]

    file = 'MDARK/Halos_0.194.MDARK'

    build_test_set(2, 0.194, 'Data/MDARK_Test/ID_list_5.csv')

    print('Done building test set')

    IDs = []
    # Load the list of IDs - this is a csv file
    with open('Data/MDARK_Test/ID_list_5.csv', 'r') as f:
        lines = f.readlines() # Read the lines
        # Skip the first line
        lines = lines[1:]
        for line in lines:
            ID = line.split(',')[0]
            IDs.append(ID)
    
    for ID in IDs:
        halos = find_halos(ID, 0.194)