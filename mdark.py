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

        mass = self.mass * 2 * 10**30 # Mass in kilograms
        # Print the mass in exponent form (mass is currently in kg as a numpy array)
        print(f'Mass: {mass[0]:.2e} kg')
        R200 = R200 * 3.086 * 10**19 # Convert to meters

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
    chunk_size = 10000  # Adjust based on your memory constraints
    chunks = pd.read_csv(file, dtype=data_types, chunksize=chunk_size)
    return chunks


def plot_mass_dist(z):
    file = dir + 'fixed_key_{}.MDARK'.format(z) 
    chunks = chunk_data(file)

    Mass = []
    size = []
    for chunk in chunks:
        Mass.append(chunk[' Total Mass'].values)
        size.append(chunk[' Characteristic Size'].values)
    
    Mass = np.concatenate(Mass)
    size = np.concatenate(size)
    # Plot the mass distribution (log scale)
    # Use 1000 bins
    
    fig, ax = plt.subplots()
    fancy_hist(Mass, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\rm dark}$ [$M_{\odot}$]')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig(dir + 'mass_dist_{}.png'.format(z))

    fig, ax = plt.subplots()
    fancy_hist(size, bins=1000, histtype='step', density=True, ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$R_{\rm dark}$ [Mpc]')
    ax.set_ylabel('Probability Density')
    ax.set_title(r'$Multidark: z = {}$'.format(z))
    fig.tight_layout()
    fig.savefig(dir + 'size_dist_{}.png'.format(z))


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
    output = []
    for chunk in chunks:
        mask = chunk['MainHaloID'] == ID
        if np.any(mask):
            output.append(chunk[mask])
    # Flatten the list
    output = pd.concat(output)
    
    # Now we have the correct objects, we can return the relevant information
    xhalo = output['x'].values
    yhalo = output['y'].values
    zhalo = output['z'].values
    chalo = output['concentration_NFW'].values
    masshalo = output['HaloMass'].values

    halos = Halo(xhalo, yhalo, zhalo, chalo, masshalo)

    return halos


def choose_ID(z, mass_range, min_halos):
    # Given a set of criteria, choose a cluster ID

    file = dir + 'fixed_key_{}.MDARK'.format(z)
    chunks = chunk_data(file)

    # Choose a random cluster with mass in the given range
    # and more than 1 halo
    options = []
    for chunk in chunks:
        mask = (chunk[' Total Mass'].values > mass_range[0]) & (chunk[' Total Mass'].values < mass_range[1]) & (chunk[' Halo Number'].values > min_halos)        
        if np.any(mask):
            options.append(chunk[mask]['MainHaloID'].values)
    # Flatten the list
    options = np.concatenate(options)
    # Choose one of the clusters at random
    IDs = np.random.choice(options)

    return IDs
    
    # Now we have the correct objects, we can return the relevant information


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
    4. Generate a convergence map from the candidate lenses, 
        compute the mass of each lens, and compare to the
        input mass.
    5. Generate a ROC curve for the pipeline
    6. Generate a confusion matrix
    7. Save the results to a file
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
    print(len(lenses.x))
    print(np.mean(lenses.x))
    print(np.std(lenses.x))

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # to work correctly
    centroid = np.mean(lenses.x), np.mean(lenses.y)
    lenses.x -= np.mean(lenses.x)
    lenses.y -= np.mean(lenses.y)

    xmax = np.max([np.max(lenses.x), np.max(lenses.y)])
    # Set the maximum extent of the field of view
    # to be the maximum extent of the lenses

    # Generate a set of background galaxies
    ns = 0.01
    Nsource = int(ns * (xmax*2)**2)
    sources = utils.createSources(lenses, Nsource, randompos=True, sigs=0.1, sigf=0.01, sigg=0.02, xmax=xmax)

    candidate_lenses, _ = pipeline.fit_lensing_field(sources, xmax, [True, True, True])

    return lenses, candidate_lenses


def build_test_set(Nhalos, z):
    # Select a number of halos, spaced evenly in log space across the mass range
    M_min = 1e13
    M_max = 1e15
    masses = np.logspace(np.log10(M_min), np.log10(M_max), Nhalos)
    IDs = []
    # mass range will be between this mass and the next mass, for each ID
    for i in range(Nhalos):
        ID = choose_ID(z, (masses[i], masses[i+1]), 2)
        IDs.append(ID)
    return IDs



def process_results(halos, lenses, candidate_lenses, z):
    # Given the results of the pipeline, process the results
    # We can directly compare the input lenses to the candidate lenses
    # We can also compare the true mass from the halos to the inferred mass
    # from the candidate lenses

    # Get the true mass of the system
    xmax = np.max([np.max(candidate_lenses.x), np.max(candidate_lenses.y)])
    extent = [-xmax, xmax, -xmax, xmax]
    kappa = utils.calculate_kappa(candidate_lenses, extent, 0.1, 0.01, 0.02)
    mass = utils.calculate_mass(kappa, z, 0.5, 1)
    true_mass = np.sum(halos.mass)

    # Now compare the lenses to the candidate lenses
    
    # Match the lenses to the candidate lenses
    # This is a nearest neighbour problem
    # We can use the sklearn.metrics.pairwise_distances function
    # to find the nearest neighbour for each lens
    # We can then compare the true mass to the inferred mass
    # We can also look at the number of lenses that were correctly
    # identified, the number of false positives, and the number of
    # false negatives


    x_true = lenses.x
    y_true = lenses.y
    x_pred = candidate_lenses.x
    y_pred = candidate_lenses.y

    # Create a distance matrix
    distance_matrix = sklearn.metrics.pairwise_distances(x_true, x_pred, y_true, y_pred)

    # Find the nearest neighbour for each lens
    nearest_neighbour = np.argmin(distance_matrix, axis=1)

    # Compare the strengths of the lenses
    true_strength = lenses.te
    pred_strength = candidate_lenses.te

    # Assess the quality of the match
    # Match lenses to candidate lenses based on distance
    # If they are within a certain distance, and the strengths
    # are similar, then we have a match

    # Also, we aren't going to use a confusion matrix, because 
    # we don't have a binary classification problem
    # We have a regression problem

    # Lets get started. First, match the lenses to the candidate lenses
    true_positives = 0
    for i in range(len(nearest_neighbour)):
        # If the distance is small, and the strengths are similar
        # then we have a match
        if distance_matrix[i, nearest_neighbour[i]] < 0.1 and abs(true_strength[i] - pred_strength[nearest_neighbour[i]]) < 1:
            true_positives += 1
    
    # Now, count the false positives
    false_positives = len(candidate_lenses) - true_positives

    # Now, count the false negatives
    false_negatives = len(lenses) - true_positives

    # Now, we can save the results to a file
    results = {'true_mass': true_mass, 
                'inferred_mass': mass,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives}
    
    # Save the results to a file
    with open('results.txt', 'w') as f:
        f.write('True Mass: {}\n'.format(true_mass))
        f.write('Inferred Mass: {}\n'.format(mass))
        f.write('True Positives: {}\n'.format(true_positives))
        f.write('False Positives: {}\n'.format(false_positives))
        f.write('False Negatives: {}\n'.format(false_negatives))
    return results

if __name__ == '__main__':
    zs = [0.194, 0.221, 0.248, 0.276]

    z = 0.194
    d = cosmo.angular_diameter_distance(z).to(u.meter)
    # Given a distance in Mpc, convert to arcseconds

    x = np.linspace(0, 1000, 1000)

    y = ((x * 3.086 * 10**22) / d).value * 206265

    plt.plot(x, y)
    plt.show()

    # plot_mass_dist(0.194)

    raise SystemExit

    ID = choose_ID(0.194, (1e13, 1e14), 2)
    halos = find_halos(ID, 0.194)
    lenses = run_analysis(halos, 0.194)
    plt.figure()
    plt.scatter(lenses.x, lenses.y, c = lenses.te, cmap='plasma', s = lenses.te * 100, alpha=0.5)
    plt.colorbar()
    plt.show()

    raise SystemExit

    R200 = halos.calc_R200(0.194)
    lenses = halos.build_lenses(0.194, R200)
    print(lenses.x, lenses.y, lenses.te)
    stop = time.time()
    print('Time taken: {}'.format(stop - start))


    raise SystemExit

    for z in zs:
        count_clusters(z)
        print('Done with z = {}'.format(z))