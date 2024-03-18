import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import hist as fancy_hist
import pipeline
import utils
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
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

        f1_1 = np.zeros(len(xs))
        f1_2 = np.zeros(len(ys))

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
            f1_1[i] += np.sum(F_mag * np.cos(phi))
            f1_2[i] += np.sum(F_mag * np.sin(phi))

        return f1_1, f1_2


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

        f2_1 = np.zeros(len(xs))
        f2_2 = np.zeros(len(ys))

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
            f2_1[i] += np.sum(G_mag * np.cos(3 * phi))
            f2_2[i] += np.sum(G_mag * np.sin(3 * phi))
        
        return f2_1, f2_2


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


def build_lensing_field(halos, z):
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

    lenses = halos # Placeholder for the lenses

    # Center the lenses at (0, 0)
    # This is a necessary step for the pipeline
    # to work correctly
    # Let the centroid be the location of the most massive halo
    # Offset by a small amount, so that we are looking at the
    # center of the cluster, but not directly at the most massive halo

    largest_halo = np.argmax(lenses.mass)
    centroid = [lenses.x[largest_halo], lenses.y[largest_halo]]
    lenses.x -= centroid[0] + np.random.uniform(-10, 10)
    lenses.y -= centroid[1] + np.random.uniform(-10, 10)

    xmax = np.max((lenses.x**2 + lenses.y**2)**0.5)
    print('xmax: {}'.format(xmax))
    
    # Don't allow the field of view to be larger than 2 arcminutes - or smaller than 1 arcminute
    xmax = np.min([xmax, 2*60])
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


def compute_masses(candidate_lenses, z):
    # Given the results of the pipeline, process the results
    # We can directly compare the input lenses to the candidate lenses
    # We can also compare the true mass from the halos to the inferred mass
    # from the candidate lenses

    # Get the true mass of the system
    xmax = np.max([np.max(candidate_lenses.x), np.max(candidate_lenses.y)])
    extent = [-xmax, xmax, -xmax, xmax]
    _,_,kappa = utils.calculate_kappa(candidate_lenses, extent, 5)
    mass = utils.calculate_mass(kappa, z, 0.5, 1)
    return mass


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


if __name__ == '__main__':
    zs = [0.194, 0.221, 0.248, 0.276]

    file = 'MDARK/Halos_0.194.MDARK'
    test_number = 6
    ID_file = 'Data/MDARK_Test/Test{}/ID_file_{}.csv'.format(test_number, test_number)
    result_file = 'Data/MDARK_Test/Test{}/results_{}.csv'.format(test_number, test_number)
    plot_name = 'Images/MDARK/mass_correlation_{}.png'.format(test_number)

    # build_test_set(30, zs[0], ID_file)
    run_test(ID_file, result_file, zs[0])
    build_mass_correlation_plot(result_file, plot_name)

    raise SystemExit

    '''
    TEST TO MAKE SURE NFW PRODUCED LENSING SIGNALS ARE REASONABLE
    '''

    # pick a random cluster
    ID = 11494083558
    halos = find_halos(ID, zs[0])
    # Convert x and y to arcseconds (from Mpc)
    x = halos.x * 3.086 * 10**22 / cosmo.angular_diameter_distance(zs[0]).to(u.meter).value * 206265
    y = halos.y * 3.086 * 10**22 / cosmo.angular_diameter_distance(zs[0]).to(u.meter).value * 206265

    centroid = np.mean(x), np.mean(y)
    x = x - centroid[0]
    y = y - centroid[1]
    xc = x[0] + 5
    yc = y[0]

    # update the halos object
    halos.x = x
    halos.y = y

    r200 = halos.calc_R200()

    # Evaluate the lensing signals at this centroid
    gamma1, gamma2 = halos.calc_shear_signal([xc], [yc])
    f1, f2 = halos.calc_F_signal([xc], [yc])
    g1, g2 = halos.calc_G_signal([xc], [yc])

    print('For cluster {}:'.format(ID))
    print('With {} halos'.format(len(halos.x)))
    print('And a total mass of {:.2e}'.format(np.sum(halos.mass)))
    print('The r200 is')
    print(r200)
    print('And 5 arcseconds from the primary halo, the signals are:')
    print('Gamma: ({}, {})'.format(gamma1, gamma2))
    print('F: ({}, {})'.format(f1, f2))
    print('G: ({}, {})'.format(g1, g2))
    print('Analysis complete!')