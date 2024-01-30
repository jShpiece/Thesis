import numpy as np 
import matplotlib.pyplot as plt
import time
import pandas as pd
from astropy.visualization import hist as fancy_hist

dir = 'MDARK/'
column_names = ['MainHaloID', ' Total Mass', ' Redshift', 'Halo Number', ' Mass Fraction', ' Characteristic Size'] # Column names for the key files
plt.style.use('scientific_presentation.mplstyle') # Use the scientific presentation style sheet for all plots


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
    for chunk in chunks:
        Mass.append(chunk[' Total Mass'].values)
    
    Mass = np.concatenate(Mass)
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


def count_clusters(z):
    file = dir + 'fixed_key_{}.MDARK'.format(z)
    chunks = chunk_data(file)

    count1 = 0
    count2 = 0

    for chunk in chunks:
        # Count each cluster with mass > 10^14 M_sun
        count1 += np.sum(chunk[' Total Mass'].values > 1e14)
        # Count each cluster of mass > 10^14 M_sun and more than 1 halo
        count2 += np.sum((chunk[' Total Mass'].values > 1e14) & (chunk[' Halo Number'].values > 1))

    print('Number of clusters with mass > 10^14 M_sun: {}'.format(count1))
    print('Number of clusters with mass > 10^14 M_sun and more than 1 halo: {}'.format(count2))

if __name__ == '__main__':
    zs = [0.194, 0.221, 0.248, 0.276]

    for z in zs:
        count_clusters(z)
        print('Done with z = {}'.format(z))