#Subhalo mass function
import numpy as np
import matplotlib.pyplot as plt

def cumulative_mass_function(mu, mu1 = 0.01, mu_cut = 0.1):
    #mu is the mass of the subhalo divided by the mass of the host halo
    return (mu/mu1)**(-0.94) * np.exp(-(mu/mu_cut)**1.2)

#Plotting
mu = np.logspace(-5, -2, 1000)

'''
plt.plot(mu, cumulative_mass_function(mu))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\mu = M_{sub}/M_{host}$')
plt.ylabel(r'$N(>\mu)$')
'''

#Let us now calculate the number of subhalos per mass bin
#Take Abell 2744 as an example
#The mass of the host halo is 1.6e15 Msun
#The mass of the largest subhalo is 1.6e14 Msun
#Let us assume that the mass of the smallest subhalo is 1e9 Msun
#The number of subhalos per mass bin is then

#The mass of the host halo
M_host = 1.6e15
#The mass of the largest subhalo
M_largest = 1.6e14
#The mass of the smallest subhalo
M_smallest = 1e9


#Create this plot as a histogram
plt.plot(mu * M_host, cumulative_mass_function(mu))
#Label each bin with the number of subhalos in that bin
'''
for i in range(len(mass_bins)-1):
    plt.text(mass_bins[i], cumulative_mass_function[i], str(int(N_subhalos[i])))
'''
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{sub}$')
plt.ylabel(r'$N(>M_{sub})$')
plt.title('Cumulative subhalo mass function for Abell 2744 - like cluster')
plt.show()

#Okay, so this gives us several hundred subhalos at the low mass end
#What is the angular separation between these subhalos?
#Abell 2744 is at a redshift of 0.308
#The angular diameter distance to Abell 2744 is 1.2 Gpc
#Let Abell 2744 span 350 x 350 arcsec
#Let us assume that the subhalos are distributed uniformly in the cluster
#Let the number of subhalos be 500

n_small = 900
d = 350 #arcsec
n_dist = n_small / (d**2)
print(n_dist)
avg_sep = np.sqrt(1/n_dist)
print(avg_sep)