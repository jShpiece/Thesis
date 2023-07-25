#Subhalo mass function
import numpy as np
import matplotlib.pyplot as plt

def cumulative_mass_function(mu, mu1 = 0.01, mu_cut = 0.1):
    #mu is the mass of the subhalo divided by the mass of the host halo
    return (mu/mu1)**(-0.94) * np.exp(-(mu/mu_cut)**1.2)

def stn_flexion(eR, n, sigma, rmin, rmax):
    term1 = eR * np.sqrt(np.pi * n) / (sigma * rmin)
    term2 = np.log(rmax / rmin) / np.sqrt(rmax**2 / rmin**2 - 1)
    return term1 * term2

def stn_shear(eR, n, sigma, rmin, rmax):
    term1 = eR * np.sqrt(np.pi * n) / (sigma)
    term2 = (1 - rmin/rmax) / np.sqrt(1 - (rmin/rmax)**2)
    return term1 * term2
 
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

#Now load the flexion data for Abell 2744
file = 'a2744_updated_simplified.csv'

#Read in the data
data = np.loadtxt(file, delimiter=',', skiprows=1)

#Order: ID, x, y, f1, f2
#Read in all the data
IDs = data[:,0]
xs = data[:,1]
ys = data[:,2]
f1s = data[:,3]
f2s = data[:,4]

#Compute the angular size of the cluster
size = np.max(np.sqrt(xs**2 + ys**2))
N = len(IDs)
#The number density n of subhalos is given by n = N / (size**2)
n = N / (size**2)

sigma = 10**-2
rmin = 1
rmax = 20

#Calculate the signal to noise ratio as a function of eR
eR = np.linspace(10e-3, 1, 1000)
stnf = stn_flexion(eR, n, sigma, rmin, rmax)
stns = stn_shear(eR, n, sigma, rmin, rmax)

plt.plot(eR, stnf, label='Flexion')
plt.plot(eR, stns, label='Shear')
plt.xlabel('eR')
plt.ylabel('Signal to noise ratio')
plt.hlines(1, 0, 1, colors='r', linestyles='dashed')
plt.title('Signal to noise ratio as a function of Einstein radius')
plt.xscale('log')
plt.legend()
plt.show()