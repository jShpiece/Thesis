# Test the method for getting projected mass profiles from a lens model

import numpy as np
import utils
import halo_obj
import matplotlib.pyplot as plt

mass = 1e14 # Msun
halo = halo_obj.NFW_Lens([0],[0],[0],[0],[mass],0.308,[0])
halo.calculate_concentration()

R_200 = halo.calc_R200()[0] # in meters
# Convert to kpc
R_200 = R_200 * 3.24078e-20

r = np.linspace(R_200 / 2, R_200 * 3/2, 100)

# Calculate the projected mass profile
projected_mass = utils.nfw_projected_mass(halo, r)

plt.figure()
plt.plot(r, projected_mass)
plt.yscale('log')
plt.xlabel('Radius (kpc)')
plt.ylabel('Projected Mass (Msun)')
plt.title('Projected Mass Profile')
plt.vlines(R_200, 1e13, 1e15, color='r', linestyle='--', label='R_200')
plt.hlines(mass, R_200 / 2, R_200 * 3/2, color='r', linestyle='--', label='M_200')
plt.xlim(R_200 / 2, R_200 * 3/2)
plt.ylim(1e13, 1e15)
plt.show()


# find the projected mass at R_200
projected_mass_R_200 = utils.nfw_projected_mass(halo, R_200)
# This should be equal to the mass 
assert np.isclose(projected_mass_R_200, mass, rtol=1e-2), 'projected_mass_R_200 = {:.3e}, mass = {:.3e}, diff = {:.3e}'.format(projected_mass_R_200, mass, projected_mass_R_200 - mass)