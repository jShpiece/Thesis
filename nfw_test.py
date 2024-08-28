import numpy as np
import matplotlib.pyplot as plt
import pipeline
from astropy.constants import c, G
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo

M_solar = 1.989e30 # Solar mass in kg


def angular_diameter_distances(z1, z2):
    dl = cosmo.angular_diameter_distance(z1).to(u.m).value
    ds = cosmo.angular_diameter_distance(z2).to(u.m).value
    dls = cosmo.angular_diameter_distance_z1z2(z1, z2).to(u.m).value
    return dl, ds, dls


def critical_surface_density(z1, z2):
    dl, ds, dls = angular_diameter_distances(z1, z2)
    kappa_c = (c.value**2 / (4 * np.pi * G.value)) * (ds / (dl * dls))
    return kappa_c


def calc_signals(halos, sources, z_source):

    # Begin with the distances and angles between the source and the halo
    dx = sources.x - halos.x[:, np.newaxis]
    dy = sources.y - halos.y[:, np.newaxis]
    r = np.sqrt(dx**2 + dy**2)
    r = np.where(r == 0, 0.01, r) # Avoid division by zero
    cos_phi = dx / r # Cosine of the angle between the source and the halo
    sin_phi = dy / r # Sine of the angle between the source and the halo
    cos2phi = cos_phi**2 - sin_phi**2 # Cosine of 2*phi
    sin2phi = 2 * cos_phi * sin_phi # Sine of 2*phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi # Cosine of 3*phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi # Sine of 3*phi

    # Compute lensing magnitudes (in arcseconds)
    shear_mag = - kappa_s[:, np.newaxis] * term_2
    flex_mag = (-2 * flexion_s[:, np.newaxis]) * ((2 * x * term_1 / (x**2 - 1)**2) - term_3 / (x**2 - 1)) / 206265 # Using Wright & Brainerd notation
    g_flex_mag = (2 * flexion_s[:, np.newaxis]) * ((8 / x**3) * np.log(x / 2) + ((3/x)*(1 - 2*x**2) + term_4) / (x**2 - 1)**2) / 206265

    # Sum over all halos, resulting array should have shape (n_sources,)
    shear_1 = np.sum(shear_mag * cos2phi, axis=0)
    shear_2 = np.sum(shear_mag * sin2phi, axis=0)
    flexion_1 = np.sum(flex_mag * cos_phi, axis=0)
    flexion_2 = np.sum(flex_mag * sin_phi, axis=0)
    g_flexion_1 = np.sum(g_flex_mag * cos3phi, axis=0)
    g_flexion_2 = np.sum(g_flex_mag * sin3phi, axis=0)

    return shear_1, shear_2, flexion_1, flexion_2, g_flexion_1, g_flexion_2


def calculate_concentration(self):
    # Compute the concentration parameter for each halo
    # This is done with the Duffy et al. (2008) relation
    # This relation is valid for 0 < z < 2 - this covers the range of redshifts we are interested in
    # Note - numpy doesn't like negative powers on lists, even if the answer isn't complex
    # Get around this by taking taking the absolute value of arrays (then multiplying by -1 if necessary) (actually don't do this - mass and concentration should be positive)
    # It also breaks down if the mass is 0 (we'll be dividing by the mass)
    self.mass += 1e-10 # Add a small value to the mass to avoid division by zero
    self.concentration = 5.71 * (np.abs(self.mass) / (2 * 10**12))**(-0.084) * (1 + self.redshift)**(-0.47) 


def calc_R200(mass, redshift):
    # Compute the R200 radius for each halo
    rho_c = cosmo.critical_density(redshift).to(u.kg / u.m**3).value
    R200 = (3 / (800 * np.pi) * (np.abs(mass) * M_solar / rho_c))**(1/3) # In meters
    # Convert to arcseconds
    R200_arcsec = (R200 / cosmo.angular_diameter_distance(redshift).to(u.meter).value) * 206265
    return R200, R200_arcsec


def calc_delta_c(concentration):
    # Compute the characteristic density contrast for each halo
    delta_c = (200/3) * (concentration**3) / (np.log(1 + concentration) - concentration / (1 + concentration))
    return delta_c


if __name__ == '__main__':
    z_halo = 0.35
    z_source = 0.8
    mass = 1e12
    concentration = 7.2

    x_range = np.linspace(0.1, 10, 1000)
    # Correct for small values of x
    # x_range = np.where(np.abs(x_range) < 0.01, 0.01, x_range)

    # Define angular diameter distances
    Dl = angular_diameter_distances(z_halo, z_source)[0]

    # Compute R200
    r200, r200_arcsec = calc_R200(mass, z_halo)
    rs = r200 / concentration

    # Compute the critical surface density at the halo redshift
    rho_c = cosmo.critical_density(z_halo).to(u.kg / u.m**3).value 
    rho_s = rho_c * calc_delta_c(concentration)
    sigma_c = critical_surface_density(z_halo, z_source)
    # Each halo has a characteristic surface density kappa_s
    kappa_s = rho_s * rs / sigma_c
    flexion_s = kappa_s * Dl / rs

    # Define the radial terms that go into lensing calculations - these are purely functions of x
    x = (x_range / (r200_arcsec / concentration))

    def radial_term_1(x):
        # This is called f(x) in theory
        # Note that the two statements are almost identical - switching arctanh and arctan, and 1-x and x-1
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1

        sol[mask1] = 1 - (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1])))
        sol[mask2] = 1 - (2 / np.sqrt(x[mask2]**2 - 1)) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2])))

        return sol
    
    def radial_term_2(x):
        # This is called g(x) in theory
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x > 1
        mask3 = x == 1

        sol[mask1] = 8 * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1]))) / (x[mask1]**2 * np.sqrt(1 - x[mask1]**2)) \
                    + 4 * np.log(x[mask1] / 2) / x[mask1]**2 \
                    - 2 / (x[mask1]**2 - 1) \
                    + 4 * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1]))) / ((x[mask1]**2 - 1) * np.sqrt(1 - x[mask1]**2))
        
        sol[mask2] = 8 * np.arctan(np.sqrt((x[mask2] - 1) / (x[mask2] + 1))) / (x[mask2]**2 * np.sqrt(x[mask2]**2 - 1)) \
                    + 4 * np.log(x[mask2] / 2) / x[mask2]**2 \
                    - 2 / (x[mask2]**2 - 1) \
                    + 4 * np.arctan(np.sqrt((x[mask2] - 1) / (x[mask2] + 1))) / ((x[mask2]**2 - 1)**(3/2))
        
        sol[mask3] = 10 / 3 + 4 * np.log(1/2)
        
        return sol

    def radial_term_3(x):
        # Compute the radial term - this is called h(x) in theory
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1

        sol[mask1] = (1 / (1 - x[mask1]**2)) * (1 / x[mask1] - ((2*x[mask1]) / np.sqrt(1 - x[mask1]**2)) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1]))))
        sol[mask2] = (1 / (x[mask2]**2 - 1)) * (((2 * x[mask2]) / np.sqrt(x[mask2]**2 - 1)) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2]))) - 1 / x[mask2])

        return sol
    
    def radial_term_4(x):
        # Compute the radial term - this is called I(x) in theory
        sol = np.zeros_like(x)
        mask1 = x < 1
        mask2 = x >= 1
        leading_term = 8 / x**3 - 20 / x + 15 * x

        sol[mask1] = leading_term[mask1] * (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(np.sqrt((1 - x[mask1]) / (1 + x[mask1])))
        sol[mask2] = leading_term[mask2] * (2 / np.sqrt(x[mask2]**2 - 1)) * np.arctan(np.sqrt((x[mask2] - 1) / (1 + x[mask2])))

        return sol

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)

    # Calculate the lensing signals
    kappa_mag = 2 * kappa_s * (term_1 / (x**2 - 1))
    shear_mag = -kappa_s * term_2
    flex_mag = (-2 * flexion_s) * ((2 * x * term_1 / (x**2 - 1)**2) - term_3 / (x**2 - 1)) / 206265 # Using Wright & Brainerd notation
    g_flex_mag = (2 * flexion_s) * ((8 / x**3) * np.log(x / 2) + ((3/x)*(1 - 2*(x**2)) + term_4) / (x**2 - 1)**2) / 206265

    '''
    # Compare shear from nfw to sis
    theta_e = 0.215
    shear_sis = - theta_e / (2 * x_range)

    plt.figure()
    plt.plot(x_range, np.abs(shear_mag), label='NFW')
    plt.plot(x_range, np.abs(shear_sis), label='SIS')
    plt.yscale('log')
    plt.xlabel('x')
    plt.xlabel('x')
    plt.ylabel('Shear')
    plt.title('Shear from NFW and SIS')
    plt.legend()
    plt.show()

    raise ValueError
    '''

    '''
    # Plot the lensing signals
    fig, ax = plt.subplots(1, 5, figsize=(20, 10))
    ax[0].plot(x_range, np.abs(kappa_mag), label=r'$\kappa$')
    ax[0].plot(x_range, np.abs(shear_mag), label=r'$\gamma$')
    ax[0].plot(x_range, np.abs(flex_mag), label=r'$\mathcal{F}$-Flexion')
    ax[0].plot(x_range, np.abs(g_flex_mag), label=r'$\mathcal{G}$-Flexion')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('Lensing signal')
    ax[0].set_title('Lensing signals')
    ax[0].legend()

    # The first flexion should be the same as the first derivative of the convergence
    first_deriv = np.gradient(kappa_mag, x_range)
    ax[1].plot(x_range[1:], np.abs(first_deriv[1:]), label='First derivative of $\kappa$')
    ax[1].plot(x_range[1:], np.abs(flex_mag[1:]), label='Flexion')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('Lensing signal')
    ax[1].set_title('First derivative of $\kappa$ and flexion')
    ax[1].legend()

    ax[2].plot(x_range, first_deriv - flex_mag)
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('Difference')    
    ax[2].set_title('Residual')

    first_deriv = np.gradient(shear_mag, x_range)
    # Leave out the first value when plotting, to account for the fact that the first derivative is not defined at x = 0
    ax[3].plot(x_range[1:], np.abs(first_deriv[1:]), label='First derivative of $\gamma$')
    ax[3].plot(x_range[1:], np.abs(g_flex_mag[1:]), label='G-Flexion')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('Lensing signal')
    ax[3].set_title('First derivative of $\gamma$ and G-Flexion')
    ax[3].legend()

    ax[4].plot(x_range[1:], first_deriv[1:] - g_flex_mag[1:])
    ax[4].set_xlabel('x')
    ax[4].set_ylabel('Difference')
    ax[4].set_title('Residual')


    plt.show()
    '''

    # Focus on the shear and the second flexion
    
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    ax = ax.flatten()

    ax[0].plot(x_range, np.abs(shear_mag), label=r'$\gamma$')
    ax[0].plot(x_range, np.abs(g_flex_mag), label=r'$\mathcal{G}$')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('Lensing signal')
    ax[0].set_title('Shear and G-Flexion')
    ax[0].legend()

    first_deriv = np.gradient(shear_mag, x_range)
    ax[1].plot(x_range[1:], np.abs(first_deriv[1:]), label='First derivative of $\gamma$')
    ax[1].plot(x_range[1:], np.abs(g_flex_mag[1:]), label='Flexion')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('Lensing signal')
    ax[1].set_title('First derivative of $\gamma$ and flexion')
    ax[1].legend()

    ax[2].plot(x_range[1:], first_deriv[1:] - g_flex_mag[1:])
    ax[2].axhline(0, color='black', linestyle='--')
    ax[2].set_xlabel('x')
    ax[2].set_ylabel('Residual')
    ax[2].set_title('Residual')    

    # Do this for the SIS
    theta_e = 0.215
    shear = -theta_e / (2 * x_range)
    g_flexion = 3*theta_e / (2 * x_range**2)

    ax[3].plot(x_range, np.abs(shear), label=r'$\gamma$')
    ax[3].plot(x_range, np.abs(g_flexion), label=r'$\mathcal{G}$')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('x')
    ax[3].set_ylabel('Lensing signal')
    ax[3].legend()

    first_deriv = np.gradient(shear, x_range)
    ax[4].plot(x_range[1:], np.abs(first_deriv[1:]), label='First derivative of $\gamma$')
    ax[4].plot(x_range[1:], np.abs(g_flexion[1:]), label='G-Flexion')
    ax[4].set_yscale('log')
    ax[4].set_xlabel('x')
    ax[4].set_ylabel('Lensing signal')
    ax[4].legend()

    ax[5].plot(x_range[1:], first_deriv[1:] / g_flexion[1:])
    ax[5].axhline(0, color='black', linestyle='--')
    ax[5].set_xlabel('x')
    ax[5].set_ylabel('Residual')
    ax[5].set_title('Residual')

    plt.savefig('nfw_test.png')

    plt.show()