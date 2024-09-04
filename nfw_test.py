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

def compute_flexion(gamma1, gamma2, dx):
    """
    Computes the second flexion (G) from the shear components (gamma1, gamma2).
    
    Parameters:
    - gamma1: 2D array representing the first component of shear (gamma1)
    - gamma2: 2D array representing the second component of shear (gamma2)
    - dx: Scalar representing the grid spacing
    
    Returns:
    - G1: 2D array representing the first component of second flexion (G1)
    - G2: 2D array representing the second component of second flexion (G2)
    """
    # Compute the numerical derivatives
    dgamma1_dx = np.gradient(gamma1, dx, axis=1)
    dgamma1_dy = np.gradient(gamma1, dx, axis=0)
    dgamma2_dx = np.gradient(gamma2, dx, axis=1)
    dgamma2_dy = np.gradient(gamma2, dx, axis=0)
    
    # Compute components of the flexion G = (G1, G2)
    G1 = dgamma1_dx - dgamma2_dy
    G2 = dgamma2_dx + dgamma1_dy
    
    return G1, G2

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

    k = (1 - x) / (1 + x)

    sol[mask1] = 8 * np.arctanh(np.sqrt(k[mask1])) / (x[mask1]**2 * np.sqrt(1 - x[mask1]**2)) \
                + 4 * np.log(x[mask1]/2) / x[mask1]**2 \
                - 2 / (x[mask1]**2 - 1) \
                + 4 * np.arctanh(np.sqrt(k[mask1])) / ((x[mask1]**2 - 1) * np.sqrt(1 - x[mask1]**2))
    
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

    # Introduce a constant
    k = (1 - x) / (1 + x)

    sol[mask1] = (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(np.sqrt(k[mask1]))
    sol[mask2] = (2 / np.sqrt(x[mask2]**2 - 1)) * np.arctan(np.sqrt(-k[mask2]))
    sol *= leading_term

    return sol


def one_dimensional_calc():
    # Do this in 1D
    z_halo = 0.35
    z_source = 0.8
    mass = 1e12
    concentration = 7.2
    dx = np.linspace(-10, 10, 1000)

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
    x = np.abs(dx / (r200_arcsec / concentration))

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)

    def calc_f_flex(flexion_s, x, term_1, term_3):
        I_1 = 2 * flexion_s 
        I_2 = 2 * x * term_1 / (x**2 - 1)**2
        I_3 = term_3 / (x**2 - 1)
        return I_1 * (I_2 - I_3) / 206265

    def calc_g_flex(flexion_s, x, term_4):
        I_1 = 2 * flexion_s
        I_2 = (8 / x**3) * np.log(x / 2)
        I_3 = ((3 / x) * (1 - 2 * x**2) + term_4)
        I_4 = (x**2 - 1)**2
        return (I_1 * (I_2 + (I_3 / I_4))) / 206265
    
    # Compute the magnitudes of the lensing quantities
    kappa = 2*kappa_s * term_1 / (x**2 - 1)
    gamma = kappa_s * term_2 
    flex_mag = calc_f_flex(flexion_s, x, term_1, term_3)
    g_flex_mag = calc_g_flex(flexion_s, x, term_4)

    # Plot the lensing quantities
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    ax[0].plot(dx, kappa, label='Convergence')
    ax[0].plot(dx, np.abs(gamma), label='Shear')
    ax[0].plot(dx, flex_mag, label='Flexion')
    ax[0].plot(dx, g_flex_mag, label='G Flexion')
    ax[0].set_title('Lensing quantities')
    ax[0].legend()
    ax[0].set_xlabel('dx (arcsec)')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_yscale('log')

    # Compare the gradient of the convergence to the first flexion
    grad_kappa = np.gradient(kappa, dx)
    # Get the magnitude
    grad_kappa = np.abs(grad_kappa)
    ax[1].plot(dx, grad_kappa, label='Gradient of convergence', linestyle='--')
    ax[1].plot(dx, flex_mag, label='Flexion', linestyle='-.')
    ax[1].set_title('Comparison of gradients: Convergence and F Flexion')
    ax[1].legend()
    ax[1].set_xlabel('dx (arcsec)')
    ax[1].set_ylabel('Magnitude')
    ax[1].set_yscale('log')

    # Compare the gradient of the shear to the first flexion
    grad_gamma = np.gradient(gamma, dx) + 2 * gamma / dx
    # Get the magnitude
    grad_gamma = np.abs(grad_gamma)
    ax[2].plot(dx, grad_gamma, label='Gradient of shear', linestyle='--')
    ax[2].plot(dx, flex_mag, label='F Flexion', linestyle='-.')
    # ax[2].plot(dx, grad_gamma / g_flex_mag, label='Ratio', linestyle=':')
    ax[2].set_title('Comparison of gradients - Shear and F Flexion')
    ax[2].legend()
    ax[2].set_xlabel('dx (arcsec)')
    ax[2].set_ylabel('Magnitude')
    ax[2].set_yscale('log')

    # Compare the gradient of the shear to the second flexion
    grad_gamma = np.gradient(gamma, dx) - 2 * gamma / dx
    # Get the magnitude
    grad_gamma = np.abs(grad_gamma)
    ax[3].plot(dx, grad_gamma, label='Gradient of shear', linestyle='--')
    ax[3].plot(dx, g_flex_mag, label='G Flexion', linestyle='-.')
    # ax[3].plot(dx, grad_gamma / g_flex_mag, label='Ratio', linestyle=':')
    ax[3].set_title('Comparison of gradients - Shear and G Flexion')
    ax[3].legend()
    ax[3].set_xlabel('dx (arcsec)')
    ax[3].set_ylabel('Magnitude')
    ax[3].set_yscale('log')

    plt.savefig('1d_lensing_quantities.png')

    plt.show()
    plt.close()
    raise ValueError('Stop here')
    # Take the second derivative of the first and second flexion - they should match
    f_first = np.gradient(flex_mag, dx)
    f_second = np.gradient(f_first, dx)
    g_first = np.gradient(g_flex_mag, dx)
    g_second = np.gradient(g_first, dx)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(dx, f_second, label='Second derivative of first flexion')
    ax.plot(dx, g_second, label='Second derivative of second flexion')
    # Add a vertical line to show where our scale radius is - this is where the piecewise function changes
    ax.axvline(x=r200_arcsec/concentration, color='r', linestyle='--')
    ax.set_title('Second derivatives of flexion')
    ax.legend()
    ax.set_xlabel('dx (arcsec)')
    ax.set_ylabel('Magnitude')
    ax.set_yscale('log')

    plt.savefig('1d_flexion_second_derivative.png')
    plt.show()
    


def two_dimensional_calc():
    # Do this in 2D
    z_halo = 0.35
    z_source = 0.8
    mass = 1e12
    concentration = 7.2
    dx = np.linspace(-10, 10, 1000)
    dy = np.linspace(-10, 10, 1000)
    dx, dy = np.meshgrid(dx, dy)
    r = np.sqrt(dx**2 + dy**2)
    # Mask the center
    r[r < 1e-3] = 1e-3

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
    x = np.abs(r / (r200_arcsec / concentration))

    term_1 = radial_term_1(x)
    term_2 = radial_term_2(x)
    term_3 = radial_term_3(x)
    term_4 = radial_term_4(x)

    def calc_f_flex(flexion_s, x, term_1, term_3):
        I_1 = 2 * flexion_s 
        I_2 = 2 * x * term_1 / (x**2 - 1)**2
        I_3 = term_3 / (x**2 - 1)
        return I_1 * (I_2 - I_3) / 206265

    def calc_g_flex(flexion_s, x, term_4):
        I_1 = 2 * flexion_s
        I_2 = (8 / x**3 ) * np.log(x / 2)
        I_3 = ((3 / x) * (1 - 2 * x**2)) + term_4
        I_4 = (x**2 - 1)**2
        return (I_1 * (I_2 + (I_3 / I_4))) / 206265
    
    # Compute the magnitudes of the lensing quantities
    kappa = 2*kappa_s * term_1 / (x**2 - 1)
    gamma = kappa_s * term_2 
    flex_mag = calc_f_flex(flexion_s, x, term_1, term_3)
    g_flex_mag = calc_g_flex(flexion_s, x, term_4)

    # Create the angles between the source and the halo
    cos_phi = dx / r # Cosine of the angle between the source and the halo
    sin_phi = dy / r # Sine of the angle between the source and the halo
    cos2phi = cos_phi**2 - sin_phi**2 # Cosine of 2*phi
    sin2phi = 2 * cos_phi * sin_phi # Sine of 2*phi
    cos3phi = cos2phi * cos_phi - sin2phi * sin_phi # Cosine of 3*phi
    sin3phi = sin2phi * cos_phi + cos2phi * sin_phi # Sine of 3*phi

    # Compute the lensing quantities
    gamma_1 = gamma * cos2phi
    gamma_2 = gamma * sin2phi
    flex_1 = flex_mag * cos_phi
    flex_2 = flex_mag * sin_phi
    g_flex_1 = g_flex_mag * cos3phi
    g_flex_2 = g_flex_mag * sin3phi

    # Plot these signals
    extent = [-10, 10, -10, 10]
    signals = [kappa, gamma_1, gamma_2, flex_1, flex_2, g_flex_1, g_flex_2]
    titles = ['Convergence', 'Shear 1', 'Shear 2', 'Flexion 1', 'Flexion 2', 'G Flexion 1', 'G Flexion 2']
    fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    ax = ax.flatten()
    for i, signal in enumerate(signals):
        ax[i].imshow(signal, origin='lower', cmap='gray', extent=extent)
        ax[i].set_title(titles[i])

    # Remove the unused axes (0,3)
    ax[-1].axis('off')
    plt.savefig('2d_lensing_quantities.png')
    # plt.show()
    plt.close()

    # Now perform some tests
    # The second derivatives of the two flexions should match
    # The gradient of the convergence should match the first flexion
    # The gradient of the shear should match the second flexion

    # Compute the first derivatives of the flexions
    # (redefine dx and dy, because we made them into a meshgrid)
    '''
    dx = np.linspace(-10, 10, 1000)
    F1_1 = np.gradient(flex_1, dx, axis=0)
    F1_2 = np.gradient(flex_1, dx, axis=1)
    F2_1 = np.gradient(flex_2, dx, axis=0)
    F2_2 = np.gradient(flex_2, dx, axis=1)

    G1_1 = np.gradient(g_flex_1, dx, axis=0)
    G1_2 = np.gradient(g_flex_1, dx, axis=1)
    G2_1 = np.gradient(g_flex_2, dx, axis=0)
    G2_2 = np.gradient(g_flex_2, dx, axis=1)

    # Compute the second derivatives of the flexions
    F1_11 = np.gradient(F1_1, dx, axis=0)
    F1_22 = np.gradient(F1_2, dx, axis=1)
    F2_11 = np.gradient(F2_1, dx, axis=0)
    F2_22 = np.gradient(F2_2, dx, axis=1)
    F1_12 = np.gradient(F1_1, dx, axis=1)
    F2_12 = np.gradient(F2_1, dx, axis=1)

    G1_11 = np.gradient(G1_1, dx, axis=0)
    G1_22 = np.gradient(G1_2, dx, axis=1)
    G2_11 = np.gradient(G2_1, dx, axis=0)
    G2_22 = np.gradient(G2_2, dx, axis=1)
    G1_12 = np.gradient(G1_1, dx, axis=1)
    G2_12 = np.gradient(G2_1, dx, axis=1)

    # It should be the case that G1_11 + G1_22 = F1_11 - F1_22 + 2F2_12
    # And G2_11 + G2_22 = F2_11 - F2_22 + 2F1_12
    # We will test this
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Second derivative test - all values should be zero')
    ax[0].imshow(G1_11 + G1_22 - F1_11 + F1_22 - 2*F2_12, origin='lower', cmap='gray', extent=extent)
    ax[0].set_title('G1_11 + G1_22 - F1_11 + F1_22 - 2F2_12')
    ax[1].imshow(G2_11 + G2_22 - F2_11 + F2_22 - 2*F1_12, origin='lower', cmap='gray', extent=extent)
    ax[1].set_title('G2_11 - G2_22 - F2_11 + F2_22 - 2F1_12')
    plt.savefig('2d_flexion_second_derivative_test.png')
    plt.show()
    '''

    # Now we will test the gradients
    # The gradient of the convergence should match the first flexion
    # The gradient of the shear should match the second flexion
    dx = np.linspace(-10, 10, 1000)
    grad_kappa = np.gradient(kappa, dx, axis=0)
    # Kappa is symmetric, so we can just take the gradient in one direction
    # grad_kappa = np.abs(grad_kappa)
    G1, G2 = compute_flexion(gamma_1, gamma_2, 1)

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    fig.suptitle('Gradient test - all values should be 1')
    ax[0].imshow(grad_kappa - flex_mag, origin='lower', cmap='gray', extent=extent)
    ax[0].set_title('Gradient of convergence - Flexion')
    ax[1].imshow(G1 - g_flex_1, origin='lower', cmap='gray', extent=extent)
    ax[1].set_title('Gradient of shear 1 - G1')
    ax[2].imshow(G2 - g_flex_2, origin='lower', cmap='gray', extent=extent)
    ax[2].set_title('Gradient of shear 2 - G2')
    G_tot = np.sqrt(G1**2 + G2**2)
    ax[3].imshow(G_tot - g_flex_mag, origin='lower', cmap='gray', extent=extent)
    ax[3].set_title('Total gradient - G Flexion')
    plt.savefig('2d_flexion_gradient_test.png')
    plt.show()


if __name__ == '__main__':
    one_dimensional_calc()
    # two_dimensional_calc()