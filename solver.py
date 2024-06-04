import numpy as np
from scipy.special import hyp2f1
from scipy.optimize import fsolve

# Constants
c = 3.0e8  # speed of light in m/s
G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
H0 = 70.0  # Hubble constant in km/s/Mpc
H0_SI = H0 * 1.0e3 / 3.086e22  # Hubble constant in s^-1

# Example observed values (replace with actual observed data)
gamma_obs = 0.05
flexion_obs = 0.01
z_lens = 0.3
z_source = 1.0

# Angular diameter distance (D_l, D_s, D_ls) - example functions, replace with actual calculations
def angular_diameter_distance(z1, z2):
    # Placeholder function, replace with actual distance calculation
    return 1e22  # Example value in meters

D_l = angular_diameter_distance(0, z_lens)
D_s = angular_diameter_distance(0, z_source)
D_ls = angular_diameter_distance(z_lens, z_source)

# Critical surface density
Sigma_cr = c**2 / (4 * np.pi * G) * D_s / (D_l * D_ls)

# NFW profile functions
def kappa_nfw(x, kappa_s):
    if x < 1:
        f_x = np.arctanh(np.sqrt(1 - x**2)) / np.sqrt(1 - x**2)
    elif x == 1:
        f_x = 1
    else:
        f_x = np.arctan(np.sqrt(x**2 - 1)) / np.sqrt(x**2 - 1)
    
    return 2 * kappa_s / (x**2 - 1) * (1 - f_x)

def gamma_nfw(x, kappa_s):
    return kappa_s * (1 - 2 * hyp2f1(1, 1, 2, -x**2) / x**2)

def flexion_nfw(x, kappa_s):
    dkappa_dx = (2 * kappa_s * (1 - (2 * x) / (x**2 - 1)**2 * (1 - np.arctanh(np.sqrt(1 - x**2)) / np.sqrt(1 - x**2)))) / x
    return dkappa_dx

# Define the system of equations
def equations(vars):
    x, kappa_s = vars
    gamma = gamma_nfw(x, kappa_s)
    flexion = flexion_nfw(x, kappa_s)
    eq1 = gamma - gamma_obs
    eq2 = flexion - flexion_obs
    return [eq1, eq2]

# Initial guesses for x and kappa_s
initial_guess = [1.0, 0.005]

# Solve the system of equations
solution = fsolve(equations, initial_guess)
x_solution, kappa_s_solution = solution

# Calculate the mass from kappa_s
r_s = 1.0  # Example scale radius in meters, replace with actual value
M_200 = 4 * np.pi * r_s**2 * Sigma_cr * kappa_s_solution

# Print the results
print(f"Solved x: {x_solution}")
print(f"Solved kappa_s: {kappa_s_solution}")
print(f"Estimated mass M_200: {M_200 / 1.989e30} solar masses")
