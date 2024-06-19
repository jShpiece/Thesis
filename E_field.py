# Plot electric field in an insulating sphere
import numpy as np
import matplotlib.pyplot as plt

# Constants
eps0 = 8.85e-12
k = 8.99e9
g = 9.8

def coulomb(q1,q2,r):
    return k*q1*q2/r**2

def E_field(q_0,r):
    return q_0/(4*np.pi*eps0*r**2)

def E_field_insulator(rho,r):
    return r*rho/(3*eps0)

qA = -6.5e-6
qB = 8.75e-6
E = 1.85e8
d = 2.5e-2

F_c = np.abs(coulomb(qA,qB,d))
F_e = np.abs(E*qB)

print("F_c = {:.2e} N".format(F_c))
print("F_E = {:.2e} N".format(F_e))
print("T = F_E - F_C = {:.2e} N".format(F_e - F_c))
