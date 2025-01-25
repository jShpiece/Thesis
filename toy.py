# Toy file
# Create a fourier series that models a piecewise function

import numpy as np
import matplotlib.pyplot as plt


# Define the piecewise function
def f(x):
    if x < 0:
        return 0
    elif x < 1:
        return x
    elif x < 2:
        return 2 - x
    else:
        return 0
    
# Define the fourier series

def fourier_series(x, n):
    a0 = 1/2
    a = np.zeros(n)
    b = np.zeros(n)
    for i in range(1,n):
        a[i] = 2/(np.pi**2 * i**2) * (1 - (-1)**i)
        b[i] = 2/(np.pi * i) * (1 - (-1)**i)
    fs = a0
    for i in range(n):
        fs += a[i] * np.cos(i * np.pi * x) + b[i] * np.sin(i * np.pi * x)
    return fs

# Plot the function and the fourier series
x = np.linspace(-2, 2, 1000)
y = np.array([f(xi) for xi in x])
n = [1, 2, 3, 4, 5, 10, 20, 50, 100, 1000]
plt.plot(x, y, label='f(x)')
for ni in n:
    fs = np.array([fourier_series(xi, ni) for xi in x])
    plt.plot(x, fs, label='n = ' + str(ni))
plt.legend()
plt.show()
