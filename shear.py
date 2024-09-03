import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    dx = np.linspace(-9, 9, 10000)
    x = np.abs(dx)

    I_1 = radial_term_2(x)
    
    plt.figure()
    plt.plot(dx, I_1, label='g(x)')
    plt.yscale('log')
    plt.legend()
    plt.show()
