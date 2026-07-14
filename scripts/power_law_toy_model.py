import numpy as np
import matplotlib.pyplot as plt

def power_law(theta, kappa_star, theta_star, n):
    return kappa_star * (theta / theta_star) ** -n

if __name__ == "__main__":
    theta = np.linspace(0.1, 10, 100)
    kappa_star = 1.0
    theta_star = 1.0
    n = 3.0

    kappa = power_law(theta, kappa_star, theta_star, n)

    
    plt.plot(theta, kappa)
    plt.xlabel("theta")
    plt.ylabel("kappa")
    plt.title("Power Law Toy Model")
    plt.yscale("log")
    # plt.show()
    

    # Perform numerical integration from 0 to theta to determine kappa_avg
    import scipy.integrate as integrate
    kappa_avg = np.zeros_like(theta)
    for i, theta_lim in enumerate(theta):
        kappa_avg[i], _ = integrate.quad(lambda t: power_law(t, kappa_star, theta_star, n), 0, theta_lim)
    kappa_avg *= 2/theta**2

    plt.figure()
    plt.plot(theta, kappa_avg)
    plt.xlabel("theta")
    plt.ylabel("average kappa")
    plt.title("Average Kappa for Theta <= Theta_lim")
    plt.yscale("symlog")
    plt.show()