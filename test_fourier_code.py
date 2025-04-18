import numpy as np
import matplotlib.pyplot as plt
import utils
import source_obj

def shear_from_kappa(kappa):
    kappa_ft = np.fft.fft2(kappa)
    ny, nx = kappa.shape
    lx = np.fft.fftfreq(nx) * 2 * np.pi * nx / (2 * xmax)
    ly = np.fft.fftfreq(ny) * 2 * np.pi * ny / (2 * xmax)
    Lx, Ly = np.meshgrid(lx, ly)
    L2 = Lx**2 + Ly**2
    L2[0, 0] = 1  # avoid division by zero

    D = (Lx + 1j * Ly)**2 / L2
    gamma_ft = D * kappa_ft
    gamma = np.fft.ifft2(gamma_ft)
    return np.real(gamma), np.imag(gamma)

def flexion_from_kappa(kappa):
    # Flexion is just the gradient of the convergence
    kappa = np.array(kappa)
    flexion_1 = np.gradient(kappa, axis=1)
    flexion_2 = np.gradient(kappa, axis=0)
    return flexion_1, flexion_2


def sine_wave_test():
    xmax = 120  # size of the grid

    # create a grid
    x = np.linspace(-xmax, xmax, 2*xmax) 
    y = np.linspace(-xmax, xmax, 2*xmax) 
    X, Y = np.meshgrid(x, y)

    # Create a sine wave kappa
    wavelength = 5
    kappa = np.sin(np.sqrt(X**2 + Y**2) / wavelength)
    kappa = np.clip(kappa, -1, 1)  # clip to avoid extreme values

    # From the convergence, determine the shear and flexion
    flexion1, flexion2 = flexion_from_kappa(kappa)
    gamma1, gamma2 = shear_from_kappa(kappa)
    # Turn into source objects
    sources = source_obj.Source(X.flatten(), Y.flatten(), gamma1.flatten(), gamma2.flatten(), flexion1.flatten(), flexion2.flatten(), 
                                np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape))
    X1,Y1,kappa_f = utils.perform_kaiser_squire_reconstruction(sources, [-xmax, xmax, -xmax, xmax], 'flexion', smoothing_sigma = 0)
    X2,Y2,kappa_s = utils.perform_kaiser_squire_reconstruction(sources, [-xmax, xmax, -xmax, xmax], 'shear', smoothing_sigma = 0)

    # Determine common color limits across all relevant arrays
    vmin = min(np.min(kappa_f), np.min(kappa), np.min(kappa_f - kappa),
            np.min(kappa_s), np.min(kappa_s - kappa))
    vmax = max(np.max(kappa_f), np.max(kappa), np.max(kappa_f - kappa),
            np.max(kappa_s), np.max(kappa_s - kappa))

    # plot the results
    fig, ax = plt.subplots(2, 3, figsize=(20, 6))

    # Row 1
    im = ax[0,0].imshow(kappa_f, extent=[np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[0,0].set_title('Flexion Reconstruction')

    ax[0,1].imshow(kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[0,1].set_title('True Kappa')

    ax[0,2].imshow(kappa_f - kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray')
    ax[0,2].set_title('Difference')

    # Row 2
    ax[1,0].imshow(kappa_s, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[1,0].set_title('Shear Reconstruction')

    ax[1,1].imshow(kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[1,1].set_title('True Kappa')

    ax[1,2].imshow(kappa_s - kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray')
    ax[1,2].set_title('Difference')

    # get an rms average of the residual
    print('RMS of flexion residual:')
    print(np.sqrt(np.mean((kappa_f - kappa)**2)))
    print('RMS of shear residual:')
    print(np.sqrt(np.mean((kappa_s - kappa)**2)))

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save the figure
    plt.savefig('test_fourier_code.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    xmax = 120  # size of the grid

    # create a grid
    resolution = 1
    x = np.linspace(-xmax, xmax, 2*xmax) 
    y = np.linspace(-xmax, xmax, 2*xmax) 
    X, Y = np.meshgrid(x, y)

    # Create a softened isothermal sphere 
    # Place at the origin
    r_einstein = 10 
    r_core = 0.1
    r_trunc = xmax
    R = np.sqrt(X**2 + Y**2)
    x = R / r_einstein
    kappa = (x**2 + r_core**2) / (2 * x**2 + r_core**2) ** (3/2)
    gamma1 = - (x**2 / (2 * (x**2 + r_core**2) ** (3/2))) * np.cos(2 * np.arctan2(Y, X))
    gamma2 = - (x**2 / (2 * (x**2 + r_core**2) ** (3/2))) * np.sin(2 * np.arctan2(Y, X))

    # From the convergence, determine the flexion
    flexion1, flexion2 = flexion_from_kappa(kappa)

    # Turn into source objects
    sources = source_obj.Source(X.flatten(), Y.flatten(), gamma1.flatten(), gamma2.flatten(), flexion1.flatten(), flexion2.flatten(), 
                                np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape), np.zeros(X.flatten().shape))
    X1,Y1,kappa_f = utils.perform_kaiser_squire_reconstruction(sources, [-xmax, xmax, -xmax, xmax], 'flexion', smoothing_scale = 0, resolution_scale=1/resolution)
    X2,Y2,kappa_s = utils.perform_kaiser_squire_reconstruction(sources, [-xmax, xmax, -xmax, xmax], 'shear', smoothing_scale = 0, resolution_scale=1/resolution)

    # Determine common color limits across all relevant arrays
    vmin = min(np.min(kappa_f), np.min(kappa), np.min(kappa_f - kappa),
            np.min(kappa_s), np.min(kappa_s - kappa))
    vmax = max(np.max(kappa_f), np.max(kappa), np.max(kappa_f - kappa),
            np.max(kappa_s), np.max(kappa_s - kappa))

    # plot the results
    fig, ax = plt.subplots(2, 3, figsize=(20, 6))

    # Row 1
    im = ax[0,0].imshow(kappa_f, extent=[np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[0,0].set_title('Flexion Reconstruction')

    ax[0,1].imshow(kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[0,1].set_title('True Kappa')

    ax[0,2].imshow(kappa_f - kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray')
    ax[0,2].set_title('Difference')

    # Row 2
    ax[1,0].imshow(kappa_s, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[1,0].set_title('Shear Reconstruction')

    ax[1,1].imshow(kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray', vmin=vmin, vmax=vmax)
    ax[1,1].set_title('True Kappa')

    ax[1,2].imshow(kappa_s - kappa, extent=[-xmax, xmax, -xmax, xmax], origin='lower', cmap='gray')
    ax[1,2].set_title('Difference')

    # get an rms average of the residual
    print('RMS of flexion residual:')
    print(np.sqrt(np.mean((kappa_f - kappa)**2)))
    print('RMS of shear residual:')
    print(np.sqrt(np.mean((kappa_s - kappa)**2)))

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save the figure
    plt.savefig('softened_SIS.png', dpi=300, bbox_inches='tight')
    plt.show()