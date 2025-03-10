import utils
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Define signal parameters
    f = 1  # Frequency of sine wave
    fs = 100  # Sampling frequency
    t = np.arange(0, 1, 1/fs)  # Time vector
    x = 0.5 * np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 15 * f * t)  + 0.5 * np.sin(2 * np.pi * 60 * f * t) # Sine wave

    # Plot the original sine wave
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].plot(t, x)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Original Sine Wave")

    # Compute the Fourier Transform
    X = np.fft.fft(x)  # Keep complex values to retain phase
    freqs = np.fft.fftfreq(len(X), 1/fs)

    # Plot the magnitude spectrum (only positive frequencies)
    ax[1].plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]))
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_title("Fourier Transform (Positive Frequencies)")

    # Reconstruct the original signal using the full complex FFT
    x_reconstructed = np.fft.ifft(X).real  # Take only the real part

    # Plot the reconstructed signal
    ax[2].plot(t, x_reconstructed, linestyle="--")
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("Amplitude")
    ax[2].set_title("Reconstructed Signal")

    plt.tight_layout()
    plt.show()


def main_2d():
    # Define 2D sine wave parameters
    fx, fy = 2, 2  # Frequencies along x and y directions
    Nx, Ny = 100, 100  # Number of points in x and y directions
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    X, Y = np.meshgrid(x, y)  # Create a 2D grid

    # Generate a 2D sine wave
    z = np.sin(2 * np.pi * fx * X) + np.sin(2 * np.pi * fy * Y)
    # Smooth the signal
    z_smooth = utils.gaussian_filter(z, sigma=10, mode = 'wrap')

    # Compute the 2D Fourier Transform
    Z = np.fft.fft2(z_smooth)  # Preserve complex values
    Z_magnitude = np.abs(Z)  # Compute magnitude for visualization
    freq_x = np.fft.fftfreq(Nx, d=(x[1] - x[0]))
    freq_y = np.fft.fftfreq(Ny, d=(y[1] - y[0]))

    # Reconstruct the 2D sine wave using the inverse Fourier Transform
    z_reconstructed = np.fft.ifft2(Z).real  # Ensure real values

    # Plot original, Fourier Transform, and reconstructed signal
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(z, cmap="viridis", extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    ax[0].set_title("Original 2D Sine Wave")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    ax[1].imshow(z_smooth, cmap="viridis", extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    ax[1].set_title("Smoothed 2D Sine Wave")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    ax[2].imshow(z_reconstructed, cmap="viridis", extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
    ax[2].set_title("Reconstructed 2D Sine Wave")
    ax[2].set_xlabel("X")
    ax[2].set_ylabel("Y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_2d()