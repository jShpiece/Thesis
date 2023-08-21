import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

def normalize_distribution(y):
    volume = np.sum(y)
    normalized_y = y / volume
    return normalized_y

def combine_distributions(y1, y2):
    normalized_y1 = normalize_distribution(y1)
    normalized_y2 = normalize_distribution(y2)
    
    combined_distribution = convolve(normalized_y1, normalized_y2, mode='same', method='auto')
    combined_distribution /= np.sum(combined_distribution)
    
    return combined_distribution

# Generate 3D grids for x, y, and z
x, y, z = np.meshgrid(np.linspace(-5, 5, 39), np.linspace(-5, 5, 39), np.linspace(-5, 5, 39))

# Generate 3D distributions y1 and y2 (example functions)
y1 = np.exp(-((x - 4)**2 + y**2 + z**2))
y2 = np.exp(-(x**2 + y**2 + z**2) / 2)
y3 = np.exp(-((x + 2)**2 + y**2 + z**2))

combined_distribution = combine_distributions(y1, y2)
combined_distribution = combine_distributions(combined_distribution, y3)

# Visualize the combined distribution using contourf
plt.figure()
plt.contourf(x[:,:,0], y[:,:,0], combined_distribution[:,:,0], cmap='viridis')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Combined Probability Distribution')
plt.colorbar(label='Probability')
plt.show()
