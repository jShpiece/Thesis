import numpy as np
import matplotlib.pyplot as plt 



x = np.linspace(0, 10, 100)
# Put source at origin, put lens at x=5
lens_x = 5
eR = 1
r = x - lens_x

true_flexion = eR / (2 * lens_x**2)
true_G_flexion = 3* eR / (2 * lens_x**2)

eR_F = true_flexion * r**2 * 2
eR_G = true_G_flexion * r**3 * 2 / 3

plt.plot(x, eR_F, label='Flexion')
plt.plot(x, eR_G, label='G-Flexion')
plt.show()