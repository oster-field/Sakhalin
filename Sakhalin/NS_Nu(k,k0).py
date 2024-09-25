import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

x, y = np.meshgrid(np.arange(0, 2, 0.01), np.arange(0.001, 1, 0.01))
nu = np.sqrt(((2 / 3) * ((3 * x ** (5 / 6)) / (10 * y ** (2 / 3) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (5 / 6)) + (
        15 * np.sqrt(x) * np.sqrt(y) + 20 * x + 4 * y) / (
                                 30 * x ** (2 / 3) * y ** (1 / 6) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (
                                 5 / 6)))) / (((5 * x ** (2 / 3)) / (
        12 * y ** (1 / 3) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (2 / 3)) + (
                                                       8 * np.sqrt(x) * y ** (1 / 6) + 3 * y ** (
                                                       2 / 3)) / (12 * x ** (1 / 3) * (
        (np.sqrt(x) + np.sqrt(y)) ** 2) ** (2 / 3))) ** 2) - 1)
difference = np.exp(- 2 * (x**2)) - 1 + (((np.exp(-2 * x**2)) / (1 + np.sqrt(1 - y**2))) * (np.sqrt(1 - y**2) * (np.sqrt(1 - y**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / y) + np.exp(2 * x**2) - erf((x * np.sqrt(2) * np.sqrt(1 - y**2)) / y) - 1) + (y**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / y)))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, difference, cmap='inferno')
ax.set_xlabel('a / $a_{s}$', fontsize=12)
ax.set_ylabel('ϵ', fontsize=12)
ax.set_zlabel('$F_{r}$ - $F_{M}$', fontsize=12)
plt.show()
