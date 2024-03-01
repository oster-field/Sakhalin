import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D

x, y = np.meshgrid(np.arange(0.0001, 0.5, 0.01), np.arange(0, 1000, 1))
nu = np.sqrt(((2 / 3) * ((3 * x ** (5 / 6)) / (10 * y ** (2 / 3) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (5 / 6)) + (
        15 * np.sqrt(x) * np.sqrt(y) + 20 * x + 4 * y) / (
                                 30 * x ** (2 / 3) * y ** (1 / 6) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (
                                 5 / 6)))) / (((5 * x ** (2 / 3)) / (
        12 * y ** (1 / 3) * ((np.sqrt(x) + np.sqrt(y)) ** 2) ** (2 / 3)) + (
                                                       8 * np.sqrt(x) * y ** (1 / 6) + 3 * y ** (
                                                       2 / 3)) / (12 * x ** (1 / 3) * (
        (np.sqrt(x) + np.sqrt(y)) ** 2) ** (2 / 3))) ** 2) - 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, nu, cmap='inferno')
ax.set_xlabel('K', fontsize=12)
ax.set_ylabel('K_0', fontsize=12)
ax.set_zlabel('ν', fontsize=12)
plt.show()
