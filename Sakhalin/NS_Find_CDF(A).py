import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

x = np.arange(0, 1.25, 0.01, dtype=np.float64)
p = 0.6
multiplier = (1 + np.sqrt(1 - p**2)) / (2 * np.sqrt(1 - p ** 2))
f1 = np.full(shape=len(x), fill_value=-1 * np.sqrt(1 - p**2) / p**2, dtype=np.float64)
f2 = (np.sqrt(1 - p**2) / p**2) * np.exp(-2 * x**2)
f3 = np.full(shape=len(x), fill_value=1 / p**2, dtype=np.float64)
f4 = -1 * (1 / p**2) * erf(x * np.sqrt(2) / p)
f5 = (np.sqrt(1 - p**2) / p**2) * erf(x * np.sqrt(2) / p)
f6 = (np.sqrt(1 - p**2) / p**2) * np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p)
f7 = np.exp(-2 * x**2)
f8 = -1 * np.exp(-2 * x**2) * (1 / p**2)
f9 = np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p)
f10 = -1 * np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p) * (1 / p**2)

reference1 = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10
reference2 = np.exp(-2 * x**2)
plt.plot(x, reference1, c='black', linestyle='dashed')
plt.plot(x, reference2, c='black')
plt.show()