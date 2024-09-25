"""Функциональная комбинаторика"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import itertools

NL_x = np.load('NonGlobalCDF_x.npy')
NL_y = np.load('NonGlobalCDF_y.npy')
x = np.arange(0, 2, 0.01, dtype=np.float64)
p = 0.45
multiplier = (1 + np.sqrt(1 - p**2)) / (2 * np.sqrt(1 - p**2))
f1 = np.full(shape=len(x), fill_value=0, dtype=np.float64)
f2 = np.exp(-2 * x**2)
f3 = np.full(shape=len(x), fill_value=0, dtype=np.float64)
f4 = -erf(x * np.sqrt(2) / p)
f5 = erf(x * np.sqrt(2) / p)
f6 = np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p)
f7 = np.exp(-2 * x**2)
f8 = -1 * np.exp(-2 * x**2)
f9 = np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p)
f10 = -1 * np.exp(-2 * x**2) * erf(x * np.sqrt(2 - 2 * p**2) / p)
functions = np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
reference1 = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10
reference2 = np.exp(-2 * x**2)
for i in range(2, 9):
    for combination in itertools.combinations(range(0, 10), i):
        F = 0
        for j in range(len(combination)):
            F += functions[combination[j]]
        if F[0] > F[-1]:
            print(combination)
            plt.plot(NL_x, NL_y, c='black', linestyle='dotted')
            plt.plot(x, reference1, c='black', linestyle='dashed')
            plt.plot(x, reference2, c='black')
            plt.plot(x, F, c='red')
            plt.show()
