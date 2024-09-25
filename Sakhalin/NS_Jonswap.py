import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm

a = 8.1 * 10**(-3)
b = 0.003
g = 9.81
Sw = np.arange(0)
Sy = np.arange(0)
WDT = 1000
dt = 2 * np.pi / WDT
N = 200000
dw = WDT / N
L = dt * N
t = np.arange(0, L, dt, dtype=np.float64)
y, w = 0, 0
for i in tqdm(range(N), desc='Processing: ', colour='blue'):
    w = w + dw
    v = random.uniform(0, 2 * np.pi)
    #  S = a * g**2 * w**(-5) * np.exp(-b * g**4 * w**(-4))
    S = a * np.exp(-0.1 * (w - 7.5)**2)
    Sw = np.append(Sw, w)
    Sy = np.append(Sy, S)
    MonochromaticWave = (np.sqrt(2 * dw * S)) * (np.cos(w * t + v))
    y = y + MonochromaticWave
plt.plot(Sw, Sy)
plt.show()
m0 = np.trapz(Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
eps_theoretical = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
print(eps_theoretical)
plt.plot(t, y)
plt.show()
np.save('sequence_to_reduce_t_gauss', t)
np.save('sequence_to_reduce_y_gauss', y)
