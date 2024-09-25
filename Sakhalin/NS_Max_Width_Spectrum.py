"""Сравнение всех полученных распределений"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.special import erf
from scipy.optimize import curve_fit

'''num_realizations = 300
w0_displacement = 2.1
WDT = 200
N = 2 ** 13
dt = 2 * np.pi / WDT
dw = WDT / N
L = dt * N
amplitudes = np.arange(0, dtype=np.float64)
extremas = np.arange(0, dtype=np.float64)
t = np.arange(0, L, dt, dtype=np.float64)
c = 0
for counter in tqdm(range(0, num_realizations), desc='Realizations: ', colour='yellow'):
    y = 0
    w = 0
    Sy = np.arange(0)
    Sw = np.arange(0)
    for i in range(0, N):
        w = w + dw
        v = random.uniform(0, 2*np.pi)
        S = 1 / ((w - w0_displacement) ** 4 + 1)
        Sy = np.append(Sy, S)
        Sw = np.append(Sw, w)
        MonochromaticWave = (np.sqrt(2*dw*S))*(np.cos(w*t+v))
        y = y + MonochromaticWave
    y_1 = y
    t_1 = t
    As = 2 * np.sqrt(np.trapz(Sy, dx=dw))
    tc, ti = pyaC.zerocross1d(t, y, getIndices=True)
    tnew = np.sort(np.append(t, tc))
    for i in range(1, len(tnew + 1)):
        if tnew[i] in tc:
            tzm1 = np.where(tnew == tnew[i - 1])[0]
            yzm1 = np.where(y == y[tzm1])[0]
            y = np.insert(y, yzm1 + 1, [0])
    q = np.arange(0)
    for j in y:
        if j == 0:
            q = np.append(q, 0)
            if np.sum(q) < 0:
                q = np.abs(q)
                amplitudes = np.append(amplitudes, np.max(q) / As)
            q = np.arange(0)
        q = np.append(q, j)
Fy_A = np.linspace(1, 0, len(amplitudes), endpoint=False)
Fx_A = np.sort(amplitudes)
m0 = np.trapz(Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
print(p)'''
Fx_A = np.load('MWX.npy')
Fy_A = np.load('MWY.npy')
p = 0.896852388991725
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.plot(Fx_A, Fy_A, linewidth=2.5, color='red')
ax.plot(Fx_A, (np.exp(-2 * Fx_A**2) / 2) * (1 + erf((Fx_A * np.sqrt(2 - 2 * p**2))/p)), linewidth=2.5, color='green')
ax.plot(Fx_A, Fy_A - (np.exp(-2 * Fx_A**2) / 2) * (1 + erf((Fx_A * np.sqrt(2 - 2 * p**2))/p)), linewidth=2, color='blue')
a = 1.35
ax.plot(Fx_A, 1/2 * (1 - erf(a * Fx_A)), linewidth=2.5, color='#A64B00')
ax.plot(Fx_A, np.exp(-2 * Fx_A**2), color='black')
ax.tick_params(labelsize=20)
np.save('MWX.npy', Fx_A)
np.save('MWY.npy', Fy_A)
ax.grid()
plt.show()
