"""Сравнение всех полученных распределений"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.special import erf

num_realizations = 1000
Wc = 1e5
W = 1e-5
w0_displacement = 0
Q = ((W * Wc) / (W + 2 * np.sqrt(W * Wc) + Wc)) ** (1 / 3)
w0 = np.sqrt(Q / Wc) + w0_displacement
WDT = 19 * (w0 + np.sqrt(Q / W))
N = 2 ** 13
dt = 2 * np.pi / WDT
dw = WDT / N
L = dt * N
amplitudes = np.arange(0, dtype=np.float64)
extremas = np.arange(0, dtype=np.float64)
t = np.arange(0, L, dt, dtype=np.float64)
c = 0
for counter in tqdm(range(0, num_realizations), colour='green', desc='Creating realizations '):
    y = 0
    w = 0
    Sy = np.arange(0)
    Sw = np.arange(0)
    for i in range(0, N):
        w = w + dw
        v = random.uniform(0, 2*np.pi)
        if w0 - np.sqrt(Q / Wc) <= w <= w0:
            S = -Wc * (w - w0)**2 + Q
        elif w0 < w <= w0 + np.sqrt(Q / W):
            S = -W * (w - w0)**2 + Q
        else:
            S = 0
        Sy = np.append(Sy, S)
        Sw = np.append(Sw, w)
        MonochromaticWave = (np.sqrt(2*dw*S))*(np.cos(w*t+v))
        y = y + MonochromaticWave
    y_1 = y
    t_1 = t
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
                amplitudes = np.append(amplitudes, np.max(q) / 1.63)
            q = np.arange(0)
        q = np.append(q, j)
Fy_A = np.linspace(1, 0, len(amplitudes), endpoint=False)
Fx_A = np.sort(amplitudes)
m0 = np.trapz(Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
F = (np.exp(-2 * Fx_A**2) / 2) * (1 + erf((Fx_A * np.sqrt(2 - 2 * p**2))/p))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fx_A, Fy_A, color='red', linewidth=3.5, label='$F_{A}$')
ax.plot(Fx_A, Fy_A - F, color='#5F2580', linewidth=3.5, label='$F_{A}$')
np.save('Find_CDF_F', Fy_A)
np.save('Find_CDF_x', Fx_A)
ax.plot(Fx_A, F, color='black', alpha=1, linestyle='--', linewidth=2, label='$F_{r}$')
ax.plot(Fx_A, np.exp(-2 * Fx_A**2), color='black', alpha=1, linestyle='-.', linewidth=2, label='$F_{r}$')
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=15, title=f'ϵ={p}', title_fontsize='15')
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.grid()
plt.show()
