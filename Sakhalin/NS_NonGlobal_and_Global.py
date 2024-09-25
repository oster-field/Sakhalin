"""Сравнение всех полученных распределений"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
import sys
from scipy.special import erf
import os

num_realizations = 500
Wc = 1e5
W = 1e-5
w0_displacement = 0
Q = ((W * Wc) / (W + 2 * np.sqrt(W * Wc) + Wc)) ** (1 / 3)
w0 = np.sqrt(Q / Wc) + w0_displacement
part0 = np.arange(0, w0_displacement, 0.0001)
part1 = np.arange(w0_displacement, w0, 0.0001)
part2 = np.arange(w0, w0 + np.sqrt(Q / W), 0.0001)
Sw = np.append(np.append(part0, part1), part2)
Sy = np.append(np.append(0 * part0, -Wc * (part1 - w0)**2 + Q), -W * (part2 - w0)**2 + Q)
m0 = np.trapz(Sy, dx=0.0001)
As = 2 * np.sqrt(m0)
m1 = np.trapz(Sw * Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
nu = np.sqrt(m0 * m2 / (m1 ** 2) - 1)
rho = (2 * np.sqrt(1 - p**2)) / (1 + np.sqrt(1 - p**2))
WDT = 30 * (w0 + np.sqrt(Q / W))
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
    for i in range(0, N):
        w = w + dw
        v = random.uniform(0, 2*np.pi)
        if w0 - np.sqrt(Q / Wc) <= w <= w0:
            S = -Wc * (w - w0)**2 + Q
        elif w0 < w <= w0 + np.sqrt(Q / W):
            S = -W * (w - w0)**2 + Q
        else:
            S = 0
        MonochromaticWave = (np.sqrt(2*dw*S))*(np.cos(w*t+v))
        y = y + MonochromaticWave
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
            if np.sum(q) > 0:
                q = np.abs(q)
                q = np.append(q, 0)
                amplitudes = np.append(amplitudes, np.max(q) / As)
                for i in range(1, len(q) - 1):
                    if q[i] > q[i - 1] and q[i] > q[i + 1] and q[i] != np.max(q):
                        extremas = np.append(extremas, q[i] / As)
            q = np.arange(0)
        q = np.append(q, j)
    if len(extremas) > 10000:
        np.save(f'Extr{c}', extremas)
        extremas = np.arange(0, dtype=np.float64)
        c += 1
        print('Scatters saved')
for i in range(0, sys.maxsize):
    try:
        extremas = np.append(extremas, np.load(f'Extr{i}.npy'))
        os.remove(f'Extr{i}.npy')
    except FileNotFoundError:
        break
c = 0
Fy = np.linspace(1, 0, len(extremas), endpoint=False)
Fy_A = np.linspace(1, 0, len(amplitudes), endpoint=False)
Fx = np.sort(extremas)
Fx_A = np.sort(amplitudes)
x = np.arange(0, 10, 0.001)
ralaigh = np.exp(- 2 * (x**2))
raleigh_modified = 1 - (((np.exp(-2 * x**2)) / (1 + np.sqrt(1 - p**2))) * (np.sqrt(1 - p**2) * (np.sqrt(1 - p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p) + np.exp(2 * x**2) - erf((x * np.sqrt(2) * np.sqrt(1 - p**2)) / p) - 1) + (p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p)))
psi = np.sqrt(1 - p ** 10)
CDF_A = (1 - erf(x * np.sqrt(2) * psi / p) + np.exp(-2 * x ** 2) * (1 + erf(x * np.sqrt(2 - 2 * p ** 2) / p))) / 2
CDF_L = (1 - erf(x * np.sqrt(2) / p) - np.sqrt(1 - p**2) * (1 - erf(x * np.sqrt(2) * psi / p))) / (1 - np.sqrt(1 - p**2))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fx, Fy, color='blue', alpha=.85, linewidth=1.5, marker='.', label='$F_{L}$ numerical curve')
ax.plot(Fx_A, Fy_A, color='red', alpha=.85, linewidth=1.5, marker='.', label='$F_{A}$ numerical curve')
ax.plot(x, ralaigh, color='black', alpha=1, linestyle='--', linewidth=2, label='$F_{r}$')
ax.plot(x, raleigh_modified, color='black', alpha=1, linewidth=2, linestyle='-.', label='$F_{M}$')
ax.plot(x, CDF_A, color='black', alpha=1, linewidth=2, label='$F_{A}$')
ax.plot(x, CDF_L, color='black', alpha=1, linewidth=2, linestyle='dotted', label='$F_{L}$')
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=15, title=f'ν={np.round(nu,2)}, ϵ={np.round(p,2)}, ρ={np.round(rho,2)}', title_fontsize='15')
"""ax.set(ylim=[10e-7, 1])
ax.set(xlim=[0, 3])
plt.yscale('log')"""
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.grid()
plt.show()
