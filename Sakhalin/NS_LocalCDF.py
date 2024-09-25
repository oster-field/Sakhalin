"""Функция распределения локальных экстремумов"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.special import erf

num_realizations = 200
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
            dq_1 = np.diff(q)
            dq_2 = np.diff(q, n=2)
            for i in range(1, len(dq_2)):
                if dq_1[i - 1] > 0 >= dq_1[i] and dq_2[i - 1] < 0 and q[i] > 0:
                    extremas = np.append(extremas, q[i] / 1.63)
            q = np.arange(0)
        q = np.append(q, j)
Fy = np.linspace(1, 0, len(extremas), endpoint=False)
Fx = np.sort(extremas)
x = np.arange(0, 10, 0.001)
m0 = np.trapz(Sy, dx=0.0001)
m1 = np.trapz(Sw * Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
nu = np.sqrt(m0 * m2 / (m1 ** 2) - 1)
ralaigh = np.exp(- 2 * (x**2))
raleigh_modified = 1 - (((np.exp(-2 * x**2)) / (1 + np.sqrt(1 - p**2))) * (np.sqrt(1 - p**2) * (np.sqrt(1 - p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p) + np.exp(2 * x**2) - erf((x * np.sqrt(2) * np.sqrt(1 - p**2)) / p) - 1) + (p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p)))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fx, Fy, color='red', alpha=.65, linewidth=5.5, label=f'ν={np.round(nu,2)}, ϵ={np.round(p,2)}')
ax.plot(x, ralaigh, color='black', alpha=1, linestyle='--', linewidth=2, label='Rayleigh CDF')
ax.plot(x, raleigh_modified, color='black', alpha=1, linewidth=2, linestyle='-.', label='CDF(ϵ)')
ax.set_xlabel('Normalized value of a positive local maximum', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.grid()
np.save('WP_3', p)
np.save('LM_Max_width_x', Fx)
np.save('LM_Max_width_y', Fy)
plt.show()
