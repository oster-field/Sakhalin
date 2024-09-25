"""Попытка найти зависимость параметра ширины от характеристики поля"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm

num_realizations = 15
a = 0.01
b = 0.5
F = np.arange(0)
x = np.arange(0)
local_extr = 0
global_extr = 0
for n in tqdm(range(0, 100), colour='green', desc='Creating realizations '):
    a += 0.1
    b += 0.2
    c = 1 / (b - a)
    WDT = 19 * b
    N = 2 ** 13
    dt = 2 * np.pi / WDT
    dw = WDT / N
    L = dt * N
    t = np.arange(0, L, dt, dtype=np.float64)
    extremas = np.arange(0)
    for counter in range(0, num_realizations):
        y = 0
        w = 0
        Sy = np.arange(0)
        Sw = np.arange(0)
        for i in range(0, N):
            w = w + dw
            v = random.uniform(0, 2*np.pi)
            if a <= w <= b:
                S = c
            else:
                S = 0
            Sw = np.append(Sw, w)
            Sy = np.append(Sy, S)
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
                q = np.abs(q)
                global_extr += 1
                local_extr += 1
                dq_1 = np.diff(q)
                dq_2 = np.diff(q, n=2)
                for i in range(1, len(dq_2)):
                    if dq_1[i - 1] > 0 >= dq_1[i] and dq_2[i - 1] < 0 and q[i] != np.max(q):
                        local_extr += 1
                q = np.arange(0)
            q = np.append(q, j)
    m0 = np.trapz(Sy, dx=0.0001)
    m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
    m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
    x = np.append(x, np.sqrt(1 - (m2 ** 2) / (m0 * m4)))
    F = np.append(F, global_extr/local_extr)
    local_extr = 0
    global_extr = 0
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, F, color='#007439', alpha=0.7, linewidth=3, marker='o')
ax.set_xlabel('ϵ', fontsize=20)
ax.set_ylabel('Nₐ/Nₘ', fontsize=20)
ax.tick_params(labelsize=20)
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 1])
p = np.arange(0, 1.01, 0.01)
ax.plot(p, np.sqrt(1-p**2), color='black', linestyle='dashdot', label='√(1-ϵ²)')
ax.plot(p, 1-p, color='black', label='1-ϵ', linewidth='3')
ax.plot(p, np.sqrt(1-p), color='black', linestyle='dashed', label='√(1-ϵ)')
ax.plot(p, 1-p**2, color='black', linestyle='dotted', label='1-ϵ²')
ax.plot(p, 2 * np.sqrt(1-p**2) / (1+np.sqrt(1-p**2)), color='black', label='(2√(1-ϵ²))/(1+√(1-ϵ²))')
ax.grid()
ax.legend(fontsize=20)
plt.show()
