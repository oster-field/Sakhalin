import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from scipy.special import erf
from tqdm import tqdm


def hsas(arr, cutnumber):
    arr1 = np.sort(arr)
    for i in range(len(arr1) - 1):
        if arr1[i] <= cutnumber < arr1[i + 1]:
            arr1 = arr1[i + 1: -1]
            break
    return np.mean(arr1[int(len(arr1) / 3):-1])


def hights(ymax, ymin):
    wavelenght = np.arange(0)
    if len(ymax) >= len(ymin):
        for i in range(0, len(ymin)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    if len(ymax) < len(ymin):
        for i in range(0, len(ymax)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    return wavelenght


num_realizations = 3000
w0_displacement = 0
Wc = 1e5
W = 1e-5
Q = ((W * Wc) / (W + 2 * np.sqrt(W * Wc) + Wc)) ** (1 / 3)
w0 = np.sqrt(Q / Wc) + w0_displacement
WDT = 19 * (w0 + np.sqrt(Q / W))
N = 2**13
dt = 2*np.pi / WDT
dw = WDT / N
L = dt * N
t = np.arange(0, L, dt, dtype=np.float64)
all_amplitudes = np.arange(0)
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
    if counter == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=20)
        ax.set_xlabel('ω, [rad/sec.]', fontsize=20)
        ax.set_ylabel('S(ω)', fontsize=20)
        ax.plot(Sw, Sy, color='#48036F', linewidth=5)
        ax.axvline(x=w0, color='black', linewidth=2, linestyle='dashdot')
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=20)
        ax.set_xlabel('t, [sec.]', fontsize=20)
        ax.set_ylabel('η(x=0, t), [m]', fontsize=20)
        ax.plot(t, y, color='#0B61A4', alpha=.8)
        ax.axhline(y=0, color='black', linewidth=1)
        plt.show()
    q = np.arange(0)
    ymax = np.arange(0)
    x = t
    xc, xi = pyaC.zerocross1d(x, y, getIndices=True)
    xnew = np.sort(np.append(x, xc))
    for i in range(1, len(xnew + 1)):
        if xnew[i] in xc:
            xzm1 = np.where(xnew == xnew[i - 1])[0]
            yzm1 = np.where(y == y[xzm1])[0]
            y = np.insert(y, yzm1 + 1, [0])
    q = np.arange(0)
    for j in abs(y):
        if j == 0:
            if q[len(q) - 1] > 0:
                ymax = np.append(ymax, np.max(q))
            q = np.arange(0)
        q = np.append(q, j)
    significant_parameter = 2 * np.sqrt(np.trapz(Sy, dx=dw))
    all_amplitudes = np.append(all_amplitudes, ymax / significant_parameter)

Fy = np.linspace(1, 0, len(all_amplitudes), endpoint=False)
Fx = np.sort(all_amplitudes)
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(0, 10, 0.001)
m0 = np.trapz(Sy, dx=0.0001)
m1 = np.trapz(Sw * Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
nu = np.sqrt(m0 * m2 / (m1 ** 2) - 1)
ralaigh = np.exp(- 2 * (x**2))
raleigh_modified = 1 - (((np.exp(-2 * x**2)) / (1 + np.sqrt(1 - p**2))) * (np.sqrt(1 - p**2) * (np.sqrt(1 - p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p) + np.exp(2 * x**2) - erf((x * np.sqrt(2) * np.sqrt(1 - p**2)) / p) - 1) + (p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p)))
# ax.plot(Fx, Fy, color='red', alpha=.65, linewidth=5.5, label=f'ν={np.round(nu,2)}, ϵ={np.round(p,2)}')  # Linear
ax.plot(Fx, Fy, color='red', alpha=.85, linewidth=1.5, marker='.', label=f'ν={np.round(nu,2)}, ϵ={np.round(p,2)}')
ax.tick_params(labelsize=10)
ax.set(ylim=[10e-7, 1])
ax.set(xlim=[0, 3])
ax.plot(x, ralaigh, color='black', alpha=1, linestyle='--', linewidth=2, label='Rayleigh CDF')
ax.plot(x, raleigh_modified, color='black', alpha=1, linewidth=2, linestyle='-.', label='CDF(ϵ)')
ax.legend(fontsize=15)
ax.set_xlabel('a/aₛ', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
plt.yscale('log')
plt.show()

