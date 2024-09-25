import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.special import erf
from tqdm import tqdm

num_realizations = 250
w0_displacement = 21
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
extremas = np.arange(0)
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
    significant_parameter = 2 * np.sqrt(np.trapz(Sy, dx=dw))
    dy_1 = np.diff(y)
    dy_2 = np.diff(y, n=2)
    for i in range(1, len(dy_2)):
        if dy_1[i-1] > 0 >= dy_1[i] and dy_2[i-1] < 0 and y[i] > 0:
            extremas = np.append(extremas, y[i] / significant_parameter)
Fy = np.linspace(1, 0, len(extremas), endpoint=False)
Fx = np.sort(extremas)
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
ax.plot(Fx, Fy, color='red', alpha=.65, linewidth=5.5, label=f'ν={np.round(nu,2)}, ϵ={np.round(p,2)}')
ax.set(ylim=[0, 1])
ax.set(xlim=[0, 2])
ax.plot(x, ralaigh, color='black', alpha=1, linestyle='--', linewidth=2, label='Rayleigh CDF')
ax.plot(x, raleigh_modified, color='black', alpha=1, linewidth=2, linestyle='-.', label='CDF(ϵ)')
ax.legend(fontsize=15)
ax.set_xlabel('local positive maxima normalized to aₛ', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
np.save('WP_2', p)
np.save('LM_Medium_width_x', Fx)
np.save('LM_Medium_width_y', Fy)
ax.grid()
plt.show()

