import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
import sys
import os

num_realizations = 2
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
F = np.arange(0, dtype=int)
x = np.arange(0, dtype=np.float64)
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
    extr_per_zerocrossing = 0
    for j in y:
        if j == 0:
            q = np.abs(q)
            dq_1 = np.diff(q)
            dq_2 = np.diff(q, n=2)
            for i in range(1, len(dq_2)):
                if dq_1[i - 1] > 0 >= dq_1[i] and dq_2[i - 1] < 0:
                    extr_per_zerocrossing += 1
            if extr_per_zerocrossing != 0:
                F = np.append(F, extr_per_zerocrossing)
                x = np.append(x, np.max(q) / 1.63)
            else:
                F = np.append(F, 1)
                x = np.append(x, np.max(q) / 1.63)
            extr_per_zerocrossing = 0
            q = np.arange(0)
        q = np.append(q, j)
    if len(F) > 10000:
        np.save(f'F{c}', F)
        np.save(f'x{c}', x)
        F = np.arange(0, dtype=int)
        x = np.arange(0, dtype=np.float64)
        c += 1
        print('Scatters saved')
for i in range(0, sys.maxsize):
    try:
        F = np.append(F, np.load(f'F{i}.npy'))
        os.remove(f'F{i}.npy')
        x = np.append(x, np.load(f'x{i}.npy'))
        os.remove(f'x{i}.npy')
    except FileNotFoundError:
        break
pairs = zip(F, x)
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
sorted_F, sorted_x = zip(*sorted_pairs)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(sorted_x, sorted_F, color='#007439', alpha=0.1, linewidth=0, marker='o')
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel('Number of local maxima in an individual wave', fontsize=20)
ax.tick_params(labelsize=20)
ax.grid()
plt.show()
