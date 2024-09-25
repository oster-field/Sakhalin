import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.fftpack import rfft, rfftfreq

num_realizations = 50
Wc = 1e5
W = 1e-5
Q = ((W * Wc) / (W + 2 * np.sqrt(W * Wc) + Wc)) ** (1 / 3)
w0 = np.sqrt(Q / Wc)
part1 = np.arange(0, w0, 0.0001)
part2 = np.arange(w0, w0 + np.sqrt(Q / W), 0.0001)
Sw = np.append(part1, part2)
Sy = np.append(-Wc * (part1 - np.sqrt(Q / Wc))**2 + Q, -W * (part2 - np.sqrt(Q / Wc))**2 + Q)
eps_experimental = np.arange(0)
eps_spectral = np.arange(0)
x = np.arange(0)
c = 0.1
for _ in tqdm(range(85), desc='Processing: ', colour='blue'):
    WDT = c * (w0 + np.sqrt(Q / W))
    dt = 2 * np.pi / WDT
    N = int(3000 + 500 * c**1.5)
    dw = WDT / N
    L = dt * N
    t = np.arange(0, L, dt, dtype=np.float64)
    Sensor_Frequency = np.sum((0 <= t) & (t <= 1))
    eps_spectral_array = np.arange(0)
    amplitudes, extremas = 0, 0
    coord = np.arange(0)
    for counter in range(num_realizations):
        y, w = 0, 0
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
        s_w = rfftfreq(len(y), (1 / Sensor_Frequency) / (2 * np.pi))
        s_a = np.abs(rfft(y).real)
        eps_spectral_array = np.append(eps_spectral_array, np.sqrt(1 - (np.trapz((s_w ** 2) * s_a, dx=0.0001) ** 2) / (np.trapz(s_a, dx=0.0001) * np.trapz((s_w ** 4) * s_a, dx=0.0001))))
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
                q = np.append(q, 0)
                amplitudes += 1
                for i in range(1, len(q) - 1):
                    if q[i] > q[i - 1] and q[i] > q[i + 1]:
                        extremas += 1
                q = np.arange(0)
            q = np.append(q, j)
    rho = amplitudes / extremas
    eps_experimental = np.append(eps_experimental, (2 * np.sqrt(1 - rho)) / (2 - rho))
    eps_spectral = np.append(eps_spectral, np.mean(eps_spectral_array))
    x = np.append(x, int(Sensor_Frequency))
    c += 0.1

m0 = np.trapz(Sy, dx=0.0001)
m2 = np.trapz((Sw ** 2) * Sy, dx=0.0001)
m4 = np.trapz((Sw ** 4) * Sy, dx=0.0001)
eps_theoretical = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
pairs = zip(x, eps_experimental, eps_spectral)
sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
x, eps_experimental, eps_spectral = zip(*sorted_pairs)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axhline(eps_theoretical, color='black', linewidth=3)
ax.plot(x, eps_experimental, color='#A63E00', alpha=.85, linewidth=2.5, marker='.', label='$ϵ_{direct}$')
ax.plot(x, eps_spectral, color='#006C51', alpha=.85, linewidth=2.5, marker='.', label='$ϵ_{spectral}$')
np.save('x_approaches_DP', x)
np.save('eps_experimental_DP', eps_experimental)
np.save('eps_spectral_DP', eps_spectral)
np.save('eps_theoretical_DP', eps_theoretical)
ax.set_xlabel('Discretization frequency', fontsize=20)
ax.set_ylabel('Parameter value', fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=15)
ax.grid()
plt.show()
