import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


w0_displacement = 100
Wc = 1e5
W = 1e-5
K = 0.12
Q = ((W * Wc) / (W + 2 * np.sqrt(W * Wc) + Wc)) ** (1 / 3)
eps_arr = np.arange(0)
nu_arr = np.arange(0)
for i in tqdm(range(0, 99), colour='green', desc='Calculating '):
    w = np.arange(0, 110, 0.0001)
    S = np.exp(-((w-w0_displacement)**2)/(2 * K**2))
    m0 = np.trapz(S, dx=0.0001)
    m1 = np.trapz(w * S, dx=0.0001)
    m2 = np.trapz((w ** 2) * S, dx=0.0001)
    m4 = np.trapz((w ** 4) * S, dx=0.0001)
    eps = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
    nu = np.sqrt(m0 * m2 / (m1 ** 2) - 1)
    eps_arr = np.append(eps_arr, eps)
    nu_arr = np.append(nu_arr, nu)
    w0_displacement -= 1
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(eps_arr, color='#A63100', marker='o', linewidth=2, label='ϵ')
ax.plot(nu_arr, color='#007143', linewidth=2, marker='o', label='ν')
ax.legend(fontsize=30)
ax.set_xlabel('Spectrum expansion', fontsize=20)
ax.set_ylabel('Value of parameter', fontsize=20)
ax.grid()
plt.show()
