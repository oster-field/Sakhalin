import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.interpolate import CubicSpline

t = np.load('sequence_to_reduce_t_gauss.npy')
y = np.load('sequence_to_reduce_y_gauss.npy')
dt = t[1]
ddtf = 0.0001
N = int((1 - dt) / ddtf)
Tmax = np.max(t)
s = CubicSpline(t, y)
amplitudes, extremas = 0, 0
F = np.arange(0)
x = np.arange(0)
for _ in tqdm(range(N), desc='Processing: ', colour='blue'):
    t = np.arange(0, Tmax, dt)
    y = s(t)
    t_back = t
    y_back = y
    tc, ti = pyaC.zerocross1d(t_back, y_back, getIndices=True)
    tnew = np.sort(np.append(t_back, tc))
    for i in range(1, len(tnew + 1)):
        if tnew[i] in tc:
            tzm1 = np.where(tnew == tnew[i - 1])[0]
            yzm1 = np.where(y_back == y_back[tzm1])[0]
            y_back = np.insert(y_back, yzm1 + 1, [0])
    q = np.arange(0)
    for j in y_back:
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
    eps = (2 * np.sqrt(1 - rho)) / (2 - rho)
    F = np.append(F, eps)
    x = np.append(x, 1 / dt)
    dt += ddtf
plt.plot(x, F, marker='.')
plt.show()
np.save('x_approaches_DP_gauss', x)
np.save('eps_experimental_DP_gauss', F)
np.save('eps_theoretical_DP_gauss', 0.48806094204453676)
