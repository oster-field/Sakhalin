import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from scipy.stats import kurtosis, skew
from tqdm import tqdm
from scipy.special import erf

Kc = 1e10
K = 1e-10
Q = ((K * Kc) / (K + 2 * np.sqrt(K * Kc) + Kc)) ** (1 / 3)
k0 = np.sqrt(Q / Kc) + 700


def spectrum_values(K):
    Q = ((K * Kc) / (K + 2 * np.sqrt(K * Kc) + Kc)) ** (1 / 3)
    k1 = np.arange(k0 - np.sqrt(Q / Kc), k0, 0.0001)
    k2 = np.arange(k0, k0 + np.sqrt(Q / K), 0.0001)
    global LastK
    LastK = max(k2)
    y1 = -Kc * (k1 - k0) ** 2 + Q
    y2 = -K * (k2 - k0) ** 2 + Q
    k = np.append(k1, k2)
    y = np.append(y1, y2)
    return k, y


def spectum_plotting(K):
    x, y = spectrum_values(K)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('k, [1/m]', fontsize=20)
    ax.set_ylabel('S(k)', fontsize=20)
    ax.set(xlim=[x.min(), x.max()],
           ylim=[y.min(), y.max()])
    ax = plt.gca()
    m0 = np.trapz(y, dx=0.0001)
    w0 = (np.trapz(x * y, dx=0.0001) / m0)
    m2 = np.trapz(((x - w0) ** 2) * y, dx=0.0001)
    width = np.sqrt(m2 / (w0 * w0 * m0))
    ax.plot(x, y, color='b', label=f'Width={width}')
    ax.vlines(k0, y.min(), y.max(), color='dimgray', alpha=0.6, linestyle='--')
    ax.legend(fontsize=15)
    plt.show()


def wavefield_values(K):
    N = 2 ** 12
    dk = (15 * LastK) / N
    dx = 2 * np.pi / (15 * LastK)
    L = dx * N
    x = np.arange(0, L, dx, dtype=np.float64)
    k = 0
    y = 0
    Q = ((K * Kc) / (K + 2 * np.sqrt(K * Kc) + Kc)) ** (1 / 3)
    for i in range(0, N):
        k = k + dk
        v = random.uniform(0, 2 * np.pi)
        if k < k0:
            S = -Kc * (k - k0) ** 2 + Q
            if S < 0:
                S = 0
        elif k > k0:
            S = -K * (k - k0) ** 2 + Q
            if S < 0:
                S = 0
        MonochromaticWave = (np.sqrt(2 * dk * S)) * (np.cos(k * x + v))
        y = y + MonochromaticWave
    return x, y


def wavefield_plotting(K):
    x, y = wavefield_values(K)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax = plt.gca()
    ax.set_xlabel('x, [m]', fontsize=20)
    ax.set_ylabel('η(x, t = 0), [m]', fontsize=20)
    plt.title(
        'M0 = ' + str(np.mean(y)) + ' M1 = ' + str(np.var(y)) + 'M2 = ' + str(skew(y)) + ' M3 = ' + str(kurtosis(y)))
    ax.plot(x, y, color='b')
    plt.show()


def measurments(K):
    x, y = wavefield_values(K)
    ymax = np.arange(0)
    ymin = np.arange(0)
    xc, xi = pyaC.zerocross1d(x, y, getIndices=True)
    xnew = np.sort(np.append(x, xc))
    for i in range(1, len(xnew + 1)):
        if xnew[i] in xc:
            xzm1 = np.where(xnew == xnew[i - 1])[0]
            yzm1 = np.where(y == y[xzm1])[0]
            y = np.insert(y, yzm1 + 1, [0])
    q = np.arange(0)
    for j in y:
        if j == 0:
            if q[len(q) - 1] > 0:
                ymax = np.append(ymax, np.max(q))
            else:
                ymin = np.append(ymin, np.min(q))
            q = np.arange(0)
        q = np.append(q, j)
    wavelenght = np.arange(0)
    if len(ymax) >= len(ymin):
        for i in range(0, len(ymin)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    if len(ymax) < len(ymin):
        for i in range(0, len(ymax)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    return ymax, ymin, wavelenght


def creating_series(N, K):
    allymax = np.arange(0)
    allymin = np.arange(0)
    allheights = np.arange(0)
    for i in tqdm(range(0, N), colour='green', desc='Creating realizations '):
        ymax, ymin, wavelenght = measurments(K)
        allymax = np.append(allymax, ymax)
        allymin = np.append(allymin, ymin)
        allheights = np.append(allheights, wavelenght)
    return allymax, allymin, allheights


def distribution_function(arr, Significant):
    y = np.linspace(1, 0, len(arr), endpoint=False)
    x = np.sort(arr)
    x /= Significant
    return x, y


def distribution_value(K):
    allymax, allymin, allwavelenght = creating_series(2, K)
    x0, y0 = distribution_function(allymax, 2 * np.sqrt(2 / 3))
    x2, y2 = distribution_function(allwavelenght, 4 * np.sqrt(2 / 3))
    return x0, y0, x2, y2


def distribution_measurments_plotting(K):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = np.arange(0, 10, 0.001)
    xs, ys = spectrum_values(K)
    m0 = np.trapz(ys, dx=0.0001)
    m1 = np.trapz(xs * ys, dx=0.0001)
    m2 = np.trapz((xs ** 2) * ys, dx=0.0001)
    m4 = np.trapz((xs ** 4) * ys, dx=0.0001)
    p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
    nu = np.sqrt(m0 * m2 / (m1 ** 2) - 1)
    # ralaigh = np.exp((-1 * (x ** 2)) / (2 * m0))
    ralaigh = np.exp(- 2 * (x**2))
    theta = x / np.sqrt(m0)
    # raleigh_modified = 1 - (((np.exp((-1 * theta**2) / (2))) / (1 + np.sqrt(1 - p**2))) * (np.sqrt(1 - p**2) * (np.sqrt(1 - p**2) * np.exp((theta**2) / 2) * erf(theta / (np.sqrt(2) * p)) + np.exp((theta**2) / 2) - erf((theta * np.sqrt(1 - p**2)) / (np.sqrt(2) * p)) - 1) + (p**2) * np.exp((theta**2)/2) * erf(theta / (p * np.sqrt(2)))))
    raleigh_modified = 1 - (((np.exp(-2 * x**2)) / (1 + np.sqrt(1 - p**2))) * (np.sqrt(1 - p**2) * (np.sqrt(1 - p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p) + np.exp(2 * x**2) - erf((x * np.sqrt(2) * np.sqrt(1 - p**2)) / p) - 1) + (p**2) * np.exp(2 * x**2) * erf((x * np.sqrt(2)) / p)))
    ax.set_xlabel('H/Hₛ', fontsize=20)
    ax.set_ylabel('CDF', fontsize=20)
    x0, y0, x2, y2 = distribution_value(K)
    ax.plot(x2, y2, color='green', alpha=.65, linewidth=5.5, label=f'ν={np.round(nu,3)}, ϵ={np.round(p,3)}')
    ax.tick_params(labelsize=10)
    ax.set(ylim=[10e-7, 1])
    ax.set(xlim=[0, 2])
    ax.plot(x, ralaigh, color='black', alpha=1, linestyle='--', linewidth=2, label='Rayleigh CDF')
    ax.plot(x, raleigh_modified, color='black', alpha=1, linewidth=2, linestyle='-.', label='CDF(ϵ)')
    ax.legend(fontsize=15)
    ax.tick_params(labelsize=20)
    # plt.yscale('log')
    plt.show()


distribution_measurments_plotting(K)
