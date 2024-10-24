"""Сравнение всех полученных распределений"""
import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.special import erf


def formula(a, epsilon):
    term1 = (np.sqrt(2) * (1 + epsilon ** 4)) / epsilon * a
    term2 = np.exp(-2 * a ** 2)
    term3 = (np.sqrt(2) * np.sqrt(1 - epsilon ** 2)) / epsilon * a
    result = 0.5 * (1 - erf(term1) + term2 * (1 + erf(term3)))
    return result


o = input('Create / Picture? (C/P): ')
if o == 'C':
    for _ in range(2):
        num_realizations = 2000
        w0_displacement = 1.5
        WDT = 400
        N = 2 ** 13
        dt = 2 * np.pi / WDT
        dw = WDT / N
        L = dt * N
        amplitudes = np.arange(0, dtype=np.float64)
        extremas = np.arange(0, dtype=np.float64)
        t = np.arange(0, L, dt, dtype=np.float64)
        c = 0
        for counter in tqdm(range(0, num_realizations), desc='Realizations: ', colour='yellow'):
            y = 0
            w = 0
            Sy = np.arange(0)
            Sw = np.arange(0)
            for i in range(0, N):
                w = w + dw
                v = random.uniform(0, 2*np.pi)
                if _ == 0:
                    S = 1 / ((w - w0_displacement) ** 6 + 1)
                else:
                    S = np.exp(-0.91 * (w - w0_displacement) ** 2)
                Sy = np.append(Sy, S)
                Sw = np.append(Sw, w)
                MonochromaticWave = (np.sqrt(2*dw*S))*(np.cos(w*t+v))
                y = y + MonochromaticWave
            y_1 = y
            t_1 = t
            '''if counter == 0:
                plt.plot(t, y)
                plt.show()'''
            As = 2 * np.sqrt(np.trapz(Sy, dx=dw))
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
                    q = np.append(q, 0)
                    if np.sum(q) < 0:
                        q = np.abs(q)
                        amplitudes = np.append(amplitudes, np.max(q) / As)
                    q = np.arange(0)
                q = np.append(q, j)
        Fy_A = np.linspace(1, 0, len(amplitudes), endpoint=False)
        Fx_A = np.sort(amplitudes)
        m0 = np.trapz(Sy, dx=dw)
        m2 = np.trapz((Sw ** 2) * Sy, dx=dw)
        m4 = np.trapz((Sw ** 4) * Sy, dx=dw)
        p = np.sqrt(1 - (m2 ** 2) / (m0 * m4))
        np.save('MWP_my.npy', p)
        if _ == 0:
            np.save('MWX_my.npy', Fx_A)
            np.save('MWY_my.npy', Fy_A)
            print(f'Mine p is {p} and len(a) is {len(amplitudes)}')
        else:
            np.save('MWX_gaussian.npy', Fx_A)
            np.save('MWY_gaussian.npy', Fy_A)
            print(f'Gaussian p is {p} and len(a) is {len(amplitudes)}')

elif o == 'P':
    Fx_A = np.load('MWX_my.npy')
    Fy_A = np.load('MWY_my.npy')
    p = np.load('MWP_my.npy')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
    ax.set_ylabel(f'CDF, p={np.round(p, 2)}', fontsize=20)
    ax.plot(Fx_A, Fy_A, linewidth=2.5, color='red')
    ax.plot(Fx_A, formula(Fx_A, p), linewidth=2.5, color='green')
    ax.plot(Fx_A, np.exp(-2 * Fx_A**2), color='black')
    Fx_A = np.load('MWX_gaussian.npy')
    Fy_A = np.load('MWY_gaussian.npy')
    ax.plot(Fx_A, Fy_A, linewidth=2.5, color='blue')
    ax.tick_params(labelsize=20)
    ax.grid()
    plt.show()
