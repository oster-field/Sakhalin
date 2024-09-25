"""Выделяет из каждой 20-минутной записи (то есть перекрытие окнами в оконном Фурье) характеристики спектра."""
import numpy as np
import matplotlib.pyplot as plt
from functions import Sensor_Frequency, DateStart, DateEnd, newdates
from scipy.signal.windows import hann
from scipy.fftpack import rfft, rfftfreq, irfft
from PyAstronomy import pyaC
import sys
import datetime
from tqdm import tqdm

width = np.arange(0)
width_eps = np.arange(0)
width_eps_rho = np.arange(0)
goda = np.arange(0)
dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
if Sensor_Frequency == 8:
    n = 5.75
else:
    n = np.pi

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            mask = hann(len(arr))
            s = np.abs(rfft(arr * mask))
            w = rfftfreq(len(s), (1 / Sensor_Frequency) / (2 * np.pi))
            ind = np.where(w < n)
            s = s[ind]
            w = w[ind]
            dx = w[1]
            m0 = np.trapz(s, dx=dx)
            m1 = np.trapz(w * s, dx=dx)
            m2 = np.trapz((w**2) * s, dx=dx)
            m4 = np.trapz((w**4) * s, dx=dx)
            Q = np.trapz(w * s**2, dx=dx) / m0**2
            goda = np.append(goda, Q)
            np.save('Data/' + filename + ' reading ' + str(i) + ' goda', Q)
            nu = np.sqrt(((m0 * m2) / m1**2) - 1)
            np.save('Data/' + filename + ' reading ' + str(i) + ' nu', nu)
            eps = np.sqrt(1 - (m2**2)/(m0 * m4))
            np.save('Data/' + filename + ' reading ' + str(i) + ' eps_width', eps)
            width = np.append(width, nu)
            width_eps = np.append(width_eps, eps)
            amplitudes, extremas = 0, 0
            y = irfft(rfft(arr)[ind])
            t = np.arange(len(y))
            tc, ti = pyaC.zerocross1d(t, y, getIndices=True)
            tnew = np.sort(np.append(t, tc))
            for c1 in range(1, len(tnew + 1)):
                if tnew[c1] in tc:
                    tzm1 = np.where(tnew == tnew[c1 - 1])[0]
                    yzm1 = np.where(y == y[tzm1])[0]
                    y = np.insert(y, yzm1 + 1, [0])
            q = np.arange(0)
            for j in y:
                if j == 0:
                    q = np.abs(q)
                    q = np.append(q, 0)
                    amplitudes += 1
                    for c2 in range(1, len(q) - 1):
                        if q[c2] > q[c2 - 1] and q[c2] > q[c2 + 1]:
                            extremas += 1
                    q = np.arange(0)
                q = np.append(q, j)
            rho = amplitudes / extremas
            if rho > 1:
                rho = 1
            eps = (2 * np.sqrt(1 - rho)) / (2 - rho)
            if eps == 0:
                eps = np.mean(width_eps_rho)
            np.save('Data/' + filename + ' reading ' + str(i) + ' rho', eps)
            width_eps_rho = np.append(width_eps_rho, eps)
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    ds += Deltadate

'''np.save('Data/All_nu', width)
np.save('Data/All_eps_width', width_eps)
np.save('Data/All_rho', width_eps_rho)
np.save('Data/All_goda', goda)'''

fig = plt.figure(num='Spectrum characteristics')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(width_eps, linewidth=1, color='red', marker='o', alpha=.45, label='ϵ')
ax.plot(width_eps_rho, linewidth=1, color='green', marker='o', alpha=.45, label='$ϵ_d$')
#ax.plot(goda, linewidth=1, color='blue', marker='o', alpha=.45, label='$Q_p$')
#ax.plot(width, linewidth=1, color='black', marker='o', alpha=.45, label='ν')
ax.set_xlabel('N', fontsize=20)
ax.set_ylabel('Spectral characteristics', fontsize=22)
ax.grid()
plt.legend(fontsize=16)
plt.show()
