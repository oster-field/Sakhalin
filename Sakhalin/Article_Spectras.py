"""Выделяет из каждой 20-минутной записи (то есть перекрытие окнами в оконном Фурье) характеристики спектра."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, rfftfreq
import sys
import datetime
from tqdm import tqdm
import pandas as pd
from matplotlib import ticker


def newdates(ds, de, number):
    dates = pd.date_range(ds, de).strftime('%d.%m').tolist()
    deleteddates = []
    for i in range(1, 3):
        for line in open(f'Data{number}/log' + str(i) + '.txt').readlines():
            day = line.strip()[8::]
            month = line.strip()[5:7]
            deleteddates.append(day + '.' + month)
    dates = [date for date in dates if date not in deleteddates]
    Deltadate = datetime.timedelta(days=1)
    ds += Deltadate * len(
        open(f'Data{number}/log1.txt').readlines())  # Даты начала и конца записи волнения (без погружений)
    de -= Deltadate * len(open(f'Data{number}/log2.txt').readlines())
    return dates, ds, de


def Spectrum(lowerbound, upperbound, number):
    try:
        DateStart = datetime.datetime.strptime((open(f'DataTXT{number}_done/INFO.dat').readlines()[5].strip()),
                                               '%Y.%m.%d %H:%M:%S.%f').date()
        DateEnd = datetime.datetime.strptime((open(f'DataTXT{number}_done/INFO.dat').readlines()[7].strip()),
                                             '%Y.%m.%d %H:%M:%S.%f').date()
    except FileNotFoundError:
        DateStart = None
        DateEnd = None
    dates, ds, de = newdates(DateStart, DateEnd, number)
    Deltadate = datetime.timedelta(days=1)
    pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
    rec = np.arange(0)
    interval = 0
    Sensor_Frequency = 8
    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = np.load(f'Data{number}/{filename} reading {i}.npy')
                interval += len(arr) / Sensor_Frequency
                if lowerbound * 60 * 60 < interval < upperbound * 60 * 60:
                    rec = np.append(rec, arr)
            except FileNotFoundError:
                Error = True
            if Error:
                break
        pbar.update(1)
        ds += Deltadate
    s = np.abs(rfft(rec))
    w = rfftfreq(len(s), (1 / Sensor_Frequency) / (2 * np.pi))
    s = (s ** 2) / (len(rec) * np.max(w))
    ind = np.where(w < 5.75)
    return w[ind], s[ind]


w1, s1 = Spectrum(0, 100, 3)
w2, s2 = Spectrum(0, 100, 2)
w3, s3 = Spectrum(200, 400, 1)
w4, s4 = Spectrum(430, 650, 3)
fig, ax = plt.subplots(2, 2)
for i in range(2):
    for j in range(2):
        ax[i, j].tick_params(labelsize=20)
        ax[i, j].set_xlabel('ω, [rad/s]', fontsize=20)
        ax[i, j].set_ylabel('S(ω), [m²/s]', fontsize=20)
        ax[i, j].set(xlim=[0, 2])
        ax[i, j].grid(axis="y")
        ax[i, j].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
ax[0, 1].plot(w1, s1, linewidth=2, color='#007241')
ax[0, 1].set(ylim=[0, np.max(s1) + 0.05])
ax[1, 1].plot(w2, s2, linewidth=2, color='#007241')
ax[1, 1].set(ylim=[0, np.max(s2) + 0.004])
ax[0, 0].plot(w3, s3, linewidth=2, color='#007241')
ax[0, 0].set(ylim=[0, np.max(s3) + 0.01])
ax[1, 0].plot(w4, s4, linewidth=2, color='#007241')
ax[1, 0].set(ylim=[0, np.max(s4) + 0.02])
'''wp = 0.625
gamma = 1.33
const = 0.00088'''
wp = 0.61
gamma = 5.5
const = 0.00023
x1 = np.arange(0.001, wp, 0.001)
x2 = np.arange(wp, np.max(w4), 0.001)
JONSWAP1 = (const * (9.8**2) / (x1 ** 5)) * np.exp(-1.25 * (wp / x1)**4) * gamma**(np.exp(-(x1 - wp)**2 / (2 * (0.07 * wp)**2)))
JONSWAP2 = (const * (9.8**2) / (x2 ** 5)) * np.exp(-1.25 * (wp / x2)**4) * gamma**(np.exp(-(x2 - wp)**2 / (2 * (0.09 * wp)**2)))
ax[1, 0].plot(x1, JONSWAP1, linewidth=2, color='#A40004', label=f'JONSWAP, γ={gamma}')
ax[1, 0].plot(x2, JONSWAP2, linewidth=2, color='#A40004')
ax[1, 0].legend(fontsize=16)
plt.subplots_adjust(left=0.064, bottom=0.083, right=0.97, top=0.974, wspace=0.2, hspace=0.2)
plt.show()

