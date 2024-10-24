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
    ND = 11
    l = 0
    r = int(len(rec) / ND)
    w_smooth = rfftfreq(int(len(rec) / ND), (1 / Sensor_Frequency) / (2 * np.pi))
    s_smooth = np.zeros(len(w_smooth))
    for _ in range(2 * ND - 1):
        s_smooth = np.add(s_smooth, np.abs(rfft(rec[l:r])))
        l += int(len(rec) / (ND * 2))
        r += int(len(rec) / (ND * 2))
    s_smooth /= ND
    s_smooth = (s_smooth ** 2) / (int(len(rec) / ND) * np.max(w_smooth))
    return w, s, w_smooth, s_smooth


w, s, w_smooth, s_smooth = Spectrum(0, 100, 2)
fig, ax = plt.subplots(1, 1)
ax.tick_params(labelsize=20)
ax.set_xlabel('ω, [rad/s]', fontsize=20)
ax.set_ylabel('S(ω), [m²/s]', fontsize=20)
ax.set(xlim=[np.pi / (10 * 30), 5], ylim=[10**(-7), 10**(-2)])
ax.grid(axis="y")
# ax.plot(w, s, linewidth=2, color='#447BD4')
ax.plot(w_smooth, s_smooth, linewidth=2, color='#447BD4')
solitary = np.where(w < 0.228)[0]
tail = np.where(w > 1)[0]
'''ax.plot(w[solitary], s[solitary], linewidth=2, color='#A64B00')
ax.plot(w[tail], s[tail], linewidth=2, color='#7109AA')'''
solitary_reference_1 = [0.201656, 0.000246749]
solitary_reference_2 = [0.065, 0.00056]
tail_reference = [1.54, 8.79718e-05]
a = solitary_reference_1[0] * solitary_reference_1[1]
b = solitary_reference_2[0]**(11/3) * solitary_reference_2[1]
c = tail_reference[0]**(4/3) * tail_reference[1]
d = tail_reference[0] ** 4 * tail_reference[1]
x = w[solitary]
z = w[tail]
ax.plot(x, a * x**(-1), c='black', linewidth=2, label=r"$ω^{-1}$")
ax.plot(x, b * x**(-11/3), c='r', linewidth=2, linestyle='dashed', label=r"$ω^{-11/3}$")
ax.plot(z, d * z**(-4), c='green', linewidth=2, linestyle='dashed', label=r"$ω^{-4}$")
ax.plot(z, c * z**(-4/3), c='#E439A1', linewidth=2, label=r"$ω^{-4/3}$")
ax.legend(fontsize=21)
plt.subplots_adjust(left=0.064, bottom=0.083, right=0.97, top=0.974, wspace=0.2, hspace=0.2)
plt.yscale('log')
plt.xscale('log')
plt.show()

