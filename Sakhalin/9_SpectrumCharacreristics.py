"""Выделяет из каждой 20-минутной записи (то есть перекрытие окнами в оконном Фурье) характеристики спектра."""
import numpy as np
import matplotlib.pyplot as plt
from functions import spectrum_width, Sensor_Frequency, DateStart, DateEnd, newdates
from scipy.signal.windows import hann
from scipy.fftpack import rfft, rfftfreq
import sys
import datetime
from tqdm import tqdm

WindowSize = 20
DeltaWindow = 1200
width = np.arange(0)
E = np.arange(0)
w0 = np.arange(0)
y = np.load('Data/FullRec.npy')
window = WindowSize * 60 * Sensor_Frequency
n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))
w = rfftfreq(window, 1 / Sensor_Frequency)

for i in tqdm(range(0, n), desc="Progress: ", colour='green'):
    arr = y[i*DeltaWindow:window + i*DeltaWindow]
    mask = hann(len(arr))
    s = np.abs(rfft(arr * mask))
    dx = 1 / len(w)
    m0 = np.trapz(s, dx=dx)
    E = np.append(E, m0)
    w0 = np.append(w0, np.trapz(w * s, dx=dx) / m0)
    width = np.append(width, spectrum_width(w, s))

np.save('Data/All_width', width)
np.save('Data/All_w0', w0)
np.save('Data/All_E', E)

dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
c = 0
pbar = tqdm(total=len(dates), desc="Saving: ", colour='green')

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            np.save('Data/' + filename + ' reading ' + str(i) + ' spectrum width', width[c])
            np.save('Data/' + filename + ' reading ' + str(i) + ' spectrum w0', w0[c])
            np.save('Data/' + filename + ' reading ' + str(i) + ' spectrum E', E[c])
            c += 1
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    ds += Deltadate

fig = plt.figure(num='Spectrum characteristics')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
rightlim = WindowSize / 60 + n * DeltaWindow / 3600
norm = np.max(np.arange(len(width / np.max(width)))) / rightlim
ax.plot(np.arange(len(E / np.max(E))) / norm, E / np.max(E),
        linewidth=2, color='#A63C00', marker='o', alpha=.3)
ax.plot(np.arange(len(w0 / np.max(w0))) / norm, w0 / np.max(w0),
        linewidth=2, color='#4F2982', marker='o', alpha=.3)
ax.plot(np.arange(len(width / np.max(width))) / norm, width / np.max(width),
        linewidth=2, color='#008110', marker='o', alpha=.3)
ax.plot(np.arange(0), np.arange(0), linewidth=2, color='#A63C00', label='Energy', marker='o')
ax.plot(np.arange(0), np.arange(0), linewidth=2, color='#4F2982', label='w0', marker='o')
ax.plot(np.arange(0), np.arange(0), linewidth=2, color='#008110', label='Width', marker='o')
ax.set_xlabel('Hours', fontsize=20)
ax.set_xlim(left=0, right=rightlim)
ax.set_ylabel('Spectral characteristics (normalized to maximum)', fontsize=22)
ax.grid()
plt.legend(fontsize=16)
plt.show()
