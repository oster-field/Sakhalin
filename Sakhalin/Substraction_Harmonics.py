"""Выделяет из каждой 20-минутной записи (то есть перекрытие окнами в оконном Фурье) характеристики спектра."""
import numpy as np
import matplotlib.pyplot as plt
from functions import Sensor_Frequency, DateStart, DateEnd, newdates
from scipy.signal.windows import hann
from scipy.fftpack import fft, fftfreq, ifft
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
rec = np.arange(0)
l = 0

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            l += len(arr) / Sensor_Frequency
            if 600 * 60 * 60 < l < 700 * 60 * 60:
                rec = np.append(rec, arr)
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    ds += Deltadate
s = fft(rec)
w = fftfreq(len(s), (1 / Sensor_Frequency) / (2 * np.pi))
for freq in range(len(w)):
    if abs(w[freq]) > 0.2:
        s[freq] = 0 + 0j
y = ifft(s).real
fig = plt.figure()
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot((np.arange(len(y)) / 8) / 60, y, linewidth=2, color='#046176')
ax.set_xlabel('t, min', fontsize=20)
ax.set_ylabel('η(t), m', fontsize=20)
ax.axhline(y=0, color='black', linewidth=1)
ax.grid(axis="y")
plt.show()

