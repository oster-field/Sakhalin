"""Удаление ненужных файлов, удаление спайков, сплайн-интерполяция."""
import numpy as np
import datetime
from functions import DateStart, DateEnd, rmsValue, Sensor_Frequency
import sys
from tqdm import tqdm
import os
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.fftpack import fft, ifft, fftfreq

minRMSvalue = 0.019  # Минимальное значение rms, при котором запись не удаляется
interpolationrate = 8  # Сколько точек в секунду будет после сплайн-интерполяции
print('Delete spikes? (Y/N)')
spikes = input()
print('Spline interpolation? (Y/N)')
interpolation = input()
dates = pd.date_range(DateStart, DateEnd).strftime('%d.%m').tolist()
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
Fl = False

while DateStart <= DateEnd:
    filename = DateStart.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            if len(arr) == 0 or np.mean(arr) == 0:
                os.remove('Data/' + filename + ' reading ' + str(i) + '.npy')
                print('Data/' + filename + ' reading ' + str(i) + '.npy removed as empty')
            else:
                s = fft(arr)
                x = fftfreq(len(arr), (1 / Sensor_Frequency) / (2 * np.pi))
                for freq in range(len(x)):
                    if abs(x[freq]) < np.pi / (10 * 30):  # Удаление гармоник длительностью > TMax минут
                        s[freq] = 0 + 0j
                arr_t = ifft(s).real
                if rmsValue(arr_t) < minRMSvalue:
                    os.remove('Data/' + filename + ' reading ' + str(i) + '.npy')
                    print('Data/' + filename + ' reading ' + str(i) + '.npy removed as LowRMS')
                if spikes == 'Y':
                    for j in range(len(arr) - 1):
                        if np.abs(arr[j + 1] - arr[j]) > 6 * np.sqrt(np.var(arr)):
                            arr[j + 1] = (arr[j] + arr[j + 2]) / 2
                            Fl = True
                    if Fl:
                        np.save('Data/' + filename + ' reading ' + str(i), arr)
                        print('Spike removed')
                        Fl = False
                if interpolation == 'Y':
                    cs = CubicSpline(np.arange(len(arr)), arr)
                    xs = np.arange(0, len(arr), 1 / interpolationrate)
                    np.save('Data/' + filename + ' reading ' + str(i), cs(xs))
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    DateStart += Deltadate
