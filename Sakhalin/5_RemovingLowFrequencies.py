"""Удаление высокочастотных компонент для стабильного нулевого уровня. Построение графика до/после удаления.
Нужно запустить два раза: первый - преобразование, второй - график."""
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.fftpack import fft, ifft, fftfreq
from functions import DateStart, DateEnd, seriesreducer, split_array, Sensor_Frequency, newdates, edge_effect
import sys
from tqdm import tqdm

Times = 10  # Насколько разредить запись, количество точек уменьшается в (2**Times)
TMax = 10  # Минимальная длительность волн, которые останутся после преобразования (в минутах)
dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Progress: ", colour='green')
y = np.arange(0)
ticks = np.arange(0)
sizes = np.arange(0)
filenames = []
c = 0
istransformed = open('Data/istransformed.txt').readlines()[0].strip()

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            if len(arr) != 0 and istransformed == 'Not transformed':
                y = np.append(y, arr)
                sizes = np.append(sizes, len(arr))
                filenames.append('Data/' + filename + ' reading ' + str(i) + '.npy')
            else:
                y = np.append(y, seriesreducer(arr, Times))
        except FileNotFoundError:
            Error = True
        if Error:
            break
    ticks = np.append(ticks, c + round((len(y) - c) / 2))
    c = len(y)
    pbar.update(1)
    ds += Deltadate

if istransformed == 'Not transformed':
    s = fft(y)
    x = fftfreq(len(y), (1 / Sensor_Frequency) / (2 * np.pi))
    for freq in range(len(x)):
        if abs(x[freq]) < np.pi / (TMax * 30):  # Удаление гармоник длительностью > TMax минут
            s[freq] = 0 + 0j
    y = ifft(s).real
    splittedy = split_array(y, sizes)
    np.save('Data/SpectrumY', s)
    np.save('Data/SpectrumX', x)
    np.save('Data/Samples', len(y))
    y = y[sizes[0]:-sizes[-1]]
    np.save('Data/FullRec', y)
    for i in range(len(filenames)):
        if i == 1 or i == len(filenames) - 1:
            np.save(filenames[i], edge_effect(splittedy[i]))
        elif i == 0 or i == len(filenames):
            np.save(filenames[i], np.arange(0))
        else:
            np.save(filenames[i], splittedy[i])
    with open('Data/istransformed.txt', 'w') as file:
        file.write('Transformed')
    print('Transformed, lounch one more time')
else:
    fig = plt.figure(num='Before and after Fourier transforms plot')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.plot((len(y)/len(np.load('Data/ConvertedPlot.npy'))) * np.arange(len(np.load('Data/ConvertedPlot.npy'))),
            np.load('Data/ConvertedPlot.npy'), linewidth=2,
            color='#760461', label='Before Fourier transforms', alpha=.4)
    ax.plot(np.arange(len(y)), y, linewidth=2, color='#046176', label='After Fourier transforms')
    ax.set_xlabel('Days', fontsize=20)
    ax.set_ylabel('η(t), [m]', fontsize=20)
    ax.set_xticks(ticks[::3])
    ax.set_xticklabels(dates[::3], rotation=30)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis="y")
    plt.legend(fontsize=16)
    plt.show()
