"""Вычитание среднего, вычисление глубины погружения, построение графика смещений.
График можно строить несколько раз, пересчет будет только при первом
запуске программы. Удаленные дни на оси абсцисс пересчитываются."""
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functions import DateStart, DateEnd, seriesreducer, newdates
import sys
from tqdm import tqdm

Times = 10  # Насколько разредить запись, количество точек уменьшается в (2**Times)
isconverted = open('Data/isconverted.txt').readlines()[0].strip()
MeasurmentError = np.load('Data/MeasurmentError.npy')
dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Depth calculating: ", colour='green')
pbar2 = tqdm(total=len(dates), desc="Progress: ", colour='green')
y = np.arange(0)
ticks = np.arange(0)
meanarr = np.arange(0)
c = 0
Fl = False

if isconverted == 'Not converted':
    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
                arr = np.delete(arr, np.where(arr == 0))
                if len(arr) != 0:
                    depth = np.mean(arr) - MeasurmentError
                    meanarr = np.append(meanarr, depth)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' Depth', depth)
                np.save('Data/' + filename + ' reading ' + str(i) + '.npy', arr)
            except FileNotFoundError:
                Error = True
            if Error:
                break
        pbar.update(1)
        ds += Deltadate
    mean = np.mean(meanarr)
    np.save('Data/Average Depth', mean)
    dates, ds, de = newdates(DateStart, DateEnd)


while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            if isconverted == 'Not converted':
                Fl = True
                np.save('Data/' + filename + ' reading ' + str(i) + '.npy', arr - mean)
                y = np.append(y, seriesreducer(arr - mean, Times))
            elif isconverted == 'Converted':
                y = np.append(y, seriesreducer(arr, Times))
        except FileNotFoundError:
            Error = True
        if Error:
            break
    ticks = np.append(ticks, c + round((len(y) - c) / 2))
    c = len(y)
    pbar2.update(1)
    ds += Deltadate

if Fl:
    with open('Data/isconverted.txt', 'w') as file:
        file.write('Converted')
    np.save('Data/ConvertedPlot', y)

fig = plt.figure(num='Converted plot')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(np.arange(len(y)), y, linewidth=2, color='#046176', label='Water surface displacement')
ax.set_xlabel('Days', fontsize=20)
ax.set_ylabel('η(t), [m]', fontsize=20)
ax.set_xticks(ticks[::3])
ax.set_xticklabels(dates[::3], rotation=30)
ax.axhline(y=0, color='black', linewidth=1)
ax.grid(axis="y")
plt.legend(fontsize=16)
plt.show()
