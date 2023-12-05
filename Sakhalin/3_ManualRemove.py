"""Удаление точек, не являющихся записями волнения. Вспомогательной функцией находятся промежутки погружения
и строится их график с указанием названия файла. Нажатие на точку на открывшемся графике удаляет точки до/после."""
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functions import lowmean, extractlastnumber, DateStart, DateEnd
import pandas as pd
import sys


def onclick(event):
    if event.inaxes == ax[0] or event.inaxes == ax[1]:
        xclick = round(event.xdata)
        if event.inaxes == ax[0]:
            if xclick <= len(np.load(filenames[0])):
                ax[0].axvspan(0, xclick, color='red', alpha=.4)
                ax[0].plot(np.arange(0, xclick), y[np.arange(0, xclick)], linewidth=2, color='black')
                plt.draw()
                ar = np.load(filenames[0])
                for point in range(0, xclick):
                    ar[point] = 0
                np.save(filenames[0], ar)
                for reading in range(1, extractlastnumber(filenames[0])):
                    np.save('Data/' + filenames[0][5:15] + ' reading ' + str(reading) + '.npy',
                            0*np.load('Data/' + filenames[0][5:15] + ' reading ' + str(reading) + '.npy'))
                    print(filenames[0][5:15] + ' reading ' + str(reading) + '.npy nullified')
                print(filenames[0][5::] + ' partically nullified')
            else:
                ax[0].axvspan(0, xclick, color='red', alpha=.4)
                ax[0].plot(np.arange(0, xclick), y[np.arange(0, xclick)], linewidth=2, color='black')
                plt.draw()
                ar = np.load(filenames[1])
                for point in range(0, xclick - len(np.load(filenames[0]))):
                    ar[point] = 0
                np.save(filenames[1], ar)
                for reading in range(1, extractlastnumber(filenames[1])):
                    np.save('Data/' + filenames[1][5:15] + ' reading ' + str(reading) + '.npy',
                            0*np.load('Data/' + filenames[1][5:15] + ' reading ' + str(reading) + '.npy'))
                    print(filenames[1][5:15] + ' reading ' + str(reading) + '.npy nullified')
                print(filenames[1][5::] + ' partically nullified')
        if event.inaxes == ax[1]:
            if xclick <= len(np.load(filenames[2])):
                ax[1].axvspan(xclick, len(p), color='red', alpha=.4)
                ax[1].plot(np.arange(xclick, len(p)), p[np.arange(xclick, len(p))], linewidth=2, color='black')
                plt.draw()
                ar = np.load(filenames[2])
                for point in range(xclick, len(ar)):
                    ar[point] = 0
                np.save(filenames[2], ar)
                print(filenames[2][5::] + ' partically nullified')
                for reading in range(extractlastnumber(filenames[2]) + 1, sys.maxsize):
                    Err = False
                    try:
                        np.save('Data/' + filenames[2][5:15] + ' reading ' + str(reading) + '.npy',
                                0 * np.load('Data/' + filenames[2][5:15] + ' reading ' + str(reading) + '.npy'))
                        print(filenames[2][5:15] + ' reading ' + str(reading) + '.npy nullified')
                    except FileNotFoundError:
                        Err = True
                    if Err:
                        break
            else:
                ax[1].axvspan(xclick, len(p), color='red', alpha=.4)
                ax[1].plot(np.arange(xclick, len(p)), p[np.arange(xclick, len(p))], linewidth=2, color='black')
                plt.draw()
                ar = np.load(filenames[3])
                for point in range(xclick - len(np.load(filenames[2])), len(ar)):
                    ar[point] = 0
                np.save(filenames[3], ar)
                print(filenames[3][5::] + ' partically nullified')
                for reading in range(extractlastnumber(filenames[3]) + 1, sys.maxsize):
                    Err = False
                    try:
                        np.save('Data/' + filenames[3][5:15] + ' reading ' + str(reading) + '.npy',
                                0 * np.load('Data/' + filenames[3][5:15] + ' reading ' + str(reading) + '.npy'))
                        print(filenames[3][5:15] + ' reading ' + str(reading) + '.npy nullified')
                    except FileNotFoundError:
                        Err = True
                    if Err:
                        break


filenames = lowmean()
Deltadate = datetime.timedelta(days=1)
DaysToDelete1 = pd.date_range(DateStart, datetime.datetime.strptime(filenames[0][5:15], '%Y.%m.%d').date() -
                              Deltadate).strftime('%Y.%m.%d').tolist()
DaysToDelete2 = pd.date_range(datetime.datetime.strptime(filenames[3][5:15], '%Y.%m.%d').date() + Deltadate,
                              DateEnd).strftime('%Y.%m.%d').tolist()
with open('Data/log1.txt', 'w') as file:
    file.write('')
with open('Data/log2.txt', 'w') as file:
    file.write('')

for i in range(len(DaysToDelete1)):
    print('Day ' + DaysToDelete1[i] + ' nullified')
    with open('Data/log1.txt', 'a') as file:
        file.write(DaysToDelete1[i] + '\n')
    Error = False
    for j in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + DaysToDelete1[i] + ' reading ' + str(j) + '.npy')
            np.save('Data/' + DaysToDelete1[i] + ' reading ' + str(j) + '.npy', arr * 0)
        except FileNotFoundError:
            Error = True
        if Error:
            break
for i in range(len(DaysToDelete2)):
    print('Day ' + DaysToDelete2[i] + ' nullified')
    with open('Data/log2.txt', 'a') as file:
        file.write(DaysToDelete2[i] + '\n')
    Error = False
    for j in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + DaysToDelete2[i] + ' reading ' + str(j) + '.npy')
            np.save('Data/' + DaysToDelete2[i] + ' reading ' + str(j) + '.npy', arr * 0)
        except FileNotFoundError:
            Error = True
        if Error:
            break

y = np.append(np.load(filenames[0]), np.load(filenames[1]))
p = np.append(np.load(filenames[2]), np.load(filenames[3]))
fig, ax = plt.subplots(2, num='Press left mouse button to delete')
for i in range(2):
    ax[i].tick_params(labelsize=15)
    ax[i].set_ylabel('Pressure, [MPa]', fontsize=15)
    ax[i].grid(axis="y")
ax[1].set_xlabel('Point number', fontsize=15)
ax[0].plot(np.arange(len(y)), y, linewidth=2, color='b')
ax[1].plot(np.arange(len(p)), p, linewidth=2, color='b')
ax[0].axvline(x=len(np.load(filenames[0])), color='black', linewidth=1)
ax[1].axvline(x=len(np.load(filenames[2])), color='black', linewidth=1)
ax[0].set_title(filenames[0][5::] + ' | ' + filenames[1][5::])
ax[1].set_title(filenames[2][5::] + ' | ' + filenames[3][5::])
fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
