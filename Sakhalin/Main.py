import numpy as np
import datetime
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal.windows import hann
from matplotlib.patches import Rectangle
from scipy.fftpack import fft, ifft, fftfreq, rfft, rfftfreq
from sympy import *
from PyAstronomy import pyaC
from scipy.signal import savgol_filter
from tkinter import *
from tkinter.ttk import Checkbutton, Combobox, Progressbar
from tkinter.messagebox import showinfo
import re

DateStart = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[5].strip()),
                                       '%Y.%m.%d %H:%M:%S.%f').date()
DateEnd = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[7].strip()),
                                     '%Y.%m.%d %H:%M:%S.%f').date()
Sensor_Frequency = int(open('DataTXT/INFO.dat').readlines()[2].strip()[15:17])


def seriesreducer(arr, times, n=2):
    for i in range(0, times):
        arr = arr[::n]
    return arr


def divedetector(series):
    a = series[0:len(series) // 2]
    b = series[len(series) // 2:-1]
    variances_a = []
    for i in range(len(a) - 1):
        variances_a.append(np.var(a[i:i + 2]))
    if np.max(variances_a) > 15 * np.std(variances_a):
        xbegin = np.arange(np.argmax(variances_a) + 50)
        ybegin = series[xbegin]
    else:
        xbegin = 0
        ybegin = 0
    variances_b = []
    for i in range(len(b) - 1):
        variances_b.append(np.var(b[i:i + 2]))
    if np.max(variances_b) > 15 * np.std(variances_b):
        xend = np.arange(np.argmax(variances_b) - 50 + len(a), len(series))
        yend = series[xend]
    else:
        xend = 0
        yend = 0
    return xbegin, ybegin, xend, yend


def lowmean(ds=DateStart, de=DateEnd):
    dt = datetime.timedelta(days=1)
    mean = np.arange(0)
    readings = []
    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = seriesreducer(np.load('Data/' + filename + ' reading ' + str(i) + '.npy'), 5)
                mean = np.append(mean, np.mean(arr))
                readings.append('Data/' + filename + ' reading ' + str(i) + '.npy')
            except FileNotFoundError:
                Error = True
            if Error:
                break
        ds += dt
    res = []
    for i in range(len(readings)-1):
        if mean[i] < np.sum(mean) / len(readings) < mean[i + 1] or mean[i+1] < np.sum(mean) / len(readings) < mean[i]:
            res.append(readings[i])
            res.append(readings[i+1])
    return res


def extractlastnumber(string):
    last_space_index = string.rfind(" ")
    if last_space_index != -1:
        substring = string[last_space_index + 1:]
        numbers = "".join([c for c in substring if c.isdigit()])
        if numbers:
            return int(numbers)
    return None


def newdates(ds, de):
    dates = pd.date_range(ds, de).strftime('%d.%m').tolist()
    deleteddates = []
    for i in range(1, 3):
        for line in open('Data/log' + str(i) + '.txt').readlines():
            day = line.strip()[8::]
            month = line.strip()[5:7]
            deleteddates.append(day + '.' + month)
    dates = [date for date in dates if date not in deleteddates]
    Deltadate = datetime.timedelta(days=1)
    ds += Deltadate * len(
        open('Data/log1.txt').readlines())  # Даты начала и конца записи волнения (без погружений)
    de -= Deltadate * len(open('Data/log2.txt').readlines())
    return dates, ds, de


def split_array(array, sizes):
    result = []
    current_index = 0
    for size in sizes:
        sub_array = array[current_index:current_index + size]
        result.append(sub_array)
        current_index += size
    return result


def find_max_values(array, N):
    max_values = []
    max_indices = []
    for i in range(0, len(array), N):
        sub_array = array[i:i + N]
        max_value = np.max(sub_array)
        max_index = np.argmax(sub_array) + i
        max_values.append(max_value)
        max_indices.append(max_index)
    return max_values, max_indices


def find_closest_index(arr, target):
    closest_index = 0
    closest_difference = abs(arr[0] - target)
    for i in range(1, len(arr)):
        difference = abs(arr[i] - target)
        if difference < closest_difference:
            closest_difference = difference
            closest_index = i
    return closest_index


def average_neighbours(arr):
    if len(arr) % 2 != 0:
        arr = arr[:-1]
    arr = (arr[:-1] + arr[1:]) / 2
    return arr


def individualwaves(y):
    sigma = np.sqrt(np.var(y))
    Hs = 4 * sigma
    As = 2 * sigma
    x = np.arange(len(y))
    xc, xi = pyaC.zerocross1d(x, y, getIndices=True)
    Tz = ((len(y)/Sensor_Frequency) / len(xc)) * 2
    xnew = np.sort(np.append(x, xc))
    for i in range(1, len(xnew + 1)):
        if xnew[i] in xc:
            xzm1 = np.where(xnew == xnew[i - 1])[0]
            yzm1 = np.where(y == y[xzm1])[0]
            y = np.insert(y, yzm1 + 1, [0])
    q = np.arange(0)
    ymax = np.arange(0)
    ymin = np.arange(0)
    for j in y:
        if j == 0:
            if q[len(q) - 1] > 0:
                ymax = np.append(ymax, np.max(q))
            else:
                ymin = np.append(ymin, np.min(q))
            q = np.arange(0)
        q = np.append(q, j)
    wavelenght = np.arange(0)
    if len(ymax) >= len(ymin):
        for i in range(len(ymin)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    else:
        for i in range(len(ymax)):
            wavelenght = np.append(wavelenght, [ymax[i] + abs(ymin[i])])
    return As, Hs, Tz, ymax, ymin, wavelenght


def distribution_function(arr):
    y = np.arange(0)
    for i in range(0, len(arr)):
        y = np.append(y, 1 - i / len(arr))
    x = np.sort(arr)
    return x, y


def edge_effect(arr):
    s = fft(arr * hann(len(arr)))
    x = fftfreq(len(arr), Sensor_Frequency / (2 * np.pi))
    for freq in range(len(x)):
        if abs(x[freq]) < np.pi / 150:
            s[freq] = 0 + 0j
    arr = ifft(s).real
    return arr/np.hamming(len(arr))


def spectrum_width(w, s):
    dx = 1 / len(w)
    m0 = np.trapz(s, dx=dx)
    m1 = np.trapz(w * s, dx=dx)
    w0 = m1 / m0
    m2 = np.trapz(((w - w0) ** 2) * s, dx=dx)
    return np.sqrt(m2 / ((w0 ** 2) * m0))


def rmsValue(arr):
    square = 0
    for element in arr:
        square += (element ** 2)
    mean = (square / len(arr))
    rms = np.sqrt(mean)
    return rms


def kh_solver(h, Tz):
    x = Symbol('x')
    equation = x * tan(x) - (4 * (np.pi ** 2) * h) / (9.81 * (Tz ** 2))
    kh = nsolve(equation, x, (0, np.pi / 2), solver='bisect')
    return float(kh)


def pointInRect(x0, y0, w, h, x, y):
    x2, y2 = x0+w, y0+h
    if x0 < x < x2:
        if y0 < y < y2:
            return True
    return False


def decompose(data, window_size, degree, deriv):
    trend = savgol_filter(data, window_size, degree, deriv=deriv)
    seasonality = data - trend
    noise = data - seasonality
    return trend, seasonality, noise


def data_from_txt(DateStart=DateStart):
    if not os.path.isdir("Data"):
        os.mkdir("Data")
    Pressure = np.arange(0)
    ReadingsPerFile = Sensor_Frequency * 1200  # Сколько точек будет в файле .npy (Для 8 Гц 9600 точек -  20 мин.)
    Deltadate = datetime.timedelta(days=1)

    while DateStart <= DateEnd:
        counter = 0
        filename = 'DataTXT/' + re.findall(r'\d+', open('DataTXT/INFO.dat').readlines()[1].strip())[0] + \
                   '_Press_meters_' + DateStart.strftime('%Y.%m.%d') + '.dat'
        num_lines = len(open(filename).readlines())
        with open(filename, 'r') as file:
            for line in file:
                Pressure = np.append(Pressure, float(line.strip().replace(',', '.')))
                if (len(Pressure) == ReadingsPerFile) or (len(Pressure) == num_lines % ReadingsPerFile) and (
                        counter * ReadingsPerFile + num_lines % ReadingsPerFile == num_lines):
                    np.save('Data/' + DateStart.strftime('%Y.%m.%d') + ' reading ' + str(counter + 1),
                            Pressure.astype(float))
                    Pressure = np.arange(0)
                    counter += 1
        pb1.step(1)
        DateStart += Deltadate

    with open('Data/isconverted.txt', 'w') as file:
        file.write('Not converted')
    with open('Data/istransformed.txt', 'w') as file:
        file.write('Not transformed')
    with open('Data/isprocessed.txt', 'w') as file:
        file.write('Not processed')
    with open('Data/dba.txt', 'w') as file:
        file.write('Not calculated')
    with open('Data/dbl.txt', 'w') as file:
        file.write('Not calculated')


def pressure_plotting(Times, DateStart=DateStart):
    Deltadate = datetime.timedelta(days=1)
    dates = pd.date_range(DateStart, DateEnd).strftime('%d.%m').tolist()
    y = np.arange(0)
    ticks = np.arange(0)
    c = 0
    np.save('Data/MeasurmentError', np.load('Data/' + DateStart.strftime('%Y.%m.%d') + ' reading 1.npy')[0])

    while DateStart <= DateEnd:
        filename = DateStart.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = seriesreducer(np.load('Data/' + filename + ' reading ' + str(i) + '.npy'), Times)
                y = np.append(y, arr)
            except FileNotFoundError:
                Error = True
            if Error:
                break
        ticks = np.append(ticks, c + round((len(y) - c) / 2))
        c = len(y)
        DateStart += Deltadate

    fig = plt.figure(num='Pressure Plot')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.plot(np.arange(len(y)), y, linewidth=2, color='#2D9B27', label='Pressure fluctuations')
    ax.set_xlabel('Days', fontsize=20)
    ax.set_ylabel('Pressure, [MPa]', fontsize=20)
    ax.set_xticks(ticks[::3])
    ax.set_xticklabels(dates[::3], rotation=30)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(axis="y")
    xbegin, ybegin, xend, yend = divedetector(y)
    ax.plot(xbegin, ybegin, linewidth=2.3, color='#BE2F33', label='Sensor immersion')
    ax.plot(xend, yend, linewidth=2.3, color='#BE2F33')
    plt.legend(fontsize=16)
    plt.show()


def manual_remove(DateStart=DateStart):
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
                                0 * np.load('Data/' + filenames[0][5:15] + ' reading ' + str(reading) + '.npy'))
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
                                0 * np.load('Data/' + filenames[1][5:15] + ' reading ' + str(reading) + '.npy'))
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


def conversion(Times, DateStart=DateStart):
    isconverted = open('Data/isconverted.txt').readlines()[0].strip()
    MeasurmentError = np.load('Data/MeasurmentError.npy')
    dates, ds, de = newdates(DateStart, DateEnd)
    Deltadate = datetime.timedelta(days=1)
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


def fourier_transforms(Times, TMax, DateStart=DateStart):
    dates, ds, de = newdates(DateStart, DateEnd)
    Deltadate = datetime.timedelta(days=1)
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
        ax.plot((len(y) / len(np.load('Data/ConvertedPlot.npy'))) * np.arange(len(np.load('Data/ConvertedPlot.npy'))),
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


def lowrms_spikes_emptyfiles_spline(minRMSvalue, interpolationrate, spikes, interpolation, DateStart=DateStart):
    dates = pd.date_range(DateStart, DateEnd).strftime('%d.%m').tolist()
    Deltadate = datetime.timedelta(days=1)
    Fl = False

    while DateStart <= DateEnd:
        filename = DateStart.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
                if len(arr) == 0 or np.mean(arr) == 0 or rmsValue(arr) < minRMSvalue:
                    os.remove('Data/' + filename + ' reading ' + str(i) + '.npy')
                else:
                    if spikes == 'Y':
                        for j in range(len(arr) - 1):
                            if np.abs(arr[j + 1] - arr[j]) > 6 * np.sqrt(np.var(arr)):
                                arr[j + 1] = (arr[j] + arr[j + 2]) / 2
                                Fl = True
                        if Fl:
                            np.save('Data/' + filename + ' reading ' + str(i), arr)
                            Fl = False
                    if interpolation == 'Y':
                        cs = CubicSpline(np.arange(len(arr)), arr)
                        xs = np.arange(0, len(arr), 1 / interpolationrate)
                        np.save('Data/' + filename + ' reading ' + str(i), cs(xs))
            except FileNotFoundError:
                Error = True
            if Error:
                break
        DateStart += Deltadate


def spectrum_plotting(Times=0):
    s = np.load('Data/SpectrumY.npy')
    w = np.load('Data/SpectrumX.npy')
    samples = np.load('Data/Samples.npy')
    s = seriesreducer(s, Times)
    w = seriesreducer(w, Times)
    s = np.sqrt((2 * np.abs(s.real)) / samples)
    positivepart = 0
    for freq in w:
        if freq >= 0:
            positivepart += 1
    positivew = w[0:positivepart]
    positives = s[0:positivepart]
    envelope, ind = find_max_values(positives, 10 * 2 ** (11 - Times))
    wenvelope = w[ind]
    np.save('Data/EnvelopeY', envelope)
    np.save('Data/EnvelopeX', wenvelope)

    fig = plt.figure(num='Wave spectrum approximation')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.plot(positivew, positives, linewidth=2, color='#007241', label='Wave spectrum approximation')
    ax.plot(wenvelope, envelope, linewidth=2, color='#A62F00', label='Envelope')
    ax.set_xlabel('ω, [rad/sec]', fontsize=20)
    ax.set_ylabel('A(ω), [m]', fontsize=20)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set(xlim=[0, np.max(w)])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    plt.legend(fontsize=16)
    plt.show()


def window_ft(WindowSize, DeltaWindow, part, o):
    if o == 'Log':
        part *= 2
    y = np.load('Data/FullRec.npy')
    window = WindowSize * 60 * Sensor_Frequency
    n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))
    w = rfftfreq(window, 1 / Sensor_Frequency)
    z = []

    for i in range(0, n):
        arr = y[i * DeltaWindow:window + i * DeltaWindow]
        mask = hann(len(arr))
        s = np.abs(rfft(arr * mask))
        s = np.sqrt((2 * s[0:int(len(w) * .01 * part)]) / len(arr))
        z.append(np.flip(s))
    z = np.asarray(z)

    fig = plt.figure(num=f'Window FT {o} scale')
    ax = fig.add_subplot(111)
    if o == 'Log':
        img = ax.imshow(np.flip(np.flip(np.log10(z)).T),
                        extent=[0, WindowSize / 60 + n * DeltaWindow / 3600, 0, w[int(len(w) * .01 * part)]],
                        cmap='rainbow')
        colorbar = plt.colorbar(img, format='10e%1.0f m', shrink=0.75)
        plt.tight_layout(h_pad=1)
    elif o == 'Lin':
        img = ax.imshow(np.flip(np.flip(z).T),
                        extent=[0, WindowSize / 60 + n * DeltaWindow / 3600, 0, w[int(len(w) * .01 * part)]],
                        cmap='PuOr')
        colorbar = plt.colorbar(img, format='%1.2f m', shrink=0.75)
        plt.tight_layout(h_pad=1)
    colorbar.ax.tick_params(labelsize=20)
    ratio = 0.5
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('t, [hours]', fontsize=20)
    ax.set_ylabel('f, [Hz]', fontsize=20)
    plt.show()


def specrtum_characteristics(DateStart=DateStart):
    WindowSize = 20
    DeltaWindow = 1200
    width = np.arange(0)
    E = np.arange(0)
    w0 = np.arange(0)
    y = np.load('Data/FullRec.npy')
    window = WindowSize * 60 * Sensor_Frequency
    n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))
    w = rfftfreq(window, 1 / Sensor_Frequency)

    for i in range(0, n):
        arr = y[i * DeltaWindow:window + i * DeltaWindow]
        mask = hann(len(arr))
        s = np.abs(rfft(arr * mask))
        dx = 1 / len(w)
        m0 = np.trapz(s, dx=dx)
        E = np.append(E, m0)
        w0 = np.append(w0, np.trapz(w * s, dx=dx) / m0)
        width = np.append(width, spectrum_width(w, s))

    np.save('Data/All_width', width)
    np.save('Data/All_w0', w0)
    np.save('Data/All_energy', E)

    dates, ds, de = newdates(DateStart, DateEnd)
    Deltadate = datetime.timedelta(days=1)
    c = 0

    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
                np.save('Data/' + filename + ' reading ' + str(i) + ' width', width[c])
                np.save('Data/' + filename + ' reading ' + str(i) + ' w0', w0[c])
                np.save('Data/' + filename + ' reading ' + str(i) + ' energy', E[c])
                c += 1
            except FileNotFoundError:
                Error = True
            if Error:
                break

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


def processing(DateStart=DateStart):
    dates, ds, de = newdates(DateStart, DateEnd)
    Deltadate = datetime.timedelta(days=1)

    isprocessed = open('Data/isprocessed.txt').readlines()[0].strip()
    df_pa = np.arange(0)
    df_na = np.arange(0)
    df_l = np.arange(0)
    arrHs = np.arange(0)
    arrkh = np.arange(0)
    arreps = np.arange(0)
    arra = np.arange(0)
    arrUr = np.arange(0)

    if isprocessed == 'Not processed':
        with open('Data/isprocessed.txt', 'w') as file:
            file.write('Processed')
        while ds <= de:
            filename = ds.strftime('%Y.%m.%d')
            Error = False
            for i in range(1, sys.maxsize):
                try:
                    arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
                    depth = np.load('Data/' + filename + ' reading ' + str(i) + ' Depth.npy')
                    As, Hs, Tz, ymax, ymin, wavelenght = individualwaves(arr)
                    kh = kh_solver(depth, Tz)
                    k = kh / depth
                    eps = (k * Hs) / 4
                    a = As / depth
                    Ur = (3 * k * Hs) / ((2 * k * depth) ** 3)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' As', As)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' Hs', Hs)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' Tz', Tz)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' PA', ymax)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' NA', ymin)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' L', wavelenght)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' kh', kh)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' eps', eps)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' a', a)
                    np.save('Data/' + filename + ' reading ' + str(i) + ' Ur', Ur)
                    df_pa = np.append(df_pa, ymax / As)
                    df_na = np.append(df_na, -1 * ymin / As)
                    df_l = np.append(df_l, wavelenght / Hs)
                    arrHs = np.append(arrHs, Hs)
                    arrkh = np.append(arrkh, kh)
                    arreps = np.append(arreps, eps)
                    arra = np.append(arra, a)
                    arrUr = np.append(arrUr, Ur)
                except FileNotFoundError:
                    Error = True
                if Error:
                    break

            ds += Deltadate
        np.save('Data/DF_PA', df_pa)
        np.save('Data/DF_NA', df_na)
        np.save('Data/DF_L', df_l)
        np.save('Data/MeanHs', np.mean(arrHs))
        np.save('Data/MeanAs', np.mean(arrHs) / 2)
        np.save('Data/All_kh', arrkh)
        np.save('Data/All_eps', arreps)
        np.save('Data/All_a', arra)
        np.save('Data/All_Ur', arrUr)


def amplitudes_distribution(o):
    iscalculated = open('Data/dba.txt').readlines()[0].strip()
    if iscalculated == 'Not calculated':
        xp, yp = distribution_function(np.load('Data/DF_PA.npy'))
        xn, yn = distribution_function(np.load('Data/DF_NA.npy'))
        np.save('Data/DF_PA_X', xp)
        np.save('Data/DF_PA_Y', yp)
        np.save('Data/DF_NA_X', xn)
        np.save('Data/DF_NA_Y', yn)
        with open('Data/dba.txt', 'w') as file:
            file.write('Calculated')
    else:
        xp = np.load('Data/DF_PA_X.npy')
        yp = np.load('Data/DF_PA_Y.npy')
        xn = np.load('Data/DF_NA_X.npy')
        yn = np.load('Data/DF_NA_Y.npy')

    x = np.arange(0, np.max(np.array([np.max(xp), np.max(xn)])), 0.0001)
    yr = np.exp(-2 * x * x)
    c = np.sqrt(np.pi / 8)
    p = np.load('Data/MeanAs.npy') / np.load('Data/Average Depth.npy')
    yg = np.exp(-1 * (np.pi / (4 + p)) * ((x / c) ** (2 / (1 - p * c))))

    fig = plt.figure(num=f'Amplitudes distribution {o} scale')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)

    if o == 'Log':
        ax.plot(xn, yn, linewidth=2, marker='.', alpha=.8, color='#2D9B27', label='Negative amplitudes')
        ax.plot(xp, yp, linewidth=2, marker='.', alpha=.8, color='#BE2F33', label='Positive amplitudes')
        ax.plot(x, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution', alpha=.8)
        ax.plot(x, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
        ax.set_xlim(left=0)
        ax.set_ylim(top=1, bottom=np.min(np.array([np.min(yp), np.min(yn)])) - 10e-7)
        plt.yscale('log')
    elif o == 'Lin':
        ax.plot(xn, yn, linewidth=8, color='#2D9B27', label='Negative amplitudes')
        ax.plot(xp, yp, linewidth=6, color='#BE2F33', label='Positive amplitudes')
        ax.plot(x, yr, linewidth=3, color='black', linestyle='--', label='Rayleigh distribution', alpha=.8)
        ax.plot(x, yg, linewidth=4, color='black', label='Glukhovskiy distribution')
        ax.set_xlim(left=0, right=2)
        ax.set_ylim(top=1, bottom=0)
    ax.set_xlabel('A/As', fontsize=20)
    ax.set_ylabel('F(A)', fontsize=20)
    ax.grid()
    plt.legend(fontsize=16)
    plt.show()


def height_distribution(o):
    iscalculated = open('Data/dbl.txt').readlines()[0].strip()
    if iscalculated == 'Not calculated':
        x, y = distribution_function(np.load('Data/DF_L.npy'))
        np.save('Data/DF_L_X', x)
        np.save('Data/DF_L_Y', y)
        with open('Data/dbl.txt', 'w') as file:
            file.write('Calculated')
    else:
        x = np.load('Data/DF_L_X.npy')
        y = np.load('Data/DF_L_Y.npy')

    xrg = np.arange(0, np.max(x), 0.0001)
    yr = np.exp(-2 * xrg * xrg)
    c = np.sqrt(np.pi / 8)
    p = np.load('Data/MeanHs.npy') / np.load('Data/Average Depth.npy')
    yg = np.exp(-1 * (np.pi / (4 + p)) * ((xrg / c) ** (2 / (1 - p * c))))

    fig = plt.figure(num=f'Height distribution {o} scale')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)

    if o == 'Log':
        ax.plot(x, y, linewidth=2, marker='.', alpha=.65, color='#06276F', label='Heights')
        ax.plot(xrg, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution')
        ax.plot(xrg, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
        ax.set_xlim(left=0)
        ax.set_ylim(top=1, bottom=np.min(y) - 10e-7)
        plt.yscale('log')
    elif o == 'Lin':
        ax.plot(x, y, linewidth=6, alpha=.8, color='#06276F', label='Heights')
        ax.plot(xrg, yr, linewidth=4, color='black', linestyle='--', label='Rayleigh distribution')
        ax.plot(xrg, yg, linewidth=4, color='black', label='Glukhovskiy distribution')
        ax.set_xlim(left=0, right=2)
        ax.set_ylim(top=1, bottom=0)
    ax.set_xlabel('H/Hs', fontsize=20)
    ax.set_ylabel('F(H)', fontsize=20)
    ax.grid()
    plt.legend(fontsize=16)
    plt.show()


def heatmap(o, m, n, DateStart=DateStart):
    x = np.arange(0)
    Hs = np.arange(0)
    dates, ds, de = newdates(DateStart, DateEnd)
    Deltadate = datetime.timedelta(days=1)


    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' ' + str(o) + '.npy'))
                Hs = np.append(Hs, np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy'))
            except FileNotFoundError:
                Error = True
            if Error:
                break

        ds += Deltadate

    z = np.zeros((m, n), dtype=int)
    h = np.max(Hs) / m
    w = np.max(x) / n
    p = len(Hs)
    x0, y0 = 0, 0
    for i in range(m):
        x0 = 0
        for j in range(n):
            for q in range(p):
                if pointInRect(x0, y0, w, h, x[q], Hs[q]):
                    z[i, j] += 1
            x0 += w
        y0 += h

    text = np.asarray(z)
    z = np.flipud(np.asarray(z))
    fig = plt.figure(num=f'Heatmap for {o}')
    ax = fig.add_subplot(111)
    x0, y0 = 0, 0
    xlabs = np.arange(w / 2, np.max(x), w)
    ylabs = np.arange(h / 2, np.max(Hs), h)
    for i in range(len(ylabs)):
        x0 = 0
        for j in range(len(xlabs)):
            if text[i, j] > (3 / 4) * np.max(text):
                color = 'black'
            else:
                color = 'white'
            ax.add_patch(Rectangle((x0, y0), w, h, edgecolor='white', linewidth=1, facecolor='none'))
            ax.annotate(str(text[i, j]), xy=(xlabs[j], ylabs[i]), ha='center', va='center', color=color)
            x0 += w
        y0 += h
    ax.set_xticks(xlabs, labels=np.round(xlabs, 2), rotation=30)
    ax.set_yticks(ylabs, labels=np.round(ylabs, 2))
    img = ax.imshow(z, extent=[0, np.max(x), 0, np.max(Hs)], cmap='bone')
    plt.tight_layout(h_pad=1)
    ratio = 0.6
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    ax.tick_params(labelsize=17)
    if o == 'kh':
        ax.set_xlabel('kh', fontsize=20)
    elif o == 'Tz':
        ax.set_xlabel('Tz, [sec]', fontsize=20)
    elif o == 'a':
        ax.set_xlabel('a', fontsize=20)
    elif o == 'eps':
        ax.set_xlabel('ε', fontsize=20)
    elif o == 'Ur':
        ax.set_xlabel('Ur', fontsize=20)
    elif o == 'width':
        ax.set_xlabel('Spectrum width', fontsize=20)
    elif o == 'w0':
        ax.set_xlabel('w0, [rad/sec]', fontsize=20)
    elif o == 'energy':
        ax.set_xlabel('Energy', fontsize=20)
    ax.set_ylabel('Hs, [m]', fontsize=20)
    plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.98, wspace=0.2, hspace=0.2)
    plt.show()


def distribution_for_set(o, DateStart=DateStart):
    dates, ds, de = newdates(DateStart, DateEnd)

    Deltadate = datetime.timedelta(days=1)
    all_p = np.sort(np.load('Data/All_' + str(o) + '.npy'))
    splitted_all_p = split_array(all_p, np.array([len(all_p) // 4, len(all_p) // 4, len(all_p) // 4,
                                                  len(all_p) // 4 + len(all_p) % 4]))
    p0 = splitted_all_p[0][0]
    p1 = splitted_all_p[1][0]
    p2 = splitted_all_p[2][0]
    p3 = splitted_all_p[3][0]
    p4 = splitted_all_p[3][-1]
    hight1 = np.arange(0)
    hight2 = np.arange(0)
    hight3 = np.arange(0)
    hight4 = np.arange(0)

    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                hight = np.load('Data/' + filename + ' reading ' + str(i) + ' L.npy')
                Hs = np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy')
                p = np.load('Data/' + filename + ' reading ' + str(i) + ' ' + str(o) + '.npy')
                if p0 <= p <= p1:
                    hight1 = np.append(hight1, hight / Hs)
                elif p1 < p <= p2:
                    hight2 = np.append(hight2, hight / Hs)
                elif p2 < p <= p3:
                    hight3 = np.append(hight3, hight / Hs)
                elif p3 < p <= p4:
                    hight4 = np.append(hight4, hight / Hs)
            except FileNotFoundError:
                Error = True
            if Error:
                break

        ds += Deltadate

    x1, y1 = distribution_function(hight1)
    x2, y2 = distribution_function(hight2)
    x3, y3 = distribution_function(hight3)
    x4, y4 = distribution_function(hight4)
    xrg = np.arange(0, np.max(np.array([np.max(x1), np.max(x2), np.max(x3), np.max(x4)])), 0.0001)
    yr = np.exp(-2 * xrg * xrg)
    c = np.sqrt(np.pi / 8)
    p = np.load('Data/MeanHs.npy') / np.load('Data/Average Depth.npy')
    yg = np.exp(-1 * (np.pi / (4 + p)) * ((xrg / c) ** (2 / (1 - p * c))))

    fig = plt.figure(num=f'Distribution function splitted by {o}')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.plot(x1, y1, linewidth=2, marker='.', alpha=.65,
            color='#412C84', label=f'{o} ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
    ax.plot(x2, y2, linewidth=2, marker='.', alpha=.65,
            color='#269926',
            label=f'{o} ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
    ax.plot(x3, y3, linewidth=2, marker='.', alpha=.65,
            color='#BF3030', label=f'{o} ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
    ax.plot(x4, y4, linewidth=2, marker='.', alpha=.65,
            color='#FF6A00', label=f'{o} ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
    ax.plot(xrg, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution')
    ax.plot(xrg, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
    ax.set_xlabel('H/Hs', fontsize=20)
    ax.set_ylabel('F(H)', fontsize=20)
    ax.set_xlim(left=0)
    ax.set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])) - 10e-7)
    ax.grid()
    plt.yscale('log')
    plt.legend(fontsize=16)
    plt.show()


def quasiperiodicity_check(o, DateStart=DateStart):
    dates, ds, de = newdates(DateStart, DateEnd)

    Deltadate = datetime.timedelta(days=1)
    p = np.arange(0)

    while ds <= de:
        filename = ds.strftime('%Y.%m.%d')
        Error = False
        for i in range(1, sys.maxsize):
            try:
                if o == 'Hs':
                    p = np.append(p, np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy'))
                if o == 'Tz':
                    p = np.append(p, np.load('Data/' + filename + ' reading ' + str(i) + ' Tz.npy'))
            except FileNotFoundError:
                Error = True
            if Error:
                break

        ds += Deltadate

    p /= np.max(p)
    xp = np.arange(len(p)) / 3
    trendp = decompose(p, 50, 3, 0)[0]
    dtrendp = decompose(p, 50, 3, 1)[0]

    fig = plt.figure(num=f'{o} fluctuations')
    ax = fig.add_subplot(111)
    ax.tick_params(labelsize=20)
    ax.plot(xp, p, linewidth=2, color='#006E4A', marker='.', label=f'{o}', alpha=.5)
    ax.plot(xp, trendp, linewidth=2, color='black', label=f'{o} trend')
    ax.scatter(np.array([]), np.array([]), color='#A62000', label='Small variance', s=100)
    ax.scatter(np.array([]), np.array([]), color='#FFAA00', label='Small variance and derivative', s=100)
    for i in range(2, len(p), 2):
        if np.var(np.array([p[i], p[i - 1], p[i - 2]])) < 5e-6:
            ax.plot(np.array([xp[i], xp[i - 1], xp[i - 2]]), np.array([p[i], p[i - 1], p[i - 2]]),
                    linewidth=2, color='#A62000', marker='.', alpha=.5)
            if np.abs(dtrendp[i - 2]) < 0.01 / 15 and np.abs(dtrendp[i - 1]) < 0.01 / 15 and np.abs(
                    dtrendp[i]) < 0.01 / 15:
                ax.plot(np.array([xp[i], xp[i - 1], xp[i - 2]]), np.array([p[i], p[i - 1], p[i - 2]]),
                        linewidth=2, color='#FFAA00', marker='.', alpha=.5)
    ax.set_xlabel('Hours', fontsize=20)
    ax.set_ylabel(f'{o} / max({o})', fontsize=20)
    ax.grid()
    plt.legend(fontsize=16)
    plt.show()


def spectrum_approximation():
    def onclick_2(event):
        global degree, x, y, curve, averagecurve
        if event.button == 1 and event.inaxes == ax:
            line = curve.pop(0)
            line.remove()
            poly = np.poly1d(np.polyfit(x, y, degree))
            degree += 5
            curve = ax.plot(x, poly(x), linewidth=2, alpha=.9, color='#320A71')
            ax.set_title('Polynom of ' + str(degree) + ' degree')
            plt.draw()
        elif event.button == 2 and event.inaxes == ax:
            line = averagecurve.pop(0)
            line.remove()
            x = average_neighbours(x)
            y = average_neighbours(y)
            averagecurve = ax.plot(x, y, linewidth=4, alpha=.6, color='#007439')
            plt.draw()
        elif event.button == 3 and event.inaxes == ax:
            imax = find_closest_index(x, event.xdata)
            x = x[:imax:]
            y = y[:imax:]
            ax.set_xlim(left=0, right=np.max(x))
            plt.draw()

    degree = 3
    x = np.load('Data/EnvelopeX.npy')
    y = np.load('Data/EnvelopeY.npy')
    p = np.poly1d(np.polyfit(x, y, degree))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    curve = ax.plot(0, 0)
    ax.tick_params(labelsize=20)
    averagecurve = ax.plot(0, 0)
    ax.plot(x, y, linewidth=6, alpha=.3, color='#A62F00', label='Envelope of spectrum approximation')
    ax.set_xlabel('ω, [rad/sec]', fontsize=20)
    ax.set_ylabel('A(ω), [m]', fontsize=20)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlim(left=0, right=np.max(x))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y")
    plt.legend(fontsize=16)
    fig.canvas.mpl_connect('button_press_event', onclick_2)
    plt.show()


window = Tk()
window.title("Data processing and analysis")
window.geometry('1920x1080')

lbl0 = Label(window, text="  Data processing.  ", font=("Arial Bold", 16)).grid(column=0, row=0)
lblreduce = Label(window, text="  Reduse point numbers for plotting by 2^", font=("Arial Bold", 16)).grid(column=1, row=0)
txtr = Entry(window, width=3, font=("Arial Bold", 10))
txtr.insert(-1, '10')
txtr.grid(column=2, row=0)
lbl1 = Label(window, text="  Step 1: Convert .dat to .npy:  ", font=("Arial Bold", 16)).grid(column=0, row=1)
pb1 = Progressbar(window, maximum=len(pd.date_range(DateStart, DateEnd).strftime('%d.%m').tolist()))
pb1.grid(column=2, row=1)
lbl2 = Label(window, text="  Step 2: Pressure plotting:  ", font=("Arial Bold", 16)).grid(column=0, row=2)
pb2 = Progressbar(window).grid(column=2, row=2)
lbl3 = Label(window, text="  Step 3: Manual remove:  ", font=("Arial Bold", 16)).grid(column=0, row=3)
lbl4 = Label(window, text="  Step 4: Converting and plotting:  ", font=("Arial Bold", 16)).grid(column=0, row=4)
lbl5 = Label(window, text="  Step 5: Removing low frequencies:  ", font=("Arial Bold", 16)).grid(column=0, row=5)
lbltmax = Label(window, text="max period (minutes): ", font=("Arial Bold", 16)).grid(column=2, row=5)
txttmax = Entry(window, width=3, font=("Arial Bold", 10))
txttmax.insert(-1, '10')
txttmax.grid(column=3, row=5)
lbl6 = Label(window, text="  Step 6: Modify folder and data:  ", font=("Arial Bold", 16)).grid(column=0, row=6)
lblinterp = Label(window, text="interpolate: ", font=("Arial Bold", 16)).grid(column=2, row=6)
chk_stateinterp = BooleanVar()
chk_stateinterp.set(False)
chkinterp = Checkbutton(window, var=chk_stateinterp)
chkinterp.grid(column=3, row=6)
lbllowr = Label(window, text="remove spikes: ", font=("Arial Bold", 16)).grid(column=4, row=6)
chk_statelowr = BooleanVar()
chk_statelowr.set(False)
chklowr = Checkbutton(window, var=chk_statelowr)
chklowr.grid(column=5, row=6)
lblinterpr = Label(window, text=" interpolation rate: ", font=("Arial Bold", 16)).grid(column=6, row=6)
txtinterpr = Entry(window, width=3, font=("Arial Bold", 10))
txtinterpr.insert(-1, '8')
txtinterpr.grid(column=7, row=6)
lbllowrv = Label(window, text=" min. RMS: ", font=("Arial Bold", 16)).grid(column=8, row=6)
txtlowrv = Entry(window, width=11, font=("Arial Bold", 10))
txtlowrv.insert(-1, '0.019')
txtlowrv.grid(column=9, row=6)
lbl7 = Label(window, text="  Step 7: Spectum plotting:  ", font=("Arial Bold", 16)).grid(column=0, row=7)
lbl8 = Label(window, text="  Step 8: Window Fourier transform:  ", font=("Arial Bold", 16)).grid(column=0, row=8)
lblws = Label(window, text=" window size (min.): ", font=("Arial Bold", 16))
lblws.grid(column=2, row=8)
txtws = Entry(window, width=3, font=("Arial Bold", 10))
txtws.insert(-1, '10')
txtws.grid(column=3, row=8)
lbldw = Label(window, text=" delta (sec.): ", font=("Arial Bold", 16))
lbldw.grid(column=4, row=8)
txtdw = Entry(window, width=3, font=("Arial Bold", 10))
txtdw.insert(-1, '10')
txtdw.grid(column=5, row=8)
lblscale = Label(window, text=" scale: ", font=("Arial Bold", 16))
lblscale.grid(column=6, row=8)
combo8 = Combobox(window, font=("Arial Bold", 16), width=5)
combo8['values'] = ('Lin', 'Log')
combo8.current(0)
combo8.grid(column=7, row=8)
lblpart = Label(window, text="  part to show (%): ", font=("Arial Bold", 16))
lblpart.grid(column=8, row=8)
txtpart = Entry(window, width=3, font=("Arial Bold", 10))
txtpart.insert(-1, '10')
txtpart.grid(column=9, row=8)
lbl9 = Label(window, text="  Step 9: Spectum characteristics:  ", font=("Arial Bold", 16)).grid(column=0, row=9)
lbl10 = Label(window, text="  Step 10: Individual waves processing:  ", font=("Arial Bold", 16)).grid(column=0, row=10)
lbld = Label(window, text="  Data analysis.  ", font=("Arial Bold", 16))
lbld.grid(column=0, row=11)
lbl11 = Label(window, text="  Step 11: Amplitudes distribution:  ", font=("Arial Bold", 16)).grid(column=0, row=12)
lblscale = Label(window, text=" scale: ", font=("Arial Bold", 16)).grid(column=6, row=12)
combo11 = Combobox(window, font=("Arial Bold", 16), width=5)
combo11['values'] = ('Lin', 'Log')
combo11.current(1)
combo11.grid(column=7, row=12)
lbl12 = Label(window, text="  Step 12: Height distribution:  ", font=("Arial Bold", 16))
lbl12.grid(column=0, row=13)
lblscale = Label(window, text=" scale: ", font=("Arial Bold", 16))
lblscale.grid(column=6, row=13)
combo12 = Combobox(window, font=("Arial Bold", 16), width=5)
combo12['values'] = ('Lin', 'Log')
combo12.current(1)
combo12.grid(column=7, row=13)
lbl13 = Label(window, text="  Step 13: Heatmap MxN:  ", font=("Arial Bold", 16)).grid(column=0, row=14)
lblm = Label(window, text=" M: ", font=("Arial Bold", 16))
lblm.grid(column=2, row=14)
txtm = Entry(window, width=3, font=("Arial Bold", 10))
txtm.insert(-1, '24')
txtm.grid(column=3, row=14)
lbln = Label(window, text=" N: ", font=("Arial Bold", 16))
lbln.grid(column=4, row=14)
txtn = Entry(window, width=3, font=("Arial Bold", 10))
txtn.insert(-1, '21')
txtn.grid(column=5, row=14)
lblparam = Label(window, text=" parameter: ", font=("Arial Bold", 16))
lblparam.grid(column=6, row=14)
combo13 = Combobox(window, font=("Arial Bold", 16), width=5)
combo13['values'] = ('kh', 'Tz', 'a', 'eps', 'Ur', 'width', 'w0', 'energy')
combo13.current(0)
combo13.grid(column=7, row=14)
lbl14 = Label(window, text="  Step 15: Distribution for set:  ", font=("Arial Bold", 16)).grid(column=0, row=15)
lblsparam = Label(window, text=" parameter: ", font=("Arial Bold", 16))
lblsparam.grid(column=6, row=15)
combo14 = Combobox(window, font=("Arial Bold", 16), width=5)
combo14['values'] = ('kh', 'a', 'eps', 'Ur', 'width', 'w0', 'energy')
combo14.current(0)
combo14.grid(column=7, row=15)
lbl15 = Label(window, text="  Step 16: Quasiperiosicity:  ", font=("Arial Bold", 16)).grid(column=0, row=16)
lblqparam = Label(window, text=" parameter: ", font=("Arial Bold", 16)).grid(column=6, row=16)
combo15 = Combobox(window, font=("Arial Bold", 16), width=5)
combo15['values'] = ('Hs', 'Tz')
combo15.current(0)
combo15.grid(column=7, row=16)


def command0():
    fourier_transforms(int(txtr.get()), float(txttmax.get()))
    showinfo(title="Info", message="Transformed, lounch one more time")


def command1():
    if chk_stateinterp.get():
        interpolation = 'Y'
    else:
        interpolation = 'N'
    if chk_statelowr.get():
        spikes = 'Y'
    else:
        spikes = 'N'
    lowrms_spikes_emptyfiles_spline(float(txtlowrv.get()), float(txtinterpr.get()), spikes, interpolation)


btn1 = Button(window, text="Convert", font=("Arial Bold", 14), command=data_from_txt).grid(column=1, row=1)
btn2 = Button(window, text="Plot", font=("Arial Bold", 14),
              command=lambda: pressure_plotting(int(txtr.get()))).grid(column=1, row=2)
btn3 = Button(window, text="Show", font=("Arial Bold", 14), command=manual_remove).grid(column=1, row=3)
btn4 = Button(window, text="Convert and plot", font=("Arial Bold", 14),
              command=lambda: conversion(int(txtr.get()))).grid(column=1, row=4)
btn5 = Button(window, text="Run", font=("Arial Bold", 14), command=command0).grid(column=1, row=5)
btn6 = Button(window, text="Modify", font=("Arial Bold", 14),
              command=command1).grid(column=1, row=6)
btn7 = Button(window, text="Plot", font=("Arial Bold", 14), command=spectrum_plotting).grid(column=1, row=7)
btn8 = Button(window, text="Plot", font=("Arial Bold", 14),
              command=lambda: window_ft(int(txtws.get()), int(txtdw.get()),
                                        float(txtpart.get()), combo8.get())).grid(column=1, row=8)
btn9 = Button(window, text="Plot", font=("Arial Bold", 14), command=specrtum_characteristics).grid(column=1, row=9)
btn10 = Button(window, text="Start", font=("Arial Bold", 14), command=processing).grid(column=1, row=10)
btn11 = Button(window, text="Plot", font=("Arial Bold", 14),
               command=lambda: amplitudes_distribution(combo11.get())).grid(column=1, row=12)
btn12 = Button(window, text="Plot", font=("Arial Bold", 14),
               command=lambda: height_distribution(combo12.get())).grid(column=1, row=13)
btn13 = Button(window, text="Plot", font=("Arial Bold", 14),
               command=lambda: heatmap(combo13.get(), int(txtm.get()), int(txtn.get()))).grid(column=1, row=14)
btn14 = Button(window, text="Plot", font=("Arial Bold", 14),
               command=lambda: distribution_for_set(combo14.get())).grid(column=1, row=15)
btn15 = Button(window, text="Plot", font=("Arial Bold", 14),
               command=lambda: quasiperiodicity_check(combo15.get())).grid(column=1, row=16)

window.mainloop()
