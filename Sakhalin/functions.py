"""1) Функция удаляет каждую n-ю точку "times" раз, она нужна для построения
графиков в больших масштабах (дни, месяцы) для ускорения работы;
2)Функция среднего значения между соседними точками;
3)Маркировка точек ряда, соответствующих погружению датчика;
4)Поиск записей, содержащих момент погружения;
5)Получение номера записи;
6)Разбиение массива на подмассивы, размеры которых заданы массивом;
7)Максимум элементов массива с некоторым шагом;
8)Индекс элемента массива, ближайшего к заданному числу;
9)Усреднение соседних элементов массива;
10)Получение характеристик поля;
11)Функция распределения вероятности;
12)Сглаживание краевого эффекта преобразования Фурье;
13)Спектральная ширина;
14)Root mean square;
15)Решатель уравнения x * tgx = const;
16)Выясняет находится ли точка в прямоугольнике;
17)Декомпозиция временного ряда;"""
import numpy as np
import sys
import datetime
from PyAstronomy import pyaC
import pandas as pd
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal.windows import hann
from sympy.solvers import nsolve
from sympy import Symbol, tan
from scipy.signal import savgol_filter

DateStart = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[5].strip()),
                                       '%Y.%m.%d %H:%M:%S.%f').date()
DateEnd = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[7].strip()),
                                     '%Y.%m.%d %H:%M:%S.%f').date()
Sensor_Frequency = int(open('DataTXT/INFO.dat').readlines()[2].strip()[15:17])


def seriesreducer(arr, times, n=2):
    for i in range(0, times):
        arr = arr[::n]
    return arr


def meanvalueplot(series):
    if len(series) % 2 != 0:
        series = np.delete(series, 0)
    a = series[::2]
    b = series[1::2]
    mean = a / b
    xmean = 2 * np.arange(0, len(mean))
    return xmean, mean


def divedetector(series):
    x, seriessplit = meanvalueplot(series)
    a, b = np.split(seriessplit, 2)
    a = np.flip(a)
    for i in a:
        if 0.5 < i < 1.5:
            a = np.delete(a, [0])
        else:
            break
    for i in b:
        if 0.5 < i < 1.5:
            b = np.delete(b, [0])
        else:
            break
    xbegin = np.arange(0, len(series[0:2 * len(a)]))
    ybegin = series[0:2 * len(a)]
    xend = np.arange(len(series) - len(series[0:2 * len(b)]), len(series))
    yend = np.flip(np.flip(series)[0:2 * len(b)])
    return xbegin, ybegin, xend, yend


def lowmean():
    ds = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[5].strip()),
                                    '%Y.%m.%d %H:%M:%S.%f').date()
    de = datetime.datetime.strptime((open('DataTXT/INFO.dat').readlines()[7].strip()),
                                    '%Y.%m.%d %H:%M:%S.%f').date()
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

