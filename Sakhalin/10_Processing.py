"""Выделение индивидуальных волн, их высот и амплитуд, и нескольких параметров для каждой записи."""
import numpy as np
from functions import individualwaves, DateStart, DateEnd, newdates, kh_solver
import sys
import datetime
from tqdm import tqdm

dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Progress: ", colour='green')
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
        pbar.update(1)
        ds += Deltadate
    np.save('Data/DF_PA', df_pa)
    np.save('Data/DF_NA', df_na)
    np.save('Data/DF_L', df_l)
    np.save('Data/MeanHs', np.mean(arrHs))
    np.save('Data/MeanAs', np.mean(arrHs) / 2)
    np.save('Data/All_kh', arrkh)
    np.save('Data/All_e', arreps)
    np.save('Data/All_a', arra)
    np.save('Data/All_Ur', arrUr)
