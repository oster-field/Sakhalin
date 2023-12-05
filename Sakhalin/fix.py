from functions import DateStart, DateEnd, newdates
import numpy as np
import sys
import datetime
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller

dates, ds, de = newdates(DateStart, DateEnd)
pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
Deltadate = datetime.timedelta(days=1)

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            arr = np.load('Data/' + filename + ' reading ' + str(i) + '.npy')
            arr = np.append(arr, np.load('Data/' + filename + ' reading ' + str(i + 1) + '.npy'))
            arr = np.append(arr, np.load('Data/' + filename + ' reading ' + str(i + 2) + '.npy'))
            result = adfuller(arr)
            if result[0] < result[4]["1%"]:
                print("Нулевая гипотеза отвергнута – Временной ряд стационарен")
            else:
                print("Нулевая гипотеза не отвергнута – Временной ряд не стационарен")
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    ds += Deltadate

