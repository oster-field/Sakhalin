"""Тепловая карта, размещающая записи в несколько прямоугольников опциональной длины по параметрам по всем
параметрам. Программа нужна для определения границ множеств, на которые бъется пространство параметров."""
import numpy as np
import datetime
from tqdm import tqdm
import sys
import pandas as pd

Deltadate = datetime.timedelta(days=1)

for n in tqdm(range(1, 14), desc='Processing: ', colour='green'):
    try:
        ds = datetime.datetime.strptime((open(f'DataTXT{n}_done/INFO.dat').readlines()[5].strip()),
                                        '%Y.%m.%d %H:%M:%S.%f').date()
        de = datetime.datetime.strptime((open(f'DataTXT{n}_done/INFO.dat').readlines()[7].strip()),
                                        '%Y.%m.%d %H:%M:%S.%f').date()
        dates = pd.date_range(ds, de).strftime('%d.%m').tolist()
        while ds <= de:
            filename = ds.strftime('%Y.%m.%d')
            Error = False
            for i in range(1, sys.maxsize):
                try:
                    eps = np.load(f'Data{n}/{filename} reading {str(i)} eps.npy')
                    Q = np.load(f'Data{n}/{filename} reading {str(i)} goda.npy')
                    BFI = eps / Q
                    np.save(f'Data{n}/{filename} reading {str(i)} BFI_goda_divide.npy', BFI)
                except FileNotFoundError:
                    Error = True
                if Error:
                    break
            ds += Deltadate
    except FileNotFoundError:
        break
