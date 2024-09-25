"""Функция распределения для конкретных значений параметров для всех данных. Используется multiprocessing."""
import numpy as np
import sys
import datetime
from tqdm import tqdm
import pandas as pd
from multiprocessing import Process


def distribution_function(arr, num, name):
    x = np.sort(arr)
    np.save(f'df(p)_all_{name}_small_a_x{num}', x)


if __name__ == '__main__':
    print('kh/Tz/a/eps/Ur/width/w0/energy?')
    o = input()
    Deltadate = datetime.timedelta(days=1)
    all_p = np.arange(0)
    MeanHs1 = np.arange(0)
    MeanHs2 = np.arange(0)
    MeanHs3 = np.arange(0)
    MeanHs4 = np.arange(0)
    MeanDepth1 = np.arange(0)
    MeanDepth2 = np.arange(0)
    MeanDepth3 = np.arange(0)
    MeanDepth4 = np.arange(0)
    for i in range(1, sys.maxsize):
        try:
            all_p = np.append(all_p, np.load(f'Data{i}/All_{str(o)}.npy'))
        except FileNotFoundError:
            break

    p0 = np.load(f'{o}0.npy')
    p1 = np.load(f'{o}1.npy')
    p2 = np.load(f'{o}2.npy')
    p3 = np.load(f'{o}3.npy')
    p4 = np.load(f'{o}4.npy')

    hight1 = np.arange(0)
    hight2 = np.arange(0)
    hight3 = np.arange(0)
    hight4 = np.arange(0)
    for n in range(1, sys.maxsize):
        try:
            ds = datetime.datetime.strptime((open(f'DataTXT{n}_done/INFO.dat').readlines()[5].strip()),
                                            '%Y.%m.%d %H:%M:%S.%f').date()
            de = datetime.datetime.strptime((open(f'DataTXT{n}_done/INFO.dat').readlines()[7].strip()),
                                            '%Y.%m.%d %H:%M:%S.%f').date()
            dates = pd.date_range(ds, de).strftime('%d.%m').tolist()
            pbar = tqdm(total=len(dates), desc=f'Folder {n}: ', colour='green')
            while ds <= de:
                filename = ds.strftime('%Y.%m.%d')
                Error = False
                for i in range(1, sys.maxsize):
                    try:
                        hight = np.load(f'Data{n}/{filename} reading {str(i)} L.npy')
                        Hs = np.load(f'Data{n}/{filename} reading {str(i)} Hs.npy')
                        p = np.load(f'Data{n}/{filename} reading {str(i)} {str(o)}.npy')
                        a = np.load(f'Data{n}/{filename} reading {str(i)} a.npy')
                        Depth = np.load(f'Data{n}/{filename} reading {str(i)} Depth.npy')
                        if a < 0.04:
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
                pbar.update(1)
                ds += Deltadate
        except FileNotFoundError:
            break

    process1 = Process(target=distribution_function, args=([hight1], 1, o))
    process2 = Process(target=distribution_function, args=([hight2], 2, o))
    process3 = Process(target=distribution_function, args=([hight3], 3, o))
    process4 = Process(target=distribution_function, args=([hight4], 4, o))
    process1.start()
    process2.start()
    process3.start()
    process4.start()
    process1.join()
    process2.join()
    process3.join()
    process4.join()
