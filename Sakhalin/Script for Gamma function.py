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
                    kh = np.load(f'Data{n}/{filename} reading {str(i)} kh.npy')
                    eps = np.load(f'Data{n}/{filename} reading {str(i)} eps.npy')
                    Q = np.load(f'Data{n}/{filename} reading {str(i)} goda.npy')
                    v = 1 + (2 * kh) / (np.sinh(2 * kh))
                    a = -v**2 + 2 + 8 * (kh**2) * ((np.cosh(2 * kh)) / ((np.sinh(2 * kh))**2))
                    b = (np.cosh(4 * kh) + 8 - 2 * (np.tanh(kh))**2) / (8 * (np.sinh(kh))**4) - ((2 * (np.cosh(kh))**2 + 0.5 * v)**2) / ((np.sinh(2 * kh))**2 * ((kh / (np.tanh(kh))) - (v / 2)**2))
                    if a < 0:
                        print(kh)
                    g = v * np.sqrt(np.abs(b) / a)
                    np.save(f'Data{n}/{filename} reading {str(i)} gamma.npy', g)
                    BFI = np.sqrt(2 * np.pi) * eps * Q * g
                    np.save(f'Data{n}/{filename} reading {str(i)} BFI_proper.npy', BFI)
                    BFI_goda = np.sqrt(2 * np.pi) * eps * Q
                    np.save(f'Data{n}/{filename} reading {str(i)} BFI_goda.npy', BFI_goda)
                except FileNotFoundError:
                    Error = True
                if Error:
                    break
            ds += Deltadate
    except FileNotFoundError:
        break
