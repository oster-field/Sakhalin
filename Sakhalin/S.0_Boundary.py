import numpy as np
import datetime
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt

o = 'BFI_proper'

x = np.arange(0)
Hs = np.arange(0)
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
                    Hs = np.append(Hs, np.load(f'Data{n}/{filename} reading {str(i)} Hs.npy'))
                    x = np.append(x, np.load(f'Data{n}/{filename} reading {str(i)} {str(o)}.npy'))
                except FileNotFoundError:
                    Error = True
                if Error:
                    break
            ds += Deltadate
    except FileNotFoundError:
        break

np.save(f'{o}0', 0)
np.save(f'{o}4', np.max(x))
# Custom parameters:

np.save(f'{o}1', 0.016)
np.save(f'{o}2', 0.067)
np.save(f'{o}3', 0.2)

fig, ax = plt.subplots(1, 1)
ax.scatter(x, Hs, s=1, alpha=.18, color='r')
ax.tick_params(labelsize=20)
ax.set_ylabel('Hs', fontsize=20)
ax.set_xlabel(f'{o}', fontsize=20)
plt.subplots_adjust(left=0.067, bottom=0.088, right=0.97, top=0.974, wspace=0.2, hspace=0.2)
plt.show()
