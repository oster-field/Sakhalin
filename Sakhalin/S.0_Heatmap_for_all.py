"""Тепловая карта, размещающая записи в несколько прямоугольников опциональной длины по параметрам по всем
параметрам."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functions import pointInRect
import datetime
import sys
import pandas as pd

print('kh/Tz/a/eps/Ur/width/w0/energy?')
o = input()
m = 25  # Число строк
r = 25  # Число столбцов
x = np.arange(0)
Hs = np.arange(0)
Deltadate = datetime.timedelta(days=1)

for n in range(1, sys.maxsize):
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

np.save(f'{o}1', 0.5)
np.save(f'{o}2', 1.03)
np.save(f'{o}3', 2.01)

z = np.zeros((m, r), dtype=int)
h = np.max(Hs) / m
w = np.max(x) / r
p = len(Hs)
x0, y0 = 0, 0
for i in range(m):
    x0 = 0
    for j in range(r):
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
xlabs = np.arange(w/2, np.max(x), w)
ylabs = np.arange(h/2, np.max(Hs), h)
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
