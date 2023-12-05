"""Тепловая карта, размещающая записи в несколько прямоугольников опциональной длины по параметрам."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functions import DateStart, DateEnd, newdates, pointInRect
import datetime
import sys
from tqdm import tqdm

print('kh/Tz/a/e/Ur/WDT/w0/E?')
o = input()
m = 24  # Число строк
n = 21  # Число столбцов
x = np.arange(0)
Hs = np.arange(0)
dates, ds, de = newdates(DateStart, DateEnd)
Deltadate = datetime.timedelta(days=1)
pbar = tqdm(total=len(dates), desc="Progress: ", colour='green')

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            if o == 'kh':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' kh.npy'))
            elif o == 'Tz':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' Tz.npy'))
            elif o == 'a':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' a.npy'))
            elif o == 'e':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' eps.npy'))
            elif o == 'Ur':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' Ur.npy'))
            elif o == 'WDT':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' spectrum width.npy'))
            elif o == 'w0':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' spectrum w0.npy'))
            elif o == 'E':
                x = np.append(x, np.load('Data/' + filename + ' reading ' + str(i) + ' spectrum E.npy'))
            Hs = np.append(Hs, np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy'))
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
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
elif o == 'e':
    ax.set_xlabel('ε', fontsize=20)
elif o == 'Ur':
    ax.set_xlabel('Ur', fontsize=20)
elif o == 'WDT':
    ax.set_xlabel('Spectrum width', fontsize=20)
elif o == 'w0':
    ax.set_xlabel('w0, [rad/sec]', fontsize=20)
elif o == 'E':
    ax.set_xlabel('Energy', fontsize=20)
ax.set_ylabel('Hs, [m]', fontsize=20)
plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.98, wspace=0.2, hspace=0.2)
plt.show()
