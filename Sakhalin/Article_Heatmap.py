"""Тепловая карта, размещающая записи в несколько прямоугольников опциональной длины по параметрам по всем
параметрам. Программа нужна для определения границ множеств, на которые бъется пространство параметров."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functions import pointInRect
import datetime
from tqdm import tqdm
import sys
import pandas as pd

o = 'goda'
m = 30  # Число строк
r = 40  # Число столбцов
x = np.arange(0)
a = np.arange(0)
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
                    a = np.append(a, np.load(f'Data{n}/{filename} reading {str(i)} a.npy'))
                    x = np.append(x, np.load(f'Data{n}/{filename} reading {str(i)} {str(o)}.npy'))
                except FileNotFoundError:
                    Error = True
                if Error:
                    break
            ds += Deltadate
    except FileNotFoundError:
        break

x = 2 * x  # recalculating Qp
z = np.zeros((m, r), dtype=int)
h = np.max(a) / m
w = np.max(x) / r
p = len(a)
x0, y0 = 0, 0
for i in range(m):
    x0 = 0
    for j in range(r):
        for q in range(p):
            if pointInRect(x0, y0, w, h, x[q], a[q]):
                z[i, j] += 1
        x0 += w
    y0 += h

text = np.asarray(z)
borders = np.array([921, 458, 65, 15])
colors = ['#4065FF', '#00C7FF',  '#006E33', '#A3FF40']
labels = ['12 часов', 'день', 'неделю', 'месяц']
z = np.flipud(np.asarray(z))
fig = plt.figure(num=f'Heatmap for {o}')
ax = fig.add_subplot(111)
x0, y0 = 0, 0
xlabs = np.arange(w/2, np.max(x), w)
ylabs = np.arange(h/2, np.max(a), h)
for i in range(len(ylabs)):
    x0 = 0
    for j in range(len(xlabs)):
        if text[i, j] > (1 / 2) * np.max(text):
            color = 'black'
        else:
            color = 'white'
        for _ in reversed(range(len(borders))):
            if i != 0 and j != 0 and i != len(ylabs) - 1 and j != len(xlabs) - 1:
                if text[i, j] >= borders[_] > text[i + 1, j]:
                    ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                if text[i, j] >= borders[_] > text[i - 1, j]:
                    ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                if text[i, j] >= borders[_] > text[i, j - 1]:
                    ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
                if text[i, j] >= borders[_] > text[i, j + 1]:
                    ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
            else:
                if i == 0 and j == 0:
                    if text[i, j] >= borders[_] > text[i + 1, j]:
                        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j + 1]:
                        ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
                if i == 0 and j == len(xlabs) - 1:
                    if text[i, j] >= borders[_] > text[i + 1, j]:
                        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j - 1]:
                        ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
                if i == len(ylabs) - 1 and j == 0:
                    if text[i, j] >= borders[_] > text[i - 1, j]:
                        ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j + 1]:
                        ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
                if i == len(ylabs) - 1 and j == len(xlabs) - 1:
                    if text[i, j] >= borders[_] > text[i - 1, j]:
                        ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j - 1]:
                        ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
                if i == 0 and j != 0 and j != len(xlabs) - 1:
                    if text[i, j] >= borders[_] > text[i + 1, j]:
                        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j - 1]:
                        ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j + 1]:
                        ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
                if j == 0 and i != 0 and i != len(ylabs) - 1:
                    if text[i, j] >= borders[_] > text[i + 1, j]:
                        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i - 1, j]:
                        ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j + 1]:
                        ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
                if i == len(ylabs) - 1 and j != 0 and j != len(xlabs) - 1:
                    if text[i, j] >= borders[_] > text[i - 1, j]:
                        ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j - 1]:
                        ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j + 1]:
                        ax.plot([x0 + w, x0 + w], [y0, y0 + h], color=colors[_], linewidth=3)
                if j == len(xlabs) - 1 and i != 0 and i != len(ylabs) - 1:
                    if text[i, j] >= borders[_] > text[i + 1, j]:
                        ax.plot([x0, x0 + w], [y0 + h, y0 + h], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i - 1, j]:
                        ax.plot([x0, x0 + w], [y0, y0], color=colors[_], linewidth=3)
                    if text[i, j] >= borders[_] > text[i, j - 1]:
                        ax.plot([x0, x0], [y0, y0 + h], color=colors[_], linewidth=3)
        ax.annotate(str(text[i, j]), xy=(xlabs[j], ylabs[i]), ha='center', va='center', color=color, size=12)
        x0 += w
    y0 += h
for _ in range(len(borders)):
    ax.plot([], color=colors[_], linewidth=3, label=f'Раз в {labels[_]}')
ax.set_xticks(xlabs, labels=np.round(xlabs, 2), rotation=35)
ax.set_yticks(ylabs, labels=np.round(ylabs, 2))
img = ax.imshow(z, extent=[0, np.max(x), 0, np.max(a)], cmap='hot')
plt.tight_layout(h_pad=1)
ratio = 0.5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax.tick_params(labelsize=18.5)
ax.set_xlabel('Qₚ', fontsize=20)
ax.set_ylabel('a', fontsize=20)
ax.legend(fontsize=16, framealpha=0.9, facecolor='#414141', labelcolor='white')
plt.subplots_adjust(left=0, bottom=0.1, right=1, top=0.98, wspace=0.2, hspace=0.2)
plt.show()
