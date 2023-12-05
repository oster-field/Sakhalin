"""Выделение тренда (аппроксимация по точкам полиномом заданной степени), поиск троек значений с малой
дисперсией, из этих троек выделяются значения с небольшой производной. Таким образом находятся промежутки,
когда Hs и Tz не сильно меняются, колеблются с малой дисперсией вокруг локального горизонтального среднего,
тем самым являются индикаторами квазистационарности (в силу выбранных параметров) самих записей волнения."""
import numpy as np
import matplotlib.pyplot as plt
from functions import DateStart, DateEnd, newdates, decompose
import sys
import datetime
from tqdm import tqdm

print('Hs / Tz?')
o = input()
dates, ds, de = newdates(DateStart, DateEnd)
pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
Deltadate = datetime.timedelta(days=1)
p = np.arange(0)

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            if o == 'Hs':
                p = np.append(p, np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy'))
            if o == 'Tz':
                p = np.append(p, np.load('Data/' + filename + ' reading ' + str(i) + ' Tz.npy'))
        except FileNotFoundError:
            Error = True
        if Error:
            break
    pbar.update(1)
    ds += Deltadate

p /= np.max(p)
xp = np.arange(len(p)) / 3
trendp = decompose(p, 50, 3, 0)[0]
dtrendp = decompose(p, 50, 3, 1)[0]

fig = plt.figure(num=f'{o} fluctuations')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(xp, p, linewidth=2, color='#006E4A', marker='.', label=f'{o}', alpha=.5)
ax.plot(xp, trendp, linewidth=2, color='black', label=f'{o} trend')
ax.scatter(np.array([]), np.array([]), color='#A62000', label='Small variance', s=100)
ax.scatter(np.array([]), np.array([]), color='#FFAA00', label='Small derivative', s=100)
for i in range(2, len(p), 2):
    if np.var(np.array([p[i], p[i-1], p[i-2]])) < 5e-6:
        ax.plot(np.array([xp[i], xp[i-1], xp[i-2]]), np.array([p[i], p[i-1], p[i-2]]),
                linewidth=2, color='#A62000', marker='.', alpha=.5)
        if np.abs(dtrendp[i-2]) < 0.01 / 15 and np.abs(dtrendp[i-1]) < 0.01 / 15 and np.abs(dtrendp[i]) < 0.01 / 15:
            ax.plot(np.array([xp[i], xp[i - 1], xp[i - 2]]), np.array([p[i], p[i - 1], p[i - 2]]),
                    linewidth=2, color='#FFAA00', marker='.', alpha=.5)
ax.set_xlabel('Hours', fontsize=20)
ax.set_ylabel(f'{o} / max({o})', fontsize=20)
ax.grid()
plt.legend(fontsize=16)
plt.show()
