"""Функция распределения для конкретных значений параметров."""
import numpy as np
import matplotlib.pyplot as plt
from functions import DateStart, DateEnd, newdates, distribution_function, split_array
import sys
import datetime
from tqdm import tqdm

print('kh/Tz/a/eps/Ur/width/w0/energy?')
o = input()
dates, ds, de = newdates(DateStart, DateEnd)
pbar = tqdm(total=len(dates), desc="Processing: ", colour='green')
Deltadate = datetime.timedelta(days=1)
all_p = np.sort(np.load('Data/All_' + str(o) + '.npy'))
splitted_all_p = split_array(all_p, np.array([len(all_p) // 4, len(all_p) // 4, len(all_p) // 4,
                                              len(all_p) // 4 + len(all_p) % 4]))
p0 = splitted_all_p[0][0]
p1 = splitted_all_p[1][0]
p2 = splitted_all_p[2][0]
p3 = splitted_all_p[3][0]
p4 = splitted_all_p[3][-1]
hight1 = np.arange(0)
hight2 = np.arange(0)
hight3 = np.arange(0)
hight4 = np.arange(0)

while ds <= de:
    filename = ds.strftime('%Y.%m.%d')
    Error = False
    for i in range(1, sys.maxsize):
        try:
            hight = np.load('Data/' + filename + ' reading ' + str(i) + ' L.npy')
            Hs = np.load('Data/' + filename + ' reading ' + str(i) + ' Hs.npy')
            p = np.load('Data/' + filename + ' reading ' + str(i) + ' ' + str(o) + '.npy')
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

x1, y1 = distribution_function(hight1)
x2, y2 = distribution_function(hight2)
x3, y3 = distribution_function(hight3)
x4, y4 = distribution_function(hight4)
xrg = np.arange(0, np.max(np.array([np.max(x1), np.max(x2), np.max(x3), np.max(x4)])), 0.0001)
yr = np.exp(-2 * xrg * xrg)
c = np.sqrt(np.pi / 8)
p = np.load('Data/MeanHs.npy') / np.load('Data/Average Depth.npy')
yg = np.exp(-1 * (np.pi / (4 + p)) * ((xrg / c) ** (2 / (1 - p * c))))

fig = plt.figure(num=f'Distribution function splitted by {o}')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(x1, y1, linewidth=2, marker='.', alpha=.65,
        color='#412C84', label=f'{o} ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax.plot(x2, y2, linewidth=2, marker='.', alpha=.65,
        color='#269926',
        label=f'{o} ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax.plot(x3, y3, linewidth=2, marker='.', alpha=.65,
        color='#BF3030', label=f'{o} ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax.plot(x4, y4, linewidth=2, marker='.', alpha=.65,
        color='#FF6A00', label=f'{o} ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax.plot(xrg, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution')
ax.plot(xrg, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
ax.set_xlabel('H/Hs', fontsize=20)
ax.set_ylabel('F(H)', fontsize=20)
ax.set_xlim(left=0)
ax.set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])) - 10e-7)
ax.grid()
plt.yscale('log')
plt.legend(fontsize=16)
plt.show()
