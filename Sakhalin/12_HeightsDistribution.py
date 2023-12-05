"""Функция распределения высот, нормированные на значительные, первый раз считает довольно долго,
остальные запуски программы по заранее посчитанным данным. Можно менять масштаб вертикальной оси."""
import numpy as np
import matplotlib.pyplot as plt
from functions import distribution_function

print('Log/Lin?')
o = input()
iscalculated = open('Data/dbl.txt').readlines()[0].strip()
if iscalculated == 'Not calculated':
    x, y = distribution_function(np.load('Data/DF_L.npy'))
    np.save('Data/DF_L_X', x)
    np.save('Data/DF_L_Y', y)
    with open('Data/dbl.txt', 'w') as file:
        file.write('Calculated')
else:
    x = np.load('Data/DF_L_X.npy')
    y = np.load('Data/DF_L_Y.npy')

xrg = np.arange(0, np.max(x), 0.0001)
yr = np.exp(-2 * xrg * xrg)
c = np.sqrt(np.pi / 8)
p = np.load('Data/MeanHs.npy') / np.load('Data/Average Depth.npy')
yg = np.exp(-1 * (np.pi / (4 + p)) * ((xrg / c) ** (2 / (1 - p * c))))

fig = plt.figure(num=f'Height distribution {o} scale')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)

if o == 'Log':
    ax.plot(x, y, linewidth=2, marker='.', alpha=.65, color='#06276F', label='Heights')
    ax.plot(xrg, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution')
    ax.plot(xrg, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
    ax.set_xlim(left=0)
    ax.set_ylim(top=1, bottom=np.min(y) - 10e-7)
    plt.yscale('log')
elif o == 'Lin':
    ax.plot(x, y, linewidth=6, alpha=.8, color='#06276F', label='Heights')
    ax.plot(xrg, yr, linewidth=4, color='black', linestyle='--', label='Rayleigh distribution')
    ax.plot(xrg, yg, linewidth=4, color='black', label='Glukhovskiy distribution')
    ax.set_xlim(left=0, right=2)
    ax.set_ylim(top=1, bottom=0)
ax.set_xlabel('H/Hs', fontsize=20)
ax.set_ylabel('F(H)', fontsize=20)
ax.grid()
plt.legend(fontsize=16)
plt.show()
