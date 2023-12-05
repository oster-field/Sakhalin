"""Функция распределения амплитуд, нормированные на значительные, первый раз считает довольно долго,
остальные запуски программы по заранее посчитанным данным. Можно менять масштаб вертикальной оси."""
import numpy as np
import matplotlib.pyplot as plt
from functions import distribution_function

print('Log/Lin?')
o = input()
iscalculated = open('Data/dba.txt').readlines()[0].strip()
if iscalculated == 'Not calculated':
    xp, yp = distribution_function(np.load('Data/DF_PA.npy'))
    xn, yn = distribution_function(np.load('Data/DF_NA.npy'))
    np.save('Data/DF_PA_X', xp)
    np.save('Data/DF_PA_Y', yp)
    np.save('Data/DF_NA_X', xn)
    np.save('Data/DF_NA_Y', yn)
    with open('Data/dba.txt', 'w') as file:
        file.write('Calculated')
else:
    xp = np.load('Data/DF_PA_X.npy')
    yp = np.load('Data/DF_PA_Y.npy')
    xn = np.load('Data/DF_NA_X.npy')
    yn = np.load('Data/DF_NA_Y.npy')

x = np.arange(0, np.max(np.array([np.max(xp), np.max(xn)])), 0.0001)
yr = np.exp(-2 * x * x)
c = np.sqrt(np.pi / 8)
p = np.load('Data/MeanAs.npy') / np.load('Data/Average Depth.npy')
yg = np.exp(-1 * (np.pi / (4 + p)) * ((x / c) ** (2 / (1 - p * c))))

fig = plt.figure(num=f'Amplitudes distribution {o} scale')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)

if o == 'Log':
    ax.plot(xn, yn, linewidth=2, marker='.', alpha=.8, color='#2D9B27', label='Negative amplitudes')
    ax.plot(xp, yp, linewidth=2, marker='.', alpha=.8, color='#BE2F33', label='Positive amplitudes')
    ax.plot(x, yr, linewidth=2, color='black', linestyle='--', label='Rayleigh distribution', alpha=.8)
    ax.plot(x, yg, linewidth=2, color='black', label='Glukhovskiy distribution')
    ax.set_xlim(left=0)
    ax.set_ylim(top=1, bottom=np.min(np.array([np.min(yp), np.min(yn)])) - 10e-7)
    plt.yscale('log')
elif o == 'Lin':
    ax.plot(xn, yn, linewidth=8, color='#2D9B27', label='Negative amplitudes')
    ax.plot(xp, yp, linewidth=6, color='#BE2F33', label='Positive amplitudes')
    ax.plot(x, yr, linewidth=3, color='black', linestyle='--', label='Rayleigh distribution', alpha=.8)
    ax.plot(x, yg, linewidth=4, color='black', label='Glukhovskiy distribution')
    ax.set_xlim(left=0, right=2)
    ax.set_ylim(top=1, bottom=0)
ax.set_xlabel('A/As', fontsize=20)
ax.set_ylabel('F(A)', fontsize=20)
ax.grid()
plt.legend(fontsize=16)
plt.show()
