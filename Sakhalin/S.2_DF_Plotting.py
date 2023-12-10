import numpy as np
import matplotlib.pyplot as plt

x1 = np.load('df(p)_all_x1.npy')
x2 = np.load('df(p)_all_x2.npy')
x3 = np.load('df(p)_all_x3.npy')
x4 = np.load('df(p)_all_x4.npy')
y1 = np.load('df(p)_all_y1.npy')
y2 = np.load('df(p)_all_y2.npy')
y3 = np.load('df(p)_all_y3.npy')
y4 = np.load('df(p)_all_y4.npy')
MeanHs = np.load('MeanHs_all')
MeanDepth = np.load('MeanDepth_all')

xrg = np.arange(0, np.max(np.array([np.max(x1), np.max(x2), np.max(x3), np.max(x4)])), 0.0001)
yr = np.exp(-2 * xrg * xrg)
c = np.sqrt(np.pi / 8)
g = MeanHs / MeanDepth
yg = np.exp(-1 * (np.pi / (4 + g)) * ((xrg / c) ** (2 / (1 - g * c))))

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