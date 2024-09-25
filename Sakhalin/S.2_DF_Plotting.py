import numpy as np
import matplotlib.pyplot as plt

print('kh/Tz/a/eps/Ur/width/w0/energy?')
o = input()
x1 = np.load(f'df(p)_all{o}_x1.npy')[0]
x2 = np.load(f'df(p)_all{o}_x2.npy')[0]
x3 = np.load(f'df(p)_all{o}_x3.npy')[0]
x4 = np.load(f'df(p)_all{o}_x4.npy')[0]
y1 = np.linspace(1, 0, len(x1), endpoint=False)
y2 = np.linspace(1, 0, len(x2), endpoint=False)
y3 = np.linspace(1, 0, len(x3), endpoint=False)
y4 = np.linspace(1, 0, len(x4), endpoint=False)
p0 = np.load(f'{o}0.npy')
p1 = np.load(f'{o}1.npy')
p2 = np.load(f'{o}2.npy')
p3 = np.load(f'{o}3.npy')
p4 = np.load(f'{o}4.npy')

fig = plt.figure(num=f'Distribution function splitted by {o}')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
colors = ['#412C84', '#269926', '#BF3030', '#FF6A00']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
ax.plot(x1, y1, linewidth=2, marker='.', alpha=.65,
        color=colors[0], label=f'{o} ∈ [{np.round(p0, 3)};{np.round(p1, 3)})')
ax.plot(x2, y2, linewidth=2, marker='.', alpha=.65,
        color=colors[1],
        label=f'{o} ∈ [{np.round(p1, 3)};{np.round(p2, 3)})')
ax.plot(x3, y3, linewidth=2, marker='.', alpha=.65,
        color=colors[2], label=f'{o} ∈ [{np.round(p2, 3)};{np.round(p3, 3)})')
ax.plot(x4, y4, linewidth=2, marker='.', alpha=.65,
        color=colors[3], label=f'{o} ∈ [{np.round(p3, 3)};{np.round(p4, 3)}]')

xrg = np.arange(0, np.max(np.array([np.max(x1), np.max(x2), np.max(x3), np.max(x4)])), 0.0001)
c = np.sqrt(np.pi / 8)
"""for i in range(1, 5):
    g = np.load(f'MeanHs{i}.npy')[0] / np.load(f'MeanDepth{i}.npy')[0]
    yg = np.exp(-1 * (np.pi / (4 + g)) * ((xrg / c) ** (2 / (1 - g * c))))
    ax.plot(xrg, yg, linewidth=2, color=colors[i-1], label='Glukhovskiy distribution', linestyle=linestyles[i-1])"""

ax.set_xlabel('H/Hs', fontsize=20)
ax.set_ylabel('F(H)', fontsize=20)
ax.set_xlim(left=0)
ax.set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])))
ax.grid()
plt.yscale('log')
plt.legend(fontsize=16)
plt.show()
