import numpy as np
import matplotlib.pyplot as plt

o = 'Ur'
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

fig, ax = plt.subplots(1, 3)
rx = np.arange(0, 3, 0.0001)
for i in range(3):
    ax[i].tick_params(labelsize=20)
    ax[i].set_xlabel('H/4σ', fontsize=20)
    ax[i].set_xlim(left=0, right=3)
    ax[i].set_ylim(top=1, bottom=1.55e-07)
    ax[i].grid()
    ax[i].set_yscale('log')
    ax[i].plot(rx, np.exp(- 2 * rx**2), linewidth=2, linestyle='dashed', color='black', label='Rayleigh CDF')

colors = ['#412C84', '#269926', '#BF3030', '#FF6A00']
ax[0].set_ylabel('F(H)', fontsize=20)

ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[0], label=f'Ur ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[1], label=f'Ur ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[2], label=f'Ur ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[3], label=f'Ur ∈ [{np.round(p3, 2)};{np.round(p4, 2)})')

ax[0].plot(x4, y4, linewidth=2, marker='.', alpha=.65, color=colors[3])
ax[0].plot(x3[:-1], y3[:-1], linewidth=2, marker='.', alpha=.65, color=colors[2])
ax[0].plot(x2[:-2], y2[:-2], linewidth=2, marker='.', alpha=.65, color=colors[1])
ax[0].plot(x1, y1, linewidth=2, marker='.', alpha=.65, color=colors[0])
ax[0].legend(fontsize=16, title="")

o = 'eps'
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

ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[0], label=f'kσ ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[1], label=f'kσ ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[2], label=f'kσ ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[3], label=f'kσ ∈ [{np.round(p3, 2)};{np.round(p4, 2)})')

ax[1].plot(x4, y4, linewidth=2, marker='.', alpha=.65, color=colors[3])
ax[1].plot(x3[:-1], y3[:-1], linewidth=2, marker='.', alpha=.65, color=colors[2])
ax[1].plot(x2[:-2], y2[:-2], linewidth=2, marker='.', alpha=.65, color=colors[1])
ax[1].plot(x1, y1, linewidth=2, marker='.', alpha=.65, color=colors[0])
ax[1].legend(fontsize=16, title="")

o = 'kh'
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

ax[2].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[0], label=f'kh ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[2].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[1], label=f'kh ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[2].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[2], label=f'kh ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[2].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[3], label=f'kh ∈ [{np.round(p3, 2)};{np.round(p4, 2)})')

ax[2].plot(x4, y4, linewidth=2, marker='.', alpha=.65, color=colors[3])
ax[2].plot(x3[:-1], y3[:-1], linewidth=2, marker='.', alpha=.65, color=colors[2])
ax[2].plot(x2[:-2], y2[:-2], linewidth=2, marker='.', alpha=.65, color=colors[1])
ax[2].plot(x1, y1, linewidth=2, marker='.', alpha=.65, color=colors[0])
ax[2].legend(fontsize=16, title="")

plt.subplots_adjust(left=0.064, bottom=0.079, right=0.979, top=0.983, wspace=0.2, hspace=0.2)
plt.show()