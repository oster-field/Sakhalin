import numpy as np
import matplotlib.pyplot as plt

o = 'BFI_goda'
x1 = np.load(f'df(p)_all{o}_x1.npy')[0]
x2 = np.load(f'df(p)_all{o}_x2.npy')[0]
x3 = np.load(f'df(p)_all{o}_x3.npy')[0]
x4 = np.load(f'df(p)_all{o}_x4.npy')[0]
y1 = np.linspace(1, 0, len(x1), endpoint=False)
y2 = np.linspace(1, 0, len(x2), endpoint=False)
y3 = np.linspace(1, 0, len(x3), endpoint=False)
y4 = np.linspace(1, 0, len(x4), endpoint=False)
p0 = np.load(f'{o}0.npy') * 2
p1 = np.load(f'{o}1.npy') * 2
p2 = np.load(f'{o}2.npy') * 2
p3 = np.load(f'{o}3.npy') * 2
p4 = np.load(f'{o}4.npy') * 2

fig, ax = plt.subplots(1, 2)
for i in range(2):
    ax[i].tick_params(labelsize=20)
    ax[i].set_xlabel('H/4σ', fontsize=20)
    ax[i].set_xlim(left=0, right=3)
    ax[i].grid()
    ax[i].set_yscale('log')

ax[0].set_ylabel('F(H)', fontsize=20)
colors = ['#412C84', '#269926', '#BF3030', '#FF6A00']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']

ax[0].plot(x1, y1, linewidth=2, marker='.', alpha=.65,
           color=colors[0], label=f'BFI ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[0].plot(x2, y2, linewidth=2, marker='.', alpha=.65,
           color=colors[1],
           label=f'BFI ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[0].plot(x3, y3, linewidth=2, marker='.', alpha=.65,
           color=colors[2], label=f'BFI ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[0].plot(x4, y4, linewidth=2, marker='.', alpha=.65,
           color=colors[3], label=f'BFI ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax[0].set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])))
ax[0].legend(fontsize=16)

o = 'BFI_proper'
x1 = np.load(f'df(p)_all{o}_x1.npy')[0]
x2 = np.load(f'df(p)_all{o}_x2.npy')[0]
x3 = np.load(f'df(p)_all{o}_x3.npy')[0]
x4 = np.load(f'df(p)_all{o}_x4.npy')[0]
y1 = np.linspace(1, 0, len(x1), endpoint=False)
y2 = np.linspace(1, 0, len(x2), endpoint=False)
y3 = np.linspace(1, 0, len(x3), endpoint=False)
y4 = np.linspace(1, 0, len(x4), endpoint=False)
p0 = np.load(f'{o}0.npy') * 2
p1 = np.load(f'{o}1.npy') * 2
p2 = np.load(f'{o}2.npy') * 2
p3 = np.load(f'{o}3.npy') * 2
p4 = np.load(f'{o}4.npy') * 2

ax[1].plot(x1, y1, linewidth=2, marker='.', alpha=.65,
           color=colors[0], label=f'BFI (*Г) ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[1].plot(x2, y2, linewidth=2, marker='.', alpha=.65,
           color=colors[1],
           label=f'BFI (*Г) ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[1].plot(x3, y3, linewidth=2, marker='.', alpha=.65,
           color=colors[2], label=f'BFI (*Г) ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[1].plot(x4, y4, linewidth=2, marker='.', alpha=.65,
           color=colors[3], label=f'BFI (*Г) ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax[1].set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])))
ax[1].legend(fontsize=16)

plt.subplots_adjust(left=0.064, bottom=0.079, right=0.979, top=0.983, wspace=0.2, hspace=0.2)
plt.show()
