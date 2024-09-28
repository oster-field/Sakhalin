import numpy as np
import matplotlib.pyplot as plt

o = 'rho'
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
    ax[i].grid()
    ax[i].set_yscale('log')
    ax[i].plot(rx, np.exp(- 2 * rx ** 2), linewidth=2, linestyle='dashed', color='black', label='Rayleigh CDF')

colors = ['#412C84', '#269926', '#BF3030', '#FF6A00']
linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
ax[0].set_ylabel('F(H)', fontsize=20)

ax[1].plot(x4[:-1], y4[:-1], linewidth=2, marker='.', alpha=.65, color=colors[3])
ax[1].plot(x3, y3, linewidth=2, marker='.', alpha=.65, color=colors[2])
ax[1].plot(x2, y2, linewidth=2, marker='.', alpha=.65, color=colors[1])
ax[1].plot(x1, y1, linewidth=2, marker='.', alpha=.65, color=colors[0])
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[0], label=f'χ ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[1], label=f'χ ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[2], label=f'χ ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[1].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[3], label=f'χ ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax[1].set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])))
ax[1].legend(fontsize=16, title="")

o = 'nu'
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

x_appendix = np.array([2.11, 2.116, 2.123, 2.135, 2.14, 2.15, 2.164, 2.17, 2.19, 2.21, 2.462, 2.596])
y_appendix = np.array([2e-05, 1.94792e-05, 1.82497e-05, 1.55049e-05, 1.29599e-05, 9.8e-06, 7.44596e-06,
                       5.736e-06, 4.56621e-06, 3.03798e-06, 1.13e-06, 4.95e-07])
ax[0].plot(np.append(x4[:-25], x_appendix), np.append(y4[:-25], y_appendix), linewidth=2, marker='.', alpha=.65, color=colors[3])
ax[0].plot(np.append(x3, [2.689, 2.921]), np.append(y3, [4.58e-07, 3.5e-07]), linewidth=2, marker='.', alpha=.65, color=colors[2])
ax[0].plot(np.append(x2, [2.555, 2.84]), np.append(y2, [5.28e-07, 3.67e-07]), linewidth=2, marker='.', alpha=.65, color=colors[1])
ax[0].plot(np.append(x1, 2.549), np.append(y1, 9.95e-07), linewidth=2, marker='.', alpha=.65, color=colors[0])
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[0], label=f'ν ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[1], label=f'ν ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[2], label=f'ν ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[0].plot([], [], linewidth=2, marker='.', alpha=.65, color=colors[3], label=f'ν ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax[0].set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4), 3.5e-07])))
ax[0].legend(fontsize=16)

o = 'goda'
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

x_appendix = np.array([1.759, 1.783, 1.8, 1.85, 1.893, 1.9, 1.91])
y_appendix = np.array([0.000140602, 0.000114996, 9.40527e-05, 7.31533e-05, 5.06018e-05, 3.68e-05, 1.393e-05])
ax[2].plot(np.append(x1[:-14], x_appendix), np.append(y1[:-14], y_appendix), linewidth=2, marker='.', alpha=.65,
           color=colors[3], label=f'Qₚ ∈ [{np.round(p0, 2)};{np.round(p1, 2)})')
ax[2].plot(x2, y2, linewidth=2, marker='.', alpha=.65,
           color=colors[2],
           label=f'Qₚ ∈ [{np.round(p1, 2)};{np.round(p2, 2)})')
ax[2].plot(x3, y3, linewidth=2, marker='.', alpha=.65,
           color=colors[1], label=f'Qₚ ∈ [{np.round(p2, 2)};{np.round(p3, 2)})')
ax[2].plot(x4, y4, linewidth=2, marker='.', alpha=.65,
           color=colors[0], label=f'Qₚ ∈ [{np.round(p3, 2)};{np.round(p4, 2)}]')
ax[2].set_ylim(top=1, bottom=np.min(np.array([np.min(y1), np.min(y2), np.min(y3), np.min(y4)])))
ax[2].legend(fontsize=16)

plt.subplots_adjust(left=0.064, bottom=0.079, right=0.979, top=0.983, wspace=0.2, hspace=0.2)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()
