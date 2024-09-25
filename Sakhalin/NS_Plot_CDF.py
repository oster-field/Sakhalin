import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


Fy = np.load('Find_CDF_F.npy')
Fx = np.load('Find_CDF_x.npy')

p = 0.4469770245124512

d = Fy - np.exp(-2 * Fx ** 2)
a = - np.min(d) - 0.003
x0 = Fx[np.argmin(d)] + 0.008
b = (np.log(2 * a) / x0**2)
b = 21.5
a, b, x0 = 0.7612893466924386, 21.678556702157092, 0.21310166587641477

F = np.exp(-2 * Fx ** 2)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title(f'a={a}, b={b} x0={x0}')
ax.plot(Fx, Fy, color='#5F2580', alpha=.5, linewidth=3.5, label='$F$')
ax.plot(Fx, F, color='black', alpha=1, linestyle='--', linewidth=2, label='$F_{ref}$')
ax.plot(Fx, F - Fy, color='magenta', alpha=.5, linewidth=3.5)
ax.set_xlabel('Normalized value of individual wave amplitude', fontsize=20)
ax.set_ylabel('CDF', fontsize=20)
ax.tick_params(labelsize=20)
ax.legend(fontsize=15, title=f'ϵ={np.round(p, 2)}, max={np.max(np.abs(Fy - F))}', title_fontsize='15')
ax.grid()
plt.show()
