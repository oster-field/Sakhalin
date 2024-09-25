import numpy as np
import matplotlib.pyplot as plt

x = np.load('x_approaches_DP.npy')
eps_experimental = np.load('eps_experimental_DP.npy')
eps_theoretical = np.load('eps_theoretical_DP.npy')
fig = plt.figure()
ax = fig.add_subplot(111)
y = 100 * np.abs(eps_experimental - eps_experimental[0]) / eps_experimental[0]
ax.plot(x, y, color='#A63E00', alpha=.85, linewidth=2.5, marker='.', label='$ϵ_{experimental}$')
ax.set_xlabel('Sampling frequency, Hz', fontsize=20)
ax.set_ylabel('Measurement error value, %', fontsize=20)
ax.tick_params(labelsize=20)
plt.xticks(np.arange(min(x), max(x)+1, 7), rotation=45)
#  ax.legend(fontsize=15)
ax.grid()
plt.show()