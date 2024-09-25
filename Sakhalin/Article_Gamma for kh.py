import numpy as np
import datetime
from tqdm import tqdm
import sys
import pandas as pd
import matplotlib.pyplot as plt

kh = np.arange(0.001, 10, 0.001)
v = 1 + (2 * kh) / (np.sinh(2 * kh))
a = -v**2 + 2 + 8 * (kh**2) * ((np.cosh(2 * kh)) / ((np.sinh(2 * kh))**2))
b = (np.cosh(4 * kh) + 8 - 2 * (np.tanh(kh))**2) / (8 * (np.sinh(kh))**4) - ((2 * (np.cosh(kh))**2 + 0.5 * v)**2) / ((np.sinh(2 * kh))**2 * ((kh / (np.tanh(kh))) - (v / 2)**2))
g = v * np.sqrt(np.abs(b) / a)
fig, ax = plt.subplots(1, 1)
ax.plot(kh, g, linewidth=3, alpha=.95, color='b')
ax.set_ylim(top=2.125, bottom=0)
ax.set_xlim(left=0, right=10)
ax.tick_params(labelsize=20)
ax.set_xlabel('kh', fontsize=20)
ax.set_ylabel('Г(kh)', fontsize=20)
ax.grid()
ax.set_xticks(np.arange(0, 10, 0.5))
ax.set_xticks(np.arange(0, 10, 0.1), minor=True)
plt.yticks(np.arange(0, 2.125, 0.125))
plt.subplots_adjust(left=0.067, bottom=0.088, right=0.97, top=0.974, wspace=0.2, hspace=0.2)
plt.show()
