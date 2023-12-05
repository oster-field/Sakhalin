"""Построение графика спектра для записи целиком. Число точек для построения сокращено."""
import numpy as np
import matplotlib.pyplot as plt
from functions import seriesreducer, find_max_values

Times = 0  # Насколько разредить спектр, количество точек уменьшается в (2**Times)
s = np.load('Data/SpectrumY.npy')
w = np.load('Data/SpectrumX.npy')
samples = np.load('Data/Samples.npy')
s = seriesreducer(s, Times)
w = seriesreducer(w, Times)
s = np.sqrt((2 * np.abs(s.real)) / samples)
positivepart = 0
for freq in w:
    if freq >= 0:
        positivepart += 1
positivew = w[0:positivepart]
positives = s[0:positivepart]
envelope, ind = find_max_values(positives, 10 * 2 ** (11 - Times))
wenvelope = w[ind]
np.save('Data/EnvelopeY', envelope)
np.save('Data/EnvelopeX', wenvelope)

fig = plt.figure(num='Wave spectrum approximation')
ax = fig.add_subplot(111)
ax.tick_params(labelsize=20)
ax.plot(positivew, positives, linewidth=2, color='#007241', label='Wave spectrum approximation')
ax.plot(wenvelope, envelope, linewidth=2, color='#A62F00', label='Envelope')
ax.set_xlabel('ω, [rad/sec]', fontsize=20)
ax.set_ylabel('A(ω), [m]', fontsize=20)
ax.axhline(y=0, color='black', linewidth=1)
ax.set(xlim=[0, np.max(w)])
ax.set_ylim(bottom=0)
ax.grid(axis="y")
plt.legend(fontsize=16)
plt.show()
