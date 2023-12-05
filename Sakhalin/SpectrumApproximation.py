"""Аппроксимация спектра полиномом, ПКМ - обрезать спектр справа, СКМ - усреднить соседние значения спектра,
ЛКМ - увеличение степени полинома."""
import numpy as np
import matplotlib.pyplot as plt
from functions import find_closest_index, average_neighbours


def onclick_2(event):
    global degree, x, y, curve, averagecurve
    if event.button == 1 and event.inaxes == ax:
        line = curve.pop(0)
        line.remove()
        poly = np.poly1d(np.polyfit(x, y, degree))
        degree += 5
        curve = ax.plot(x, poly(x), linewidth=2, alpha=.9, color='#320A71')
        ax.set_title('Polynom of ' + str(degree) + ' degree')
        plt.draw()
    elif event.button == 2 and event.inaxes == ax:
        line = averagecurve.pop(0)
        line.remove()
        x = average_neighbours(x)
        y = average_neighbours(y)
        averagecurve = ax.plot(x, y, linewidth=4, alpha=.6, color='#007439')
        plt.draw()
    elif event.button == 3 and event.inaxes == ax:
        imax = find_closest_index(x, event.xdata)
        x = x[:imax:]
        y = y[:imax:]
        ax.set_xlim(left=0, right=np.max(x))
        plt.draw()


degree = 3
x = np.load('Data/EnvelopeX.npy')
y = np.load('Data/EnvelopeY.npy')
p = np.poly1d(np.polyfit(x, y, degree))
fig = plt.figure()
ax = fig.add_subplot(111)
curve = ax.plot(0, 0)
ax.tick_params(labelsize=20)
averagecurve = ax.plot(0, 0)
ax.plot(x, y, linewidth=6, alpha=.3, color='#A62F00', label='Envelope of spectrum approximation')
ax.set_xlabel('ω, [rad/sec]', fontsize=20)
ax.set_ylabel('A(ω), [m]', fontsize=20)
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlim(left=0, right=np.max(x))
ax.set_ylim(bottom=0)
ax.grid(axis="y")
plt.legend(fontsize=16)
fig.canvas.mpl_connect('button_press_event', onclick_2)
plt.show()
