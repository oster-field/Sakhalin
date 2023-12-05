"""Вычисляет оконное преобразование Фурье с оконной функцией Ханна. Сохраняет ширины на каждом сдвиге окна."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, rfftfreq
from tqdm import tqdm
from functions import Sensor_Frequency
from scipy.signal.windows import hann

WindowSize = 10  # Окно преобразования, в минутах
DeltaWindow = 10  # Сдвиг окна, в секундах
part = 10  # Процент от спектра, достаточный для отображения
print('Log/Lin?')
o = input()
if o == 'Log':
    part *= 2
y = np.load('Data/FullRec.npy')
window = WindowSize * 60 * Sensor_Frequency
n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))
w = rfftfreq(window, 1 / Sensor_Frequency)
z = []
width = np.arange(0)
w0 = np.arange(0)
E = np.arange(0)

for i in tqdm(range(0, n), desc="Progress: ", colour='green'):
    arr = y[i*DeltaWindow:window + i*DeltaWindow]
    mask = hann(len(arr))
    s = np.abs(rfft(arr * mask))
    s = np.sqrt((2 * s[0:int(len(w) * .01 * part)]) / len(arr))
    z.append(np.flip(s))
z = np.asarray(z)

fig = plt.figure(num=f'Window FT {o} scale')
ax = fig.add_subplot(111)
if o == 'Log':
    img = ax.imshow(np.flip(np.flip(np.log10(z)).T),
                    extent=[0, WindowSize / 60 + n * DeltaWindow / 3600, 0, w[int(len(w) * .01 * part)]], cmap='rainbow')
    colorbar = plt.colorbar(img, format='10e%1.0f m', shrink=0.75)
    plt.tight_layout(h_pad=1)
elif o == 'Lin':
    img = ax.imshow(np.flip(np.flip(z).T),
                    extent=[0, WindowSize / 60 + n * DeltaWindow / 3600, 0, w[int(len(w) * .01 * part)]], cmap='PuOr')
    colorbar = plt.colorbar(img, format='%1.2f m', shrink=0.75)
    plt.tight_layout(h_pad=1)
colorbar.ax.tick_params(labelsize=20)
ratio = 0.5
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax.tick_params(labelsize=20)
ax.set_xlabel('t, [hours]', fontsize=20)
ax.set_ylabel('f, [Hz]', fontsize=20)
plt.show()
