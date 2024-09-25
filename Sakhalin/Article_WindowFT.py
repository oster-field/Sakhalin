"""Вычисляет оконное преобразование Фурье с оконной функцией Ханна. Сохраняет ширины на каждом сдвиге окна."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import rfft, rfftfreq
from tqdm import tqdm
from scipy.signal.windows import hann

Sensor_Frequency = 8
WindowSize = 10  # Окно преобразования, в минутах
DeltaWindow = 60  # Сдвиг окна, в секундах
part = 20  # Процент от спектра, достаточный для отображения
y = np.load('Data2/FullRec.npy')
window = WindowSize * 60 * Sensor_Frequency
n = int((len(y) - window) / (DeltaWindow * Sensor_Frequency))
w = rfftfreq(window, (1 / Sensor_Frequency) / (2 * np.pi))
z = []

for i in tqdm(range(0, n), desc="Progress: ", colour='green'):
    arr = y[i*DeltaWindow:window + i*DeltaWindow]
    mask = hann(len(arr))
    s = np.abs(rfft(arr * mask))[0:int(len(w) * .01 * part)]
    s = (s ** 2) / (len(arr) * np.max(w))
    s = np.log10(s)
    z.append(np.flip(s))
z = np.asarray(z)

fig = plt.figure(num=f'Window FT')
ax = fig.add_subplot(111)
img = ax.imshow(np.flip(np.flip(z).T),
                extent=[0, WindowSize / 60 + n * DeltaWindow / 3600, 0, w[int(len(w) * .01 * part)]], cmap='gist_heat', vmin=-10)
colorbar = plt.colorbar(img, shrink=0.75)
colorbar.ax.set_ylabel('Spectral density, [m²/s]', size=20)
plt.tight_layout(h_pad=1)
colorbar.ax.tick_params(labelsize=20)
ratio = 0.25
x_left, x_right = ax.get_xlim()
y_low, y_high = ax.get_ylim()
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
ax.tick_params(labelsize=20)
ax.set_xlabel('t, [hours]', fontsize=20)
ax.set_ylabel('ω, [rad/s]', fontsize=20)
plt.show()
