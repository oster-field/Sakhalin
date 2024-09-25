import numpy as np
import matplotlib.pyplot as plt
import random
from PyAstronomy import pyaC
from tqdm import tqdm
from scipy.fftpack import rfft, rfftfreq
from scipy.signal.windows import hann

w0_displacement = 4
WDT = 150
N = 2 ** 15
dt = 2 * np.pi / WDT
dw = WDT / N
L = dt * N
t = np.arange(0, L, dt, dtype=np.float64)
c = 0
y = 0
w = 0
Sy = np.arange(0)
Sw = np.arange(0)
for i in tqdm(range(0, N)):
    w = w + dw
    v = random.uniform(0, 2*np.pi)
    S = np.exp(-2 * (w-w0_displacement)**2)
    Sy = np.append(Sy, S)
    Sw = np.append(Sw, w)
    MonochromaticWave = (np.sqrt(2*dw*S))*(np.cos(w*t+v))
    y = y + MonochromaticWave
mask = hann(len(y))
s = np.abs(rfft(y * mask))
w = rfftfreq(len(t), dt / (2*np.pi))
s = (s ** 2) / (len(y) * np.max(w))
SD = np.exp(-2 * (w-w0_displacement)**2)
plt.plot(w, SD)
plt.plot(w, s)
plt.show()
