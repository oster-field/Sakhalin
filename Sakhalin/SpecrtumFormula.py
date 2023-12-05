import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.fftpack import fft, ifft, fftfreq, rfft, rfftfreq, irfft
from functions import DateStart, DateEnd, seriesreducer, newdates
import pandas as pd
import sys
from tqdm import tqdm
from scipy.signal.windows import hann
from scipy.interpolate import CubicSpline
from sympy import *


def formula():
    arr = np.load('Data/2022.10.19 reading 52.npy')[0:3]  # Чем больше тем дольше :(
    xarr = np.arange(len(arr))
    omega = np.arange(0.1, 25, 0.1)
    cs = CubicSpline(xarr, arr)
    coefs = np.asarray(cs.c.T)
    x = Symbol('x')
    w = Symbol('w')
    expr = 0
    npfunk = np.zeros(len(omega))
    for i in tqdm(range(len(xarr) - 1)):
        f = (coefs[i][0] * x**3 + coefs[i][1] * x**2 + coefs[i][2] * x + coefs[i][3]) * exp(-1j * w * x)
        fourier = integrate(f, (x, i, i+1), conds='none')
        expr += fourier
    print(simplify(expr))


def transform():
    y = np.load('Data/Fullrec.npy')[0:2000000]
    x = np.arange(len(y))
    omega = np.arange(0.0001, 6.30, 0.1)
    I = np.zeros(len(omega))
    cs = CubicSpline(x, y)
    coefs = np.asarray(cs.c.T)
    npfunk = np.zeros(len(omega))
    for i in tqdm(range(len(x) - 1)):
        q = i + 1
        p = i
        a = coefs[i][0]
        b = coefs[i][1]
        c = coefs[i][2]
        d = coefs[i][3]
        I += (omega * ((q * (q * (a * q + b) + c) + d) * omega ** 2 - 6 * a * q - 2 * b) * np.sin(q * omega) + (
                    (q * (3 * a * q + 2 * b) + c) * omega ** 2 - 6 * a) * np.cos(q * omega) - omega * (
                          (p * (p * (a * p + b) + c) + d) * omega ** 2 - 6 * a * p - 2 * b) * np.sin(p * omega) - (
                          (p * (3 * a * p + 2 * b) + c) * omega ** 2 - 6 * a) * np.cos(p * omega)) / omega ** 4
    I = I / np.sqrt(2 * np.pi)
    plt.plot(omega, np.sqrt(2 * np.abs(I) / len(y)))
    plt.show()


def interpolatewindow():
    y = np.load('Data/2022.11.01 reading 4.npy')
    interpolationrate = 10
    w = rfftfreq(len(y), (1 / 8))
    mask = hann(len(y))
    s = np.abs(rfft(y * mask))
    plt.plot(w, np.sqrt(2 * s / len(y)))
    cs = CubicSpline(np.arange(len(y)), y)
    y = cs(np.arange(0, len(y), (1 / 8) / interpolationrate))
    wi = rfftfreq(len(y), (1 / 8)) * 8 * interpolationrate
    mask = hann(len(y))
    si = np.abs(rfft(y * mask))
    plt.plot(wi, np.sqrt(2 * si / len(y)))
    plt.show()

