import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.fftpack import fft, ifft, fftfreq, rfft, rfftfreq, irfft
from functions import *
import pandas as pd
import sys
from tqdm import tqdm
from scipy.signal.windows import hann
from sympy.solvers import nsolve
from sympy import Symbol, tan
from scipy.interpolate import CubicSpline

arr = np.load('Data/DF_NA.npy')
arr2 = arr
arr3 = arr
r = split_array(arr, np.array([len(arr) // 3, len(arr) // 3, len(arr) // 3 + len(arr) % 3]))
print(r)

