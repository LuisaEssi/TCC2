# IR = S + M
# RD = (ra X S) + (rv x M)

# M = sinal de movimento gerado pela pulsacao do sangue venoso
# S = sinal arterial pulsatil
# ra = razao da densidade optica da saturacao arterial (contem o SpO2)
# rv = razao da densidade optica da saturacao venosa
# r' = saturação SPO2 (iterada)

# RS = r' x IR - RD 
# RS = (r'-ra) x S + (r'-rv) x M
from sklearn.preprocessing import StandardScaler
from cmath import nan, sin
import numpy as np
# import pylab as ps
import scipy as sp
from numpy import mean
from numpy import nanmean
from numpy import reshape
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import csv
import statistics
from matplotlib import rc
from scipy.signal import find_peaks
import scipy.integrate 
from sympy import sign
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
from threshold_pev import adaptative_th, fix_pev, loc_pev
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import scipy.signal
import fir_filter
import serial
import time
from numpy import trapz
import padasip as pa


ZERO_PHASE_FILTERS = True
PLOTS = True
# In[]:
# Plot Settings
rc("font", **{"size": 11})
# plt.close("all")
# plt.ion()


# In[]:
# Load Subject Data using helper function
from utils import SubjectPPGRecord

MAT_FILE = "Subject14"
sub = SubjectPPGRecord(MAT_FILE, db_path = "TCC2/ppg/", mat = True)
rec = sub.record
fs = sub.fs
red_ppg = rec.red_a
ir_ppg = rec.ir_a
spo2 = rec.spo2
inc_6_sec = 2560 
# inicio = 355072  # Spo2 = 89
# inicio = 470*256 # Spo2 = 95/96
# inicio = 1140*256 # Spo2 = 94
inicio = 350*256 # Spo2 = 127
# inicio = 3487*256 # Spo2 = 96


rd = red_ppg[inicio:inicio+inc_6_sec]
ir = ir_ppg[inicio:inicio+inc_6_sec]




# print(spo2[inicio:inicio+inc_6_sec])


t = np.linspace(0, ((len(rd) / fs)), len(rd))
rd = rd/np.max(rd)
ir = ir/np.max(ir)


plt.plot(t,rd, label= 'red norm')
plt.plot(t, ir, label= 'ir norm')
plt.legend()
plt.show()


# LMS ADAPTATIVE FILTER
# signal_corrupted = sinal de referencia RS
# signal = sinal original (IR) sem filtro
filt = pa.filters.FilterLMS(10,mu=0.1)
def lms(signal_corrupted, signal, step, order):
    #signal_corrupted: Input Vector.
    #signal: Desired Signal.
    #step: The value that will be used to computate the error vector.
    #order: Filter Order.

    # Number of loops to count because of the buffer delay.
    I = signal_corrupted.size - order + 1
    # Filter output.
    signal_approx = np.zeros([I, 1])
    # Error vector.
    error = np.zeros((I, 1))
    # Matrix that will store the filter coeficients for each loop.
    w = np.zeros((order, I))

    # mse = np.zeros(I,)

    for i in range(I):
        # u = Auxiliar variable. Store the piece of the corruptedInput
        # that will be multiplied by the current set of coefficients.
        u = np.flipud(signal_corrupted[i:i + order])[:, np.newaxis]
        signal_approx[i] = w[:, [i - 1]].T @ u

        error[i] = signal[i + order - 1] - signal_approx[i]
        # mse[i] = np.absolute(error[i]) ** 2 / I

        # Updating the filter coeficients
        w[:, [i]] = w[:, [i - 1]] + step * u * (error[i])
        signal_approx[i] = w[:, [i]].T @ u
    
    return signal_approx, error, w


r = 0
f_list = []
DSP_list = []
power_list = []
signal_recev_list = []

RS_list = []
passo = 0.002
ordem = 200


for i in range (r,101,1):
    r_ir = ((i/100)) * ir 
    RS = rd - r_ir
    RS_list.append(RS)
    step = passo
    order = ordem
    

    y = np.empty(len(rd))
    f = fir_filter.FIR_filter(np.zeros(ordem))

    for j in range(len(rd)):
        # ref_noise = np.sin(2.0 * np.pi * fnoise/fs* 1)
        canceller = f.filter(RS[j]) 
        output_signal = rd[j]-canceller
        f.lms(output_signal, passo)
        y[j] = output_signal


    power = sp.sum(y)**2
    power_list.append(power)
    # signal_recev_list.append(mse)
    # plt.plot(t2, signal_recev, label= i)

# print(power_list[1], 'valor de power')
# print(signal_recev_list[1], 'valor mse')

plt.plot(t, RS_list[89], label = 'RS')
plt.legend()
plt.show()

plt.figure()
plt.plot(y, label = 'filtro')
plt.plot(rd, label= 'RED')
plt.legend()
plt.show()


# t2 = np.linspace(0, ((len(rd) / fs)), len(rd)-(ordem-1))
# plt.plot(t2, signal_recev_list[1], label= 'adaptativo')
# plt.plot(t, rd, label= 'RED')


# plt.legend()
# plt.show()
    # (f, S) = scipy.signal.periodogram(signal_recev, fs)
    # f_list.append(f)
    # DSP_list = max(S)   
    
    # DSP_list.append(S)  

# recev_fft = np.abs(fftshift(fft(signal_recev_list[90]))) 
# f2 = np.linspace(0, fs/2, num = len(recev_fft))
# plt.show()
# for i in range(0,101,1):
#     recev_fft = np.abs(fftshift(fft(signal_recev_list[i]))) 
#     f2 = np.linspace(0, fs/2, num = len(recev_fft))

#     plt.plot(f2, recev_fft, label= i)

#     plt.legend()
    
# plt.plot(freqs[idx],DSP_list[89][idx], label = '89')

# print(signal_recev.shape)
plt.stem(power_list)

plt.legend()
plt.show()
# print(power_list.index(max(power_list)))
# print(coefs)

