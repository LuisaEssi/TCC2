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
import signal_hw
import padasip as pa
from serial.tools.list_ports import comports


import serial
import time


MAX_BUFF_LEN = 15
SETUP 		 = False
port 		 = None

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

MAT_FILE = "Subject1"
# MAT_FILE = "Subject3"
# MAT_FILE = "Subject5"
# MAT_FILE = "Subject6"
# MAT_FILE = "Subject8"
# MAT_FILE = "Subject14"
sub = SubjectPPGRecord(MAT_FILE, db_path = "ppg/", mat = True)
rec = sub.record
fs = sub.fs
red_ppg = rec.red_a
ir_ppg = rec.ir_a
spo2 = rec.spo2
inc_sec = 2560 


#SUBJECT14

# inicio = 355072  # Spo2 = 89 - ok
# inicio = 1380*256  # Spo2 = 89 - _______ok
# inicio = 470*256 # Spo2 = 95/96 -ok
# inicio = 1140*256 # Spo2 = 94 - cagado
# inicio = 1660*256 # Spo2 = 87
# inicio = 3024*256 # Spo2 = 91 - ok

#SUBJECT6

# inicio = 3171*256 # Spo2 = 94 - ok ORDEM = 150 E PASSO = 0,04s
# inicio = 1543*256 # Spo2 = 94 - ok ORDEM = 150 E PASSO = 0,04s
# inicio = 1615*256 # Spo2 = 94 - ok _________OK

#SUBJECT8

# inicio = 2315*256 # Spo2 = 96 - _____OK


#SUBJECT1

inicio = 1925 *256 #Spo2 = 95%  Tempo = 93 ____OK

# inicio = 2120 *256 #Spo2 = 95%  Tempo = 93

#SUBJECT3
# inicio = 1990 *256 #Spo2 = 97%  Tempo = 93

#SUBJECT5
# inicio = 2400*256 #Spo2 = 97%  Tempo = 

# DADOS HARDWARE
# inicio = 200
# inc_sec = 2500
# fs = 250
# sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/weliton.txt",freq = fs)
# sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/manuella.txt",freq = fs)
# sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste_18_01_anso.txt",freq = fs)

red_bac = red_ppg[inicio:inicio+inc_sec]
ir_bac = ir_ppg[inicio:inicio+inc_sec]

#FUNCAO
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

    for i in range(I):
        # u = Auxiliar variable. Store the piece of the corruptedInput
        # that will be multiplied by the current set of coefficients.
        u = np.flipud(signal_corrupted[i:i + order])[:, np.newaxis]
        signal_approx[i] = w[:, [i - 1]].T @ u

        error[i] = signal[i + order - 1] - signal_approx[i]

        # Updating the filter coeficients
        w[:, [i]] = w[:, [i - 1]] + step * u * (error[i])
        signal_approx[i] = w[:, [i]].T @ u
    
    return signal_approx, error, w


# USB
while(1):
    ports = [port for port in comports()]
    portIndex = 0
    if len(ports) <= 0:
        print("Nenhum aparelho conectado, abortando codigo")
        exit()
    elif len(ports) == 1:
        print("Usando porta USB {}".format(ports[portIndex].device))
    else:
        print("Portas USB disponiveis:")
        i=0
        for port in ports:
            print("{}- {}".format(i, port.device))
            i+=1

    port = ports[portIndex].device
    # port = "/dev/ttyUSB0"
    baudrate = 9600
    # fileName = "data.txt"
    fs = int((500/baudrate)*1000)
    inc_sec = int(fs* 10)
    samples = inc_sec # 6sec for processing data 


    ser = serial.Serial(port,baudrate)

    print("Connected to ESP port:" + port)
    ser.flushInput()
    print("Abrindo Serial")

    line = 0
    vec = []
    while line < samples:
        data = str(ser.readline().decode("utf-8"))
        vec.append(data)
        line = line+1
    print("Final de leituras")

    red_ppg = []
    ir_ppg = []

    inicio = 0
    vec = vec[1:len(vec)-2]
    for i in range(len(vec)):
        red_usb,ir_usb = vec[i].split(",")
        red_ppg.append(float(red_usb))
        ir_ppg.append(float(ir_usb))


    red_bac = red_ppg[inicio:inicio+inc_sec]
    ir_bac = ir_ppg[inicio:inicio+inc_sec]
    print(len(red_bac))
    t = np.linspace(0, ((samples / fs)), len(red_bac))




    # BANDPASS SIGNAL - Filtro AC
    cutoff_lac = 0.4
    cutoff_hac = 8
    BP_ORDER = 8

    rd = butter_bandpass_filter(red_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
    ir = butter_bandpass_filter(ir_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)


    # print(spo2[inicio:inicio+inc_6_sec])


    t = np.linspace(0, ((len(rd) / fs)), len(rd))
    rd = rd/np.max(rd)
    ir = ir/np.max(ir)


    # plt.plot(t,rd, label= 'red norm')
    # plt.plot(t, ir, label= 'ir norm')
    # plt.legend()
    # plt.show()


    #CALCULOS PARA FFT

    #IR

    c_ir_fil = np.abs(fftshift(fft(ir)))

    abs_ir_cfil = c_ir_fil[:int(np.floor(len(c_ir_fil)/2))]

    f2 = np.linspace(0, fs/2, num = len(abs_ir_cfil))

    abs_ir_fil  = abs_ir_cfil /np.max(abs_ir_cfil)
    abs_ir_fil = np.flipud(abs_ir_fil[0::])


    #RED

    c_red_fil = np.abs(fftshift(fft(rd)))

    abs_red_cfil = c_red_fil[:int(np.floor(len(c_red_fil)/2))]

    f4 = np.linspace(0, fs/2, num = len(abs_red_cfil))

    abs_red_fil  = abs_red_cfil /np.max(abs_red_cfil)
    abs_red_fil = np.flipud(abs_red_fil[0::])


    #PLOTS


    # plt.subplot(2, 1, 1)
    # plt.plot(f2,abs_ir_fil,"b", label = "FFT do sinal IR com filtragem")
    # plt.ylabel("Amplitude [mV]")
    # plt.xlabel("Frequencia [Hz]")
    # plt.legend()
    # plt.subplot(2, 1, 2)
    # plt.plot(f4,abs_red_fil,"r", label = "FFT do sinal RED com filtragem")
    # plt.ylabel("Amplitude [mV]")
    # plt.xlabel("Frequencia [Hz]")
    # plt.legend()
    # plt.show()

    # calculo na frequencia

    frequencia = f4[np.where(abs_ir_fil ==np.max(abs_ir_fil))]
    BPM = frequencia*60
    print(BPM, "bpm")


    # LMS ADAPTATIVE FILTER
    # signal_corrupted = sinal de referencia RS
    # signal = sinal original (IR) sem filtro
    filt = pa.filters.FilterLMS(10,mu=0.1)

    r = 0
    f_list = []
    DSP_list = []
    power_list = []
    signal_recev_list = []

    RS_list = []
    # passo = 0.0004
    passo = 0.004 #HW
    # passo = 0.001 #BANCO
    ordem = 150

    for i in range (r,101,1):
        r_ir = ((i/100)) * ir 
        RS = r_ir - rd
        RS_list.append(RS)

        signal_recev, mse ,w = lms(RS,ir, passo, ordem)

        fft_saida = np.abs(fft(signal_recev))    
        fft_saida = fft_saida[:int(np.floor(len(fft_saida)/2))]
        fr = np.linspace(0, fs/2, num = len(fft_saida))

        power = np.sum(np.abs(fft_saida)**2)
        power_list.append(power)
        signal_recev_list.append(signal_recev)
        power = 0

    # for i in range(len(RS_list)-1): 
    #     plt.plot(t, RS_list[i], label = i)
    # plt.legend()
    # plt.show()



    # power_max = [np.max(power_list)] * 50

    # for i in range(len(power_list)):
    #     power_max.append(power_list[i])


    t2 = np.linspace(0, ((len(ir) / fs)), len(ir)-(ordem-1))

    # Filtragem LMS

    # plt.subplot(2, 1, 1)
    # plt.plot(t, ir,"r", label = "Sinal original IR")
    # plt.ylabel("Amplitude [mV]")
    # plt.xlabel("Tempo [segundos]")
    # plt.legend()

    # plt.subplot(2, 1, 2)
    # plt.plot(t2,signal_recev_list[9],"b", label = "Sinal filtrado pelo LMS")
    # plt.ylabel("Amplitude [mV]")
    # plt.xlabel("Tempo [segundos]")
    # plt.legend()
    # plt.show()

    # for i in range(len(signal_recev_list)-1):
    #     plt.plot(t2, signal_recev_list[i], label= i)
    # plt.plot(t, ir, label= 'IR')

    # plt.legend()
    # plt.show()
    # plt.plot(np.max(power_list)-power_list, label = "Curva de potência")
    # plt.axvline(power_list.index(min(power_list)), color = 'red', linestyle = '--')
    # plt.legend()
    # plt.xlabel("SpO2(%)")
    # plt.ylabel("Potência")
    # plt.show()
    print('O valor de SpO2 (f) é :', power_list.index(min(power_list)), '%')

    def write_ser(cmd):
        cmd = cmd + '\n'
        ser.write(cmd.encode())

    write_ser('SpO2: ' + str(int(power_list.index(min(power_list))))+ '%' + ' FC: '+ str(int(BPM))+'bpm')
    ser.flushInput()
    ser.flushOutput()
ser.close()