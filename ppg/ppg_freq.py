# In[]:
# from turtle import distance
# from cv2 import threshold
from cmath import nan
import numpy as np
from numpy import mean
from numpy import nanmean
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import csv
import statistics
from matplotlib import rc
from scipy.signal import find_peaks
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
from threshold_pev import adaptative_th, fix_pev, loc_pev
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

import serial
import time


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
sub = SubjectPPGRecord(MAT_FILE, db_path=".", mat=True)
rec = sub.record
fs = sub.fs
red_ppg = rec.red_a
ir_ppg = rec.ir_a
inc_10_sec = 2560
spo2 = rec.spo2
soma_freq = inc_10_sec
spo2_filtred = []
r_windowed = []
passos = 0

t = np.linspace(0, ((len(red_ppg) / fs)), len(red_ppg))
red_ppg_n = red_ppg/max(red_ppg)

if PLOTS:
    plt.plot(t,spo2,"b-", linewidth = 1, label = "FFT IR filtrado (AC)")
    plt.legend()
    plt.figure()
    
if PLOTS:
    plt.plot(t,ir_ppg,"b-", linewidth = 1, label = "IR")
    plt.legend()
    plt.show()
    



# print(spo2[355072:355072+2560])
# # red = red_ppg[200000:]
# # ir = ir_ppg[200000:]
# # print(len(spo2))
# red = red_ppg
# # ir = ir_ppg[soma_freq:soma_freq_t]
# # spo2_ver = (spo2[soma_freq:soma_freq_t])
# # # print(spo2_ver)

# ## retirando valores de spo2 maiores que 100%
# # for i in range(len(spo2)):
# #     if ((spo2[i])<=100):
# #        spo2_filtred.append(spo2[i]) 

# print(nanmean(spo2))
# spo2_mean = nanmean(spo2)
# for i in range(soma_freq,len(spo2),inc_10_sec):
#     if ((spo2[i])<=100):
#         spo2_filtred.append(spo2[i]) 
#         # spo2_mean = mean((spo2_filtred))
#     else:
#         spo2_filtred.append(spo2_mean) 

# print("spo2 do banco: ",len(spo2_filtred))
# #NOSSO PPG

# # lista = [] 
# # fs = 512
# # with open('dadosnosso.csv', newline='') as csvfile:
# #     # o nome 'spamreader' abaixo é só exemplo, poderia ser qq. coisa
# #     spamreader = csv.reader(csvfile, delimiter=',') # separe por vírgula

# #     # o módulo csv detectará novas linhas automaticamente
# #     for linha in spamreader:
# #         lista.append(linha)
# #     lista = np.array(lista,float)

# # # print(lista[0::,0], "RED") #
# # # print(lista[0::,1], "IR") #

# # red = lista[0::,0]
# # ir = lista[0::,1]
# # __________________________________________________

# # Time array
# t = np.linspace(0, ((len(red) / fs)), len(red))
# print(f"PPG Signal contains {len(red)} samples and {len(red)/fs} sec. duration.")


# # In[]:
# # print(red_ppg[0:inc_10_sec])
# ### BUTTER
# for i in range(soma_freq,len(spo2),inc_10_sec):
#     soma_freq_t= i + inc_10_sec
#     passos = passos + 1
# # LOWPASS FILTER - Filtro inicial
#     cutoff_lp = 6  # desired cutoff frequency of the filter, Hz
#     red = red_ppg[i:soma_freq_t]
#     ir = ir_ppg[i:soma_freq_t]

#     LP_ORDER = 8
#     red_lp = butter_filter(red, cutoff_lp, fs, order=LP_ORDER)
#     ir_lp = butter_filter(ir, cutoff_lp, fs, order=LP_ORDER)



#     # # HIGHPASS SIGNAL
#     # cutoff_dc = 0.4
#     # HP_ORDER = 8
#     # red_hp = butter_filter(red_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")
#     # ir_hp = butter_filter(ir_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")

#     # BANDPASS SIGNAL - Filtro AC
#     cutoff_lac = 0.45
#     cutoff_hac = 5
#     BP_ORDER = 8
#     red_bac = butter_bandpass_filter(red_lp, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
#     ir_bac = butter_bandpass_filter(ir_lp, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)

#     # LOWPASS FILTER - Filtro DC
#     cutoff_ldc = 0.45  # desired cutoff frequency of the filter, Hz
#     LP_ORDER = 8
#     red_ldc = butter_filter(red_lp, cutoff_ldc, fs, order=LP_ORDER)
#     ir_ldc = butter_filter(ir_lp, cutoff_ldc, fs, order=LP_ORDER)


#     abs_ir_dc = np.abs(fftshift(fft(ir_ldc)))
#     abs_ir_dc = abs_ir_dc[:int(np.floor(len(abs_ir_dc)/2))]

#     abs_red_dc = np.abs(fftshift(fft(red_ldc)))
#     abs_red_dc = abs_red_dc[:int(np.floor(len(abs_red_dc)/2))]

#     abs_ir_ac = np.abs(fftshift(fft(ir_bac)))
#     abs_ir_ac = abs_ir_ac[:int(np.floor(len(abs_ir_ac)/2))]

#     abs_red_ac = np.abs(fftshift(fft(red_bac)))
#     abs_red_ac = abs_red_ac[:int(np.floor(len(abs_red_ac)/2))]

#     f = np.linspace(0, fs/2, num = len(abs_ir_dc))

#     # abs_ir_dc  = abs_ir_dc /np.max(abs_ir_dc)
#     abs_ir_dc = np.flipud(abs_ir_dc[0::])
#     abs_red_dc = np.flipud(abs_red_dc[0::])
#     abs_ir_ac = np.flipud(abs_ir_ac[0::])
#     abs_red_ac = np.flipud(abs_red_ac[0::])
#     if (abs_red_dc.all()>0 and abs_ir_dc.all()>0 and abs_ir_ac.all()>0):
#         r = ((max(abs_red_ac)/max(abs_red_dc))/(max(abs_ir_ac)/max(abs_ir_dc)))/2
#         r_windowed.append(r)

#     # if(passos == 77):
#         # if PLOTS:
#         #     plt.plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR filtrado (DC)")
#         #     plt.legend()
#         #     plt.show()
#         #     plt.figure()

#         # if PLOTS:
#         #     plt.plot(f,abs_red_dc,"b-", linewidth = 1, label = "FFT RED filtrado (DC)")
#         #     plt.legend()
#         #     plt.show()
#         #     plt.figure()

#         # if PLOTS:
#         #     plt.plot(f,abs_ir_ac,"b-", linewidth = 1, label = "FFT IR filtrado (AC)")
#         #     plt.legend()
#         #     plt.show()
#         #     plt.figure()

#         # if PLOTS:
#         #     plt.plot(f,abs_red_ac,"b-", linewidth = 1, label = "FFT RED filtrado (AC)")
#         #     plt.legend()
#         #     plt.show()
#         #     plt.figure()



# print("tamanho: ",len(r_windowed))
# r_win_array = np.array(r_windowed)
# print(mean(r_windowed))

# vi = 150
# vf = 165

# rx = r_win_array[vi:vf]
# sy = spo2_filtred [vi:vf]
# polynomial_features= PolynomialFeatures(degree=2)
# R_sm = polynomial_features.fit_transform(rx.reshape(-1, 1))
# # print(R_sm.shape())
# # R_sm = sm.add_constant(r_windowed)
# poly_reg_model = LinearRegression()
# print(len(R_sm))
# # results = sm.OLS(spo2_filtred, R_sm).fit()
# poly_reg_model.fit(R_sm,sy)
# # print(results.summary())
# y_predicted = poly_reg_model.predict(R_sm)
# # print(y_predicted)
# # print(poly_reg_model)


# t3 = np.linspace(0, (len(r_win_array)), len(r_win_array))

# # if PLOTS:
# #     plt.plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR filtrado (DC)")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # if PLOTS:
# #     plt.plot(f,abs_red_dc,"b-", linewidth = 1, label = "FFT RED filtrado (DC)")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # if PLOTS:
# #     plt.plot(f,abs_ir_ac,"b-", linewidth = 1, label = "FFT IR filtrado (AC)")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # if PLOTS:
# #     plt.plot(f,abs_red_ac,"b-", linewidth = 1, label = "FFT RED filtrado (AC)")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # if PLOTS:
# #     plt.plot(t, ir_bac, "k", label="PPG IR Filtro Passa Banda (AC)")
# #     plt.plot(t, red_bac, "r", label="PPG RED Filtro Passa Banda (AC)")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # if PLOTS:
# #     plt.plot(t, ir_ldc, "k", label="PPG IR Filtro Passa Baixa (DC)")
# #     plt.plot(t, red_ldc, "r", label="PPG RED Filtro Passa Baixa (DC)")
# #     plt.legend()
# #     plt.show()

# #y = a + bR + cR^2

# t2 = np.linspace(0, (len(sy) ), len(sy))
# print(mean(spo2_filtred))
# print("coeficientes: ",poly_reg_model.coef_ )
# print("constante: ",poly_reg_model.intercept_ )
# if PLOTS:
#     plt.plot(t2, sy,"k.", label="SpO2")
#     plt.plot(t2, rx,"r.", label="R")
#     plt.plot(t2, y_predicted,"b", label="SpO2 ajustado")
#     plt.legend()
#     plt.show()

# # if PLOTS:
# #     plt.plot(t, spo2_ver, "m.", label="SpO2 Faixa")
# #     plt.legend()
# #     plt.show()


# # print(len(r_windowed))
# # print(len(spo2_filtred))


# #Regressão linear para achar um A e B 

# # xfreqir_ac = [] 




# #SPO2 function

# A = 110
# B = -25


# spo2_final = A + B * r_win_array
# spo2_rl = 96 + (-23.2 * r_win_array) + (16 * r_win_array**2)

# t3 = np.linspace(0, (len(spo2_final) ), len(spo2_final))
# t4 = np.linspace(0, (len(spo2_rl) ), len(spo2_rl))



# plt.plot(t3, spo2_final, "m.", label="SpO2 110")
# plt.legend()
# plt.show()

# plt.plot(t4, spo2_rl, "m.", label="SpO2 RL")
# plt.legend()
# plt.show()


# # spo2_1 = (1000.0 - 550.0 * r_windowed[len(r)-1]) / (900.0 - 350.0 * r_windowed[len(r)-1]) * 100.0
# # spo2_2 = 10.0002 * (r_windowed[len(r)-1]**3) - 52.887 * (r_windowed[len(r)-1]**2) + 26.871 * r_windowed[len(r)-1] + 98.283

# # print((127 in spo2_ver), "%")
# # print(spo2_1, "%")
# # print(spo2_2, "%")

# # k = []
# # for i in range(len(r)):
# #     if ((r[i] <= 1.0) and r[i] >= 0):
# #         k.append(r[i])
# # k = np.array(k)
# # rfunction = lambda R: A + B * R 
# # spo2_final = rfunction(r)

# # print(spo2_rl)
# # print(spo2_filtred)


# print("FIM")


# print("FIM")



# # # plots sem correção
# # if PLOTS:

# #     plt.plot(t, ir_dc_cheby2, "k", label="PPG")
# #     plt.plot(time_p_ir, peak_ir, "r.", label="Threshold Peaks IR")
# #     plt.plot(time_v_ir, valley_ir, "g.", label="Threshold Valley IR")
# #     plt.plot(vec_media_peak_ir_time, vec_media_peak_ir,"y", label="Media peak IR")
# #     plt.plot(vec_media_valley_ir_time, vec_media_valley_ir,"m", label="Media valley IR")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()
    
# # if PLOTS:
# #     plt.plot(t, red_dc_cheby2, "k", label="PPG")
# #     plt.plot(time_p_red, peak_red, "r.", label="Threshold Peaks RED")
# #     plt.plot(time_v_red, valley_red, "g.", label="Threshold Valley RED")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()

# # # # plots com correção

# # if PLOTS:

# #     plt.plot(t, ir_dc_cheby2, "k", label="PPG")
# #     plt.plot(time_p_ir_c, peak_ir_c, "r.", label="Threshold Peaks IR C")
# #     plt.plot(time_v_ir_c, valley_ir_c, "g.", label="Threshold Valley IR C")
# #     plt.plot(vec_media_peak_time_ir_c, vec_media_peak_ir_c,"y", label="Media peak IR")
# #     plt.plot(vec_media_valley_time_ir_c, vec_media_valley_ir_c,"m", label="Media valley IR")
# #     plt.legend()
# #     plt.show()
# #     plt.figure()
    
# # if PLOTS:
# #     plt.plot(t, red_dc_cheby2, "k", label="PPG")
# #     plt.plot(time_p_red_c, peak_red_c, "r.", label="Threshold Peaks RED")
# #     plt.plot(time_v_red_c, valley_red_c, "g.", label="Threshold Valley RED")
# #     plt.plot(vecr = (max(abs_red_ac)/max(abs_red_dc))/(max(abs_ir_ac)/


# # normalização 

# # r_ir = (np.array(peak_ir) - np.array(valley_ir))/np.array(peak_ir)
# # r_red = (np.array(peak_red[1:]) - np.array(valley_red))/np.array(peak_red[1:])
# # r_ir = (np.array(peak_ir[1:]) - np.array(valley_ir))/np.array(peak_ir[1:])
# # r_ir = (np.array[1:peak_ir] - np.array[1:valley_ir])/np.array(peak_ir)



# # r = r_red[69:]/r_ir

# # k = []
# # for i in range(len(r)):
# #     if ((r[i] <= 1.0) and r[i] >= 0):
# #         k.append(r[i])

# # k = np.array(k)
# # print(r)
# # print(len(r_red))
# # print(len(r_ir))
# # print(type(peak_red[0]))
# # # print(type(peak_ir[0]))

# # #%%
# # #SPO2 function

# # A = 110
# # B = -25
# # rfunction = lambda R: A + B * R 

# # spo2_test = rfunction(k)

# # if PLOTS:
# #     plt.stem(spo2_test, label = 'SpO2')  # type: ignore
# #     plt.legend()
# #     plt.show()


    

# # # value_r = r[k]

# # a_ = 94 #110
# # b_ = -25
# # rfunction_ = lambda R_: a_ + b_ * R_

# # # spo2_ = rfunction_(value_r)

# # print(np.mean(spo2_test))
# # # print(np.mean(spo2_))
# # print("FIM")

# ## USB communication

# MAX_BUFF_LEN = 255
# SETUP 		 = False
# port 		 = None

# prev = time.time()
# while(not SETUP):
# 	try:
# 		# 					 Serial port(windows-->COM), baud rate, timeout msg
# 		port = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

# 	except: # Bad way of writing excepts (always know your errors)
# 		if(time.time() - prev > 2): # Don't spam with msg
# 			print("No serial detected, please plug your uController")
# 			prev = time.time()

# 	if(port is not None): # We're connected
# 		SETUP = True


# # read one char (default))

# # Write whole strings
# def write_ser(cmd):
# 	cmd = cmd 
# 	port.write(cmd.encode())

# # Super loop
# while(1):
#     cmd = (str(spo2_final[0])[0:10]) + '\n'# Blocking, there're solutions for this ;)
#     write_ser(cmd)






