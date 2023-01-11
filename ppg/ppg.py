# In[]:
# from turtle import distance
from cv2 import threshold
import numpy as np
import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import find_peaks
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
from threshold_pev import adaptative_th, fix_pev, loc_pev

ZERO_PHASE_FILTERS = True
PLOTS = True

# In[]:
# Load Subject Data using helper function
from utils import SubjectPPGRecord

# MAT_FILE = "Subject6"
MAT_FILE = "Subject8"
# MAT_FILE = "Subject14"
sub = SubjectPPGRecord(MAT_FILE, db_path="ppg/", mat=True)
rec = sub.record
fs = sub.fs
red_ppg = rec.red_a
ir_ppg = rec.ir_a
spo2 = rec.spo2
inc_sec = 2560


#SUBJECT14

# inicio = 355072  # Spo2 = 89 - ok
# inicio = 470*256 # Spo2 = 95/96 -ok
# inicio = 1140*256 # Spo2 = 94 - não
# inicio = 1660*256 # Spo2 = 87
# inicio = 3024*256 # Spo2 = 91 - ok

#SUBJECT6

# inicio = 3171*256 # Spo2 = 94 - ok ORDEM = 150 E PASSO = 0,04s

#SUBJECT8

inicio = 2305*256 # Spo2 = 96 - OK
# inicio = 2680*256 # Spo2 = 98 - OK


# Time array

print(f"PPG Signal contains {len(red_ppg)} samples and {len(red_ppg)/fs} sec. duration.")


# In[]:

red = red_ppg[inicio:inicio+inc_sec]
ir = ir_ppg[inicio:inicio+inc_sec]

t = np.linspace(0, ((len(red) / fs)), len(red))


### BUTTER
# LOWPASS FILTER
cutoff_lp = 8  # desired cutoff frequency of the filter, Hz
LP_ORDER = 8
red_lp = butter_filter(red, cutoff_lp, fs, order=LP_ORDER)
ir_lp = butter_filter(ir, cutoff_lp, fs, order=LP_ORDER)

# HIGHPASS SIGNAL
cutoff_dc = 0.4
HP_ORDER = 8
red_hp = butter_filter(red_lp, cutoff_dc, fs, order=HP_ORDER, btype="high") 
ir_hp = butter_filter(ir_lp, cutoff_dc, fs, order=HP_ORDER, btype="high")

# BANDPASS SIGNAL
BP_ORDER = 8
red_bp = butter_bandpass_filter(red, cutoff_dc, cutoff_lp, fs, order=BP_ORDER)
ir_bp = butter_bandpass_filter(ir, cutoff_dc, cutoff_lp, fs, order=BP_ORDER)

# sinal red_bp e ir_bp filtrado pelo butter 

###CHEBY2

# LOWPASS FILTER
cutoff_lp2 = 6 # frequencia de corte
rs = 26

red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=LP_ORDER, rs = rs) # btype="low"
ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=LP_ORDER, rs = rs)

# HIGHPASS SIGNAL    # freq de corte pra remover sinal DC
HP_ORDER_CH = 6
red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")
ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")

# sinal red_dc_cheby2 e ir_dc_cheby2 finaç filtrado pelo cheby2



#funcao calcula os valores de limiares para que armazer os possiceis picos e os vales do sinal

def find_peak_valley_2(sample, time, threshold, order,gain):

    i = 1
    num_upsteps = 0
    possible_peak = False 
    possible_valley = False
    value_possible_peak = 0
    time_possible_peak = 0
    value_possible_valley = 0
    time_possible_valley = 0
    value_peak = []
    value_valley = []
    time_valley = []
    time_peak = []
    media_peak = 0 
    media_valley = 0 
    peak_vector =[0]*order
    valley_vector = [0]*order
    vec_media_peak = []
    vec_media_peak_time = []
    vec_media_valley = []
    vec_media_valley_time = []
    limite = 0.3
    
    peak_lock = False
    valley_lock = False

    while (i < len(sample)):
        if i>= 2:
            if (sample[i]>sample[i-1]):
                num_upsteps = num_upsteps + 1 
                if (possible_valley == False):
                    possible_valley = True
                    value_possible_valley = sample[i-1]
                    time_possible_valley = time[i-1]
            else:
                if (num_upsteps >= threshold):
                    possible_peak = True
                    value_possible_peak = sample[i-1]
                    time_possible_peak = time[i-1]
                else:
                    if (possible_valley == True):
                        if (sample[i] <= value_possible_valley): 
                            value_possible_valley = sample[i]
                            time_possible_valley = time[i]
                                                        
                    if ((possible_peak == True)):                        
                        if (sample[i-1]>value_possible_peak):
                            peak_vector[0] = sample[i-1]
                            media_peak = ((np.sum(peak_vector))/order)
                            peak_vector=np.roll(peak_vector,1)
                            vec_media_peak.append(media_peak*limite)
                            vec_media_peak_time.append(time[i-1])
                            if ((abs(media_peak*limite)<=abs(sample[i-1])) and (peak_lock == False)):
                                time_peak.append(time[i-1])
                                value_peak.append(sample[i-1])
                                peak_lock = True
                                valley_lock = False
                                
                        else:
                            peak_vector[0]=value_possible_peak
                            media_peak = ((np.sum(peak_vector))/order)
                            peak_vector=np.roll(peak_vector,1)
                            vec_media_peak.append(media_peak*limite)
                            vec_media_peak_time.append(time_possible_peak)
                            if ((abs(media_peak*limite)<=abs(value_possible_peak)) and (peak_lock == False)):
                                    time_peak.append(time_possible_peak)
                                    value_peak.append(value_possible_peak)
                                    peak_lock = True
                                    valley_lock = False
                                                                        
                        if ((possible_valley == True) ): 
                            valley_vector[0]=value_possible_valley
                            media_valley = ((np.sum(valley_vector))/order)
                            valley_vector= np.roll(valley_vector,1) 
                            vec_media_valley.append(media_valley*limite)
                            vec_media_valley_time.append(time_possible_valley)
                            if (((media_valley*limite)>=(value_possible_valley)) and (valley_lock == False)):
                                time_valley.append(time_possible_valley)
                                value_valley.append(value_possible_valley)
                                possible_valley = False 
                                valley_lock = True
                                peak_lock = False
                                                                                                          
                        possible_peak = False
                num_upsteps = 0
               
        i = i + 1
                
    threshold = 0.6 * num_upsteps
    return value_peak, value_valley, time_peak, time_valley,vec_media_peak,\
           vec_media_valley,vec_media_peak_time,vec_media_valley_time

vec_media_peak_ir = []
vec_media_valley_ir  = []
vec_media_peak_red = []
vec_media_valley_red  = []
vec_media_peak_red_time = []
vec_media_valley_red_time  = []
vec_media_peak_ir_time = []
vec_media_valley_ir_time  = []

avg_order = 5
threshold_peak_valley = 25


peak_ir,valley_ir, time_p_ir,time_v_ir,vec_media_peak_ir,\
vec_media_valley_ir,vec_media_peak_ir_time,\
vec_media_valley_ir_time = find_peak_valley_2(ir_dc_cheby2, t, threshold_peak_valley,avg_order,2)


peak_red,valley_red, time_p_red,time_v_red,vec_media_peak_red,\
vec_media_valley_red,vec_media_peak_red_time,\
vec_media_valley_red_time = find_peak_valley_2(red_dc_cheby2, t, threshold_peak_valley, avg_order,2)




#PLOT IR com sinal vindo do filtro Cheby2, limiar e pontos de Picos e vales encontrados 
plt.plot(t, ir_dc_cheby2, "k", label="PPG Cheby2")
plt.plot(time_p_ir, peak_ir, "r.", label="Threshold Peaks IR")
plt.plot(time_v_ir, valley_ir, "g.", label="Threshold Valley IR")
plt.plot(vec_media_peak_ir_time, vec_media_peak_ir,"y", label="Media peak IR")
plt.plot(vec_media_valley_ir_time, vec_media_valley_ir,"m", label="Media valley IR")
plt.legend()
plt.show()

#PLOT RED com sinal vindo do filtro Cheby2, limiar e pontos de Picos e vales encontrados
    
plt.plot(t, red_dc_cheby2, "k", label="Sinal PPG ")
plt.plot(time_p_red, peak_red, "b.", label="Picos e vales")
plt.plot(time_v_red, valley_red, "b.")#, label="Vales")
plt.plot(vec_media_peak_red_time, vec_media_peak_red,"r", label="Limiar de picos e vales")
plt.plot(vec_media_valley_red_time, vec_media_valley_red,"r")#, label="Limiar de vales")
plt.legend()
plt.show()



#igualando os valores de picos e vales para o cálculo de R

if((len(peak_ir)>len(valley_ir))):
    r_ir = np.array(peak_ir[0:len(valley_ir)])-np.array(valley_ir)/np.array(peak_ir[0:len(valley_ir)])
else:
    r_ir = np.array(peak_ir)-np.array(valley_ir[0:len(peak_ir)])/np.array(peak_ir)

if((len(peak_red)>len(valley_red))):
    r_red = np.array(peak_red[0:len(valley_red)])-np.array(valley_red)/np.array(peak_red[0:len(valley_red)])
else:
    r_red = np.array(peak_red)-np.array(valley_red[0:len(peak_red)])/np.array(peak_red)

# Calcular o valor R 

if(len(r_ir) > len(r_red)):
    R = r_red/r_ir[0:len(r_red)]
else:
    R = r_red[0:len(r_ir)]/r_ir


# # Calculando R

# r_ir = np.array(peak_ir)-np.array(valley_ir)/np.array(peak_ir)
# r_red = np.array(peak_red)-np.array(valley_red)/np.array(peak_red)

# R = r_red/r_ir


#SPO2 function

A = 110
B = -25
rfunction = lambda R: A + B * R 

SPO2 = rfunction(R)

SPO2_media = numpy.mean(SPO2)

print('O valor de SpO2 é :', np.int(SPO2_media))

