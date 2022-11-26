# In[]:
import numpy as np
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import find_peaks
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter

ZERO_PHASE_FILTERS = True
PLOTS = True
# In[]:
# Plot Settings
rc("font", **{"size": 18})
plt.close("all")
plt.ion()


# In[]:
# Load Subject Data using helper function
from utils import SubjectPPGRecord

MAT_FILE = "Subject14"
sub = SubjectPPGRecord(MAT_FILE, db_path=".", mat=True)
rec = sub.record
fs = sub.fs
red = rec.red_a
ir = rec.ir_a
spo2 = rec.spo2

# Time array
t = np.linspace(0, ((len(red) / fs)), len(red))
print(f"PPG Signal contains {len(red)} samples and {len(red)/fs} sec. duration.")


# In[]:
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


###CHEBY2

# LOWPASS FILTER
cutoff_lp2 = 8  # frequencia de corte
rs = 26

red_lp_cheby2 = cheby2_filter(red, cutoff_lp, fs, order=LP_ORDER, rs = rs) # btype="low"
ir_lp_cheby2 = cheby2_filter(ir, cutoff_lp, fs, order=LP_ORDER, rs = rs)

# HIGHPASS SIGNAL    # freq de corte pra remover sinal DC
HP_ORDER_CH = 8
red_dc_cheby2 = cheby2_filter(red_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")
ir_dc_cheby2 = cheby2_filter(ir_lp_cheby2, cutoff_dc, fs, order=HP_ORDER_CH, rs = rs, btype="high")

#%%
# Split Signal by Stages

# warmup = [:300*fs]
# aerob = [300*fs:720*fs]
# rec_at = [720*fs:1080*fs]
# anaer1 = [1380*fs:1470*fs]
# rec_at1 = [1470*fs:1530*fs]
# anaer2 = [1530*fs:1620*fs]
# rec_at2 = [1620*fs:1680*fs]
# anaer3 = [1680*fs:1770*fs]
# rec_at3 = [1770*fs:1830*fs]
#cool_down = [1830*fs:]
#(((1.5*60 + 1*60)*3 + 5*60)) , 5*60

stages_duration = np.array([5*60, 12*60, 6*60, (((1.5*60 + 1*60)*3)+5*60), (((1.5*60 + 1*60)*3)+3*60), 5*60], dtype=np.int64) * fs
stages_duration_sum = stages_duration.cumsum()

def split_by_stage(sig):
    sig_stage = []
    prev_duration = 0
    for duration in stages_duration_sum:
        sig_stage.append(sig[prev_duration:duration])
        prev_duration = duration
        if duration == stages_duration_sum[-1]:
            sig_stage.append(sig[duration:])
    return sig_stage

ir_st = split_by_stage(ir_dc_cheby2)
red_st = split_by_stage(red_dc_cheby2)
time = split_by_stage(t)


# # Time array
# t = np.linspace(0, ((len(red) / fs)), len(red))
# t = np.linspace(0, stages_duration_sum[-1] / fs, stages_duration_sum[-1])

#%%
# LMS ADAPTATIVE FILTER
#signal_corrupted = sinal de referencia 
#signal = sinal original (IR) sem filtro

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

    mse = np.zeros(I,)

    for i in range(I):
        # u = Auxiliar variable. Store the piece of the corruptedInput
        # that will be multiplied by the current set of coefficients.
        u = np.flipud(signal_corrupted[i:i + order])[:, np.newaxis]
        signal_approx[i] = w[:, [i - 1]].T @ u

        error[i] = signal[i + order - 1] - signal_approx[i]
        mse[i] = np.absolute(error[i]) ** 2 / I

        # Updating the filter coeficients
        w[:, [i]] = w[:, [i - 1]] + step * u * (error[i])
        signal_approx[i] = w[:, [i]].T @ u
    
    return signal_approx, mse

#%%

step = 0.05
order = 2
lenght = 2048

MSE = np.zeros(lenght - order + 1)
signal_recev = np.zeros(lenght - order + 1)

signal_recev, mse = lms(ir[0*fs:8*fs], ir_lp_cheby2[0*fs:8*fs], step, order)


#%%

#reference signal RS = r'*IRsignal - REDsignal 



#%%
# COMENTÁRIOS DO PROFESSOR SOBRE A FFT
# janela deslizante => tem que descartar as amostras 
# overlap de metade da janela = percentual de overlap de 50%....99%???
# não ter efeito de borda
# fazer 512 e faz a transformada, a próxima janela de tamanho 512 
# joga fora os primeiros 256, desloca a  origem pro 257 e enche a nova janela da metade pra frente. 
# overlap, deslizar a janela entre transformadas
# amostras de 0 511 estima a saturação a próxima joga fora do 0 a 255 e pega a 256 a 511+767 


# shiftLen = tempo de deslocamento 
# dataOverlap = calcula o número de amostras de deslocamento 
#               com base na taxa de amostragem

#%%
#TESTE 1 

#Recorte do sinal + fft 

winlen = fs*8
stepsize = winlen // 2

ir = ir_st[0]

idxlist = range(0, len(ir)-winlen, stepsize)
C = np.zeros((len(idxlist), winlen))
tf = []
flip_sig = []
crop_sig = []

for o, i in enumerate(idxlist):
    C[o, :] = ir[i:i+winlen]
    tf.append(np.abs(fftshift(fft(C[o]))))
    crop_sig.append(tf[o][:int(np.floor(len(tf[o])/2))])
    flip_sig.append(np.flipud(crop_sig[o][0::]))


#%%
##TESTE 2 

def fftsegm(data,duration,dataoverlap):
    
    n_segms = int(np.ceil((len(data)-dataoverlap)/(duration-dataoverlap)))

    idx = range(0,len(data),(duration-int(dataoverlap)))

    signal_crop = [np.abs(fftshift(fft(data[i:i+duration]))) for i in idx]

    ## acrescentar vários 0 no final do sinal para finalizar a janela no último segm
    signal_crop[n_segms-1] = np.pad(signal_crop[n_segms-1],
        (0,duration-signal_crop[n_segms-1].shape[0]),'constant')
    crop = np.vstack(signal_crop[0:n_segms]) #sequencia para um lista em vez de vários arrays

    crop_sig = []
    flip_sig = []
    for c,i in enumerate(crop):
        crop_sig.append(crop[c][:int(np.floor(len(crop[c])/2))])
        flip_sig.append(np.flipud(crop_sig[c][0::]))

    return flip_sig

windowLen = 20             
tempo_desloc = 18           # shift len of next window 
duration = int(windowLen*fs)
dataoverlap = (windowLen-tempo_desloc)*fs

print(dataoverlap)

data = ir_st[0]

ir_fftseg = fftsegm(data,duration,dataoverlap)



#%%
#TESTE para conferir se a fft tá certa

ir_fftseg2 = ir_fftseg[0][:int(np.floor(len(ir_fftseg[0])/2))]
fir = np.linspace(0, fs/2, num = len(ir_fftseg))

abs_ir_dc = np.abs(fftshift(fft(ir_st[0])))
abs_ir_dc = abs_ir_dc[:int(np.floor(len(abs_ir_dc)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_dc))

abs_ir_dc  = abs_ir_dc /np.max(abs_ir_dc)
abs_ir_dc = np.flipud(abs_ir_dc[0::])

fig, ax = plt.subplots(2)
ax[0].plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR filtrado")
ax[0].grid()
ax[1].plot(ir_fftseg2,"r-", linewidth = 1, label = "FFT IR filtrado")
ax[1].grid()


#%%

#Recorte do sinal + fft e psd
#Definir o deltaT

deltaT = round(4 * fs)

# Recorte do sinal em deltaT
n_win = np.ceil(len(ir_dc_cheby2) / deltaT).astype('int')
pad_length = int(n_win * deltaT - len(ir_dc_cheby2))
ir_s = np.append(ir_dc_cheby2, np.zeros([pad_length]))

# Recorte do sinal em deltaT usando ordem F (Matlab/Fortran)
crop_ir_s = ir_s.reshape((deltaT, -1), order='F')

# Matriz para armazenar FFT de cada janela
psd_ir_s = np.zeros(crop_ir_s.shape)
# PSD = densidade espectral de potência
mean_psd = np.zeros([deltaT], dtype='complex')
for i in range(n_win):
    col = crop_ir_s[:, i]
    t_f = fftshift(fft(col, deltaT))
    tf = (np.abs(t_f)**2)
    mean_psd += t_f
    psd_ir_s[:, i] = tf
# Calculamos o valor médio (cortamos as freqs negativas) dividindo pela duração
mean_psd = fftshift(mean_psd / n_win)
mean_psd = mean_psd[:int(np.floor(len(mean_psd) / 2))]
freqs = np.linspace(0, fs / 2, num=len(mean_psd))