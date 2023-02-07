#%%
import numpy as np
from numpy.fft import fft, fftshift,rfft
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import find_peaks, butter, lfilter, filtfilt
import signal_hw


#%%

sr_dados_red = []
sr_dados_ir = []
dados_red = []
dados_ir = []
conteudo = []

fs = 512    #frequencia de amostragem do sinal



# file1 = open('novo.txt','r')
# for linha in file1:
#     conteudo.append(linha.rstrip())
# file1.close()

# for i in range(len(conteudo)):
#     sr_valor_red, sr_valor_ir, valor_red, valor_ir = conteudo[i].split(",")
#     sr_dados_red.append(sr_valor_red)
#     sr_dados_ir.append(sr_valor_ir)
#     dados_red.append(valor_red)
#     dados_ir.append(valor_ir)



# for i in range(len(conteudo)):
#     sr_dados_red[i] = float(sr_dados_red[i])
#     sr_dados_ir[i] = float(sr_dados_ir[i])
#     dados_red[i] =float(dados_red[i])
#     dados_ir[i] = float(dados_ir[i])

sred, sir, redppg, irppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/weliton.txt",freq = fs)

# sred = sr_dados_red
# sir = sr_dados_ir

# redppg = dados_red
# irppg = dados_ir

#%%

t = np.linspace(0, len(redppg) / fs, len(redppg))
# print(len(t))

# Plotando um trecho do sinal

Tppg = 100
fig, ax = plt.subplots()
plt.plot(redppg[100: Tppg * fs],"r-", linewidth=1, label="PPG sem RED")
plt.plot(irppg[100: Tppg * fs],"b-", linewidth=1, label="PPG sem IR")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Time [samples]")
plt.legend()
plt.grid()
#%%

t = np.linspace(0, len(sred) / fs, len(sred))
# print(len(t))

# Plotando um trecho do sinal

Tppg = 100
fig, ax = plt.subplots()
plt.plot(sred[100: Tppg * fs],"r-", linewidth=1, label="PPG RED com REMOÇÃO LUZ")
plt.plot(sir[100: Tppg * fs],"b-", linewidth=1, label="PPG IR com REMOÇÃO LUZ")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Time [samples]")
plt.legend()
plt.grid()



#%%

ir_or = np.abs(fftshift(fft(sir)))

abs_ir_or = ir_or[:int(np.floor(len(ir_or)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_or))

abs_ir_dc  = abs_ir_or /np.max(abs_ir_or)
abs_ir_dc = np.flipud(abs_ir_dc[0::])


# fig, ax = plt.subplots()
# plt.plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR com remoção")
# plt.ylabel("Amplitude [mV]")
# plt.xlabel("Frequencia [Hz]")
# plt.legend()


#%%

c_ir_or = np.abs(fftshift(fft(irppg)))

abs_ir_cor = c_ir_or[:int(np.floor(len(c_ir_or)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_cor))

abs_ir_cr  = abs_ir_cor /np.max(abs_ir_cor)
abs_ir_cr = np.flipud(abs_ir_cr[0::])

#AQUI
fig, ax = plt.subplots()
plt.plot(f,abs_ir_cr,"b-", linewidth = 1, label = "FFT do sinal IR sem filtragem")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.legend()







#%%
# Filtrando o sinal PPG
# Validacao do teorema de Nyquiest
def butter_design(cutoff, fs, order, btype="low"):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    z, p, k = butter(order, normal_cutoff, btype=btype, analog=False, output='zpk')
    print(k)
    return b, a


def butter_filter(data, cutoff, fs, order, btype="low"):
    b, a = butter_design(cutoff, fs, btype=btype, order=order)
    y = filtfilt(b, a, data)  # zero-phase
    return y

# FILTRO PASSA-BAIXA
cutoff_lp = 8  # frequencia de corte

red_lp = butter_filter(redppg, cutoff_lp, fs, order=4) # btype="low"
ir_lp = butter_filter(irppg, cutoff_lp, fs, order=4)

# FILTRO PASSA-ALTA
cutoff_dc = 0.2    # freq de corte pra remover sinal DC
red_dc = butter_filter(red_lp, cutoff_dc, fs, order=4, btype="high")
ir_dc = butter_filter(ir_lp, cutoff_dc, fs, order=4, btype="high")

#PLOT REMOCAO DC
fig, ax = plt.subplots(2)
start = 0
end = 500
ax[1].plot(
    t[start * fs : end * fs], ir_dc[start * fs : end * fs], "b-", linewidth=1, label="filtered ir"
)

ax[0].plot(
    t[start * fs : end * fs], red_dc[start * fs : end * fs], "r-", linewidth=1, label="filtered red"
)
plt.xlabel("Time [sec]")
ax[1].grid()
ax[1].legend()
ax[0].grid()
ax[0].legend()

#PLOT SEM REMOCAO DC

fig, ax = plt.subplots()
plt.plot(t[start * fs : end * fs], ir_lp[start * fs : end * fs], "b-", linewidth=1, label="filtered ir")
plt.plot(t[start * fs : end * fs], red_lp[start * fs : end * fs], "r-", linewidth=1, label="filtered red")
plt.xlabel("Time [sec]")
plt.legend()
plt.grid()
#%%

def fftsegm(data,duration,dataoverlap):
    n_segms = int(np.ceil((len(data)-dataoverlap)/(duration-dataoverlap)/3))

    idx = range(0,len(data),(duration-int(dataoverlap)))

    tempBuf = [data[i:i+duration] for i in idx]
    # tempBuf = tempBuf[:int(np.floor(len(tempBuf)/2))]
    
    tempBuf[n_segms-1] = np.pad(tempBuf[n_segms-1],
        (0,duration-tempBuf[n_segms-1].shape[0]),'constant')
    tempBuf2 = np.vstack(tempBuf[0:n_segms]) #sequencia

    return tempBuf2
# np.abs(fftshift(fft
windowLen = 2              # in seconds
tempo_desloc = 1           # shift len of next window in 1 second
duration = int(windowLen*fs)
dataoverlap = (windowLen-tempo_desloc)*fs

print(dataoverlap)

data = ir_dc

ir_fftseg = fftsegm(data,duration,dataoverlap)

plt.figure
plt.plot(ir_dc)
plt.show()
plt.plot(ir_fftseg[0])
plt.plot(ir_fftseg[1])

#%%

def fftsegm(data,duration,dataoverlap):
    n_segms = int(np.ceil((len(data)-dataoverlap)/(duration-dataoverlap)/3))

    idx = range(0,len(data),(duration-int(dataoverlap)))

    tempBuf = [np.abs(fftshift(fft(data[i:i+duration]))) for i in idx]
    # tempBuf = tempBuf[:int(np.floor(len(tempBuf)/2))]
    
    tempBuf[n_segms-1] = np.pad(tempBuf[n_segms-1],
        (0,duration-tempBuf[n_segms-1].shape[0]),'constant')
    tempBuf2 = np.vstack(tempBuf[0:n_segms]) #sequencia

    return tempBuf2
# np.abs(fftshift(fft
windowLen = 2              # in seconds
tempo_desloc = 1           # shift len of next window in 1 second
duration = int(windowLen*fs)
dataoverlap = (windowLen-tempo_desloc)*fs

print(dataoverlap)

data = ir_dc

ir_fftseg = fftsegm(data,duration,dataoverlap)
fir = np.linspace(0, fs/2, num = len(ir_fftseg))

abs_ir_dc = np.abs(fftshift(fft(ir_dc)))
abs_ir_dc = abs_ir_dc[:int(np.floor(len(abs_ir_dc)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_dc))

abs_ir_dc  = abs_ir_dc /np.max(abs_ir_dc)
abs_ir_dc = np.flipud(abs_ir_dc[0::])

fig, ax = plt.subplots(2)
ax[0].plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR filtrado")
ax[0].grid()
ax[1].plot(ir_fftseg[0],"r-", linewidth = 1, label = "FFT IR filtrado")
ax[1].grid()

#%%

windowLen = 2              # in seconds
tempo_desloc = 1           # shift len of next window in 1 second
duration = int(windowLen*fs)
dataoverlap = (windowLen-tempo_desloc)*fs

print(dataoverlap)

data = ir_dc



n_segms = int(np.ceil((len(data)-dataoverlap)/(duration-dataoverlap)/3))

idx = range(0,len(data),(duration-int(dataoverlap)))

tempBuf = [np.abs(fftshift(fft(data[i:i+duration]))) for i in idx]
tempBuf = tempBuf[:int(np.floor(len(tempBuf)/2))]
 
tempBuf[n_segms-1] = np.pad(tempBuf[n_segms-1],
    (0,duration-tempBuf[n_segms-1].shape[0]),'constant')
tempBuf2 = np.vstack(tempBuf[0:n_segms])

# f = np.linspace(0, fs/2, num = len(abs_ir_or))

#--------- FILTRADO -------------
# -------- IR --------
abs_ir_dc = np.abs(fftshift(fft(ir_dc)))
abs_ir_dc = abs_ir_dc[:int(np.floor(len(abs_ir_dc)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_dc))

abs_ir_dc  = abs_ir_dc /np.max(abs_ir_dc)
abs_ir_dc = np.flipud(abs_ir_dc[0::])

fig, ax = plt.subplots()
plt.plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FFT IR filtrado")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.grid()

# tempBuf = []
# for i in idx:
#     tempBuf.append(np.abs(fftshift(fft(data[i:i+duration]))))

# temp1 = []
# for i in range(len(tempBuf)):
#     temp[i] = tempBuf[i]
#     temp = temp[:int(np.floor(len(temp)/2))]
#     temp1.append(temp)



#%%

#raise SystemExit(0) #HACK: DEBUG

#%%

duration=int(4*fs)

n_segms = int(np.ceil(2*(len(ir_dc)/duration)-1))

winlen = 512
stepsize = winlen // 2

idxlist = range(0, len(ir_dc)-winlen, stepsize)
C = np.zeros((len(idxlist), winlen))

for o, i in enumerate(idxlist):
    C[o, :] = ir_dc[i:i+winlen]

plt.plot(C[13])
plt.show()
plt.plot(C[14])
plt.show()
plt.plot(ir_dc)


#%%

#Definir o deltaT

deltaT = round(2 * fs)

# Recorte do sinal em deltaT
n_win = np.ceil(len(ir_dc) / deltaT).astype('int')
pad_length = int(n_win * deltaT - len(ir_dc))
ir_s = np.append(ir_dc, np.zeros([pad_length]))

# Recorte do sinal em deltaT usando ordem F (Matlab/Fortran)
crop_ir_s = ir_s.reshape((deltaT, -1), order='F')

# Matriz para armazenar FFT de cada janela
psd_ir_s = np.zeros(crop_ir_s.shape)
# PSD
mean_psd = np.zeros([deltaT])
for i in range(n_win):
    col = crop_ir_s[:, i]
    t_f = fftshift(fft(col, deltaT))
    #tf = 10 * np.log10(np.abs(t_f)**2)
    # mean_psd += t_f
    #psd_ir_s[:, i] = tf
# Calculamos o valor médio (cortamos as freqs negativas)
mean_psd = fftshift(mean_psd / n_win)
mean_psd = mean_psd[:int(np.floor(len(mean_psd) / 2))]
freqs = np.linspace(0, fs / 2, num=len(mean_psd))




#%%





# ------FFT do sinal original e filtrado------

#--------- ORIGINAL -------------
# -------- IR --------
abs_ir_or = np.abs(fftshift(fft(irppg)))
abs_ir_or = abs_ir_or[:int(np.floor(len(abs_ir_or)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_or))

abs_ir_or  = abs_ir_or /np.max(abs_ir_or)
abs_ir_or = np.flipud(abs_ir_or[0::])

#
fig, ax = plt.subplots()
plt.plot(f,abs_ir_or, "b-", linewidth = 1, label = "FTT IR original")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.grid()

# -------- RED --------
abs_red_or = np.abs(fftshift(fft(redppg)))
abs_red_or = abs_red_or[:int(np.floor(len(abs_red_or)/2))]

abs_red_or  = abs_red_or /np.max(abs_red_or)
abs_red_or = np.flipud(abs_ir_or[0::])
#
fig, ax = plt.subplots()
plt.plot(f,abs_ir_or,"r-", linewidth = 1, label = "FTT RED original")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.grid()

#--------- FILTRADO -------------
# -------- IR --------
abs_ir_dc = np.abs(fftshift(fft(ir_dc)))
abs_ir_dc = abs_ir_dc[:int(np.floor(len(abs_ir_dc)/2))]

f = np.linspace(0, fs/2, num = len(abs_ir_dc))

abs_ir_dc  = abs_ir_dc /np.max(abs_ir_dc)
abs_ir_dc = np.flipud(abs_ir_dc[0::])
#
fig, ax = plt.subplots()
plt.plot(f,abs_ir_dc,"b-", linewidth = 1, label = "FTT IR filtrado")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.grid()


# -------- RED --------
abs_red_dc = np.abs(fftshift(fft(red_dc)))
abs_red_dc = abs_red_dc[:int(np.floor(len(abs_red_dc)/2))]

abs_red_dc  = abs_red_dc /np.max(abs_red_dc)
abs_red_dc = np.flipud(abs_red_dc[0::])
#
fig, ax = plt.subplots()
plt.plot(f,abs_red_dc,"r-", linewidth = 1, label = "FTT RED filtrado")
plt.ylabel("Amplitude [mV]")
plt.xlabel("Frequencia [Hz]")
plt.grid()
plt.show()

#%%

list_pks = [-0.17780045, -0.46764584, 0.59981918, 0.60001265, -0.20320578, 0.23450197]

l = []


for pos in range(len(list_pks)-1): #and not found and not stop
    pos =pos+1
    l.append(list_pks[0])
    
    if list_pks[pos] < 0:
        if list_pks[pos] < list_pks[pos-1]:
            list_pks[pos] = 0.0
            l.append(list_pks[pos])
            pos = pos+1 #do something to remove the index of the atual position
        else: 
            l.append(list_pks[pos])
            pos = pos + 1
    elif list_pks[pos] > 0: 
        if list_pks[pos] < list_pks[pos-1]:
            l.append(list_pks[pos])
            pos = pos+1        
        else:
            list_pks[pos] = 0.0 #do something to remove the index of the atual position             
            l.append(list_pks[pos])
            pos = pos+1            
print(l)
# %%
ppg = ir_dc_cheby2
l_n = []
for i in range(len(ppg)):
    if ppg[i] < 0:
        a = ppg[i]
        l_n.append(a)

l_neg = np.array(l_n)
# Initial Parameters
WIN_DURATION = 900e-3
OFF_WIN_DURATION = 250e-3
MASK_WIN_DURATION = 120e-3

# Threshold Functions Valley
updateThresholdn = (
    lambda th_prev, idx: (th_prev + np.min(l_neg[idx]) + (np.max(l_neg[idx]) - np.min(l_neg[idx])) / 2.8) / 2
)
offThresholdn = lambda th: 0.25 * th

win = np.arange(fs)  # first search window
win_step = int(WIN_DURATION * fs)  # window step
force_step = False

# Threshold Lists
lthn = [np.min(l_neg[win]) + (np.min(l_neg[win]) + np.max(l_neg[win])) / 3.4]  # threshold list
lnnn = [0]
lnnn_off = [0]
lnnn_th = [lthn[-1]]

while True:
    # Update Window for next search
    win_start = lnnn[-1] + win_step + int(force_step) * win_step
    force_step = False
    win = np.arange(win_start, win_start + win_step)

    # Adjust Window
    if win[-1] > len(l_neg):
        win = win[win < len(l_neg)]
    # Break (no more samples) :)
    if not win.size:
        break

    # Update Threshold
    lthn.append(updateThreshold(lthn[-1], win))

    # Find next Burst
    lnn = l_neg[win]
    lnnn[lnn <= lthn[-1]] = 0
    lnnn = np.where(lnn)[0]
    # Short Time Rejection
    if lnnn[-1] != 0:  # only mask if this is not the first iteration...
        lnnn = lnnn[lnnn > int(MASK_WIN_DURATION * fs)]
    # Nothing above the threshold
    if not lnnn.size:  # force a step and continue ...
        force_step = True
        continue
    # Append to list of Coarse Burst Limits
    lnnn.append(lnn[0] + win[0] - 1)
    # Append to Threshold Values
    lnnn_th.append(lthn[-1])

    # Find burst end
    if lnnn[-2] != 0:  # excluded first iteration
        #  lnn[-2] bc lnn was updated 6 lines above
        win_off = win - win_step  # take a step back
        win_off = np.arange(win_off[0], win_off[0] + int(OFF_WIN_DURATION * fs))  # crop window
        win_off = win_off[win_off < len(l_neg)]  # crop on borders
        # Locate Burst End
        lnn_off = l_neg[win_off]
        lnn_off[lnn_off <= offThreshold(lthn[-1])] = 0
        lnn_off = np.where(lnn_off)[0]
        # Append to List
        try:
            lnnn_off.append(win_off[0] + ln_off[-1] - 1)
        except IndexError:
            # print('invalid ln_off!')
            pass

# Remove first element in threshold indexes
lnnn = np.array(lnnn[1:])
lnnn_off = np.array(lnnn_off[1:])
lnnn_th = np.array(lnnn_th[1:])


#%%

lthn = [np.min(l_neg[win]) + (np.min(l_neg[win]) + np.max(l_neg[win])) / 3.4]  # threshold list
lnnn = [0]
lnnn_off = [0]
lnnn_th = [lthn[-1]]

while True:
    # Update Window for next search
    win_start = lnnn[-1] + win_step + int(force_step) * win_step
    force_step = False
    win = np.arange(win_start, win_start + win_step)

    # Adjust Window
    if win[-1] > len(l_neg):
        win = win[win < len(l_neg)]
    # Break (no more samples) :)
    if not win.size:
        break

    # Update Threshold
    lthn.append(updateThresholdn(lthn[-1], win))

    # Find next Burst
    lnn = l_neg[win]
    lnn[lnn <= lthn[-1]] = 0
    lnn = np.where(lnn)[0]
    # Short Time Rejection
    if lnnn[-1] != 0:  # only mask if this is not the first iteration...
        lnnn = lnnn[lnnn > int(MASK_WIN_DURATION * fs)]
    # Nothing above the threshold
    if not lnn.size:  # force a step and continue ...
        force_step = True
        continue
    # Append to list of Coarse Burst Limits
    lnnn.append(lnn[0] + win[0] - 1)
    # Append to Threshold Values
    lnnn_th.append(lthn[-1])

    # Find burst end
    if lnnn[-2] != 0:  # excluded first iteration
        #  lnn[-2] bc lnn was updated 6 lines above
        win_off = win - win_step  # take a step back
        win_off = np.arange(win_off[0], win_off[0] + int(OFF_WIN_DURATION * fs))  # crop window
        win_off = win_off[win_off < len(l_neg)]  # crop on borders
        # Locate Burst End
        lnn_off = l_neg[win_off]
        lnn_off[lnn_off <= offThresholdn(lthn[-1])] = 0
        lnn_off = np.where(lnn_off)[0]
        # Append to List
        try:
            lnnn_off.append(win_off[0] + lnn_off[-1] - 1)
        except IndexError:
            # print('invalid ln_off!')
            pass

# Remove first element in threshold indexes
lnnn = np.array(lnnn[1:])
lnnn_off = np.array(lnnn_off[1:])
lnnn_th = np.array(lnnn_th[1:])

