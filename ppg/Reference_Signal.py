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
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift
from matplotlib import rc
from scipy.signal import find_peaks
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
import serial
import padasip as pa
from serial.tools.list_ports import comports
import signal_hw


# Plot Settings
rc("font", **{"size": 11})

banco_serial = 1 # 0 => testar dados do banco; 1 => testar dados da porta serial
sem_dedo = 0
plots = 0 # 0 => não plotar gráficos; 1 => plotar gráficos
umsegundo = False #um segundo de amostra

# In[]:
# Load Subject Data using helper function
from utils import SubjectPPGRecord

def write_ser(cmd):
        cmd = cmd + '\n'
        ser.write(cmd.encode())


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

if (banco_serial == 0):
    MAX_BUFF_LEN = 15
    SETUP 		 = False
    port 		 = None
    ZERO_PHASE_FILTERS = True
    PLOTS = True

    # DADOS DO BANCO DE SINAIS

    # Load Subject Data using helper function
    from utils import SubjectPPGRecord

    MAT_FILE = "Subject1"
    # MAT_FILE = "Subject3"
    # MAT_FILE = "Subject5"
    # MAT_FILE = "Subject6"
    # MAT_FILE = "Subject8"
    # MAT_FILE = "Subject14"

    sub = SubjectPPGRecord(MAT_FILE, db_path="ppg/", mat=True)
    rec = sub.record
    fs = sub.fs
    red_ppg = rec.red_a
    ir_ppg = rec.ir_a
    spo2 = rec.spo2
    inc_sec = 2560
    umsegundo = False

    #SUBJECT1
    inicio = 1925 *256 #Spo2 = 95%  Tempo = 93___OK

    #SUBJECT3
    # inicio = 370 *256 #Spo2 = 97%  Tempo = 93

    #SUBJECT5
    # inicio = 2400*256 #Spo2 = 97%  Tempo = 95 ___OK

    #SUBJECT6
    # inicio = 1615*256 # Spo2 = 94 - ______________OK 

    #SUBJECT8
    # inicio = 2315*256 # Spo2 = 96 - ____OK

    #SUBJECT14
    # inicio = 1380*256  # Spo2 = 89 - ______ok

    #DADOS DO HARDWARE

    # inc_sec = 2500 # 10 segundos 
    # inicio = 600
    # fs = 250

    # sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste1_18_01.txt",freq = fs, dois_ou_quatro = 0)
    # sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste_20_01.txt",freq = fs, dois_ou_quatro = 0)
    # red_ppg, ir_ppg, sred, sir = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste3_30_01.txt",freq = fs, dois_ou_quatro = 0)
    # red_ppg, ir_ppg, sred, sir= signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste4_02_02.txt",freq = fs, dois_ou_quatro = 0)
 
    # Time array
    # print(f"PPG Signal contains {len(red_ppg)} samples and {len(red_ppg)/fs} sec. duration.")

    red_bac = red_ppg [inicio: inicio+inc_sec]
    ir_bac = ir_ppg [inicio: inicio+inc_sec]
    
    # BANDPASS SIGNAL - Filtro AC
    cutoff_lac = 0.4
    cutoff_hac = 8
    BP_ORDER = 8

    rd = butter_bandpass_filter(red_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
    ir = butter_bandpass_filter(ir_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)


    t = np.linspace(0, ((len(rd) / fs)), len(rd))

    rd = rd/np.max(rd)
    ir = ir/np.max(ir)

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


    # calculo na frequencia

    frequencia = f4[np.where(abs_ir_fil ==np.max(abs_ir_fil))]
    BPM = frequencia*60


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
    
    print('O valor de SpO2 (f) é :', power_list.index(min(power_list)), '%')
    print('O valor de FC é:', int(BPM),'bpm')

    if(plots == 1):
        # Sinais normalizados

        plt.plot(t,rd, label= 'red norm')
        plt.plot(t, ir, label= 'ir norm')
        plt.legend()
        plt.show()

        # PLOTS FFT
    
        plt.subplot(2, 1, 1)
        plt.plot(f2,abs_ir_fil,"b", label = "FFT do sinal IR com filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(f4,abs_red_fil,"r", label = "FFT do sinal RED com filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.show()

        # PLOT sinal de referencia

        for i in range(len(RS_list)-1): 
            plt.plot(t, RS_list[i], label = i)
        plt.legend()
        plt.show()

        # Plots da Filtragem LMS

        plt.subplot(2, 1, 1)
        plt.plot(t, ir,"r", label = "Sinal original IR")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()

        t2 = np.linspace(0, ((len(ir) / fs)), len(ir)-(ordem-1))
        plt.subplot(2, 1, 2)
        plt.plot(t2,signal_recev_list[9],"b", label = "Sinal filtrado pelo LMS")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.show()

        for i in range(len(signal_recev_list)-1):
            plt.plot(t2, signal_recev_list[i], label= i)
        plt.plot(t, ir, label= 'IR')

        # plots da curva de potencia com o valor de SpO2

        plt.legend()
        plt.show()
        plt.plot(np.max(power_list)-power_list, label = "Curva de potência")
        plt.axvline(power_list.index(min(power_list)), color = 'red', linestyle = '--')
        plt.legend()
        plt.xlabel("SpO2(%)")
        plt.ylabel("Potência")
        plt.show()

else:  # SERIAL USB
   
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
        baudrate = 9600
        fs = int((500/baudrate)*1000)
        inicio = 0        
    

        if (umsegundo == False):
            inc_sec =int(np.ceil(fs* 10)) # 10 segundos 
        
            samples = inc_sec # 10sec for processing data 

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

            
            vec = vec[1:len(vec)-2]
            for i in range(len(vec)):
                red_usb,ir_usb = vec[i].split(",")
                red_ppg.append(float(red_usb))
                ir_ppg.append(float(ir_usb))

            for j in range(len(red_ppg)):
                if(red_ppg[j] <= 3.23):
                    sem_dedo = 1
                else:
                    sem_dedo = 0

            umsegundo = True

        else:
            # 1 segundos 
        
            samples = int(np.ceil(fs)) # 1sec for processing data 

            ser = serial.Serial(port,baudrate)
            ser.flushInput()

            line = 0
            vec = []
            while line < samples:
                data = str(ser.readline().decode("utf-8"))
                vec.append(data)
                line = line+1
        
            inicio = 0
            vec = vec[1:len(vec)-1]
            red_ppg = red_ppg[samples-2:]      
            ir_ppg = ir_ppg[samples-2:]
            print(len(vec))
            for i in range(len(vec)):
                red_usb,ir_usb = vec[i].split(",")
                red_ppg.append(float(red_usb))
                ir_ppg.append(float(ir_usb))

            for j in range(len(red_ppg)):
                if(red_ppg[j] <= 3.23):
                    sem_dedo = 1
                else:
                    sem_dedo = 0

        #igualando o tamanho dos vetores
        if(sem_dedo == 0):

            red_bac = red_ppg
            ir_bac = ir_ppg

            if(len(red_bac)>len(ir_bac)):
                red_bac = red_bac[0:len(ir_bac)]
            else:
                ir_bac = ir_bac[0:len(red_bac)]

            t = np.linspace(0, ((samples / fs)), len(red_bac))

            # BANDPASS SIGNAL - Filtro AC
            cutoff_lac = 0.4
            cutoff_hac = 8
            BP_ORDER = 8

            rd = butter_bandpass_filter(red_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
            ir = butter_bandpass_filter(ir_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)

            t = np.linspace(0, ((len(rd) / fs)), len(rd))
            rd = rd/np.max(rd)
            ir = ir/np.max(ir)

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


            # calculo na frequencia de FC

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

            print('O valor de SpO2 (f) é :', power_list.index(min(power_list)), '%')
            
            write_ser('SpO2: ' + str(int(power_list.index(min(power_list))))+ '%' + ' FC: '+ str(int(BPM))+'bpm')
        else:
            write_ser('            ' + 'Coloque   o dedo')
        ser.flushInput()
        ser.flushOutput()
    ser.close()