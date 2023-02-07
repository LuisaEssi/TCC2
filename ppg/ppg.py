import numpy as np
import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
from filterbanks import cheby2_filter, cheby2_bandpass_filter, butter_filter, butter_bandpass_filter
from serial.tools.list_ports import comports
from utils import SubjectPPGRecord
import serial
import signal_hw # para teste de banco (HW)

banco_serial = 1 # 0 => testar dados do banco; 1 => testar dados da porta serial
sem_dedo = 0
plots = 0 # 0 => não plotar gráficos; 1 => plotar gráficos
umsegundo = False
inc_sec = 0 
inicio = 0 
fs = 0
t = 0

#FUNCOES

def write_ser(cmd):
    cmd = cmd + '\n'
    ser.write(cmd.encode())

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
        limite = 0.4
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

if (banco_serial == 0):
    MAX_BUFF_LEN = 15
    SETUP 		 = False
    port 		 = None
    ZERO_PHASE_FILTERS = True
    PLOTS = True

    # DADOS DO BANCO DE SINAIS

    # Load Subject Data using helper function
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

    # sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste1_18_01.txt",freq = fs, dois_ou_quatro = 0) #
    # sred, sir, red_ppg, ir_ppg = signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste_20_01.txt",freq = fs, dois_ou_quatro = 0) #
    # red_ppg, ir_ppg, sred, sir= signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste3_30_01.txt",freq = fs, dois_ou_quatro = 0) #200
    red_ppg, ir_ppg, sred, sir= signal_hw.signal_extract(arquivo = "ppg/txt_hw/teste4_02_02.txt",freq = fs, dois_ou_quatro = 0) #600

    # Time array
    print(f"PPG Signal contains {len(red_ppg)} samples and {len(red_ppg)/fs} sec. duration.")


    red_bac = red_ppg [inicio: inicio+inc_sec]
    ir_bac = ir_ppg [inicio: inicio+inc_sec]
        
    cutoff_lac = 0.4
    cutoff_hac = 8
    BP_ORDER = 8

    red = butter_bandpass_filter(red_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
    ir = butter_bandpass_filter(ir_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)

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

    t = np.linspace(0, ((len(red_dc_cheby2)/ fs)), len(red_dc_cheby2))

    #CALCULOS PARA FFT

    #IR sem filtro

    c_ir_or = np.abs(fftshift(fft(ir_ppg)))

    abs_ir_cor = c_ir_or[:int(np.floor(len(c_ir_or)/2))]

    f1 = np.linspace(0, fs/2, num = len(abs_ir_cor))

    abs_ir_cr  = abs_ir_cor /np.max(abs_ir_cor)
    abs_ir_cr = np.flipud(abs_ir_cr[0::])

    #IR com filtro

    c_ir_fil = np.abs(fftshift(fft(ir_dc_cheby2)))

    abs_ir_cfil = c_ir_fil[:int(np.floor(len(c_ir_fil)/2))]

    f2 = np.linspace(0, fs/2, num = len(abs_ir_cfil))

    abs_ir_fil  = abs_ir_cfil /np.max(abs_ir_cfil)
    abs_ir_fil = np.flipud(abs_ir_fil[0::])


    #RED sem filtro

    c_red_or = np.abs(fftshift(fft(red_ppg)))

    abs_red_cor = c_red_or[:int(np.floor(len(c_red_or)/2))]

    f3 = np.linspace(0, fs/2, num = len(abs_red_cor))

    abs_red_cr  = abs_red_cor /np.max(abs_red_cor)
    abs_red_cr = np.flipud(abs_red_cr[0::])


    #RED com filtro

    c_red_fil = np.abs(fftshift(fft(red_dc_cheby2)))

    abs_red_cfil = c_red_fil[:int(np.floor(len(c_red_fil)/2))]

    f4 = np.linspace(0, fs/2, num = len(abs_red_cfil))

    abs_red_fil  = abs_red_cfil /np.max(abs_red_cfil)
    abs_red_fil = np.flipud(abs_red_fil[0::])
  

    # calculo na frequencia
    frequencia = f4[np.where(abs_ir_fil ==np.max(abs_ir_fil))]
    BPM = frequencia*60
    print(BPM, "bpm")


    #funcao calcula os valores de limiares para que armazer os possiveis picos e os vales do sinal

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


    #SPO2 function
    # SpO2 = -15 * R + 110;

    A = 110
    B = -15

    rfunction = lambda R: A + B * R 

    SPO2 = rfunction(R)

    SPO2_media = numpy.mean(SPO2)


    periodo = inc_sec/fs   #10 seg 
    periodo2 = abs(time_p_ir[0] - time_p_ir[len(time_p_ir)-1])
    quant_peak = len(peak_ir)
    BPM = (quant_peak/periodo)*60
    print('FC:', np.int(BPM),"bpm")
    print('SpO2:', np.int(SPO2_media), "%")
    if (plots == 1):

        #  Plot original
        plt.plot(t, red_bac, "r", label= "Sinal Vermelho")
        plt.plot(t, ir_bac,"b", label= "Sinal Infravermelho")
        # plt.plot(t, ambiente,"k", label= "Luz ambiente", linewidth = 1.0)
        plt.xlabel("Tempo (s)")
        plt.ylabel("Amplitude (mV)")
        plt.legend()
        plt.show()
        # Sinais Cheby2, limiar e pontos de Picos e vales encontrados com duas imagens IR e R 

        # Filtragem do sinal R com Chebychev tipo II
        plt.subplot(2, 2, 1)
        plt.plot(t,red_bac,"r", label = "Sinal RED original")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.subplot(2, 2, 2)
        plt.plot(t,ir_bac,"b", label = "Sinal IR original")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(t,red_dc_cheby2,"r", label = "Sinal RED filtrado com Chebychev tipo II")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(t,ir_dc_cheby2,"b", label = "Sinal IR filtrado com Chebychev tipo II")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.show()

        #PLOTS FFT

        plt.subplot(2, 2, 2)
        plt.plot(f1,abs_ir_cr,"b", label = "FFT do sinal IR sem filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.subplot(2, 2, 1)
        plt.plot(f3,abs_red_cr,"r", label = "FFT do sinal RED sem filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.subplot(2, 2, 4)
        plt.plot(f2,abs_ir_fil,"b", label = "FFT do sinal IR com filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.subplot(2, 2, 3)
        plt.plot(f4,abs_red_fil,"r", label = "FFT do sinal RED com filtragem")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Frequencia [Hz]")
        plt.legend()
        plt.show()

        plt.plot(t, ir, "r",  linewidth = 1, label="Sinal IR original")
        plt.plot(t, ir_dc_cheby2, "k",  linewidth = 1, label="Sinal IR filtrado")
        plt.ylabel("Amplitude [mV]")
        plt.xlabel("Tempo [segundos]")
        plt.legend()
        plt.show()

        #PLOT IR com sinal vindo do filtro Cheby2, limiar e pontos de Picos e vales encontrados 

        plt.plot(t, ir_dc_cheby2, "k", label="PPG Cheby2")
        plt.plot(time_p_ir, peak_ir, "r.", label="Threshold Peaks IR")
        plt.plot(time_v_ir, valley_ir, "g.", label="Threshold Valley IR")
        plt.plot(vec_media_peak_ir_time, vec_media_peak_ir,"y", label="Media peak IR")
        plt.plot(vec_media_valley_ir_time, vec_media_valley_ir,"m", label="Media valley IR")
        plt.legend()
        plt.show()

        # PLOT RED e IR com sinal vindo do filtro Cheby2, limiar e pontos de Picos e vales encontrados

        plt.subplot(2, 1, 1)
        plt.plot(t, red_dc_cheby2, "r", label="Sinal RED PPG ")
        plt.plot(time_p_red, peak_red, "g.", label="Picos e vales")
        plt.plot(time_v_red, valley_red, "g.")#, label="Vales")
        plt.plot(vec_media_peak_red_time, vec_media_peak_red,"k", label="Limiar de picos e vales")
        plt.plot(vec_media_valley_red_time, vec_media_valley_red,"k")#, label="Limiar de vales")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t, ir_dc_cheby2, "b", label="Sinal IR PPG ")
        plt.plot(time_p_ir, peak_ir, "g.", label="Picos e vales")
        plt.plot(time_v_ir, valley_ir, "g.")#, label="Vales")
        plt.plot(vec_media_peak_ir_time, vec_media_peak_ir,"k", label="Limiar de picos e vales")
        plt.plot(vec_media_valley_ir_time, vec_media_valley_ir,"k")#, label="Limiar de vales")
        plt.legend()
        plt.show()

else:
    
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

        ##SERIAL
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
            # remover 1 segundo de amostras (samples) dos vetores red_ppg e ir_ppg


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
        if(sem_dedo == 0):
            red_bac = red_ppg
            ir_bac = ir_ppg

            if(len(red_bac)>len(ir_bac)):
                red_bac = red_bac[0:len(ir_bac)]
            else:
                ir_bac = ir_bac[0:len(red_bac)]

            t = np.linspace(0, ((samples / fs)), len(red_bac))

            cutoff_lac = 0.4
            cutoff_hac = 8
            BP_ORDER = 8

            red = butter_bandpass_filter(red_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)
            ir = butter_bandpass_filter(ir_bac, cutoff_lac, cutoff_hac, fs, order=BP_ORDER)

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

            #funcao calcula os valores de limiares para que armazer os possiceis picos e os vales do sinal

            vec_media_peak_ir = []
            vec_media_valley_ir  = []
            vec_media_peak_red = []
            vec_media_valley_red  = []
            vec_media_peak_red_time = []
            vec_media_valley_red_time  = []
            vec_media_peak_ir_time = []
            vec_media_valley_ir_time  = []

            avg_order = 5
            threshold_peak_valley = 5


            peak_ir,valley_ir, time_p_ir,time_v_ir,vec_media_peak_ir,\
            vec_media_valley_ir,vec_media_peak_ir_time,\
            vec_media_valley_ir_time = find_peak_valley_2(ir_dc_cheby2, t, threshold_peak_valley,avg_order,2)


            peak_red,valley_red, time_p_red,time_v_red,vec_media_peak_red,\
            vec_media_valley_red,vec_media_peak_red_time,\
            vec_media_valley_red_time = find_peak_valley_2(red_dc_cheby2, t, threshold_peak_valley, avg_order,2)

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
             # R = r_red/r_ir

            if(len(r_ir) > len(r_red)):
                R = r_red/r_ir[0:len(r_red)]
            else:
                R = r_red[0:len(r_ir)]/r_ir       


            #SPO2 function
            # SpO2 = -15 * R + 110;

            A = 110
            B = -15

            rfunction = lambda R: A + B * R 

            SPO2 = rfunction(R)

            SPO2_media = numpy.mean(SPO2)
        

            periodo = inc_sec/fs   #10 seg 
            periodo2 = abs(time_p_ir[0] - time_p_ir[len(time_p_ir)-1])
            quant_peak = len(peak_ir)
            BPM = (quant_peak/periodo)*60
            print('FC:', np.int(BPM),"bpm")

            if ((SPO2_media > 100) or (SPO2_media < 0) ):
                SPO2_media = 'ERRO'
                write_ser('SpO2: ' + str(SPO2_media) + 'FC: '+ str(int(BPM))+'bpm')
                print('ERRO')
            else:
                print('O valor de SpO2 é (t) :', np.int(SPO2_media),'%')
                write_ser('SpO2: ' + str(int(SPO2_media))+ '%' + ' FC: '+ str(int(BPM))+'bpm')
        else:
            write_ser('            ' + 'Coloque   o dedo')
        ser.flushInput()
        ser.flushOutput()
    ser.close()


