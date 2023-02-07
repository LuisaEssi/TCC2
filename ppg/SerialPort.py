import serial
from serial.tools.list_ports import comports

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
baudrate = 115200
fileName = "data.csv"
fs = 512
samples = 6*fs # 6sec for processing data 


ser = serial.Serial(port,baudrate)
print("Connected to ESP port:" + port)
ser.flushInput()
print("Abrindo Serial")

file = open(fileName, "w")
print("Arquivo criado")

line = 0

while line <= samples:
    data = str(ser.readline().decode("utf-8"))
    # data = data[0:][:-2]
    print(data)
    file = open(fileName,"a")
    file.write(data)
    # file.write(data + "\n")
    line = line+1


print("Final de leituras")
file.close()
ser.close()