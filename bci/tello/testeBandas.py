import threading 
import socket
import sys
import time
import platform
import sys; sys.path.append('./pylsl')
from pylsl import StreamInlet, resolve_stream

host = ''
port = 9000
locaddr = (host,port) 



sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

msg = "command"
msg = msg.encode(encoding="utf-8") 
sent = sock.sendto(msg, tello_address)
time.sleep(1)

msg = "takeoff"
msg = msg.encode(encoding="utf-8") 
sent = sock.sendto(msg, tello_address)


while True:
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    sample = inlet.pull_sample()
    time.sleep(1)
    x=sample[0]
    delta=x[0]
    theta=x[1]
    alpha = x[2]
    beta = x[3]
    gama = x[4]
    print(alpha,beta)
    
   

##    if esquerda >= direita:
##        print('descer')
##        msg = "down 40"
##        msg = msg.encode(encoding="utf-8") 
##        sent = sock.sendto(msg, tello_address)
##        time.sleep(2)
##
##        
##
##    if direita >= esquerda:
##        print('subir')
##        msg = "up 40"
##        msg = msg.encode(encoding="utf-8") 
##        sent = sock.sendto(msg, tello_address)
        ##time.sleep(2)
