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


while True:
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    sample = inlet.pull_sample()
    x=sample[0]
    direita=x[1]
    esquerda=x[0]
    print(sample)
    msg = ""

    if esquerda == 1:
        print('voar')
        msg = "command"
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)
        time.sleep(3)

        msg = "takeoff"
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)

    if esquerda == 0:
        print('pousar')
        msg = "land"
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)
        time.sleep(3)
        
