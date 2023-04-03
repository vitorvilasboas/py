#
# Tello Python3 Control Demo 
#
# http://www.ryzerobotics.com/
#
# 1/1/2018

import threading 
import socket
import sys
import time
import platform
import sys; sys.path.append('./pylsl')
import pyautogui
from pylsl import StreamInlet, resolve_stream




host = ''
port = 9000
locaddr = (host,port) 


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

tello_address = ('192.168.10.1', 8889)

sock.bind(locaddr)

def recv():
    count = 0
    while True: 
        try:
            data, server = sock.recvfrom(1518)
            print(data.decode(encoding="utf-8"))
        except Exception:
            print ('\nExit . . .\n')
            break


print ('\r\n\r\nTello Python3 Demo.\r\n')

print ('Tello: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')

print ('end -- quit demo.\r\n')




#recvThread create
recvThread = threading.Thread(target=recv)
recvThread.start()


cd = 0
while cd != 10000:
    cd += 1
    try:
        python_version = str(platform.python_version())
        version_init_num = int(python_version.partition('.')[0]) 
       # print (version_init_num)
        if version_init_num == 3:
            msg = input("");
       
        ##streams = resolve_stream('type', 'EEG')
        ##inlet = StreamInlet(streams[0])
        ##sample = inlet.pull_sample()
       
        ##x=sample[0]
        ##direita=x[1]
        ##esquerda=x[0]

        ##if direita >= 0:
            ##msg = "cw 90"
        elif version_init_num == 2:
            msg = raw_input("");
            
        ##if esquerda >= 0:
            ##msg = "cw 90"
        if not msg:
            break  

        if 'end' in msg:
            print ('...')
            sock.close()  
            break

        # Send data
        msg = msg.encode(encoding="utf-8") 
        sent = sock.sendto(msg, tello_address)
    except KeyboardInterrupt:
        print ('\n . . .\n')
        sock.close()  
        break




