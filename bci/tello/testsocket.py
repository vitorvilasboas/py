import socket
import time
host = '127.0.0.1'
port = 6878
locaddr = (host,port) 


# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind(locaddr)


while True:
    data, server = sock.recvfrom(6878)
    print(data.decode('ascii'))
    time.sleep(0.1)

