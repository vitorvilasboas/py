import matplotlib.pyplot as plt
from numpy import abs, arange, asarray, cos, dot, linspace, log10, pi, ones, sin
from numpy.fft import fft
from numpy.linalg import lstsq
from numpy.random import rand
from scipy.signal import filtfilt, firwin, freqz

from sin_basis import make_basis

####### Make signal
T = 1000
noise = rand(T)

####### Typical filter - get coefficients
b = firwin(200, [0.2, 0.4], window=('kaiser', 8), pass_zero=False)

####### Typical filter - get frequency response
w, h = freqz(b)

####### Typical filter - empirical time response
a = ones(1)
y1 = filtfilt(b, a, noise)

####### Dictionary projection
X0 = make_basis(1000, 90, 0.1, 0.2)
b = lstsq(X0, noise)[0]
y2 = dot(X0, b)

####### Transform both to frequency domain
Y1 = 20 * log10(abs(fft(y1)[0:500]))
Y2 = 20 * log10(abs(fft(y2)[0:500]))

####### Plot

fig_height = 4.7

plt.figure(figsize=(15,fig_height))
plt.title('Digital filter theoretical frequency response')
plt.plot(w /(2*pi), 20 * log10(abs(h)), 'b')
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
plt.grid()
plt.axis('tight')
plt.show()

f = linspace(0, 0.5, 500)

plt.figure(figsize=(15,fig_height))
plt.title('Digital filter empirical frequency response')
plt.plot(f, Y1)
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
plt.grid()
plt.axis('tight')
plt.show()

plt.figure(figsize=(15,fig_height))
plt.title('Projected signal on sinusoidal base')
plt.plot(f, Y2)
plt.ylabel('Amplitude [dB]', color='b')
plt.xlabel('Frequency [rad/sample]')
plt.grid()
plt.axis('tight')
plt.show()
