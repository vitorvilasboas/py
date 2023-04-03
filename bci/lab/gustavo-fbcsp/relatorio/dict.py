import matplotlib.pyplot as plt
from numpy import arange
from numpy.linalg import lstsq, inv
from numpy.random import normal

def get_projection_matrix(X0):
    COV = dot(X0, X0.T)
    G0 = dot(X0.T, pinv(COV))
    P0 = dot(G0, X0)
    return P0
    
signal_length = 200
freq_limit = 0.2

# Number of components of the first basis
m = 40

# Space (resolution) between frequency bins
B = freq_limit/m

# Make basis and get projection matrix
n = arange(signal_length)
X0 = asarray([cos(2*pi*f*n) for f in arange(0, freq_limit, B)])
P0 = get_projection_matrix(X0)

# Plot first matrix
fsize = 8
fig = plt.figure(figsize(fsize,fsize))
plt.imshow(P0, cmap='gray', interpolation='None')
plt.colorbar()
plt.show()
#plt.savefig('../relatorio-1/matrix-p0.pdf', bbox_inches='tight')

# Number of components of each basis
mv = [10,30,60]

# Plotting
fsize = 12
fig = plt.figure(figsize(fsize,fsize/3.))

for m, i in zip(mv, arange(1,4)):

    B = freq_limit/m

    # Make dictionary and get projection matrix
    # Not using make_basis: here there are only cosines
    X0 = asarray([cos(2*pi*f*n) for f in arange(0, freq_limit, B)])
    COV = dot(X0, X0.T)
    G0 = dot(X0.T, inv(COV))
    P0 = dot(G0, X0)

    # Plot
    plt.subplot(1,3,i)
    plt.imshow(P0, cmap='gray', interpolation='None')

fig.tight_layout()
plt.show()
#plt.savefig('../relatorio-1/matrices-p0.pdf', bbox_inches='tight')
