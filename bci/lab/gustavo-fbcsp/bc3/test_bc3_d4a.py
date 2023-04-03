from cPickle import dump
from numpy import asarray, diff, ravel
from scipy.io import loadmat

subject = 'ay'

mat = loadmat('../data/bci3_iva/mat/data_set_IVa_'+subject+'.mat')
cnt = mat['cnt']
pos = mat['mrk'][0][0][0][0]

y = ravel(loadmat('../data/bci3_iva/mat/true_labels_'+subject+'.mat')['true_y'])

epochs = []
for p in pos:
    epochs.append(cnt[p:p+500])

epochs = asarray(epochs).transpose(0,2,1)

print min(diff(pos))
print epochs.shape
print y.shape

data = [epochs, y]

with open('../data/bci3_iva/'+subject+'.pickle', 'wb') as handle:
    dump(data, handle)
