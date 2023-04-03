from pickle import dump
from numpy import asarray, diff, ravel
from scipy.io import loadmat

subject = 'aa'

mat = loadmat('D:/bci_tools/dset34a/matlab/'+subject+'.mat')
cnt = mat['cnt']
pos = mat['mrk'][0][0][0][0]

y = ravel(loadmat('D:/bci_tools/dset34a/true_labels/trues_'+subject+'.mat')['true_y'])

epochs = []
for p in pos:
    epochs.append(cnt[p:p+500])

epochs = asarray(epochs).transpose(0,2,1)

print(min(diff(pos)))
print(epochs.shape)
print(y.shape)

data = [epochs, y]

with open('D:/bci_tools/dset34a/'+subject+'.pickle', 'wb') as handle:
    dump(data, handle)
