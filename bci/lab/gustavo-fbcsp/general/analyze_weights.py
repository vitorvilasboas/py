from numpy import arange, dot, linspace, zeros
from cPickle import load
from matplotlib import pyplot as plt

#~ subjectsT = arange(1,10)
subjectsT = [2]
#~ classesT = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
classesT = [[3,4]]

for SUBJECT in subjectsT:
    for classes in classesT:
       
        filename = 'freqs/S' + str(SUBJECT) + 'C' + str(classes)
        struct = load(open(filename))
        
        w = struct['w']
        
        try:
            w_acc += w
        except:
            w_acc = w

fl = 0
fh = 51

f = linspace(fl, fh, len(w_acc))

#~ w_acc = w_acc / max(w_acc)

plt.figure(figsize=(22,11))
plt.grid()
plt.plot(f, w_acc)
plt.scatter (f, w_acc)



#~ subjectsT = arange(1,10)
subjectsT = [2]
#~ classesT = [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]
classesT = [[1,2]]

for SUBJECT in subjectsT:
    for classes in classesT:
       
        filename = 'freqs/S' + str(SUBJECT) + 'C' + str(classes)
        struct = load(open(filename))
        
        w = struct['w']
        
        try:
            w_acc += w
        except:
            w_acc = w

#~ w_acc = w_acc / max(w_acc)

plt.plot(f, w_acc, c='r')
plt.scatter (f, w_acc, c='r')

axes = plt.gca()
axes.set_xlim([fl,fh])

plt.show()
