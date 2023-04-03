from cPickle import dump, load

t = load(open('/home/gustavo/bci/results/bc3/proj_8_30'))


#~ How many trials before cleaning
print len(t.trials)

to_be_del = []

for i in range(len(t.trials)):
    try:
        loss = t.trials[i]['result']['loss']
        if loss > 60 or loss <= 1:
            to_be_del.append(i)
    except:
        to_be_del.append(i)

for i in reversed(to_be_del):
    del t.trials[i]


#~ How many trials after cleaning
print len(t.trials)

dump(t, open('/home/gustavo/bci/results/bc3/proj_8_30', 'wb'))
