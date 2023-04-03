# -*- coding: utf-8 -*-
from numpy import dtype, fromfile, transpose, concatenate


def load_data(SUBJECT, classes, atraso):
    # folder = "C:/Users/Josi/OneDrive/datasets/c4_2a/epocas"
    folder = "D:/PPCA/tools/dset_iv2a/epocas"
    # folder = "D:/OneDrive/datasets/c4_2a/epocas"
    X = [[], []]
    set = ['T_', 'E_']
    for i in range(2):
        for j in range(2):
            path = folder + '/A0' + str(SUBJECT) + set[j] + str(classes[i]) + '.fdt'
            fid = open(path, 'rb')
            data = fromfile(fid, dtype('f'))
            data = data.reshape((72, 1000, 22))
            X[j].append(transpose(data, (0,2,1)))  # comentado após implementação de atrasos
            # SBCSP com deslocamento amostral...
            '''
            data = transpose(data, (2, 0, 1))
            Xa = data[:, :, 0:1000 - atraso]
            for cont in range(1, atraso + 1):
                qn = (1000 - atraso) + cont
                Xa = concatenate([Xa, data[:, :, cont:qn]])
            Xa = transpose(Xa, (1, 0, 2))
            X[j].append(Xa)
            '''
            '''
            X[0][0] = A0sT_a  # onde s = sujeito, a/b = id da classe, T = set de treinamento, E = set de avaliação
            X[0][1] = A0sE_a
            X[1][0] = A0sT_b
            X[1][1] = A0sE_b
            '''
    return X

if __name__ == "__main__":

    SUBJECT = 1
    classes = [1, 2]

    X = load_data(SUBJECT, classes, 0)
    
#    X[train/validation set][classe]
#    X_Tset_classe1 = X[0][0]
#    X_Tset_classe2 = X[0][1]
#    X_Vset_classe1 = X[1][0]
#    X_Vset_classe2 = X[1][1]

    print(type(X))
    print(type(X[0]))
    print(type(X[0][0]))
    print(X[0][0].shape)
