from numpy import transpose, concatenate
def windowing(X, t_0, t_start, t_end, fs, atraso):
    W_Start = int((t_start - t_0) * fs)
    W_End = int((t_end - t_0) * fs)
    for i in range(2):
        for j in range(2):
            # X[i][j] = X[i][j][:, :, W_Start:W_End]
            Xa = X[i][j][:, :, W_Start:W_End]
            for cont in range(1, atraso + 1):
                Xb = X[i][j][:, :, W_Start - cont:W_End - cont]
                Xa = transpose(Xa, (1, 0, 2))
                Xb = transpose(Xb, (1, 0, 2))
                Xa = concatenate([Xa, Xb])
                Xa = transpose(Xa, (1, 0, 2))
            X[i][j] = Xa
    return X
