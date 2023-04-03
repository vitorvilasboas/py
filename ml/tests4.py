import numpy as np
import matplotlib.pyplot as plt
from prml.linear import FishersLinearDiscriminant
import sklearn.datasets as ds
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler

iris = ds.load_iris()
X = iris.data  # features
t = iris.target
t_names = iris.target_names

Xa = np.asarray(X[t == 0,:])
Xb = np.asarray(X[t == 1,:])
Xc = np.asarray(X[t == 2,:])

Xa_train = Xa[:25,:]
Xb_train = Xb[:25,:]
Xc_train = Xc[:25,:]

Xa_test = Xa[25:,:]
Xb_test = Xb[25:,:]
Xc_test = Xc[25:,:]

X_train = np.concatenate([Xa_train,Xb_train])
t_train = np.concatenate([np.zeros(25, dtype=int), np.ones(25, dtype=int)])

#Xa_train = StandardScaler().fit_transform(Xa_train)
#Xb_train = StandardScaler().fit_transform(Xb_train)

m_a = np.mean(Xa_train, axis=0) #~ np.asarray([np.mean(Xa_train[:,i]) for i in range(0,4)])
m_b = np.mean(Xb_train, axis=0) 

S_a = Xa_train.T @ Xa_train - m_a.T @ m_a # @ ~ .dot()
S_b = Xb_train.T @ Xb_train - m_b.T @ m_b
Sw = S_a + S_b # cov_inclass

w = np.linalg.inv(Sw) @ m_b-m_a

y_train1 = X_train @ w

X_test = np.concatenate([Xa_test,Xb_test])
y_test1 = X_test @ w

limiar = 0
y_test_cl = (y_test1 > limiar).astype(np.int)

#plt.figure()
#plt.scatter(X_train[:, 0], X_train[:, 1], c=t_train)

test= np.eye(np.max(t) + 1)[t]

#x_plot = 3
#plt.figure(figsize=(10,7))
#for i, name in zip([0, 1], [t_names[1],t_names[2]]):
#    plt.scatter(X_test[t_train == i,x_plot], y_test1[t_train == i], alpha=0.7, label=name)
#x_ = np.linspace(min(X_test[:,x_plot]), max(X_test[:,x_plot]), 100) # np.linspace(0.8,2.6,1000)
#y_ = np.linspace(min(y_test1), max(y_test1), 100) # np.linspace(-0.05,0.08,1000)
#plt.plot(x_,y_, c='r')
#plt.legend(loc='best', scatterpoints=1)
# plt.plot(y_test1, y_test1, c='k')


#X1e2 = Xa.T.dot(Xa) # soma dos quadrados
#Xa = Xa.dot(Xa.T)
#z = np.cov(Xa.T)
#teste = sum(Xa**2)
#variance = np.var(Xa[:,0]) #~
#variance2 = sum([(Xa[i,0] - np.mean(Xa[:,0]))**2 for i in range(len(Xa[:,0]))])/len(Xa[:,0])
#std = np.std(Xa[:,0]) # np.sqrt(variance)
#var_all_feat = [np.var(Xa[:,i]) for i in range(len(Xa.T))]


############################################################

### Opção 2
#Sa2 = np.cov(Xa_train, rowvar=False) # rowvar=False assume que os exemplos esto nas linhas
#Sb2 = np.cov(Xb_train, rowvar=False)
#S2 = Sa2 + Sb2 # cov_inclass
#w2 = np.linalg.solve(S2, m_a - m_b) # ~ np.linalg.inv(S2) @ (m_a - m_b)
#w2 /= np.linalg.norm(w2).clip(min=1e-10)
#y_train2 = X_train @ w2
#y_test2 = X_test @ w2
#plt.figure(figsize=(10,7))
#plt.plot(y_test2)

##########################################################

### Opção 3 PRML
#model = FishersLinearDiscriminant()
#model.fit(X_train, t_train)
#y = model.classify(X_test)


###########################################################

### Opção 4 - Sklearn

# formata os rótulos de classes sempre iniciando de 0 até k-1
t_ = LabelEncoder().fit_transform(t)



# A padronização de um conjunto de dados é um requisito comum para muitos estimadores 
# de aprendizado de máquina: eles podem se comportar mal se os recursos individuais não 
# se parecerem mais ou menos com dados padrão normalmente distribuídos
# Muitos elementos usados ​​na função objetivo de um algoritmo de aprendizado (como o kernel RBF 
# do Support Vector Machines ou os regularizadores L1 e L2 dos modelos lineares) assumem que 
# todos os recursos estão centralizados em torno de 0 e têm variação na mesma ordem. 
# Se um recurso tem uma variação que é ordens de magnitude maior que os outros, ele pode 
# dominar a função objetivo e tornar o estimador incapaz de aprender com outros recursos 
# corretamente, conforme o esperado.
# X_train_std ~ score z = (x-u)/s normaliza/transforma os dados a partir da média (u) e o desvio_padrao (s)
X_train_std = StandardScaler().fit_transform(X)
S_W = np.zeros((4,4))
for i in range(3): S_W += np.cov(X_train_std[t_==i].T) # to 3 classes


X_train_std = StandardScaler().fit_transform(X_train)
S_W = np.zeros((4,4))
for i in range(2): S_W += np.cov(X_train_std[t_train==i].T)

N = np.bincount(t_train) # number of samples for given class
vecs=[]
[vecs.append(np.mean(X_train_std[t_train==i],axis=0)) for i in range(2)] # class means

mean_overall = np.mean(X_train_std, axis=0) # overall mean

S_B=np.zeros((4,4))
for i in range(2): S_B += N[i] * ( ( (vecs[i]-mean_overall).reshape(4,1) ).dot(
        ( (vecs[i]-mean_overall).reshape(1,4) ) ) )


#lda = LinearDiscriminantAnalysis()
#lda.fit(X_train, t_train)
#X_r2 = lda.transform(X_test)
#lda_pred = lda.predict(X_test)
#plt.figure()
#for i, name in zip( [0, 1, 2], t_names):
#    plt.scatter(X_r2[t == i, 0], X_r2[t == i, 1], alpha=.8, label=name)
#plt.legend(loc='best', shadow=False, scatterpoints=1)
