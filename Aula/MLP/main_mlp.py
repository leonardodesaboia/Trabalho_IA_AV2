import numpy as np
import matplotlib.pyplot as plt
from neural_network import MultilayerPerceptron
data = np.loadtxt("spiral_d.csv",delimiter=',')

X = data[:,:-1]

Y_treino = data[:,-1].reshape(1,1400)
n1 = np.sum(Y_treino[:]==1)
n2 = np.sum(Y_treino[:]==-1)

Y_treino = np.tile(
    np.array([1,-1]).reshape(2,1),(1,n1)
)

Y_treino = np.hstack((
    Y_treino, np.tile(
    np.array([1,-1]).reshape(2,1),(1,n1)
    )
))

#Normalização (min-max)
X_treino = 2*(X-np.min(X))/(np.max(X)-np.min(X))-1
mlp = MultilayerPerceptron(X_treino.T, Y_treino, [1000])
mlp.fit()
bp = 1
