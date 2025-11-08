import numpy as np

class PerceptronMulticlasse:
    def __init__(self, X_train, Y_train, learning_rate=1e-3, max_epochs=500):
        self.p, self.N = X_train.shape
        self.C = Y_train.shape[0]
        
        self.X_train = np.vstack((
            -np.ones((1, self.N)), X_train
        ))
        
        self.d = Y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.W = np.random.random_sample((self.C, self.p+1)) - 0.5
        self.hist_eqm = []
    
    def activation_function(self, u):
        return 1 if u >= 0 else -1
    
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p+1, 1)
            u_k = self.W @ x_k
            y_k = np.array([[self.activation_function(u_k[c, 0])] for c in range(self.C)])
            d_k = self.d[:, k].reshape(self.C, 1)
            e_k = d_k - y_k
            s += np.sum(e_k**2)
        return s / (2 * self.N)
    
    def fit(self):
        epochs = 0
        error = True
        
        while error and epochs < self.max_epochs:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p+1, 1)
                u_k = self.W @ x_k
                y_k = np.array([[self.activation_function(u_k[c, 0])] for c in range(self.C)])
                d_k = self.d[:, k].reshape(self.C, 1)
                e_k = d_k - y_k
                
                if np.any(e_k != 0):
                    error = True
                    
                self.W = self.W + self.lr * (e_k @ x_k.T)
            
            self.hist_eqm.append(self.EQM())
            epochs += 1


class MADALINEMulticlasse:
    def __init__(self, X_train, Y_train, n_hidden=30, learning_rate=1e-2, max_epochs=1000, tol=1e-6):
        self.p, self.N = X_train.shape
        self.C = Y_train.shape[0]
        self.n_hidden = n_hidden
        
        self.X_train = np.vstack((
            -np.ones((1, self.N)), X_train
        ))
        
        self.d = Y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        
        # Inicialização Xavier
        limit1 = np.sqrt(6.0 / (self.p + 1 + n_hidden))
        self.W1 = np.random.uniform(-limit1, limit1, (n_hidden, self.p+1))
        
        limit2 = np.sqrt(6.0 / (n_hidden + 1 + self.C))
        self.W2 = np.random.uniform(-limit2, limit2, (self.C, n_hidden+1))
        
        self.hist_eqm = []
    
    def g(self, u):
        # Sigmoid ao invés de tanh - mais estável
        return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))
    
    def g_d(self, a):
        # Derivada da sigmoid: g'(u) = g(u) * (1 - g(u))
        # Já temos a = g(u), então usamos direto
        return a * (1.0 - a)
    
    def forward(self, x):
        # Camada oculta
        self.u1 = self.W1 @ x
        self.a1 = self.g(self.u1)
        
        # Adiciona bias
        a1_bias = np.vstack((-np.ones((1, 1)), self.a1))
        
        # Camada de saída (linear)
        self.u2 = self.W2 @ a1_bias
        self.a2 = self.u2
        
        return self.a2
    
    def backward(self, x, d):
        # Forward
        y = self.forward(x)
        
        # Erro na saída
        e2 = d - y
        
        # Gradiente camada de saída (linear, sem derivada)
        a1_bias = np.vstack((-np.ones((1, 1)), self.a1))
        grad_W2 = e2 @ a1_bias.T
        
        # Retropropaga erro para camada oculta
        delta1 = (self.W2[:, 1:].T @ e2) * self.g_d(self.a1)
        grad_W1 = delta1 @ x.T
        
        # Atualiza pesos
        self.W2 = self.W2 + self.lr * grad_W2
        self.W1 = self.W1 + self.lr * grad_W1
    
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p+1, 1)
            y_k = self.forward(x_k)
            d_k = self.d[:, k].reshape(self.C, 1)
            e_k = d_k - y_k
            s += np.sum(e_k**2)
        return s / (2 * self.N)
    
    def fit(self):
        EQM_ant = self.EQM()
        self.hist_eqm.append(EQM_ant)
        
        for epoch in range(self.max_epochs):
            # Treina em todas as amostras
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p+1, 1)
                d_k = self.d[:, k].reshape(self.C, 1)
                self.backward(x_k, d_k)
            
            # Calcula EQM
            EQM_atual = self.EQM()
            self.hist_eqm.append(EQM_atual)
            
            # Verifica convergência
            if abs(EQM_ant - EQM_atual) < self.tol:
                break
            
            EQM_ant = EQM_atual


class MultilayerPerceptron:
    def __init__(self, X_train, Y_train, topology, learning_rate=1e-3, max_epoch=1000, tol=1e-8):
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        
        self.X_train = np.vstack((
            -np.ones((1, self.N)), X_train
        ))
        self.tol = tol
        self.lr = learning_rate
        self.d = Y_train
        
        topology.append(self.m)
        self.W = [None] * len(topology)
        
        for i in range(len(self.W)):
            if i == 0:
                W = np.random.random_sample((topology[i], self.p+1)) - 0.5
            else:
                W = np.random.random_sample((topology[i], topology[i-1]+1)) - 0.5
            self.W[i] = W
        
        self.max_epoch = max_epoch
        self.y = [None] * len(topology)
        self.u = [None] * len(topology)
        self.delta = [None] * len(topology)
        self.hist_eqm = []
    
    def g(self, u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))
    
    def g_d(self, u):
        y = self.g(u)
        return 0.5 * (1 - y**2)
    
    def backward(self, e, x):
        for i in range(len(self.W)-1, -1, -1):
            if i == len(self.W)-1:
                self.delta[i] = self.g_d(self.u[i]) * e
                yb = np.vstack((
                    -np.ones((1, 1)),
                    self.y[i-1]
                ))
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)
            elif i == 0:
                Wnbt = (self.W[i+1][:, 1:]).T
                self.delta[i] = self.g_d(self.u[i]) * (Wnbt @ self.delta[i+1])
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ x.T)
            else:
                Wnbt = (self.W[i+1][:, 1:]).T
                self.delta[i] = self.g_d(self.u[i]) * (Wnbt @ self.delta[i+1])
                yb = np.vstack((
                    -np.ones((1, 1)),
                    self.y[i-1]
                ))
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)
    
    def forward(self, x):
        for i, W in enumerate(self.W):
            if i == 0:
                self.u[i] = W @ x
            else:
                yb = np.vstack((
                    -np.ones((1, 1)), self.y[i-1]
                ))
                self.u[i] = W @ yb
            self.y[i] = self.g(self.u[i])
    
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p+1, 1)
            self.forward(x_k)
            y = self.y[-1]
            d = self.d[:, k].reshape(self.m, 1)
            e = d - y
            s += np.sum(e**2)
        return s / (2 * self.N)
    
    def fit(self):
        epoch = 0
        EQM1 = 1
        
        while epoch < self.max_epoch and EQM1 > self.tol:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p+1, 1)
                self.forward(x_k)
                y = self.y[-1]
                d = self.d[:, k].reshape(self.m, 1)
                e = d - y
                self.backward(e, x_k)
            
            EQM1 = self.EQM()
            self.hist_eqm.append(EQM1)
            epoch += 1