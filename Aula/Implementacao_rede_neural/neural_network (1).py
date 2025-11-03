import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self,X_train,y_train,learning_rate=1e-3,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.d = y_train
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1,self.d[:]==1],
                            self.X_train[2,self.d[:]==1],marker='s',s=120)
            self.ax.scatter(self.X_train[1,self.d[:]==-1],
                            self.X_train[2,self.d[:]==-1],marker='o',s=120)
            self.ax.set_xlim(-1,7)
            self.ax.set_ylim(-1,7)
            self.x1 = np.linspace(-2,10)
            self.draw_line()
        
    def draw_line(self,c='k',alpha=1,lw=2):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x1,x2,c=c,alpha=alpha,lw=lw)
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    
    def fit(self):
        epochs = 0
        error = True
        while error:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                y_k = self.activation_function(u_k)
                d_k = self.d[k]
                e_k = d_k - y_k
                if e_k!=0:
                    error = True
                self.w = self.w + self.lr*e_k*x_k
            
            plt.pause(.4)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
        plt.pause(.4)
        self.draw_line(c='g',alpha=1,lw=4)
        plt.show()
class ADALINE:
    def __init__(self,X_train,y_train,learning_rate=1e-3,max_epochs=1000,tol=1e-5,plot=True):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((
            -np.ones((1,self.N)), X_train
        ))
        self.max_epochs = max_epochs
        self.tol = tol
        self.d = y_train
        self.lr = learning_rate
        self.w = np.zeros((self.p+1,1))
        self.w = np.random.random_sample((self.p+1,1))-.5
        self.plot = plot
        if plot:
            self.fig = plt.figure(1)
            self.ax = self.fig.add_subplot()
            self.ax.scatter(self.X_train[1,self.d[:]==1],
                            self.X_train[2,self.d[:]==1],marker='s',s=120)
            self.ax.scatter(self.X_train[1,self.d[:]==-1],
                            self.X_train[2,self.d[:]==-1],marker='o',s=120)
            self.ax.set_xlim(-1,7)
            self.ax.set_ylim(-1,7)
            self.x1 = np.linspace(-2,10)
            self.draw_line()
        
    def draw_line(self,c='k',alpha=1,lw=2):
        x2 = -self.w[1,0]/self.w[2,0]*self.x1 + self.w[0,0]/self.w[2,0]
        x2 = np.nan_to_num(x2)
        plt.plot(self.x1,x2,c=c,alpha=alpha,lw=lw)
        
    def activation_function(self, u):
        return 1 if u>=0 else -1
    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:,k].reshape(self.p+1,1)
            u_k = (self.w.T@x_k)[0,0]
            d_k = self.d[k]
            s += (d_k-u_k)**2
        return s/(2*self.N)
    def fit(self):
        epochs = 0
        EQM1 = 0
        EQM2 = 1
        hist_eqm = []
        while abs(EQM1-EQM2)>self.tol and epochs < self.max_epochs:
            EQM1 = self.EQM()
            hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:,k].reshape(self.p+1,1)
                u_k = (self.w.T@x_k)[0,0]
                d_k = self.d[k]
                e_k = d_k - u_k
                self.w = self.w + self.lr*x_k*e_k
            EQM2 = self.EQM()
            plt.pause(.1)
            self.draw_line(c='r',alpha=.5)
            epochs+=1
        plt.pause(.1)
        self.draw_line(c='g',alpha=1,lw=4)
        plt.figure(2)
        plt.plot(hist_eqm)
        plt.grid()
        plt.title("Curva de aprendizado do ADALINE")
        plt.xlabel("Épocas")
        plt.ylabel("EQM")
        plt.show()
        
        


bp=1