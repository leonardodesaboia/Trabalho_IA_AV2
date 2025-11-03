import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CLASSES DE REDES NEURAIS
# ============================================================

class Perceptron:
    """Perceptron Simples para classificacao binaria"""
    def __init__(self, X_train, y_train, learning_rate=1e-3, max_epochs=1000):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5
        self.hist_eqm = []

    def activation_function(self, u):
        return 1 if u >= 0 else -1

    def fit(self, verbose=False):
        epochs = 0
        error = True
        while error and epochs < self.max_epochs:
            error = False
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = (self.w.T @ x_k)[0, 0]
                y_k = self.activation_function(u_k)
                d_k = self.d[k]
                e_k = d_k - y_k
                if e_k != 0:
                    error = True
                self.w = self.w + self.lr * e_k * x_k
            epochs += 1
            # Calcular EQM para acompanhamento
            eqm = self.EQM()
            self.hist_eqm.append(eqm)
            if verbose and epochs % 100 == 0:
                print(f"Epoca {epochs}, EQM: {eqm:.6f}")

    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            u_k = (self.w.T @ x_k)[0, 0]
            y_k = self.activation_function(u_k)
            d_k = self.d[k]
            s += (d_k - y_k) ** 2
        return s / (2 * self.N)

    def predict(self, X):
        p_test, N_test = X.shape
        X_test = np.vstack((-np.ones((1, N_test)), X))
        predictions = []
        for k in range(N_test):
            x_k = X_test[:, k].reshape(p_test + 1, 1)
            u_k = (self.w.T @ x_k)[0, 0]
            y_k = self.activation_function(u_k)
            predictions.append(y_k)
        return np.array(predictions)


class ADALINE:
    """ADALINE (Adaptive Linear Neuron) para classificacao binaria"""
    def __init__(self, X_train, y_train, learning_rate=1e-3, max_epochs=1000, tol=1e-5):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.max_epochs = max_epochs
        self.tol = tol
        self.d = y_train
        self.lr = learning_rate
        self.w = np.random.random_sample((self.p + 1, 1)) - 0.5
        self.hist_eqm = []

    def activation_function(self, u):
        return 1 if u >= 0 else -1

    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            u_k = (self.w.T @ x_k)[0, 0]
            d_k = self.d[k]
            s += (d_k - u_k) ** 2
        return s / (2 * self.N)

    def fit(self, verbose=False):
        epochs = 0
        EQM1 = 0
        EQM2 = 1
        while abs(EQM1 - EQM2) > self.tol and epochs < self.max_epochs:
            EQM1 = self.EQM()
            self.hist_eqm.append(EQM1)
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = (self.w.T @ x_k)[0, 0]
                d_k = self.d[k]
                e_k = d_k - u_k
                self.w = self.w + self.lr * x_k * e_k
            EQM2 = self.EQM()
            epochs += 1
            if verbose and epochs % 100 == 0:
                print(f"Epoca {epochs}, EQM: {EQM2:.6f}")

    def predict(self, X):
        p_test, N_test = X.shape
        X_test = np.vstack((-np.ones((1, N_test)), X))
        predictions = []
        for k in range(N_test):
            x_k = X_test[:, k].reshape(p_test + 1, 1)
            u_k = (self.w.T @ x_k)[0, 0]
            y_k = self.activation_function(u_k)
            predictions.append(y_k)
        return np.array(predictions)


class MultilayerPerceptron:
    """Perceptron de Multiplas Camadas (MLP)"""
    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, topology: list,
                 learning_rate=1e-3, max_epoch=10000, tol=1e-12, verbose=False):
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.tol = tol
        self.lr = learning_rate
        self.d = Y_train
        self.verbose = verbose

        # Adicionar camada de saida
        topology.append(self.m)
        self.W = [None] * len(topology)
        Z = 0

        for i in range(len(self.W)):
            if i == 0:
                W = np.random.random_sample((topology[i], self.p + 1)) - 0.5
            else:
                W = np.random.random_sample((topology[i], topology[i - 1] + 1)) - 0.5
            self.W[i] = W
            Z += W.size

        if verbose:
            print(f"Rede MLP com {Z} parametros")

        self.max_epoch = max_epoch
        self.y = [None] * len(topology)
        self.u = [None] * len(topology)
        self.delta = [None] * len(topology)
        self.hist_eqm = []

    def g(self, u):
        return (1 - np.exp(-u)) / (1 + np.exp(-u))

    def g_d(self, u):
        y = self.g(u)
        return 0.5 * (1 - y ** 2)

    def backward(self, e, x):
        for i in range(len(self.W) - 1, -1, -1):
            if i == len(self.W) - 1:
                self.delta[i] = self.g_d(self.u[i]) * e
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1]))
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)
            elif i == 0:
                Wnbt = (self.W[i + 1][:, 1:]).T
                self.delta[i] = self.g_d(self.u[i]) * (Wnbt @ self.delta[i + 1])
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ x.T)
            else:
                Wnbt = (self.W[i + 1][:, 1:]).T
                self.delta[i] = self.g_d(self.u[i]) * (Wnbt @ self.delta[i + 1])
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1]))
                self.W[i] = self.W[i] + self.lr * (self.delta[i] @ yb.T)

    def forward(self, x):
        for i, W in enumerate(self.W):
            if i == 0:
                self.u[i] = W @ x
            else:
                yb = np.vstack((-np.ones((1, 1)), self.y[i - 1]))
                self.u[i] = W @ yb
            self.y[i] = self.g(self.u[i])

    def EQM(self):
        s = 0
        for k in range(self.N):
            x_k = self.X_train[:, k].reshape(self.p + 1, 1)
            self.forward(x_k)
            y = self.y[-1]
            d = self.d[:, k].reshape(self.m, 1)
            e = d - y
            s += np.sum(e ** 2)
        return s / (2 * self.N)

    def fit(self):
        epoch = 0
        EQM1 = 1

        while epoch < self.max_epoch and EQM1 > self.tol:
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                self.forward(x_k)
                y = self.y[-1]
                d = self.d[:, k].reshape(self.m, 1)
                e = d - y
                self.backward(e, x_k)

            EQM1 = self.EQM()
            self.hist_eqm.append(EQM1)
            epoch += 1

            if self.verbose and epoch % 100 == 0:
                print(f"Epoca: {epoch}, EQM: {EQM1:.10f}")

    def predict(self, X):
        p_test, N_test = X.shape
        X_test = np.vstack((-np.ones((1, N_test)), X))
        predictions = []
        for k in range(N_test):
            x_k = X_test[:, k].reshape(p_test + 1, 1)
            self.forward(x_k)
            y = self.y[-1]
            predictions.append(y)
        return np.hstack(predictions).T


# ============================================================
# FUNCOES DE METRICAS
# ============================================================

def confusion_matrix_manual(y_true, y_pred):
    """Calcula matriz de confusao manualmente"""
    # Para classificacao binaria com labels 1 e -1
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    return np.array([[TP, FN], [FP, TN]])


def calculate_metrics(confusion_matrix):
    """Calcula metricas a partir da matriz de confusao"""
    TP = confusion_matrix[0, 0]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    TN = confusion_matrix[1, 1]

    # Acuracia
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Sensibilidade (Recall)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Especificidade
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Precisao
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # F1-Score
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }


def plot_confusion_matrix(cm, title='Matriz de Confusao', filename=None):
    """Plota matriz de confusao usando seaborn"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positivo', 'Negativo'],
                yticklabels=['Positivo', 'Negativo'],
                cbar_kws={'label': 'Contagem'})
    plt.xlabel('Predito', fontsize=12, fontweight='bold')
    plt.ylabel('Real', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# MAIN - TAREFA DE CLASSIFICACAO
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TAREFA DE CLASSIFICACAO - SPIRAL DATASET")
    print("=" * 80)

    # ============================================================
    # 1. CARREGAR E ORGANIZAR OS DADOS
    # ============================================================
    print("\n1. CARREGAMENTO E ORGANIZACAO DOS DADOS")
    print("-" * 80)

    data = np.loadtxt("spiral_d.csv", delimiter=',')
    print(f"Dataset carregado: {data.shape[0]} amostras, {data.shape[1]} colunas")

    X = data[:, :-1]  # Features
    Y = data[:, -1]   # Labels

    n1 = np.sum(Y == 1)
    n2 = np.sum(Y == -1)
    print(f"Classe +1: {n1} amostras ({n1/len(Y)*100:.1f}%)")
    print(f"Classe -1: {n2} amostras ({n2/len(Y)*100:.1f}%)")

    # ============================================================
    # 2. VISUALIZACAO INICIAL DOS DADOS
    # ============================================================
    print("\n2. VISUALIZACAO INICIAL DOS DADOS")
    print("-" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Dados originais
    axes[0].scatter(X[Y == 1, 0], X[Y == 1, 1],
                    c='#FF6B6B', marker='o', s=50, alpha=0.6,
                    edgecolors='black', linewidth=0.5, label='Classe +1')
    axes[0].scatter(X[Y == -1, 0], X[Y == -1, 1],
                    c='#4ECDC4', marker='s', s=50, alpha=0.6,
                    edgecolors='black', linewidth=0.5, label='Classe -1')
    axes[0].set_xlabel('Feature X', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Feature Y', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribuicao Original dos Dados\n(Spiral Dataset)',
                      fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Estatisticas
    axes[1].text(0.1, 0.9, 'ESTATISTICAS DO DATASET',
                 fontsize=14, fontweight='bold', transform=axes[1].transAxes)
    axes[1].text(0.1, 0.8, f'Total de amostras: {X.shape[0]}',
                 fontsize=11, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.75, f'Numero de features: {X.shape[1]}',
                 fontsize=11, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.70, f'Classe +1: {n1} amostras ({n1/X.shape[0]*100:.1f}%)',
                 fontsize=11, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.65, f'Classe -1: {n2} amostras ({n2/X.shape[0]*100:.1f}%)',
                 fontsize=11, transform=axes[1].transAxes)
    axes[1].text(0.1, 0.55, 'Feature X:', fontsize=11, fontweight='bold',
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.50, f'Min: {X[:,0].min():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.45, f'Max: {X[:,0].max():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.40, f'Media: {X[:,0].mean():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].text(0.1, 0.30, 'Feature Y:', fontsize=11, fontweight='bold',
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.25, f'Min: {X[:,1].min():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.20, f'Max: {X[:,1].max():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].text(0.15, 0.15, f'Media: {X[:,1].mean():.2f}', fontsize=10,
                 transform=axes[1].transAxes)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('1_visualizacao_inicial.png', dpi=300, bbox_inches='tight')
    print("Visualizacao salva: 1_visualizacao_inicial.png")
    plt.close()

    print("\nContinue para os proximos passos...")
