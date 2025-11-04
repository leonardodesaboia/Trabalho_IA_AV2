"""
================================================================================
ETAPA 1 COMPLETA - REDES NEURAIS ARTIFICIAIS
Implementação integral da ETAPA 1 (itens 1-7, sem RBF)

Usando as classes corrigidas (Perceptron, ADALINE, MLP)
Apenas numpy e matplotlib (sem seaborn, pandas, sklearn)
Gráficos mínimos com plt.show()

Baseado exclusivamente nos slides do Prof. Paulo Cirillo - UNIFOR
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================================

DATA_PATH = "Trabalho/Questao1/spiral_d.csv"
R = 500
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
STRATIFY = True  # Split estratificado

# Flags globais
SHOW_PLOTS = True
SAVE_FIGS = False
SAVE_TABLES = True

# Hiperparâmetros
PS_CFG = {"learning_rate": 0.01, "max_epochs": 1000}
ADA_CFG = {"learning_rate": 0.001, "max_epochs": 1000, "tol": 1e-5}
MLP_MAIN_CFG = {"topology": [20], "learning_rate": 0.01, "max_epoch": 500, "tol": 1e-6}
MLP_UNDER_CFG = {"topology": [3], "learning_rate": 0.01, "max_epoch": 300, "tol": 1e-6}
MLP_OVER_CFG = {"topology": [100, 50], "learning_rate": 0.01, "max_epoch": 300, "tol": 1e-6}


# ============================================================================
# CLASSES (copiadas exatamente do arquivo corrigido)
# ============================================================================

class Perceptron:
    """Perceptron Simples - baseado nos slides"""

    def __init__(self, X_train, y_train, learning_rate=0.01, max_epochs=1000,
                 plot=False, save_fig=False, fig_title=None):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.d = y_train
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = np.random.uniform(-0.5, 0.5, (self.p + 1, 1))
        self.hist_eqm = []
        self.hist_epochs = []
        self.plot = plot
        self.save_fig = save_fig
        self.fig_title = fig_title if fig_title else "Perceptron"
        self.w_initial = self.w.copy()

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
            eqm = self.EQM()
            self.hist_eqm.append(eqm)
            self.hist_epochs.append(epochs)
            if verbose and epochs % 50 == 0:
                print(f"Época {epochs}, EQM: {eqm:.6f}")
        if verbose:
            print(f"Treinamento concluído em {epochs} épocas")
        if self.plot:
            self._plot_result()

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

    def _draw_line(self, w, ax, color='k', alpha=1, lw=2, label=None):
        w0, w1, w2 = w[0, 0], w[1, 0], w[2, 0]
        if abs(w2) < 1e-12:
            if abs(w1) < 1e-12:
                return
            x_const = -w0 / w1
            ax.axvline(x_const, c=color, alpha=alpha, lw=lw, label=label)
            return
        x1_range = ax.get_xlim()
        x1 = np.linspace(x1_range[0], x1_range[1], 100)
        x2 = -w1/w2 * x1 + w0/w2
        ax.plot(x1, np.nan_to_num(x2), c=color, alpha=alpha, lw=lw, label=label)

    def _plot_result(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        X = self.X_train[1:, :]
        mask_pos = self.d == 1
        mask_neg = self.d == -1
        ax.scatter(X[0, mask_pos], X[1, mask_pos], c='#FF6B6B', marker='o', s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe +1')
        ax.scatter(X[0, mask_neg], X[1, mask_neg], c='#4ECDC4', marker='s', s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe -1')
        self._draw_line(self.w_initial, ax, color='black', alpha=0.5, lw=2, label='Inicial')
        self._draw_line(self.w, ax, color='green', alpha=0.8, lw=2.5, label='Final')
        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(self.fig_title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig('perceptron_result.png', dpi=300, bbox_inches='tight')
        plt.show()


class ADALINE:
    """ADALINE - baseado nos slides"""

    def __init__(self, X_train, y_train, learning_rate=0.001, max_epochs=1000, tol=1e-5,
                 plot=False, save_fig=False, fig_title=None, plot_eqm=False):
        self.p, self.N = X_train.shape
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.max_epochs = max_epochs
        self.tol = tol
        self.d = y_train
        self.lr = learning_rate
        self.w = np.random.uniform(-0.5, 0.5, (self.p + 1, 1))
        self.hist_eqm = []
        self.hist_epochs = []
        self.plot = plot
        self.save_fig = save_fig
        self.fig_title = fig_title if fig_title else "ADALINE"
        self.plot_eqm = plot_eqm
        self.w_initial = self.w.copy()

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
            self.hist_epochs.append(epochs + 1)
            for k in range(self.N):
                x_k = self.X_train[:, k].reshape(self.p + 1, 1)
                u_k = (self.w.T @ x_k)[0, 0]
                d_k = self.d[k]
                e_k = d_k - u_k
                self.w = self.w + self.lr * x_k * e_k
            EQM2 = self.EQM()
            epochs += 1
            if verbose and epochs % 50 == 0:
                print(f"Época {epochs}, EQM: {EQM2:.6f}")
        if verbose:
            print(f"Treinamento concluído em {epochs} épocas")
        if self.plot:
            self._plot_result()
        if self.plot_eqm:
            self._plot_eqm_curve()

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

    def _draw_line(self, w, ax, color='k', alpha=1, lw=2, label=None):
        w0, w1, w2 = w[0, 0], w[1, 0], w[2, 0]
        if abs(w2) < 1e-12:
            if abs(w1) < 1e-12:
                return
            x_const = -w0 / w1
            ax.axvline(x_const, c=color, alpha=alpha, lw=lw, label=label)
            return
        x1_range = ax.get_xlim()
        x1 = np.linspace(x1_range[0], x1_range[1], 100)
        x2 = -w1/w2 * x1 + w0/w2
        ax.plot(x1, np.nan_to_num(x2), c=color, alpha=alpha, lw=lw, label=label)

    def _plot_result(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        X = self.X_train[1:, :]
        mask_pos = self.d == 1
        mask_neg = self.d == -1
        ax.scatter(X[0, mask_pos], X[1, mask_pos], c='#FF6B6B', marker='o', s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe +1')
        ax.scatter(X[0, mask_neg], X[1, mask_neg], c='#4ECDC4', marker='s', s=60,
                  alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe -1')
        self._draw_line(self.w_initial, ax, color='black', alpha=0.5, lw=2, label='Inicial')
        self._draw_line(self.w, ax, color='green', alpha=0.8, lw=2.5, label='Final')
        ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
        ax.set_title(self.fig_title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig('adaline_result.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_eqm_curve(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.hist_epochs, self.hist_eqm, 'b-', lw=2)
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('EQM', fontsize=12, fontweight='bold')
        ax.set_title('ADALINE - Curva de Aprendizado', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class MultilayerPerceptron:
    """MLP com backpropagation - baseado nos slides"""

    def __init__(self, X_train, Y_train, topology, learning_rate=0.01, max_epoch=10000,
                 tol=1e-12, plot=False, save_fig=False, fig_title=None, verbose=False):
        if X_train.ndim == 2 and X_train.shape[0] > X_train.shape[1]:
            X_train = X_train.T
        self.p, self.N = X_train.shape
        self.m = Y_train.shape[0]
        self.X_train = np.vstack((-np.ones((1, self.N)), X_train))
        self.tol = tol
        self.lr = learning_rate
        self.d = Y_train
        self.verbose = verbose
        self.plot = plot
        self.save_fig = save_fig
        self.fig_title = fig_title if fig_title else "MLP"

        hidden = list(topology)
        layers = hidden + [self.m]
        self.L = len(layers)
        self.W = [None] * self.L
        Z = 0
        for i in range(self.L):
            if i == 0:
                W = np.random.uniform(-0.5, 0.5, (layers[i], self.p + 1))
            else:
                W = np.random.uniform(-0.5, 0.5, (layers[i], layers[i - 1] + 1))
            self.W[i] = W
            Z += W.size
        if verbose:
            print(f"Rede MLP com {Z} parâmetros")

        self.max_epoch = max_epoch
        self.y = [None] * self.L
        self.u = [None] * self.L
        self.delta = [None] * self.L
        self.hist_eqm = []
        self.hist_epochs = []

    def g(self, u):
        """Tangente hiperbólica conforme slides"""
        return np.tanh(u)

    def g_d(self, u):
        """Derivada da tangente hiperbólica conforme slides"""
        y = np.tanh(u)
        return 0.5 * (1 - y**2)

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
            self.hist_epochs.append(epoch + 1)
            epoch += 1

            if self.verbose and epoch % 50 == 0:
                print(f"Época: {epoch}, EQM: {EQM1:.10f}")

        if self.verbose:
            print(f"Treinamento concluído em {epoch} épocas")

        if self.plot:
            self._plot_eqm_curve()

    def predict(self, X):
        if X.ndim == 2 and X.shape[0] > X.shape[1]:
            X = X.T
        p_test, N_test = X.shape
        X_test = np.vstack((-np.ones((1, N_test)), X))
        predictions = []
        for k in range(N_test):
            x_k = X_test[:, k].reshape(p_test + 1, 1)
            self.forward(x_k)
            y = self.y[-1]
            predictions.append(y)
        return np.hstack(predictions).T

    def _plot_eqm_curve(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.hist_epochs, self.hist_eqm, 'b-', lw=2)
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('EQM', fontsize=12, fontweight='bold')
        ax.set_title(self.fig_title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        if self.save_fig:
            plt.savefig('mlp_result.png', dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def train_test_split_numpy(X, y, train_ratio, seed, stratify=True):
    """Split estratificado simples"""
    np.random.seed(seed)
    N = X.shape[0]

    if stratify:
        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == -1)[0]

        np.random.shuffle(idx_pos)
        np.random.shuffle(idx_neg)

        n_train_pos = int(len(idx_pos) * train_ratio)
        n_train_neg = int(len(idx_neg) * train_ratio)

        train_idx = np.concatenate([idx_pos[:n_train_pos], idx_neg[:n_train_neg]])
        test_idx = np.concatenate([idx_pos[n_train_pos:], idx_neg[n_train_neg:]])

        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
    else:
        indices = np.random.permutation(N)
        split = int(N * train_ratio)
        train_idx = indices[:split]
        test_idx = indices[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def to_one_hot(y):
    """Converte rótulos {-1, +1} para one-hot (2, N)"""
    N = len(y)
    onehot = np.zeros((2, N))
    onehot[0, y == 1] = 1
    onehot[1, y == -1] = 1
    return onehot


def confusion_matrix_manual(y_true, y_pred):
    """Matriz de confusão manual"""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    return np.array([[TP, FN], [FP, TN]])


def metrics_from_cm(cm):
    """Calcula métricas da matriz de confusão"""
    TP, FN = cm[0, 0], cm[0, 1]
    FP, TN = cm[1, 0], cm[1, 1]

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    if (precision + sensitivity) > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0

    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }


def plot_scatter(X, y, title="Scatter Plot"):
    """Scatter plot 2D com matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask_pos = y == 1
    mask_neg = y == -1
    ax.scatter(X[mask_pos, 0], X[mask_pos, 1], c='#FF6B6B', marker='o', s=60,
              alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe +1')
    ax.scatter(X[mask_neg, 0], X[mask_neg, 1], c='#4ECDC4', marker='s', s=60,
              alpha=0.7, edgecolors='black', linewidth=0.8, label='Classe -1')
    ax.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGS:
        plt.savefig('scatter_inicial.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_cm_imshow(cm, title="Matriz de Confusao", filename=None):
    """Plota matriz de confusão com imshow"""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['+1', '-1'])
    ax.set_yticklabels(['+1', '-1'])
    ax.set_xlabel('Predito', fontsize=11, fontweight='bold')
    ax.set_ylabel('Real', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Anotações
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if filename and SAVE_FIGS:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def print_stats_table(summary):
    """Imprime tabela de estatísticas"""
    print("\n" + "=" * 100)
    print("ESTATÍSTICAS FINAIS (R=500 RODADAS)")
    print("=" * 100)

    metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 100)
        print(f"{'Modelo':<15} {'Média':<12} {'Desvio':<12} {'Máximo':<12} {'Mínimo':<12}")
        print("-" * 100)
        for model in summary:
            stats = summary[model][metric]
            print(f"{model:<15} {stats['mean']*100:>10.2f}% {stats['std']*100:>10.2f}% "
                  f"{stats['max']*100:>10.2f}% {stats['min']*100:>10.2f}%")


# ============================================================================
# FLUXO PRINCIPAL - ETAPA 1
# ============================================================================

def main():
    print("=" * 80)
    print("ETAPA 1 - REDES NEURAIS ARTIFICIAIS")
    print("=" * 80)
    print(f"Configurações: R={R}, TRAIN_RATIO={TRAIN_RATIO}, STRATIFY={STRATIFY}")
    print(f"SHOW_PLOTS={SHOW_PLOTS}, SAVE_FIGS={SAVE_FIGS}, SAVE_TABLES={SAVE_TABLES}")

    # ========================================================================
    # (1) ORGANIZAÇÃO DOS DADOS
    # ========================================================================
    print("\n" + "=" * 80)
    print("(1) CARREGAMENTO E VISUALIZAÇÃO INICIAL")
    print("=" * 80)

    data = np.loadtxt(DATA_PATH, delimiter=',')
    X = data[:, :2]
    y = data[:, 2]
    y = np.where(y > 0, 1, -1)

    print(f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Classe +1: {np.sum(y == 1)} amostras")
    print(f"Classe -1: {np.sum(y == -1)} amostras")

    if SHOW_PLOTS:
        plot_scatter(X, y, title="Visualização Inicial - Spiral Dataset")

    # ========================================================================
    # (2) TREINO INICIAL (SANIDADE)
    # ========================================================================
    print("\n" + "=" * 80)
    print("(2) TREINO INICIAL (SANIDADE)")
    print("=" * 80)

    X_tr, y_tr, X_te, y_te = train_test_split_numpy(X, y, TRAIN_RATIO, RANDOM_SEED, STRATIFY)
    print(f"Treino: {len(y_tr)} amostras | Teste: {len(y_te)} amostras")

    # Perceptron
    print("\n--- PERCEPTRON ---")
    ps = Perceptron(X_tr.T, y_tr, **PS_CFG, plot=False)
    ps.fit(verbose=False)
    y_pred_ps = ps.predict(X_te.T)
    acc_ps = np.mean(y_pred_ps == y_te)
    print(f"Acurácia: {acc_ps*100:.2f}%")

    # ADALINE
    print("\n--- ADALINE ---")
    ada = ADALINE(X_tr.T, y_tr, **ADA_CFG, plot=False)
    ada.fit(verbose=False)
    y_pred_ada = ada.predict(X_te.T)
    acc_ada = np.mean(y_pred_ada == y_te)
    print(f"Acurácia: {acc_ada*100:.2f}%")

    # MLP
    print("\n--- MLP ---")
    y_tr_oh = to_one_hot(y_tr)
    mlp = MultilayerPerceptron(X_tr.T, y_tr_oh, **MLP_MAIN_CFG, plot=False, verbose=False)
    mlp.fit()
    y_pred_mlp_probs = mlp.predict(X_te.T)
    y_pred_mlp = np.where(np.argmax(y_pred_mlp_probs, axis=1) == 0, 1, -1)
    acc_mlp = np.mean(y_pred_mlp == y_te)
    print(f"Acurácia: {acc_mlp*100:.2f}%")

    # ========================================================================
    # (3) UNDERFITTING / OVERFITTING (MLP)
    # ========================================================================
    print("\n" + "=" * 80)
    print("(3) UNDERFITTING / OVERFITTING (MLP)")
    print("=" * 80)

    # Underfitting
    print("\n--- UNDERFITTING: MLP [3] ---")
    mlp_under = MultilayerPerceptron(X_tr.T, y_tr_oh, **MLP_UNDER_CFG,
                                     plot=SHOW_PLOTS, save_fig=SAVE_FIGS,
                                     fig_title="MLP Underfitting - Curva EQM", verbose=False)
    mlp_under.fit()

    # Predições treino
    y_pred_train_probs = mlp_under.predict(X_tr.T)
    y_pred_train = np.where(np.argmax(y_pred_train_probs, axis=1) == 0, 1, -1)
    cm_under_train = confusion_matrix_manual(y_tr, y_pred_train)
    metrics_under_train = metrics_from_cm(cm_under_train)

    # Predições teste
    y_pred_test_probs = mlp_under.predict(X_te.T)
    y_pred_test = np.where(np.argmax(y_pred_test_probs, axis=1) == 0, 1, -1)
    cm_under_test = confusion_matrix_manual(y_te, y_pred_test)
    metrics_under_test = metrics_from_cm(cm_under_test)

    print("UNDERFITTING - Treino:", metrics_under_train)
    print("UNDERFITTING - Teste :", metrics_under_test)

    if SAVE_FIGS and mlp_under.hist_eqm:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mlp_under.hist_epochs, mlp_under.hist_eqm, 'b-', lw=2)
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('EQM', fontsize=12, fontweight='bold')
        ax.set_title('MLP Underfitting - EQM', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('mlp_under_eqm.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Overfitting
    print("\n--- OVERFITTING: MLP [100, 50] ---")
    mlp_over = MultilayerPerceptron(X_tr.T, y_tr_oh, **MLP_OVER_CFG,
                                    plot=SHOW_PLOTS, save_fig=SAVE_FIGS,
                                    fig_title="MLP Overfitting - Curva EQM", verbose=False)
    mlp_over.fit()

    # Predições treino
    y_pred_train_probs = mlp_over.predict(X_tr.T)
    y_pred_train = np.where(np.argmax(y_pred_train_probs, axis=1) == 0, 1, -1)
    cm_over_train = confusion_matrix_manual(y_tr, y_pred_train)
    metrics_over_train = metrics_from_cm(cm_over_train)

    # Predições teste
    y_pred_test_probs = mlp_over.predict(X_te.T)
    y_pred_test = np.where(np.argmax(y_pred_test_probs, axis=1) == 0, 1, -1)
    cm_over_test = confusion_matrix_manual(y_te, y_pred_test)
    metrics_over_test = metrics_from_cm(cm_over_test)

    print("OVERFITTING - Treino:", metrics_over_train)
    print("OVERFITTING - Teste :", metrics_over_test)

    if SAVE_FIGS and mlp_over.hist_eqm:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mlp_over.hist_epochs, mlp_over.hist_eqm, 'r-', lw=2)
        ax.set_xlabel('Época', fontsize=12, fontweight='bold')
        ax.set_ylabel('EQM', fontsize=12, fontweight='bold')
        ax.set_title('MLP Overfitting - EQM', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        plt.tight_layout()
        plt.savefig('mlp_over_eqm.png', dpi=300, bbox_inches='tight')
        plt.close()

    # ========================================================================
    # (4) VALIDAÇÃO R=500 RODADAS
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"(4) VALIDAÇÃO POR AMOSTRAGEM ALEATÓRIA (R={R})")
    print("=" * 80)

    results = {
        'Perceptron': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                      'precision': [], 'f1_score': [], 'cm': []},
        'ADALINE': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                   'precision': [], 'f1_score': [], 'cm': []},
        'MLP': {'accuracy': [], 'sensitivity': [], 'specificity': [],
               'precision': [], 'f1_score': [], 'cm': []}
    }

    for r in range(R):
        # Split com seed derivada
        X_tr, y_tr, X_te, y_te = train_test_split_numpy(X, y, TRAIN_RATIO,
                                                         RANDOM_SEED + r, STRATIFY)

        # Perceptron
        ps = Perceptron(X_tr.T, y_tr, **PS_CFG, plot=False)
        ps.fit(verbose=False)
        y_pred = ps.predict(X_te.T)
        cm = confusion_matrix_manual(y_te, y_pred)
        metrics = metrics_from_cm(cm)
        for key in metrics:
            results['Perceptron'][key].append(metrics[key])
        results['Perceptron']['cm'].append(cm)

        # ADALINE
        ada = ADALINE(X_tr.T, y_tr, **ADA_CFG, plot=False)
        ada.fit(verbose=False)
        y_pred = ada.predict(X_te.T)
        cm = confusion_matrix_manual(y_te, y_pred)
        metrics = metrics_from_cm(cm)
        for key in metrics:
            results['ADALINE'][key].append(metrics[key])
        results['ADALINE']['cm'].append(cm)

        # MLP
        y_tr_oh = to_one_hot(y_tr)
        mlp = MultilayerPerceptron(X_tr.T, y_tr_oh, **MLP_MAIN_CFG, plot=False, verbose=False)
        mlp.fit()
        y_pred_probs = mlp.predict(X_te.T)
        y_pred = np.where(np.argmax(y_pred_probs, axis=1) == 0, 1, -1)
        cm = confusion_matrix_manual(y_te, y_pred)
        metrics = metrics_from_cm(cm)
        for key in metrics:
            results['MLP'][key].append(metrics[key])
        results['MLP']['cm'].append(cm)

        if (r + 1) % 50 == 0:
            print(f"Rodada {r+1}/{R} concluída")

    print(f"Validação concluída: {R} rodadas")

    # ========================================================================
    # (5) MELHORES/PIORES (todas as 5 métricas)
    # ========================================================================
    print("\n" + "=" * 80)
    print("(5) SELEÇÃO DE MELHORES/PIORES RODADAS")
    print("=" * 80)

    for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
        print(f"\n--- Métrica: {metric.upper()} ---")

        for model in ['Perceptron', 'ADALINE', 'MLP']:
            values = np.array(results[model][metric])
            best_idx = np.argmax(values)
            worst_idx = np.argmin(values)

            print(f"{model}: Best={values[best_idx]*100:.2f}% (rodada {best_idx}) | "
                  f"Worst={values[worst_idx]*100:.2f}% (rodada {worst_idx})")

            # Re-treinar as rodadas selecionadas para gerar curvas
            for case, idx in [('best', best_idx), ('worst', worst_idx)]:
                # Split exatamente como foi feito na rodada original
                X_tr_sel, y_tr_sel, X_te_sel, y_te_sel = train_test_split_numpy(
                    X, y, TRAIN_RATIO, RANDOM_SEED + idx, STRATIFY)

                # Re-treinar o modelo
                if model == 'Perceptron':
                    model_sel = Perceptron(X_tr_sel.T, y_tr_sel, **PS_CFG, plot=False)
                    model_sel.fit(verbose=False)
                    hist_eqm = model_sel.hist_eqm
                    hist_epochs = model_sel.hist_epochs
                elif model == 'ADALINE':
                    model_sel = ADALINE(X_tr_sel.T, y_tr_sel, **ADA_CFG, plot=False)
                    model_sel.fit(verbose=False)
                    hist_eqm = model_sel.hist_eqm
                    hist_epochs = model_sel.hist_epochs
                elif model == 'MLP':
                    y_tr_sel_oh = to_one_hot(y_tr_sel)
                    model_sel = MultilayerPerceptron(X_tr_sel.T, y_tr_sel_oh, **MLP_MAIN_CFG,
                                                     plot=False, verbose=False)
                    model_sel.fit()
                    hist_eqm = model_sel.hist_eqm
                    hist_epochs = model_sel.hist_epochs

                # Plotar CM
                cm = results[model]['cm'][idx]
                fname_cm = f"cm_{model.lower()}_{metric}_{case}.png" if SAVE_FIGS else None
                if SHOW_PLOTS:
                    plot_cm_imshow(cm, f"{model} - {metric.upper()} {case.upper()}", fname_cm)
                elif SAVE_FIGS:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(['+1', '-1'])
                    ax.set_yticklabels(['+1', '-1'])
                    ax.set_title(f"{model} - {metric.upper()} {case.upper()}", fontsize=12, fontweight='bold')
                    for i in range(2):
                        for j in range(2):
                            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                                   color="white" if cm[i, j] > cm.max() / 2 else "black")
                    plt.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(fname_cm, dpi=300, bbox_inches='tight')
                    plt.close()

                # Plotar curva de aprendizado
                fname_lc = f"lc_{model.lower()}_{metric}_{case}.png" if SAVE_FIGS else None
                if SHOW_PLOTS or SAVE_FIGS:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(hist_epochs, hist_eqm, 'b-', lw=2)
                    ax.set_xlabel('Época', fontsize=12, fontweight='bold')
                    ax.set_ylabel('EQM', fontsize=12, fontweight='bold')
                    ax.set_title(f"{model} - {metric.upper()} {case.upper()} - Curva Aprendizado",
                                fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    if model == 'MLP':
                        ax.set_yscale('log')
                    plt.tight_layout()
                    if SAVE_FIGS:
                        plt.savefig(fname_lc, dpi=300, bbox_inches='tight')
                    if SHOW_PLOTS:
                        plt.show()
                    else:
                        plt.close()

    # ========================================================================
    # (6) ESTATÍSTICAS FINAIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("(6) ESTATÍSTICAS FINAIS")
    print("=" * 80)

    summary = {}
    for model in ['Perceptron', 'ADALINE', 'MLP']:
        summary[model] = {}
        for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
            values = np.array(results[model][metric])
            summary[model][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'min': np.min(values)
            }

    print_stats_table(summary)

    # Salvar tabelas CSV
    if SAVE_TABLES:
        print("\nSalvando tabelas CSV...")
        for metric in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
            filename = f'tabela_{metric}.csv'
            with open(filename, 'w') as f:
                f.write("Modelo,Media,Desvio_Padrao,Maximo,Minimo\n")
                for model in ['Perceptron', 'ADALINE', 'MLP']:
                    stats = summary[model][metric]
                    f.write(f"{model},{stats['mean']*100:.2f},{stats['std']*100:.2f},"
                           f"{stats['max']*100:.2f},{stats['min']*100:.2f}\n")
            print(f"  {filename}")

    # ========================================================================
    # (7) QUADRO-RESUMO FINAL
    # ========================================================================
    print("\n" + "=" * 80)
    print("(7) QUADRO-RESUMO COMPARATIVO")
    print("=" * 80)

    print(f"\n{'Modelo':<15} {'Accuracy (média±desvio)':<30} {'F1-Score (média±desvio)':<30}")
    print("-" * 80)
    for model in ['Perceptron', 'ADALINE', 'MLP']:
        acc_mean = summary[model]['accuracy']['mean'] * 100
        acc_std = summary[model]['accuracy']['std'] * 100
        f1_mean = summary[model]['f1_score']['mean'] * 100
        f1_std = summary[model]['f1_score']['std'] * 100
        print(f"{model:<15} {acc_mean:>6.2f}% ± {acc_std:>5.2f}%{' '*15} "
              f"{f1_mean:>6.2f}% ± {f1_std:>5.2f}%")

    print("\n" + "=" * 80)
    print("ETAPA 1 CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()