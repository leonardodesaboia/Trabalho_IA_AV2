import numpy as np
from modelos_rna import RBF
from PIL import Image
import os

# Carregar dados (código rápido)
base_path = r"C:\Users\Calil\Documents\VIDA\Faculdade\2025.2\IA\Trabalho AV2\Trabalho_IA_AV2\Trabalho\Questao2\amostras\RecFac"
target_size = (50, 50)

print("Carregando dados...")
pessoas = sorted([p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p))])
C = len(pessoas)

X_list = []
Y_list = []

for idx, pessoa in enumerate(pessoas):
    pessoa_path = os.path.join(base_path, pessoa)
    for img_name in sorted(os.listdir(pessoa_path)):
        img = Image.open(os.path.join(pessoa_path, img_name))
        img_array = np.array(img.resize(target_size), dtype=np.float64).flatten()
        X_list.append(img_array)
        Y_list.append(idx)

X = np.array(X_list).T
Y_labels = np.array(Y_list)
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

Y_onehot = -np.ones((C, X.shape[1]))
for i in range(X.shape[1]):
    Y_onehot[Y_labels[i], i] = 1

# Divisão treino/teste
np.random.seed(42)
idx = np.random.permutation(X.shape[1])
n_train = int(0.8 * X.shape[1])
X_train = X_norm[:, idx[:n_train]]
Y_train = Y_onehot[:, idx[:n_train]]
X_test = X_norm[:, idx[n_train:]]
Y_test = Y_onehot[:, idx[n_train:]]

print(f"✓ Dados prontos\n")

# Função rápida de acurácia
def acc_rbf(modelo, X_test, Y_test):
    acertos = 0
    for k in range(X_test.shape[1]):
        x_k = X_test[:, k].reshape(-1, 1)
        h_k = modelo.calcular_h(x_k)
        h_k_bias = np.vstack((-np.ones((1, 1)), h_k))
        u_k = modelo.W @ h_k_bias
        if np.argmax(u_k) == np.argmax(Y_test[:, k]):
            acertos += 1
    return 100 * acertos / X_test.shape[1]

# TESTE RÁPIDO (menos épocas, menos combinações)
print("="*50)
print("AJUSTE RÁPIDO DO RBF")
print("="*50)

configs = [
    (50, 5.0),
    (50, 10.0),
    (50, 15.0),
    (80, 10.0),
    (100, 10.0),
]

print(f"\n{'n_centros':<12} {'sigma':<10} {'Acurácia':<10}")
print("-"*35)

melhor_acc = 0
melhor_config = None

for n_centros, sigma in configs:
    rbf = RBF(X_train, Y_train, 
              n_centros=n_centros, 
              sigma=sigma, 
              learning_rate=1e-2, 
              max_epochs=200,  # REDUZIDO!
              tol=1e-4)  # TOLERÂNCIA MAIOR!
    rbf.fit()
    acc = acc_rbf(rbf, X_test, Y_test)
    print(f"{n_centros:<12} {sigma:<10.1f} {acc:>8.2f}%")
    
    if acc > melhor_acc:
        melhor_acc = acc
        melhor_config = (n_centros, sigma)

print(f"\n{'='*50}")
print(f"MELHOR: n_centros={melhor_config[0]}, sigma={melhor_config[1]:.1f}")
print(f"Acurácia: {melhor_acc:.2f}%")