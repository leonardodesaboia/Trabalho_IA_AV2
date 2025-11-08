import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from modelos_rna import MADALINEMulticlasse

# =============================================================================
# TESTE RÁPIDO - MADALINE
# =============================================================================
print("="*60)
print("TESTE RAPIDO - VALIDACAO DO MADALINE")
print("="*60)

base_path = r"C:\Users\Calil\Documents\VIDA\Faculdade\2025.2\IA\Trabalho AV2\Trabalho_IA_AV2\Trabalho\Questao2\amostras\RecFac"
target_size = (50, 50)

pessoas = sorted([p for p in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, p))])
C = len(pessoas)

X_list = []
Y_list = []

for idx, pessoa in enumerate(pessoas):
    pessoa_path = os.path.join(base_path, pessoa)
    for img_name in sorted(os.listdir(pessoa_path)):
        img_path = os.path.join(pessoa_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, target_size)
        img_array = img_resized.astype(np.float64).flatten()
        X_list.append(img_array)
        Y_list.append(idx)

X = np.array(X_list).T
Y_labels = np.array(Y_list)
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

Y_onehot = -np.ones((C, X.shape[1]))
for i in range(X.shape[1]):
    Y_onehot[Y_labels[i], i] = 1

print(f"Dados carregados: {X_norm.shape}\n")

# =============================================================================
# TREINAR MADALINE
# =============================================================================
print("Treinando MADALINE...")

np.random.seed(42)
idx = np.random.permutation(X_norm.shape[1])
n_train = int(0.8 * X_norm.shape[1])

X_train = X_norm[:, idx[:n_train]]
Y_train = Y_onehot[:, idx[:n_train]]
X_test = X_norm[:, idx[n_train:]]
Y_test = Y_onehot[:, idx[n_train:]]

mad = MADALINEMulticlasse(X_train, Y_train, n_hidden=30, learning_rate=1e-2, max_epochs=1000, tol=1e-6)
mad.fit()

print(f"Treinamento concluido! ({len(mad.hist_eqm)} epocas)")

# =============================================================================
# CALCULAR ACURÁCIA
# =============================================================================
N_test = X_test.shape[1]
acertos = 0

X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
for k in range(N_test):
    x_k = X_test_bias[:, k].reshape(-1, 1)
    y_k = mad.forward(x_k)
    if np.argmax(y_k) == np.argmax(Y_test[:, k]):
        acertos += 1

acuracia = 100 * acertos / N_test

print(f"\n{'='*60}")
print(f"RESULTADO: {acuracia:.2f}%")
print("="*60)

# =============================================================================
# ANÁLISE
# =============================================================================
if acuracia >= 85:
    print("EXCELENTE! MADALINE esta funcionando corretamente!")
elif acuracia >= 70:
    print("RAZOAVEL. Pode melhorar com ajuste de hiperparametros.")
else:
    print("RUIM! Algo ainda esta errado com o MADALINE.")

print(f"\nEQM final: {mad.hist_eqm[-1]:.6f}")
print(f"Epocas ate convergencia: {len(mad.hist_eqm)}")

# =============================================================================
# PLOTAR CURVA DE APRENDIZADO
# =============================================================================
plt.figure(figsize=(10, 5))
plt.plot(mad.hist_eqm, linewidth=2)
plt.xlabel('Epocas', fontsize=12)
plt.ylabel('EQM', fontsize=12)
plt.title(f'MADALINE - Curva de Aprendizado (Acuracia: {acuracia:.2f}%)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('teste_madaline.png', dpi=150)
print("\nCurva salva: teste_madaline.png")
plt.show()

print(f"\n{'='*60}")
if acuracia >= 85:
    print("PODE RODAR OS SCRIPTS COMPLETOS COM CONFIANCA!")
else:
    print("REVISE O CODIGO ANTES DE RODAR OS SCRIPTS COMPLETOS")
print("="*60)