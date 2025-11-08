import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
from modelos_rna import PerceptronMulticlasse, MADALINEMulticlasse, MultilayerPerceptron

# =============================================================================
# CARREGAR DADOS
# =============================================================================
print("="*60)
print("ANÁLISE DAS RODADAS EXTREMAS")
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

print(f"✓ Dados carregados\n")

# =============================================================================
# FUNÇÃO DE ACURÁCIA
# =============================================================================
def calcular_acuracia(modelo, X_test, Y_test, tipo):
    N_test = X_test.shape[1]
    acertos = 0
    
    if tipo == 'linear':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            u_k = modelo.W @ x_k
            if np.argmax(u_k) == np.argmax(Y_test[:, k]):
                acertos += 1
    
    elif tipo == 'madaline':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            y_k = modelo.forward(x_k)
            if np.argmax(y_k) == np.argmax(Y_test[:, k]):
                acertos += 1
    
    elif tipo == 'mlp':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            modelo.forward(x_k)
            if np.argmax(modelo.y[-1]) == np.argmax(Y_test[:, k]):
                acertos += 1
    
    return 100 * acertos / N_test

# =============================================================================
# ETAPA 1: RODAR 10 RODADAS PARA IDENTIFICAR EXTREMAS
# =============================================================================
print("="*60)
print("ETAPA 1: EXECUTANDO 10 RODADAS PARA IDENTIFICAR EXTREMAS")
print("="*60)

R = 10
acuracias_mlp = []

for r in range(R):
    np.random.seed(r)
    idx = np.random.permutation(X_norm.shape[1])
    n_train = int(0.8 * X_norm.shape[1])
    
    X_train = X_norm[:, idx[:n_train]]
    Y_train = Y_onehot[:, idx[:n_train]]
    X_test = X_norm[:, idx[n_train:]]
    Y_test = Y_onehot[:, idx[n_train:]]
    
    mlp = MultilayerPerceptron(X_train, Y_train, topology=[50, 30], learning_rate=1e-2, max_epoch=500, tol=1e-8)
    mlp.fit()
    acc = calcular_acuracia(mlp, X_test, Y_test, 'mlp')
    acuracias_mlp.append(acc)
    
    print(f"Rodada {r}: {acc:.2f}%")

idx_maior = np.argmax(acuracias_mlp)
idx_menor = np.argmin(acuracias_mlp)

print(f"\n✓ Rodada com MAIOR acurácia: {idx_maior} ({acuracias_mlp[idx_maior]:.2f}%)")
print(f"✓ Rodada com MENOR acurácia: {idx_menor} ({acuracias_mlp[idx_menor]:.2f}%)")

rodadas_extremas = [idx_maior, idx_menor]
labels_rodadas = [
    f'Rodada {idx_maior+1} (Maior Acurácia: {acuracias_mlp[idx_maior]:.2f}%)',
    f'Rodada {idx_menor+1} (Menor Acurácia: {acuracias_mlp[idx_menor]:.2f}%)'
]

# =============================================================================
# FUNÇÃO PARA CRIAR MATRIZ DE CONFUSÃO
# =============================================================================
def criar_matriz_confusao(Y_true, Y_pred, n_classes):
    matriz = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(Y_true)):
        matriz[Y_true[i], Y_pred[i]] += 1
    return matriz

# =============================================================================
# FUNÇÃO DE ACURÁCIA COM PREDIÇÕES
# =============================================================================
def calcular_acuracia_completa(modelo, X_test, Y_test, tipo):
    N_test = X_test.shape[1]
    acertos = 0
    Y_pred = np.zeros(N_test, dtype=int)
    Y_true = np.zeros(N_test, dtype=int)
    
    if tipo == 'linear':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            u_k = modelo.W @ x_k
            Y_pred[k] = np.argmax(u_k)
            Y_true[k] = np.argmax(Y_test[:, k])
            if Y_pred[k] == Y_true[k]:
                acertos += 1
    
    elif tipo == 'madaline':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            y_k = modelo.forward(x_k)
            Y_pred[k] = np.argmax(y_k)
            Y_true[k] = np.argmax(Y_test[:, k])
            if Y_pred[k] == Y_true[k]:
                acertos += 1
    
    elif tipo == 'mlp':
        X_test_bias = np.vstack((-np.ones((1, N_test)), X_test))
        for k in range(N_test):
            x_k = X_test_bias[:, k].reshape(-1, 1)
            modelo.forward(x_k)
            Y_pred[k] = np.argmax(modelo.y[-1])
            Y_true[k] = np.argmax(Y_test[:, k])
            if Y_pred[k] == Y_true[k]:
                acertos += 1
    
    acuracia = 100 * acertos / N_test
    return acuracia, Y_pred, Y_true

# =============================================================================
# ETAPA 2: ANÁLISE DETALHADA DAS RODADAS EXTREMAS
# =============================================================================
print(f"\n{'='*60}")
print("ETAPA 2: ANÁLISE DETALHADA DAS RODADAS EXTREMAS")
print("="*60)

resultados_analise = {}

for idx_rodada, r in enumerate(rodadas_extremas):
    print(f"\n{'='*60}")
    print(f"ANALISANDO {labels_rodadas[idx_rodada]}")
    print("="*60)
    
    np.random.seed(r)
    idx = np.random.permutation(X_norm.shape[1])
    n_train = int(0.8 * X_norm.shape[1])
    
    X_train = X_norm[:, idx[:n_train]]
    Y_train = Y_onehot[:, idx[:n_train]]
    X_test = X_norm[:, idx[n_train:]]
    Y_test = Y_onehot[:, idx[n_train:]]
    
    resultados_analise[r] = {}
    
    # Perceptron
    print("\nTreinando Perceptron...")
    per = PerceptronMulticlasse(X_train, Y_train, learning_rate=1e-3, max_epochs=500)
    per.fit()
    acc_per, Y_pred_per, Y_true_per = calcular_acuracia_completa(per, X_test, Y_test, 'linear')
    matriz_per = criar_matriz_confusao(Y_true_per, Y_pred_per, C)
    resultados_analise[r]['Perceptron'] = {
        'hist': per.hist_eqm,
        'acc': acc_per,
        'matriz': matriz_per
    }
    print(f"  Acurácia: {acc_per:.2f}%")
    
    # MADALINE ✅ CORRIGIDO: learning_rate=5e-3
    print("Treinando MADALINE...")
    mad = MADALINEMulticlasse(X_train, Y_train, n_hidden=30, learning_rate=5e-3, max_epochs=1000, tol=1e-6)
    mad.fit()
    acc_mad, Y_pred_mad, Y_true_mad = calcular_acuracia_completa(mad, X_test, Y_test, 'madaline')
    matriz_mad = criar_matriz_confusao(Y_true_mad, Y_pred_mad, C)
    resultados_analise[r]['MADALINE'] = {
        'hist': mad.hist_eqm,
        'acc': acc_mad,
        'matriz': matriz_mad
    }
    print(f"  Acurácia: {acc_mad:.2f}%")
    
    # MLP
    print("Treinando MLP...")
    mlp = MultilayerPerceptron(X_train, Y_train, topology=[50, 30], learning_rate=1e-2, max_epoch=500, tol=1e-8)
    mlp.fit()
    acc_mlp, Y_pred_mlp, Y_true_mlp = calcular_acuracia_completa(mlp, X_test, Y_test, 'mlp')
    matriz_mlp = criar_matriz_confusao(Y_true_mlp, Y_pred_mlp, C)
    resultados_analise[r]['MLP'] = {
        'hist': mlp.hist_eqm,
        'acc': acc_mlp,
        'matriz': matriz_mlp
    }
    print(f"  Acurácia: {acc_mlp:.2f}%")

# =============================================================================
# PLOTAR CURVAS DE APRENDIZADO
# =============================================================================
print(f"\n{'='*60}")
print("GERANDO CURVAS DE APRENDIZADO")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx_rodada, r in enumerate(rodadas_extremas):
    for idx_modelo, modelo in enumerate(['Perceptron', 'MADALINE', 'MLP']):
        ax = axes[idx_rodada, idx_modelo]
        hist = resultados_analise[r][modelo]['hist']
        acc = resultados_analise[r][modelo]['acc']
        
        if len(hist) > 0:
            ax.plot(hist, linewidth=2)
            ax.set_title(f"{modelo}\n{labels_rodadas[idx_rodada].split('(')[0].strip()}\nAcurácia: {acc:.2f}%", fontsize=10)
            ax.set_xlabel("Épocas")
            ax.set_ylabel("EQM")
            ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('curvas_aprendizado.png', dpi=150, bbox_inches='tight')
print("✓ Curvas salvas: curvas_aprendizado.png")
plt.show()

# =============================================================================
# PLOTAR MATRIZES DE CONFUSÃO
# =============================================================================
print(f"\nGERANDO MATRIZES DE CONFUSÃO")
print("="*60)

for idx_rodada, r in enumerate(rodadas_extremas):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx_modelo, modelo in enumerate(['Perceptron', 'MADALINE', 'MLP']):
        ax = axes[idx_modelo]
        matriz = resultados_analise[r][modelo]['matriz']
        acc = resultados_analise[r][modelo]['acc']
        
        sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=pessoas, yticklabels=pessoas, 
                    ax=ax, cbar_kws={'label': 'Frequência'})
        ax.set_title(f"{modelo}\n{labels_rodadas[idx_rodada].split('(')[0].strip()}\nAcurácia: {acc:.2f}%", fontsize=11)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        ax.tick_params(axis='both', labelsize=7)
    
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_rodada_{r+1}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Matriz salva: matriz_confusao_rodada_{r+1}.png")
    plt.show()

# =============================================================================
# ANÁLISE DE CONFUSÕES
# =============================================================================
print(f"\n{'='*60}")
print("ANÁLISE DAS CONFUSÕES")
print("="*60)

for modelo in ['Perceptron', 'MADALINE', 'MLP']:
    print(f"\n{modelo}:")
    
    erros_comuns = {}
    
    for r in rodadas_extremas:
        matriz = resultados_analise[r][modelo]['matriz']
        
        for i in range(C):
            for j in range(C):
                if i != j and matriz[i, j] > 0:
                    par = (pessoas[i], pessoas[j])
                    if par not in erros_comuns:
                        erros_comuns[par] = 0
                    erros_comuns[par] += matriz[i, j]
    
    if erros_comuns:
        top_erros = sorted(erros_comuns.items(), key=lambda x: x[1], reverse=True)[:3]
        print("  Confusões mais frequentes:")
        for (p1, p2), count in top_erros:
            print(f"    {p1} → {p2}: {count} vezes")
    else:
        print("  Nenhuma confusão detectada")

print(f"\n{'='*60}")
print("✅ Análise concluída!")
print("="*60)