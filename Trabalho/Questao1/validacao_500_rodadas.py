import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from questao1 import Perceptron, ADALINE, MultilayerPerceptron
from questao1 import confusion_matrix_manual, calculate_metrics, plot_confusion_matrix
import warnings
import pickle
warnings.filterwarnings('ignore')

print("=" * 80)
print("VALIDACAO COM 500 RODADAS - PERCEPTRON, ADALINE E MLP")
print("=" * 80)

# Carregar dados
data = np.loadtxt("spiral_d.csv", delimiter=',')
X = data[:, :-1]
Y = data[:, -1]

# Normalização
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

print(f"\nDados carregados: {X.shape[0]} amostras")
print(f"Features normalizadas no intervalo [{X_norm.min():.2f}, {X_norm.max():.2f}]")

# Hiperparâmetros definidos baseados na teoria do Prof. Paulo Cirillo
R = 500  # Número de rodadas
train_ratio = 0.8

print(f"\nNumero de rodadas: {R}")
print(f"Proporcao treino/teste: {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")

# Discussão dos hiperparâmetros baseada na teoria
print("\n" + "=" * 80)
print("ESCOLHA DOS HIPERPARAMETROS (Baseado na Teoria - Prof. Paulo Cirillo)")
print("=" * 80)
print("""
1. PERCEPTRON SIMPLES (Regra de Rosenblatt, 1958):
   - Learning Rate (η): 0.01
   - Max Epochs: 1000
   - Regra: w(t+1) = w(t) + η·e(t)·x(t), onde e(t) = d(t) - y(t)
   - Funcao de ativacao: sinal (degrau: +1 se u>=0, -1 caso contrario)
   Justificativa: η moderado (0 < η ≤ 1) para convergencia. Criterio de parada
   quando nao houver mais erros de classificacao em uma epoca completa.

2. ADALINE (Regra de Widrow-Hoff/LMS):
   - Learning Rate (η): 0.001
   - Max Epochs: 1000
   - Tolerancia: 1e-5
   - Regra: w(t+1) = w(t) + η·e(t)·x(t), onde e(t) = d(t) - u(t)
   - Criterio: |EQM_atual - EQM_anterior| ≤ ε
   Justificativa: η menor que Perceptron devido ao gradiente continuo.
   EQM = (1/2N)·Σ(d - u)² usado para monitorar convergencia.

3. MLP - Perceptron de Multiplas Camadas (Backpropagation):
   - Topologia: [20] (1 camada oculta com 20 neuronios)
   - Learning Rate (η): 0.01
   - Max Epochs: 500
   - Tolerancia: 1e-6
   - Funcao ativacao: tanh (tangente hiperbolica bipolar [-1,1])
   - Regra Delta Generalizada: W(L) = W(L) + η·δ(L) ⊗ y(L-1)
   Justificativa: Topologia [20] escolhida pela regra q ≈ (p+m)/2 = (2+2)/2 ≈ 2-20.
   Funcao tanh escolhida por ser totalmente diferenciavel e compativel com
   labels bipolares (-1, +1). Max epochs reduzido por ser problema nao-linear.

   Numero de parametros da rede MLP(2,20,2):
   Z = (p+1)·q1 + (q1+1)·m = (2+1)·20 + (20+1)·2 = 60 + 42 = 102 parametros
""")

# Estruturas para armazenar resultados
results = {
    'Perceptron': {
        'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                    'precision': [], 'f1_score': []},
        'confusion_matrices': [],
        'hist_eqm': []
    },
    'ADALINE': {
        'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                    'precision': [], 'f1_score': []},
        'confusion_matrices': [],
        'hist_eqm': []
    },
    'MLP': {
        'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                    'precision': [], 'f1_score': []},
        'confusion_matrices': [],
        'hist_eqm': []
    }
}

# ============================================================
# EXECUTAR 500 RODADAS
# ============================================================

print("\n" + "=" * 80)
print("INICIANDO VALIDACAO COM 500 RODADAS")
print("=" * 80)

np.random.seed(42)  # Para reproducibilidade inicial

for rodada in range(R):
    if (rodada + 1) % 50 == 0:
        print(f"Processando rodada {rodada + 1}/{R}...")

    # Embaralhar e particionar dados
    indices = np.random.permutation(len(X_norm))
    train_size = int(len(X_norm) * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X_norm[train_indices].T
    Y_train = Y[train_indices]
    X_test = X_norm[test_indices].T
    Y_test = Y[test_indices]

    # Preparar Y para MLP (one-hot encoding)
    Y_train_onehot = np.vstack((
        (Y_train == 1).astype(float),
        (Y_train == -1).astype(float)
    ))

    # ============================================================
    # TREINAR E AVALIAR PERCEPTRON
    # ============================================================
    perceptron = Perceptron(X_train, Y_train, learning_rate=0.01, max_epochs=1000)
    perceptron.fit(verbose=False)

    pred_perceptron = perceptron.predict(X_test)
    cm_perceptron = confusion_matrix_manual(Y_test, pred_perceptron)
    metrics_perceptron = calculate_metrics(cm_perceptron)

    results['Perceptron']['metrics']['accuracy'].append(metrics_perceptron['accuracy'])
    results['Perceptron']['metrics']['sensitivity'].append(metrics_perceptron['sensitivity'])
    results['Perceptron']['metrics']['specificity'].append(metrics_perceptron['specificity'])
    results['Perceptron']['metrics']['precision'].append(metrics_perceptron['precision'])
    results['Perceptron']['metrics']['f1_score'].append(metrics_perceptron['f1_score'])
    results['Perceptron']['confusion_matrices'].append(cm_perceptron)
    results['Perceptron']['hist_eqm'].append(perceptron.hist_eqm)

    # ============================================================
    # TREINAR E AVALIAR ADALINE
    # ============================================================
    adaline = ADALINE(X_train, Y_train, learning_rate=0.001, max_epochs=1000, tol=1e-5)
    adaline.fit(verbose=False)

    pred_adaline = adaline.predict(X_test)
    cm_adaline = confusion_matrix_manual(Y_test, pred_adaline)
    metrics_adaline = calculate_metrics(cm_adaline)

    results['ADALINE']['metrics']['accuracy'].append(metrics_adaline['accuracy'])
    results['ADALINE']['metrics']['sensitivity'].append(metrics_adaline['sensitivity'])
    results['ADALINE']['metrics']['specificity'].append(metrics_adaline['specificity'])
    results['ADALINE']['metrics']['precision'].append(metrics_adaline['precision'])
    results['ADALINE']['metrics']['f1_score'].append(metrics_adaline['f1_score'])
    results['ADALINE']['confusion_matrices'].append(cm_adaline)
    results['ADALINE']['hist_eqm'].append(adaline.hist_eqm)

    # ============================================================
    # TREINAR E AVALIAR MLP
    # ============================================================
    mlp = MultilayerPerceptron(X_train, Y_train_onehot, [20],
                               learning_rate=0.01, max_epoch=500, tol=1e-6, verbose=False)
    mlp.fit()

    pred_mlp_probs = mlp.predict(X_test)
    pred_mlp_classes = np.argmax(pred_mlp_probs, axis=1)
    pred_mlp = np.where(pred_mlp_classes == 0, 1, -1)

    cm_mlp = confusion_matrix_manual(Y_test, pred_mlp)
    metrics_mlp = calculate_metrics(cm_mlp)

    results['MLP']['metrics']['accuracy'].append(metrics_mlp['accuracy'])
    results['MLP']['metrics']['sensitivity'].append(metrics_mlp['sensitivity'])
    results['MLP']['metrics']['specificity'].append(metrics_mlp['specificity'])
    results['MLP']['metrics']['precision'].append(metrics_mlp['precision'])
    results['MLP']['metrics']['f1_score'].append(metrics_mlp['f1_score'])
    results['MLP']['confusion_matrices'].append(cm_mlp)
    results['MLP']['hist_eqm'].append(mlp.hist_eqm)

print("\n500 rodadas concluidas!")

# Salvar resultados
with open('resultados_500_rodadas.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Resultados salvos em: resultados_500_rodadas.pkl")

# ============================================================
# ANALISE ESTATISTICA
# ============================================================

print("\n" + "=" * 80)
print("ANALISE ESTATISTICA FINAL")
print("=" * 80)

for model_name in ['Perceptron', 'ADALINE', 'MLP']:
    print(f"\n{model_name}:")
    print("-" * 80)

    for metric_name in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
        values = np.array(results[model_name]['metrics'][metric_name])

        mean_val = np.mean(values)
        std_val = np.std(values)
        max_val = np.max(values)
        min_val = np.min(values)

        print(f"  {metric_name.capitalize():<15}: "
              f"Media={mean_val*100:>6.2f}% | "
              f"Std={std_val*100:>6.2f}% | "
              f"Max={max_val*100:>6.2f}% | "
              f"Min={min_val*100:>6.2f}%")

# ============================================================
# TABELAS RESUMO
# ============================================================

print("\n" + "=" * 80)
print("TABELAS RESUMO DAS METRICAS")
print("=" * 80)

metric_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
metric_labels = ['Acuracia', 'Sensibilidade', 'Especificidade', 'Precisao', 'F1-Score']

for metric_name, metric_label in zip(metric_names, metric_labels):
    print(f"\n{metric_label}:")
    print(f"{'Modelos':<25} {'Media':<12} {'Desvio-padrao':<15} {'Maior Valor':<15} {'Menor Valor':<15}")
    print("-" * 82)

    for model_name in ['Perceptron', 'ADALINE', 'MLP']:
        values = np.array(results[model_name]['metrics'][metric_name])
        mean_val = np.mean(values) * 100
        std_val = np.std(values) * 100
        max_val = np.max(values) * 100
        min_val = np.min(values) * 100

        print(f"{model_name:<25} {mean_val:>10.2f}% {std_val:>13.2f}% "
              f"{max_val:>13.2f}% {min_val:>13.2f}%")

print("\nContinuando para visualizacoes e analise de extremos...")
