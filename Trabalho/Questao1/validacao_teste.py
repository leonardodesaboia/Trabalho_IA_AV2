import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from questao1 import Perceptron, ADALINE, MultilayerPerceptron
from questao1 import confusion_matrix_manual, calculate_metrics
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VALIDACAO TESTE - 50 RODADAS")
print("=" * 80)

# Carregar dados
data = np.loadtxt("spiral_d.csv", delimiter=',')
X = data[:, :-1]
Y = data[:, -1]

# Normalização Min-Max para [-1, 1]
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

print(f"\nDados: {X.shape[0]} amostras, {X.shape[1]} features")
print(f"Normalizacao: [{X_norm.min():.2f}, {X_norm.max():.2f}]")

# Hiperparâmetros
R = 50  # Teste com 50 rodadas
train_ratio = 0.8

print(f"Rodadas: {R}")
print(f"Split: {train_ratio*100:.0f}%/{(1-train_ratio)*100:.0f}%")

# Estruturas para resultados
results = {
    'Perceptron': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                               'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []},
    'ADALINE': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                           'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []},
    'MLP': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                       'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []}
}

print("\nIniciando validacao...")
np.random.seed(42)

for rodada in range(R):
    if (rodada + 1) % 10 == 0:
        print(f"Rodada {rodada + 1}/{R}...")

    # Split treino/teste
    indices = np.random.permutation(len(X_norm))
    train_size = int(len(X_norm) * train_ratio)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X_norm[train_indices].T
    Y_train = Y[train_indices]
    X_test = X_norm[test_indices].T
    Y_test = Y[test_indices]

    # Y one-hot para MLP
    Y_train_onehot = np.vstack(((Y_train == 1).astype(float), (Y_train == -1).astype(float)))

    # PERCEPTRON
    perceptron = Perceptron(X_train, Y_train, learning_rate=0.01, max_epochs=1000)
    perceptron.fit(verbose=False)
    pred_p = perceptron.predict(X_test)
    cm_p = confusion_matrix_manual(Y_test, pred_p)
    met_p = calculate_metrics(cm_p)

    results['Perceptron']['metrics']['accuracy'].append(met_p['accuracy'])
    results['Perceptron']['metrics']['sensitivity'].append(met_p['sensitivity'])
    results['Perceptron']['metrics']['specificity'].append(met_p['specificity'])
    results['Perceptron']['metrics']['precision'].append(met_p['precision'])
    results['Perceptron']['metrics']['f1_score'].append(met_p['f1_score'])
    results['Perceptron']['confusion_matrices'].append(cm_p)
    results['Perceptron']['hist_eqm'].append(perceptron.hist_eqm)

    # ADALINE
    adaline = ADALINE(X_train, Y_train, learning_rate=0.001, max_epochs=1000, tol=1e-5)
    adaline.fit(verbose=False)
    pred_a = adaline.predict(X_test)
    cm_a = confusion_matrix_manual(Y_test, pred_a)
    met_a = calculate_metrics(cm_a)

    results['ADALINE']['metrics']['accuracy'].append(met_a['accuracy'])
    results['ADALINE']['metrics']['sensitivity'].append(met_a['sensitivity'])
    results['ADALINE']['metrics']['specificity'].append(met_a['specificity'])
    results['ADALINE']['metrics']['precision'].append(met_a['precision'])
    results['ADALINE']['metrics']['f1_score'].append(met_a['f1_score'])
    results['ADALINE']['confusion_matrices'].append(cm_a)
    results['ADALINE']['hist_eqm'].append(adaline.hist_eqm)

    # MLP
    mlp = MultilayerPerceptron(X_train, Y_train_onehot, [20],
                               learning_rate=0.01, max_epoch=500, tol=1e-6, verbose=False)
    mlp.fit()
    pred_mlp_probs = mlp.predict(X_test)
    pred_mlp_classes = np.argmax(pred_mlp_probs, axis=1)
    pred_mlp = np.where(pred_mlp_classes == 0, 1, -1)
    cm_mlp = confusion_matrix_manual(Y_test, pred_mlp)
    met_mlp = calculate_metrics(cm_mlp)

    results['MLP']['metrics']['accuracy'].append(met_mlp['accuracy'])
    results['MLP']['metrics']['sensitivity'].append(met_mlp['sensitivity'])
    results['MLP']['metrics']['specificity'].append(met_mlp['specificity'])
    results['MLP']['metrics']['precision'].append(met_mlp['precision'])
    results['MLP']['metrics']['f1_score'].append(met_mlp['f1_score'])
    results['MLP']['confusion_matrices'].append(cm_mlp)
    results['MLP']['hist_eqm'].append(mlp.hist_eqm)

print("\nValidacao concluida!")

# Análise estatística
print("\n" + "=" * 80)
print("ESTATISTICAS (50 RODADAS)")
print("=" * 80)

for model_name in ['Perceptron', 'ADALINE', 'MLP']:
    print(f"\n{model_name}:")
    print("-" * 80)
    for metric_name in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
        values = np.array(results[model_name]['metrics'][metric_name])
        print(f"  {metric_name.capitalize():<15}: Media={np.mean(values)*100:>6.2f}% | "
              f"Std={np.std(values)*100:>6.2f}% | Max={np.max(values)*100:>6.2f}% | "
              f"Min={np.min(values)*100:>6.2f}%")

print("\n" + "=" * 80)
print("Teste completado! Arquivos salvos.")
print("=" * 80)
