import numpy as np
from questao1 import Perceptron, ADALINE, MultilayerPerceptron
from questao1 import confusion_matrix_manual, calculate_metrics
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VALIDACAO RAPIDA - 10 RODADAS (TESTE)")
print("=" * 80)

# Carregar dados
data = np.loadtxt("spiral_d.csv", delimiter=',')
X = data[:, :-1]
Y = data[:, -1]
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

print(f"Dados: {X.shape[0]} amostras")

# Hiperparâmetros OTIMIZADOS para velocidade
R = 10
train_ratio = 0.8

results = {
    'Perceptron': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                               'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []},
    'ADALINE': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                           'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []},
    'MLP': {'metrics': {'accuracy': [], 'sensitivity': [], 'specificity': [],
                       'precision': [], 'f1_score': []}, 'confusion_matrices': [], 'hist_eqm': []}
}

print("Iniciando validacao...")
np.random.seed(42)

for rodada in range(R):
    print(f"Rodada {rodada + 1}/{R}...", end=' ')

    # Split
    indices = np.random.permutation(len(X_norm))
    train_size = int(len(X_norm) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train = X_norm[train_indices].T
    Y_train = Y[train_indices]
    X_test = X_norm[test_indices].T
    Y_test = Y[test_indices]

    Y_train_onehot = np.vstack(((Y_train == 1).astype(float), (Y_train == -1).astype(float)))

    # PERCEPTRON - Reduzido para 100 épocas
    perceptron = Perceptron(X_train, Y_train, learning_rate=0.01, max_epochs=100)
    perceptron.fit(verbose=False)
    pred_p = perceptron.predict(X_test)
    cm_p = confusion_matrix_manual(Y_test, pred_p)
    met_p = calculate_metrics(cm_p)

    for key in met_p:
        results['Perceptron']['metrics'][key].append(met_p[key])
    results['Perceptron']['confusion_matrices'].append(cm_p)
    results['Perceptron']['hist_eqm'].append(perceptron.hist_eqm)
    print(f"P={met_p['accuracy']*100:.1f}%", end=' ')

    # ADALINE - Reduzido para 100 épocas
    adaline = ADALINE(X_train, Y_train, learning_rate=0.001, max_epochs=100, tol=1e-5)
    adaline.fit(verbose=False)
    pred_a = adaline.predict(X_test)
    cm_a = confusion_matrix_manual(Y_test, pred_a)
    met_a = calculate_metrics(cm_a)

    for key in met_a:
        results['ADALINE']['metrics'][key].append(met_a[key])
    results['ADALINE']['confusion_matrices'].append(cm_a)
    results['ADALINE']['hist_eqm'].append(adaline.hist_eqm)
    print(f"A={met_a['accuracy']*100:.1f}%", end=' ')

    # MLP - Reduzido para 100 épocas
    mlp = MultilayerPerceptron(X_train, Y_train_onehot, [20],
                               learning_rate=0.01, max_epoch=100, tol=1e-6, verbose=False)
    mlp.fit()
    pred_mlp_probs = mlp.predict(X_test)
    pred_mlp_classes = np.argmax(pred_mlp_probs, axis=1)
    pred_mlp = np.where(pred_mlp_classes == 0, 1, -1)
    cm_mlp = confusion_matrix_manual(Y_test, pred_mlp)
    met_mlp = calculate_metrics(cm_mlp)

    for key in met_mlp:
        results['MLP']['metrics'][key].append(met_mlp[key])
    results['MLP']['confusion_matrices'].append(cm_mlp)
    results['MLP']['hist_eqm'].append(mlp.hist_eqm)
    print(f"M={met_mlp['accuracy']*100:.1f}%")

print("\n" + "=" * 80)
print("ESTATISTICAS")
print("=" * 80)

for model_name in ['Perceptron', 'ADALINE', 'MLP']:
    print(f"\n{model_name}:")
    for metric_name in ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']:
        values = np.array(results[model_name]['metrics'][metric_name])
        print(f"  {metric_name.capitalize():<15}: {np.mean(values)*100:>6.2f}% ± {np.std(values)*100:>5.2f}%")

# Salvar
with open('results_teste.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResultados salvos em: results_teste.pkl")
