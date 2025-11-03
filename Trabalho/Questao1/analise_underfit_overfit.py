import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from questao1 import MultilayerPerceptron, confusion_matrix_manual, calculate_metrics, plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANALISE DE UNDERFITTING E OVERFITTING - MLP")
print("=" * 80)

# Carregar dados
data = np.loadtxt("spiral_d.csv", delimiter=',')
X = data[:, :-1]
Y = data[:, -1]

# Normalização
X_norm = 2 * (X - np.min(X)) / (np.max(X) - np.min(X)) - 1

# Preparar Y no formato one-hot encoding (2 x N)
Y_onehot = np.vstack((
    (Y == 1).astype(float),
    (Y == -1).astype(float)
))

print(f"\nDados carregados: {X.shape[0]} amostras")
print(f"Features normalizadas no intervalo [{X_norm.min():.2f}, {X_norm.max():.2f}]")

# ============================================================
# TESTAR DIFERENTES TOPOLOGIAS
# ============================================================

print("\n" + "=" * 80)
print("TESTANDO DIFERENTES TOPOLOGIAS DE MLP")
print("=" * 80)

# Definir topologias a serem testadas
topologies = {
    'Underfitting (2 neuronios)': [2],
    'Underfitting (5 neuronios)': [5],
    'Balanceado (20 neuronios)': [20],
    'Overfitting (100 neuronios)': [100],
    'Overfitting (200 neuronios)': [200],
}

# Hiperparâmetros
learning_rate = 0.01
max_epochs = 500  # Reduzido de 1000 para 500
tolerance = 1e-6

results = {}

for name, topology in topologies.items():
    print(f"\n{'-' * 80}")
    print(f"Testando: {name}")
    print(f"Topologia: {topology}")
    print(f"Hiperparametros: LR={learning_rate}, Max Epochs={max_epochs}, Tol={tolerance}")

    # Criar e treinar MLP
    mlp = MultilayerPerceptron(
        X_norm.T,
        Y_onehot,
        topology.copy(),  # Copiar para não modificar a lista original
        learning_rate=learning_rate,
        max_epoch=max_epochs,
        tol=tolerance,
        verbose=False
    )

    print("Treinando...")
    mlp.fit()

    # Fazer predições
    predictions = mlp.predict(X_norm.T)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(Y_onehot, axis=0)

    # Converter para labels binários (-1, 1)
    pred_labels = np.where(pred_classes == 0, 1, -1)
    true_labels = Y

    # Calcular matriz de confusão
    cm = confusion_matrix_manual(true_labels, pred_labels)

    # Calcular métricas
    metrics = calculate_metrics(cm)

    # Salvar resultados
    results[name] = {
        'topology': topology,
        'mlp': mlp,
        'confusion_matrix': cm,
        'metrics': metrics,
        'hist_eqm': mlp.hist_eqm
    }

    print(f"Epocas treinadas: {len(mlp.hist_eqm)}")
    print(f"EQM final: {mlp.hist_eqm[-1]:.10f}")
    print(f"Acuracia: {metrics['accuracy']*100:.2f}%")
    print(f"Sensibilidade: {metrics['sensitivity']*100:.2f}%")
    print(f"Especificidade: {metrics['specificity']*100:.2f}%")
    print(f"Precisao: {metrics['precision']*100:.2f}%")
    print(f"F1-Score: {metrics['f1_score']:.4f}")

# ============================================================
# VISUALIZACAO DOS RESULTADOS
# ============================================================

print("\n" + "=" * 80)
print("GERANDO VISUALIZACOES")
print("=" * 80)

# 1. Curvas de aprendizado comparativas
plt.figure(figsize=(14, 8))
for name, result in results.items():
    plt.plot(result['hist_eqm'], label=name, linewidth=2)

plt.xlabel('Epocas', fontsize=12, fontweight='bold')
plt.ylabel('Erro Quadratico Medio (EQM)', fontsize=12, fontweight='bold')
plt.title('Comparacao de Curvas de Aprendizado - Diferentes Topologias',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('2_curvas_aprendizado_comparacao.png', dpi=300, bbox_inches='tight')
print("Salvo: 2_curvas_aprendizado_comparacao.png")
plt.close()

# 2. Matrizes de confusão para casos selecionados
# Selecionar: Underfitting (2 neurônios), Balanceado (20 neurônios), Overfitting (200 neurônios)
selected_cases = [
    'Underfitting (2 neuronios)',
    'Balanceado (20 neuronios)',
    'Overfitting (200 neuronios)'
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, case_name in enumerate(selected_cases):
    if case_name in results:
        cm = results[case_name]['confusion_matrix']
        metrics = results[case_name]['metrics']

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Classe +1', 'Classe -1'],
                    yticklabels=['Classe +1', 'Classe -1'],
                    ax=axes[idx], cbar=True)

        axes[idx].set_xlabel('Predito', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Real', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{case_name}\nAcuracia: {metrics["accuracy"]*100:.2f}%',
                            fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('3_matrizes_confusao_comparacao.png', dpi=300, bbox_inches='tight')
print("Salvo: 3_matrizes_confusao_comparacao.png")
plt.close()

# 3. Gráfico de barras com métricas
metrics_names = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score']
metrics_labels = ['Acuracia', 'Sensibilidade', 'Especificidade', 'Precisao', 'F1-Score']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, (metric_name, metric_label) in enumerate(zip(metrics_names, metrics_labels)):
    model_names = list(results.keys())
    metric_values = [results[name]['metrics'][metric_name] * 100 for name in model_names]

    bars = axes[idx].bar(range(len(model_names)), metric_values, color='skyblue', edgecolor='black')

    # Colorir barras baseado no desempenho
    for i, bar in enumerate(bars):
        if metric_values[i] >= 90:
            bar.set_color('#2ECC71')  # Verde para bom desempenho
        elif metric_values[i] >= 70:
            bar.set_color('#F39C12')  # Laranja para médio
        else:
            bar.set_color('#E74C3C')  # Vermelho para baixo

    axes[idx].set_xticks(range(len(model_names)))
    axes[idx].set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
    axes[idx].set_ylabel(f'{metric_label} (%)', fontsize=10, fontweight='bold')
    axes[idx].set_title(f'Comparacao de {metric_label}', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].set_ylim(0, 105)

    # Adicionar valores nas barras
    for i, v in enumerate(metric_values):
        axes[idx].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

# Esconder subplot extra
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('4_metricas_comparacao.png', dpi=300, bbox_inches='tight')
print("Salvo: 4_metricas_comparacao.png")
plt.close()

# 4. Tabela resumo das métricas
print("\n" + "=" * 80)
print("TABELA RESUMO DAS METRICAS")
print("=" * 80)
print(f"{'Topologia':<35} {'Acuracia':<12} {'Sensib.':<12} {'Especif.':<12} {'Precisao':<12} {'F1-Score':<12}")
print("-" * 115)
for name, result in results.items():
    m = result['metrics']
    print(f"{name:<35} {m['accuracy']*100:>10.2f}% {m['sensitivity']*100:>10.2f}% "
          f"{m['specificity']*100:>10.2f}% {m['precision']*100:>10.2f}% {m['f1_score']:>10.4f}")

# 5. Análise de fronteira de decisão para casos selecionados
print("\n" + "=" * 80)
print("GERANDO FRONTEIRAS DE DECISAO")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, case_name in enumerate(selected_cases):
    if case_name in results:
        mlp = results[case_name]['mlp']

        # Criar grid
        x_min, x_max = X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5
        y_min, y_max = X_norm[:, 1].min() - 0.5, X_norm[:, 1].max() + 0.5
        h = 0.02
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predições
        Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()].T)
        Z_class = np.argmax(Z, axis=1).reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z_class, alpha=0.4, cmap='RdYlBu', levels=1)
        axes[idx].scatter(X_norm[Y == 1, 0], X_norm[Y == 1, 1],
                          c='#FF6B6B', marker='o', s=30, alpha=0.7,
                          edgecolors='black', linewidth=0.5, label='Classe +1')
        axes[idx].scatter(X_norm[Y == -1, 0], X_norm[Y == -1, 1],
                          c='#4ECDC4', marker='s', s=30, alpha=0.7,
                          edgecolors='black', linewidth=0.5, label='Classe -1')

        axes[idx].set_xlabel('Feature X (normalizada)', fontsize=10, fontweight='bold')
        axes[idx].set_ylabel('Feature Y (normalizada)', fontsize=10, fontweight='bold')
        axes[idx].set_title(f'{case_name}', fontsize=11, fontweight='bold')
        axes[idx].legend(loc='best', fontsize=8)
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('5_fronteiras_decisao_comparacao.png', dpi=300, bbox_inches='tight')
print("Salvo: 5_fronteiras_decisao_comparacao.png")
plt.close()

print("\n" + "=" * 80)
print("ANALISE DE UNDERFITTING/OVERFITTING COMPLETA")
print("=" * 80)
print("\nArquivos gerados:")
print("  2_curvas_aprendizado_comparacao.png")
print("  3_matrizes_confusao_comparacao.png")
print("  4_metricas_comparacao.png")
print("  5_fronteiras_decisao_comparacao.png")
