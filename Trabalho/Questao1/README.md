# Trabalho IA - Questão 1: Classificação com Redes Neurais Artificiais

## Descrição do Projeto

Este projeto implementa e compara três modelos de Redes Neurais Artificiais para classificação de dados do dataset "Spiral":

1. **Perceptron Simples** (Rosenblatt, 1958)
2. **ADALINE** (Adaptive Linear Neuron - Widrow-Hoff, 1960)
3. **MLP** (Multilayer Perceptron - Backpropagation)

## Estrutura do Projeto

```
Questao1/
├── questao1.py                           # Classes base (Perceptron, ADALINE, MLP)
├── analise_underfit_overfit.py           # Análise de diferentes topologias MLP
├── validacao_500_rodadas.py              # Validação com 500 rodadas
├── validacao_rapida.py                   # Teste rápido (10 rodadas)
├── spiral_d.csv                          # Dataset original
├── 1_visualizacao_inicial.png            # Visualização dos dados
├── 2_curvas_aprendizado_comparacao.png   # Curvas de aprendizado
├── 3_matrizes_confusao_comparacao.png    # Matrizes de confusão
├── 4_metricas_comparacao.png             # Comparação de métricas
├── 5_fronteiras_decisao_comparacao.png   # Fronteiras de decisão
└── README.md                             # Este arquivo
```

## Dataset

**spiral_d.csv**: 1400 amostras, 2 features, 2 classes (+1, -1)
- Classe +1: 1000 amostras (71.4%)
- Classe -1: 400 amostras (28.6%)
- Normalização: Min-Max para [-1, 1]

## Modelos Implementados

### 1. Perceptron Simples

**Baseado em**: Rosenblatt, Frank. "The perceptron: a probabilistic model for information storage and organization in the brain." Psychological review 65.6 (1958): 386.

**Características**:
- Função de ativação: `sinal(u)` (degrau bipolar)
- Regra de aprendizagem: `w(t+1) = w(t) + η·e(t)·x(t)`
- Erro: `e(t) = d(t) - y(t)`
- Learning rate (η): 0.01
- Max épocas: 1000

**Limitações**: Apenas problemas linearmente separáveis

### 2. ADALINE

**Baseado em**: Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits.

**Características**:
- Regra LMS (Least Mean Squares)
- Regra de aprendizagem: `w(t+1) = w(t) + η·e(t)·x(t)`
- Erro: `e(t) = d(t) - u(t)` (erro contínuo)
- EQM: `(1/2N)·Σ(d - u)²`
- Learning rate (η): 0.001
- Max épocas: 1000
- Tolerância: 1e-5

**Vantagem**: Minimização do erro quadrático médio

### 3. MLP (Multilayer Perceptron)

**Baseado em**: Rumelhart, D. E., et al. "Parallel distributed processing." New York: IEEE (1988).

**Características**:
- Topologia: MLP(2, 20, 2) = 102 parâmetros
  - Entrada: 2 features
  - Camada oculta: 20 neurônios
  - Saída: 2 neurônios (one-hot encoding)
- Função de ativação: `tanh` (tangente hiperbólica)
- Regra Delta Generalizada (Backpropagation)
- Learning rate (η): 0.01
- Max épocas: 500
- Tolerância: 1e-6

**Fórmulas**:
- Forward: `i^(L) = W^(L) · y^(L-1)`
- Backward: `δ^(L) = g'(i^(L)) ⊙ (W^(L+1))^T · δ^(L+1)`
- Atualização: `W^(L) ← W^(L) + η·δ^(L) ⊗ y^(L-1)`

## Hiperparâmetros

### Justificativas (Baseadas na Teoria - Prof. Paulo Cirillo)

**Perceptron**:
- η = 0.01: Valor moderado (0 < η ≤ 1) para convergência estável
- Critério de parada: Sem erros de classificação em uma época completa

**ADALINE**:
- η = 0.001: Menor que Perceptron devido ao gradiente contínuo
- Critério: |EQM_atual - EQM_anterior| ≤ ε

**MLP**:
- Topologia [20]: Escolhida pela regra `q ≈ (p+m)/2 = (2+2)/2 ≈ 2-20`
- tanh: Totalmente diferenciável, compatível com labels bipolares [-1,+1]
- Número de parâmetros: `Z = (p+1)·q + (q+1)·m = 3·20 + 21·2 = 102`

## Métricas de Avaliação

Todas calculadas **manualmente** conforme requisitos:

1. **Acurácia**: `(VP + VN) / (VP + VN + FP + FN)`
2. **Sensibilidade** (Recall): `VP / (VP + FN)`
3. **Especificidade**: `VN / (VN + FP)`
4. **Precisão**: `VP / (VP + FP)`
5. **F1-Score**: `2 · (Precisão · Sensibilidade) / (Precisão + Sensibilidade)`

## Resultados Preliminares (10 rodadas, 100 épocas)

| Modelo     | Acurácia    | Sensibilidade | Especificidade | Precisão    | F1-Score    |
|------------|-------------|---------------|----------------|-------------|-------------|
| Perceptron | 72.07 ± 6.24% | 79.70 ± 17.29% | 52.30 ± 24.50% | 81.88 ± 4.84% | 79.33 ± 6.99% |
| ADALINE    | 77.93 ± 1.99% | 87.40 ± 2.92%  | 54.79 ± 4.25%  | 82.66 ± 2.30% | 84.91 ± 1.52% |
| **MLP**    | **80.68 ± 2.05%** | **92.56 ± 2.33%** | 51.61 ± 5.07% | **82.51 ± 2.55%** | **87.20 ± 1.44%** |

### Análise:
- ✅ **MLP** apresentou melhor desempenho geral (80.68%)
- ✅ **Alta sensibilidade** (92.56%) - boa detecção da classe +1
- ⚠️ **Baixa especificidade** (~50-54%) - dificuldade com classe -1
- ✅ **MLP mais estável** (menor desvio-padrão)

## Análise de Underfitting/Overfitting

### Topologias Testadas (MLP):

| Topologia          | Acurácia | EQM Final  | Observação    |
|--------------------|----------|------------|---------------|
| [2] neurônios      | 73.21%   | 0.2175     | **Underfitting** |
| [5] neurônios      | 66.50%   | 0.2375     | Underfitting  |
| [20] neurônios     | 59.00%   | 0.2727     | **Balanceado** |
| [100] neurônios    | 55.50%   | 0.2591     | Overfitting   |
| [200] neurônios    | 32.50%   | 0.3271     | **Overfitting severo** |

### Conclusão:
- Rede muito simples (2-5 neurônios): **Underfitting**
- Rede muito complexa (100-200 neurônios): **Overfitting**
- **Ponto ótimo**: ~20 neurônios na camada oculta

## Execução

### Pré-requisitos
```bash
pip install numpy matplotlib seaborn
```

### Scripts Disponíveis

1. **Visualização inicial e análise exploratória**:
```bash
python3 questao1.py
```

2. **Análise de underfitting/overfitting**:
```bash
python3 analise_underfit_overfit.py
```

3. **Validação rápida (10 rodadas)**:
```bash
python3 validacao_rapida.py
```

4. **Validação completa (500 rodadas)** ⚠️ Tempo estimado: 3-5 horas:
```bash
python3 validacao_500_rodadas.py
```

## Arquivos Gerados

### Visualizações
- `1_visualizacao_inicial.png`: Distribuição dos dados originais
- `2_curvas_aprendizado_comparacao.png`: EQM vs Épocas
- `3_matrizes_confusao_comparacao.png`: Matrizes de confusão
- `4_metricas_comparacao.png`: Comparação de métricas
- `5_fronteiras_decisao_comparacao.png`: Fronteiras de decisão

### Dados
- `results_teste.pkl`: Resultados serializados (10 rodadas)
- `resultados_500_rodadas.pkl`: Resultados completos (500 rodadas)

## Referências

1. **Rosenblatt, F.** (1958). The perceptron: a probabilistic model for information storage and organization in the brain. *Psychological review*, 65(6), 386.

2. **Widrow, B., & Hoff, M. E.** (1960). Adaptive switching circuits. *IRE WESCON Convention Record*, 96-104.

3. **Rumelhart, D. E., et al.** (1988). Parallel distributed processing. New York: IEEE.

4. **Material do Prof. Paulo Cirillo Souza Barbosa** - Universidade de Fortaleza (UNIFOR), Centro de Ciências Tecnológicas (CCT).

## Autores

- Leonardo de Saboia
- Disciplina: Inteligência Artificial Computacional (T296)
- Instituição: Universidade de Fortaleza (UNIFOR)

## Licença

Este projeto é de uso acadêmico.
