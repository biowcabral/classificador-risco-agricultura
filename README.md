# 🌾 Classificador de Risco de Desperdício Agrícola

Sistema completo de análise e classificação de risco de desperdício agrícola utilizando Machine Learning, com análise comparativa de 7 modelos e visualização interativa.

## 📊 Resultados Principais

- **Melhor Modelo:** Decision Tree - **98.58%** de Accuracy
- **Dados Analisados:** 13 anos (2012-2024), 124.137 registros, 399 municípios
- **Modelos Implementados:** 7 algoritmos de ML com análise comparativa completa

## 🏗️ Arquitetura do Projeto (MVC)

```
classificador-risco-agricultura/
│
├── 📁 models/                          # MODEL - Lógica de ML e Processamento
│   └── analise_rapida.py              # Script principal de análise ML
│
├── 📁 views/                           # VIEW - Interface e Visualização
│   └── dashboard_final.html           # Dashboard interativo principal
│
├── 📁 controllers/                     # CONTROLLER - Orquestração e Execução
│   └── executar_analise.py            # Script de controle e menu interativo
│
├── 📁 data/                            # Dados de Entrada e Saída
│   ├── VBP*.xls                       # Dados brutos VBP 2012-2024
│   ├── vbp*.xlsx                      # Dados brutos VBP recentes
│   ├── comparacao_modelos.csv         # Resultados comparativos
│   └── resultados_ml.json             # Resultados completos em JSON
│
├── 📁 outputs/                         # Visualizações e Gráficos
│   ├── comparacao_metricas.png        # Comparação de métricas
│   ├── confusion_matrix_melhor.png    # Matriz de confusão
│   ├── feature_importance.png         # Importância de features (XAI)
│   └── evolucao_temporal.png          # Evolução temporal
│
├── 📁 notebooks/                       # Notebooks Jupyter das Aulas
│   └── Aula/                          # 11 notebooks de ML utilizados
│
├── 📁 docs/                            # Documentação
│   ├── README_ML.md                   # Documentação técnica completa
│   ├── GUIA_RAPIDO.md                 # Guia rápido de uso
│   └── *.md                           # Outros documentos
│
└── 📁 obsoletos/                       # Arquivos Legados (não utilizados)
    ├── analise_temporal_agricultura.py
    ├── dashboard_*.html
    └── ...                            # Scripts auxiliares antigos
```

## 🚀 Como Executar

### Opção 1: Execução Rápida (Recomendado)

```bash
# Instalar dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

# Executar análise completa
python models/analise_rapida.py

# Abrir dashboard
start views/dashboard_final.html
```

### Opção 2: Menu Interativo

```bash
python controllers/executar_analise.py
```

## 📈 Modelos Implementados

| Posição | Modelo               | Accuracy | F1-Score | Tempo   |
|---------|---------------------|----------|----------|---------|
| 🏆 1º   | Decision Tree       | 98.58%   | 98.58%   | 5.29s   |
| 🥈 2º   | Gradient Boosting   | 98.44%   | 98.44%   | 107.51s |
| 🥉 3º   | Random Forest       | 97.84%   | 97.84%   | 19.05s  |
| 4º      | Extra Trees         | 65.78%   | 65.95%   | 6.37s   |
| 5º      | KNN                 | 63.23%   | 63.05%   | 12.98s  |
| 6º      | Naive Bayes         | 61.32%   | 59.48%   | 0.75s   |
| 7º      | Logistic Regression | 57.57%   | 57.62%   | 13.73s  |

## 🔍 Features Principais

- **Feature Engineering:** 7 features derivadas (produtividade, VBP por hectare, etc.)
- **Feature Selection:** SelectKBest, Feature Importance (XAI)
- **Análise Temporal:** Evolução de 13 anos de dados agrícolas
- **Interpretabilidade:** Análise detalhada do porquê de cada resultado

## 📚 Metodologias Aplicadas (11 Aulas)

1. **Iris Dataset** - KNN, Decision Tree
2. **Diabetes** - Logistic Regression, Random Forest
3. **Predictive Analytics** - Comparação de múltiplos modelos
4. **Machine Failure** - Ensemble Methods (Voting, Bagging, Boosting)
5. **Churn** - Neural Networks, SVM
6. **Breast Cancer** - Feature Selection (RFE, SFS)
7. **Wine Clustering** - K-Means
8. **Health Ageing** - Hierarchical Clustering
9. **Obesity** - PCA
10. **XAI** - SHAP, Feature Importance
11. **Groceries** - Association Rules

## 📊 Dashboard Interativo

O dashboard inclui 4 abas:

1. **📊 Visão Geral** - Estatísticas e comparação visual
2. **🤖 Modelos** - Tabela completa e métricas detalhadas
3. **🔍 Features** - Importância e seleção de atributos
4. **📖 Análise Detalhada** - Explicação completa de cada modelo

## 🎯 Análise de Resultados

### Por que Decision Tree venceu?

- ✅ Capturou perfeitamente os **thresholds naturais** (quartis de produção)
- ✅ **Interpretabilidade máxima** para stakeholders
- ✅ **Rápido** (5.29s) para produção
- ✅ Ideal para dados com **estrutura hierárquica clara**

### Aplicações Recomendadas

- **Produção:** Decision Tree (precisão + velocidade + interpretabilidade)
- **Pesquisa:** Gradient Boosting (máxima precisão)
- **Robustez:** Random Forest (equilibrado e resistente a outliers)

## 🛠️ Tecnologias Utilizadas

- **Python 3.x**
- **Pandas, NumPy** - Manipulação de dados
- **Scikit-learn** - Machine Learning
- **Matplotlib, Seaborn** - Visualizações
- **XGBoost** - Gradient Boosting avançado
- **Chart.js** - Gráficos interativos no dashboard

## 📝 Licença

Este projeto foi desenvolvido para análise de risco agrícola no Paraná.

## 👥 Contribuições

Sistema desenvolvido com base em 11 aulas práticas de Machine Learning, integrando múltiplas metodologias e técnicas avançadas.

---

**Última atualização:** Janeiro 2026
