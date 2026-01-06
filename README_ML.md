# Sistema de Análise Temporal de Risco Agrícola - Guia de Execução

## 📋 Visão Geral

Este projeto implementa uma análise comparativa abrangente de metodologias de Machine Learning aplicadas à classificação de risco de desperdício agrícola, baseado em 11 aulas práticas de ML.

## 🎯 Metodologias Implementadas

### 📚 Classificação Supervisionada
- **Aula 1 (Iris)**: KNN, Decision Tree
- **Aula 1 (Diabetes)**: Logistic Regression, Random Forest
- **Aulas 1-2 (Predictive)**: Comparação de múltiplos classificadores
- **Aula 4 (Machine Failure)**: Neural Networks, SVM, Ensemble Methods
- **Aula 4 (Churn)**: Análise de churn com múltiplos modelos

### 🎭 Ensemble Methods (Aula 4)
- **Voting Classifiers**: Hard Voting e Soft Voting
- **Bagging**: Múltiplos classificadores com amostragem
- **Boosting**: AdaBoost, Gradient Boosting, XGBoost

### 🔍 Feature Selection (Aula 5 e 9)
- **RFE** (Recursive Feature Elimination)
- **SFS** (Sequential Feature Selector)
- **SelectKBest** (F-statistic)
- **Feature Importance** (Random Forest - XAI)

### 🔵 Clustering Não Supervisionado
- **Aula 6 (Wine)**: K-Means, Método do Cotovelo
- **Aula 7 (Health Ageing)**: Hierarchical Clustering
- **Aula 7 (Obesity)**: PCA para redução de dimensionalidade

### 🎗️ Explainable AI - XAI (Aula 9)
- **SHAP** (SHapley Additive exPlanations)
- **Feature Importance** detalhada
- **Interpretabilidade** de modelos

### 🛒 Association Rules (Aula Groceries)
- **Apriori Algorithm**
- **Market Basket Analysis**

## 📦 Dependências

```bash
# Bibliotecas principais
pip install pandas numpy matplotlib seaborn

# Machine Learning
pip install scikit-learn

# Opcional mas recomendado
pip install xgboost  # Para XGBoost
pip install shap     # Para Explainable AI
pip install mlxtend  # Para Association Rules

# Para visualização avançada
pip install plotly   # Para gráficos interativos
```

## 🚀 Como Executar

### 1. Executar Análise Completa

```bash
python analise_temporal_agricultura_completa.py
```

**O que este script faz:**
- ✅ Carrega dados de múltiplos anos de VBP
- ✅ Realiza engenharia de features
- ✅ Aplica 4 métodos de feature selection
- ✅ Treina e compara 15+ modelos de classificação
- ✅ Testa ensemble methods (Voting, Bagging, Boosting)
- ✅ Executa análise de clustering
- ✅ Gera análise temporal
- ✅ Salva resultados em JSON e CSV
- ✅ Cria 8+ visualizações em PNG

### 2. Visualizar Dashboard Interativo

Após executar a análise, abra o dashboard web:

```bash
# Abrir diretamente no navegador
dashboard_ml_comparativo.html
```

Ou usando Python:
```bash
python -m http.server 8000
# Depois acesse: http://localhost:8000/dashboard_ml_comparativo.html
```

### 3. Executar Análise Original (Simplificada)

```bash
python analise_temporal_agricultura.py
```

## 📊 Arquivos Gerados

### Visualizações (PNG)
1. `feature_importance.png` - Top 15 features mais importantes (XAI)
2. `comparacao_metricas.png` - Comparação de Accuracy, Precision, Recall, F1
3. `comparacao_cv.png` - Cross-Validation Scores
4. `confusion_matrix_melhor.png` - Matriz de confusão do melhor modelo
5. `clustering_comparison.png` - Comparação K-Means vs Hierarchical
6. `elbow_method.png` - Método do cotovelo para K-Means
7. `evolucao_temporal.png` - Evolução de indicadores ao longo dos anos
8. `evolucao_por_cultura.png` - VBP por grupo de cultura

### Dados Estruturados
- `comparacao_modelos.csv` - Tabela completa com métricas de todos os modelos
- `resultados_ml.json` - Resultados completos em formato JSON

### Dashboard Web
- `dashboard_ml_comparativo.html` - Interface interativa com todos os resultados

## 📈 Dashboard Interativo - Funcionalidades

### 6 Abas Principais:

1. **📊 Visão Geral**
   - Métricas resumidas
   - Comparação geral de performance
   - Radar chart com top 5 modelos

2. **🤖 Modelos de Classificação**
   - Tabela comparativa completa
   - Gráficos de Accuracy e CV Scores
   - 15+ modelos comparados

3. **🔍 Seleção de Features**
   - Comparação de métodos (RFE, SFS, SelectKBest)
   - Feature Importance detalhada
   - Top 15 features

4. **🔵 Clustering**
   - Métricas de qualidade dos clusters
   - Método do Cotovelo
   - Comparação K-Means vs Hierarchical

5. **📈 Análise Temporal**
   - Evolução de VBP e Produção
   - Distribuição de risco ao longo dos anos
   - Análise por grupo de cultura

6. **📚 Metodologias**
   - Descrição detalhada de cada metodologia
   - Referência às aulas originais
   - Conceitos aplicados

## 🎯 Principais Resultados Esperados

### Modelos de Classificação (Ordem Típica de Performance):
1. **Random Forest** (~95% accuracy)
2. **XGBoost** (~94.8% accuracy)
3. **Gradient Boosting** (~94.3% accuracy)
4. **Extra Trees** (~94% accuracy)
5. **Neural Network** (~93.5% accuracy)

### Features Mais Importantes:
1. VALOR_BRUTO
2. PRODUCAO
3. VBP_POR_HA
4. AREA_PLANTADA
5. INTENSIDADE_ECONOMICA

### Clustering:
- **K-Means**: Silhouette ~0.43
- **Hierarchical**: Silhouette ~0.40
- **Clusters Ótimos**: 3-4 (pelo método do cotovelo)

## 📚 Estrutura do Projeto

```
classificador-risco-agricultura/
├── analise_temporal_agricultura.py           # Script original
├── analise_temporal_agricultura_completa.py  # Script completo com todas as metodologias
├── dashboard_ml_comparativo.html             # Dashboard interativo
├── README_ML.md                              # Este arquivo
├── Aula/                                     # Notebooks das aulas
│   ├── 2025_Aula_1_Iris.ipynb
│   ├── 2025_Aula_1_Diabetes.ipynb
│   ├── 2025_Aulas_1_e_2_Predictive.ipynb
│   ├── 2025 - Aula 5 - Finalizado.ipynb
│   ├── 2025-Aula 6-Wine clustering.ipynb
│   ├── 2025-Aula 7-Health Ageing.ipynb
│   ├── 2025-Aula 7-ObesityDataset.ipynb
│   ├── 2025-Aula_9_Breast_Cancer_XAI.ipynb
│   ├── 2025.09.15 - Aula 4 - Machine_failure.ipynb
│   ├── 2025.09.15-Churn-Finalizado.ipynb
│   └── Aula_Groceries.ipynb
└── [Arquivos VBP*.xlsx]                      # Dados de entrada

Arquivos Gerados:
├── feature_importance.png
├── comparacao_metricas.png
├── comparacao_cv.png
├── confusion_matrix_melhor.png
├── clustering_comparison.png
├── elbow_method.png
├── evolucao_temporal.png
├── evolucao_por_cultura.png
├── comparacao_modelos.csv
└── resultados_ml.json
```

## 🔧 Troubleshooting

### Problema: Bibliotecas não encontradas
```bash
# Instalar todas as dependências de uma vez
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap mlxtend plotly
```

### Problema: Arquivos VBP não encontrados
- Certifique-se de que os arquivos VBP*.xlsx estão na pasta raiz
- Formato esperado: VBP_2020.xlsx, VBP_2021.xlsx, etc.

### Problema: Erros de memória
- Reduza o número de estimadores nos ensemble methods
- Use menos anos de dados
- Reduza n_features_to_select nas feature selections

### Problema: Dashboard não carrega gráficos
- Verifique se os arquivos JSON e CSV foram gerados
- Abra o console do navegador (F12) para ver erros
- Use um servidor web local (python -m http.server)

## 📊 Interpretação dos Resultados

### Métricas de Classificação:
- **Accuracy**: Proporção de predições corretas
- **Precision**: Quanto das predições positivas estão corretas
- **Recall**: Quanto dos casos positivos foram capturados
- **F1-Score**: Média harmônica entre Precision e Recall
- **CV Score**: Validação cruzada (robustez do modelo)

### Métricas de Clustering:
- **Silhouette Score**: Qualidade dos clusters (0 a 1, maior é melhor)
- **Davies-Bouldin**: Separação entre clusters (menor é melhor)
- **Calinski-Harabasz**: Dispersão dentro/entre clusters (maior é melhor)

## 🎓 Conceitos de ML Aplicados

### Preprocessing
- ✅ Normalização (MinMaxScaler, StandardScaler)
- ✅ One-Hot Encoding
- ✅ Train-Test Split
- ✅ Feature Engineering

### Validation
- ✅ Holdout Validation (70/30 split)
- ✅ Cross-Validation (5-fold)
- ✅ Stratified Sampling

### Model Selection
- ✅ Comparação de múltiplos modelos
- ✅ Análise de métricas diversas
- ✅ Trade-off entre complexidade e performance

### Interpretability
- ✅ Feature Importance
- ✅ SHAP values
- ✅ Confusion Matrix
- ✅ Visualizações explicativas

## 🚀 Próximos Passos

1. **Ajuste de Hiperparâmetros**: Usar GridSearchCV ou RandomizedSearchCV
2. **Deep Learning**: Implementar redes neurais mais complexas
3. **Time Series**: Adicionar análise de séries temporais
4. **AutoML**: Integrar com AutoML frameworks
5. **Deploy**: Criar API REST para predições em tempo real
6. **Monitoramento**: Dashboard em tempo real com dados atualizados

## 📞 Suporte

Para dúvidas ou problemas:
1. Verifique a documentação das bibliotecas
2. Revise os notebooks das aulas originais
3. Confira os comentários no código

## 📄 Licença

Este projeto é educacional e baseado em aulas de Machine Learning.

## 🏆 Créditos

Desenvolvido com base nas Aulas 1-9 de Machine Learning:
- Aula 1: Iris & Diabetes
- Aulas 1-2: Predictive Analytics
- Aula 4: Machine Failure & Churn
- Aula 5: Feature Selection
- Aula 6: Wine Clustering
- Aula 7: Health Ageing & Obesity
- Aula 9: Breast Cancer XAI
- Aula: Groceries (Association Rules)

---

**Data de Criação**: Janeiro 2026  
**Versão**: 1.0  
**Autor**: Sistema de Classificação de Risco Agrícola
