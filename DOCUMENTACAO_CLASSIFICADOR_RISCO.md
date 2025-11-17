# 🚜 Sistema de Classificação de Risco de Desperdício em Municípios

## 📋 Visão Geral

Este sistema utiliza técnicas de **Machine Learning** para classificar municípios brasileiros em categorias de risco de desperdício agrícola baseado em dados do **Valor Bruto da Produção (VBP) 2024**. O modelo emprega o algoritmo **Random Forest** para categorizar municípios em três níveis de risco: **BAIXO**, **MÉDIO** e **ALTO**.

## 🎯 Objetivo

Identificar municípios com maior probabilidade de desperdício agrícola através da análise de:
- Produção agrícola
- Área plantada
- Valor bruto da produção
- Diversidade produtiva (número de culturas diferentes)
- Grupo da cultura

## 📊 Fonte de Dados

- **Arquivo**: `vbp_2024.xlsx`
- **Origem**: Dados definitivos do Valor Bruto da Produção 2024
- **Estrutura**: Dados por município, cultura e grupo de cultura
- **Período**: Safra 2024 (código 2324)

## 🏗️ Arquitetura do Sistema

### 📦 Dependências
```python
pandas              # Manipulação de dados
numpy               # Operações numéricas
scikit-learn        # Machine Learning
matplotlib          # Visualizações
seaborn            # Visualizações estatísticas
```

### 🔧 Componentes Principais

#### 1. **Carregamento e Limpeza de Dados**
```python
def main():
    df = pd.read_excel(file_path, skiprows=1)  # Pula header principal
```

**Mapeamento de Colunas:**
- `Município` → `MUNICIPIO`
- `Produção` → `PRODUCAO` 
- `Área (ha)` → `AREA_PLANTADA`
- `VBP` → `VALOR_BRUTO`
- `Grupo` → `GRUPO_CULTURA`
- `Cultura` → `CULTURA`

#### 2. **Cálculo da Diversidade Produtiva**
```python
def calcular_diversidade(df, municipio_col, cultura_col):
    return df.groupby(municipio_col)[cultura_col].nunique().rename('diversidade_produtiva')
```

**Funcionalidade:** Conta quantas culturas diferentes cada município produz, sendo um indicador de:
- Resiliência agrícola
- Distribuição de risco
- Sustentabilidade produtiva

#### 3. **Sistema de Classificação de Risco**

**Metodologia:** Baseado em quantis (33º e 66º percentis)

```python
def classificar_risco(row):
    score = 0
    # Análise de 4 dimensões:
    # - Produção: quantidade produzida
    # - Área: hectares plantados
    # - Valor: receita bruta
    # - Diversidade: variedade de culturas
    
    # Score de risco (+1 = maior risco, -1 = menor risco)
    if score >= 2: return 'ALTO'
    elif score <= -2: return 'BAIXO'
    else: return 'MEDIO'
```

**Critérios de Classificação:**

| Risco | Condições |
|-------|-----------|
| **ALTO** | ≥2 indicadores negativos (baixa produção, área, valor ou diversidade) |
| **MÉDIO** | -1 a +1 indicadores (situação intermediária) |
| **BAIXO** | ≤-2 indicadores (alta produção, área, valor e diversidade) |

### 🤖 Modelo de Machine Learning

#### **Algoritmo:** Random Forest Classifier
```python
RandomForestClassifier(
    n_estimators=100,    # 100 árvores de decisão
    random_state=42,     # Reprodutibilidade
    max_depth=10         # Profundidade máxima
)
```

**Por que Random Forest?**
- ✅ **Robustez**: Resistente a overfitting
- ✅ **Interpretabilidade**: Importância das features
- ✅ **Performance**: Excelente para dados tabulares
- ✅ **Versatilidade**: Lida bem com dados mistos

#### **Features do Modelo:**
1. `PRODUCAO` - Quantidade produzida (normalizada)
2. `AREA_PLANTADA` - Hectares plantados (normalizada)
3. `VALOR_BRUTO` - Receita em reais (normalizada)
4. `diversidade_produtiva` - Número de culturas (normalizada)
5. `GRUPO_CULTURA` - Categoria da cultura (codificada)

#### **Pré-processamento:**
- **Normalização**: StandardScaler para variáveis numéricas
- **Codificação**: LabelEncoder para variáveis categóricas
- **Divisão**: 80% treino / 20% teste com estratificação

## 📈 Métricas de Avaliação

### 🎯 Métricas Principais

#### **Acurácia Geral**
```
Acurácia: 99.88%
```
Percentual de classificações corretas sobre o total.

#### **Métricas por Classe**

| Classe | Precisão | Recall | F1-Score | Suporte |
|--------|----------|--------|----------|---------|
| ALTO   | 99.78%   | 99.93% | 99.86%   | 1,382   |
| BAIXO  | 100.00%  | 99.93% | 99.96%   | 1,363   |
| MÉDIO  | 99.86%   | 99.80% | 99.83%   | 1,481   |

#### **Definições:**
- **Precisão**: % de predições positivas que estão corretas
- **Recall**: % de casos positivos identificados corretamente
- **F1-Score**: Média harmônica entre precisão e recall
- **Suporte**: Número de amostras por classe

### 📊 Importância das Variáveis

| Variável | Importância | Descrição |
|----------|-------------|-----------|
| VALOR_BRUTO | 33.81% | **Mais importante** - Receita da produção |
| PRODUCAO | 27.24% | Quantidade produzida |
| AREA_PLANTADA | 23.04% | Extensão territorial cultivada |
| diversidade_produtiva | 15.10% | Variedade de culturas |
| GRUPO_CULTURA | 0.80% | **Menos importante** - Tipo de cultura |

## 🗺️ Análises Regionais

### 📍 Risco por Região

| Região | Alto Risco | Baixo Risco | Médio Risco |
|--------|------------|-------------|-------------|
| **Noroeste** | 54.56% | 22.01% | 23.43% |
| **Centro-ocidental** | 45.09% | 28.94% | 25.98% |
| **Norte-central** | 42.08% | 26.48% | 31.44% |
| **Centro-sul** | 35.72% | 35.03% | 29.25% |
| **Oeste** | 30.19% | 27.35% | 42.45% |
| **Sudoeste** | 27.10% | 33.23% | 39.67% |
| **Norte Pioneiro** | 22.59% | 35.55% | 41.86% |
| **Centro-oriental** | 20.48% | 44.10% | 35.43% |
| **Metropolitana** | 19.06% | 45.45% | 35.49% |
| **Sudeste** | 15.85% | 44.08% | 40.07% |

### 🌾 Risco por Grupo de Cultura

| Grupo | Alto Risco | Baixo Risco | Médio Risco |
|-------|------------|-------------|-------------|
| **Hortaliças** | 44.21% | 18.30% | 37.49% |
| **Frutas** | 34.83% | 17.75% | 47.42% |
| **Florestais** | 23.90% | 43.41% | 32.69% |
| **Pecuária** | 18.00% | 55.73% | 26.27% |
| **Grãos/Grandes Culturas** | 12.12% | 67.32% | 20.56% |

## 📊 Funções de Relatório

### 1. **Relatório Detalhado de Métricas**
```python
def generate_detailed_report(y_test, y_pred, target_names):
```
- Calcula métricas por classe
- Médias macro e ponderadas
- Formatação profissional

### 2. **Matriz de Confusão**
```python
def plot_confusion_matrix(y_test, y_pred, target_names):
```
- Visualização com heatmap
- Cores em escala azul
- Anotações numéricas

### 3. **Importância das Features**
```python
def plot_feature_importance(clf, feature_names):
```
- Gráfico de barras
- Ordenação decrescente
- Retorna série pandas

### 4. **Análise de Distribuição**
```python
def analyze_risk_distribution(df):
```
- Distribuição por região
- Distribuição por cultura
- Percentuais formatados

## 💾 Saídas do Sistema

### 1. **Arquivo CSV**
```
classificacao_risco_municipios.csv
```
**Conteúdo:**
- MUNICIPIO
- GRUPO_CULTURA
- PRODUCAO
- AREA_PLANTADA
- VALOR_BRUTO
- diversidade_produtiva
- RISCO_DESPERDICIO

### 2. **Visualizações**
- Matriz de confusão (PNG)
- Importância das features (PNG)

### 3. **Relatórios Console**
- Estatísticas descritivas
- Métricas de performance
- Análises regionais e por cultura

## 🚀 Como Executar

### **Pré-requisitos:**
```bash
pip install pandas scikit-learn matplotlib seaborn openpyxl
```

### **Execução:**
```bash
python municipio_food_waste_risk_classifier_detailed.py
```

### **Arquivos Necessários:**
- `vbp_2024.xlsx` (na mesma pasta)

## 📝 Interpretação dos Resultados

### **Municípios de Alto Risco:**
- **Características**: Baixa produção, área reduzida, baixo VBP, pouca diversidade
- **Recomendações**: 
  - Incentivos à diversificação
  - Investimento em tecnologia
  - Assistência técnica especializada
  - Políticas de crédito rural

### **Municípios de Baixo Risco:**
- **Características**: Alta produção, grandes áreas, alto VBP, grande diversidade
- **Estratégias**: 
  - Manutenção das boas práticas
  - Compartilhamento de conhecimento
  - Centros de excelência
  - Modelo para outros municípios

### **Municípios de Médio Risco:**
- **Características**: Situação intermediária
- **Ações**: 
  - Monitoramento contínuo
  - Intervenções pontuais
  - Prevenção de degradação
  - Incentivos seletivos

## 🔍 Limitações e Considerações

### **Limitações do Modelo:**
1. **Dados Faltantes**: 14,018 registros removidos (39.9%)
2. **Causalidade**: Correlação não implica causalidade
3. **Temporal**: Snapshot de um ano (2024)
4. **Fatores Externos**: Não considera clima, pragas, mercado

### **Considerações Metodológicas:**
1. **Quantis**: Classificação relativa, não absoluta
2. **Balanceamento**: Classes bem distribuídas (32-35%)
3. **Overfitting**: Risco baixo devido ao Random Forest
4. **Generalização**: Alta performance pode indicar dados muito similares

## 🔧 Personalização e Extensões

### **Parâmetros Ajustáveis:**
```python
# Quantis para classificação
quantiles = [0.33, 0.66]  # Pode ajustar para [0.25, 0.75]

# Parâmetros do Random Forest
n_estimators = 100      # Número de árvores
max_depth = 10         # Profundidade máxima
random_state = 42      # Semente aleatória

# Divisão treino/teste
test_size = 0.2        # 20% para teste
```

### **Extensões Possíveis:**
1. **Validação Cruzada**: K-fold para robustez
2. **Outros Algoritmos**: XGBoost, SVM, Neural Networks
3. **Feature Engineering**: Ratios, logs, interações
4. **Análise Temporal**: Múltiplos anos
5. **Fatores Externos**: Clima, economia, população

## 👥 Público-Alvo

- **Gestores Públicos**: Políticas agrícolas regionais
- **Pesquisadores**: Estudos de sustentabilidade
- **Consultores**: Assessoria em agronegócio
- **Produtores**: Análise de risco e oportunidades
- **Investidores**: Identificação de regiões promissoras

## 📞 Suporte e Manutenção

Para dúvidas, melhorias ou reportar bugs:
- Verificar estrutura do arquivo Excel
- Validar instalação das dependências
- Confirmar formato das colunas
- Testar com subset menor dos dados

---

**Versão**: 1.0  
**Data**: Setembro 2025  
**Linguagem**: Python 3.8+  
**Licença**: Uso educacional e pesquisa