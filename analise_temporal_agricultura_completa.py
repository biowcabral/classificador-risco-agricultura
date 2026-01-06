"""
Sistema Avançado de Análise Temporal de Risco de Desperdício Agrícola
Comparação Abrangente de Metodologias de Machine Learning
Baseado nas Aulas 1-9: Iris, Diabetes, Predictive, Machine Failure, Churn, Wine Clustering, Health Ageing, Obesity, Breast Cancer XAI, Groceries
Autor: Sistema de Classificação de Risco Agrícola
Data: Janeiro 2026
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import glob
import os
import re
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Importações para Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report, 
                            ConfusionMatrixDisplay, roc_auc_score, roc_curve,
                            silhouette_score, davies_bouldin_score, calinski_harabasz_score)

# CLASSIFICAÇÃO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                             GradientBoostingClassifier, BaggingClassifier,
                             VotingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# CLUSTERING
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

# FEATURE SELECTION
from sklearn.feature_selection import RFE, SequentialFeatureSelector, SelectKBest, f_classif

# ASSOCIATION RULES
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("⚠ mlxtend não disponível. Regras de associação desabilitadas.")

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost não disponível. Use: pip install xgboost")

# SHAP para XAI
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP não disponível. XAI desabilitado. Use: pip install shap")

# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RESULTADOS = {
    'models': {},
    'feature_selection': {},
    'clustering': {},
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

print("=" * 100)
print(" " * 15 + "SISTEMA AVANÇADO DE ANÁLISE TEMPORAL DE RISCO AGRÍCOLA")
print(" " * 20 + "Comparação Abrangente de Metodologias de ML")
print("=" * 100)

# =============================================================================
# FUNÇÕES AUXILIARES - CARREGAMENTO DE DADOS
# =============================================================================

def padronizar_grupo_existente(grupo):
    """Padroniza nomes de grupos existentes nos dados"""
    if pd.isna(grupo):
        return 'Outros'
    
    grupo_str = str(grupo).strip().lower()
    grupo_str = grupo_str.replace('ã', 'a').replace('õ', 'o').replace('ç', 'c')
    grupo_str = grupo_str.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    grupo_str = grupo_str.replace('ó', 'o').replace('ú', 'u').replace('ê', 'e')
    
    mapa_grupos = {
        'graos e outras grandes culturas': 'Grãos',
        'graos': 'Grãos',
        'hortalicas': 'Hortaliças',
        'frutas': 'Frutas',
        'florestais': 'Florestais',
        'pecuaria': 'Pecuária',
        'flores': 'Flores',
        'a': 'Grãos',
        'b': 'Hortaliças',
        'c': 'Frutas',
        'd': 'Florestais',
        'e': 'Pecuária',
    }
    
    return mapa_grupos.get(grupo_str, 'Outros')

def classificar_cultura(cultura):
    """Classifica cultura em grupos"""
    if pd.isna(cultura):
        return 'Outros'
    
    cultura_upper = str(cultura).upper().strip()
    
    graos = ['SOJA', 'MILHO', 'TRIGO', 'FEIJAO', 'FEIJÃO', 'ARROZ', 'CANA']
    hortalicas = ['TOMATE', 'BATATA', 'CEBOLA', 'CENOURA', 'ALFACE', 'MANDIOCA']
    frutas = ['LARANJA', 'BANANA', 'UVA', 'MACA', 'MAÇÃ', 'MAMAO', 'MAMÃO']
    florestais = ['MADEIRA', 'PINUS', 'EUCALIPTO', 'ERVA-MATE']
    pecuaria = ['LEITE', 'BOVINO', 'SUINO', 'FRANGO', 'OVOS', 'PEIXE']
    flores = ['ROSA', 'CRISANTEMO', 'ORQUIDEA']
    
    for keyword in graos:
        if keyword in cultura_upper:
            return 'Grãos'
    for keyword in hortalicas:
        if keyword in cultura_upper:
            return 'Hortaliças'
    for keyword in frutas:
        if keyword in cultura_upper:
            return 'Frutas'
    for keyword in florestais:
        if keyword in cultura_upper:
            return 'Florestais'
    for keyword in pecuaria:
        if keyword in cultura_upper:
            return 'Pecuária'
    for keyword in flores:
        if keyword in cultura_upper:
            return 'Flores'
    
    return 'Outros'

def obter_keywords_coluna(coluna):
    """Retorna palavras-chave para identificar colunas"""
    keywords = {
        'MUNICIPIO': ['município', 'municipio', 'cidade'],
        'CULTURA': ['cultura', 'produto', 'cultivo'],
        'PRODUCAO': ['produção', 'producao', 'quant'],
        'AREA_PLANTADA': ['área', 'area', 'ha', 'hectare'],
        'VALOR_BRUTO': ['vbp', 'valor', 'bruto']
    }
    return keywords.get(coluna, [])

def padronizar_colunas(df, ano):
    """Padroniza nomes de colunas entre diferentes anos"""
    mapeamentos = {
        'município': 'MUNICIPIO',
        'municipio': 'MUNICIPIO',
        'produção': 'PRODUCAO',
        'producao': 'PRODUCAO',
        'área (ha)': 'AREA_PLANTADA',
        'area (ha)': 'AREA_PLANTADA',
        'área': 'AREA_PLANTADA',
        'area': 'AREA_PLANTADA',
        'vbp': 'VALOR_BRUTO',
        'valor bruto': 'VALOR_BRUTO',
        'valor bruto': 'VALOR_BRUTO',
        'cultura': 'CULTURA',
        'produto': 'CULTURA',
        'grupo': 'GRUPO_CULTURA',
        'grupo cultura': 'GRUPO_CULTURA',
        'região': 'REGIAO',
        'regiao': 'REGIAO'
    }
    
    df_renamed = df.copy()
    for col_original in df.columns:
        col_lower = col_original.lower().strip()
        if col_lower in mapeamentos:
            df_renamed = df_renamed.rename(columns={col_original: mapeamentos[col_lower]})
    
    colunas_essenciais = ['MUNICIPIO', 'CULTURA', 'PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
    for col in colunas_essenciais:
        if col not in df_renamed.columns:
            for df_col in df_renamed.columns:
                if any(keyword in df_col.lower() for keyword in obter_keywords_coluna(col)):
                    df_renamed = df_renamed.rename(columns={df_col: col})
                    break
    
    if 'GRUPO_CULTURA' not in df_renamed.columns and 'CULTURA' in df_renamed.columns:
        df_renamed['GRUPO_CULTURA'] = df_renamed['CULTURA'].apply(classificar_cultura)
    elif 'GRUPO_CULTURA' in df_renamed.columns:
        df_renamed['GRUPO_CULTURA'] = df_renamed['GRUPO_CULTURA'].apply(padronizar_grupo_existente)
        mask_vazio = df_renamed['GRUPO_CULTURA'].isna() | (df_renamed['GRUPO_CULTURA'] == 'Outros')
        if mask_vazio.any() and 'CULTURA' in df_renamed.columns:
            df_renamed.loc[mask_vazio, 'GRUPO_CULTURA'] = df_renamed.loc[mask_vazio, 'CULTURA'].apply(classificar_cultura)
    
    return df_renamed

def carregar_dados_multi_anos(pasta="."):
    """Carrega dados de múltiplos anos de VBP"""
    print("\n📂 Buscando arquivos VBP...")
    
    patterns = ["VBP*.xls", "VBP*.xlsx", "vbp*.xls", "vbp*.xlsx"]
    arquivos_encontrados = []
    
    for pattern in patterns:
        arquivos_encontrados.extend(glob.glob(os.path.join(pasta, pattern)))
    
    dados_por_ano = {}
    
    for arquivo in arquivos_encontrados:
        nome_arquivo = os.path.basename(arquivo).lower()
        match = re.search(r'(20\d{2})', nome_arquivo)
        if match:
            ano = int(match.group(1))
            try:
                print(f"    ✓ Carregando: {os.path.basename(arquivo)} ({ano})")
                df = pd.read_excel(arquivo, skiprows=1)
                
                df.columns = [col.strip().replace('\n', '').replace('  ', ' ') 
                             for col in df.columns]
                
                df = padronizar_colunas(df, ano)
                df['ANO'] = ano
                
                dados_por_ano[ano] = df
                print(f"      {len(df):,} registros carregados")
                
            except Exception as e:
                print(f"    ✗ Erro ao carregar {ano}: {e}")
    
    return dados_por_ano

# =============================================================================
# FASE 1: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n" + "=" * 100)
print("FASE 1: CARREGAMENTO E EXPLORAÇÃO DOS DADOS")
print("=" * 100)

dados_por_ano = carregar_dados_multi_anos()

if not dados_por_ano:
    print("\n❌ ERRO: Nenhum dado foi carregado. Verifique os arquivos.")
    exit()

print(f"\n✓ Total de anos carregados: {len(dados_por_ano)}")
print(f"✓ Período: {min(dados_por_ano.keys())} - {max(dados_por_ano.keys())}")

# Combinar todos os anos
print("\n🔄 Combinando dados de todos os anos...")
todos_dados = pd.concat(dados_por_ano.values(), ignore_index=True)

print(f"\n📊 Informações do dataset combinado:")
print(f"   • Total de registros: {len(todos_dados):,}")
if 'MUNICIPIO' in todos_dados.columns:
    print(f"   • Total de municípios: {todos_dados['MUNICIPIO'].nunique():,}")
print(f"   • Período: {todos_dados['ANO'].min()} - {todos_dados['ANO'].max()}")

# =============================================================================
# FASE 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES
# =============================================================================

print("\n" + "=" * 100)
print("FASE 2: PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
print("=" * 100)

# Converter colunas numéricas
colunas_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO']
for col in colunas_numericas:
    if col in todos_dados.columns:
        todos_dados[col] = pd.to_numeric(todos_dados[col], errors='coerce')

print(f"   Registros antes da limpeza: {len(todos_dados):,}")
todos_dados = todos_dados.dropna(subset=colunas_numericas)
print(f"   Registros após limpeza: {len(todos_dados):,}")

# Engenharia de Features
print("\n🔧 Engenharia de Features:")
print("   • Calculando diversidade produtiva por município")
diversidade = todos_dados.groupby(['ANO', 'MUNICIPIO'])['CULTURA'].nunique().reset_index()
diversidade.columns = ['ANO', 'MUNICIPIO', 'diversidade_produtiva']
todos_dados = todos_dados.merge(diversidade, on=['ANO', 'MUNICIPIO'], how='left')

print("   • Criando features derivadas")
todos_dados['PRODUTIVIDADE'] = todos_dados['PRODUCAO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)
todos_dados['VBP_POR_HA'] = todos_dados['VALOR_BRUTO'] / todos_dados['AREA_PLANTADA'].replace(0, 1)
todos_dados['INTENSIDADE_ECONOMICA'] = todos_dados['VALOR_BRUTO'] / todos_dados['PRODUCAO'].replace(0, 1)

# Criar variável target: RISCO_DESPERDICIO
print("\n🎯 Criando variável target: RISCO_DESPERDICIO")

def classificar_risco(df_ano):
    """Classifica risco de desperdício baseado em quantis múltiplos"""
    prod_q = df_ano['PRODUCAO'].quantile([0.33, 0.66])
    area_q = df_ano['AREA_PLANTADA'].quantile([0.33, 0.66])
    valor_q = df_ano['VALOR_BRUTO'].quantile([0.33, 0.66])
    div_q = df_ano['diversidade_produtiva'].quantile([0.33, 0.66])
    
    def calcular_score(row):
        score = 0
        if row['PRODUCAO'] <= prod_q.iloc[0]: score += 1
        elif row['PRODUCAO'] >= prod_q.iloc[1]: score -= 1
        
        if row['AREA_PLANTADA'] <= area_q.iloc[0]: score += 1
        elif row['AREA_PLANTADA'] >= area_q.iloc[1]: score -= 1
        
        if row['VALOR_BRUTO'] <= valor_q.iloc[0]: score += 1
        elif row['VALOR_BRUTO'] >= valor_q.iloc[1]: score -= 1
        
        if row['diversidade_produtiva'] <= div_q.iloc[0]: score += 1
        elif row['diversidade_produtiva'] >= div_q.iloc[1]: score -= 1
        
        if score >= 2: return 'ALTO'
        elif score <= -2: return 'BAIXO'
        else: return 'MEDIO'
    
    df_ano['RISCO_DESPERDICIO'] = df_ano.apply(calcular_score, axis=1)
    return df_ano

todos_dados = pd.concat([classificar_risco(df) for ano, df in todos_dados.groupby('ANO')], 
                        ignore_index=True)

print("\n📊 Distribuição de RISCO_DESPERDICIO:")
print(todos_dados['RISCO_DESPERDICIO'].value_counts())
print("\nPercentuais:")
print(todos_dados['RISCO_DESPERDICIO'].value_counts(normalize=True) * 100)

# =============================================================================
# FASE 3: PREPARAÇÃO PARA MODELAGEM
# =============================================================================

print("\n" + "=" * 100)
print("FASE 3: PREPARAÇÃO PARA MODELAGEM")
print("=" * 100)

# Separação de features e target
features_numericas = ['PRODUCAO', 'AREA_PLANTADA', 'VALOR_BRUTO', 
                     'diversidade_produtiva', 'PRODUTIVIDADE', 'VBP_POR_HA', 
                     'INTENSIDADE_ECONOMICA', 'ANO']
X = todos_dados[features_numericas + ['GRUPO_CULTURA']].copy()
y = todos_dados['RISCO_DESPERDICIO']

print(f"\n   • Features (X): {X.shape}")
print(f"   • Target (y): {y.shape}")

# One-Hot Encoding
print("\n🔄 Aplicando One-Hot Encoding...")
X = pd.get_dummies(X, drop_first=True, dtype='int64')
print(f"   • Features após encoding: {X.shape}")

# Divisão treino/teste
print("\n✂ Dividindo dados em treino (70%) e teste (30%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"   • Treino: {X_train.shape[0]:,} registros")
print(f"   • Teste: {X_test.shape[0]:,} registros")

# Normalização
print("\n🔄 Aplicando normalizações (MinMax e Standard)...")
minmax = MinMaxScaler()
standard = StandardScaler()

X_train_minmax = pd.DataFrame(minmax.fit_transform(X_train), columns=X_train.columns)
X_test_minmax = pd.DataFrame(minmax.transform(X_test), columns=X_test.columns)

X_train_standard = pd.DataFrame(standard.fit_transform(X_train), columns=X_train.columns)
X_test_standard = pd.DataFrame(standard.transform(X_test), columns=X_test.columns)

print("   ✓ Dados normalizados!")

# =============================================================================
# FASE 4: SELEÇÃO DE FEATURES (Feature Selection)
# =============================================================================

print("\n" + "=" * 100)
print("FASE 4: SELEÇÃO DE FEATURES")
print("=" * 100)

# RFE (Recursive Feature Elimination)
print("\n🔍 1. RFE (Recursive Feature Elimination) - Aula 5")
rf_base = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(estimator=rf_base, n_features_to_select=10, step=1)
rfe.fit(X_train, y_train)
selected_rfe = X_train.columns[rfe.support_]
print(f"   Features selecionadas por RFE: {len(selected_rfe)}")
print(f"   {list(selected_rfe)}")
RESULTADOS['feature_selection']['RFE'] = list(selected_rfe)

# SFS (Sequential Feature Selector)
print("\n🔍 2. SFS (Sequential Feature Selector) - Aula 5")
sfs = SequentialFeatureSelector(estimator=rf_base, n_features_to_select=7, 
                                direction='forward', n_jobs=-1)
sfs.fit(X_train, y_train)
selected_sfs = X_train.columns[sfs.support_]
print(f"   Features selecionadas por SFS: {len(selected_sfs)}")
print(f"   {list(selected_sfs)}")
RESULTADOS['feature_selection']['SFS'] = list(selected_sfs)

# SelectKBest
print("\n🔍 3. SelectKBest (F-statistic)")
skb = SelectKBest(f_classif, k=10)
skb.fit(X_train, y_train)
selected_skb = X_train.columns[skb.get_support()]
print(f"   Features selecionadas por SelectKBest: {len(selected_skb)}")
print(f"   {list(selected_skb)}")
RESULTADOS['feature_selection']['SelectKBest'] = list(selected_skb)

# Feature Importance (Random Forest)
print("\n🔍 4. Feature Importance (Random Forest) - Aula 9 (XAI)")
rf_base.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_base.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Features mais importantes:")
print(feature_importance.head(10))
RESULTADOS['feature_selection']['Importance'] = feature_importance.to_dict('records')[:10]

# Visualizar importância
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance - Random Forest (XAI)', fontsize=14, fontweight='bold')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("\n   ✓ Salvo: feature_importance.png")
plt.close()

# =============================================================================
# FASE 5: TREINAMENTO E COMPARAÇÃO DE MODELOS DE CLASSIFICAÇÃO
# =============================================================================

print("\n" + "=" * 100)
print("FASE 5: TREINAMENTO E COMPARAÇÃO DE MODELOS DE CLASSIFICAÇÃO")
print("=" * 100)

# Codificar target para modelos que precisam
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Dicionário de modelos
modelos = {
    # Aula 1 - Iris
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    
    # Aula 1/2 - Diabetes/Predictive
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    
    # Aula 4 - Machine Failure
    'Neural Network': MLPClassifier(hidden_layers=(100, 50), max_iter=500, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    
    # Ensemble Methods - Aula 4
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    
    # Naive Bayes
    'Naive Bayes': GaussianNB(),
}

# Adicionar XGBoost se disponível
if XGBOOST_AVAILABLE:
    modelos['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, 
                                      eval_metric='mlogloss')

# Treinar e avaliar todos os modelos
resultados_comparacao = []

print("\n🤖 Treinando modelos...")
for nome, modelo in modelos.items():
    print(f"\n   Treinando {nome}...")
    
    try:
        # Treinar
        modelo.fit(X_train_minmax, y_train)
        
        # Predizer
        y_pred = modelo.predict(X_test_minmax)
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(modelo, X_train_minmax, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Armazenar resultados
        resultado = {
            'Modelo': nome,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'CV Mean': cv_mean,
            'CV Std': cv_std
        }
        resultados_comparacao.append(resultado)
        
        print(f"      ✓ Accuracy: {accuracy:.4f}")
        print(f"      ✓ F1-Score: {f1:.4f}")
        print(f"      ✓ CV Mean: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # Armazenar no dicionário global
        RESULTADOS['models'][nome] = resultado
        
    except Exception as e:
        print(f"      ✗ Erro: {e}")

# Criar DataFrame de comparação
df_comparacao = pd.DataFrame(resultados_comparacao)
df_comparacao = df_comparacao.sort_values('Accuracy', ascending=False)

print("\n" + "=" * 100)
print("📊 COMPARAÇÃO DE TODOS OS MODELOS")
print("=" * 100)
print(df_comparacao.to_string(index=False))

# Salvar comparação em CSV
df_comparacao.to_csv('comparacao_modelos.csv', index=False)
print("\n✓ Salvo: comparacao_modelos.csv")

# =============================================================================
# VISUALIZAÇÃO DA COMPARAÇÃO DE MODELOS
# =============================================================================

print("\n📊 Criando visualizações comparativas...")

# Gráfico 1: Comparação de Métricas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy
df_sorted = df_comparacao.sort_values('Accuracy')
axes[0, 0].barh(df_sorted['Modelo'], df_sorted['Accuracy'], color='skyblue')
axes[0, 0].set_xlabel('Accuracy')
axes[0, 0].set_title('Comparação de Accuracy', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# F1-Score
df_sorted = df_comparacao.sort_values('F1-Score')
axes[0, 1].barh(df_sorted['Modelo'], df_sorted['F1-Score'], color='lightcoral')
axes[0, 1].set_xlabel('F1-Score')
axes[0, 1].set_title('Comparação de F1-Score', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Precision
df_sorted = df_comparacao.sort_values('Precision')
axes[1, 0].barh(df_sorted['Modelo'], df_sorted['Precision'], color='lightgreen')
axes[1, 0].set_xlabel('Precision')
axes[1, 0].set_title('Comparação de Precision', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Recall
df_sorted = df_comparacao.sort_values('Recall')
axes[1, 1].barh(df_sorted['Modelo'], df_sorted['Recall'], color='plum')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_title('Comparação de Recall', fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('comparacao_metricas.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: comparacao_metricas.png")
plt.close()

# Gráfico 2: Cross-Validation Scores
plt.figure(figsize=(12, 6))
df_sorted = df_comparacao.sort_values('CV Mean')
plt.barh(df_sorted['Modelo'], df_sorted['CV Mean'], 
         xerr=df_sorted['CV Std'], color='steelblue', alpha=0.7)
plt.xlabel('Cross-Validation Score (Mean ± Std)')
plt.title('Comparação de Cross-Validation Scores', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_cv.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: comparacao_cv.png")
plt.close()

# Matriz de confusão do melhor modelo
melhor_modelo_nome = df_comparacao.iloc[0]['Modelo']
melhor_modelo = modelos[melhor_modelo_nome]
y_pred_melhor = melhor_modelo.predict(X_test_minmax)

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_melhor)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['ALTO', 'BAIXO', 'MEDIO'],
            yticklabels=['ALTO', 'BAIXO', 'MEDIO'])
plt.title(f'Matriz de Confusão - {melhor_modelo_nome}', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig('confusion_matrix_melhor.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: confusion_matrix_melhor.png")
plt.close()

# =============================================================================
# FASE 6: ENSEMBLE METHODS - VOTING E BAGGING (Aula 4)
# =============================================================================

print("\n" + "=" * 100)
print("FASE 6: ENSEMBLE METHODS - VOTING E BAGGING")
print("=" * 100)

# Voting Classifier - Hard Voting
print("\n🗳 Voting Classifier (Hard)...")
voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42))
    ],
    voting='hard'
)
voting_hard.fit(X_train_minmax, y_train)
acc_voting_hard = voting_hard.score(X_test_minmax, y_test)
print(f"   ✓ Accuracy (Hard Voting): {acc_voting_hard:.4f}")
RESULTADOS['models']['Voting Hard'] = {'Accuracy': acc_voting_hard}

# Voting Classifier - Soft Voting
print("\n🗳 Voting Classifier (Soft)...")
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42))
    ],
    voting='soft'
)
voting_soft.fit(X_train_minmax, y_train)
acc_voting_soft = voting_soft.score(X_test_minmax, y_test)
print(f"   ✓ Accuracy (Soft Voting): {acc_voting_soft:.4f}")
RESULTADOS['models']['Voting Soft'] = {'Accuracy': acc_voting_soft}

# Bagging
print("\n🎒 Bagging Classifier...")
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10, random_state=42),
    n_estimators=50,
    random_state=42
)
bagging.fit(X_train_minmax, y_train)
acc_bagging = bagging.score(X_test_minmax, y_test)
print(f"   ✓ Accuracy (Bagging): {acc_bagging:.4f}")
RESULTADOS['models']['Bagging'] = {'Accuracy': acc_bagging}

# =============================================================================
# FASE 7: CLUSTERING (Aula 6 - Wine Clustering, Aula 7)
# =============================================================================

print("\n" + "=" * 100)
print("FASE 7: CLUSTERING - ANÁLISE NÃO SUPERVISIONADA")
print("=" * 100)

# Preparar dados para clustering (usar apenas features numéricas)
X_clustering = X_train_minmax.copy()

# K-Means
print("\n🔵 K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X_clustering)
silhouette_kmeans = silhouette_score(X_clustering, clusters_kmeans)
davies_bouldin_kmeans = davies_bouldin_score(X_clustering, clusters_kmeans)
calinski_kmeans = calinski_harabasz_score(X_clustering, clusters_kmeans)

print(f"   ✓ Silhouette Score: {silhouette_kmeans:.4f}")
print(f"   ✓ Davies-Bouldin Index: {davies_bouldin_kmeans:.4f}")
print(f"   ✓ Calinski-Harabasz Index: {calinski_kmeans:.4f}")

RESULTADOS['clustering']['K-Means'] = {
    'Silhouette': silhouette_kmeans,
    'Davies-Bouldin': davies_bouldin_kmeans,
    'Calinski-Harabasz': calinski_kmeans
}

# Hierarchical Clustering
print("\n🌳 Hierarchical Clustering (Agglomerative)...")
hierarchical = AgglomerativeClustering(n_clusters=3)
clusters_hierarchical = hierarchical.fit_predict(X_clustering)
silhouette_hierarchical = silhouette_score(X_clustering, clusters_hierarchical)
davies_bouldin_hierarchical = davies_bouldin_score(X_clustering, clusters_hierarchical)
calinski_hierarchical = calinski_harabasz_score(X_clustering, clusters_hierarchical)

print(f"   ✓ Silhouette Score: {silhouette_hierarchical:.4f}")
print(f"   ✓ Davies-Bouldin Index: {davies_bouldin_hierarchical:.4f}")
print(f"   ✓ Calinski-Harabasz Index: {calinski_hierarchical:.4f}")

RESULTADOS['clustering']['Hierarchical'] = {
    'Silhouette': silhouette_hierarchical,
    'Davies-Bouldin': davies_bouldin_hierarchical,
    'Calinski-Harabasz': calinski_hierarchical
}

# PCA para visualização
print("\n🔍 Aplicando PCA para visualização...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

# Visualizar clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# K-Means
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_kmeans, 
                          cmap='viridis', alpha=0.6, s=30)
axes[0].set_title(f'K-Means Clustering (Silhouette: {silhouette_kmeans:.3f})', 
                 fontweight='bold')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

# Hierarchical
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_hierarchical, 
                          cmap='plasma', alpha=0.6, s=30)
axes[1].set_title(f'Hierarchical Clustering (Silhouette: {silhouette_hierarchical:.3f})', 
                 fontweight='bold')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=150, bbox_inches='tight')
print("\n   ✓ Salvo: clustering_comparison.png")
plt.close()

# Método do cotovelo para K-Means
print("\n📊 Método do Cotovelo para determinar número ótimo de clusters...")
wcss = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_clustering)
    wcss.append(kmeans_temp.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, marker='o', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (k)')
plt.ylabel('WCSS (Inércia)')
plt.title('Método do Cotovelo - K-Means', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
print("   ✓ Salvo: elbow_method.png")
plt.close()

# =============================================================================
# FASE 8: ANÁLISE TEMPORAL E TENDÊNCIAS
# =============================================================================

print("\n" + "=" * 100)
print("FASE 8: ANÁLISE TEMPORAL E TENDÊNCIAS")
print("=" * 100)

# Agregar dados por ano
dados_anuais = todos_dados.groupby('ANO').agg({
    'MUNICIPIO': 'nunique',
    'PRODUCAO': 'sum',
    'AREA_PLANTADA': 'sum',
    'VALOR_BRUTO': 'sum',
    'diversidade_produtiva': 'mean',
    'PRODUTIVIDADE': 'mean',
    'VBP_POR_HA': 'mean'
}).reset_index()

# Evolução do risco por ano
risco_por_ano = todos_dados.groupby(['ANO', 'RISCO_DESPERDICIO']).size().unstack(fill_value=0)
risco_pct_ano = risco_por_ano.div(risco_por_ano.sum(axis=1), axis=0) * 100

# Visualização temporal
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# VBP Total
axes[0, 0].plot(dados_anuais['ANO'], dados_anuais['VALOR_BRUTO'] / 1e9, 
               marker='o', linewidth=2, markersize=8)
axes[0, 0].set_title('Evolução do VBP Total', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('VBP (Bilhões R$)')
axes[0, 0].grid(True, alpha=0.3)

# Produção Total
axes[0, 1].plot(dados_anuais['ANO'], dados_anuais['PRODUCAO'] / 1e6, 
               marker='o', linewidth=2, markersize=8, color='green')
axes[0, 1].set_title('Evolução da Produção Total', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Produção (Milhões ton)')
axes[0, 1].grid(True, alpha=0.3)

# Diversidade Produtiva
axes[1, 0].plot(dados_anuais['ANO'], dados_anuais['diversidade_produtiva'], 
               marker='o', linewidth=2, markersize=8, color='purple')
axes[1, 0].set_title('Evolução da Diversidade Produtiva', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Diversidade Média')
axes[1, 0].set_xlabel('Ano')
axes[1, 0].grid(True, alpha=0.3)

# Distribuição de Risco
risco_pct_ano.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                  color=['#FF6B6B', '#4ECDC4', '#FFE66D'])
axes[1, 1].set_title('Evolução da Distribuição de Risco', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Percentual (%)')
axes[1, 1].set_xlabel('Ano')
axes[1, 1].legend(title='Risco')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('evolucao_temporal.png', dpi=150, bbox_inches='tight')
print("\n   ✓ Salvo: evolucao_temporal.png")
plt.close()

# =============================================================================
# SALVAR RESULTADOS EM JSON
# =============================================================================

print("\n💾 Salvando resultados em JSON...")
with open('resultados_ml.json', 'w', encoding='utf-8') as f:
    json.dump(RESULTADOS, f, indent=4, ensure_ascii=False)
print("   ✓ Salvo: resultados_ml.json")

# =============================================================================
# RESUMO FINAL
# =============================================================================

print("\n" + "=" * 100)
print("✅ ANÁLISE COMPLETA FINALIZADA")
print("=" * 100)

print(f"\n📊 RESUMO DOS RESULTADOS:")
print(f"   • {len(modelos)} modelos de classificação treinados e comparados")
print(f"   • Melhor modelo: {melhor_modelo_nome} (Accuracy: {df_comparacao.iloc[0]['Accuracy']:.4f})")
print(f"   • 3 métodos de feature selection aplicados")
print(f"   • 2 algoritmos de clustering avaliados")
print(f"   • {len(dados_por_ano)} anos de dados analisados")

print("\n📁 ARQUIVOS GERADOS:")
arquivos = [
    'feature_importance.png',
    'comparacao_metricas.png',
    'comparacao_cv.png',
    'confusion_matrix_melhor.png',
    'clustering_comparison.png',
    'elbow_method.png',
    'evolucao_temporal.png',
    'comparacao_modelos.csv',
    'resultados_ml.json'
]
for arquivo in arquivos:
    print(f"   ✓ {arquivo}")

print("\n🎯 METODOLOGIAS APLICADAS:")
print("   1. Classificação: KNN, Decision Tree, Random Forest, SVM, Neural Network, etc.")
print("   2. Ensemble: Voting, Bagging, Boosting (AdaBoost, Gradient Boosting, XGBoost)")
print("   3. Feature Selection: RFE, SFS, SelectKBest, Feature Importance")
print("   4. Clustering: K-Means, Hierarchical")
print("   5. Dimensionality Reduction: PCA")
print("   6. Preprocessing: MinMaxScaler, StandardScaler, One-Hot Encoding")
print("   7. Validation: Train-Test Split, Cross-Validation")

print("\n" + "=" * 100)
print("Sistema desenvolvido baseado nas Aulas 1-9 de Machine Learning")
print("=" * 100)
